"""
eth_bot.py — Main bot controller.

Run modes:
  python eth_bot.py --paper          # Paper trading (simulated fills)
  python eth_bot.py --live           # Live trading on HTX (requires API keys)
  python eth_bot.py --backtest       # Run backtest and exit
  python eth_bot.py --backtest --days 365

The loop fires once per closed bar:
  1. Fetch latest OHLCV + daily bars
  2. Compute indicators
  3. Detect setups → score → filter by threshold
  4. Place pending limit orders via trade_manager
  5. Update trade_manager (TP/SL/BE checks)
  6. Sleep until next bar close
"""

import argparse
import asyncio
import logging
import math
import time
from datetime import datetime, timezone

import pandas as pd

from config import BotConfig, cfg as default_cfg
from market_data import make_exchange, fetch_ohlcv, fetch_daily_ohlcv, calc_indicators, get_atr_mult, inject_htf_trend
from strategy_engine import detect_setups_df, get_signals, calc_cancel_bars_dyn
from risk_engine import get_stop_price, calc_qty, calc_tp_levels, round_tick
from execution_engine import ExecutionEngine
from trade_manager import TradeManager
from paper_trader import PaperExecutionEngine
from backtest import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eth_bot.log"),
    ],
    force=True,
)
# Silence noisy third-party HTTP loggers
for _lib in ("ccxt", "urllib3", "requests", "asyncio"):
    logging.getLogger(_lib).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


# ── P9: Funding rate blackout ─────────────────────────────────────────────────

FUNDING_HOURS_UTC = {0, 8, 16}   # HTX perpetual funding settlement hours

def is_in_funding_blackout(config: BotConfig) -> bool:
    """Returns True if we are within `funding_blackout_mins` of a funding event."""
    if not config.funding_blackout_enabled:
        return False
    now = datetime.now(timezone.utc)
    # Minutes elapsed since the last hour boundary
    mins_into_hour = now.minute
    # Distance to next / just-passed funding hour (in minutes)
    for fh in FUNDING_HOURS_UTC:
        # Forward distance
        hours_ahead = (fh - now.hour) % 24
        mins_to = hours_ahead * 60 - mins_into_hour
        if 0 <= mins_to < config.funding_blackout_mins:
            return True
        # Backward distance (just passed)
        if hours_ahead == 0 and mins_into_hour < config.funding_blackout_mins:
            return True
    return False


# ── Bar timing ────────────────────────────────────────────────────────────────

def seconds_until_bar_close(timeframe: str) -> float:
    """Seconds remaining until the current bar closes (rounded up)."""
    tf_seconds = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "4h": 14400,
    }.get(timeframe, 900)
    now_ts = time.time()
    elapsed = now_ts % tf_seconds
    remaining = tf_seconds - elapsed
    return remaining


def sleep_until_next_bar(timeframe: str, buffer_seconds: float = 3.0) -> None:
    """Block until the next bar is confirmed closed."""
    wait = seconds_until_bar_close(timeframe) + buffer_seconds
    logger.info(f"Next bar in {wait:.0f}s — sleeping…")
    time.sleep(wait)


# ── Signal processing ─────────────────────────────────────────────────────────

def process_signals(
    df: pd.DataFrame,
    manager: TradeManager,
    config: BotConfig,
    bar_index: int,
    atr_mult: float,
) -> None:
    """
    Evaluate signals on the last confirmed bar and open pending orders if
    the score meets the threshold.
    """
    if manager.has_trade or manager.has_pending:
        return  # Already in a trade or pending — do not stack

    if len(df) < 2:
        return

    row      = df.iloc[-1]
    prev_row = df.iloc[-2]

    signals = get_signals(row, prev_row, config)

    for sig in signals:
        if not sig.auto_trade:
            logger.info(f"Signal {sig.side} setup={sig.setup} score={sig.score:.0f} — below threshold ({config.auto_trade_threshold}), skipped")
            continue

        tick  = config.tick_size
        is_long = sig.side == "long"
        entry = round_tick(
            sig.retest_level - config.limit_offset_ticks * tick if is_long
            else sig.retest_level + config.limit_offset_ticks * tick,
            tick,
        )

        stop = get_stop_price(
            is_long=is_long,
            entry=entry,
            atr=float(row["atr"]),
            swing_low=float(row["swing_low"]),
            swing_high=float(row["swing_high"]),
            prev_low=float(row["prev_low"]),
            band_dn=float(row["band_dn"]),
            pdl=float(row["pdl"]),
            prev_high=float(row["prev_high"]),
            band_up=float(row["band_up"]),
            pdh=float(row["pdh"]),
            config=config,
            atr_mult=atr_mult,
        )

        qty  = calc_qty(entry, stop, config)
        tp1, tp2, tp3 = calc_tp_levels(is_long, entry, stop, config)

        if config.adaptive_cancel:
            avg_tr  = float(row["atr"])
            n_bars  = calc_cancel_bars_dyn(float(row["close"]), sig.retest_level, avg_tr, config)
        else:
            n_bars = config.cancel_bars

        expiry = bar_index + n_bars

        logger.info(
            f"Signal {sig.side.upper()} | setup={sig.setup} score={sig.score:.0f} | "
            f"entry={entry} stop={stop} tp1={tp1} tp2={tp2} tp3={tp3} qty={qty} expiry_bar={expiry}"
        )

        manager.open_pending(
            side=sig.side,
            retest_level=sig.retest_level,
            stop=stop,
            tp1=tp1, tp2=tp2, tp3=tp3,
            qty=qty,
            expiry_bar=expiry,
            score=sig.score,
        )
        break  # one pending at a time


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_live(config: BotConfig, paper: bool = False) -> None:
    logger.info(f"Starting ETH Scalper Bot — mode={'PAPER' if paper else 'LIVE'}")

    exchange = make_exchange(config)

    if paper:
        exec_engine = PaperExecutionEngine(config)
    else:
        exec_engine = ExecutionEngine(exchange, config)
        exec_engine.set_leverage(config.leverage)

    manager   = TradeManager(exec_engine, config)
    atr_mult  = get_atr_mult(config.timeframe, config)
    bar_index = 0
    current_day = ""

    while True:
        try:
            # ── P8: Daily reset ───────────────────────────────────────────────
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != current_day:
                current_day = today
                if paper:
                    exec_engine.reset_daily_stats()  # type: ignore[union-attr]
                logger.info(f"New trading day: {today}")

            # ── P8: Circuit breaker check ─────────────────────────────────────
            if paper:
                if exec_engine.daily_loss_limit_hit():  # type: ignore[union-attr]
                    logger.warning("Daily loss limit hit — no new signals until next day")
                    manager._cancel_pending()
                    sleep_until_next_bar(config.timeframe)
                    continue
                if exec_engine.daily_trade_limit_hit():  # type: ignore[union-attr]
                    logger.warning("Daily trade limit reached — no new signals until next day")
                    sleep_until_next_bar(config.timeframe)
                    continue

            # ── P9: Funding rate blackout ─────────────────────────────────────
            if is_in_funding_blackout(config):
                logger.info("Funding rate blackout — skipping signal generation this bar")
                manager.update_expiry_only(bar_index)
                sleep_until_next_bar(config.timeframe)
                bar_index += 1
                continue

            # ── Fetch data ────────────────────────────────────────────────────
            df = fetch_ohlcv(exchange, config.symbol, config.timeframe, limit=config.ohlcv_limit)

            daily_df = fetch_daily_ohlcv(exchange, config.symbol, limit=5)
            if not daily_df.empty:
                last_daily = daily_df.iloc[-2]  # previous completed day
                pdh = float(last_daily["high"])
                pdl = float(last_daily["low"])
            else:
                pdh = pdl = float("nan")

            df = calc_indicators(df, config, pdh=pdh, pdl=pdl)

            # ── P3: Inject HTF trend ──────────────────────────────────────────
            df = inject_htf_trend(df, exchange, config)

            df = detect_setups_df(df, config)
            df.dropna(subset=["atr", "ema_fast"], inplace=True)

            if len(df) < 3:
                logger.warning("Not enough bars after indicator warmup — skipping")
                sleep_until_next_bar(config.timeframe)
                continue

            # ── Paper: simulate current bar fills ─────────────────────────────
            if paper:
                exec_engine.simulate_bar(df.iloc[-1])  # type: ignore[union-attr]

            # ── Trade manager update (TP/SL/BE/expiry) ────────────────────────
            manager.update(df.iloc[-1], bar_index)

            # ── Signal processing ─────────────────────────────────────────────
            process_signals(df, manager, config, bar_index, atr_mult)

            # ── Heartbeat ─────────────────────────────────────────────────────
            if paper:
                summary = exec_engine.summary()  # type: ignore[union-attr]
                logger.info(
                    f"[PAPER] Balance=${summary.get('balance', 0):.2f} | "
                    f"Trades={summary.get('trades', 0)} | WR={summary.get('win_rate', 0)}% | "
                    f"DailyLoss=${exec_engine.daily_loss():.2f}"  # type: ignore[union-attr]
                )
            else:
                bal = exec_engine.get_balance()
                row = df.iloc[-1]
                atr_pct = float(row['atr']) / float(row['close']) * 100
                ema_spread_pct = abs(float(row['ema_fast']) - float(row['ema_slow'])) / float(row['close']) * 100
                logger.info(
                    f"[LIVE] Balance=${bal:.2f} | ETH=${float(row['close']):.2f} | "
                    f"ATR%={atr_pct:.3f}% | EMAspread%={ema_spread_pct:.3f}%"
                )
                # Setup diagnostics — show what's firing/blocking each bar
                logger.info(
                    f"[DIAG] sweep_l={bool(row.get('long_sweep',False))} "
                    f"sweep_s={bool(row.get('short_sweep',False))} "
                    f"vmr_l={bool(row.get('long_vmr',False))} "
                    f"vmr_s={bool(row.get('short_vmr',False))} "
                    f"mom_l={bool(row.get('long_mom',False))} "
                    f"mom_s={bool(row.get('short_mom',False))} "
                    f"vol_ok={bool(row.get('vol_ok',False))} "
                    f"htf_l={bool(row.get('htf_trend_long',True))} "
                    f"htf_s={bool(row.get('htf_trend_short',True))}"
                )

            bar_index += 1

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            if paper:
                print("\n── Final Paper Trading Summary ──────────────")
                for k, v in exec_engine.summary().items():  # type: ignore[union-attr]
                    print(f"  {k:<18} {v}")
                print("─────────────────────────────────────────────")
            break
        except Exception as exc:
            logger.error(f"Loop error: {exc}", exc_info=True)

        sleep_until_next_bar(config.timeframe)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETH Scalper Bot")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper",     action="store_true", help="Paper trading mode")
    mode.add_argument("--live",      action="store_true", help="Live trading mode (requires API keys)")
    parser.add_argument("--yes",       action="store_true", help="Skip live mode confirmation (for systemd/non-interactive)")
    mode.add_argument("--backtest",  action="store_true", help="Run historical backtest and exit")
    parser.add_argument("--days",       type=int,   default=180,   help="Backtest history days")
    parser.add_argument("--timeframe",  type=str,   default=None,  help="Override timeframe (default: from config.py)")
    parser.add_argument("--risk",       type=float, default=None,  help="Override risk per trade %%")
    parser.add_argument("--leverage",   type=int,   default=None)
    parser.add_argument("--deposit",    type=float, default=None)
    parser.add_argument("--threshold",  type=int,   default=None,  help="Override auto-trade score threshold")
    args = parser.parse_args()

    # Build config from config.py defaults; only override if explicitly passed on CLI
    config = BotConfig()
    if args.timeframe is not None:
        config.timeframe = args.timeframe
    if args.risk is not None:
        config.risk_pct = args.risk
    if args.leverage is not None:
        config.leverage = args.leverage
    if args.deposit is not None:
        config.init_dep = args.deposit
    if args.threshold is not None:
        config.auto_trade_threshold = args.threshold

    if args.backtest:
        results = run_backtest(config, days=args.days)
        print("\n── Backtest Results ──────────────────────────────")
        for k, v in results.items():
            print(f"  {k:<18} {v}")
        print("─────────────────────────────────────────────────")
    elif args.paper:
        run_live(config, paper=True)
    elif args.live:
        if not args.yes:
            confirm = input("⚠  LIVE trading mode — this will place REAL orders on HTX. Type YES to continue: ")
            if confirm.strip().upper() != "YES":
                print("Aborted.")
                return
        run_live(config, paper=False)


if __name__ == "__main__":
    main()
