"""
backtest.py — Historical simulation over a pre-loaded OHLCV DataFrame.

Usage:
    python backtest.py --days 180
    python backtest.py --days 365 --timeframe 15m

The backtest fetches historical data from HTX, runs indicators and the
strategy engine bar-by-bar, and simulates fills using paper_trader logic.

Metrics printed at the end:
  - Total trades, Win rate, Avg RR
  - Total PnL, Max drawdown
  - Sharpe ratio (daily returns)
  - Trades per day
"""

import argparse
import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import BotConfig, cfg as default_cfg
from market_data import calc_indicators, make_exchange, fetch_ohlcv, fetch_daily_ohlcv, get_atr_mult, inject_htf_trend
from strategy_engine import detect_setups_df, calc_score, get_retest_level, calc_cancel_bars_dyn, SetupFlags, get_signals
from risk_engine import get_stop_price, calc_qty, calc_tp_levels, round_tick
from paper_trader import PaperExecutionEngine

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_flags_from_row(row: pd.Series) -> SetupFlags:
    f = SetupFlags()
    f.long_sweep  = bool(row.get("long_sweep",  False))
    f.short_sweep = bool(row.get("short_sweep", False))
    f.long_vmr    = bool(row.get("long_vmr",    False))
    f.short_vmr   = bool(row.get("short_vmr",   False))
    f.long_mom    = bool(row.get("long_mom",    False))
    f.short_mom   = bool(row.get("short_mom",   False))
    f.vol_ok      = bool(row.get("vol_ok",      False))
    return f


# ── Backtest state ────────────────────────────────────────────────────────────

class BacktestState:
    """Lightweight equivalent of TradeManager for the backtester."""

    def __init__(self, paper: PaperExecutionEngine, config: BotConfig):
        self.paper = paper
        self.cfg = config
        self.pending_side: str = ""
        self.pending_entry: float = 0.0
        self.pending_expiry: int = 0
        self.pending_stop: float = 0.0
        self.pending_tp1: float = 0.0
        self.pending_tp2: float = 0.0
        self.pending_tp3: float = 0.0
        self.pending_qty: float = 0.0

        self.in_trade: bool = False
        self.trade_side: str = ""
        self.trade_entry: float = 0.0
        self.trade_stop: float = 0.0
        self.trade_tp1: float = 0.0
        self.trade_tp2: float = 0.0
        self.trade_tp3: float = 0.0
        self.trade_qty: float = 0.0
        self.trade_rem_qty: float = 0.0   # P2: remaining after partial closes
        self.be_moved: bool = False
        self.tp1_hit: bool = False        # P2
        self.tp2_hit: bool = False        # P2/P7
        self.balance_history: list[float] = [config.init_dep]
        # P8: daily tracking
        self._current_day: str = ""

    def open_pending(self, side, entry, stop, tp1, tp2, tp3, qty, expiry):
        self.pending_side   = side
        self.pending_entry  = entry
        self.pending_stop   = stop
        self.pending_tp1    = tp1
        self.pending_tp2    = tp2
        self.pending_tp3    = tp3
        self.pending_qty    = qty
        self.pending_expiry = expiry

    def cancel_pending(self):
        self.pending_side = ""

    def update(self, row: pd.Series, bar_index: int) -> None:
        high, low = row["high"], row["low"]

        # Check pending fill
        if self.pending_side and not self.in_trade:
            filled = (
                (self.pending_side == "long"  and low  <= self.pending_entry) or
                (self.pending_side == "short" and high >= self.pending_entry)
            )
            if filled:
                self.in_trade      = True
                self.trade_side    = self.pending_side
                self.trade_entry   = self.pending_entry
                self.trade_stop    = self.pending_stop
                self.trade_tp1     = self.pending_tp1
                self.trade_tp2     = self.pending_tp2
                self.trade_tp3     = self.pending_tp3
                self.trade_qty     = self.pending_qty
                self.trade_rem_qty = self.pending_qty   # P2: tracks remaining after partials
                self.be_moved      = False
                self.tp1_hit       = False
                self.tp2_hit       = False
                self.pending_side  = ""
            elif bar_index > self.pending_expiry:
                self.pending_side = ""

        # Check active trade
        if self.in_trade:
            c = self.cfg
            side = self.trade_side
            atr_val = float(row.get("atr", 0))

            # Stop hit → close remaining
            if (side == "long" and low <= self.trade_stop) or (side == "short" and high >= self.trade_stop):
                self._close("stop", self.trade_stop, self.trade_rem_qty)
                return
            # TP3 → close remaining
            if (side == "long" and high >= self.trade_tp3) or (side == "short" and low <= self.trade_tp3):
                self._close("tp3", self.trade_tp3, self.trade_rem_qty)
                return

            # P2: TP1 → partial close 25% + move stop to BE
            if not self.tp1_hit:
                tp1_hit = (side == "long" and high >= self.trade_tp1) or (side == "short" and low <= self.trade_tp1)
                if tp1_hit:
                    self.tp1_hit = True
                    self.be_moved = True
                    partial = round(self.trade_qty * c.partial_close_tp1_pct / c.qty_step) * c.qty_step
                    partial = max(c.qty_step, min(partial, self.trade_rem_qty))
                    self._partial_close("tp1", self.trade_tp1, partial)
                    self.trade_rem_qty = max(c.qty_step, round((self.trade_rem_qty - partial) / c.qty_step) * c.qty_step)
                    self.trade_stop = self.trade_entry

            # P2: TP2 → partial close 50% of original
            if self.tp1_hit and not self.tp2_hit:
                tp2_hit = (side == "long" and high >= self.trade_tp2) or (side == "short" and low <= self.trade_tp2)
                if tp2_hit:
                    self.tp2_hit = True
                    partial = round(self.trade_qty * c.partial_close_tp2_pct / c.qty_step) * c.qty_step
                    partial = max(c.qty_step, min(partial, self.trade_rem_qty))
                    self._partial_close("tp2", self.trade_tp2, partial)
                    self.trade_rem_qty = max(c.qty_step, round((self.trade_rem_qty - partial) / c.qty_step) * c.qty_step)

            # P7: trailing stop after TP2
            if self.tp2_hit and c.trail_after_tp2 and atr_val > 0:
                trail_dist = atr_val * c.trail_atr_mult
                if side == "long":
                    new_trail = round(round(row["close"] - trail_dist, 2) / c.tick_size) * c.tick_size
                    if new_trail > self.trade_stop:
                        self.trade_stop = new_trail
                else:
                    new_trail = round(round(row["close"] + trail_dist, 2) / c.tick_size) * c.tick_size
                    if new_trail < self.trade_stop:
                        self.trade_stop = new_trail

    def _partial_close(self, reason: str, exit_price: float, qty: float) -> None:
        self.paper.record_trade(
            side=self.trade_side,
            entry=self.trade_entry,
            exit_price=exit_price,
            qty=qty,
            exit_reason=reason,
        )
        self.balance_history.append(self.paper._balance)

    def _close(self, reason: str, exit_price: float, qty: float) -> None:
        self.paper.record_trade(
            side=self.trade_side,
            entry=self.trade_entry,
            exit_price=exit_price,
            qty=qty,
            exit_reason=reason,
        )
        self.balance_history.append(self.paper._balance)
        self.in_trade = False


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(paper: PaperExecutionEngine, balance_history: list[float], n_days: float) -> dict:
    s = paper.summary()
    if s["trades"] == 0:
        return s

    bal = np.array(balance_history)
    daily_returns = np.diff(bal) / bal[:-1]
    sharpe = (
        (daily_returns.mean() / daily_returns.std() * math.sqrt(252))
        if daily_returns.std() > 0 else 0.0
    )

    # Max drawdown
    peak = bal[0]
    max_dd = 0.0
    for b in bal:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_dd:
            max_dd = dd

    s["sharpe"]       = round(sharpe, 2)
    s["max_drawdown"] = round(max_dd * 100, 2)  # %
    s["trades_per_day"] = round(s["trades"] / max(n_days, 1), 2)
    return s


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(config: BotConfig = default_cfg, days: int = 180) -> dict:
    print(f"Fetching {days} days of {config.timeframe} data from HTX…")
    exchange = make_exchange(config)

    # Bars needed: days × bars_per_day
    bars_per_day = {"15m": 96, "1h": 24, "30m": 48}.get(config.timeframe, 96)
    limit = min(days * bars_per_day + 200, 1000)  # ccxt max varies; chunk if needed

    df = fetch_ohlcv(exchange, config.symbol, config.timeframe, limit=limit)

    # Daily bars for PDH/PDL
    daily_df = fetch_daily_ohlcv(exchange, config.symbol, limit=days + 5)

    # Map each intraday bar to previous day H/L
    def get_pdh_pdl(ts: pd.Timestamp):
        day = ts.floor("D")
        prev_daily = daily_df[daily_df.index < day]
        if prev_daily.empty:
            return float("nan"), float("nan")
        return prev_daily.iloc[-1]["high"], prev_daily.iloc[-1]["low"]

    df["pdh"], df["pdl"] = zip(*[get_pdh_pdl(ts) for ts in df.index])

    df = calc_indicators(df, config)
    # P3: inject HTF trend from pre-fetched data (no live exchange call needed)
    df = inject_htf_trend(df, exchange, config)
    df = detect_setups_df(df, config)
    df.dropna(subset=["atr", "ema_fast", "ema_slow", "vwap"], inplace=True)

    print(f"Running backtest on {len(df)} bars…")

    paper  = PaperExecutionEngine(config, log_path="backtest_trades.csv")
    state  = BacktestState(paper, config)
    atr_mult = get_atr_mult(config.timeframe, config)

    for bar_index, (ts, row) in enumerate(df.iterrows()):
        state.update(row, bar_index)

        # P8: daily reset
        day_str = ts.strftime("%Y-%m-%d")
        if day_str != state._current_day:
            state._current_day = day_str
            paper.reset_daily_stats()

        # P8: circuit breaker — skip signals if daily limit hit
        if paper.daily_loss_limit_hit() or paper.daily_trade_limit_hit():
            continue

        # Only generate new signals when flat (no active trade / pending)
        if state.in_trade or state.pending_side:
            continue

        flags = _make_flags_from_row(row)

        for is_long in (True, False):
            # Side filter
            if is_long and config.side_filter == "Short only":
                continue
            if not is_long and config.side_filter == "Long only":
                continue

            # Setup active?
            active = (is_long and flags.raw_long) or (not is_long and flags.raw_short)
            if not active:
                continue

            score = calc_score(is_long, flags, row, config)
            if score < config.auto_trade_threshold:
                continue

            retest = get_retest_level(is_long, flags, row)
            if retest is None:
                continue

            tick = config.tick_size
            entry = round_tick(
                retest - config.limit_offset_ticks * tick if is_long
                else retest + config.limit_offset_ticks * tick,
                tick,
            )

            stop = get_stop_price(
                is_long=is_long,
                entry=entry,
                atr=row["atr"],
                swing_low=row["swing_low"],
                swing_high=row["swing_high"],
                prev_low=row["prev_low"],
                band_dn=row["band_dn"],
                pdl=row["pdl"],
                prev_high=row["prev_high"],
                band_up=row["band_up"],
                pdh=row["pdh"],
                config=config,
                atr_mult=atr_mult,
            )

            qty = calc_qty(entry, stop, config)
            tp1, tp2, tp3 = calc_tp_levels(is_long, entry, stop, config)

            if config.adaptive_cancel:
                avg_tr = row["atr"]  # using ATR as proxy for avg TR
                expiry = bar_index + calc_cancel_bars_dyn(row["close"], retest, avg_tr, config)
            else:
                expiry = bar_index + config.cancel_bars

            state.open_pending("long" if is_long else "short", entry, stop, tp1, tp2, tp3, qty, expiry)
            break  # one signal per bar

    n_days = len(df) / bars_per_day
    metrics = compute_metrics(paper, state.balance_history, n_days)
    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETH Scalper Backtest")
    parser.add_argument("--days", type=int, default=180, help="History days to test")
    parser.add_argument("--timeframe", type=str, default="15m")
    args = parser.parse_args()

    config = BotConfig(timeframe=args.timeframe)
    results = run_backtest(config, days=args.days)

    print("\n── Backtest Results ──────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<18} {v}")
    print("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
