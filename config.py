from dataclasses import dataclass, field


@dataclass
class BotConfig:
    # ── Exchange ──────────────────────────────────────────────────────────────
    symbol: str = "ETH/USDT:USDT"       # CCXT unified symbol (perpetual swap)
    timeframe: str = "5m"               # Primary timeframe (5m = 288 bars/day vs 15m = 96)
    ohlcv_limit: int = 500              # Bars to fetch each cycle

    # ── Entry ─────────────────────────────────────────────────────────────────
    entry_mode: str = "retest"          # "retest" | "bar_close"
    setups_opt: str = "All"             # "All" | "Sweep&Reversal" | "VWAP Mean Revert" | "Momentum Pullback"
    side_filter: str = "Both"           # "Both" | "Long only" | "Short only"
    cancel_bars: int = 5                # Static pending order lifetime (bars)
    limit_offset_ticks: int = 0         # Limit price offset (ticks) from retest level
    close_only: bool = True             # Only fire signals on confirmed (closed) bars

    # ── Adaptive cancel ───────────────────────────────────────────────────────
    adaptive_cancel: bool = True
    cancel_tr_len: int = 14             # ATR/TR length for adaptive cancel
    cancel_scale: float = 1.5
    cancel_min: int = 2
    cancel_max: int = 8

    # ── Risk ──────────────────────────────────────────────────────────────────
    init_dep: float = 20.0              # Deposit (USDT)
    risk_pct: float = 1.0               # Risk per trade (%)
    leverage: int = 10
    qty_step: float = 0.01              # Minimum lot size increment (1 HTX contract = 0.01 ETH)

    # ── Stop ─────────────────────────────────────────────────────────────────
    sl_mode: str = "Hybrid"             # "ATR" | "Swing" | "Hybrid"
    atr_len: int = 14
    atr_mult_15m: float = 1.0
    atr_mult_1h: float = 1.5
    pad_swing_atr: float = 0.2          # Swing stop padding (× ATR)
    stop_buffer_ticks: int = 4
    use_liq_stop: bool = True           # Push stop beyond nearest liquidity level
    liq_pad_ticks: int = 3
    stop_cap_atr_mult: float = 2.5      # Max stop distance (× ATR)

    # ── Take profit ───────────────────────────────────────────────────────────
    r_tp1: float = 1.3                  # TP1 in R multiples
    r_tp2: float = 1.8
    r_tp3: float = 3.0

    # ── Indicators ────────────────────────────────────────────────────────────
    ema_fast_len: int = 20
    ema_slow_len: int = 50
    dev_len: int = 100                  # VWAP stdev window
    sigma_k: float = 1.0               # VWAP band width (sigmas)
    vol_mult: float = 0.8              # Volume multiplier for sweep detection
    sweep_len: int = 15                 # Lookback bars for range high/low
    min_dev_pct_vw: float = 0.08       # Min deviation from VWAP (%) for VMR

    # ── Scoring weights ───────────────────────────────────────────────────────
    w_sweep: int = 40
    w_vmr: int = 30
    w_mom: int = 30
    w_vol: int = 15
    w_vwap: int = 10
    w_ema: int = 10
    auto_trade_threshold: int = 45

    # ── P1: Scoring — confluence bonus ────────────────────────────────────────
    w_confluence: int = 20              # Bonus when 2+ setups fire on the same bar

    # ── P2: Partial closes ────────────────────────────────────────────────────
    partial_close_tp1_pct: float = 0.25   # Close 25% of position at TP1
    partial_close_tp2_pct: float = 0.50   # Close 50% of original position at TP2

    # ── P3: Higher-timeframe (HTF) trend filter ───────────────────────────────
    htf_filter: bool = True
    htf_timeframe: str = "15m"
    htf_ema_fast_len: int = 20
    htf_ema_slow_len: int = 50

    # ── P4: ATR volatility regime gate ────────────────────────────────────────
    vol_gate_enabled: bool = True
    min_atr_pct: float = 0.05           # Skip signals when ATR/close < this (dead market)
    max_atr_pct: float = 1.50           # Skip signals when ATR/close > this (news/cascade)

    # ── P5: Sweep depth quality filter ───────────────────────────────────────
    sweep_depth_filter: bool = True
    min_sweep_depth_atr: float = 0.10   # Sweep must extend >= 0.10 × ATR beyond range

    # ── P6: EMA spread anti-chop filter ──────────────────────────────────────
    chop_filter_enabled: bool = True
    min_ema_spread_pct: float = 0.005   # |(ema_fast - ema_slow)| / close must exceed this

    # ── P7: ATR trailing stop after TP2 ──────────────────────────────────────
    trail_after_tp2: bool = True
    trail_atr_mult: float = 1.0         # Trail stop by N × ATR from close after TP2 hit

    # ── P8: Daily loss circuit breaker ───────────────────────────────────────
    daily_loss_limit_pct: float = 10.0  # Pause trading when daily loss >= this % of deposit
    max_trades_per_day: int = 8         # Hard cap on daily trade count

    # ── P9: Funding rate session blackout ─────────────────────────────────────
    funding_blackout_enabled: bool = True
    funding_blackout_mins: int = 8      # Silence signals N minutes around funding (0,8,16 UTC)

    # ── Commission (for P&L simulation) ──────────────────────────────────────
    maker_fee_pct: float = 0.02         # 0.02%
    taker_fee_pct: float = 0.055        # 0.055%

    # ── Misc ──────────────────────────────────────────────────────────────────
    tick_size: float = 0.01             # ETH/USDT min price increment
    rsi_len: int = 14
    vol_sma_len: int = 20               # Volume SMA length for sweep check


# Singleton default config — import and override fields as needed
cfg = BotConfig()
