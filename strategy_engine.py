"""
strategy_engine.py — Setup detection and signal scoring.

Logic mirrors the Pine Script in context.md exactly:
  - Sweep & Reversal
  - VWAP Mean Reversion
  - Momentum Pullback
  - Composite scoring
  - Retest level selection
  - Adaptive cancel bars
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from config import BotConfig, cfg as default_cfg


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SetupFlags:
    long_sweep: bool = False
    short_sweep: bool = False
    long_vmr: bool = False
    short_vmr: bool = False
    long_mom: bool = False
    short_mom: bool = False
    vol_ok: bool = False
    # P3: HTF trend alignment
    htf_trend_long: bool = True   # True = neutral/bullish on HTF (default allow)
    htf_trend_short: bool = True  # True = neutral/bearish on HTF (default allow)

    @property
    def raw_long(self) -> bool:
        return self.long_sweep or self.long_vmr or self.long_mom

    @property
    def raw_short(self) -> bool:
        return self.short_sweep or self.short_vmr or self.short_mom


@dataclass
class Signal:
    side: str            # "long" | "short"
    setup: str           # "Sweep&Reversal" | "VWAP Mean Revert" | "Momentum Pullback"
    score: float
    retest_level: float  # limit entry price (before tick offset)
    auto_trade: bool     # score >= threshold


# ── Crossover helpers ─────────────────────────────────────────────────────────

def _crossover(series: pd.Series, reference: pd.Series) -> pd.Series:
    """True where series crosses above reference."""
    return (series > reference) & (series.shift(1) <= reference.shift(1))


def _crossunder(series: pd.Series, reference: pd.Series) -> pd.Series:
    """True where series crosses below reference."""
    return (series < reference) & (series.shift(1) >= reference.shift(1))


# ── Vectorised setup detection (full DataFrame) ───────────────────────────────

def detect_setups_df(df: pd.DataFrame, config: BotConfig = default_cfg) -> pd.DataFrame:
    """
    Add boolean setup columns to the DataFrame.
    Useful for backtesting over all bars at once.
    """
    c = config

    vol_ok = df["volume"] > df["vol_sma"] * c.vol_mult

    # P4: ATR volatility regime gate
    if c.vol_gate_enabled and "atr" in df.columns:
        atr_pct = df["atr"] / df["close"] * 100
        vol_regime_ok = (atr_pct >= c.min_atr_pct) & (atr_pct <= c.max_atr_pct)
    else:
        vol_regime_ok = pd.Series(True, index=df.index)

    # P6: EMA spread anti-chop filter
    if c.chop_filter_enabled and "ema_fast" in df.columns and "ema_slow" in df.columns:
        ema_spread_pct = (df["ema_fast"] - df["ema_slow"]).abs() / df["close"] * 100
        not_choppy = ema_spread_pct >= c.min_ema_spread_pct
    else:
        not_choppy = pd.Series(True, index=df.index)

    # P3: HTF trend filter (columns injected by market_data if htf_filter=True)
    if c.htf_filter and "htf_trend_long" in df.columns:
        htf_long_ok  = df["htf_trend_long"].fillna(True).astype(bool)
        htf_short_ok = df["htf_trend_short"].fillna(True).astype(bool)
    else:
        htf_long_ok  = pd.Series(True, index=df.index)
        htf_short_ok = pd.Series(True, index=df.index)

    # Sweep & Reversal — requires vol_ok (0.8× avg) to filter dead-bar fakeouts
    long_sweep_raw  = (df["low"] < df["prev_low"]) & (df["close"] > df["prev_low"]) & vol_ok
    short_sweep_raw = (df["high"] > df["prev_high"]) & (df["close"] < df["prev_high"]) & vol_ok

    # P5: Sweep depth quality filter
    if c.sweep_depth_filter and "atr" in df.columns:
        long_sweep_depth  = (df["prev_low"] - df["low"]) >= (df["atr"] * c.min_sweep_depth_atr)
        short_sweep_depth = (df["high"] - df["prev_high"]) >= (df["atr"] * c.min_sweep_depth_atr)
    else:
        long_sweep_depth  = pd.Series(True, index=df.index)
        short_sweep_depth = pd.Series(True, index=df.index)

    long_sweep  = long_sweep_raw  & long_sweep_depth  & vol_regime_ok & not_choppy & htf_long_ok
    short_sweep = short_sweep_raw & short_sweep_depth & vol_regime_ok & not_choppy & htf_short_ok

    # VWAP Mean Reversion
    dev_pct = ((df["close"] - df["vwap"]) / df["vwap"]).abs() * 100
    long_vmr  = (
        (df["close"].shift(1) < df["band_dn"].shift(1))
        & (df["close"] > df["band_dn"])
        & (dev_pct >= c.min_dev_pct_vw)
        & vol_regime_ok & htf_long_ok
    )
    short_vmr = (
        (df["close"].shift(1) > df["band_up"].shift(1))
        & (df["close"] < df["band_up"])
        & (dev_pct >= c.min_dev_pct_vw)
        & vol_regime_ok & htf_short_ok
    )

    # Momentum Pullback
    long_mom  = (
        df["trend_long"]
        & _crossover(df["close"], df["ema_fast"])
        & (df["low"] <= df["ema_fast"])
        & (df["rsi"] > 50)
        & vol_regime_ok & not_choppy & htf_long_ok
    )
    short_mom = (
        df["trend_short"]
        & _crossunder(df["close"], df["ema_fast"])
        & (df["high"] >= df["ema_fast"])
        & (df["rsi"] < 50)
        & vol_regime_ok & not_choppy & htf_short_ok
    )

    df["vol_ok"]      = vol_ok
    df["long_sweep"]  = long_sweep
    df["short_sweep"] = short_sweep
    df["long_vmr"]    = long_vmr
    df["short_vmr"]   = short_vmr
    df["long_mom"]    = long_mom
    df["short_mom"]   = short_mom

    return df


# ── Single-bar setup detection (live use) ─────────────────────────────────────

def detect_setups_row(row: pd.Series, prev_row: pd.Series, config: BotConfig = default_cfg) -> SetupFlags:
    """
    Detect setups for the latest confirmed bar.
    `row`      — current bar (iloc[-1] after indicators)
    `prev_row` — previous bar (iloc[-2])
    """
    c = config

    flags = SetupFlags()
    flags.vol_ok = row["volume"] > row["vol_sma"] * c.vol_mult

    # P4: ATR volatility regime gate
    atr_val = float(row.get("atr", 0))
    close_val = float(row["close"])
    atr_pct = (atr_val / close_val * 100) if close_val != 0 else 0
    vol_regime_ok = (
        not c.vol_gate_enabled
        or (c.min_atr_pct <= atr_pct <= c.max_atr_pct)
    )

    # P6: EMA spread anti-chop filter
    ema_spread_pct = (
        abs(float(row["ema_fast"]) - float(row["ema_slow"])) / close_val * 100
        if close_val != 0 else 0
    )
    not_choppy = not c.chop_filter_enabled or (ema_spread_pct >= c.min_ema_spread_pct)

    # P3: HTF trend flags (injected by caller via row if available)
    flags.htf_trend_long  = bool(row.get("htf_trend_long",  True)) if c.htf_filter else True
    flags.htf_trend_short = bool(row.get("htf_trend_short", True)) if c.htf_filter else True

    # Sweep base conditions — requires vol_ok (0.8× avg) to filter dead-bar fakeouts
    long_sweep_base  = (row["low"] < row["prev_low"])  and (row["close"] > row["prev_low"])  and flags.vol_ok
    short_sweep_base = (row["high"] > row["prev_high"]) and (row["close"] < row["prev_high"]) and flags.vol_ok

    # P5: Sweep depth quality filter
    if c.sweep_depth_filter and atr_val > 0:
        long_depth_ok  = (float(row["prev_low"])  - float(row["low"]))  >= atr_val * c.min_sweep_depth_atr
        short_depth_ok = (float(row["high"]) - float(row["prev_high"])) >= atr_val * c.min_sweep_depth_atr
    else:
        long_depth_ok = short_depth_ok = True

    flags.long_sweep  = long_sweep_base  and long_depth_ok  and vol_regime_ok and not_choppy and flags.htf_trend_long
    flags.short_sweep = short_sweep_base and short_depth_ok and vol_regime_ok and not_choppy and flags.htf_trend_short

    # VWAP Mean Reversion
    dev_pct = abs((row["close"] - row["vwap"]) / row["vwap"]) * 100 if row["vwap"] != 0 else 0
    flags.long_vmr  = (
        (prev_row["close"] < prev_row["band_dn"])
        and (row["close"] > row["band_dn"])
        and (dev_pct >= c.min_dev_pct_vw)
        and vol_regime_ok and flags.htf_trend_long
    )
    flags.short_vmr = (
        (prev_row["close"] > prev_row["band_up"])
        and (row["close"] < row["band_up"])
        and (dev_pct >= c.min_dev_pct_vw)
        and vol_regime_ok and flags.htf_trend_short
    )

    # Momentum Pullback
    flags.long_mom  = (
        bool(row["trend_long"])
        and (prev_row["close"] <= prev_row["ema_fast"]) and (row["close"] > row["ema_fast"])
        and (row["low"] <= row["ema_fast"])
        and (row["rsi"] > 50)
        and vol_regime_ok and not_choppy and flags.htf_trend_long
    )
    flags.short_mom = (
        bool(row["trend_short"])
        and (prev_row["close"] >= prev_row["ema_fast"]) and (row["close"] < row["ema_fast"])
        and (row["high"] >= row["ema_fast"])
        and (row["rsi"] < 50)
        and vol_regime_ok and not_choppy and flags.htf_trend_short
    )

    return flags


# ── Setup selector (respects setups_opt and side_filter) ─────────────────────

def _select_setups(flags: SetupFlags, config: BotConfig):
    c = config
    opt = c.setups_opt

    want_long  = c.side_filter != "Short only"
    want_short = c.side_filter != "Long only"

    if opt == "All":
        sel_long  = flags.raw_long
        sel_short = flags.raw_short
    elif opt == "Sweep&Reversal":
        sel_long  = flags.long_sweep
        sel_short = flags.short_sweep
    elif opt == "VWAP Mean Revert":
        sel_long  = flags.long_vmr
        sel_short = flags.short_vmr
    elif opt == "Momentum Pullback":
        sel_long  = flags.long_mom
        sel_short = flags.short_mom
    else:
        sel_long = sel_short = False

    return sel_long and want_long, sel_short and want_short


# ── Scoring ───────────────────────────────────────────────────────────────────

def calc_score(is_long: bool, flags: SetupFlags, row: pd.Series, config: BotConfig = default_cfg) -> float:
    c = config

    if is_long:
        s_sweep = c.w_sweep if flags.long_sweep else 0
        s_vmr   = c.w_vmr   if flags.long_vmr   else 0
        s_mom   = c.w_mom   if flags.long_mom    else 0
        vol_bonus  = c.w_vol  if flags.long_sweep and flags.vol_ok else 0
        vwap_bonus = c.w_vwap if (row["vwap"] < row["close"] < row["band_up"]) else 0
        ema_bonus  = c.w_ema  if row["trend_long"] else 0
    else:
        s_sweep = c.w_sweep if flags.short_sweep else 0
        s_vmr   = c.w_vmr   if flags.short_vmr   else 0
        s_mom   = c.w_mom   if flags.short_mom    else 0
        vol_bonus  = c.w_vol  if flags.short_sweep and flags.vol_ok else 0
        vwap_bonus = c.w_vwap if (row["band_dn"] < row["close"] < row["vwap"]) else 0
        ema_bonus  = c.w_ema  if row["trend_short"] else 0

    # P1: Sum all active setup scores (was: max). Add confluence bonus when 2+ setups fire.
    setup_score = s_sweep + s_vmr + s_mom
    active_setups = sum([bool(s_sweep), bool(s_vmr), bool(s_mom)])
    confluence_bonus = c.w_confluence if active_setups >= 2 else 0
    total = setup_score + vol_bonus + vwap_bonus + ema_bonus + confluence_bonus
    return min(float(total), 100.0)


# ── Retest level ──────────────────────────────────────────────────────────────

def get_retest_level(is_long: bool, flags: SetupFlags, row: pd.Series) -> Optional[float]:
    """
    Returns the retest (limit entry) level for the given setup.
    Priority: Sweep > VMR > Mom (matches Pine getRetestLevelLong/Short).
    """
    if is_long:
        if flags.long_sweep:
            return float(row["prev_low"])
        if flags.long_vmr:
            return float(row["band_dn"])
        if flags.long_mom:
            return float(row["ema_fast"])
    else:
        if flags.short_sweep:
            return float(row["prev_high"])
        if flags.short_vmr:
            return float(row["band_up"])
        if flags.short_mom:
            return float(row["ema_fast"])
    return None


# ── Adaptive cancel bars ──────────────────────────────────────────────────────

def calc_cancel_bars_dyn(
    close: float,
    retest_level: float,
    avg_tr: float,
    config: BotConfig = default_cfg,
) -> int:
    c = config
    if avg_tr <= 0 or math.isnan(avg_tr):
        exp_bars = 0.0
    else:
        dist = abs(close - retest_level)
        exp_bars = math.ceil(dist / avg_tr)
    raw = round(c.cancel_scale * exp_bars)
    return int(max(c.cancel_min, min(c.cancel_max, raw)))


# ── High-level signal builder ─────────────────────────────────────────────────

def get_signals(
    row: pd.Series,
    prev_row: pd.Series,
    config: BotConfig = default_cfg,
) -> list[Signal]:
    """
    Returns 0-2 Signal objects for the confirmed bar.
    Called once per bar in the live loop.
    """
    flags = detect_setups_row(row, prev_row, config)
    sig_long, sig_short = _select_setups(flags, config)

    signals: list[Signal] = []

    for is_long, active in [(True, sig_long), (False, sig_short)]:
        if not active:
            continue
        retest = get_retest_level(is_long, flags, row)
        if retest is None:
            continue
        score = calc_score(is_long, flags, row, config)

        # Determine setup name
        if is_long:
            setup = "Sweep&Reversal" if flags.long_sweep else ("VWAP Mean Revert" if flags.long_vmr else "Momentum Pullback")
        else:
            setup = "Sweep&Reversal" if flags.short_sweep else ("VWAP Mean Revert" if flags.short_vmr else "Momentum Pullback")

        signals.append(Signal(
            side="long" if is_long else "short",
            setup=setup,
            score=score,
            retest_level=retest,
            auto_trade=score >= config.auto_trade_threshold,
        ))

    return signals
