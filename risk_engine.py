"""
risk_engine.py — Stop price calculation, position sizing, and TP levels.

Mirrors Pine Script getStopPrice(), calcQty(), and TP formulas from context.md.
"""

import math
from typing import Optional

from config import BotConfig, cfg as default_cfg


# ── Tick rounding ─────────────────────────────────────────────────────────────

def round_tick(price: float, tick_size: float) -> float:
    return round(round(price / tick_size) * tick_size, 10)


def round_qty(qty: float, qty_step: float) -> float:
    return max(qty_step, round(round(qty / qty_step) * qty_step, 10))


# ── Liquidity level helpers ───────────────────────────────────────────────────

def nearest_below(price: float, prev_low: float, band_dn: float, pdl: float) -> Optional[float]:
    """
    Returns the nearest key level below `price`.
    Mirrors Pine nearestBelow(): considers prev_low, band_dn, pdl.
    """
    candidates = []
    for level in (prev_low, band_dn, pdl):
        if not math.isnan(level) and level < price:
            candidates.append(level)
    return max(candidates) if candidates else None


def nearest_above(price: float, prev_high: float, band_up: float, pdh: float) -> Optional[float]:
    """
    Returns the nearest key level above `price`.
    Mirrors Pine nearestAbove(): considers prev_high, band_up, pdh.
    """
    candidates = []
    for level in (prev_high, band_up, pdh):
        if not math.isnan(level) and level > price:
            candidates.append(level)
    return min(candidates) if candidates else None


# ── Stop price ────────────────────────────────────────────────────────────────

def get_stop_price(
    is_long: bool,
    entry: float,
    atr: float,
    swing_low: float,
    swing_high: float,
    prev_low: float,
    band_dn: float,
    pdl: float,
    prev_high: float,
    band_up: float,
    pdh: float,
    config: BotConfig = default_cfg,
    atr_mult: Optional[float] = None,
) -> float:
    """
    Full stop price logic matching Pine getStopPrice():
      1. ATR stop
      2. Swing stop (last swing low/high ± padding)
      3. Base = ATR | Swing | Hybrid (min/max)
      4. Liquidity stop (nearest level beyond entry)
      5. Buffer ticks added
      6. Cap at stop_cap_atr_mult × ATR
    """
    c = config
    tick = c.tick_size
    if atr_mult is None:
        atr_mult = c.atr_mult_15m

    # 1. ATR stop
    atr_stop = entry - atr * atr_mult if is_long else entry + atr * atr_mult

    # 2. Swing stop
    swing_stop: Optional[float] = None
    if c.sl_mode != "ATR":
        sw = swing_low if is_long else swing_high
        if not math.isnan(sw):
            pad = atr * c.pad_swing_atr
            swing_stop = sw - pad if is_long else sw + pad

    # 3. Base
    if c.sl_mode == "ATR":
        base = atr_stop
    elif c.sl_mode == "Swing":
        base = swing_stop if swing_stop is not None else atr_stop
    else:  # Hybrid
        if swing_stop is None:
            base = atr_stop
        else:
            base = min(atr_stop, swing_stop) if is_long else max(atr_stop, swing_stop)

    # 4. Liquidity stop
    liq_stop: Optional[float] = None
    if c.use_liq_stop:
        if is_long:
            lb = nearest_below(entry, prev_low, band_dn, pdl)
            if lb is not None:
                liq_stop = lb - c.liq_pad_ticks * tick
        else:
            la = nearest_above(entry, prev_high, band_up, pdh)
            if la is not None:
                liq_stop = la + c.liq_pad_ticks * tick

    if liq_stop is None:
        with_liq = base
    else:
        with_liq = min(base, liq_stop) if is_long else max(base, liq_stop)

    # 5. Buffer
    with_buf = with_liq - c.stop_buffer_ticks * tick if is_long else with_liq + c.stop_buffer_ticks * tick

    # 6. Cap
    cap = atr * c.stop_cap_atr_mult
    if is_long:
        capped = max(entry - cap, with_buf)
    else:
        capped = min(entry + cap, with_buf)

    return round_tick(capped, tick)


# ── Position sizing ───────────────────────────────────────────────────────────

def calc_qty(
    entry: float,
    stop: float,
    config: BotConfig = default_cfg,
) -> float:
    """
    Mirrors Pine calcQty():
      qty = (deposit × risk_pct / 100) / stop_distance × leverage
    """
    c = config
    dist = abs(entry - stop)
    if dist == 0:
        return c.qty_step
    risk_usdt = c.init_dep * (c.risk_pct / 100.0)
    base_qty = risk_usdt / dist
    qty = base_qty * c.leverage
    return round_qty(qty, c.qty_step)


# ── TP levels ─────────────────────────────────────────────────────────────────

def calc_tp_levels(
    is_long: bool,
    entry: float,
    stop: float,
    config: BotConfig = default_cfg,
) -> tuple[float, float, float]:
    """Returns (tp1, tp2, tp3) rounded to tick."""
    c = config
    tick = c.tick_size
    dist = abs(entry - stop)
    sign = 1 if is_long else -1
    tp1 = round_tick(entry + sign * c.r_tp1 * dist, tick)
    tp2 = round_tick(entry + sign * c.r_tp2 * dist, tick)
    tp3 = round_tick(entry + sign * c.r_tp3 * dist, tick)
    return tp1, tp2, tp3


# ── Estimated risk from qty ───────────────────────────────────────────────────

def est_risk_from_qty(entry: float, stop: float, qty: float) -> float:
    """Mirrors Pine estRiskFromQty()."""
    return abs(entry - stop) * qty
