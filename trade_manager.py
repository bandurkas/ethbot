"""
trade_manager.py — Active trade lifecycle management.

Mirrors Pine Script post-fill logic:
  - TP1 hit  → move stop to break-even
  - TP2/TP3  → (optional) partial close
  - Stop hit → close position
  - Pending retest order expiry → cancel

The manager holds at most one pending order and one active trade per side.
In practice the Pine script handles one side at a time; we follow the same
convention (new signal cancels previous pending of the opposite side).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import BotConfig, cfg as default_cfg
from execution_engine import ExecutionEngine
from risk_engine import round_tick, round_qty as _round_qty
from telegram_notify import notify_trade_open, notify_partial_close, notify_trade_close

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PendingOrder:
    side: str               # "long" | "short"
    order_id: str
    retest_level: float     # limit price
    expiry_bar: int         # bar_index after which the order is cancelled
    score: float


@dataclass
class ActiveTrade:
    side: str               # "long" | "short"
    entry: float
    stop: float
    tp1: float
    tp2: float
    tp3: float
    qty: float              # original full size
    remaining_qty: float = 0.0  # updated after partial closes
    stop_order_id: Optional[str] = None
    be_moved: bool = False
    tp1_hit: bool = False
    tp2_hit: bool = False
    fill_bar: int = 0

    def __post_init__(self):
        if self.remaining_qty == 0.0:
            self.remaining_qty = self.qty


# ── Trade Manager ─────────────────────────────────────────────────────────────

class TradeManager:
    """
    Manages pending limit orders and active trades.
    Call `update(row, bar_index)` once per confirmed bar.
    """

    def __init__(self, execution: ExecutionEngine, config: BotConfig = default_cfg):
        self.exec = execution
        self.cfg = config
        self.pending: Optional[PendingOrder] = None
        self.trade: Optional[ActiveTrade] = None

    # ── Public: place a new pending limit order ───────────────────────────────

    def open_pending(
        self,
        side: str,
        retest_level: float,
        stop: float,
        tp1: float,
        tp2: float,
        tp3: float,
        qty: float,
        expiry_bar: int,
        score: float,
    ) -> None:
        # Cancel any existing pending order first
        self._cancel_pending()
        # Cancel any active trade on opposite side (one direction at a time)
        if self.trade and self.trade.side != side:
            self._close_trade("new_signal_opposite_side", self.trade.entry)

        c_side = "buy" if side == "long" else "sell"
        tick = self.cfg.tick_size
        limit_price = round_tick(
            retest_level - self.cfg.limit_offset_ticks * tick if side == "long"
            else retest_level + self.cfg.limit_offset_ticks * tick,
            tick,
        )

        order = self.exec.place_limit_order(c_side, qty, limit_price)

        self.pending = PendingOrder(
            side=side,
            order_id=order["id"],
            retest_level=limit_price,
            expiry_bar=expiry_bar,
            score=score,
        )
        # Store TP/SL on the pending object so we can build the trade on fill
        self.pending._stop = stop          # type: ignore[attr-defined]
        self.pending._tp1  = tp1           # type: ignore[attr-defined]
        self.pending._tp2  = tp2           # type: ignore[attr-defined]
        self.pending._tp3  = tp3           # type: ignore[attr-defined]
        self.pending._qty  = qty           # type: ignore[attr-defined]
        logger.info(f"Pending {side} @ {limit_price} | expiry bar {expiry_bar}")

    # ── Public: bar update ────────────────────────────────────────────────────

    def update(self, row: pd.Series, bar_index: int) -> None:
        """
        Called once per confirmed bar. Checks:
          1. Has the pending order filled?
          2. Has the pending order expired?
          3. Have TP/SL levels been hit on the active trade?
        """
        if self.pending:
            self._check_pending_fill(row, bar_index)

        if self.trade:
            self._check_trade(row, bar_index)

    # ── Pending order handling ────────────────────────────────────────────────

    def _check_pending_fill(self, row: pd.Series, bar_index: int) -> None:
        p = self.pending
        assert p is not None

        filled = False
        if p.side == "long" and row["low"] <= p.retest_level:
            filled = True
        elif p.side == "short" and row["high"] >= p.retest_level:
            filled = True

        if filled:
            logger.info(f"Pending {p.side} filled @ {p.retest_level}")
            self.trade = ActiveTrade(
                side=p.side,
                entry=p.retest_level,
                stop=p._stop,          # type: ignore[attr-defined]
                tp1=p._tp1,            # type: ignore[attr-defined]
                tp2=p._tp2,            # type: ignore[attr-defined]
                tp3=p._tp3,            # type: ignore[attr-defined]
                qty=p._qty,            # type: ignore[attr-defined]
                remaining_qty=p._qty,  # type: ignore[attr-defined]
                fill_bar=bar_index,
            )
            # Place software stop (exchange stop if possible)
            stop_side = "sell" if p.side == "long" else "buy"
            stop_order = self.exec.place_stop_market_order(stop_side, p._qty, p._stop)  # type: ignore[attr-defined]
            if stop_order:
                self.trade.stop_order_id = stop_order["id"]
            self.pending = None
            notify_trade_open(self.trade.side, self.trade.entry, self.trade.stop, self.trade.tp3, self.trade.qty)
            return

        # Check expiry
        if bar_index > p.expiry_bar:
            logger.info(f"Pending {p.side} expired at bar {bar_index}")
            self._cancel_pending()

    # ── Active trade handling ─────────────────────────────────────────────────

    def _check_trade(self, row: pd.Series, bar_index: int) -> None:
        t = self.trade
        assert t is not None
        c = self.cfg

        high, low = row["high"], row["low"]
        tick = c.tick_size

        # Stop hit? Close remaining position.
        if t.side == "long" and low <= t.stop:
            logger.info(f"LONG stopped out @ {t.stop}")
            self._close_trade("stop_hit", t.stop)
            return
        if t.side == "short" and high >= t.stop:
            logger.info(f"SHORT stopped out @ {t.stop}")
            self._close_trade("stop_hit", t.stop)
            return

        # TP3 hit → close remaining position
        if t.side == "long" and high >= t.tp3:
            logger.info(f"LONG TP3 hit @ {t.tp3}")
            self._close_trade("tp3", t.tp3)
            return
        if t.side == "short" and low <= t.tp3:
            logger.info(f"SHORT TP3 hit @ {t.tp3}")
            self._close_trade("tp3", t.tp3)
            return

        # P2: TP1 hit → partial close (25%) + move stop to BE
        if not t.tp1_hit:
            tp1_hit = (t.side == "long" and high >= t.tp1) or (t.side == "short" and low <= t.tp1)
            if tp1_hit:
                t.tp1_hit = True
                t.be_moved = True

                # Partial close: 25% of original qty
                close_qty = _round_qty(t.qty * c.partial_close_tp1_pct, c.qty_step)
                close_side = "sell" if t.side == "long" else "buy"
                if close_qty >= c.qty_step:
                    self.exec.place_market_order(close_side, close_qty)
                    t.remaining_qty = max(c.qty_step, _round_qty(t.remaining_qty - close_qty, c.qty_step))
                    logger.info(f"TP1 partial close {close_qty} @ ~{t.tp1} | remaining={t.remaining_qty}")
                    notify_partial_close(t.side, "TP1", t.tp1, close_qty, t.entry)

                # Move stop to BE on remaining position
                old_stop = t.stop
                t.stop = t.entry
                if t.stop_order_id:
                    self.exec.cancel_order(t.stop_order_id)
                new_stop_order = self.exec.place_stop_market_order(close_side, t.remaining_qty, t.entry)
                t.stop_order_id = new_stop_order["id"] if new_stop_order else None
                logger.info(f"Stop moved {old_stop} → {t.entry} (BE)")

        # P2: TP2 hit → partial close (50% of original qty)
        if t.tp1_hit and not t.tp2_hit:
            tp2_hit = (t.side == "long" and high >= t.tp2) or (t.side == "short" and low <= t.tp2)
            if tp2_hit:
                t.tp2_hit = True
                close_qty = _round_qty(t.qty * c.partial_close_tp2_pct, c.qty_step)
                close_side = "sell" if t.side == "long" else "buy"
                actual_close = min(close_qty, t.remaining_qty)
                if actual_close >= c.qty_step:
                    self.exec.place_market_order(close_side, actual_close)
                    t.remaining_qty = max(c.qty_step, _round_qty(t.remaining_qty - actual_close, c.qty_step))
                    logger.info(f"TP2 partial close {actual_close} @ ~{t.tp2} | remaining={t.remaining_qty}")
                    notify_partial_close(t.side, "TP2", t.tp2, actual_close, t.entry)

        # P7: Trailing stop after TP2
        if t.tp2_hit and c.trail_after_tp2:
            atr_val = float(row.get("atr", 0))
            if atr_val > 0:
                trail_dist = atr_val * c.trail_atr_mult
                if t.side == "long":
                    new_trail = round_tick(row["close"] - trail_dist, tick)
                    if new_trail > t.stop:
                        old_stop = t.stop
                        t.stop = new_trail
                        if t.stop_order_id:
                            self.exec.cancel_order(t.stop_order_id)
                        new_stop_order = self.exec.place_stop_market_order("sell", t.remaining_qty, t.stop)
                        t.stop_order_id = new_stop_order["id"] if new_stop_order else None
                        logger.info(f"Trail stop raised {old_stop} → {t.stop}")
                else:
                    new_trail = round_tick(row["close"] + trail_dist, tick)
                    if new_trail < t.stop:
                        old_stop = t.stop
                        t.stop = new_trail
                        if t.stop_order_id:
                            self.exec.cancel_order(t.stop_order_id)
                        new_stop_order = self.exec.place_stop_market_order("buy", t.remaining_qty, t.stop)
                        t.stop_order_id = new_stop_order["id"] if new_stop_order else None
                        logger.info(f"Trail stop lowered {old_stop} → {t.stop}")

    # ── Close trade ───────────────────────────────────────────────────────────

    def _close_trade(self, reason: str, exit_price: float = 0.0) -> None:
        t = self.trade
        if t is None:
            return
        # Cancel existing stop order if it was placed on exchange
        if t.stop_order_id:
            self.exec.cancel_order(t.stop_order_id)
        # Send market close for whatever remains
        close_side = "sell" if t.side == "long" else "buy"
        close_qty = t.remaining_qty if t.remaining_qty > 0 else t.qty
        try:
            self.exec.place_market_order(close_side, close_qty)
        except Exception as exc:
            logger.error(f"Market close failed: {exc}")
        logger.info(f"Trade closed — reason: {reason} | side: {t.side} | entry: {t.entry} | stop: {t.stop}")
        notify_trade_close(t.side, t.entry, exit_price or t.entry, close_qty, reason)
        self.trade = None

    # ── Cancel pending ────────────────────────────────────────────────────────

    def _cancel_pending(self) -> None:
        if self.pending:
            self.exec.cancel_order(self.pending.order_id)
            self.pending = None

    # ── Funding blackout: manage trades but cancel pending new orders ─────────

    def update_expiry_only(self, bar_index: int) -> None:
        """
        Used during funding-rate blackout. Cancels any pending limit order
        (we don't want fills during high-volatility funding windows) but keeps
        monitoring an active trade that was already open.
        """
        self._cancel_pending()

    # ── State accessors ───────────────────────────────────────────────────────

    @property
    def has_pending(self) -> bool:
        return self.pending is not None

    @property
    def has_trade(self) -> bool:
        return self.trade is not None

    @property
    def is_flat(self) -> bool:
        return not self.has_pending and not self.has_trade
