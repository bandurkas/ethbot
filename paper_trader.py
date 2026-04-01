"""
paper_trader.py — Simulated execution layer (no real orders).

Implements the same interface as ExecutionEngine so TradeManager can run
identically in paper mode. Fills are simulated bar-by-bar against OHLCV.

Usage:
    paper_exec = PaperExecutionEngine(config)
    manager    = TradeManager(paper_exec, config)
    # then drive manager.update(row, bar_index) as normal
"""

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from config import BotConfig, cfg as default_cfg

logger = logging.getLogger(__name__)


# ── Fake order / position objects ─────────────────────────────────────────────

_order_counter = 0

def _new_order_id() -> str:
    global _order_counter
    _order_counter += 1
    return f"PAPER-{_order_counter:06d}"


@dataclass
class PaperOrder:
    id: str
    side: str       # "buy" | "sell"
    type: str       # "limit" | "market" | "stop"
    qty: float
    price: float    # limit price (0 for market/stop)
    stop_price: float = 0.0
    status: str = "open"    # "open" | "filled" | "cancelled"
    fill_price: float = 0.0


# ── Paper trade record (for P&L log) ─────────────────────────────────────────

@dataclass
class PaperTradeRecord:
    side: str
    entry: float
    exit: float
    qty: float
    exit_reason: str    # "stop" | "tp1" | "tp2" | "tp3" | "manual"
    pnl_usdt: float     # unrealised, then realised
    fee_usdt: float


# ── Paper execution engine ────────────────────────────────────────────────────

class PaperExecutionEngine:
    """
    Drop-in replacement for ExecutionEngine for paper trading.
    Exposes the same public methods; stores orders in memory.
    """

    def __init__(self, config: BotConfig = default_cfg, log_path: str = "paper_trades.csv"):
        self.cfg = config
        self.symbol = config.symbol
        self._orders: dict[str, PaperOrder] = {}
        self._balance = config.init_dep
        self._trades: list[PaperTradeRecord] = []
        self._log_path = Path(log_path)
        # P8: Daily tracking
        self._daily_start_balance: float = config.init_dep
        self._daily_trade_count: int = 0
        self._current_day: str = ""
        self._init_csv()

    # ── ExecutionEngine interface ─────────────────────────────────────────────

    def place_limit_order(self, side: str, qty: float, price: float) -> dict:
        order = PaperOrder(id=_new_order_id(), side=side, type="limit", qty=qty, price=price)
        self._orders[order.id] = order
        logger.info(f"[PAPER] Limit {side} {qty} @ {price} | id={order.id}")
        return {"id": order.id, "price": price, "amount": qty, "side": side}

    def place_market_order(self, side: str, qty: float) -> dict:
        order = PaperOrder(id=_new_order_id(), side=side, type="market", qty=qty, price=0.0)
        order.status = "filled"   # market orders fill immediately
        self._orders[order.id] = order
        logger.info(f"[PAPER] Market {side} {qty} | id={order.id}")
        return {"id": order.id, "amount": qty, "side": side}

    def place_stop_market_order(self, side: str, qty: float, stop_price: float) -> Optional[dict]:
        order = PaperOrder(id=_new_order_id(), side=side, type="stop", qty=qty, price=0.0, stop_price=stop_price)
        self._orders[order.id] = order
        logger.info(f"[PAPER] Stop-market {side} {qty} trigger @ {stop_price} | id={order.id}")
        return {"id": order.id, "stopPrice": stop_price, "amount": qty, "side": side}

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = "cancelled"
            logger.info(f"[PAPER] Cancelled {order_id}")
            return True
        return False

    def get_open_orders(self) -> list[dict]:
        return [
            {"id": o.id, "side": o.side, "price": o.price, "amount": o.qty}
            for o in self._orders.values()
            if o.status == "open"
        ]

    def get_position(self) -> Optional[dict]:
        return None  # TradeManager tracks position state

    def get_balance(self) -> float:
        return self._balance

    def set_leverage(self, leverage: int) -> bool:
        return True

    # ── Bar simulation ────────────────────────────────────────────────────────

    def simulate_bar(self, row: pd.Series) -> None:
        """
        Simulate order fills against a bar. Call BEFORE TradeManager.update()
        so that stop orders already track the current bar's price action.
        (TradeManager.update() handles the strategy-level TP/SL checks.)
        """
        for order in list(self._orders.values()):
            if order.status != "open":
                continue
            high, low = row["high"], row["low"]

            if order.type == "limit":
                if order.side == "buy" and low <= order.price:
                    order.fill_price = order.price
                    order.status = "filled"
                    logger.info(f"[PAPER] Limit buy filled @ {order.fill_price}")
                elif order.side == "sell" and high >= order.price:
                    order.fill_price = order.price
                    order.status = "filled"
                    logger.info(f"[PAPER] Limit sell filled @ {order.fill_price}")

            elif order.type == "stop":
                if order.side == "sell" and low <= order.stop_price:
                    order.fill_price = order.stop_price
                    order.status = "filled"
                    logger.info(f"[PAPER] Stop sell triggered @ {order.fill_price}")
                elif order.side == "buy" and high >= order.stop_price:
                    order.fill_price = order.stop_price
                    order.status = "filled"
                    logger.info(f"[PAPER] Stop buy triggered @ {order.fill_price}")

    # ── P&L recording ─────────────────────────────────────────────────────────

    def record_trade(
        self,
        side: str,
        entry: float,
        exit_price: float,
        qty: float,
        exit_reason: str,
    ) -> PaperTradeRecord:
        c = self.cfg
        sign = 1 if side == "long" else -1
        gross_pnl = sign * (exit_price - entry) * qty
        fee = (entry * qty * c.taker_fee_pct / 100) + (exit_price * qty * c.taker_fee_pct / 100)
        net_pnl = gross_pnl - fee
        self._balance += net_pnl
        # P8: update daily trade count
        self._daily_trade_count += 1
        rec = PaperTradeRecord(
            side=side, entry=entry, exit=exit_price, qty=qty,
            exit_reason=exit_reason, pnl_usdt=net_pnl, fee_usdt=fee,
        )
        self._trades.append(rec)
        self._append_csv(rec)
        logger.info(f"[PAPER] Trade closed | {side} | entry={entry} exit={exit_price} | PnL=${net_pnl:.2f} ({exit_reason})")
        return rec

    # ── P8: Daily circuit breaker helpers ────────────────────────────────────

    def reset_daily_stats(self) -> None:
        """Call at the start of each new trading day."""
        self._daily_start_balance = self._balance
        self._daily_trade_count = 0

    def daily_loss(self) -> float:
        """Returns today's loss as a positive USDT value (0 if no loss)."""
        return max(0.0, self._daily_start_balance - self._balance)

    def daily_loss_limit_hit(self) -> bool:
        c = self.cfg
        limit = c.init_dep * c.daily_loss_limit_pct / 100
        return self.daily_loss() >= limit

    def daily_trade_limit_hit(self) -> bool:
        return self._daily_trade_count >= self.cfg.max_trades_per_day

    # ── Metrics ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self._trades:
            return {"trades": 0}
        wins = [t for t in self._trades if t.pnl_usdt > 0]
        losses = [t for t in self._trades if t.pnl_usdt <= 0]
        total_pnl = sum(t.pnl_usdt for t in self._trades)
        win_rate = len(wins) / len(self._trades) * 100
        avg_win  = sum(t.pnl_usdt for t in wins) / len(wins)   if wins   else 0
        avg_loss = sum(t.pnl_usdt for t in losses) / len(losses) if losses else 0
        avg_rr   = abs(avg_win / avg_loss) if avg_loss != 0 else math.inf
        return {
            "trades":    len(self._trades),
            "wins":      len(wins),
            "losses":    len(losses),
            "win_rate":  round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win":   round(avg_win, 2),
            "avg_loss":  round(avg_loss, 2),
            "avg_rr":    round(avg_rr, 2),
            "balance":   round(self._balance, 2),
        }

    # ── CSV logging ───────────────────────────────────────────────────────────

    def _init_csv(self) -> None:
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["side", "entry", "exit", "qty", "exit_reason", "pnl_usdt", "fee_usdt"])

    def _append_csv(self, rec: PaperTradeRecord) -> None:
        with open(self._log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([rec.side, rec.entry, rec.exit, rec.qty, rec.exit_reason, rec.pnl_usdt, rec.fee_usdt])
