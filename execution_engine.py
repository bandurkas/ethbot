"""
execution_engine.py — CCXT-based order placement and management for HTX.

All order actions go through this module so paper_trader.py can swap out
the implementation without touching the rest of the codebase.
"""

import logging
from typing import Optional

import ccxt

from config import BotConfig, cfg as default_cfg

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Thin wrapper around ccxt.htx for ETH/USDT:USDT perpetual swap.

    Stop-loss management strategy:
      HTX supports conditional (trigger) orders. We place a market stop order
      via create_order with 'stopPrice'. If that fails, the TradeManager falls
      back to software stop monitoring.
    """

    CONTRACT_SIZE = 0.01  # 1 HTX contract = 0.01 ETH

    def __init__(self, exchange: ccxt.htx, config: BotConfig = default_cfg):
        self.exchange = exchange
        self.cfg = config
        self.symbol = config.symbol

    def _contracts(self, qty_eth: float) -> int:
        """Convert ETH qty → whole contracts (min 1). HTX requires integer contracts."""
        return max(1, round(qty_eth / self.CONTRACT_SIZE))

    # ── Order placement ───────────────────────────────────────────────────────

    def place_limit_order(self, side: str, qty: float, price: float) -> dict:
        """
        Place a GTC limit order.
        side: "buy" | "sell"
        Returns the raw ccxt order dict.
        """
        contracts = self._contracts(qty)
        logger.info(f"Placing limit {side} {contracts} contracts ({qty} ETH) @ {price}")
        order = self.exchange.create_order(
            symbol=self.symbol,
            type="limit",
            side=side,
            amount=contracts,
            price=price,
            params={"timeInForce": "GTC", "lever_rate": self.cfg.leverage, "margin_mode": "cross"},
        )
        logger.info(f"Limit order placed: {order['id']}")
        return order

    def place_market_order(self, side: str, qty: float) -> dict:
        """Place a market order (used for immediate entry or emergency close)."""
        contracts = self._contracts(qty)
        logger.info(f"Placing market {side} {contracts} contracts ({qty} ETH)")
        order = self.exchange.create_order(
            symbol=self.symbol,
            type="market",
            side=side,
            amount=contracts,
            params={"lever_rate": self.cfg.leverage, "margin_mode": "cross"},
        )
        logger.info(f"Market order placed: {order['id']}")
        return order

    def place_stop_market_order(self, side: str, qty: float, stop_price: float) -> Optional[dict]:
        """
        Place a stop-market order (trigger order on HTX).
        Returns None if not supported; trade_manager will monitor in software.
        """
        try:
            contracts = self._contracts(qty)
            logger.info(f"Placing stop-market {side} {contracts} contracts trigger @ {stop_price}")
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="stop",
                side=side,
                amount=contracts,
                params={"stopPrice": stop_price, "triggerType": "market", "lever_rate": self.cfg.leverage, "margin_mode": "cross"},
            )
            logger.info(f"Stop order placed: {order['id']}")
            return order
        except ccxt.BaseError as exc:
            logger.warning(f"Stop order placement failed ({exc}); will use software stop")
            return None

    # ── Order management ──────────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        try:
            self.exchange.cancel_order(order_id, self.symbol)
            logger.info(f"Cancelled order {order_id}")
            return True
        except ccxt.OrderNotFound:
            logger.warning(f"Order {order_id} not found when cancelling")
            return False
        except ccxt.BaseError as exc:
            logger.error(f"Cancel failed for {order_id}: {exc}")
            return False

    def amend_stop_order(self, stop_order_id: str, new_stop_price: float) -> bool:
        """
        Move an existing stop order to a new price (break-even move).
        HTX does not support amend in unified API — we cancel & replace.
        """
        self.cancel_order(stop_order_id)
        return True  # Caller must re-place stop via place_stop_market_order

    # ── Account / position queries ────────────────────────────────────────────

    def get_open_orders(self) -> list[dict]:
        try:
            return self.exchange.fetch_open_orders(self.symbol)
        except ccxt.BaseError as exc:
            logger.error(f"fetch_open_orders failed: {exc}")
            return []

    def get_position(self) -> Optional[dict]:
        """Returns the current position dict or None if flat."""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos.get("contracts", 0) != 0:
                    return pos
        except ccxt.BaseError as exc:
            logger.error(f"fetch_positions failed: {exc}")
        return None

    def get_balance(self) -> float:
        """Returns USDT available balance from the swap (unified) account."""
        try:
            bal = self.exchange.fetch_balance({"type": "swap"})
            return float(bal.get("USDT", {}).get("free", 0.0))
        except ccxt.BaseError as exc:
            logger.error(f"fetch_balance failed: {exc}")
            return 0.0

    def set_leverage(self, leverage: int) -> bool:
        try:
            self.exchange.set_leverage(leverage, self.symbol)
            logger.info(f"Leverage set to {leverage}x")
            return True
        except ccxt.BaseError as exc:
            logger.warning(f"set_leverage failed: {exc}")
            return False
