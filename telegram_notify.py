"""
telegram_notify.py — Lightweight Telegram push notifications.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
All functions are fire-and-forget (errors are logged, never raised).
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


def _send(text: str) -> None:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        resp = requests.post(
            _API.format(token=token),
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
        if not resp.ok:
            logger.warning(f"Telegram error {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        logger.warning(f"Telegram send failed: {exc}")


def notify_trade_open(side: str, entry: float, stop: float, tp3: float, qty: float) -> None:
    icon = "🟢" if side == "long" else "🔴"
    risk = abs(entry - stop) * qty
    text = (
        f"{icon} <b>{side.upper()} OPENED</b>\n"
        f"Entry : <code>${entry:.2f}</code>\n"
        f"Stop  : <code>${stop:.2f}</code>\n"
        f"TP3   : <code>${tp3:.2f}</code>\n"
        f"Qty   : <code>{qty} ETH</code>  Risk: <code>${risk:.2f}</code>"
    )
    _send(text)


def notify_partial_close(side: str, label: str, exit_price: float, qty: float, entry: float) -> None:
    pnl  = (exit_price - entry) * qty * (1 if side == "long" else -1)
    sign = "+" if pnl >= 0 else ""
    text = (
        f"📊 <b>{label} hit</b> — partial close\n"
        f"Exit  : <code>${exit_price:.2f}</code>  Qty: <code>{qty}</code>\n"
        f"PnL   : <code>{sign}${pnl:.2f}</code>"
    )
    _send(text)


def notify_trade_close(side: str, entry: float, exit_price: float, qty: float, reason: str) -> None:
    pnl  = (exit_price - entry) * qty * (1 if side == "long" else -1)
    sign = "+" if pnl >= 0 else ""
    icon = "✅" if pnl >= 0 else "❌"
    label_map = {
        "stop_hit": "Stop hit",
        "tp3": "TP3",
        "new_signal_opposite_side": "Reversed",
    }
    label = label_map.get(reason, reason)
    text = (
        f"{icon} <b>{side.upper()} CLOSED</b> [{label}]\n"
        f"Entry : <code>${entry:.2f}</code> → <code>${exit_price:.2f}</code>\n"
        f"PnL   : <b><code>{sign}${pnl:.2f}</code></b>"
    )
    _send(text)
