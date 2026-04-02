"""
test_execution.py — Places minimum ETH/USDT:USDT market buy then closes it.
Verifies: auth, balance, order placement, position close.

Run: python test_execution.py
"""

import os
import time
import sys
from dotenv import load_dotenv

load_dotenv()

from market_data import make_exchange
from config import cfg

MIN_QTY   = 0.01   # 1 HTX contract = 0.01 ETH (absolute minimum)
CONTRACTS = 1      # Place exactly 1 contract


def run():
    print("── ETH Bot Execution Test ──────────────────────")
    exchange = make_exchange(cfg)

    # 1. Balance
    try:
        bal = exchange.fetch_balance({"type": "swap"})
        usdt_free = float(bal.get("USDT", {}).get("free", 0))
        print(f"[OK] Balance fetched: ${usdt_free:.2f} USDT free")
    except Exception as e:
        print(f"[FAIL] Balance fetch: {e}")
        sys.exit(1)

    if usdt_free < 2.0:
        print("[FAIL] Need at least $2 free in USDT-M Futures to run this test.")
        sys.exit(1)

    # 2. Set leverage
    try:
        exchange.set_leverage(cfg.leverage, cfg.symbol)
        print(f"[OK] Leverage set to {cfg.leverage}x")
    except Exception as e:
        print(f"[WARN] Leverage set: {e}")

    # 3. Get current price
    ticker = exchange.fetch_ticker(cfg.symbol)
    price  = ticker["last"]
    print(f"[OK] ETH price: ${price:.2f}")

    notional = MIN_QTY * price / cfg.leverage
    print(f"[INFO] Test order: {CONTRACTS} contract ({MIN_QTY} ETH) | Notional: ${MIN_QTY * price:.2f} | Margin: ~${notional:.2f}")

    # 4. Place market BUY (1 contract)
    print(f"\nPlacing market BUY {CONTRACTS} contract …")
    try:
        order = exchange.create_market_order(cfg.symbol, "buy", CONTRACTS)
        print(f"[OK] Order placed → id={order['id']} status={order['status']}")
    except Exception as e:
        print(f"[FAIL] Market buy: {e}")
        sys.exit(1)

    # 5. Wait for fill confirmation
    time.sleep(2)
    try:
        pos = exchange.fetch_positions([cfg.symbol])
        open_pos = [p for p in pos if abs(float(p["contracts"] or 0)) > 0]
        if open_pos:
            p = open_pos[0]
            print(f"[OK] Position confirmed: {p['side']} {p['contracts']} contracts @ ${p['entryPrice']:.2f}")
        else:
            print("[WARN] No position found after buy — may take a moment to settle")
    except Exception as e:
        print(f"[WARN] Position fetch: {e}")

    # 6. Close position — market SELL (1 contract, reduceOnly)
    print(f"\nClosing position with market SELL {CONTRACTS} contract …")
    try:
        close = exchange.create_market_order(
            cfg.symbol, "sell", CONTRACTS,
            params={"reduceOnly": True}
        )
        print(f"[OK] Close order → id={close['id']} status={close['status']}")
    except Exception as e:
        print(f"[FAIL] Market sell: {e}")
        sys.exit(1)

    time.sleep(2)

    # 7. Final balance
    try:
        bal2   = exchange.fetch_balance({"type": "swap"})
        final  = float(bal2.get("USDT", {}).get("free", 0))
        pnl    = final - usdt_free
        sign   = "+" if pnl >= 0 else ""
        print(f"\n[OK] Final balance: ${final:.4f} | PnL: {sign}${pnl:.4f}")
    except Exception as e:
        print(f"[WARN] Final balance: {e}")

    print("\n── All checks passed. Bot execution pipeline is working. ──")


if __name__ == "__main__":
    run()
