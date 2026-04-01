"""
market_data.py — OHLCV fetching and indicator calculation.

All indicator logic mirrors the Pine Script in context.md:
  - EMA fast/slow
  - VWAP with stdev bands (rolling, NOT session-reset — matches Pine ta.vwap behaviour
    which resets per session; for 15m intraday we approximate with daily reset)
  - ATR
  - RSI
  - Volume SMA
  - Prev high/low (sweep range)
  - Previous day high/low (PDH/PDL)
  - Pivot high/low (3-bar)
"""

import os
import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import BotConfig, cfg as default_cfg

load_dotenv()


# ── Exchange factory ──────────────────────────────────────────────────────────

def make_exchange(config: BotConfig = default_cfg) -> ccxt.htx:
    exchange = ccxt.htx({
        "apiKey": os.getenv("HTX_API_KEY", ""),
        "secret": os.getenv("HTX_API_SECRET", ""),
        "options": {
            "defaultType": "swap",
            "unified": True,            # HTX unified account (new account type)
        },
    })
    if os.getenv("HTX_TESTNET", "false").lower() == "true":
        exchange.set_sandbox_mode(True)
    exchange.load_markets()
    return exchange


# ── OHLCV fetch ───────────────────────────────────────────────────────────────

def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> pd.DataFrame:
    """Fetch OHLCV bars and return as DataFrame."""
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df


def fetch_daily_ohlcv(exchange: ccxt.Exchange, symbol: str, limit: int = 5) -> pd.DataFrame:
    """Fetch daily bars for PDH/PDL."""
    return fetch_ohlcv(exchange, symbol, "1d", limit=limit)


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _vwap_with_bands(df: pd.DataFrame, dev_len: int, sigma_k: float):
    """
    Rolling VWAP that resets each calendar day (UTC), matching Pine's ta.vwap.
    Returns (vwap, band_up, band_dn, price_dev, stdev_dev) as Series.
    """
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"]

    # Day group identifier
    day = df.index.floor("D")

    cum_pv = (hlc3 * vol).groupby(day).cumsum()
    cum_v = vol.groupby(day).cumsum()
    vwap = cum_pv / cum_v

    price_dev = df["close"] - vwap
    stdev_dev = price_dev.rolling(dev_len).std()
    band_up = vwap + sigma_k * stdev_dev
    band_dn = vwap - sigma_k * stdev_dev

    return vwap, band_up, band_dn, price_dev, stdev_dev


def _pivot_high(high: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    """Returns pivot high value at each bar where pivot was confirmed (right bars ago)."""
    result = pd.Series(np.nan, index=high.index)
    arr = high.values
    for i in range(left, len(arr) - right):
        window = arr[i - left : i + right + 1]
        if arr[i] == max(window):
            result.iloc[i] = arr[i]
    return result


def _pivot_low(low: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    result = pd.Series(np.nan, index=low.index)
    arr = low.values
    for i in range(left, len(arr) - right):
        window = arr[i - left : i + right + 1]
        if arr[i] == min(window):
            result.iloc[i] = arr[i]
    return result


# ── Main indicator calculation ────────────────────────────────────────────────

def calc_indicators(df: pd.DataFrame, config: BotConfig = default_cfg, pdh: float = None, pdl: float = None) -> pd.DataFrame:
    """
    Adds all indicator columns to df in-place and returns it.
    pdh/pdl: previous day high/low (pass from daily fetch).
    """
    c = config

    df["ema_fast"] = _ema(df["close"], c.ema_fast_len)
    df["ema_slow"] = _ema(df["close"], c.ema_slow_len)
    df["trend_long"] = df["ema_fast"] > df["ema_slow"]
    df["trend_short"] = df["ema_fast"] < df["ema_slow"]

    df["vwap"], df["band_up"], df["band_dn"], df["price_dev"], df["stdev_dev"] = _vwap_with_bands(
        df, c.dev_len, c.sigma_k
    )

    df["atr"] = _atr(df, c.atr_len)
    df["rsi"] = _rsi(df["close"], c.rsi_len)
    df["vol_sma"] = df["volume"].rolling(c.vol_sma_len).mean()

    # Sweep range: previous bar's rolling high/low
    df["prev_high"] = df["high"].shift(1).rolling(c.sweep_len).max()
    df["prev_low"] = df["low"].shift(1).rolling(c.sweep_len).min()

    # Previous day high/low (scalar, injected from daily fetch)
    df["pdh"] = pdh if pdh is not None else np.nan
    df["pdl"] = pdl if pdl is not None else np.nan

    # Swing pivots (3-bar) — last known value propagated forward
    ph_raw = _pivot_high(df["high"], 3, 3)
    pl_raw = _pivot_low(df["low"], 3, 3)
    df["swing_high"] = ph_raw.ffill()
    df["swing_low"] = pl_raw.ffill()

    return df


# ── P3: HTF trend indicator injection ────────────────────────────────────────

def inject_htf_trend(
    df: pd.DataFrame,
    exchange,
    config: BotConfig = default_cfg,
) -> pd.DataFrame:
    """
    Fetch HTF (default 1h) bars, compute EMA fast/slow, and inject
    htf_trend_long / htf_trend_short columns into the intraday df.

    Each intraday bar gets the HTF trend of the most recently closed HTF bar
    (forward-fill, no lookahead).
    """
    if not config.htf_filter:
        df["htf_trend_long"]  = True
        df["htf_trend_short"] = True
        return df

    htf_df = fetch_ohlcv(exchange, config.symbol, config.htf_timeframe, limit=200)
    htf_df["htf_ema_fast"] = _ema(htf_df["close"], config.htf_ema_fast_len)
    htf_df["htf_ema_slow"] = _ema(htf_df["close"], config.htf_ema_slow_len)
    htf_df["htf_trend_long"]  = htf_df["htf_ema_fast"] > htf_df["htf_ema_slow"]
    htf_df["htf_trend_short"] = htf_df["htf_ema_fast"] < htf_df["htf_ema_slow"]

    # Reindex to intraday timestamps via forward-fill (no lookahead)
    htf_trend = htf_df[["htf_trend_long", "htf_trend_short"]]
    combined = df.join(htf_trend, how="left")
    combined["htf_trend_long"]  = combined["htf_trend_long"].ffill().fillna(True).astype(bool)
    combined["htf_trend_short"] = combined["htf_trend_short"].ffill().fillna(True).astype(bool)

    df["htf_trend_long"]  = combined["htf_trend_long"]
    df["htf_trend_short"] = combined["htf_trend_short"]
    return df


# ── ATR multiplier selector (matches Pine getTimeTF) ─────────────────────────

def get_atr_mult(timeframe: str, config: BotConfig = default_cfg) -> float:
    if timeframe == "15m":
        return config.atr_mult_15m
    if timeframe in ("1h", "60m"):
        return config.atr_mult_1h
    return config.atr_mult_15m
