from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Dict
import pandas as pd
import polars as pl
import yfinance as yf

from signalgraph.schemas import Signal
from signalgraph.utils import stable_uid

def fetch_ohlcv_yf(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Returns pandas DataFrame with DatetimeIndex and OHLCV columns.
    """
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    # ensure tz-aware UTC
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns=str.lower
    )

def ohlcv_to_signals(ticker: str, df: pd.DataFrame) -> pl.DataFrame:
    rows: List[Dict] = []
    for ts, r in df.iterrows():
        # align asof to trading date (UTC)
        asof = ts.date().isoformat()
        ts_iso = ts.to_pydatetime().astimezone(timezone.utc).isoformat()

        # price close
        uid_p = stable_uid("yfinance", ticker, ts_iso, "market.price", "close")
        rows.append(Signal(
            uid=uid_p, ts=ts.to_pydatetime(), entity=ticker, kind="market.price",
            value=float(r["close"]), source="yfinance", meta={"field": "close"}, asof=asof
        ).model_dump())

        # volume
        uid_v = stable_uid("yfinance", ticker, ts_iso, "market.volume", "volume")
        rows.append(Signal(
            uid=uid_v, ts=ts.to_pydatetime(), entity=ticker, kind="market.volume",
            value=float(r["volume"]), source="yfinance", meta={}, asof=asof
        ).model_dump())

        # realized volatility proxy (daily range / close)
        if r["close"] and r["close"] != 0:
            rv = (float(r["high"]) - float(r["low"])) / float(r["close"])
            uid_s = stable_uid("yfinance", ticker, ts_iso, "market.volatility", "rng_close")
            rows.append(Signal(
                uid=uid_s, ts=ts.to_pydatetime(), entity=ticker, kind="market.volatility",
                value=rv, source="yfinance", meta={"formula": "range/close"}, asof=asof
            ).model_dump())

    return pl.DataFrame(rows)
