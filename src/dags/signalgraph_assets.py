from __future__ import annotations
from dagster import asset
import polars as pl

from ingest.sources.market.yfinance_ohlcv import fetch_ohlcv_yf, ohlcv_to_signals
from signalgraph.io import write_signals

@asset
def sg_market_daily(context):
    tickers = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL"]
    frames = []
    for t in tickers:
        try:
            raw = fetch_ohlcv_yf(t, period="1y", interval="1d")
            df = ohlcv_to_signals(t, raw)
            frames.append(df)
            context.log.info(f"{t}: {len(df)} signals")
        except Exception as e:
            context.log.warning(f"{t} failed: {e}")

    if not frames:
        return {"tickers": [], "rows": 0}

    all_df = pl.concat(frames, how="diagonal_relaxed")
    # idempotency: drop dup uids if we re-run
    all_df = all_df.unique(subset=["uid"])
    write_signals(all_df)
    return {"tickers": tickers, "rows": int(all_df.height)}
