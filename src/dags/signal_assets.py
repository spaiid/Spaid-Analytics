from dagster import asset, AssetExecutionContext
import polars as pl
from signalgraph.io import write_signals
from ingest.sources.market.ohlcv import fetch_ohlcv, ohlcv_to_signals
from ingest.sources.google_trends.fetch import fetch_trends, trends_to_signals

@asset
def sg_market_daily(context: AssetExecutionContext):
    tickers = ["NVDA","MSFT","AAPL","AMZN","GOOGL", "SPY"]
    all_df = []
    for t in tickers:
        rows = fetch_ohlcv(t)            # implement with Polygon/Yahoo
        df = ohlcv_to_signals(t, rows)
        all_df.append(df)
    write_signals(pl.concat(all_df))
    return {"tickers": tickers, "rows": sum(len(x) for x in all_df)}

@asset
def sg_trends_weekly(context: AssetExecutionContext):
    keywords = {"NVDA":"nvidia", "MSFT":"microsoft"}
    all_df = []
    for tic, kw in keywords.items():
        series = fetch_trends(kw)        # implement with pytrends or cached csv
        df = trends_to_signals(kw, series, tic)
        all_df.append(df)
    write_signals(pl.concat(all_df))
    return {"keywords": list(keywords.values())}
