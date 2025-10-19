import polars as pl
from datetime import date, timedelta

def load_signals(kind: str, start: str, end: str, root="_data/signalgraph") -> pl.DataFrame:
    # naive loader: glob partitions in range (optimize later)
    import glob, os
    paths = []
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    cur = d0
    while cur <= d1:
        p = os.path.join(root, f"kind={kind}", f"asof={cur.isoformat()}", "*.parquet")
        paths += glob.glob(p)
        cur += timedelta(days=1)
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")

def features_daily(start: str, end: str) -> pl.DataFrame:
    price = load_signals("market.price", start, end)
    vol   = load_signals("market.volume", start, end)
    jobs  = load_signals("jobs.open_roles", start, end, )
    trends= load_signals("behavior.google_trends", start, end)

    # pivot/value → per-entity time series
    def pivot_mean(df):
        return (df.group_by(["asof","entity"])
                 .agg(pl.col("value").mean().alias("val"))
                 .pivot(values="val", index=["asof","entity"], columns=None))

    # rollups
    f_price = price.group_by(["entity","asof"]).agg(
        pl.col("value").mean().alias("close"),
    )
    f_trend = pivot_mean(trends).rename({"val":"trend"})
    f_jobs  = pivot_mean(jobs).rename({"val":"open_roles"})
    f_vol   = pivot_mean(vol).rename({"val":"volume"})

    base = f_price.join(f_vol, on=["entity","asof"], how="left")\
                  .join(f_trend, on=["entity","asof"], how="left")\
                  .join(f_jobs, on=["entity","asof"], how="left")

    # simple deriveds
    base = base.with_columns([
        pl.col("close").pct_change().over("entity").alias("ret_1d"),
        pl.col("trend").pct_change().over("entity").alias("trend_chg"),
    ])
    return base.sort(["entity","asof"])
