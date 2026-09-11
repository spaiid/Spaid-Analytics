from __future__ import annotations
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

FEATURES = [
  # returns & momentum
  "ret_1d","ret_5d","ret_20d","ret_60d",
  # moving averages / position vs MA
  "ma10","ma20","ma50","px_over_ma20",
  # volatility
  "vol_cc_20","vol_cc_60","rv_rc_20",
  # volume structure
  "vol_ma20","vol_rel_20",
  # oscillators
  "rsi14","macd","macd_signal","macd_hist",
]


TARGET = "ret_fwd_20d"

def train_test_split_by_date(df: pl.DataFrame, split_frac: float = 0.8):
    dates = df.select("asof").unique().to_series().to_list()
    dates.sort()
    cut = int(len(dates)*split_frac)
    train_dates = set(dates[:cut])
    test_dates  = set(dates[cut:])
    return (df.filter(pl.col("asof").is_in(list(train_dates))),
            df.filter(pl.col("asof").is_in(list(test_dates))))

def fit_random_forest(df: pl.DataFrame):
    df = df.drop_nulls(subset=FEATURES+[TARGET])
    if df.is_empty():
        raise ValueError("No rows after dropping nulls.")

    train_df, test_df = train_test_split_by_date(df)

    Xtr = train_df.select(FEATURES).to_pandas()
    ytr = train_df.select(TARGET).to_series().to_pandas()
    Xte = test_df.select(FEATURES).to_pandas()
    yte = test_df.select(TARGET).to_series().to_pandas()

    model = RandomForestRegressor(n_estimators=400, max_depth=7, random_state=7, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    metrics = {
        "r2": float(r2_score(yte, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }
    # simple per-entity average predicted return (last test date)
    test_pdf = test_df.to_pandas()
    test_pdf["pred"] = pred
    latest = test_pdf["asof"].max()
    latest_slice = test_pdf[test_pdf["asof"] == latest]
    ranking = (latest_slice.groupby("entity")["pred"].mean()
               .sort_values(ascending=False).to_dict())

    return model, metrics, ranking
