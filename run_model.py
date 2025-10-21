# run_model.py
import os, sys, time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# UI reporter (HTML/JSON)
from tools.reporter import RunReport


# -------------- Config --------------
MODEL = os.environ.get("MODEL", "gbm").lower()               # "gbm" | "rf"
TOPN = int(os.environ.get("TOPN", "3"))                      # portfolio size for backtest
FWD_H = int(os.environ.get("FWD_H", "20"))                   # forward horizon (days) used in target names
TEST_FRACTION = float(os.environ.get("TEST_FRAC", "0.2"))    # last 20% time as test
TC_BPS = float(os.environ.get("TC_BPS", "5"))                # one-way transaction cost (bps)
REBAL_EVERY = int(os.environ.get("REBAL_EVERY", "1"))        # rebalance cadence in bars
LONG_SHORT = os.environ.get("LONG_SHORT", "0") == "1"        # long-short if 1, else long-only
MAX_WEIGHT = float(os.environ.get("MAX_WEIGHT", "0.10"))     # per-name cap (abs)
DECILES = int(os.environ.get("DECILES", "5"))                # decile bins for small universes

# Optional toggles
CS_ZSCORE = os.environ.get("CS_ZSCORE", "0") == "1"          # cross-sectional z-score features per date
DEMEAN_TARGET = os.environ.get("DEMEAN_TARGET", "0") == "1"   # demean target by date
LEAKAGE_ON_ALL = os.environ.get("LEAKAGE_ON_ALL", "0") == "1" # audit on train+test preds for power
LEAKAGE_USE_PAST_FEATURE = os.environ.get("LEAKAGE_USE_PAST_FEATURE", "1") == "1"  # use ret_{H}d baseline
PRINT_GLOBAL_RIC = os.environ.get("PRINT_GLOBAL_RIC", "1") == "1"  # print global Rank-ICs too

# UI report toggle
UI_ON = os.environ.get("SF_UI_REPORT", "1") == "1"

# ---- feature registry ----
BASE_FEATURES: List[str] = [
    # returns & momentum
    "ret_1d","ret_5d","ret_20d","ret_60d",
    # moving averages / price vs MA
    "ma10","ma20","ma50","px_over_ma20",
    # volatility
    "vol_cc_20","vol_cc_60","rv_rc_20",
    # volume structure
    "vol_ma20","vol_rel_20",
    # oscillators
    "rsi14","ema12","ema26","macd","macd_signal","macd_hist",
]

RESID_FEATURES: List[str] = [
    "resid_cum_20d","resid_cum_60d","resid_vol_20d","resid_vol_60d","beta_mkt_60d",
    # "beta_sec_60d",  # uncomment only if present in your features parquet
]

# (optional) long-horizon adds when using FWD_H=90+
LONG_H_FEATURES: List[str] = [
    "ret_90d","ret_120d","ret_180d",
    "ma100","ma200","px_over_ma100","px_over_ma200",
    "vol_cc_90","rv_rc_60","rv_rc_90",
]

# -------------- Utils --------------
def ensure_date(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure `asof` is pl.Date (YYYY-MM-DD)."""
    dt = df.schema.get("asof")
    if dt == pl.Date:
        return df
    if dt == pl.Datetime:
        return df.with_columns(pl.col("asof").dt.date().alias("asof"))
    return df.with_columns(pl.col("asof").str.to_date(format="%Y-%m-%d").alias("asof"))

def detect_target(df: pl.DataFrame, horizon: int) -> str:
    # Support both new and legacy names
    candidates = [
        f"fwd{horizon}_ret_ex_mkt",  # new market-excess
        f"fwd{horizon}_ret_ex_sec",  # new sector-excess
        f"fwd{horizon}_ret",         # new raw forward
        f"excess_fwd_{horizon}d",    # legacy
        f"ret_fwd_{horizon}d",       # legacy
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # last resort: any forward target
    for c in df.columns:
        if c.startswith(("fwd", "excess_fwd_", "ret_fwd_")) and c.endswith(("ret","d")):
            return c
    raise SystemExit("No forward target column found. Build features with forward returns first.")

def time_split(df: pl.DataFrame, test_fraction: float):
    df = ensure_date(df).sort(["asof", "entity"])
    dates = df.select("asof").unique().sort("asof")["asof"].to_list()

    # guardrails
    if len(dates) == 0:
        raise SystemExit("No dates available after filtering/drop_nulls; cannot split.")
    if len(dates) == 1:
        # everything must be train; empty test
        train = df
        test = pl.DataFrame(schema=df.schema)  # empty
        return train, test

    # clamp split index to [1, len(dates)-1]
    k = int(len(dates) * (1.0 - max(0.0, min(0.9, test_fraction))))  # cap test_fraction < 1
    k = max(1, min(len(dates) - 1, k))
    split_date = dates[k]

    train = df.filter(pl.col("asof") < split_date)
    test  = df.filter(pl.col("asof") >= split_date)
    if train.is_empty() or test.is_empty():
        # fallback: last date is test, rest train
        split_date = dates[-1]
        train = df.filter(pl.col("asof") < split_date)
        test  = df.filter(pl.col("asof") == split_date)
    return train, test

def to_xy(df: pl.DataFrame, features: List[str], target: str):
    X = df.select(features).to_numpy()
    y = df.select(target).to_numpy().ravel()
    return X, y

def fit_model(model_name: str):
    if model_name == "gbm":
        return GradientBoostingRegressor(
            n_estimators=600, learning_rate=0.02, max_depth=3, subsample=0.8, random_state=42
        )
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=600, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
        )
    raise ValueError(f"Unknown MODEL={model_name}")

def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    ybar = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - ybar) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"r2": r2, "rmse": rmse}

def sanitize_features(
    df: pl.DataFrame, features: list[str], target: str, asof_col: str = "asof", ent_col: str = "entity"
) -> pl.DataFrame:
    """Replace inf/NaN with null, median-impute features, and drop bad targets."""
    # 1) Replace infinities with nulls
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in features
    ])
    # 2) Convert NaN -> null for features & target
    df = df.with_columns([
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c) for c in (features + [target])
    ])
    # 3) Compute medians safely
    meds_df = df.select([pl.col(c).median().alias(c) for c in features])
    meds = meds_df.to_dicts()[0] if meds_df.height else {}
    # 4) Impute per-feature; fall back to 0 if median is None
    fill_exprs = []
    for c in features:
        val = meds.get(c, 0.0)
        if val is None or not np.isfinite(val):
            val = 0.0
        fill_exprs.append(pl.col(c).fill_null(val).alias(c))
    df = df.with_columns(fill_exprs)
    # 5) Drop any remaining bad targets
    df = df.filter(~pl.col(target).is_null() & ~pl.col(target).is_nan())
    return df.sort([asof_col, ent_col])

# ---------- Cross-sectional scaling (optional) ----------
def zscore_by_date(df: pl.DataFrame, cols: list[str], asof_col: str = "asof") -> pl.DataFrame:
    out = df
    for c in cols:
        out = out.with_columns(
            ((pl.col(c) - pl.col(c).mean().over(asof_col)) /
             (pl.col(c).std(ddof=1).over(asof_col) + 1e-9)).alias(c)
        )
    return out

# ---------- Signal quality: IC / Rank-IC & decile curve ----------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float); y = y.astype(float)
    if x.size < 3: return float("nan")
    x = x - x.mean(); y = y - y.mean()
    denom = (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum()))
    if denom <= 0: return float("nan")
    return float((x @ y) / denom)

def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a)+1, dtype=float)
    uniq, idx_start, counts = np.unique(a[order], return_index=True, return_counts=True)
    for s, c in zip(idx_start, counts):
        if c > 1:
            avg = (s+1 + s+c) / 2.0
            ranks[order[s:s+c]] = avg
    return ranks

def cs_ic_rankic_by_date(df: pl.DataFrame, pred_col: str, target_col: str, asof_col="asof") -> pl.DataFrame:
    outs = []
    for d, g in df.group_by(asof_col, maintain_order=True):
        p = g.get_column(pred_col).to_numpy()
        y = g.get_column(target_col).to_numpy()
        if len(p) >= 3 and np.isfinite(p).all() and np.isfinite(y).all():
            ic = _pearson(p, y)
            rp = _rankdata(p); ry = _rankdata(y)
            ric = _pearson(rp, ry)
            outs.append((d, float(ic), float(ric), int(len(p))))
    return pl.DataFrame(outs, schema=[asof_col, "ic", "rank_ic", "n"], orient="row")

def decile_curve(df: pl.DataFrame, pred_col: str, ret_col: str, asof_col="asof", n_bins: int = 10) -> pl.DataFrame:
    ranked = (
        df.with_columns([
            pl.col(pred_col).rank(method="average").over(asof_col).alias("_rk"),
            pl.len().over(asof_col).alias("_n")
        ])
        .with_columns(((pl.col("_rk") - 0.5) / (pl.col("_n") + 1e-9)).alias("_pct"))
        .with_columns(((pl.col("_pct") * n_bins).floor() + 1).clip(1, n_bins).cast(pl.Int32).alias("decile"))
    )
    return (
        ranked.group_by("decile")
              .agg(pl.col(ret_col).mean().alias("mean_ret"), pl.len().alias("count"))
              .sort("decile")
    )

def backtest_topN_tranches(
    df: pl.DataFrame,
    pred_col="y_pred",
    ret_col="y_true",
    asof_col="asof",
    ent_col="entity",
    N=3,
    H=90,
    K=9,
    tc_bps=5.0,
    max_weight=0.10,
    long_short=False,
):
    dates = df.select(asof_col).unique().sort(asof_col)[asof_col].to_list()
    step = max(1, H // K)
    prev_ws = [pl.DataFrame({ent_col: [], "w": []}) for _ in range(K)]
    tr_daily = [[] for _ in range(K)]

    for i, d in enumerate(dates):
        k = i % K
        snap = df.filter(pl.col(asof_col) == d).select([ent_col, pred_col])
        if snap.height:
            snap = snap.with_columns(
                pl.col(pred_col).rank("dense", descending=True).alias("_r_hi"),
                pl.col(pred_col).rank("dense", descending=False).alias("_r_lo")
            )
            longs = snap.filter(pl.col("_r_hi") <= N)[ent_col].to_list()
            shorts = snap.filter(pl.col("_r_lo") <= N)[ent_col].to_list() if long_short else []
            names = longs + shorts
            if names:
                sgn = np.array([1]*len(longs) + ([-1]*len(shorts)), float)
                w = sgn / np.sum(np.abs(sgn))
                w_now = pl.DataFrame({ent_col: names, "w": w})
            else:
                w_now = pl.DataFrame({ent_col: [], "w": []})
            w_now = w_now.with_columns(pl.col("w").clip(-max_weight, max_weight))
            s = float(w_now.select(pl.col("w").abs().sum().alias("s")).item())
            if s > 0:
                w_now = w_now.with_columns((pl.col("w")/s).alias("w"))

            prev_w = prev_ws[k]
            if prev_w.height:
                mv = w_now.join(prev_w.rename({"w":"w_prev"}), on=ent_col, how="full").fill_null(0.0)
                turn = float(mv.select((pl.col("w") - pl.col("w_prev")).abs().sum().alias("t")).item())
            else:
                mv = w_now.with_columns(pl.lit(0.0).alias("w_prev"))
                turn = float(mv.select(pl.col("w").abs().sum().alias("t")).item())
            cost = (tc_bps / 1e4) * turn
            prev_ws[k] = w_now
        else:
            cost = 0.0

        today = df.filter(pl.col(asof_col) == d).select([ent_col, ret_col])
        pnl = today.join(prev_ws[k], on=ent_col, how="left").fill_null(0.0)
        day_ret = float(pnl.select((pl.col(ret_col) * pl.col("w")).sum().alias("r")).item()) - cost
        tr_daily[k].append(day_ret)

    L = max(len(x) for x in tr_daily) if tr_daily else 0
    agg = np.zeros(L, float); cnt = np.zeros(L, float)
    for x in tr_daily:
        for j, r in enumerate(x):
            agg[j] += r; cnt[j] += 1.0
    r = np.divide(agg, np.maximum(cnt, 1.0), where=(cnt>0))

    if r.size == 0:
        return {"n_periods": 0, "CAGR": float("nan"), "Sharpe": float("nan"), "MaxDD": float("nan")}
    eq = (1.0 + r).cumprod()
    sharpe = (r.mean() / (r.std(ddof=1) + 1e-12)) * np.sqrt(252.0)
    years = len(r) / 252.0
    cagr = eq[-1]**(1.0/years) - 1.0 if years > 0 else float("nan")
    peak = np.maximum.accumulate(eq)
    mdd = float(((peak - eq) / peak).max())
    return {"n_periods": int(len(r)), "CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": mdd, "Tranches": K}

def backtest_topN_realistic(
    df: pl.DataFrame,
    pred_col="y_pred",
    ret_col="y_true",
    asof_col="asof",
    ent_col="entity",
    N=3,
    long_short=False,
    rebalance_every=1,
    tc_bps=5.0,
    max_weight=0.10,
) -> Dict[str, float]:
    dates = df.select(asof_col).unique().sort(asof_col)[asof_col].to_list()
    prev_w = pl.DataFrame({ent_col: [], "w": []})
    daily = []
    turn_series = []

    for i, d in enumerate(dates):
        cost = 0.0
        if i % rebalance_every == 0:
            snap = df.filter(pl.col(asof_col) == d).select([ent_col, pred_col])
            if snap.height == 0:
                daily.append(0.0); continue

            snap = snap.with_columns(
                pl.col(pred_col).rank("dense", descending=True).alias("_r_hi"),
                pl.col(pred_col).rank("dense", descending=False).alias("_r_lo")
            )
            longs = snap.filter(pl.col("_r_hi") <= N)[ent_col].to_list()
            shorts = snap.filter(pl.col("_r_lo") <= N)[ent_col].to_list() if long_short else []

            names = longs + shorts
            if len(names) == 0:
                w_now = pl.DataFrame({ent_col: [], "w": []})
            else:
                sgn = np.array([1]*len(longs) + ([-1]*len(shorts)), float)
                w = sgn / np.sum(np.abs(sgn))
                w_now = pl.DataFrame({ent_col: names, "w": w})

            if w_now.height:
                w_now = w_now.with_columns(pl.col("w").clip(-max_weight, max_weight))
                scale = float(w_now.select(pl.col("w").abs().sum().alias("s")).item())
                if scale > 0:
                    w_now = w_now.with_columns((pl.col("w")/scale).alias("w"))

            if prev_w.height:
                mv = w_now.join(prev_w.rename({"w": "w_prev"}), on=ent_col, how="full").fill_null(0.0)
                turn = float(mv.select((pl.col("w") - pl.col("w_prev")).abs().sum().alias("t")).item())
            else:
                mv = w_now.with_columns(pl.lit(0.0).alias("w_prev"))
                turn = float(mv.select(pl.col("w").abs().sum().alias("t")).item())
            cost = (tc_bps / 1e4) * turn
            prev_w = w_now
            turn_series.append(turn)

        today = df.filter(pl.col(asof_col) == d).select([ent_col, ret_col])
        pnl = today.join(prev_w, on=ent_col, how="left").fill_null(0.0)
        day_ret = float(pnl.select((pl.col(ret_col) * pl.col("w")).sum().alias("r")).item()) - cost
        daily.append(day_ret)

    r = np.array(daily, float)
    if r.size == 0:
        return {"n_periods": 0, "CAGR": float("nan"), "Sharpe": float("nan"), "MaxDD": float("nan")}
    eq = (1.0 + r).cumprod()
    sharpe = (r.mean() / (r.std(ddof=1) + 1e-12)) * np.sqrt(252.0 / max(1.0, FWD_H))
    cagr = (eq[-1] ** (252.0 / (len(eq) * max(1.0, FWD_H))) - 1.0)
    peak = np.maximum.accumulate(eq)
    mdd = float(((peak - eq) / peak).max())
    return {
        "n_periods": int(len(r)),
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDD": mdd,
        "Turnover_avg": float(np.mean(turn_series) if turn_series else float("nan")),
        "Cost_bps": float(tc_bps),
    }

def rolling_cv(df: pl.DataFrame, features: List[str], target: str, folds: int = 4) -> Dict[str, float]:
    df = df.with_columns(pl.col("asof").cast(pl.Datetime)).sort(["asof", "entity"])
    dates = df.select("asof").unique().sort("asof")["asof"].to_list()
    if len(dates) < folds + 1:
        return {"r2_mean": float("nan"), "rmse_mean": float("nan")}

    r2s, rmses = [], []
    for frac in np.linspace(0.5, 0.9, num=folds):
        k = max(5, int(len(dates) * frac))
        d_train_end = dates[k - 1]
        d_test_end_idx = min(k + max(5, len(dates) // 20), len(dates) - 1)
        d_test_end = dates[d_test_end_idx]

        tr = df.filter(pl.col("asof") <= d_train_end)
        te = df.filter((pl.col("asof") > d_train_end) & (pl.col("asof") <= d_test_end))
        if tr.is_empty() or te.is_empty():
            continue

        Xtr, ytr = to_xy(tr, features, target)
        Xte, yte = to_xy(te, features, target)
        mdl = fit_model(MODEL)
        mdl.fit(Xtr, ytr)
        p = mdl.predict(Xte)
        m = metrics_from_preds(yte, p)
        r2s.append(m["r2"]); rmses.append(m["rmse"])

    if not r2s:
        return {"r2_mean": float("nan"), "rmse_mean": float("nan")}
    return {"r2_mean": float(np.mean(r2s)), "rmse_mean": float(np.mean(rmses))}

# ---------- Rank-IC helpers ----------
def rank_ic(df: pl.DataFrame, pred_col="y_pred", target_col="y_true", asof_col="asof") -> float:
    outs = []
    for d, g in df.group_by(asof_col, maintain_order=True):
        p = g.get_column(pred_col).to_numpy()
        y = g.get_column(target_col).to_numpy()
        if len(p) >= 3 and np.isfinite(p).all() and np.isfinite(y).all():
            rp = p.argsort().argsort().astype(float); rp -= rp.mean()
            ry = y.argsort().argsort().astype(float); ry -= ry.mean()
            denom = (np.sqrt((rp**2).sum()) * np.sqrt((ry**2).sum()))
            if denom > 0:
                outs.append(float((rp @ ry) / denom))
    return float(np.mean(outs)) if outs else float("nan")

def global_rank_ic(df: pl.DataFrame, pred="y_pred", target="y_true") -> float:
    p = df[pred].to_numpy()
    y = df[target].to_numpy()
    if p.size < 3: return float("nan")
    rp = p.argsort().argsort().astype(float); rp -= rp.mean()
    ry = y.argsort().argsort().astype(float); ry -= ry.mean()
    denom = np.sqrt((rp**2).sum()) * np.sqrt((ry**2).sum())
    return float((rp @ ry) / denom) if denom > 0 else float("nan")


# -------------- Main --------------
if __name__ == "__main__":
    report = RunReport("SignalForge · Model Run")
    report.set_config(
        MODEL=MODEL, TOPN=TOPN, FWD_H=FWD_H,
        TEST_FRACTION=TEST_FRACTION, TC_BPS=TC_BPS,
        REBAL_EVERY=REBAL_EVERY, LONG_SHORT=LONG_SHORT,
        MAX_WEIGHT=MAX_WEIGHT, DECILES=DECILES,
        CS_ZSCORE=CS_ZSCORE, DEMEAN_TARGET=DEMEAN_TARGET,
        LEAKAGE_ON_ALL=LEAKAGE_ON_ALL, LEAKAGE_USE_PAST_FEATURE=LEAKAGE_USE_PAST_FEATURE,
        PRINT_GLOBAL_RIC=PRINT_GLOBAL_RIC,
    )

    t0 = time.perf_counter()
    os.environ.setdefault("SF_DATA_ROOT", "./_data")
    feat_root = Path(os.environ["SF_DATA_ROOT"]) / "features"
    cands = sorted(feat_root.glob("daily_*.parquet"))
    if not cands:
        raise SystemExit("No features file found. Run run_features.py first.")
    report.add_step("Scan features directory", time.perf_counter()-t0, {"count": len(cands)})

    # accept files like *_fwdH_90.parquet or *_fwdH_20-90.parquet
    tags = (f"_fwdH_{FWD_H}.", f"_fwdH_{FWD_H}-")
    pref = [p for p in cands if any(t in p.name for t in tags)]

    use_list = pref if pref else cands

    def end_date_key(p: Path):
        try:
            base = p.name[:-8]
            tail = base.split("_to_")[1]
            end_str = tail.split("_fwdH_")[0]
            return end_str
        except Exception:
            return ""

    use_list = sorted(use_list, key=end_date_key)

    def detect_in_file(p: Path) -> bool:
        try:
            df_head = pl.read_parquet(p, n_rows=10)
            _ = detect_target(df_head, FWD_H)
            return True
        except Exception:
            return False

    t1 = time.perf_counter()
    candidates = [p for p in use_list[::-1] if detect_in_file(p)]
    if not candidates:
        raise SystemExit(f"No features file contains a target for FWD_H={FWD_H}.")
    chosen = candidates[0]
    report.add_step("Choose features parquet", time.perf_counter()-t1, {"file": chosen.name})

    # --- Manual override for debugging/toy data ---
    override_path = os.environ.get("SF_FEATURES_PATH")
    if override_path:
        override = Path(override_path)
        if not override.exists():
            raise SystemExit(f"[error] SF_FEATURES_PATH not found: {override}")
        t2 = time.perf_counter()
        df = pl.read_parquet(override)
        report.add_step("Load features parquet (override)", time.perf_counter()-t2, {"rows": df.height, "cols": df.width})
        print(f"[info] using features file (override): {override.name} from {override.parent}")
    else:
        t2 = time.perf_counter()
        df = pl.read_parquet(chosen)
        report.add_step("Load features parquet", time.perf_counter()-t2, {"rows": df.height, "cols": df.width})
        print(f"[info] using features file: {chosen.name} from {chosen.parent}")


    # Build requested feature set
    FEATURES: List[str] = []
    FEATURES += BASE_FEATURES
    FEATURES += RESID_FEATURES
    if FWD_H >= 90:
        FEATURES += LONG_H_FEATURES

    present = set(df.columns)
    missing = [f for f in FEATURES if f not in present]
    FEATURES = [f for f in FEATURES if f in present]
    if missing:
        msg = f"dropping {len(missing)} missing features"
        report.add_step("Prune missing features", 0.0, {"missing_sample": missing[:12], "count": len(missing)})
    if not FEATURES:
        raise SystemExit("No usable FEATURES found in features parquet. Rebuild features or adjust list.")

    target = detect_target(df, FWD_H)
    cols_needed = ["asof", "entity"] + FEATURES + [target]
    miss2 = [c for c in cols_needed if c not in df.columns]
    if miss2:
        raise SystemExit(f"Missing columns {miss2}. Rebuild features or adjust FEATURES list.")

    df = df.select(cols_needed).drop_nulls()
    df = ensure_date(df)

    if CS_ZSCORE:
        t3 = time.perf_counter()
        df = zscore_by_date(df, FEATURES, "asof")
        report.add_step("Cross-sectional z-score", time.perf_counter()-t3, {"cols": len(FEATURES)})
    if DEMEAN_TARGET:
        df = df.with_columns((pl.col(target) - pl.col(target).mean().over("asof")).alias(target))

    df = sanitize_features(df, FEATURES, target)

    # Split
    t4 = time.perf_counter()
    train_df, test_df = time_split(df, TEST_FRACTION)
    report.add_step("Time split", time.perf_counter()-t4, {"n_train": int(train_df.height), "n_test": int(test_df.height)})

    # Fit & predict
    t5 = time.perf_counter()
    Xtr, ytr = to_xy(train_df, FEATURES, target)
    Xte, yte = to_xy(test_df, FEATURES, target)
    model = fit_model(MODEL)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    m = metrics_from_preds(yte, pred)
    report.add_step("Fit & predict", time.perf_counter()-t5, {"r2": round(m["r2"],6), "rmse": round(m["rmse"],6)})

    test_preds = test_df.with_columns(pl.Series("pred", pred)).select(["asof", "entity", "pred", target])

    # Latest ranking (last test date)
    test_dates = test_df.select("asof").unique().sort("asof")["asof"].to_list()
    last_day = test_dates[-1]
    last_slice = test_df.filter(pl.col("asof") == last_day)
    last_pred = model.predict(last_slice.select(FEATURES).to_numpy())
    ranking = dict(sorted(zip(last_slice["entity"].to_list(), last_pred), key=lambda x: -x[1]))
    top_show = list(ranking.items())[:TOPN]

    # Quick backtest
    def topn_backtest_quick(preds: pl.DataFrame, target_col: str, n: int = 3) -> Dict[str, float]:
        picks = preds.with_columns(pl.col("pred").rank("dense", descending=True).over("asof").alias("rank")) \
                     .filter(pl.col("rank") <= n)
        pnl = picks.group_by("asof").agg(pl.col(target_col).mean().alias("port_ret")).sort("asof")
        pnl = pnl.with_columns(((1.0 + pl.col("port_ret")).cum_prod().alias("cum")))
        series = pnl.select("port_ret").to_numpy().ravel()
        mu = float(series.mean()) if series.size else float("nan")
        sd = float(series.std(ddof=1)) if series.size > 1 else float("nan")
        sharpe = (mu / sd) * (252.0 ** 0.5) / max(1.0, FWD_H) ** 0.5 if (sd and sd > 0 and np.isfinite(sd)) else float("nan")
        last_cum = float(pnl["cum"][-1]) if pnl.height > 0 else 1.0
        return {"n_periods": int(pnl.height), "cum": last_cum, "avg": mu, "vol": sd, "sharpe_approx": sharpe}

    bt_quick = topn_backtest_quick(test_preds, target_col=target, n=TOPN)

    # Realistic backtest + tranches
    bt = backtest_topN_realistic(
        test_preds.rename({target:"y_true","pred":"y_pred"}),
        pred_col="y_pred", ret_col="y_true",
        N=TOPN, long_short=LONG_SHORT,
        rebalance_every=REBAL_EVERY, tc_bps=TC_BPS, max_weight=MAX_WEIGHT,
    )
    bt_tr = backtest_topN_tranches(
        test_preds.rename({target:"y_true","pred":"y_pred"}),
        pred_col="y_pred", ret_col="y_true",
        N=TOPN, H=FWD_H, K=max(3, FWD_H // 10),
        tc_bps=TC_BPS, max_weight=MAX_WEIGHT, long_short=LONG_SHORT,
    )

    # Rolling CV
    cv = rolling_cv(df, FEATURES, target, folds=4)

    # Signal diagnostics
    dfp = test_preds.rename({target: "y_true", "pred": "y_pred"}).sort(["asof", "entity"])
    ic_tbl = cs_ic_rankic_by_date(dfp, "y_pred", "y_true")
    ic_mean = float(ic_tbl["ic"].mean()) if ic_tbl.height else float("nan")
    ic_std  = float(ic_tbl["ic"].std(ddof=1)) if ic_tbl.height > 1 else float("nan")
    ric_mean = float(ic_tbl["rank_ic"].mean()) if ic_tbl.height else float("nan")
    deciles = decile_curve(dfp, "y_pred", "y_true", n_bins=DECILES)

    # Feature importance
    feat_imp = None
    try:
        fi = getattr(model, "feature_importances_", None)
        if fi is not None:
            order = np.argsort(fi)[::-1]
            feat_imp = [(FEATURES[i], float(fi[i])) for i in order[:20]]
    except Exception:
        pass

    # Leakage audit
    H = max(1, FWD_H)
    def _align_cols(df_in: pl.DataFrame, pred_col: str, tgt_col: str) -> pl.DataFrame:
        return df_in.select(["asof", "entity", pred_col, tgt_col]).rename({pred_col: "y_pred", tgt_col: "y_true"})

    if LEAKAGE_ON_ALL:
        pred_all = model.predict(df.select(FEATURES).to_numpy())
        all_preds = df.select(["asof","entity", target]).with_columns(pl.Series("y_pred", pred_all)).rename({target:"y_true"})
        audit_df = all_preds.sort(["asof","entity"])
        df_all_for_past = df
    else:
        audit_df = dfp
        df_all_for_past = df

    ric_future = rank_ic(audit_df, "y_pred", "y_true")

    ric_pastH_feat = float("nan")
    if LEAKAGE_USE_PAST_FEATURE:
        past_feat_col = f"ret_{H}d"
        if past_feat_col in df_all_for_past.columns:
            past_join = (
                df_all_for_past.select(["asof","entity", past_feat_col])
                               .rename({past_feat_col: "y_pastH"})
                               .join(audit_df, on=["asof","entity"], how="inner")
                               .drop_nulls(subset=["y_pastH"])
            )
            if past_join.height:
                ric_pastH_feat = rank_ic(_align_cols(past_join, "y_pred", "y_pastH"), "y_pred", "y_true")

    pred_fwdH = audit_df.with_columns(pl.col("y_pred").shift(H).over("entity").alias("y_pred_shift")).drop_nulls(subset=["y_pred_shift"])
    ric_future_fwdH = rank_ic(_align_cols(pred_fwdH, "y_pred_shift", "y_true"), "y_pred", "y_true")

    tgt_backH = audit_df.with_columns(pl.col("y_true").shift(H).over("entity").alias("y_true_shift")).drop_nulls(subset=["y_true_shift"])
    ric_backshiftH = rank_ic(_align_cols(tgt_backH, "y_pred", "y_true_shift"), "y_pred", "y_true")

    rng = np.random.default_rng(123)
    def shuffle_within_date(g: pl.DataFrame) -> pl.DataFrame:
        y = g["y_true"].to_numpy()
        n = y.shape[0]
        if n <= 1:
            y_shuf = y
        else:
            idx = rng.permutation(n)
            y_shuf = y[idx].copy()
        return g.with_columns(pl.Series("y_true_shuf", y_shuf))

    shuf = audit_df.group_by("asof", maintain_order=True).map_groups(shuffle_within_date)
    ric_shuf = rank_ic(_align_cols(shuf, "y_pred", "y_true_shuf"), "y_pred", "y_true")

    def permute_entities(g: pl.DataFrame) -> pl.DataFrame:
        y = g["y_true"].to_numpy()
        n = y.shape[0]
        if n > 1:
            y_perm = np.concatenate([y[-1:], y[:-1]], axis=0).copy()
        else:
            y_perm = y
        return g.with_columns(pl.Series("y_true_perm", y_perm))

    perm = audit_df.group_by("asof", maintain_order=True).map_groups(permute_entities)
    ric_perm = rank_ic(_align_cols(perm, "y_pred", "y_true_perm"), "y_pred", "y_true")

    global_stats = {}
    if PRINT_GLOBAL_RIC:
        global_stats = {
            "RIC_future_global": global_rank_ic(audit_df),
            "RIC_pastH_feat_global": global_rank_ic(_align_cols(past_join, "y_pred", "y_pastH")) if LEAKAGE_USE_PAST_FEATURE and 'past_join' in locals() and past_join.height else float("nan"),
            "RIC_future_fwdH_global": global_rank_ic(_align_cols(pred_fwdH, "y_pred_shift", "y_true")) if pred_fwdH.height else float("nan"),
            "RIC_backshiftH_global": global_rank_ic(_align_cols(tgt_backH, "y_pred", "y_true_shift")) if tgt_backH.height else float("nan"),
        }

    # ---------------- UI REPORT ----------------
    # Dataset section
    report.set_dataset(
        rows=df.height,
        cols=df.width,
        columns=", ".join((["asof","entity"] + FEATURES)[:20]) + (" ..." if len(FEATURES) > 18 else "")
    )

    # Key tables via "Targets" block and "Notes" (keeps reporter simple)
    # 1) Core metrics
    report.add_target_diag(
        horizon=FWD_H,
        cols=["r2", "rmse"],
        null_rates={"r2": m["r2"] if np.isfinite(m["r2"]) else float("nan"),
                    "rmse": m["rmse"] if np.isfinite(m["rmse"]) else float("nan")}
    )
    # 2) Backtests
    report.add_note(f"Quick backtest (Top{TOPN}): { {k: round(v,6) if isinstance(v,float) else v for k,v in bt_quick.items()} }")
    report.add_note(f"Realistic backtest: { {k: round(v,6) if isinstance(v,float) else v for k,v in bt.items()} }")
    report.add_note(f"Tranche backtest: { {k: round(v,6) if isinstance(v,float) else v for k,v in bt_tr.items()} }")
    # 3) ICs / Rank-IC
    report.add_note(f"Signal quality: IC_mean={round(ic_mean,6)}, IC_std={round(ic_std,6) if np.isfinite(ic_std) else ic_std}, RankIC_mean={round(ric_mean,6)}")
    # 4) CV
    report.add_note(f"Rolling CV (expanding): r2_mean={round(cv['r2_mean'],6) if np.isfinite(cv['r2_mean']) else cv['r2_mean']}, rmse_mean={round(cv['rmse_mean'],6) if np.isfinite(cv['rmse_mean']) else cv['rmse_mean']}")
    # 5) Feature importance (top 10) + latest ranking (top N)
    if feat_imp:
        report.add_note("Top feature importance: " + ", ".join([f"{k}:{round(v,4)}" for k,v in feat_imp[:10]]))
    if top_show:
        report.add_note("Latest ranking (top): " + ", ".join([f"{a}:{round(b,4)}" for a,b in top_show]))

    # Leakage audit summary
    leak_summary = {
        "dataset": "all" if LEAKAGE_ON_ALL else "test_only",
        "H": H,
        "RankIC_future": ric_future,
        "RankIC_pastH_feat": ric_pastH_feat if LEAKAGE_USE_PAST_FEATURE else None,
        "RankIC_future_fwdH": ric_future_fwdH,
        "RankIC_backshiftH": ric_backshiftH,
        "RankIC_shuffled": ric_shuf,
        "RankIC_entity_perm": ric_perm,
    }
    if PRINT_GLOBAL_RIC:
        leak_summary.update(global_stats)
    report.add_note("Leakage audit: " + str({k: (round(v,6) if isinstance(v,float) else v) for k,v in leak_summary.items()}))

    # Write UI report
    report.finish()
    if UI_ON:
        ui_dir = Path("./_state/reports")
        html_path = report.write(ui_dir, basename="model")
        print(f"[ui] report: {html_path}")
    else:
        print("[ui] SF_UI_REPORT=0 (skipping HTML report)")

    # Minimal console tail
    print("metrics (holdout):", {"r2": m["r2"], "rmse": m["rmse"],
                                 "n_train": int(train_df.height), "n_test": int(test_df.height)})
    print(f"backtest top{TOPN} (approx):", bt_quick)
    print(f"backtest top{TOPN} (realistic):", {k: bt[k] for k in ["n_periods","CAGR","Sharpe","MaxDD","Turnover_avg","Cost_bps"]})
