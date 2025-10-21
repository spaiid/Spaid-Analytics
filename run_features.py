# run_features.py
import os, sys
from pathlib import Path

# make `src` importable when running as a script
sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl
from features.signal_features import build_daily_features
from features.residuals import add_residual_features
# NEW: targets module (place add_forward_return there as discussed)
from features.targets import add_forward_return


def parse_horizons(env_val: str) -> list[int]:
    return [int(x.strip()) for x in env_val.split(",") if x.strip()]


if __name__ == "__main__":
    os.environ.setdefault("SF_DATA_ROOT", "./_data")

    # -----------------------------
    # Config (via environment vars)
    # -----------------------------
    # Example:
    #   FWD_HS="20,90"
    #   TARGET="ex_mkt"   # one of: raw | ex_mkt | ex_sec | both
    #   BENCHMARK_ENTITY="SPY"
    #   RESID_ON="1"
    #   RESID_WIN="60"
    #   SECTOR_COL="sector"
    #   FEAT_LOOKBACK_DAYS="450"
    #   USE_LOG_RET="1"
    # -----------------------------
    H_LIST = parse_horizons(os.environ.get("FWD_HS", os.environ.get("FWD_H", "20")))
    LOOKBACK_DAYS = int(os.environ.get("FEAT_LOOKBACK_DAYS", "450"))

    TARGET_MODE = os.environ.get("TARGET", "ex_mkt").lower()  # raw | ex_mkt | ex_sec | both
    BENCHMARK_ENTITY = os.environ.get("BENCHMARK_ENTITY", "SPY")
    USE_LOG_RET = os.environ.get("USE_LOG_RET", "1") == "1"

    RESID_ON = os.environ.get("RESID_ON", "1") == "1"
    RESID_WIN = int(os.environ.get("RESID_WIN", "60"))
    SECTOR_COL = os.environ.get("SECTOR_COL", "sector")

    print(f"Config: H_LIST={H_LIST}, TARGET={TARGET_MODE}, BENCHMARK={BENCHMARK_ENTITY}, "
    f"RESID_ON={RESID_ON}, RESID_WIN={RESID_WIN}, LOOKBACK={LOOKBACK_DAYS}, USE_LOG_RET={USE_LOG_RET}")


    # -----------------------------
    # Detect available date range
    # -----------------------------
    sg_root = Path(os.environ["SF_DATA_ROOT"]) / "signalgraph" / "kind=market.price"
    asofs = sorted(p.name.split("asof=")[-1] for p in sg_root.glob("asof=*") if p.is_dir())
    if not asofs:
        raise SystemExit("No signal partitions found. Run run_signalgraph.py first.")

    start_idx = max(0, len(asofs) - LOOKBACK_DAYS)
    start = asofs[start_idx]
    end = asofs[-1]

    # -----------------------------
    # Build core daily features
    # -----------------------------
    # Expected at minimum: asof (datetime), entity (ticker/id),
    # price column ("close" or "adj_close"), and optional sector column.
    df = build_daily_features(start, end)

    # -----------------------------
    # Residualized features (optional)
    # -----------------------------
    if RESID_ON:
        df = add_residual_features(df, window=RESID_WIN, sector_col=SECTOR_COL)

    # diagnostics
    res_cols = [c for c in df.columns if c.startswith(("resid_", "beta_mkt", "beta_sec", "alpha_"))]
    print(f"[info] residual columns present: {res_cols[:20]}{' ...' if len(res_cols)>20 else ''}")
    if not res_cols:
        print("[warn] residualization produced no columns. Check that 'ret_1d' exists and RESID_WIN is not too large.")

    # -----------------------------
    # Forward targets (raw / excess)
    # -----------------------------
    # We create the requested targets for each horizon.
    # add_forward_return will append:
    #   - fwd{H}_ret                        (always)
    #   - fwd{H}_ret_ex_mkt  (if TARGET in {ex_mkt, both} and benchmark present)
    #   - fwd{H}_ret_ex_sec  (if TARGET in {ex_sec, both} and sector_col provided)
    # -----------------------------
    # Ensure expected price column exists
    price_col = "adj_close" if "adj_close" in df.columns else ("close" if "close" in df.columns else None)
    if price_col is None:
        raise SystemExit("No 'close' or 'adj_close' column found in features frame.")

    # Confirm benchmark existence if needed
    if TARGET_MODE in ("ex_mkt", "both"):
        if BENCHMARK_ENTITY not in df["entity"].unique().to_list():
            print(f"[warn] BENCHMARK_ENTITY '{BENCHMARK_ENTITY}' not found in entity column; market-excess targets will be null.")

    # Attach forward return targets per horizon
    # (The helper accepts DataFrame or LazyFrame; we pass DataFrame for simplicity.)
    for H in H_LIST:
        df = add_forward_return(
            df,
            horizon=H,
            date_col="asof",
            entity_col="entity",
            price_col=price_col,
            sector_col=SECTOR_COL if SECTOR_COL in df.columns else None,
            benchmark_entity=(BENCHMARK_ENTITY if TARGET_MODE in ("ex_mkt", "both") else None),
            sector_relative=(TARGET_MODE in ("ex_sec", "both")),
            use_log_returns=USE_LOG_RET,
        ).collect() if isinstance(df, pl.LazyFrame) else add_forward_return(
            df,
            horizon=H,
            date_col="asof",
            entity_col="entity",
            price_col=price_col,
            sector_col=SECTOR_COL if SECTOR_COL in df.columns else None,
            benchmark_entity=(BENCHMARK_ENTITY if TARGET_MODE in ("ex_mkt", "both") else None),
            sector_relative=(TARGET_MODE in ("ex_sec", "both")),
            use_log_returns=USE_LOG_RET,
        )

    # -----------------------------
    # Past returns (for reference)
    # -----------------------------
    # Adds arithmetic trailing returns ret_{H}d if missing.
    df = df.sort(["entity", "asof"])
    for H in H_LIST:
        col_name = f"ret_{H}d"
        if col_name not in df.columns:
            df = df.with_columns(
                ((pl.col(price_col) / pl.col(price_col).shift(H).over("entity")) - 1.0).alias(col_name)
            )

    # -----------------------------
    # Write output
    # -----------------------------
    out_dir = Path(os.environ["SF_DATA_ROOT"]) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    h_tag = "-".join(str(h) for h in H_LIST)
    tgt_tag = TARGET_MODE
    out_path = out_dir / f"daily_{start}_to_{end}_fwdH_{h_tag}_tgt_{tgt_tag}.parquet"

    df.write_parquet(out_path)

    print(f"[info] wrote: {out_path}")
    print(df.head(8))
    print(df.tail(8))
    print("rows:", df.height, "cols:", df.width)
