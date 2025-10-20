import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl
from features.signal_features import build_daily_features, add_forward_return
from features.residuals import add_residual_features

def parse_horizons(env_val: str) -> list[int]:
    return [int(x.strip()) for x in env_val.split(",") if x.strip()]

if __name__ == "__main__":
    os.environ.setdefault("SF_DATA_ROOT", "./_data")

    # --- Config ---
    # One or many forward horizons, e.g. "20" or "20,90"
    H_LIST = parse_horizons(os.environ.get("FWD_HS", os.environ.get("FWD_H", "20")))
    # How many calendar days of partitions to load (more for longer MAs/targets)
    LOOKBACK_DAYS = int(os.environ.get("FEAT_LOOKBACK_DAYS", "450"))

    # --- Detect available date range from signalgraph partitions ---
    sg_root = Path(os.environ["SF_DATA_ROOT"]) / "signalgraph" / "kind=market.price"
    asofs = sorted(p.name.split("asof=")[-1] for p in sg_root.glob("asof=*") if p.is_dir())
    if not asofs:
        raise SystemExit("No signal partitions found. Run run_signalgraph.py first.")

    start_idx = max(0, len(asofs) - LOOKBACK_DAYS)
    start = asofs[start_idx]
    end   = asofs[-1]

    # --- Build core daily features from your SignalGraph data ---
    df = build_daily_features(start, end)  # expects at least: asof, entity, price columns, etc.

    RESID_ON = os.environ.get("RESID_ON", "1") == "1"      # enable by default
    RESID_WIN = int(os.environ.get("RESID_WIN", "60"))     # rolling window for betas
    SECTOR_COL = os.environ.get("SECTOR_COL", "sector")    # if sector column exists

    if RESID_ON:
        df = add_residual_features(df, window=RESID_WIN, sector_col=SECTOR_COL)

        # --- diagnostic: confirm residual columns appeared ---
    res_cols = [c for c in df.columns if c.startswith(("resid_", "beta_mkt", "beta_sec", "alpha_"))]
    print(f"[info] residual columns present: {res_cols[:20]}{' ...' if len(res_cols)>20 else ''}")
    if not res_cols:
        print("[warn] residualization produced no columns. Check that 'ret_1d' exists and RESID_WIN is not too large.")

    # --- Add forward targets for each horizon ---
    for H in H_LIST:
        df = add_forward_return(df, horizon=H)  # creates ret_fwd_{H}d (and/or excess_fwd_{H}d)

    # --- (Optional) add matching past returns ret_{H}d if we have a price column ---
    close_col = "adj_close" if "adj_close" in df.columns else ("close" if "close" in df.columns else None)
    if close_col:
        df = df.sort(["entity", "asof"])
        for H in H_LIST:
            col_name = f"ret_{H}d"
            if col_name not in df.columns:
                df = df.with_columns(
                    ((pl.col(close_col) / pl.col(close_col).shift(H).over("entity")) - 1.0).alias(col_name)
                )

    # --- Write output ---
    out_dir = Path(os.environ["SF_DATA_ROOT"]) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    h_tag = "-".join(str(h) for h in H_LIST)
    out_path = out_dir / f"daily_{start}_to_{end}_fwdH_{h_tag}.parquet"
    df.write_parquet(out_path)

    print(f"wrote: {out_path}")
    print(df.head(8))
    print(df.tail(8))
    print("rows:", df.height, "cols:", df.width)
