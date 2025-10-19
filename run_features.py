import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl
from datetime import date, timedelta
from features.signal_features import build_daily_features, add_forward_return

if __name__ == "__main__":
    os.environ.setdefault("SF_DATA_ROOT", "./_data")

    # auto-detect available date range from partitions of market.price
    root = Path(os.environ["SF_DATA_ROOT"]) / "signalgraph" / "kind=market.price"
    asofs = sorted(p.name.split("asof=")[-1] for p in root.glob("asof=*") if p.is_dir())
    if not asofs:
        raise SystemExit("No signal partitions found. Run run_signalgraph.py first.")
    start = asofs[max(0, len(asofs)-180)]  # ~6 months
    end   = asofs[-1]

    df = build_daily_features(start, end)
    df = add_forward_return(df, horizon=20)

    out_dir = Path(os.environ["SF_DATA_ROOT"]) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_{start}_to_{end}.parquet"
    df.write_parquet(out_path)

    print("wrote:", out_path)
    print(df.head(8))
    print(df.tail(8))
    print("rows:", df.height, "cols:", df.width)
