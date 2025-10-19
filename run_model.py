import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl
from models.baseline import fit_random_forest

if __name__ == "__main__":
    os.environ.setdefault("SF_DATA_ROOT", "./_data")
    feats = sorted((Path(os.environ["SF_DATA_ROOT"]) / "features").glob("daily_*.parquet"))
    if not feats:
        raise SystemExit("No features file found. Run run_features.py first.")
    df = pl.read_parquet(feats[-1])
    model, metrics, ranking = fit_random_forest(df)
    print("metrics:", metrics)
    print("latest ranking (top 5):", list(ranking.items())[:5])
