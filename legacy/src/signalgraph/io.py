import os
import polars as pl

def _root() -> str:
    return os.environ.get("SF_DATA_ROOT", "_data")

def write_signals(df: pl.DataFrame, kind_col: str = "kind", asof_col: str = "asof"):
    root = os.path.join(_root(), "signalgraph")
    df = df.drop_nulls(subset=[kind_col, asof_col])
    for kind, asof in df.select(kind_col, asof_col).unique().iter_rows():
        out_dir = os.path.join(root, f"kind={kind}", f"asof={asof}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"part-{os.urandom(4).hex()}.parquet")
        df.filter((pl.col(kind_col) == kind) & (pl.col(asof_col) == asof)).write_parquet(path)
