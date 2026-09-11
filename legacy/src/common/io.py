import os
import polars as pl

DATA_ROOT = os.getenv("SF_DATA_ROOT", "./_data")

def _path(layer: str, source: str, dt: str) -> str:
    return f"{DATA_ROOT}/{layer}={source}/dt={dt}"

def write_parquet_partition(df: pl.DataFrame, layer: str, source: str, dt: str) -> str:
    path = _path(layer, source, dt)
    os.makedirs(path, exist_ok=True)
    file = f"{path}/part-{dt}.parquet"
    df.write_parquet(file)
    return file

def read_parquet_partition(layer: str, source: str, dt: str) -> pl.DataFrame | None:
    path = _path(layer, source, dt)
    if not os.path.isdir(path):
        return None
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    if not files:
        return None
    return pl.concat([pl.read_parquet(f"{path}/{f}") for f in files])
