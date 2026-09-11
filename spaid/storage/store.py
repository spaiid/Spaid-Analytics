"""The parquet store.

One module knows where every dataset lives, what shape it should be, and how
fresh it is, so the pipeline, the API and the data-health view all read the same
truth rather than three slightly different ones.

Writes go through `write()`, which validates against the declared schema first.
That is the point: a table that does not match its contract fails at the moment
it is produced, next to the code that produced it, instead of six steps later as
a column of nulls in a recommendation.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from spaid.config.settings import ARTIFACTS, CURATED, DERIVED, RAW
from spaid.storage import migrations
from spaid.storage.schema import ALL_TABLES, SCHEMA_VERSION, TableSpec, validate

log = logging.getLogger(__name__)

LAYERS: dict[str, Path] = {
    "raw": RAW,
    "curated": CURATED,
    "derived": DERIVED,
    "artifacts": ARTIFACTS,
}


@dataclass
class DatasetInfo:
    """The sidecar written beside every parquet file."""

    name: str
    layer: str
    path: str
    rows: int
    columns: int
    schema_version: int
    schema: dict[str, str]
    updated_at: str
    entities: int | None = None
    date_min: str | None = None
    date_max: str | None = None
    source: str | None = None
    note: str | None = None
    # Per-column null counts: the data-health view reads these rather than
    # rescanning a 600 MB file, and a coverage regression becomes visible.
    null_counts: dict[str, int] = field(default_factory=dict)

    @property
    def age_hours(self) -> float:
        try:
            then = datetime.fromisoformat(self.updated_at)
        except ValueError:
            return float("inf")
        if then.tzinfo is None:
            then = then.replace(tzinfo=UTC)
        return (datetime.now(UTC) - then).total_seconds() / 3600.0


def _spec(name: str) -> TableSpec:
    try:
        return ALL_TABLES[name]
    except KeyError:
        raise KeyError(
            f"unknown table {name!r}; declare it in spaid.storage.schema"
        ) from None


def path_for(name: str) -> Path:
    spec = _spec(name)
    return LAYERS[spec.layer] / f"{name}.parquet"


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def write(
    df: pl.DataFrame,
    name: str,
    *,
    source: str | None = None,
    note: str | None = None,
    strict: bool = True,
) -> Path:
    """Validate and persist `df` as the canonical copy of table `name`.

    The write is atomic: a crash midway leaves the previous version intact
    rather than a truncated parquet that reads as an empty universe.
    """
    spec = _spec(name)
    out = validate(df, spec, strict=strict)

    path = path_for(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        out.write_parquet(tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

    ecol, dcol = spec.entity_col, spec.date_col
    info = DatasetInfo(
        name=name,
        layer=spec.layer,
        path=str(path),
        rows=out.height,
        columns=out.width,
        schema_version=SCHEMA_VERSION,
        schema={c: str(t) for c, t in out.schema.items()},
        updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        entities=int(out[ecol].n_unique()) if ecol and ecol in out.columns else None,
        date_min=str(out[dcol].min()) if dcol and dcol in out.columns and out.height else None,
        date_max=str(out[dcol].max()) if dcol and dcol in out.columns and out.height else None,
        source=source,
        note=note or spec.description or None,
        null_counts={c: int(out[c].null_count()) for c in out.columns},
    )
    _meta_path(path).write_text(json.dumps(asdict(info), indent=2))
    log.info("wrote %s: %d rows x %d cols -> %s", name, out.height, out.width, path)
    return path


def read(name: str, *, required: bool = False) -> pl.DataFrame | None:
    """Read a table, migrating it forward and checking its declared schema.

    Declared migrations are applied first and written back, so a rename costs one
    read rather than being paid forever or forcing a refetch. Anything still
    mismatched after that raises rather than returning a frame the caller will
    misinterpret: a column of nulls where a number should be is worse than an
    error, because it propagates silently into a recommendation.
    """
    path = path_for(name)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"table {name!r} has not been built yet (expected at {path})"
            )
        return None

    df = pl.read_parquet(path)
    migrated, applied = migrations.apply(df, name)
    df = write_migrated(migrated, name) if applied else migrated
    return validate(df, _spec(name), strict=False)


def write_migrated(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Persist a migrated frame, keeping the read path working if the write fails.

    A migration that cannot be written back is still a correct read: the caller
    gets the migrated data and the next read repeats the work. Failing the read
    because the disk is full would be worse.
    """
    try:
        write(df, name, note=f"migrated to schema version {SCHEMA_VERSION}", strict=False)
    except Exception as exc:
        log.warning("could not persist the migrated %s table (%s); it will migrate again "
                    "on the next read", name, exc)
    return df


def scan(name: str) -> pl.LazyFrame | None:
    """Lazily scan a table. For the large ones, this avoids loading 600 MB."""
    path = path_for(name)
    if not path.exists():
        return None
    return pl.scan_parquet(path)


def exists(name: str) -> bool:
    return path_for(name).exists()


def info(name: str) -> DatasetInfo | None:
    mp = _meta_path(path_for(name))
    if not mp.exists():
        return None
    raw = json.loads(mp.read_text())
    known = {f for f in DatasetInfo.__dataclass_fields__}
    return DatasetInfo(**{k: v for k, v in raw.items() if k in known})


def age_hours(name: str) -> float | None:
    path = path_for(name)
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 3600.0


def is_fresh(name: str, max_age_hours: float) -> bool:
    age = age_hours(name)
    return age is not None and age < max_age_hours


def is_current_schema(name: str) -> bool:
    """Whether the stored copy of `name` still matches its declared schema."""
    path = path_for(name)
    if not path.exists():
        return True
    try:
        stored = pl.read_parquet_schema(path)
    except Exception:
        return False
    declared = _spec(name).schema
    if all(col in stored for col in declared):
        return True
    # A table with a pending migration is not stale; it is one read away from
    # being current.
    return bool(migrations.pending(name, set(stored)))


def upsert(
    df: pl.DataFrame,
    name: str,
    *,
    source: str | None = None,
    note: str | None = None,
) -> Path:
    """Merge `df` into an existing table on its primary key, new rows winning.

    Used for tables that accumulate forward -- estimate snapshots, journal
    outcomes -- where a full rebuild would throw away history that the provider
    can no longer supply.

    When the stored copy predates a schema change it cannot be merged with, so it
    is replaced and that is logged rather than silently swallowed. This is safe
    only because every table reaching this path is refetchable; append-only
    tables use `append`, which refuses.
    """
    spec = _spec(name)
    incoming = validate(df, spec, strict=False)

    if not is_current_schema(name):
        log.warning(
            "%s: the stored table predates the current schema and cannot be merged; "
            "replacing it with the freshly fetched data. Any history the provider no "
            "longer supplies is lost, which is why this is a warning and not silent.",
            name,
        )
        return write(incoming, name, source=source, note=note)

    current = read(name)
    if current is None or current.is_empty():
        return write(incoming, name, source=source, note=note)

    pk = list(spec.primary_key)
    merged = (
        pl.concat([current, incoming], how="vertical_relaxed")
        .unique(subset=pk, keep="last")
        .sort(pk)
    )
    return write(merged, name, source=source, note=note)


def append(df: pl.DataFrame, name: str, *, source: str | None = None) -> Path:
    """Append rows to an append-only table (the research journal).

    Existing rows are never modified. A duplicate primary key is an error, not
    an overwrite, because silently replacing a historical recommendation is
    exactly what the mission forbids.
    """
    spec = _spec(name)
    incoming = validate(df, spec, strict=False)
    current = read(name)
    if current is None or current.is_empty():
        return write(incoming, name, source=source)

    pk = list(spec.primary_key)
    clash = incoming.join(current.select(pk), on=pk, how="semi")
    if clash.height:
        raise ValueError(
            f"{name}: {clash.height} rows would overwrite existing entries "
            f"(primary key {pk}); this table is append-only"
        )
    merged = pl.concat([current, incoming], how="vertical_relaxed")
    return write(merged, name, source=source)


def write_json(obj, name: str, *, layer: str = "artifacts") -> Path:
    base = LAYERS[layer]
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}.json"
    fd, tmp_name = tempfile.mkstemp(dir=base, suffix=".json.tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(obj, indent=2, default=str))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_json(name: str, *, layer: str = "artifacts"):
    path = LAYERS[layer] / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def health() -> list[dict]:
    """A row per declared table: present, how big, how fresh, how complete.

    This is the data-health view's only source, so what the user sees is exactly
    what the pipeline wrote.
    """
    from spaid.config.settings import SETTINGS

    thresholds = {
        "prices": SETTINGS.freshness.prices_hours,
        "observations": SETTINGS.freshness.fundamentals_hours,
        "fundamentals": SETTINGS.freshness.fundamentals_hours,
        "companies": SETTINGS.freshness.universe_hours,
        "securities": SETTINGS.freshness.universe_hours,
        "universe_membership": SETTINGS.freshness.universe_hours,
        "estimates": SETTINGS.freshness.estimates_hours,
        "earnings_events": SETTINGS.freshness.estimates_hours,
    }

    out: list[dict] = []
    for name, spec in ALL_TABLES.items():
        meta = info(name)
        if meta is None:
            out.append(
                {
                    "name": name,
                    "layer": spec.layer,
                    "status": "missing",
                    "rows": None,
                    "entities": None,
                    "updated_at": None,
                    "age_hours": None,
                    "span": None,
                    "detail": spec.description,
                }
            )
            continue
        age = meta.age_hours
        limit = thresholds.get(name)
        status = "ok"
        if meta.rows == 0:
            status = "empty"
        elif limit is not None and age > limit:
            status = "stale"
        span = (
            f"{meta.date_min} to {meta.date_max}"
            if meta.date_min and meta.date_max
            else None
        )
        out.append(
            {
                "name": name,
                "layer": spec.layer,
                "status": status,
                "rows": meta.rows,
                "entities": meta.entities,
                "updated_at": meta.updated_at,
                "age_hours": round(age, 1),
                "max_age_hours": limit,
                "span": span,
                "source": meta.source,
                "detail": meta.note or spec.description,
                "null_counts": meta.null_counts,
            }
        )
    return out
