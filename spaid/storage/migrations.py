"""Schema migrations for the parquet store.

A stored table is a persisted contract. When the contract changes, the data on
disk does not, and something has to reconcile them. Without this module the
options are both bad: refuse to read the old file, which makes a rename
unrecoverable, or read it loosely, which is how a renamed column silently
becomes a column of nulls.

Migrations are declared, versioned and forward-only. Each one states what it
changes and why, so the history of the schema is readable rather than implied by
whatever the code currently expects.

Renames are the common case and are safe to apply on read: the data is
unchanged, only its label moved. Anything that changes *values* belongs in a
migration function instead, and is applied once and written back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Migration:
    """One change to a table's stored shape."""

    table: str
    version: int
    reason: str
    # Old column name -> new column name. Applied on read, since the values are
    # untouched and only the label moved.
    renames: dict[str, str] = field(default_factory=dict)
    # For changes that transform values rather than names. Applied on read and
    # written back, so it runs once.
    transform: Callable[[pl.DataFrame], pl.DataFrame] | None = None


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        table="earnings_events",
        version=2,
        reason=(
            "Renamed `surprise_pct` to `surprise` and changed the stored units from "
            "percentage points to a fraction, so it matches every other ratio in the "
            "schema. The old name reported a 5% earnings beat as 512% once the display "
            "layer applied its own percentage conversion."
        ),
        renames={"surprise_pct": "surprise"},
        transform=lambda df: _percentage_points_to_fraction(df, "surprise"),
    ),
)


def _percentage_points_to_fraction(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Convert a column from percentage points to a fraction, idempotently.

    The unit is inferred from the data rather than assumed, so running this
    twice is harmless: earnings surprises cluster around a few percent, so a
    median absolute value above one means the column is still in percentage
    points, and below it means the conversion has already happened. A migration
    that corrupts data when re-run is worse than no migration.
    """
    values = df[column].drop_nulls()
    if values.len() == 0:
        return df
    if float(values.abs().median()) <= 1.0:
        return df  # already a fraction
    return df.with_columns(
        (pl.col(column) / 100.0).clip(-10.0, 10.0).alias(column)
    )


def pending(table: str, columns: set[str]) -> list[Migration]:
    """Migrations that apply to a stored table with these columns."""
    return [
        m
        for m in MIGRATIONS
        if m.table == table and any(old in columns for old in m.renames)
    ]


def apply(df: pl.DataFrame, table: str) -> tuple[pl.DataFrame, list[Migration]]:
    """Bring a stored frame up to the current schema.

    Returns the migrated frame and the migrations that ran, so the caller can
    write the result back and stop paying the cost on every read.
    """
    applied: list[Migration] = []
    out = df
    for migration in pending(table, set(out.columns)):
        renames = {old: new for old, new in migration.renames.items() if old in out.columns}
        if renames:
            out = out.rename(renames)
        if migration.transform is not None:
            out = migration.transform(out)
        applied.append(migration)
        log.info(
            "migrated %s to version %d: %s", table, migration.version, migration.reason
        )
    return out, applied
