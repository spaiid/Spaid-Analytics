"""Persistence: canonical table schemas and the parquet store."""

from spaid.storage import store
from spaid.storage.schema import (
    ALL_TABLES,
    SCHEMA_VERSION,
    SchemaError,
    TableSpec,
    coerce,
    validate,
)

__all__ = [
    "ALL_TABLES",
    "SCHEMA_VERSION",
    "SchemaError",
    "TableSpec",
    "coerce",
    "store",
    "validate",
]
