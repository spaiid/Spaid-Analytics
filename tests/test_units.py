"""Unit-consistency checks across the metric layer.

Unit confusion is the quietest class of bug in a financial application: the
number renders, the chart draws, nothing throws, and the value is wrong by a
factor of a hundred. The earnings-surprise metric arrived from the provider in
percentage points while every other ratio in the system is a fraction, so a 5%
average beat displayed as 512% and ranked the company against peers on a scale
nothing else used.

So the declared unit of every metric is checked against the distribution of its
actual values. These run against the real store and skip when it is empty.
"""

from __future__ import annotations

import polars as pl
import pytest

from spaid.api.service import UNIT_BY_METRIC
from spaid.config.scoring import ACTIVE_SCORING_SPEC
from spaid.storage import store

# A metric expressed as a fraction should not routinely exceed this magnitude.
# Growth rates genuinely can: a company doubling revenue is +1.0, and a recovery
# from a small base can be several times that. Ten is far beyond any of those
# while still catching a value that is a hundred times too large.
FRACTION_SANITY_LIMIT = 10.0

# Metrics whose fractional value legitimately runs large, and why.
WIDE_FRACTION_METRICS = {
    "eps_growth_1y",  # recovery from a near-zero base
    "eps_growth_3y",
    "fcf_growth_3y",
    "revenue_growth_1y",
    "growth_acceleration",
    "earnings_surprise",  # a beat against an estimate near zero
    "forward_eps_growth",
}


@pytest.fixture(scope="module")
def metric_scores() -> pl.DataFrame:
    df = store.read("metric_scores")
    if df is None or df.is_empty():
        pytest.skip("no metric scores in the store")
    return df.filter(pl.col("status") == "scored")


class TestDeclaredUnits:
    def test_every_scored_metric_declares_a_unit(self):
        """A metric the interface must render needs to say how."""
        undeclared = [m.key for m in ACTIVE_SCORING_SPEC.metrics if m.key not in UNIT_BY_METRIC]
        assert not undeclared, f"metrics with no declared display unit: {undeclared}"

    def test_percentage_metrics_hold_fractions_not_percentage_points(self, metric_scores):
        """0.05 is a 5% margin. 5.0 is a bug the interface will render as 500%."""
        offenders: list[str] = []
        for metric, unit in UNIT_BY_METRIC.items():
            if unit != "percent" or metric in WIDE_FRACTION_METRICS:
                continue
            values = metric_scores.filter(pl.col("metric") == metric)["raw_value"].drop_nulls()
            if values.len() < 50:
                continue
            median_magnitude = float(values.abs().median())
            if median_magnitude > 1.5:
                offenders.append(
                    f"{metric}: median magnitude {median_magnitude:.2f}, which reads as "
                    f"{median_magnitude * 100:.0f}% and is almost certainly percentage points"
                )
        assert not offenders, "metrics declared as percentages but holding percentage points:\n" + "\n".join(offenders)

    def test_wide_fraction_metrics_are_still_bounded(self, metric_scores):
        """Even a growth rate from a small base has a plausible ceiling."""
        for metric in WIDE_FRACTION_METRICS:
            values = metric_scores.filter(pl.col("metric") == metric)["raw_value"].drop_nulls()
            if values.len() < 50:
                continue
            median_magnitude = float(values.abs().median())
            assert median_magnitude < FRACTION_SANITY_LIMIT, (
                f"{metric} has a median magnitude of {median_magnitude:.1f}, which is not a "
                "fraction on any reading"
            )

    def test_multiples_are_positive_and_plausible(self, metric_scores):
        """A valuation multiple is a small positive number, not a fraction."""
        for metric, unit in UNIT_BY_METRIC.items():
            if unit != "times":
                continue
            values = metric_scores.filter(pl.col("metric") == metric)["raw_value"].drop_nulls()
            if values.len() < 50:
                continue
            median = float(values.median())
            assert 0.1 < median < 200.0, (
                f"{metric} has a median of {median:.2f}, implausible for a multiple"
            )

    def test_scores_are_always_zero_to_one_hundred(self, metric_scores):
        scores = metric_scores["score"].drop_nulls()
        assert scores.len() > 0
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 100.0

    def test_percentiles_are_always_zero_to_one(self, metric_scores):
        pct = metric_scores["percentile"].drop_nulls()
        assert float(pct.min()) >= 0.0
        assert float(pct.max()) <= 1.0


class TestDerivedTableUnits:
    def test_upside_is_a_fraction(self):
        valuations = store.read("valuations")
        if valuations is None or valuations.is_empty():
            pytest.skip("no valuations")
        upside = valuations["upside"].drop_nulls()
        assert float(upside.abs().median()) < 5.0, "upside looks like percentage points"

    def test_confidence_is_zero_to_one(self):
        for table, column in (
            ("valuations", "confidence"),
            ("opportunity_scores", "confidence"),
        ):
            df = store.read(table)
            if df is None or df.is_empty():
                continue
            values = df[column].drop_nulls()
            if values.len() == 0:
                continue
            assert float(values.min()) >= 0.0
            assert float(values.max()) <= 1.0, f"{table}.{column} exceeds 1.0"

    def test_coverage_is_zero_to_one(self):
        for table in ("opportunity_scores", "category_scores"):
            df = store.read(table)
            if df is None or df.is_empty():
                continue
            values = df["coverage"].drop_nulls()
            assert float(values.min()) >= 0.0
            assert float(values.max()) <= 1.0001

    def test_risk_and_trap_scores_are_zero_to_one_hundred(self):
        for table, column in (("risk", "risk_score"), ("value_trap", "trap_score")):
            df = store.read(table)
            if df is None or df.is_empty():
                continue
            values = df[column].drop_nulls()
            if values.len() == 0:
                continue
            assert float(values.min()) >= 0.0
            assert float(values.max()) <= 100.0


class TestSchemaMigration:
    """A renamed column must not make a refetchable table permanently unreadable.

    Renaming `surprise_pct` to `surprise` left the stored table unmergeable, and
    the merge path had no route forward: it read the old table, failed schema
    validation, and the refetch could never land.
    """

    def test_stored_tables_match_their_declared_schema(self):
        from spaid.storage import store
        from spaid.storage.schema import ALL_TABLES

        stale = [
            name
            for name in ALL_TABLES
            if store.exists(name) and not store.is_current_schema(name)
        ]
        assert not stale, (
            f"these stored tables predate the current schema and will fail to read: {stale}. "
            "Rebuild them with `spaid run`."
        )
