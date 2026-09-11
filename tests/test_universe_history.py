"""Reconstructing index membership from periodic observations.

The span builder turns a sequence of "who was in the list that month" snapshots
into entry and exit dates. Everything that can go wrong here is quiet: a
company that left and rejoined merged into one continuous spell, an exit date
taken from the wrong side of the bracket, a dual-class filer occupying two
slots. None of those shows up as an error; they show up as a backtest that
holds things it could not have held.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from spaid.pipeline.security_master import (
    _classify_coverage,
    _collapse_share_classes,
    security_id_for,
)
from spaid.providers.wikipedia.history import build_spans, month_ends, spans_frame


def snapshot(rows: list[tuple[str, int]], as_of: date) -> pl.DataFrame:
    """A parsed constituents table for one month."""
    return pl.DataFrame(
        [
            {
                "ticker": ticker,
                "name": f"{ticker} Inc",
                "sector": "Test",
                "industry": "Test",
                "cik": cik,
                "date_added": None,
                "snapshot_date": as_of,
                "revid": 1,
            }
            for ticker, cik in rows
        ],
        schema={
            "ticker": pl.Utf8, "name": pl.Utf8, "sector": pl.Utf8, "industry": pl.Utf8,
            "cik": pl.Int64, "date_added": pl.Date, "snapshot_date": pl.Date, "revid": pl.Int64,
        },
    )


MONTHS = [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31), date(2020, 4, 30),
          date(2020, 5, 31), date(2020, 6, 30)]


class TestMonthEnds:
    def test_the_last_calendar_day_of_each_month_is_used(self):
        out = month_ends(date(2020, 1, 1), date(2020, 4, 15))
        assert out == [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)]

    def test_a_partial_final_month_is_excluded_rather_than_truncated(self):
        out = month_ends(date(2020, 1, 1), date(2020, 3, 30))
        assert date(2020, 3, 31) not in out


class TestSpans:
    def test_a_company_present_throughout_is_one_open_spell(self):
        snaps = {m: snapshot([("AAA", 1)], m) for m in MONTHS}
        spans = build_spans(snaps)
        assert len(spans) == 1
        assert spans[0].still_member
        assert spans[0].entry_observed_by == MONTHS[0]
        assert spans[0].exit_observed_by is None

    def test_an_exit_is_bracketed_by_the_two_snapshots_around_it(self):
        snaps = {m: snapshot([("AAA", 1)] if i < 3 else [("BBB", 2)], m)
                 for i, m in enumerate(MONTHS)}
        spans = {s.ticker: s for s in build_spans(snaps)}
        gone = spans["AAA"]
        assert not gone.still_member
        # Last seen present in March, first seen absent in April.
        assert gone.exit_observed_after == MONTHS[2]
        assert gone.exit_observed_by == MONTHS[3]

    def test_leaving_and_rejoining_produces_two_spells_not_one(self):
        """EQT did exactly this. Merging the runs would let a backtest hold it
        through a period when it was not in the index at all."""
        present = [0, 1, 4, 5]
        snaps = {
            m: snapshot([("AAA", 1)] if i in present else [("BBB", 2)], m)
            for i, m in enumerate(MONTHS)
        }
        spans = [s for s in build_spans(snaps) if s.ticker == "AAA"]
        assert len(spans) == 2
        first, second = sorted(spans, key=lambda s: s.entry_observed_by)
        assert first.entry_observed_by == MONTHS[0]
        assert first.exit_observed_by == MONTHS[2]
        assert second.entry_observed_by == MONTHS[4]
        assert second.still_member

    def test_a_company_added_midway_has_no_earlier_bracket_edge(self):
        snaps = {
            m: snapshot([("AAA", 1)] + ([("CCC", 3)] if i >= 2 else []), m)
            for i, m in enumerate(MONTHS)
        }
        spans = {s.ticker: s for s in build_spans(snaps)}
        joined = spans["CCC"]
        assert joined.entry_observed_after == MONTHS[1]
        assert joined.entry_observed_by == MONTHS[2]

    def test_a_stated_date_added_inside_the_bracket_makes_the_entry_exact(self):
        snaps = {}
        for i, m in enumerate(MONTHS):
            rows = [("AAA", 1)]
            if i >= 2:
                rows.append(("CCC", 3))
            frame = snapshot(rows, m)
            frame = frame.with_columns(
                pl.when(pl.col("ticker") == "CCC")
                .then(pl.lit(date(2020, 3, 12)))
                .otherwise(None)
                .alias("date_added")
            )
            snaps[m] = frame
        spans = {s.ticker: s for s in build_spans(snaps)}
        assert spans["CCC"].entry_date == date(2020, 3, 12)

    def test_a_stated_date_outside_the_bracket_is_not_trusted(self):
        """The snapshots saw it arrive in March; a claim that it joined in 2015
        contradicts the observation and is discarded rather than believed."""
        snaps = {}
        for i, m in enumerate(MONTHS):
            rows = [("AAA", 1)] + ([("CCC", 3)] if i >= 2 else [])
            frame = snapshot(rows, m).with_columns(
                pl.when(pl.col("ticker") == "CCC")
                .then(pl.lit(date(2015, 6, 1)))
                .otherwise(None)
                .alias("date_added")
            )
            snaps[m] = frame
        spans = {s.ticker: s for s in build_spans(snaps)}
        assert spans["CCC"].entry_date is None

    def test_the_frame_carries_every_span(self):
        snaps = {m: snapshot([("AAA", 1), ("BBB", 2)], m) for m in MONTHS}
        frame = spans_frame(build_spans(snaps))
        assert frame.height == 2
        assert set(frame["ticker"]) == {"AAA", "BBB"}

    def test_an_empty_history_produces_an_empty_frame_with_the_right_shape(self):
        frame = spans_frame([])
        assert frame.height == 0
        assert "entry_observed_by" in frame.columns


class TestIdentity:
    def test_the_central_index_key_is_the_identifier_when_it_exists(self):
        assert security_id_for(320193, "AAPL", date(2020, 1, 1)) == "CIK0000320193"

    def test_a_missing_key_falls_back_to_a_dated_ticker(self):
        """Dating the fallback matters: a ticker reassigned years later must
        become a different security, not a continuation of the old one."""
        first = security_id_for(None, "XYZ", date(2015, 1, 1))
        second = security_id_for(None, "XYZ", date(2022, 1, 1))
        assert first != second
        assert first.startswith("TKRXYZ")


class TestShareClasses:
    def test_a_dual_class_filer_occupies_one_membership_row(self):
        membership = pl.DataFrame(
            [
                {"index_name": "sp500", "security_id": "CIK1", "ticker": "GOOG",
                 "entry_observed_by": date(2020, 1, 31), "price_coverage": "full",
                 "n_snapshots": 12},
                {"index_name": "sp500", "security_id": "CIK1", "ticker": "GOOGL",
                 "entry_observed_by": date(2020, 1, 31), "price_coverage": "full",
                 "n_snapshots": 12},
            ]
        )
        primary, secondary = _collapse_share_classes(membership)
        assert primary.height == 1
        assert secondary.height == 1
        # Deterministic: the same input must always choose the same class.
        again, _ = _collapse_share_classes(membership.reverse())
        assert primary["ticker"][0] == again["ticker"][0]

    def test_the_priced_class_is_preferred_over_an_unpriced_one(self):
        membership = pl.DataFrame(
            [
                {"index_name": "sp500", "security_id": "CIK1", "ticker": "AAA",
                 "entry_observed_by": date(2020, 1, 31), "price_coverage": "none",
                 "n_snapshots": 12},
                {"index_name": "sp500", "security_id": "CIK1", "ticker": "ZZZ",
                 "entry_observed_by": date(2020, 1, 31), "price_coverage": "full",
                 "n_snapshots": 12},
            ]
        )
        primary, _ = _collapse_share_classes(membership)
        assert primary["ticker"][0] == "ZZZ"


class TestPriceCoverage:
    DATA_START = date(2013, 1, 2)
    DATA_END = date(2026, 9, 10)

    def test_no_prices_is_none_not_partial(self):
        assert _classify_coverage(None, None, date(2016, 1, 1), None,
                                  self.DATA_START, self.DATA_END) == "none"

    def test_a_spell_fully_inside_the_priced_window_is_full(self):
        assert _classify_coverage(
            date(2013, 1, 2), date(2026, 9, 10), date(2016, 1, 1), date(2020, 1, 1),
            self.DATA_START, self.DATA_END,
        ) == "full"

    def test_a_membership_that_predates_our_price_history_is_still_full(self):
        """Abbott joined the index in 1957. The question is whether we can price
        the part of that membership our data covers, not the whole of it."""
        assert _classify_coverage(
            date(2013, 1, 2), date(2026, 9, 10), date(1957, 3, 4), None,
            self.DATA_START, self.DATA_END,
        ) == "full"

    def test_a_series_that_stops_before_the_spell_does_is_partial(self):
        assert _classify_coverage(
            date(2013, 1, 2), date(2018, 6, 1), date(2016, 1, 1), None,
            self.DATA_START, self.DATA_END,
        ) == "partial"

    def test_a_spell_that_ended_before_our_data_began_has_nothing_to_cover(self):
        assert _classify_coverage(
            date(2013, 1, 2), date(2026, 9, 10), date(2009, 1, 1), date(2011, 1, 1),
            self.DATA_START, self.DATA_END,
        ) == "none"


class TestStoredUniverse:
    """Properties of the reconstruction actually on disk."""

    @pytest.fixture(scope="class")
    def membership(self) -> pl.DataFrame:
        from spaid.storage import store

        df = store.read("universe_membership")
        if df is None or df.is_empty():
            pytest.skip("no membership in the store")
        return df

    def test_exits_are_recorded_so_the_universe_is_not_survivor_only(self, membership):
        historical = membership.filter(pl.col("universe_version") != "current-list")
        if historical.is_empty():
            pytest.skip("only the current list is stored")
        assert historical.filter(pl.col("exit_date").is_not_null()).height > 0

    def test_no_company_holds_two_membership_rows_on_the_same_entry(self, membership):
        duplicated = (
            membership.group_by(["index_name", "security_id", "entry_observed_by"])
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
        )
        assert duplicated.height == 0

    def test_every_exit_is_after_its_entry(self, membership):
        backwards = membership.filter(
            pl.col("exit_date").is_not_null() & (pl.col("exit_date") <= pl.col("entry_date"))
        )
        assert backwards.height == 0

    def test_the_precision_of_each_edge_is_recorded(self, membership):
        historical = membership.filter(pl.col("universe_version") != "current-list")
        if historical.is_empty():
            pytest.skip("only the current list is stored")
        assert historical["entry_precision"].null_count() == 0
        assert historical["exit_precision"].null_count() == 0
        assert set(historical["exit_precision"].unique()) <= {
            "exact", "bracketed", "unknown", "still_member"
        }
