"""Tests for point-in-time fundamental assembly.

These cover the failure modes that actually bite: year-to-date cash-flow tags
being triple-counted, a trailing-twelve-month figure silently built from three
quarters, a restatement being ignored, and a value becoming visible before it
was filed.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from spaid.pipeline.fundamentals import (
    AtomicPeriod,
    Fact,
    assemble_ttm,
    decompose,
    instant_series,
    ttm_series,
)


def d(s: str) -> date:
    return date.fromisoformat(s)


def flow(start: str, end: str, filed: str, value: float, *, rank: int = 0, accn: str = "a") -> Fact:
    """A duration fact. `available_at` is the session after filing."""
    filed_d = d(filed)
    return Fact(
        start=d(start),
        end=d(end),
        filed=filed_d,
        available_at=filed_d + timedelta(days=1),
        value=value,
        accession=accn,
        tag_rank=rank,
    )


def instant(end: str, filed: str, value: float, *, rank: int = 0, accn: str = "a") -> Fact:
    filed_d = d(filed)
    return Fact(
        start=None,
        end=d(end),
        filed=filed_d,
        available_at=filed_d + timedelta(days=1),
        value=value,
        accession=accn,
        tag_rank=rank,
    )


def known_from(facts: list[Fact]) -> dict:
    return {(f.start, f.end): f for f in facts}


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


class TestDecompose:
    def test_discrete_quarters_pass_through(self):
        """Income-statement style: four separate quarters, no differencing."""
        facts = [
            flow("2025-01-01", "2025-03-31", "2025-04-25", 100.0),
            flow("2025-04-01", "2025-06-30", "2025-07-25", 110.0),
            flow("2025-07-01", "2025-09-30", "2025-10-24", 120.0),
            flow("2025-10-01", "2025-12-31", "2026-02-10", 130.0),
        ]
        atomics = decompose(known_from(facts))
        assert len(atomics) == 4
        assert sorted(a.value for a in atomics) == [100.0, 110.0, 120.0, 130.0]

    def test_year_to_date_is_differenced(self):
        """Cash-flow style: cumulative windows sharing a start must be differenced.

        Reading 3mo/6mo/9mo/12mo cumulative facts as four quarters would report
        2.5x the true annual figure.
        """
        facts = [
            flow("2025-01-01", "2025-03-31", "2025-04-25", 100.0),
            flow("2025-01-01", "2025-06-30", "2025-07-25", 210.0),
            flow("2025-01-01", "2025-09-30", "2025-10-24", 330.0),
            flow("2025-01-01", "2025-12-31", "2026-02-10", 460.0),
        ]
        atomics = decompose(known_from(facts))
        values = sorted(a.value for a in atomics)
        assert values == pytest.approx([100.0, 110.0, 120.0, 130.0])
        # The four slices must tile the year without overlap.
        assert sum(values) == pytest.approx(460.0)

    def test_differenced_slice_needs_both_filings(self):
        """An incremental quarter is only knowable once both cuts are public."""
        facts = [
            flow("2025-01-01", "2025-06-30", "2025-07-25", 210.0),
            flow("2025-01-01", "2025-09-30", "2025-10-24", 330.0),
        ]
        atomics = decompose(known_from(facts))
        q3 = next(a for a in atomics if a.start == d("2025-06-30"))
        assert q3.value == pytest.approx(120.0)
        assert q3.available_at == d("2025-10-25")  # the later of the two filings

    def test_implausible_durations_rejected(self):
        facts = [
            flow("2025-01-01", "2025-01-05", "2025-02-01", 5.0),  # 4 days
            flow("2020-01-01", "2025-01-01", "2025-02-01", 999.0),  # 5 years
        ]
        assert decompose(known_from(facts)) == []


# ---------------------------------------------------------------------------
# TTM assembly
# ---------------------------------------------------------------------------


class TestAssembleTtm:
    def _quarters(self) -> list[AtomicPeriod]:
        return [
            AtomicPeriod(d("2025-01-01"), d("2025-03-31"), 100.0, ("a",), d("2025-04-26")),
            AtomicPeriod(d("2025-03-31"), d("2025-06-30"), 110.0, ("b",), d("2025-07-26")),
            AtomicPeriod(d("2025-06-30"), d("2025-09-30"), 120.0, ("c",), d("2025-10-25")),
            AtomicPeriod(d("2025-09-30"), d("2025-12-31"), 130.0, ("d",), d("2026-02-11")),
        ]

    def test_sums_four_quarters(self):
        result = assemble_ttm(self._quarters())
        assert result is not None
        assert result.value == pytest.approx(460.0)
        assert result.period_end == d("2025-12-31")
        assert result.n_sources == 4

    def test_availability_is_the_last_piece_to_arrive(self):
        result = assemble_ttm(self._quarters())
        assert result.available_at == d("2026-02-11")

    def test_returns_none_when_a_quarter_is_missing(self):
        """A broken chain must produce nothing, never a three-quarter total."""
        quarters = self._quarters()
        del quarters[1]  # lose Q2
        assert assemble_ttm(quarters) is None

    def test_exclusive_date_convention_still_chains(self):
        """Some filers start a quarter the day after the previous one ends."""
        quarters = [
            AtomicPeriod(d("2025-01-01"), d("2025-03-31"), 100.0, ("a",), d("2025-04-26")),
            AtomicPeriod(d("2025-04-01"), d("2025-06-30"), 110.0, ("b",), d("2025-07-26")),
            AtomicPeriod(d("2025-07-01"), d("2025-09-30"), 120.0, ("c",), d("2025-10-25")),
            AtomicPeriod(d("2025-10-01"), d("2025-12-31"), 130.0, ("d",), d("2026-02-11")),
        ]
        result = assemble_ttm(quarters)
        assert result is not None
        assert result.value == pytest.approx(460.0)

    def test_annual_only_filer(self):
        annual = [
            AtomicPeriod(d("2024-01-01"), d("2024-12-31"), 400.0, ("a",), d("2025-02-15")),
            AtomicPeriod(d("2025-01-01"), d("2025-12-31"), 460.0, ("b",), d("2026-02-15")),
        ]
        result = assemble_ttm(annual)
        assert result is not None
        assert result.value == pytest.approx(460.0)

    def test_prefers_quarter_over_cumulative_at_the_same_end(self):
        """A 10-Q reports both the quarter and the year-to-date; use the quarter."""
        periods = self._quarters() + [
            AtomicPeriod(d("2025-01-01"), d("2025-09-30"), 330.0, ("ytd",), d("2025-10-25")),
        ]
        result = assemble_ttm(periods, target_end=d("2025-12-31"))
        assert result is not None
        assert result.value == pytest.approx(460.0)

    def test_fifty_two_week_fiscal_year(self):
        """Retail-style 13-week quarters summing to 364 days are a valid year."""
        starts = [d("2025-01-05")]
        periods = []
        for i, v in enumerate([100.0, 110.0, 120.0, 130.0]):
            s = starts[-1]
            e = s + timedelta(days=91)
            periods.append(AtomicPeriod(s, e, v, (f"a{i}",), e + timedelta(days=30)))
            starts.append(e)
        result = assemble_ttm(periods)
        assert result is not None
        assert result.value == pytest.approx(460.0)


# ---------------------------------------------------------------------------
# Point-in-time series
# ---------------------------------------------------------------------------


class TestTtmSeries:
    def test_value_never_visible_before_it_was_filed(self):
        """The defining property: nothing is knowable before its last piece arrived."""
        facts = [
            flow("2025-01-01", "2025-03-31", "2025-04-25", 100.0),
            flow("2025-04-01", "2025-06-30", "2025-07-25", 110.0),
            flow("2025-07-01", "2025-09-30", "2025-10-24", 120.0),
            flow("2025-10-01", "2025-12-31", "2026-02-10", 130.0),
        ]
        series = ttm_series(facts)
        assert series, "expected at least one trailing-twelve-month observation"
        first = series[0]
        assert first.available_at >= d("2026-02-11")
        assert first.value == pytest.approx(460.0)

    def test_restatement_supersedes_the_original(self):
        """A refiled quarter replaces the original from the refiling date onward."""
        facts = [
            flow("2025-01-01", "2025-03-31", "2025-04-25", 100.0),
            flow("2025-04-01", "2025-06-30", "2025-07-25", 110.0),
            flow("2025-07-01", "2025-09-30", "2025-10-24", 120.0),
            flow("2025-10-01", "2025-12-31", "2026-02-10", 130.0),
            # Q1 restated downward in the 10-K.
            flow("2025-01-01", "2025-03-31", "2026-02-10", 90.0, accn="restated"),
        ]
        series = ttm_series(facts)
        assert series[-1].value == pytest.approx(450.0)

    def test_original_value_stands_before_the_restatement_was_filed(self):
        facts = [
            flow("2024-10-01", "2024-12-31", "2025-02-10", 90.0),
            flow("2025-01-01", "2025-03-31", "2025-04-25", 100.0),
            flow("2025-04-01", "2025-06-30", "2025-07-25", 110.0),
            flow("2025-07-01", "2025-09-30", "2025-10-24", 120.0),
            flow("2025-10-01", "2025-12-31", "2026-02-10", 130.0),
            flow("2025-01-01", "2025-03-31", "2026-02-10", 60.0, accn="restated"),
        ]
        series = ttm_series(facts)
        early = [p for p in series if p.available_at < d("2026-02-11")]
        assert early, "expected a trailing figure that predates the restatement"
        assert early[-1].value == pytest.approx(90.0 + 100.0 + 110.0 + 120.0)

    def test_more_specific_tag_wins_within_one_filing(self):
        facts = [
            flow("2025-01-01", "2025-03-31", "2025-04-25", 100.0, rank=0),
            flow("2025-01-01", "2025-03-31", "2025-04-25", 999.0, rank=3),
            flow("2025-04-01", "2025-06-30", "2025-07-25", 110.0),
            flow("2025-07-01", "2025-09-30", "2025-10-24", 120.0),
            flow("2025-10-01", "2025-12-31", "2026-02-10", 130.0),
        ]
        assert ttm_series(facts)[-1].value == pytest.approx(460.0)

    def test_emits_one_row_per_change_not_per_filing(self):
        facts = [
            flow("2025-01-01", "2025-03-31", "2025-04-25", 100.0),
            flow("2025-04-01", "2025-06-30", "2025-07-25", 110.0),
            flow("2025-07-01", "2025-09-30", "2025-10-24", 120.0),
            flow("2025-10-01", "2025-12-31", "2026-02-10", 130.0),
            # A later filing repeating the same facts adds no information.
            flow("2025-10-01", "2025-12-31", "2026-03-01", 130.0, accn="dup"),
        ]
        series = ttm_series(facts)
        assert len({(p.period_end, p.value) for p in series}) == len(series)

    def test_empty_input(self):
        assert ttm_series([]) == []


class TestInstantSeries:
    def test_uses_latest_known_period_not_largest_in_file(self):
        """A mis-dated future balance sheet must not shadow real later values.

        This is the Colgate failure: four 2010 filings all claiming a
        2010-12-31 period end, which froze the share count for a decade.
        """
        facts = [
            instant("2025-03-31", "2025-04-25", 500.0),
            instant("2025-06-30", "2025-07-25", 520.0),
            instant("2025-09-30", "2025-10-24", 540.0),
        ]
        series = instant_series(facts)
        assert [p.value for p in series] == [500.0, 520.0, 540.0]
        assert series[-1].period_end == d("2025-09-30")

    def test_restated_balance_sheet_supersedes(self):
        facts = [
            instant("2025-03-31", "2025-04-25", 500.0),
            instant("2025-03-31", "2025-10-24", 480.0, accn="restated"),
        ]
        series = instant_series(facts)
        assert series[0].value == 500.0
        assert series[-1].value == 480.0

    def test_ignores_duration_facts(self):
        facts = [
            instant("2025-03-31", "2025-04-25", 500.0),
            flow("2025-01-01", "2025-03-31", "2025-04-25", 100.0),
        ]
        series = instant_series(facts)
        assert len(series) == 1
        assert series[0].value == 500.0

    def test_availability_follows_the_filing(self):
        facts = [instant("2025-03-31", "2025-04-25", 500.0)]
        assert instant_series(facts)[0].available_at == d("2025-04-26")


class TestLatestLevel:
    """Quantities reported over a window that must not be summed or differenced."""

    def test_takes_the_most_recent_quarter_not_the_sum(self):
        """Four quarters of a share count describe the same shares four times."""
        from spaid.pipeline.fundamentals import latest_level

        facts = [
            flow("2025-01-01", "2025-03-31", "2025-04-25", 7_500_000_000.0),
            flow("2025-04-01", "2025-06-30", "2025-07-25", 7_450_000_000.0),
            flow("2025-07-01", "2025-09-30", "2025-10-24", 7_420_000_000.0),
        ]
        result = latest_level(known_from(facts))
        assert result is not None
        assert result.value == pytest.approx(7_420_000_000.0)

    def test_prefers_the_quarter_over_the_year_to_date_window(self):
        """A 10-Q reports both a quarterly and a nine-month average share count."""
        from spaid.pipeline.fundamentals import latest_level

        facts = [
            flow("2025-07-01", "2025-09-30", "2025-10-24", 7_420_000_000.0),
            flow("2025-01-01", "2025-09-30", "2025-10-24", 7_460_000_000.0),
        ]
        result = latest_level(known_from(facts))
        assert result.value == pytest.approx(7_420_000_000.0)

    def test_never_differences_cumulative_windows(self):
        """Differencing non-additive windows produced negative share counts."""
        from spaid.pipeline.fundamentals import latest_level

        facts = [
            flow("2025-01-01", "2025-03-31", "2025-04-25", 7_500_000_000.0),
            flow("2025-01-01", "2025-06-30", "2025-07-25", 7_480_000_000.0),
            flow("2025-01-01", "2025-09-30", "2025-10-24", 7_460_000_000.0),
        ]
        result = latest_level(known_from(facts))
        assert result.value > 0
        assert result.value == pytest.approx(7_460_000_000.0)

    def test_rejects_non_positive_values(self):
        from spaid.pipeline.fundamentals import latest_level

        facts = [flow("2025-07-01", "2025-09-30", "2025-10-24", -4_000_000.0)]
        assert latest_level(known_from(facts)) is None

    def test_series_is_point_in_time(self):
        from spaid.providers.sec.concepts import Aggregation

        facts = [
            flow("2025-01-01", "2025-03-31", "2025-04-25", 7_500_000_000.0),
            flow("2025-04-01", "2025-06-30", "2025-07-25", 7_450_000_000.0),
        ]
        series = ttm_series(facts, Aggregation.LAST)
        assert [p.value for p in series] == [7_500_000_000.0, 7_450_000_000.0]
        assert series[0].available_at == d("2025-04-26")


class TestScaleOutliers:
    """Filers occasionally submit a value with the wrong scale.

    FedEx's fiscal-2026 net income was filed correctly as $4.433bn and again, by
    a different filing agent, as $4,433. Because a later filing normally
    supersedes an earlier one, the broken value won and the company appeared to
    have earned four thousand dollars. Waters' diluted share count arrived as 98
    billion against a real figure near 60 million.
    """

    def _series(self, values: list[float]) -> list[Fact]:
        out = []
        for i, v in enumerate(values):
            year, month = 2020 + i // 4, 1 + 3 * (i % 4)
            start = date(year, month, 1)
            end = date(year, month + 2, 28)
            out.append(
                Fact(
                    start=start,
                    end=end,
                    filed=end + timedelta(days=30),
                    available_at=end + timedelta(days=31),
                    value=v,
                    accession=f"a{i}",
                    tag_rank=0,
                )
            )
        return out

    def test_a_thousandfold_scale_error_is_rejected(self):
        from spaid.pipeline.fundamentals import reject_scale_outliers

        values = [1.0e9] * 8 + [1.0e6]  # the last is a thousand times too small
        kept, dropped = reject_scale_outliers(self._series(values))
        assert len(dropped) == 1
        assert dropped[0].value == pytest.approx(1.0e6)
        assert len(kept) == 8

    def test_an_absurdly_large_value_is_rejected(self):
        from spaid.pipeline.fundamentals import reject_scale_outliers

        values = [6.0e7] * 8 + [9.8e10]  # Waters' share count
        kept, dropped = reject_scale_outliers(self._series(values))
        assert len(dropped) == 1
        assert dropped[0].value == pytest.approx(9.8e10)
        assert all(f.value == pytest.approx(6.0e7) for f in kept)

    def test_ordinary_growth_is_never_rejected(self):
        """A company doubling over five years must survive the guard."""
        from spaid.pipeline.fundamentals import reject_scale_outliers

        values = [1.0e9 * (1.15**i) for i in range(12)]
        kept, dropped = reject_scale_outliers(self._series(values))
        assert dropped == []
        assert len(kept) == 12

    def test_a_genuine_zero_survives(self):
        """No buybacks this quarter is information, not an error."""
        from spaid.pipeline.fundamentals import reject_scale_outliers

        values = [5.0e8] * 8 + [0.0]
        kept, dropped = reject_scale_outliers(self._series(values))
        assert dropped == []
        assert any(f.value == 0.0 for f in kept)

    def test_a_loss_is_not_treated_as_a_scale_error(self):
        from spaid.pipeline.fundamentals import reject_scale_outliers

        values = [1.0e9] * 8 + [-9.0e8]
        _, dropped = reject_scale_outliers(self._series(values))
        assert dropped == []

    def test_thin_series_are_left_alone(self):
        """With three observations there is nothing to compare against."""
        from spaid.pipeline.fundamentals import reject_scale_outliers

        kept, dropped = reject_scale_outliers(self._series([1.0e9, 1.0e6, 2.0e9]))
        assert dropped == []
        assert len(kept) == 3

    def test_quarters_and_years_are_compared_separately(self):
        """An annual figure is legitimately four times a quarterly one."""
        from spaid.pipeline.fundamentals import reject_scale_outliers

        quarters = self._series([1.0e9] * 8)
        annuals = [
            Fact(
                start=date(2020 + i, 1, 1),
                end=date(2020 + i, 12, 31),
                filed=date(2021 + i, 2, 15),
                available_at=date(2021 + i, 2, 16),
                value=4.0e9,
                accession=f"y{i}",
                tag_rank=0,
            )
            for i in range(6)
        ]
        kept, dropped = reject_scale_outliers(quarters + annuals)
        assert dropped == []
        assert len(kept) == 14
