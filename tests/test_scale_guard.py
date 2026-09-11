"""Tests for the scale-error guard on incoming filings.

The guard exists because filers submit values with the wrong scale: FedEx's
net income arrived once as $4.433bn and again as $4,433, and Waters' diluted
share count as 98 billion against a real 60 million. A later filing normally
supersedes an earlier one, so without the guard the broken value wins.

It has to make a distinction that is not obvious from a single number: a value
far from its history is either a scale error or a company that changed size.
AMD's amortisation of intangibles went from $4m a quarter to $550m when it
acquired Xilinx, and judging that against the median of its whole history
rejected every real figure for four years.
"""

from __future__ import annotations

from datetime import date, timedelta

from spaid.pipeline.fundamentals import Fact, reject_scale_outliers


def quarter(i: int, value: float) -> Fact:
    """One quarterly fact, `i` quarters after the start of 2019."""
    end = date(2019, 3, 31) + timedelta(days=91 * i)
    return Fact(
        end=end,
        filed=end + timedelta(days=30),
        available_at=end + timedelta(days=30),
        value=value,
        accession=f"acc-{i}",
        tag_rank=0,
        start=end - timedelta(days=90),
    )


def values(facts: list[Fact]) -> list[float]:
    return sorted(f.value for f in facts)


class TestScaleErrors:
    def test_a_thousandfold_error_among_good_neighbours_is_rejected(self):
        """The FedEx case: one filing off by a factor of a million."""
        facts = [quarter(i, 4.4e9) for i in range(12)]
        facts[6] = quarter(6, 4_433.0)
        kept, rejected = reject_scale_outliers(facts)
        assert values(rejected) == [4_433.0]
        assert len(kept) == 11

    def test_an_absurdly_large_value_is_rejected(self):
        """The Waters case: a share count three orders of magnitude too big."""
        facts = [quarter(i, 60_000_000.0) for i in range(12)]
        facts[3] = quarter(3, 98_000_000_000.0)
        kept, rejected = reject_scale_outliers(facts)
        assert values(rejected) == [98_000_000_000.0]
        assert len(kept) == 11

    def test_sparse_history_is_left_alone(self):
        """With nothing to compare against, the guard must not guess."""
        facts = [quarter(i, 100.0) for i in range(3)] + [quarter(3, 1e9)]
        kept, rejected = reject_scale_outliers(facts)
        assert rejected == []
        assert len(kept) == 4

    def test_a_genuine_zero_survives(self):
        """No buyback this quarter is a fact, not an outlier."""
        facts = [quarter(i, 5e8) for i in range(12)]
        facts[5] = quarter(5, 0.0)
        kept, rejected = reject_scale_outliers(facts)
        assert rejected == []
        assert 0.0 in values(kept)


class TestStepChanges:
    def test_an_acquisition_step_change_is_preserved(self):
        """The AMD case: $4m a quarter becomes $550m and stays there."""
        before = [quarter(i, 4e6) for i in range(10)]
        after = [quarter(i, 5.5e8) for i in range(10, 22)]
        kept, rejected = reject_scale_outliers(before + after)
        assert rejected == []
        assert len(kept) == 22

    def test_the_step_survives_at_either_end_of_the_history(self):
        """A fact near the boundary must still see a full set of neighbours.

        This is where a naive centred window fails: the first post-acquisition
        quarter has only pre-acquisition neighbours behind it.
        """
        before = [quarter(i, 4e6) for i in range(10)]
        after = [quarter(i, 5.5e8) for i in range(10, 20)]
        _, rejected = reject_scale_outliers(before + after)
        assert rejected == []

    def test_a_lone_spike_inside_a_step_change_is_still_caught(self):
        """Regime change must not become a licence for anything."""
        before = [quarter(i, 4e6) for i in range(10)]
        after = [quarter(i, 5.5e8) for i in range(10, 22)]
        after[6] = quarter(16, 5.5e8 * 1_000)
        kept, rejected = reject_scale_outliers(before + after)
        assert values(rejected) == [5.5e11]
        assert len(kept) == 21

    def test_a_gradual_ramp_is_never_an_outlier(self):
        """Compounding growth, however fast, moves with its neighbours."""
        facts = [quarter(i, 1e6 * (1.35**i)) for i in range(24)]
        _, rejected = reject_scale_outliers(facts)
        assert rejected == []


class TestPeriodBuckets:
    def test_annual_and_quarterly_figures_are_judged_separately(self):
        """A year is about four quarters; that is not an error."""
        quarters = [quarter(i, 1e8) for i in range(8)]
        annuals = [
            Fact(
                end=date(2019, 12, 31) + timedelta(days=365 * i),
                filed=date(2020, 2, 1) + timedelta(days=365 * i),
                available_at=date(2020, 2, 1) + timedelta(days=365 * i),
                value=4e8,
                accession=f"fy-{i}",
                tag_rank=0,
                start=date(2019, 1, 1) + timedelta(days=365 * i),
            )
            for i in range(8)
        ]
        _, rejected = reject_scale_outliers(quarters + annuals)
        assert rejected == []
