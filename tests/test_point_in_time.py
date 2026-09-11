"""End-to-end point-in-time integrity checks against the real store.

The unit tests prove the algorithms are correct in isolation. These prove the
property survives assembly, which is where it usually dies: one join on the
wrong date column silently hands the model weeks of foresight, and every number
downstream still looks plausible.

They skip when the store is empty, so a fresh clone does not show red.
"""

from __future__ import annotations

import polars as pl
import pytest

from spaid.storage import store


@pytest.fixture(scope="module")
def observations() -> pl.DataFrame:
    df = store.read("observations")
    if df is None or df.is_empty():
        pytest.skip("no observations in the store")
    return df


@pytest.fixture(scope="module")
def fundamentals() -> pl.DataFrame:
    df = store.read("fundamentals")
    if df is None or df.is_empty():
        pytest.skip("no fundamentals in the store")
    return df


@pytest.fixture(scope="module")
def prices() -> pl.DataFrame:
    df = store.read("prices")
    if df is None or df.is_empty():
        pytest.skip("no prices in the store")
    return df


class TestObservationDates:
    def test_nothing_is_reported_before_the_period_it_describes(self, observations):
        """A filing cannot report a quarter that has not ended.

        Rows violating this froze a mis-dated value in place for years, because
        the instant series takes the largest period end as the newest.
        """
        bad = observations.filter(pl.col("period_end") > pl.col("filed"))
        assert bad.height == 0, (
            f"{bad.height} observations claim to describe a period ending after they were filed"
        )

    def test_availability_never_precedes_filing(self, observations):
        bad = observations.filter(pl.col("available_at") < pl.col("filed"))
        assert bad.height == 0

    def test_availability_follows_filing_closely(self, observations, prices):
        """Availability is the next trading session, not an arbitrary delay.

        Scoped to filings inside the price history: anything filed before our
        first session is rolled forward to it, which is correct -- we could not
        have acted on it earlier than the first day we have prices for.
        """
        first_session = prices["date"].min()
        gap = observations.filter(pl.col("filed") >= first_session).select(
            (pl.col("available_at") - pl.col("filed")).dt.total_days().alias("days")
        )
        assert gap["days"].max() <= 6, "availability lags filing by more than a week"
        assert gap["days"].min() >= 0

    def test_pre_history_filings_roll_to_the_first_session(self, observations, prices):
        first_session = prices["date"].min()
        early = observations.filter(pl.col("filed") < first_session)
        if early.is_empty():
            pytest.skip("no filings predate the price history")
        assert early.filter(pl.col("available_at") < first_session).height == 0

    def test_every_observation_carries_its_provenance(self, observations):
        """The brief's data-integrity requirement, checked rather than assumed."""
        required = [
            "company_id", "cik", "tag", "concept", "value", "unit",
            "period_end", "filed", "available_at", "source", "collected_at",
        ]
        for col in required:
            assert col in observations.columns
            assert observations[col].null_count() == 0, f"{col} has nulls"

    def test_restatements_are_flagged_and_numbered(self, observations):
        assert observations["revision"].max() >= 1, "expected at least one restatement"
        restated = observations.filter(pl.col("is_restatement"))
        assert restated.height > 0
        # A restatement is by definition not the first disclosure of its period.
        assert restated.filter(pl.col("revision") == 0).height == 0

    def test_duration_facts_have_a_start_and_instants_do_not(self, observations):
        durations = observations.filter(pl.col("period_type") == "duration")
        instants = observations.filter(pl.col("period_type") == "instant")
        assert durations["period_start"].null_count() == 0
        assert instants["period_start"].null_count() == instants.height


class TestFundamentalAvailability:
    def test_a_value_is_never_knowable_before_its_period_ends(self, fundamentals):
        bad = fundamentals.filter(pl.col("available_at") < pl.col("period_end"))
        assert bad.height == 0, (
            f"{bad.height} fundamental values become available before the period they describe"
        )

    def test_trailing_values_lag_their_period_end_realistically(self, fundamentals):
        """A company files weeks after a quarter ends, not the same afternoon."""
        ttm = fundamentals.filter(pl.col("basis") == "ttm")
        if ttm.is_empty():
            pytest.skip("no trailing values")
        lag = ttm.select(
            (pl.col("available_at") - pl.col("period_end")).dt.total_days().alias("days")
        )
        assert lag["days"].median() >= 20, (
            "trailing figures become available suspiciously soon after the period ends"
        )

    def test_each_series_is_unique_per_availability_date(self, fundamentals):
        dupes = (
            fundamentals.group_by(["company_id", "concept", "basis", "available_at"])
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
        )
        assert dupes.height == 0

    def test_share_counts_are_positive(self, fundamentals):
        shares = fundamentals.filter(pl.col("concept").is_in(["shares_diluted", "shares_basic"]))
        if shares.is_empty():
            pytest.skip("no share counts")
        assert shares.filter(pl.col("value") <= 0).height == 0


class TestCrossSourceAgreement:
    """Two independent routes to the same number must agree.

    A single source cannot detect its own errors. Berkshire's market
    capitalisation read $477 million for years because one stale share count was
    never checked against anything.
    """

    def test_share_count_agrees_with_earnings_per_share(self, fundamentals):
        """Scoped to the companies the application actually scores.

        The fundamentals table now also covers former index members, fetched so
        that the backtest can rank the historical universe rather than today's
        survivors. Their most recent filing can be years old, so their share
        counts legitimately disagree with a stale earnings figure. Including
        them would turn a check on the live universe into a check on data no
        recommendation is ever built from.
        """
        current = store.read("securities")
        if current is None or current.is_empty():
            pytest.skip("no securities table")
        scored = set(current["company_id"].to_list())

        latest = (
            fundamentals.filter(
                pl.col("concept").is_in(["shares_diluted", "eps_diluted", "net_income"])
                & pl.col("company_id").is_in(list(scored))
            )
            .sort("available_at")
            .group_by(["company_id", "concept"])
            .agg(pl.col("value").last(), pl.col("ticker").last())
        )
        wide = latest.pivot(on="concept", index="company_id", values="value")
        for col in ("shares_diluted", "eps_diluted", "net_income"):
            if col not in wide.columns:
                pytest.skip(f"{col} not available")

        checked = (
            wide.filter(
                pl.col("shares_diluted").is_not_null()
                & pl.col("eps_diluted").is_not_null()
                & pl.col("net_income").is_not_null()
                & (pl.col("eps_diluted").abs() > 0.10)
            )
            .with_columns((pl.col("net_income") / pl.col("eps_diluted")).alias("implied"))
            .with_columns(
                (pl.col("shares_diluted") / pl.col("implied") - 1.0).abs().alias("gap")
            )
        )
        assert checked.height > 300, "expected most of the universe to be checkable"
        bad = checked.filter(pl.col("gap") > 0.25)
        share = bad.height / checked.height
        assert share < 0.05, (
            f"{bad.height} of {checked.height} companies ({share:.1%}) have a diluted share "
            "count more than 25% away from the count implied by earnings per share"
        )


class TestPriceIntegrity:
    def test_both_price_series_are_present_and_ordered(self, prices):
        assert prices.filter(pl.col("close_raw") <= 0).height == 0
        assert prices.filter(pl.col("close_adj") <= 0).height == 0
        # The dividend-adjusted series is at or below the split-adjusted one.
        bad = prices.filter(pl.col("close_adj") > pl.col("close_raw") * 1.001)
        assert bad.height == 0

    def test_split_factor_is_one_at_the_most_recent_date(self, prices):
        last = prices["date"].max()
        latest = prices.filter(pl.col("date") == last)
        assert latest.filter((pl.col("split_factor") - 1.0).abs() > 1e-9).height == 0

    def test_split_factor_never_decreases_going_back_in_time(self, prices):
        """It is the product of future splits, so it can only rise going back."""
        sample = prices.filter(pl.col("ticker").is_in(["AAPL", "NVDA", "AMZN", "GOOGL"]))
        if sample.is_empty():
            pytest.skip("sample tickers absent")
        diffs = (
            sample.sort(["ticker", "date"])
            .with_columns(
                (pl.col("split_factor").diff().over("ticker")).alias("d")
            )
            .filter(pl.col("d") > 1e-9)
        )
        assert diffs.height == 0, "split factor increased going forward in time"

    def test_no_duplicate_bars(self, prices):
        dupes = (
            prices.group_by(["ticker", "date"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
        )
        assert dupes.height == 0

    def test_dollar_volume_uses_the_traded_price(self, prices):
        """Using the dividend-adjusted close biases the liquidity screen."""
        sample = prices.filter(pl.col("volume") > 0).head(5000)
        recomputed = sample["close_raw"] * sample["volume"]
        diff = (sample["dollar_volume"] - recomputed).abs().max()
        assert diff < 1.0


class TestDerivedTables:
    def test_scores_never_predate_their_inputs(self):
        scores = store.read("opportunity_scores")
        fundamentals = store.read("fundamentals")
        if scores is None or fundamentals is None:
            pytest.skip("derived tables not built")
        earliest_score = scores["date"].min()
        earliest_fact = fundamentals["available_at"].min()
        assert earliest_score >= earliest_fact

    def test_every_scored_company_has_a_category_breakdown(self):
        scores = store.read("opportunity_scores")
        categories = store.read("category_scores")
        if scores is None or categories is None:
            pytest.skip("derived tables not built")
        as_of = scores["date"].max()
        scored = scores.filter(
            (pl.col("date") == as_of) & pl.col("score").is_not_null()
        )["company_id"]
        with_categories = categories.filter(pl.col("date") == as_of)["company_id"].unique()
        missing = set(scored.to_list()) - set(with_categories.to_list())
        assert not missing, f"{len(missing)} scored companies have no category breakdown"

    def test_metric_scores_keep_the_raw_value(self):
        """The brief requires storing both the raw value and the normalised score."""
        metric_scores = store.read("metric_scores")
        if metric_scores is None:
            pytest.skip("metric scores not built")
        scored = metric_scores.filter(pl.col("status") == "scored")
        assert scored.height > 0
        assert scored["raw_value"].null_count() == 0

    def test_unscored_metrics_carry_no_score(self):
        metric_scores = store.read("metric_scores")
        if metric_scores is None:
            pytest.skip("metric scores not built")
        unscored = metric_scores.filter(pl.col("status") != "scored")
        if unscored.is_empty():
            return
        assert unscored["score"].null_count() == unscored.height, (
            "a metric marked missing or not-applicable still carries a score"
        )
