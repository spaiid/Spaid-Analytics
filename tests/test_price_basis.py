"""What basis the stored price series is actually on.

This is a property of the provider, not of our code, and it is the single
assumption the whole simulator rests on. It is tested against the real store
rather than against a fixture, because a fixture would only assert what we
believe; the point here is to catch the provider changing its mind.

The belief under test: both close series are restated onto today's share basis,
so a split leaves no discontinuity in either, and `split_factor` is what
converts back to the price the stock actually traded at.
"""

from __future__ import annotations

import polars as pl
import pytest

from spaid.storage import store


@pytest.fixture(scope="module")
def prices() -> pl.DataFrame:
    df = store.read("prices")
    if df is None or df.is_empty():
        pytest.skip("no prices in the store")
    return df


@pytest.fixture(scope="module")
def splits(prices: pl.DataFrame) -> pl.DataFrame:
    events = prices.filter(
        pl.col("split_ratio").is_not_null() & (pl.col("split_ratio") > 1.5)
    )
    if events.is_empty():
        pytest.skip("no splits in the price history")
    return events


class TestSplitBasis:
    def test_a_split_leaves_no_jump_in_the_close(self, prices, splits):
        """The close must not fall by the split ratio on the ex-date.

        If it did, the series would be genuinely as-traded and the simulator
        would have to multiply its share counts on that date. It does not, and
        the simulator must not.
        """
        with_previous = (
            prices.sort(["ticker", "date"])
            .with_columns(
                pl.col("close_raw").shift(1).over("ticker").alias("previous_close")
            )
            .join(splits.select(["ticker", "date", "split_ratio"]), on=["ticker", "date"])
            .filter(pl.col("previous_close").is_not_null() & (pl.col("previous_close") > 0))
            .with_columns(
                (pl.col("close_raw") / pl.col("previous_close")).alias("ratio")
            )
        )
        if with_previous.is_empty():
            pytest.skip("no split with a previous session to compare against")

        # A genuine as-traded series shows the close falling by *exactly* the
        # split ratio, so `ratio * split_ratio` lands on one. Anything else is
        # an ordinary day's move. The test looks for that product, not for a
        # large fall: several of these events are spin-offs the provider encodes
        # as splits, where the price really did drop because value left the
        # company, and those are not evidence about the split basis.
        as_traded_looking = with_previous.filter(
            ((pl.col("ratio") * pl.col("split_ratio")) - 1.0).abs() < 0.15
        )
        assert as_traded_looking.height == 0, (
            "the close falls by roughly the split ratio on "
            f"{as_traded_looking.height} ex-dates, so the series is as-traded after all "
            "and spaid.backtest.engine must restore its share-count adjustment: "
            f"{as_traded_looking.select(['ticker', 'date', 'split_ratio', 'ratio']).head(3).to_dicts()}"
        )

    def test_the_split_factor_recovers_the_as_traded_price(self, prices):
        """Nvidia traded above $20 in 2015; on today's basis its close reads $0.56.

        The product of the two is what a price screen has to be applied to, and
        the defect this guards against is subtle: screening the stored close
        directly lets a 2024 split decide 2015 eligibility.
        """
        nvda = prices.filter(
            (pl.col("ticker") == "NVDA") & (pl.col("date").dt.year() == 2015)
        )
        if nvda.is_empty():
            pytest.skip("no Nvidia history in 2015")
        stored = float(nvda["close_raw"][0])
        as_traded = stored * float(nvda["split_factor"][0])
        assert stored < 5.0, "expected the stored close to be on today's split basis"
        assert 10.0 < as_traded < 60.0, (
            f"the as-traded price came out at {as_traded:.2f}; Nvidia traded in the "
            "low twenties in 2015"
        )

    def test_the_split_factor_is_one_at_the_end_of_the_series(self, prices):
        """Nothing has split after the last session, so the bridge is the identity."""
        last = prices.filter(pl.col("date") == prices["date"].max())
        finite = last.filter(pl.col("split_factor").is_not_null())
        assert finite.height > 0
        assert float(finite["split_factor"].max()) == pytest.approx(1.0)


class TestDividendBasis:
    def test_dividends_are_stated_on_the_same_basis_as_the_close(self, prices):
        """A dividend must be a sane fraction of the close it is paid against.

        If the dividend were on the as-traded basis while the close was restated,
        Apple's 2015 payment would read $0.52 against a $30 close -- a 1.7%
        quarterly yield that compounds to something no large company pays. The
        check is deliberately loose; it is looking for a factor-of-N error, not
        for precision.
        """
        paying = prices.filter(
            pl.col("dividend").is_not_null()
            & (pl.col("dividend") > 0)
            & (pl.col("close_raw") > 0)
        ).with_columns((pl.col("dividend") / pl.col("close_raw")).alias("yield_"))
        if paying.is_empty():
            pytest.skip("no dividends in the price history")

        # A single quarterly payment above a tenth of the share price is not a
        # dividend, it is a units mismatch.
        implausible = paying.filter(pl.col("yield_") > 0.10)
        share = implausible.height / paying.height
        assert share < 0.005, (
            f"{implausible.height} of {paying.height} dividend payments exceed 10% of the "
            "share price, which suggests they are stated on a different split basis from "
            f"the close: {implausible.select(['ticker', 'date', 'dividend', 'close_raw']).head(3).to_dicts()}"
        )

    def test_the_two_close_series_differ_only_by_dividends(self, prices):
        """`close_adj` includes reinvested dividends, so it sits below `close_raw`
        historically and converges to it at the end of the series.

        Both being on the same split basis is what makes that true. A split
        basis mismatch would show up here as a factor-of-N gap rather than a
        gap that closes over time.
        """
        payers = (
            prices.filter(pl.col("dividend") > 0)["ticker"].unique().to_list()
        )
        if not payers:
            pytest.skip("no dividend payers")
        sample = prices.filter(pl.col("ticker").is_in(payers[:50]))
        last = sample.filter(pl.col("date") == sample["date"].max())
        converged = last.filter(
            (pl.col("close_adj") / pl.col("close_raw")).is_between(0.98, 1.02)
        )
        assert converged.height >= last.height * 0.9, (
            "the adjusted and raw closes should coincide at the end of the series; "
            f"only {converged.height} of {last.height} do"
        )
