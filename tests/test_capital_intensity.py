"""Tests for the sales-to-capital estimate that governs reinvestment.

This is the single most consequential input the discounted cash-flow model
takes, because it decides what growth costs. Getting it from the balance sheet
valued AMD at $5.54 against a $503.60 price: $57bn of invested capital, most of
it acquired intangibles already paid for, against $41bn of revenue reads as a
capital-hungry business, while the company actually spends 4% of revenue a year
on capital expenditure.
"""

from __future__ import annotations

import polars as pl
import pytest

from spaid.pipeline.analyze import sales_to_capital_table


def company_row(ticker: str, **kw) -> dict:
    """A company with five years of revenue history and a capital budget."""
    row = dict(
        ticker=ticker,
        sector="Information Technology",
        revenue=41_305_000_000.0,
        revenue__lag1y=29_600_000_000.0,
        revenue__lag3y=21_876_000_000.0,
        revenue__lag5y=13_340_000_000.0,
        capex=1_677_000_000.0,
        acquisitions=0.0,
        invested_capital=57_339_000_000.0,
    )
    row.update(kw)
    return row


def frame(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


class TestIncrementalEstimate:
    def test_it_measures_spending_not_the_balance_sheet(self):
        """AMD: 0.72 on the books, but growth actually costs a third of that."""
        table = sales_to_capital_table(frame(company_row("AMD")))
        book = 41_305_000_000.0 / 57_339_000_000.0
        expected = (41_305_000_000.0 - 13_340_000_000.0) / (5 * 1_677_000_000.0)
        assert table["AMD"] == pytest.approx(expected)
        assert table["AMD"] > book * 4

    def test_the_longest_window_available_wins(self):
        """A longer window averages out a lumpy year of building."""
        without_five = company_row("AMD")
        without_five["revenue__lag5y"] = None
        table = sales_to_capital_table(frame(without_five))
        expected = (41_305_000_000.0 - 21_876_000_000.0) / (3 * 1_677_000_000.0)
        assert table["AMD"] == pytest.approx(expected)

    def test_cash_paid_for_acquisitions_counts_as_the_cost_of_growth(self):
        """Revenue that was bought was not free."""
        organic = sales_to_capital_table(frame(company_row("ORG")))
        bought = sales_to_capital_table(
            frame(company_row("BUY", acquisitions=4_000_000_000.0))
        )
        assert bought["BUY"] < organic["ORG"]

    def test_a_divestiture_does_not_make_growth_look_cheaper(self):
        """Negative acquisition spending is floored, not treated as a rebate."""
        plain = sales_to_capital_table(frame(company_row("A")))
        sold = sales_to_capital_table(
            frame(company_row("B", acquisitions=-8_000_000_000.0))
        )
        assert sold["B"] == pytest.approx(plain["A"])


class TestFallbacks:
    def test_shrinking_revenue_falls_back_to_the_sector(self):
        """A company that shrank cannot say what growth would have cost it."""
        shrinking = company_row(
            "DOWN", revenue=10_000_000_000.0, invested_capital=4_000_000_000.0
        )
        peers = [company_row(f"PEER{i}") for i in range(3)]
        table = sales_to_capital_table(frame(shrinking, *peers))
        assert table["DOWN"] == pytest.approx(table["PEER0"])

    def test_the_book_ratio_is_the_last_resort_not_the_first(self):
        """With no sector to borrow from, the balance sheet still beats nothing."""
        alone = company_row(
            "ALONE",
            revenue=10_000_000_000.0,
            revenue__lag1y=12_000_000_000.0,
            revenue__lag3y=13_000_000_000.0,
            revenue__lag5y=14_000_000_000.0,
            invested_capital=5_000_000_000.0,
            sector=None,
        )
        table = sales_to_capital_table(frame(alone))
        assert table["ALONE"] == pytest.approx(2.0)

    def test_a_company_with_nothing_to_go_on_is_absent(self):
        """Absent, so the caller applies the spec default rather than a guess."""
        blank = company_row(
            "BLANK",
            revenue=None,
            revenue__lag1y=None,
            revenue__lag3y=None,
            revenue__lag5y=None,
            capex=None,
            invested_capital=None,
            sector=None,
        )
        assert "BLANK" not in sales_to_capital_table(frame(blank))

    def test_an_empty_cross_section_is_not_an_error(self):
        assert sales_to_capital_table(frame(company_row("X")).clear()) == {}


class TestBounds:
    def test_an_absurd_ratio_is_rejected_rather_than_clamped(self):
        """A trivial capital budget against huge growth is a data problem.

        Rejecting sends the company to its sector median; clamping would keep a
        number that arose from a rounding error in the filing.
        """
        absurd = company_row("ODD", capex=1_000.0)
        peers = [company_row(f"PEER{i}") for i in range(3)]
        table = sales_to_capital_table(frame(absurd, *peers))
        assert table["ODD"] == pytest.approx(table["PEER0"])
