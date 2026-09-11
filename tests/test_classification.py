"""Tests for business-model classification.

Getting this wrong is expensive and silent. An asset manager classified as a
bank is valued on its book equity, which for a fee business bears no relation to
its worth: Ameriprise came out 74% overvalued against a market that was not
obviously mistaken. The failure mode is a plausible-looking number, so these
cases are pinned explicitly.
"""

from __future__ import annotations

import pytest

from spaid.config.scoring import BusinessModel
from spaid.providers.wikipedia.universe import classify_business_model


class TestFinancials:
    @pytest.mark.parametrize(
        ("industry", "expected"),
        [
            ("Diversified Banks", "bank"),
            ("Regional Banks", "bank"),
            ("Consumer Finance", "bank"),
            ("Investment Banking & Brokerage", "bank"),
            ("Life & Health Insurance", "insurer"),
            ("Property & Casualty Insurance", "insurer"),
            ("Multi-line Insurance", "insurer"),
            ("Reinsurance", "insurer"),
        ],
    )
    def test_balance_sheet_businesses(self, industry, expected):
        assert classify_business_model("Financials", industry) == expected

    @pytest.mark.parametrize(
        "industry",
        [
            "Asset Management & Custody Banks",
            "Insurance Brokers",
            "Financial Exchanges & Data",
            "Transaction & Payment Processing Services",
            "Multi-Sector Holdings",
        ],
    )
    def test_fee_businesses_are_operating_companies(self, industry):
        """The word "bank" or "insurance" in a label does not make it one."""
        assert classify_business_model("Financials", industry) == "operating"

    def test_asset_managers_are_never_banks(self):
        """The specific regression: substring matching caught the wrong companies."""
        assert classify_business_model("Financials", "Asset Management & Custody Banks") != "bank"

    def test_insurance_brokers_are_never_insurers(self):
        """A broker earns commission and carries no underwriting risk."""
        assert classify_business_model("Financials", "Insurance Brokers") != "insurer"

    def test_unknown_financial_subindustry_defaults_to_operating(self):
        """The safer error: a fee business mispriced as a lender is the costly one."""
        assert classify_business_model("Financials", "Some New Category") == "operating"


class TestRealEstate:
    @pytest.mark.parametrize(
        "industry",
        ["Data Center REITs", "Retail REITs", "Telecom Tower REITs", "Timber REITs"],
    )
    def test_landlords_are_trusts(self, industry):
        assert classify_business_model("Real Estate", industry) == "reit"

    def test_services_firms_are_operating_companies(self):
        assert classify_business_model("Real Estate", "Real Estate Services") == "operating"


class TestCyclicals:
    @pytest.mark.parametrize(
        ("sector", "industry"),
        [
            ("Energy", "Oil & Gas Exploration & Production"),
            ("Materials", "Fertilizers & Agricultural Chemicals"),
            ("Materials", "Steel"),
            ("Industrials", "Airlines"),
            ("Consumer Discretionary", "Automobile Manufacturers"),
            ("Consumer Discretionary", "Homebuilding"),
        ],
    )
    def test_commodity_and_capital_cycle_businesses(self, sector, industry):
        assert classify_business_model(sector, industry) == "cyclical"


class TestOperating:
    @pytest.mark.parametrize(
        ("sector", "industry"),
        [
            ("Information Technology", "Systems Software"),
            ("Health Care", "Pharmaceuticals"),
            ("Consumer Staples", "Soft Drinks & Non-alcoholic Beverages"),
            ("Utilities", "Electric Utilities"),
            ("Communication Services", "Interactive Media & Services"),
        ],
    )
    def test_ordinary_companies(self, sector, industry):
        assert classify_business_model(sector, industry) == "operating"

    def test_every_result_is_a_known_business_model(self):
        combinations = [
            ("Financials", "Diversified Banks"),
            ("Real Estate", "Office REITs"),
            ("Energy", "Integrated Oil & Gas"),
            ("Information Technology", "Semiconductors"),
            (None, None),
            ("", ""),
        ]
        valid = {m.value for m in BusinessModel}
        for sector, industry in combinations:
            assert classify_business_model(sector, industry) in valid
