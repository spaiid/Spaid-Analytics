"""Mapping from US-GAAP XBRL tags to the canonical concepts the app reasons about.

Filers describe the same economic quantity with different tags, and the same
filer changes tag over time: Procter & Gamble reported `SalesRevenueNet` until
2018 and `Revenues` afterwards. So each concept lists every tag that can carry
it, in preference order, and the extractor merges across all of them rather than
betting on the first one that returns anything.

Preference order matters only for ties -- when two tags report the same period in
the same filing, the earlier tag in the list wins. It is ordered from most
specific to most general, because a filer that reports both
`RevenueFromContractWithCustomerExcludingAssessedTax` and `Revenues` usually
means the former as the clean top line.

Tag coverage below was measured against all 500 cached S&P 500 companies; the
counts in the comments are how many of those filers carry that tag at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PeriodType(StrEnum):
    DURATION = "duration"  # a flow measured over a window (revenue, cash flow)
    INSTANT = "instant"  # a level at a point in time (assets, debt, shares)


class Aggregation(StrEnum):
    """How four quarters combine into a trailing-twelve-month figure.

    Most flows add up: four quarters of revenue are a year of revenue. But a
    weighted-average share count is a *level* that happens to be reported over a
    window, and adding four quarters of it would report four times the company's
    shares. Earnings per share adds (four quarterly figures are the annual
    figure); shares outstanding does not.
    """

    SUM = "sum"
    LAST = "last"  # take the most recent period rather than summing


class Unit(StrEnum):
    USD = "USD"
    SHARES = "shares"
    USD_PER_SHARE = "USD/shares"
    PURE = "pure"  # ratios and rates


@dataclass(frozen=True)
class ConceptSpec:
    key: str
    label: str
    period_type: PeriodType
    unit: Unit
    tags: tuple[str, ...]
    # How quarterly values roll up into a trailing-twelve-month figure.
    aggregation: Aggregation = Aggregation.SUM
    # Whether a missing value should be read as a true zero. A company that
    # files a full cash-flow statement with no buyback line did not buy back
    # stock; a company with no `Revenues` tag is a data gap. Getting this
    # distinction right is the difference between "pays no dividend" and "we
    # don't know whether it pays a dividend".
    absent_means_zero: bool = False
    # Concepts only meaningful for particular kinds of business.
    domain: str = "general"  # general | bank | insurer | reit
    description: str = ""


# ---------------------------------------------------------------------------
# Income statement
# ---------------------------------------------------------------------------
INCOME: tuple[ConceptSpec, ...] = (
    ConceptSpec(
        "revenue", "Revenue", PeriodType.DURATION, Unit.USD,
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",  # 358/500
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",  # 400/500
            "SalesRevenueNet",  # 227/500
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
            # Financial filers report revenue net of interest expense, and
            # insurers roll premiums into their own total.
            "RevenuesNetOfInterestExpense",
            "TotalRevenuesAndOtherIncome",
            "RegulatedAndUnregulatedOperatingRevenue",
        ),
        description="Top-line revenue for the period.",
    ),
    ConceptSpec(
        "cost_of_revenue", "Cost of revenue", PeriodType.DURATION, Unit.USD,
        (
            "CostOfGoodsAndServicesSold",  # 264/500
            "CostOfRevenue",  # 136/500
            "CostOfGoodsSold",
            "CostOfServices",
        ),
        description="Direct cost of goods and services, used to derive gross profit.",
    ),
    ConceptSpec(
        "gross_profit", "Gross profit", PeriodType.DURATION, Unit.USD,
        ("GrossProfit",),  # 256/500 -- derived from revenue - cost_of_revenue when absent
        description="Revenue less cost of revenue.",
    ),
    ConceptSpec(
        "operating_income", "Operating income", PeriodType.DURATION, Unit.USD,
        (
            "OperatingIncomeLoss",  # 419/500
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        ),
        description="Profit from operations, before interest and tax.",
    ),
    ConceptSpec(
        "operating_expenses", "Operating expenses", PeriodType.DURATION, Unit.USD,
        ("OperatingExpenses", "CostsAndExpenses"),
    ),
    ConceptSpec(
        "sga", "Selling, general and administrative", PeriodType.DURATION, Unit.USD,
        (
            "SellingGeneralAndAdministrativeExpense",  # 293/500
            "GeneralAndAdministrativeExpense",
        ),
    ),
    ConceptSpec(
        "rnd", "Research and development", PeriodType.DURATION, Unit.USD,
        ("ResearchAndDevelopmentExpense",),  # 228/500
        absent_means_zero=True,
        description="R&D spend. Absent for companies that do no R&D, which is a true zero.",
    ),
    ConceptSpec(
        "pretax_income", "Pre-tax income", PeriodType.DURATION, Unit.USD,
        (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",  # 459/500
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",  # 367/500
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
        ),
    ),
    ConceptSpec(
        "tax_expense", "Income tax expense", PeriodType.DURATION, Unit.USD,
        ("IncomeTaxExpenseBenefit",),  # 497/500
    ),
    ConceptSpec(
        "net_income", "Net income", PeriodType.DURATION, Unit.USD,
        ("NetIncomeLoss", "ProfitLoss"),  # 496/500
        description="Net income attributable to the parent.",
    ),
    ConceptSpec(
        "net_income_to_common", "Net income to common", PeriodType.DURATION, Unit.USD,
        ("NetIncomeLossAvailableToCommonStockholdersBasic",),  # 303/500
        description="After preferred dividends; the numerator for earnings per share.",
    ),
    ConceptSpec(
        "interest_expense", "Interest expense", PeriodType.DURATION, Unit.USD,
        (
            "InterestExpenseNonoperating",  # 237/500
            "InterestExpenseDebt",  # 129/500
            "InterestExpense",  # 416/500 -- for banks this is a cost of revenue, see below
            "InterestAndDebtExpense",
        ),
        description=(
            "Interest on borrowings, used for coverage and for the bridge from operating "
            "income to pre-tax income. For banks the same tag means something different and "
            "the bank-specific concept is used instead."
        ),
    ),
    ConceptSpec(
        "depreciation_amortization", "Depreciation and amortisation",
        PeriodType.DURATION, Unit.USD,
        (
            "DepreciationDepletionAndAmortization",  # 365/500
            "DepreciationAmortizationAndAccretionNet",  # 73/500
            "DepreciationAndAmortization",  # 226/500
            "Depreciation",  # 354/500
        ),
        description="Taken from the cash-flow statement; the EBITDA add-back.",
    ),
    ConceptSpec(
        "amortization_intangibles", "Amortisation of intangibles",
        PeriodType.DURATION, Unit.USD,
        ("AmortizationOfIntangibleAssets",),  # 433/500
        absent_means_zero=True,
    ),
    ConceptSpec(
        "share_based_comp", "Share-based compensation", PeriodType.DURATION, Unit.USD,
        (
            "ShareBasedCompensation",  # 439/500
            "AllocatedShareBasedCompensationExpense",  # 415/500
            "ShareBasedCompensationArrangementByShareBasedPaymentAwardCompensationCost",
        ),
        absent_means_zero=True,
        description="A real cost to existing owners, expensed in the valuation model.",
    ),
    ConceptSpec(
        "eps_diluted", "Diluted earnings per share", PeriodType.DURATION, Unit.USD_PER_SHARE,
        ("EarningsPerShareDiluted", "IncomeLossFromContinuingOperationsPerDilutedShare"),  # 493/500
    ),
    ConceptSpec(
        "eps_basic", "Basic earnings per share", PeriodType.DURATION, Unit.USD_PER_SHARE,
        ("EarningsPerShareBasic", "IncomeLossFromContinuingOperationsPerBasicShare"),  # 492/500
    ),
    ConceptSpec(
        "shares_diluted", "Diluted weighted-average shares", PeriodType.DURATION, Unit.SHARES,
        (
            "WeightedAverageNumberOfDilutedSharesOutstanding",  # 492/500
            # NOT `...OutstandingAdjustment`: that tag is the *incremental*
            # dilutive share count, a few million shares, not the total. Reading
            # it as the count reports Microsoft as having minus four million
            # shares outstanding.
        ),
        aggregation=Aggregation.LAST,
        description=(
            "The consolidated diluted share count. Preferred over the cover-page count because "
            "it is non-dimensional even for dual-class issuers, where the cover page is "
            "reported per class and therefore absent from the SEC's aggregate API."
        ),
    ),
    ConceptSpec(
        "shares_basic", "Basic weighted-average shares", PeriodType.DURATION, Unit.SHARES,
        ("WeightedAverageNumberOfSharesOutstandingBasic",),  # 492/500
        aggregation=Aggregation.LAST,
    ),
    ConceptSpec(
        "dividends_per_share", "Dividends declared per share",
        PeriodType.DURATION, Unit.USD_PER_SHARE,
        (
            "CommonStockDividendsPerShareDeclared",  # 379/500
            "CommonStockDividendsPerShareCashPaid",  # 256/500
        ),
        absent_means_zero=True,
    ),
)

# ---------------------------------------------------------------------------
# Cash flow
# ---------------------------------------------------------------------------
CASH_FLOW: tuple[ConceptSpec, ...] = (
    ConceptSpec(
        "operating_cash_flow", "Operating cash flow", PeriodType.DURATION, Unit.USD,
        (
            "NetCashProvidedByUsedInOperatingActivities",  # 499/500
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ),
    ConceptSpec(
        "capex", "Capital expenditure", PeriodType.DURATION, Unit.USD,
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",  # 395/500
            "PaymentsToAcquireProductiveAssets",  # 181/500
            "PaymentsForCapitalImprovements",
            "PaymentsToAcquireOtherPropertyPlantAndEquipment",
            # Energy producers book their capital spend under exploration and
            # development rather than the generic property tag.
            "PaymentsToExploreAndDevelopOilAndGasProperties",
            "PaymentsToAcquireOilAndGasProperty",
            "PaymentsToAcquireOilAndGasPropertyAndEquipment",
        ),
        description="Reported as a positive outflow in the SEC data.",
    ),
    ConceptSpec(
        "acquisitions", "Acquisitions", PeriodType.DURATION, Unit.USD,
        ("PaymentsToAcquireBusinessesNetOfCashAcquired",),  # 432/500
        absent_means_zero=True,
        description="Used to separate acquisition-driven growth from organic growth.",
    ),
    ConceptSpec(
        "buyback", "Share repurchases", PeriodType.DURATION, Unit.USD,
        ("PaymentsForRepurchaseOfCommonStock",),  # 477/500
        absent_means_zero=True,
    ),
    ConceptSpec(
        "stock_issued", "Stock issued", PeriodType.DURATION, Unit.USD,
        (
            "ProceedsFromIssuanceOfCommonStock",
            "ProceedsFromStockOptionsExercised",
            "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlans",
        ),
        absent_means_zero=True,
    ),
    ConceptSpec(
        "dividends_paid", "Dividends paid", PeriodType.DURATION, Unit.USD,
        (
            "PaymentsOfDividendsCommonStock",  # 287/500
            "PaymentsOfDividends",  # 227/500
            "PaymentsOfDistributionsToAffiliates",
        ),
        absent_means_zero=True,
    ),
    ConceptSpec(
        "debt_issued", "Debt issued", PeriodType.DURATION, Unit.USD,
        ("ProceedsFromIssuanceOfLongTermDebt",),  # 369/500
        absent_means_zero=True,
    ),
    ConceptSpec(
        "debt_repaid", "Debt repaid", PeriodType.DURATION, Unit.USD,
        ("RepaymentsOfLongTermDebt",),  # 358/500
        absent_means_zero=True,
    ),
    ConceptSpec(
        "interest_paid", "Interest paid in cash", PeriodType.DURATION, Unit.USD,
        ("InterestPaidNet", "InterestPaid"),  # 447/500
    ),
    ConceptSpec(
        "taxes_paid", "Income taxes paid", PeriodType.DURATION, Unit.USD,
        ("IncomeTaxesPaidNet", "IncomeTaxesPaid"),  # 466/500
    ),
)

# ---------------------------------------------------------------------------
# Balance sheet (instants)
# ---------------------------------------------------------------------------
BALANCE: tuple[ConceptSpec, ...] = (
    ConceptSpec("assets", "Total assets", PeriodType.INSTANT, Unit.USD, ("Assets",)),  # 500/500
    ConceptSpec(
        "current_assets", "Current assets", PeriodType.INSTANT, Unit.USD, ("AssetsCurrent",)
    ),  # 422/500 -- banks and REITs do not present a classified balance sheet
    ConceptSpec(
        "liabilities", "Total liabilities", PeriodType.INSTANT, Unit.USD,
        ("Liabilities",),  # 374/500 -- derived as assets - equity when absent
    ),
    ConceptSpec(
        "current_liabilities", "Current liabilities", PeriodType.INSTANT, Unit.USD,
        ("LiabilitiesCurrent",),
    ),
    ConceptSpec(
        "equity", "Shareholders' equity", PeriodType.INSTANT, Unit.USD,
        ("StockholdersEquity",),  # 493/500 -- parent only, excludes minority interest
    ),
    ConceptSpec(
        "equity_incl_nci", "Equity including minority interest", PeriodType.INSTANT, Unit.USD,
        ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",),  # 395/500
    ),
    ConceptSpec(
        "minority_interest", "Minority interest", PeriodType.INSTANT, Unit.USD,
        ("MinorityInterest",),  # 347/500
        absent_means_zero=True,
        description="Added to enterprise value: it is a claim on the assets we are valuing.",
    ),
    ConceptSpec(
        "preferred_equity", "Preferred equity", PeriodType.INSTANT, Unit.USD,
        ("PreferredStockValue", "PreferredStockLiquidationPreferenceValue"),  # 309/500
        absent_means_zero=True,
    ),
    ConceptSpec(
        "cash", "Cash and equivalents", PeriodType.INSTANT, Unit.USD,
        (
            "CashAndCashEquivalentsAtCarryingValue",  # 485/500
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",  # 477/500
        ),
    ),
    ConceptSpec(
        "short_term_investments", "Short-term investments", PeriodType.INSTANT, Unit.USD,
        (
            "ShortTermInvestments",  # 176/500
            "MarketableSecuritiesCurrent",  # 86/500
            "AvailableForSaleSecuritiesDebtSecuritiesCurrent",  # 79/500
            "OtherShortTermInvestments",
        ),
        absent_means_zero=True,
        description="Counted with cash when netting debt, as it is readily realisable.",
    ),
    ConceptSpec(
        "debt_long", "Long-term debt", PeriodType.INSTANT, Unit.USD,
        (
            "LongTermDebtNoncurrent",  # 344/500
            "LongTermDebt",  # 424/500
            "LongTermDebtAndCapitalLeaseObligations",  # 189/500
        ),
        absent_means_zero=True,
    ),
    ConceptSpec(
        "debt_current", "Short-term and current debt", PeriodType.INSTANT, Unit.USD,
        (
            "LongTermDebtCurrent",  # 331/500
            "DebtCurrent",  # 183/500
            "ShortTermBorrowings",  # 243/500
            "LongTermDebtAndCapitalLeaseObligationsCurrent",  # 149/500
            "CommercialPaper",  # 168/500
            "OtherShortTermBorrowings",
        ),
        absent_means_zero=True,
    ),
    ConceptSpec(
        "operating_lease_liability", "Operating lease liability", PeriodType.INSTANT, Unit.USD,
        (
            "OperatingLeaseLiability",  # 487/500
            "OperatingLeaseLiabilityNoncurrent",  # 411/500
        ),
        absent_means_zero=True,
        description=(
            "Capitalised lease obligations. Included in enterprise value because a leased "
            "store and an owned one financed with debt are economically similar."
        ),
    ),
    ConceptSpec(
        "inventory", "Inventory", PeriodType.INSTANT, Unit.USD,
        ("InventoryNet",),  # 318/500
        absent_means_zero=True,
    ),
    ConceptSpec(
        "receivables", "Accounts receivable", PeriodType.INSTANT, Unit.USD,
        ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"),  # 341/500
    ),
    ConceptSpec(
        "payables", "Accounts payable", PeriodType.INSTANT, Unit.USD,
        ("AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"),  # 367/500
    ),
    ConceptSpec(
        "ppe_net", "Property, plant and equipment", PeriodType.INSTANT, Unit.USD,
        ("PropertyPlantAndEquipmentNet",),  # 469/500
    ),
    ConceptSpec(
        "goodwill", "Goodwill", PeriodType.INSTANT, Unit.USD,
        ("Goodwill",),  # 467/500
        absent_means_zero=True,
    ),
    ConceptSpec(
        "intangibles", "Intangible assets", PeriodType.INSTANT, Unit.USD,
        ("IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"),  # 383/500
        absent_means_zero=True,
    ),
    ConceptSpec(
        "retained_earnings", "Retained earnings", PeriodType.INSTANT, Unit.USD,
        ("RetainedEarningsAccumulatedDeficit",),  # 488/500
    ),
    ConceptSpec(
        "shares_outstanding", "Shares outstanding (cover page)", PeriodType.INSTANT, Unit.SHARES,
        (
            "dei:EntityCommonStockSharesOutstanding",  # 472/500
            "CommonStockSharesOutstanding",  # 394/500
        ),
        description=(
            "Cover-page share count. Absent for dual-class issuers, whose counts are reported "
            "per class and therefore excluded from the aggregate API -- hence the diluted "
            "weighted-average fallback."
        ),
    ),
)

# ---------------------------------------------------------------------------
# Bank-specific
# ---------------------------------------------------------------------------
BANK: tuple[ConceptSpec, ...] = (
    ConceptSpec(
        "interest_income", "Interest and dividend income", PeriodType.DURATION, Unit.USD,
        ("InterestAndDividendIncomeOperating", "InterestIncomeOperating"),
        domain="bank",
    ),
    ConceptSpec(
        "net_interest_income", "Net interest income", PeriodType.DURATION, Unit.USD,
        (
            "InterestIncomeExpenseNet",
            "InterestIncomeExpenseAfterProvisionForLoanLoss",
        ),
        domain="bank",
    ),
    ConceptSpec(
        "noninterest_income", "Non-interest income", PeriodType.DURATION, Unit.USD,
        ("NoninterestIncome",),
        domain="bank",
    ),
    ConceptSpec(
        "noninterest_expense", "Non-interest expense", PeriodType.DURATION, Unit.USD,
        ("NoninterestExpense",),
        domain="bank",
    ),
    ConceptSpec(
        "credit_provision", "Provision for credit losses", PeriodType.DURATION, Unit.USD,
        (
            "ProvisionForLoanLeaseAndOtherLosses",
            "ProvisionForLoanLossesExpensed",
            "ProvisionForCreditLossesExpensed",
        ),
        domain="bank",
        description="A rising provision is the earliest visible sign of a deteriorating book.",
    ),
    ConceptSpec(
        "loans", "Loans and leases", PeriodType.INSTANT, Unit.USD,
        (
            "LoansAndLeasesReceivableNetReportedAmount",
            "NotesReceivableNet",
            "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
        ),
        domain="bank",
    ),
    ConceptSpec(
        "deposits", "Deposits", PeriodType.INSTANT, Unit.USD, ("Deposits",), domain="bank"
    ),
)

# ---------------------------------------------------------------------------
# Insurance-specific
# ---------------------------------------------------------------------------
INSURER: tuple[ConceptSpec, ...] = (
    ConceptSpec(
        "premiums_earned", "Premiums earned", PeriodType.DURATION, Unit.USD,
        ("PremiumsEarnedNet", "PremiumsEarnedNetPropertyAndCasualty"),
        domain="insurer",
    ),
    ConceptSpec(
        "policy_benefits", "Policy benefits and claims", PeriodType.DURATION, Unit.USD,
        (
            "PolicyholderBenefitsAndClaimsIncurredNet",
            "BenefitsLossesAndExpenses",
            "LiabilityForClaimsAndClaimsAdjustmentExpenseClaimsIncurredNet",
        ),
        domain="insurer",
        description="With premiums earned, gives the loss ratio: the core underwriting measure.",
    ),
)

# ---------------------------------------------------------------------------
# REIT-specific
# ---------------------------------------------------------------------------
REIT: tuple[ConceptSpec, ...] = (
    ConceptSpec(
        "real_estate_gross", "Real estate at cost", PeriodType.INSTANT, Unit.USD,
        ("RealEstateInvestmentPropertyAtCost", "RealEstateInvestmentPropertyNet"),
        domain="reit",
    ),
    ConceptSpec(
        "real_estate_accum_dep", "Accumulated depreciation on real estate",
        PeriodType.INSTANT, Unit.USD,
        ("RealEstateInvestmentPropertyAccumulatedDepreciation",),
        domain="reit",
        description=(
            "Added back to book value for net-asset-value work: property is carried at "
            "depreciated cost, which understates it badly after a decade of appreciation."
        ),
    ),
)

ALL_CONCEPTS: tuple[ConceptSpec, ...] = INCOME + CASH_FLOW + BALANCE + BANK + INSURER + REIT
CONCEPT_BY_KEY: dict[str, ConceptSpec] = {c.key: c for c in ALL_CONCEPTS}

FLOW_CONCEPTS: tuple[ConceptSpec, ...] = tuple(
    c for c in ALL_CONCEPTS if c.period_type is PeriodType.DURATION
)
INSTANT_CONCEPTS: tuple[ConceptSpec, ...] = tuple(
    c for c in ALL_CONCEPTS if c.period_type is PeriodType.INSTANT
)

# Reverse index: tag -> the concepts that may claim it, with their rank.
TAG_TO_CONCEPTS: dict[str, list[tuple[str, int]]] = {}
for _c in ALL_CONCEPTS:
    for _rank, _tag in enumerate(_c.tags):
        TAG_TO_CONCEPTS.setdefault(_tag, []).append((_c.key, _rank))

# Every tag we care about, for a fast membership test while streaming the JSON.
WANTED_TAGS: frozenset[str] = frozenset(TAG_TO_CONCEPTS)

# Concepts whose absence in an otherwise complete filing is a genuine zero.
ZERO_WHEN_ABSENT: frozenset[str] = frozenset(
    c.key for c in ALL_CONCEPTS if c.absent_means_zero
)


def concepts_for_tag(tag: str) -> list[tuple[str, int]]:
    return TAG_TO_CONCEPTS.get(tag, [])
