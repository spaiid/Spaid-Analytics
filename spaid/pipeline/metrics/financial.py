"""Quality, growth and valuation metrics derived from point-in-time financials.

Everything here is built on a company-date grid by as-of joining the latest
*knowable* fundamental value onto each date. That single discipline is what
makes the metrics honest: a metric dated 2 May uses the numbers a reader could
have pulled from EDGAR that morning, not the quarter that had ended but not yet
been filed.

Two calculations are worth reading closely because both were wrong in the
previous build and both are load-bearing.

**Market capitalisation.** The share count comes from a filing, stated on that
filing's share basis. The price series is restated onto today's basis. Bridging
them requires the split factor *as of the filing*, not as of the price date. Get
it backwards and a company's market capitalisation collapses by the split ratio
for the two months between a split and the next quarterly report, which turns
the largest company in the index into the cheapest stock in it.

**Growth.** Comparing today's trailing-twelve-month revenue against the value
that was *knowable a year ago* keeps both sides of the ratio point-in-time. Using
the value that was eventually reported for a year ago quietly imports hindsight
into the denominator.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import polars as pl

from spaid.config.valuation import DcfSpec

log = logging.getLogger(__name__)

EPS = 1e-9

# Concepts the metric layer needs from the point-in-time fundamentals table.
NEEDED_CONCEPTS: tuple[str, ...] = (
    "revenue", "cost_of_revenue", "gross_profit", "operating_income",
    "pretax_income", "tax_expense", "net_income", "net_income_to_common",
    "interest_expense", "depreciation_amortization", "share_based_comp",
    "eps_diluted", "eps_basic", "shares_diluted", "shares_basic",
    "rnd", "sga", "operating_expenses",
    "operating_cash_flow", "capex", "acquisitions", "buyback", "dividends_paid",
    "stock_issued", "debt_issued", "debt_repaid",
    "assets", "current_assets", "liabilities", "current_liabilities",
    "equity", "equity_incl_nci", "minority_interest", "preferred_equity",
    "cash", "short_term_investments", "debt_long", "debt_current",
    "operating_lease_liability", "inventory", "receivables", "payables",
    "ppe_net", "goodwill", "intangibles", "retained_earnings",
    "shares_outstanding",
    # Financial-sector concepts, used by the bank and insurer templates.
    "interest_income", "net_interest_income", "noninterest_income",
    "noninterest_expense", "credit_provision", "loans", "deposits",
    "premiums_earned", "policy_benefits",
)


def safe_div(num: pl.Expr, den: pl.Expr, *, positive_only: bool = False) -> pl.Expr:
    """Divide, returning null rather than infinity when the denominator fails.

    `positive_only` additionally rejects negative denominators, which is what
    makes a ratio like return on equity silently skip companies with negative
    book value instead of reporting a positive return for a loss-making,
    balance-sheet-insolvent business.
    """
    guard = (den > EPS) if positive_only else (den.abs() > EPS)
    return pl.when(guard).then(num / den).otherwise(None)


def build_grid(
    prices: pl.DataFrame,
    securities: pl.DataFrame,
    *,
    dates: list | None = None,
) -> pl.DataFrame:
    """The company-date frame every metric is computed on.

    Only primary share classes appear, so a dual-class filer occupies one row
    per date rather than two.
    """
    primary = securities.filter(pl.col("is_primary")).select(["company_id", "ticker"])
    grid = prices.join(primary, on="ticker", how="inner")
    if dates is not None:
        grid = grid.filter(pl.col("date").is_in(dates))
    return grid.select(
        ["company_id", "ticker", "date", "close_raw", "close_adj", "split_factor"]
    ).sort(["company_id", "date"])


def attach_fundamentals(
    grid: pl.DataFrame,
    fundamentals: pl.DataFrame,
    *,
    concepts: tuple[str, ...] = NEEDED_CONCEPTS,
) -> pl.DataFrame:
    """As-of join the latest knowable value of each concept onto the grid.

    Emits, for every concept, the value plus the date it became available, so
    that staleness is visible to the confidence system and to the user rather
    than being silently absorbed.
    """
    out = grid.sort("date")
    available = set(fundamentals["concept"].unique().to_list())

    for concept in concepts:
        if concept not in available:
            out = out.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(concept),
                pl.lit(None, dtype=pl.Date).alias(f"{concept}__asof"),
            )
            continue
        sub = (
            fundamentals.filter(pl.col("concept") == concept)
            .select(["company_id", "available_at", "value"])
            .rename({"value": concept})
            .sort("available_at")
        )
        out = (
            out.sort("date")
            .join_asof(
                sub,
                left_on="date",
                right_on="available_at",
                by="company_id",
                strategy="backward",
            )
            .rename({"available_at": f"{concept}__asof"})
        )
    return out.sort(["company_id", "date"])


def attach_share_basis(panel: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """The split factor in force when the share count was filed.

    Multiplying a reported share count by this restates it onto the price
    series' basis. Using the split factor at the *price* date instead is the
    defect that made Nvidia's market capitalisation read $300bn for 55 trading
    days after its 2024 split.
    """
    factors = (
        prices.select(["ticker", "date", "split_factor"])
        .rename({"date": "_factor_date", "split_factor": "shares_split_factor"})
        .sort("_factor_date")
    )

    # Prefer the diluted count's availability date; fall back to the cover-page
    # count's when a filer reports only that.
    out = panel.with_columns(
        pl.coalesce(
            pl.col("shares_diluted__asof"), pl.col("shares_outstanding__asof")
        ).alias("_shares_asof")
    )

    matched = (
        out.filter(pl.col("_shares_asof").is_not_null())
        .sort("_shares_asof")
        .join_asof(
            factors,
            left_on="_shares_asof",
            right_on="_factor_date",
            by="ticker",
            strategy="backward",
        )
    )
    unmatched = out.filter(pl.col("_shares_asof").is_null()).with_columns(
        pl.lit(None, dtype=pl.Date).alias("_factor_date"),
        pl.lit(None, dtype=pl.Float64).alias("shares_split_factor"),
    )
    combined = pl.concat([matched, unmatched], how="diagonal_relaxed")

    # A filing that predates our price history gets the earliest factor we have,
    # which is the correct limiting value.
    earliest = (
        prices.sort("date")
        .group_by("ticker")
        .agg(pl.col("split_factor").first().alias("_earliest_factor"))
    )
    return (
        combined.join(earliest, on="ticker", how="left")
        .with_columns(
            pl.coalesce(
                pl.col("shares_split_factor"), pl.col("_earliest_factor"), pl.lit(1.0)
            ).alias("shares_split_factor")
        )
        .drop("_factor_date", "_earliest_factor")
        .sort(["company_id", "date"])
    )


def add_building_blocks(panel: pl.DataFrame, *, dcf: DcfSpec | None = None) -> pl.DataFrame:
    """Intermediate quantities that several metrics share."""
    dcf = dcf or DcfSpec()
    c = pl.col

    out = panel.with_columns(
        # Revenue, derived where the filer reports it only by segment. Some
        # energy producers tag revenue dimensionally (oil, gas, natural-gas
        # liquids) and the SEC's aggregate interface carries only undimensioned
        # facts, so the consolidated top line is simply absent. Operating income
        # plus operating expenses reconstructs it exactly, because that is the
        # identity the income statement is built on.
        pl.coalesce(
            c("revenue"),
            pl.when(
                c("operating_income").is_not_null() & c("operating_expenses").is_not_null()
            )
            .then(c("operating_income") + c("operating_expenses"))
            .otherwise(None),
        ).alias("revenue"),
    ).with_columns(
        # Gross profit, derived where the filer does not tag it directly. Half
        # the index does not report `GrossProfit`, and half a quality category
        # is too much to lose to a tagging convention.
        pl.coalesce(
            c("gross_profit"),
            pl.when(c("revenue").is_not_null() & c("cost_of_revenue").is_not_null())
            .then(c("revenue") - c("cost_of_revenue"))
            .otherwise(None),
        ).alias("gross_profit"),
        (c("debt_long").fill_null(0.0) + c("debt_current").fill_null(0.0)).alias("total_debt"),
        (c("cash").fill_null(0.0) + c("short_term_investments").fill_null(0.0)).alias(
            "cash_and_equivalents"
        ),
        (c("operating_cash_flow") - c("capex")).alias("free_cash_flow"),
    )

    # Trailing earnings per share, derived rather than summed.
    #
    # Summing four reported quarterly figures is wrong whenever a split falls
    # inside the window: the filer restates per-share history onto the new
    # basis, so a trailing total mixes pre- and post-split quarters. Netflix's
    # 10-for-1 split produced a trailing figure of -$9.24 for a company earning
    # $13.6bn. Dividing trailing net income by the current diluted share count
    # is always internally consistent, and is what a valuation needs anyway.
    out = out.with_columns(
        (c("total_debt") - c("cash_and_equivalents")).alias("net_debt"),
        (c("operating_income") + c("depreciation_amortization").fill_null(0.0)).alias("ebitda"),
        c("operating_income").alias("ebit"),
        # Owner free cash flow charges share-based compensation, which the cash
        # statement adds back. It is not a cost to the company's bank account
        # but it is unambiguously a cost to the existing owners.
        (
            c("operating_cash_flow") - c("capex") - c("share_based_comp").fill_null(0.0)
        ).alias("free_cash_flow_ex_sbc"),
        safe_div(c("tax_expense"), c("pretax_income"), positive_only=True)
        .clip(dcf.tax_rate_floor, dcf.tax_rate_cap)
        .alias("effective_tax_rate"),
    )

    # --- market capitalisation and enterprise value ------------------------
    # Diluted weighted-average shares lead, because the cover-page count is
    # absent for dual-class filers: it is reported per class and the SEC's
    # aggregate interface only carries undimensioned facts.
    out = out.with_columns(
        pl.coalesce(c("shares_diluted"), c("shares_outstanding"), c("shares_basic")).alias(
            "shares_reported"
        )
    ).with_columns(
        (c("shares_reported") * c("shares_split_factor")).alias("shares_current_basis")
    ).with_columns(
        # close_raw is split-adjusted onto today's basis, and the share count has
        # now been restated onto the same basis, so the product is a market
        # capitalisation in today's terms.
        safe_div(
            c("close_raw") * c("shares_current_basis"), pl.lit(1.0)
        ).alias("market_cap")
    )

    out = out.with_columns(
        pl.coalesce(
            safe_div(
                pl.coalesce(c("net_income_to_common"), c("net_income")),
                c("shares_current_basis"),
                positive_only=True,
            ),
            # Only where no share count exists at all does the filer's own
            # reported figure stand in, and it is flagged by its absence of a
            # share basis rather than silently blended.
            c("eps_diluted"),
        ).alias("eps_diluted")
    )

    out = out.with_columns(
        (
            c("market_cap")
            + c("total_debt")
            + c("minority_interest").fill_null(0.0)
            + c("preferred_equity").fill_null(0.0)
            + c("operating_lease_liability").fill_null(0.0)
            - c("cash_and_equivalents")
        ).alias("enterprise_value")
    )

    # --- invested capital --------------------------------------------------
    # Equity plus debt less surplus cash: the capital the operating business
    # actually employs. Netting cash matters for the cash-rich technology
    # companies that would otherwise look far less profitable than they are.
    out = out.with_columns(
        (
            pl.coalesce(c("equity_incl_nci"), c("equity"))
            + c("total_debt")
            - c("cash_and_equivalents")
        ).alias("invested_capital"),
        (c("operating_income") * (1.0 - c("effective_tax_rate").fill_null(0.21))).alias(
            "nopat"
        ),
    )
    return out


def _lagged(panel: pl.DataFrame, columns: list[str], years: int) -> pl.DataFrame:
    """Values as they were knowable `years` ago, joined onto each row.

    Shifting the *dates* forward and as-of joining keeps both sides of a growth
    ratio point-in-time, and tolerates gaps in a company's history where a
    fixed row offset would silently compare across a listing gap.
    """
    lag_days = 365 * years
    sub = (
        panel.select(["company_id", "date", *columns])
        .with_columns((pl.col("date") + timedelta(days=lag_days)).alias("date"))
        .rename({col: f"{col}__lag{years}y" for col in columns})
        .sort("date")
    )
    return (
        panel.sort("date")
        .join_asof(sub, on="date", by="company_id", strategy="backward")
        .sort(["company_id", "date"])
    )


def _cagr(now: pl.Expr, then: pl.Expr, years: int) -> pl.Expr:
    """Annualised growth, defined only where both endpoints are positive.

    A compound growth rate from a negative base is not a growth rate; reporting
    one produces confident nonsense like "earnings grew 340%" for a company that
    went from a large loss to a small one.
    """
    return (
        pl.when((now > EPS) & (then > EPS))
        .then((now / then) ** (1.0 / years) - 1.0)
        .otherwise(None)
    )


def add_quality(panel: pl.DataFrame) -> pl.DataFrame:
    c = pl.col
    out = panel.with_columns(
        safe_div(c("nopat"), c("invested_capital"), positive_only=True).alias("roic"),
        safe_div(c("net_income"), c("equity"), positive_only=True).alias("roe"),
        safe_div(c("gross_profit"), c("revenue"), positive_only=True).alias("gross_margin"),
        safe_div(c("operating_income"), c("revenue"), positive_only=True).alias(
            "operating_margin"
        ),
        safe_div(c("free_cash_flow"), c("revenue"), positive_only=True).alias("fcf_margin"),
        safe_div(c("free_cash_flow"), c("net_income"), positive_only=True).alias(
            "fcf_conversion"
        ),
        safe_div(c("operating_income"), c("interest_expense"), positive_only=True).alias(
            "interest_coverage"
        ),
        safe_div(c("net_debt"), c("ebitda"), positive_only=True).alias("net_debt_to_ebitda"),
        safe_div(c("current_assets"), c("current_liabilities"), positive_only=True).alias(
            "current_ratio"
        ),
        safe_div(
            pl.coalesce(c("equity_incl_nci"), c("equity")), c("assets"), positive_only=True
        ).alias("equity_to_assets"),
    )

    # Accruals: the wedge between reported profit and the cash behind it.
    out = out.with_columns(
        safe_div(c("net_income") - c("operating_cash_flow"), c("assets"), positive_only=True)
        .alias("accrual_ratio")
    )

    # Margin stability over five years. Reported as a positive score where a
    # steadier margin is better, so the direction in the spec stays +1.
    #
    # The window is expressed in *time*, not rows, so the same code is correct
    # whether the panel is sampled daily or weekly. A row-count window silently
    # becomes a three-month lookback the moment the grid changes cadence.
    out = out.sort(["company_id", "date"]).with_columns(
        c("operating_margin")
        .rolling_std_by("date", "1825d", min_samples=8)
        .over("company_id")
        .alias("_margin_std")
    ).with_columns(
        pl.when(c("_margin_std").is_not_null())
        .then(1.0 / (1.0 + c("_margin_std") * 10.0))
        .otherwise(None)
        .alias("margin_stability")
    )
    return out


def add_growth(panel: pl.DataFrame) -> pl.DataFrame:
    """Growth rates over one, three and five years, all point-in-time."""
    # Everything a downstream check needs a year-ago value for: the growth
    # metrics themselves, plus the inputs to the Piotroski tests, the
    # falling-returns trap signal and the unprofitable-growth reclassification.
    growth_cols = [
        "revenue", "eps_diluted", "free_cash_flow", "gross_profit",
        "shares_current_basis", "operating_margin",
        "net_income", "assets", "total_debt", "current_ratio",
        "gross_margin", "roic",
    ]
    out = panel
    for years in (1, 3, 5):
        present = [col for col in growth_cols if col in out.columns]
        out = _lagged(out, present, years)

    c = pl.col
    out = out.with_columns(
        safe_div(
            c("revenue") - c("revenue__lag1y"), c("revenue__lag1y").abs()
        ).alias("revenue_growth_1y"),
        _cagr(c("revenue"), c("revenue__lag3y"), 3).alias("revenue_growth_3y"),
        _cagr(c("revenue"), c("revenue__lag5y"), 5).alias("revenue_growth_5y"),
        safe_div(
            c("eps_diluted") - c("eps_diluted__lag1y"), c("eps_diluted__lag1y").abs()
        ).alias("eps_growth_1y"),
        _cagr(c("eps_diluted"), c("eps_diluted__lag3y"), 3).alias("eps_growth_3y"),
        _cagr(c("free_cash_flow"), c("free_cash_flow__lag3y"), 3).alias("fcf_growth_3y"),
        _cagr(c("gross_profit"), c("gross_profit__lag3y"), 3).alias("gross_profit_growth_3y"),
        # Negative means dilution, positive means the count shrank: buybacks
        # score well, serial issuance scores badly.
        safe_div(
            c("shares_current_basis__lag1y") - c("shares_current_basis"),
            c("shares_current_basis__lag1y").abs(),
        ).alias("share_count_change"),
        (c("operating_margin") - c("operating_margin__lag1y")).alias("margin_trend"),
    )

    out = out.with_columns(
        (c("revenue_growth_1y") - c("revenue_growth_3y")).alias("growth_acceleration"),
        # How much of the growth was bought rather than earned. Used by the
        # value-trap and confidence layers, not scored directly.
        safe_div(c("acquisitions"), c("revenue"), positive_only=True).alias(
            "acquisition_intensity"
        ),
    )
    return out


def add_valuation(panel: pl.DataFrame) -> pl.DataFrame:
    c = pl.col
    out = panel.with_columns(
        safe_div(c("enterprise_value"), c("ebit"), positive_only=True).alias("ev_ebit"),
        safe_div(c("enterprise_value"), c("ebitda"), positive_only=True).alias("ev_ebitda"),
        safe_div(c("enterprise_value"), c("revenue"), positive_only=True).alias("ev_revenue"),
        safe_div(c("free_cash_flow"), c("market_cap"), positive_only=True).alias("fcf_yield"),
        safe_div(c("net_income"), c("market_cap"), positive_only=True).alias("earnings_yield"),
        safe_div(c("market_cap"), pl.coalesce(c("equity_incl_nci"), c("equity")), positive_only=True)
        .alias("price_to_book"),
        safe_div(c("revenue"), c("market_cap"), positive_only=True).alias("sales_yield"),
        # Cash actually returned to owners, net of new stock issued.
        safe_div(
            c("dividends_paid").fill_null(0.0)
            + c("buyback").fill_null(0.0)
            - c("stock_issued").fill_null(0.0),
            c("market_cap"),
            positive_only=True,
        ).alias("shareholder_yield"),
        safe_div(c("dividends_paid"), c("market_cap"), positive_only=True).alias(
            "dividend_yield"
        ),
        safe_div(c("capex"), c("revenue"), positive_only=True).alias("capex_intensity"),
        safe_div(c("rnd"), c("revenue"), positive_only=True).alias("rnd_intensity"),
        safe_div(c("close_raw"), c("eps_diluted"), positive_only=True).alias("trailing_pe"),
    )

    # Valuation against the company's own five-year history. Expressed so that
    # cheap-relative-to-itself scores high, and computed on whichever of the two
    # primary multiples the company actually supports.
    out = out.sort(["company_id", "date"]).with_columns(
        pl.coalesce(c("ev_ebit"), c("ev_ebitda"), c("trailing_pe")).alias("_primary_multiple")
    )
    out = out.with_columns(
        c("_primary_multiple")
        .rolling_median_by("date", "1825d", min_samples=20)
        .over("company_id")
        .alias("_multiple_median"),
        c("_primary_multiple")
        .rolling_std_by("date", "1825d", min_samples=20)
        .over("company_id")
        .alias("_multiple_std"),
    ).with_columns(
        # Negative z-score: trading below its own history is a positive signal.
        pl.when(c("_multiple_std") > EPS)
        .then(-(c("_primary_multiple") - c("_multiple_median")) / c("_multiple_std"))
        .otherwise(None)
        .clip(-3.0, 3.0)
        .alias("valuation_vs_history")
    )
    return out


def build_financial_metrics(
    grid: pl.DataFrame,
    fundamentals: pl.DataFrame,
    prices: pl.DataFrame,
) -> pl.DataFrame:
    """The full financial metric panel for a company-date grid."""
    panel = attach_fundamentals(grid, fundamentals)
    panel = attach_share_basis(panel, prices)
    panel = add_building_blocks(panel)
    panel = add_quality(panel)
    panel = add_growth(panel)
    panel = add_valuation(panel)

    drop = [c for c in panel.columns if c.startswith("_") and not c.startswith("__")]
    panel = panel.drop([c for c in drop if c in panel.columns])
    log.info(
        "financial metrics: %d rows, %d companies, %d columns",
        panel.height, panel["company_id"].n_unique(), panel.width,
    )
    return panel
