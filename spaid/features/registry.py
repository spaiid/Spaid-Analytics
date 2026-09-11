"""The feature catalogue.

One declarative list drives everything downstream: which columns get built,
how they are grouped in the UI, and which direction counts as "good". Keeping
it declarative means the dashboard never hard-codes the model's internals -- it
just renders whatever is registered here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    group: str
    direction: int = 1      # +1: higher is better, -1: lower is better
    description: str = ""


FEATURES: tuple[Feature, ...] = (
    # ---- Momentum -------------------------------------------------------
    Feature("mom_12_1", "12-1 month momentum", "Momentum", 1,
            "Return over the past year excluding the most recent month, the classic momentum construction."),
    Feature("mom_6_1", "6-1 month momentum", "Momentum", 1,
            "Six-month return excluding the most recent month."),
    Feature("ret_63d", "3-month return", "Momentum", 1, "Trailing three-month total return."),
    Feature("resid_mom_126d", "Residual momentum", "Momentum", 1,
            "Six-month momentum after removing the market-beta component, so it is not just leveraged market exposure."),
    Feature("ret_5d", "1-week reversal", "Momentum", -1,
            "Very short-term returns tend to mean-revert, so a strong week is a mild negative."),

    # ---- Trend ----------------------------------------------------------
    Feature("px_over_ma50", "Price vs 50-day average", "Trend", 1, "Distance above the 50-day moving average."),
    Feature("px_over_ma200", "Price vs 200-day average", "Trend", 1, "Distance above the 200-day moving average."),
    Feature("ma_50_200", "50 vs 200-day average", "Trend", 1, "Golden-cross style trend confirmation."),
    Feature("near_52w_high", "Proximity to 52-week high", "Trend", 1,
            "How close the price sits to its one-year high; names near highs have historically kept running."),
    Feature("up_day_ratio", "Up-day consistency", "Trend", 1,
            "Share of the last 63 sessions that closed higher, a steadiness measure rather than a size-of-move measure."),

    # ---- Volatility -----------------------------------------------------
    Feature("vol_63d", "Realised volatility", "Volatility", -1,
            "Annualised 63-day volatility. Lower-volatility names have historically earned better risk-adjusted returns."),
    Feature("downside_vol_126d", "Downside volatility", "Volatility", -1,
            "Volatility computed on negative days only."),
    Feature("beta_252d", "Market beta", "Volatility", -1, "One-year beta to the S&P 500."),
    Feature("idio_vol_126d", "Idiosyncratic volatility", "Volatility", -1,
            "Volatility of the part of returns the market does not explain."),
    Feature("max_drawdown_252d", "One-year drawdown", "Volatility", 1,
            "Worst peak-to-trough fall over the past year (less negative is better)."),

    # ---- Liquidity ------------------------------------------------------
    Feature("log_dollar_volume", "Dollar volume", "Liquidity", 1, "Log of median daily traded value."),
    Feature("amihud_illiq", "Amihud illiquidity", "Liquidity", -1,
            "Price impact per dollar traded; high values mean the name is expensive to move."),
    Feature("volume_trend", "Volume trend", "Liquidity", 1,
            "Recent volume against its own baseline, a crude interest proxy."),

    # ---- Value ----------------------------------------------------------
    Feature("earnings_yield", "Earnings yield", "Value", 1, "Trailing twelve-month earnings over market cap."),
    Feature("sales_yield", "Sales yield", "Value", 1, "TTM revenue over market cap."),
    Feature("fcf_yield", "Free cash flow yield", "Value", 1,
            "Operating cash flow minus capital expenditure, over market cap."),
    Feature("book_to_price", "Book to price", "Value", 1, "Shareholders' equity over market cap."),

    # ---- Quality --------------------------------------------------------
    Feature("roe", "Return on equity", "Quality", 1, "TTM earnings over shareholders' equity."),
    Feature("roa", "Return on assets", "Quality", 1, "TTM earnings over total assets."),
    Feature("gross_margin", "Gross margin", "Quality", 1, "Gross profit over revenue."),
    Feature("operating_margin", "Operating margin", "Quality", 1, "Operating income over revenue."),
    Feature("accruals", "Accruals", "Quality", -1,
            "Earnings not backed by cash flow. High accruals have historically preceded disappointment."),
    Feature("leverage", "Leverage", "Quality", -1, "Total debt over equity."),
    Feature("current_ratio", "Current ratio", "Quality", 1, "Current assets over current liabilities."),

    # ---- Growth ---------------------------------------------------------
    Feature("revenue_growth", "Revenue growth", "Growth", 1, "TTM revenue against the same figure a year earlier."),
    Feature("earnings_growth", "Earnings growth", "Growth", 1, "TTM earnings against a year earlier."),
    Feature("margin_trend", "Margin trend", "Growth", 1, "Change in operating margin over the past year."),
    Feature("fcf_growth", "Cash flow growth", "Growth", 1, "TTM free cash flow against a year earlier."),

    # ---- Capital use ----------------------------------------------------
    Feature("buyback_yield", "Buyback yield", "Capital use", 1, "Shares repurchased over market cap."),
    Feature("dividend_yield", "Dividend yield", "Capital use", 1, "Dividends paid over market cap."),
    Feature("capex_intensity", "Capex intensity", "Capital use", -1,
            "Capital expenditure over revenue; capital-hungry businesses convert less to shareholders."),
    Feature("rnd_intensity", "R&D intensity", "Capital use", 1, "Research spend over revenue."),
)

BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}
NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)


def groups() -> tuple[str, ...]:
    seen: list[str] = []
    for f in FEATURES:
        if f.group not in seen:
            seen.append(f.group)
    return tuple(seen)


def in_group(group: str) -> tuple[Feature, ...]:
    return tuple(f for f in FEATURES if f.group == group)
