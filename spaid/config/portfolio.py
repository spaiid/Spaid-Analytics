"""Portfolio construction defaults and constraints.

These are the knobs the product brief asks to be configurable rather than buried
in code: how many names, how large a position, how much cash, how much
concentration, and how much expected improvement a trade must offer before it is
worth making.

The turnover discipline deserves a note. A recommendation engine that re-ranks
daily will suggest a trade every day, and the accumulated costs and taxes of
following it will exceed any edge the ranking has. `min_score_improvement` and
`min_trade_value` exist to make the engine stay quiet unless it has something
material to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AccountType(StrEnum):
    TAXABLE = "taxable"
    TRADITIONAL_IRA = "traditional_ira"
    ROTH_IRA = "roth_ira"
    OTHER_TAX_DEFERRED = "other_tax_deferred"

    @property
    def is_taxable(self) -> bool:
        return self is AccountType.TAXABLE


class RiskTolerance(StrEnum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ActionType(StrEnum):
    BUY = "buy"
    ADD = "add"
    HOLD = "hold"
    TRIM = "trim"
    SELL = "sell"
    BUY_BENCHMARK = "buy_benchmark"  # SPY instead of an individual name
    HOLD_CASH = "hold_cash"
    DO_NOTHING = "do_nothing"


@dataclass(frozen=True)
class RiskToleranceProfile:
    """How risk tolerance translates into concrete limits."""

    max_positions: int
    max_position_weight: float
    max_sector_weight: float
    min_cash_weight: float
    target_portfolio_beta: float
    # Names must clear this composite score to be bought at all.
    min_score_to_buy: float
    # Below this score an existing holding is a sell candidate.
    sell_score_threshold: float
    # How much a low confidence score shrinks a position, at most.
    max_confidence_haircut: float


RISK_PROFILES: dict[RiskTolerance, RiskToleranceProfile] = {
    RiskTolerance.CONSERVATIVE: RiskToleranceProfile(
        max_positions=25,
        max_position_weight=0.06,
        max_sector_weight=0.25,
        min_cash_weight=0.10,
        target_portfolio_beta=0.90,
        min_score_to_buy=70.0,
        sell_score_threshold=35.0,
        max_confidence_haircut=0.60,
    ),
    RiskTolerance.MODERATE: RiskToleranceProfile(
        max_positions=20,
        max_position_weight=0.08,
        max_sector_weight=0.30,
        min_cash_weight=0.05,
        target_portfolio_beta=1.00,
        min_score_to_buy=65.0,
        sell_score_threshold=32.0,
        max_confidence_haircut=0.50,
    ),
    RiskTolerance.AGGRESSIVE: RiskToleranceProfile(
        max_positions=15,
        max_position_weight=0.12,
        max_sector_weight=0.40,
        min_cash_weight=0.02,
        target_portfolio_beta=1.15,
        min_score_to_buy=60.0,
        sell_score_threshold=30.0,
        max_confidence_haircut=0.40,
    ),
}


@dataclass(frozen=True)
class CostSpec:
    """What a trade actually costs.

    Commission is zero at most retail brokers, but spread and slippage are not,
    and a backtest that ignores them overstates every result. These are the
    numbers used both live and in the backtester, deliberately the same object
    so the two cannot drift apart.
    """

    commission_per_trade: float = 0.0
    # Half-spread paid on entry and exit, in basis points, for a large-cap name.
    spread_bps_large_cap: float = 2.0
    spread_bps_small_cap: float = 8.0
    large_cap_threshold: float = 50_000_000_000.0
    # Extra slippage per unit of participation in daily volume.
    slippage_bps_base: float = 1.0
    # Execution happens at the next open after a signal, not at the close that
    # generated it.
    execution_delay_days: int = 1


@dataclass(frozen=True)
class TaxSpec:
    """Tax assumptions for taxable accounts.

    Used to estimate the drag of a proposed sale, never to give tax advice. Rates
    are configurable because they are personal.
    """

    short_term_rate: float = 0.35
    long_term_rate: float = 0.15
    long_term_holding_days: int = 366
    # A sale must improve the portfolio by more than the tax it triggers, times
    # this factor, before it is recommended.
    tax_hurdle_multiple: float = 1.0
    wash_sale_days: int = 31


@dataclass(frozen=True)
class TurnoverSpec:
    """The discipline that stops the engine from fidgeting."""

    # A buy must raise the portfolio's expected score by at least this much
    # (in composite-score points, position-weighted) to be worth doing.
    min_score_improvement: float = 4.0
    # Never propose a trade smaller than this, in dollars: the spread eats it.
    min_trade_value: float = 500.0
    # Nor one smaller than this share of the portfolio.
    min_trade_weight: float = 0.005
    # Do not trim a position unless it is at least this far above its target.
    rebalance_band: float = 0.25  # 25% relative deviation from target weight
    # Minimum days between trades in the same name, absent an invalidation.
    min_holding_days: int = 30
    # Target maximum annual turnover; the engine warns above this.
    max_annual_turnover: float = 1.0


@dataclass(frozen=True)
class PortfolioSpec:
    version: str
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    account_type: AccountType = AccountType.TAXABLE
    costs: CostSpec = field(default_factory=CostSpec)
    taxes: TaxSpec = field(default_factory=TaxSpec)
    turnover: TurnoverSpec = field(default_factory=TurnoverSpec)
    # Correlation above which two holdings are treated as substantially the same
    # bet for concentration purposes.
    high_correlation: float = 0.75
    # Confidence below which a name may be scored but not bought.
    min_confidence_to_buy: float = 0.40
    # When nothing clears the bar, this is what the engine recommends instead.
    default_fallback: ActionType = ActionType.BUY_BENCHMARK
    notes: str = ""

    @property
    def limits(self) -> RiskToleranceProfile:
        return RISK_PROFILES[self.risk_tolerance]


PORTFOLIO_V1 = PortfolioSpec(
    version="portfolio-2026.09.1",
    notes=(
        "Initial defaults: moderate risk tolerance, taxable account, 20 positions, 8% maximum "
        "position, 30% maximum sector, 5% minimum cash. Every limit is overridable per account."
    ),
)

ACTIVE_PORTFOLIO_SPEC = PORTFOLIO_V1
