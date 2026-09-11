"""The scoring specification: what the composite opportunity score *means*.

This module is the single source of truth for how a 0-100 score is built. It is
versioned because a score is only comparable to another score computed under the
same spec -- the research journal stores `spec_version` with every
recommendation so a historical call can always be reproduced exactly.

Two rules the mission imposes and this module enforces:

1. **Category weights are fixed and declared, never fitted.** Quality 30,
   Growth 25, Momentum 25, Valuation 20. They are configurable here; they are
   never optimised against historical returns anywhere in the codebase.
2. **Missing is not zero.** Every metric carries `required_for_category`
   coverage rules. A metric with no data contributes nothing and shrinks the
   category's evidence base rather than scoring as average.

Metric weights *within* a category are also declared, not fitted. They encode
how much each measurement is trusted as a proxy for the category, which is a
modelling judgement stated openly rather than a regression result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Category(StrEnum):
    QUALITY = "quality"
    GROWTH = "growth"
    MOMENTUM = "momentum"
    VALUATION = "valuation"


class BusinessModel(StrEnum):
    """How a company must be analysed.

    Forcing one valuation and one quality template onto every company is the
    fastest way to produce confident nonsense: a bank has no gross margin, a
    REIT's earnings are meaningless next to its AFFO, and a miner's trailing
    profits say more about the commodity cycle than about the business.
    """

    OPERATING = "operating"  # ordinary profitable non-financial company
    BANK = "bank"
    INSURER = "insurer"
    REIT = "reit"
    CYCLICAL = "cyclical"  # commodity / deeply cyclical earnings
    UNPROFITABLE_GROWTH = "unprofitable_growth"


ALL_MODELS: frozenset[BusinessModel] = frozenset(BusinessModel)
NON_FINANCIAL: frozenset[BusinessModel] = frozenset(
    {
        BusinessModel.OPERATING,
        BusinessModel.CYCLICAL,
        BusinessModel.UNPROFITABLE_GROWTH,
    }
)


class PeerBasis(StrEnum):
    """Which population a metric is percentile-ranked against.

    Comparing a utility's margin to a software company's is meaningless, so
    profitability and valuation are ranked inside the industry or sector. Price
    behaviour is a market-wide phenomenon and is ranked against the whole
    universe, with a separate explicit sector-relative-strength metric carrying
    the sector comparison.
    """

    UNIVERSE = "universe"
    SECTOR = "sector"
    INDUSTRY = "industry"  # falls back to sector when the industry is too thin
    SIZE_DECILE = "size_decile"


@dataclass(frozen=True)
class MetricSpec:
    """One measurable input to the composite score."""

    key: str
    label: str
    category: Category
    direction: int  # +1 higher is better, -1 lower is better
    weight: float  # relative weight within its category
    peer_basis: PeerBasis
    description: str
    # Which business models this metric is meaningful for. A metric not
    # applicable to a company is *absent*, not zero, and the category's
    # remaining weights are renormalised over what applies.
    applies_to: frozenset[BusinessModel] = ALL_MODELS
    # Tail clipping before ranking, as a fraction per side. Percentile ranking is
    # already robust to outliers; winsorising matters for the raw values shown in
    # the UI and for any mean-based statistic computed downstream.
    winsor: float = 0.02
    # Minimum peer-group size for a percentile to be trustworthy. Below this the
    # metric falls back to the next-broader peer group.
    min_peers: int = 8
    unit: str = "ratio"  # ratio | percent | currency | count | score | years


# --------------------------------------------------------------------------
# Quality  (30%)
# --------------------------------------------------------------------------
# Deliberately balanced between returns on capital, margin level, margin
# *stability*, cash conversion, accrual quality and balance-sheet strength.
# Leverage-driven returns are penalised twice: ROIC is weighted above ROE, and
# net-debt / balance-sheet metrics carry real weight, so a company cannot buy a
# high quality score with borrowed money.
QUALITY_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "roic", "Return on invested capital", Category.QUALITY, 1, 0.16, PeerBasis.INDUSTRY,
        "After-tax operating profit over invested capital. The cleanest single measure of "
        "whether the business earns more than its capital costs.",
        applies_to=NON_FINANCIAL,
    ),
    MetricSpec(
        "roe", "Return on equity", Category.QUALITY, 1, 0.10, PeerBasis.INDUSTRY,
        "Trailing earnings over shareholders' equity. Weighted below ROIC because leverage "
        "inflates it; the leverage metrics below are the counterweight.",
    ),
    MetricSpec(
        "gross_margin", "Gross margin", Category.QUALITY, 1, 0.08, PeerBasis.INDUSTRY,
        "Gross profit over revenue. A durable pricing-power signal in industries where it is "
        "reported meaningfully.",
        applies_to=NON_FINANCIAL,
    ),
    MetricSpec(
        "operating_margin", "Operating margin", Category.QUALITY, 1, 0.10, PeerBasis.INDUSTRY,
        "Operating income over revenue.",
    ),
    MetricSpec(
        "fcf_margin", "Free-cash-flow margin", Category.QUALITY, 1, 0.10, PeerBasis.INDUSTRY,
        "Free cash flow over revenue. Cash is harder to manufacture than accounting profit.",
    ),
    MetricSpec(
        "margin_stability", "Margin stability", Category.QUALITY, 1, 0.08, PeerBasis.INDUSTRY,
        "Inverse of the standard deviation of operating margin over the past five years. "
        "Predictable profitability is worth more than the same average delivered erratically.",
    ),
    MetricSpec(
        "fcf_conversion", "Free-cash-flow conversion", Category.QUALITY, 1, 0.08, PeerBasis.INDUSTRY,
        "Free cash flow over net income. Persistently below one means reported profit is not "
        "turning into spendable cash.",
    ),
    MetricSpec(
        "accrual_ratio", "Accrual quality", Category.QUALITY, -1, 0.07, PeerBasis.INDUSTRY,
        "(Net income minus operating cash flow) over average assets. High accruals have "
        "historically preceded disappointment and restatement.",
    ),
    MetricSpec(
        "interest_coverage", "Interest coverage", Category.QUALITY, 1, 0.07, PeerBasis.INDUSTRY,
        "Operating income over interest expense. How much room the business has before debt "
        "service threatens it.",
        applies_to=NON_FINANCIAL,
    ),
    MetricSpec(
        "net_debt_to_ebitda", "Net debt to EBITDA", Category.QUALITY, -1, 0.08, PeerBasis.INDUSTRY,
        "Net debt over trailing EBITDA. The standard leverage yardstick for operating companies.",
        applies_to=NON_FINANCIAL,
    ),
    MetricSpec(
        "share_count_change", "Share-count change", Category.QUALITY, -1, 0.08, PeerBasis.UNIVERSE,
        "Year-over-year change in diluted shares outstanding. Persistent dilution transfers value "
        "away from existing holders; genuine buybacks return it.",
    ),
)

# --------------------------------------------------------------------------
# Growth  (25%)
# --------------------------------------------------------------------------
# Split between level (how fast), durability (multi-year, and is it
# accelerating), and forward evidence (estimate revisions, surprises). Revisions
# and surprises carry real weight because they are the part of growth that is
# forward-looking rather than a description of the past.
GROWTH_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "revenue_growth_1y", "Revenue growth (1y)", Category.GROWTH, 1, 0.10, PeerBasis.INDUSTRY,
        "Trailing-twelve-month revenue against the same figure a year earlier.",
    ),
    MetricSpec(
        "revenue_growth_3y", "Revenue growth (3y)", Category.GROWTH, 1, 0.10, PeerBasis.INDUSTRY,
        "Three-year annualised revenue growth. Durability rather than a single good year.",
    ),
    MetricSpec(
        "revenue_growth_5y", "Revenue growth (5y)", Category.GROWTH, 1, 0.06, PeerBasis.INDUSTRY,
        "Five-year annualised revenue growth.",
    ),
    MetricSpec(
        "eps_growth_1y", "Earnings growth (1y)", Category.GROWTH, 1, 0.09, PeerBasis.INDUSTRY,
        "Trailing diluted earnings per share against a year earlier. Per-share so that growth "
        "bought with issued stock does not count.",
    ),
    MetricSpec(
        "eps_growth_3y", "Earnings growth (3y)", Category.GROWTH, 1, 0.09, PeerBasis.INDUSTRY,
        "Three-year annualised earnings-per-share growth.",
    ),
    MetricSpec(
        "fcf_growth_3y", "Cash-flow growth (3y)", Category.GROWTH, 1, 0.08, PeerBasis.INDUSTRY,
        "Three-year annualised free-cash-flow growth.",
    ),
    MetricSpec(
        "gross_profit_growth_3y", "Gross-profit growth (3y)", Category.GROWTH, 1, 0.06,
        PeerBasis.INDUSTRY,
        "Three-year annualised gross-profit growth. Less easily flattered by mix than revenue.",
        applies_to=NON_FINANCIAL,
    ),
    MetricSpec(
        "growth_acceleration", "Growth acceleration", Category.GROWTH, 1, 0.08, PeerBasis.INDUSTRY,
        "One-year revenue growth minus the three-year rate. Positive means the business is "
        "speeding up, negative that a good multi-year record is already fading.",
    ),
    MetricSpec(
        "eps_revision_3m", "Earnings estimate revisions", Category.GROWTH, 1, 0.12,
        PeerBasis.UNIVERSE,
        "Net direction of analyst earnings-estimate changes over the past three months. The most "
        "reliably forward-looking growth input available without a paid data feed.",
    ),
    MetricSpec(
        "revenue_revision_3m", "Revenue estimate revisions", Category.GROWTH, 1, 0.06,
        PeerBasis.UNIVERSE,
        "Net direction of analyst revenue-estimate changes over the past three months.",
    ),
    MetricSpec(
        "earnings_surprise", "Earnings-surprise history", Category.GROWTH, 1, 0.08,
        PeerBasis.UNIVERSE,
        "Average percentage earnings surprise over the last four reported quarters.",
    ),
    MetricSpec(
        "forward_eps_growth", "Forward earnings growth", Category.GROWTH, 1, 0.08,
        PeerBasis.INDUSTRY,
        "Consensus next-fiscal-year earnings growth, used only where enough analysts cover the "
        "name for the consensus to mean anything.",
    ),
)

# --------------------------------------------------------------------------
# Momentum  (25%)
# --------------------------------------------------------------------------
# Classic price momentum plus trend confirmation and consistency. The mission
# requires that momentum never independently override poor fundamentals or
# extreme valuation; that guard lives in the recommendation layer, not here,
# because the score itself should faithfully report what momentum says.
MOMENTUM_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "mom_12_1", "12-month momentum (ex last month)", Category.MOMENTUM, 1, 0.18,
        PeerBasis.UNIVERSE,
        "Return over the past year excluding the most recent month. The standard construction: "
        "the skipped month removes short-term reversal.",
    ),
    MetricSpec(
        "mom_6_1", "6-month momentum (ex last month)", Category.MOMENTUM, 1, 0.12,
        PeerBasis.UNIVERSE,
        "Six-month return excluding the most recent month.",
    ),
    MetricSpec(
        "ret_3m", "3-month return", Category.MOMENTUM, 1, 0.08, PeerBasis.UNIVERSE,
        "Trailing three-month total return.",
    ),
    MetricSpec(
        "rel_strength_spy", "Relative strength vs SPY", Category.MOMENTUM, 1, 0.12,
        PeerBasis.UNIVERSE,
        "Six-month return minus the S&P 500's. Isolates the part of the move that is not simply "
        "the market rising.",
    ),
    MetricSpec(
        "rel_strength_sector", "Relative strength vs sector", Category.MOMENTUM, 1, 0.10,
        PeerBasis.UNIVERSE,
        "Six-month return minus the company's sector ETF. Distinguishes a good company from a "
        "good sector.",
    ),
    MetricSpec(
        "px_over_ma200", "Price vs 200-day average", Category.MOMENTUM, 1, 0.08,
        PeerBasis.UNIVERSE,
        "Distance above the 200-day moving average.",
    ),
    MetricSpec(
        "ma200_slope", "200-day average direction", Category.MOMENTUM, 1, 0.08,
        PeerBasis.UNIVERSE,
        "Three-month change in the 200-day moving average. A rising long-term average is a "
        "different state from a falling one at the same price.",
    ),
    MetricSpec(
        "momentum_consistency", "Momentum consistency", Category.MOMENTUM, 1, 0.08,
        PeerBasis.UNIVERSE,
        "Share of the past twelve months in which the stock beat the market. Steady outperformance "
        "rather than one explosive month.",
    ),
    MetricSpec(
        "near_52w_high", "Proximity to 52-week high", Category.MOMENTUM, 1, 0.08,
        PeerBasis.UNIVERSE,
        "Current price relative to the one-year high.",
    ),
    MetricSpec(
        "post_earnings_drift", "Post-earnings price behaviour", Category.MOMENTUM, 1, 0.08,
        PeerBasis.UNIVERSE,
        "Cumulative market-relative return over the three days around the last earnings release. "
        "How the market actually received the most recent news.",
    ),
)

# --------------------------------------------------------------------------
# Valuation  (20%)
# --------------------------------------------------------------------------
# Multiple independent lenses, each ranked inside the industry, plus two
# cross-checks that a low multiple alone cannot satisfy: valuation against the
# company's *own* history, and growth-adjusted valuation.
VALUATION_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "ev_ebit", "EV / EBIT", Category.VALUATION, -1, 0.12, PeerBasis.INDUSTRY,
        "Enterprise value over operating profit. Capital-structure neutral, so it compares a "
        "debt-funded company with an equity-funded one fairly.",
        applies_to=NON_FINANCIAL,
    ),
    MetricSpec(
        "ev_ebitda", "EV / EBITDA", Category.VALUATION, -1, 0.10, PeerBasis.INDUSTRY,
        "Enterprise value over EBITDA.",
        applies_to=NON_FINANCIAL,
    ),
    MetricSpec(
        "fcf_yield", "Free-cash-flow yield", Category.VALUATION, 1, 0.14, PeerBasis.INDUSTRY,
        "Free cash flow over market capitalisation. What the business generates for owners "
        "relative to what they pay.",
    ),
    MetricSpec(
        "earnings_yield", "Earnings yield", Category.VALUATION, 1, 0.10, PeerBasis.INDUSTRY,
        "Trailing earnings over market capitalisation, the inverse of the price/earnings ratio.",
    ),
    MetricSpec(
        "forward_pe", "Forward P/E", Category.VALUATION, -1, 0.10, PeerBasis.INDUSTRY,
        "Price over consensus next-year earnings, where coverage is sufficient.",
    ),
    MetricSpec(
        "ev_revenue", "EV / revenue", Category.VALUATION, -1, 0.06, PeerBasis.INDUSTRY,
        "Enterprise value over revenue. Carries most of the weight for companies with no "
        "meaningful earnings yet.",
        applies_to=NON_FINANCIAL,
    ),
    MetricSpec(
        "peg", "Growth-adjusted valuation", Category.VALUATION, -1, 0.10, PeerBasis.INDUSTRY,
        "Forward price/earnings divided by expected growth. Stops the score from calling every "
        "fast-growing company expensive and every shrinking one cheap.",
    ),
    MetricSpec(
        "valuation_vs_history", "Valuation vs own history", Category.VALUATION, 1, 0.14,
        PeerBasis.UNIVERSE,
        "Where today's primary multiple sits inside the company's own five-year range, expressed "
        "so that cheap-against-itself scores high. The single best guard against mistaking a "
        "structurally low-multiple business for a bargain.",
    ),
    MetricSpec(
        "shareholder_yield", "Shareholder yield", Category.VALUATION, 1, 0.08, PeerBasis.INDUSTRY,
        "Dividends plus net buybacks over market capitalisation. Cash actually returned, rather "
        "than cash theoretically available.",
    ),
    MetricSpec(
        "price_to_book", "Price to book", Category.VALUATION, -1, 0.06, PeerBasis.INDUSTRY,
        "Market capitalisation over book equity. Weighted low for operating companies but the "
        "primary lens for banks, where the industry peer ranking gives it force.",
    ),
)

ALL_METRICS: tuple[MetricSpec, ...] = (
    QUALITY_METRICS + GROWTH_METRICS + MOMENTUM_METRICS + VALUATION_METRICS
)
METRIC_BY_KEY: dict[str, MetricSpec] = {m.key: m for m in ALL_METRICS}


@dataclass(frozen=True)
class ScoringSpec:
    """A complete, versioned definition of the composite opportunity score."""

    version: str
    category_weights: dict[Category, float]
    metrics: tuple[MetricSpec, ...]
    # A category needs at least this share of its metric weight observed before
    # it produces a score at all. Below it the category is reported as
    # "insufficient data" rather than guessed at, and the composite renormalises
    # over the categories that do have evidence.
    min_category_coverage: float = 0.40
    # A company needs at least this share of total weight observed across all
    # categories before a composite is emitted.
    min_total_coverage: float = 0.50
    # Minimum number of names on a date before cross-sectional ranking is
    # meaningful at all.
    min_universe_size: int = 30
    notes: str = ""

    def __post_init__(self) -> None:
        total = sum(self.category_weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"category weights must sum to 1.0, got {total}")
        if set(self.category_weights) != set(Category):
            missing = set(Category) - set(self.category_weights)
            raise ValueError(f"category weights missing entries for {missing}")
        for cat in Category:
            metrics = [m for m in self.metrics if m.category == cat]
            if not metrics:
                raise ValueError(f"no metrics defined for category {cat}")
            w = sum(m.weight for m in metrics)
            if abs(w - 1.0) > 1e-6:
                raise ValueError(
                    f"metric weights within {cat} must sum to 1.0, got {w:.6f}"
                )
        keys = [m.key for m in self.metrics]
        if len(keys) != len(set(keys)):
            dupes = {k for k in keys if keys.count(k) > 1}
            raise ValueError(f"duplicate metric keys: {dupes}")

    def in_category(self, cat: Category) -> tuple[MetricSpec, ...]:
        return tuple(m for m in self.metrics if m.category == cat)

    def applicable(self, model: BusinessModel) -> tuple[MetricSpec, ...]:
        return tuple(m for m in self.metrics if model in m.applies_to)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(m.key for m in self.metrics)


SCORING_V1 = ScoringSpec(
    version="score-2026.09.1",
    category_weights={
        Category.QUALITY: 0.30,
        Category.GROWTH: 0.25,
        Category.MOMENTUM: 0.25,
        Category.VALUATION: 0.20,
    },
    metrics=ALL_METRICS,
    notes=(
        "Initial specification. Category weights are those set in the product brief and are "
        "fixed by policy: they are never fitted to historical returns. Metric weights within a "
        "category are declared modelling judgements about measurement reliability, also not "
        "fitted. Any change to this spec must ship as a new version so that historical "
        "recommendations remain reproducible."
    ),
)

ACTIVE_SCORING_SPEC = SCORING_V1
SCORING_SPECS: dict[str, ScoringSpec] = {SCORING_V1.version: SCORING_V1}


def get_scoring_spec(version: str | None = None) -> ScoringSpec:
    """Return a scoring spec by version, defaulting to the active one.

    Historical recommendations are re-read with the spec they were created
    under, never with today's.
    """
    if version is None:
        return ACTIVE_SCORING_SPEC
    try:
        return SCORING_SPECS[version]
    except KeyError:
        raise KeyError(
            f"unknown scoring spec version {version!r}; known: {sorted(SCORING_SPECS)}"
        ) from None


# --------------------------------------------------------------------------
# Score -> label bands
# --------------------------------------------------------------------------
# Presentation bands over the 0-100 composite, declared here rather than in the
# frontend because the interface must not decide what a number means.
#
# The cutoffs look low until you remember what the composite is: a weighted
# average of *percentile ranks* across four categories. Averaging several
# percentile distributions concentrates the result around the middle, so on the
# S&P 500 the observed spread runs from about 25 to about 77 with a standard
# deviation near 8. A score of 65 therefore means the company sits, on average,
# at the 65th percentile of its peers on quality, growth, momentum and valuation
# simultaneously -- which roughly 3% of the index manages.
#
# The bands are absolute rather than relative on purpose: a band has to mean the
# same thing in a bull market as in a bear one, or tracking the recommendations
# over time is meaningless.
SCORE_BANDS: tuple[tuple[float, str], ...] = (
    (68.0, "Very attractive"),   # about the top 1%
    (60.0, "Attractive"),        # about the top 10%
    (42.0, "Neutral"),
    (33.0, "Unattractive"),
    (0.0, "Very unattractive"),  # about the bottom 3%
)


def score_band(score: float | None) -> str:
    if score is None:
        return "Not scored"
    for threshold, label in SCORE_BANDS:
        if score >= threshold:
            return label
    return "Very unattractive"
