"""The valuation specification: how fair value is estimated, and how confident we may be.

Like the scoring spec this is versioned, because a fair-value range is only
comparable to another one produced the same way. The research journal stores the
version alongside every estimate.

Three principles encoded here:

1. **Blend, don't pick.** A discounted-cash-flow model, a comparable-company
   multiple and the company's own valuation history each carry weight, because
   each fails in a different way. Their *disagreement* is itself information and
   feeds the confidence score.
2. **Drop a method rather than fake it.** When a method's inputs are unreliable
   or meaningless for the business (a DCF on a bank, comparables with four
   peers), its weight goes to zero and the remainder renormalise. A method is
   never run on inputs it cannot handle just to keep the weights tidy.
3. **A range, never a number.** Bear, base and bull scenarios are produced from
   declared assumption sets, and the classification depends on where price sits
   relative to the *range* and how wide that range is -- not on a P/E threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from spaid.config.scoring import BusinessModel


class ValuationMethod(StrEnum):
    DCF = "dcf"
    COMPARABLES = "comparables"
    HISTORICAL = "historical"
    # Model-specific methods, used where the generic three do not apply.
    RESIDUAL_INCOME = "residual_income"  # banks: book value + excess ROE
    NAV_AFFO = "nav_affo"  # REITs: net asset value and AFFO multiple
    NORMALIZED_EARNINGS = "normalized_earnings"  # cyclicals: mid-cycle margins


class Scenario(StrEnum):
    BEAR = "bear"
    BASE = "base"
    BULL = "bull"


class ValuationClass(StrEnum):
    SIGNIFICANTLY_UNDERVALUED = "significantly_undervalued"
    UNDERVALUED = "undervalued"
    FAIRLY_VALUED = "fairly_valued"
    OVERVALUED = "overvalued"
    SIGNIFICANTLY_OVERVALUED = "significantly_overvalued"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"


# Default method blend for an ordinary profitable operating company.
DEFAULT_METHOD_WEIGHTS: dict[ValuationMethod, float] = {
    ValuationMethod.DCF: 0.50,
    ValuationMethod.COMPARABLES: 0.30,
    ValuationMethod.HISTORICAL: 0.20,
}

# Per-business-model overrides. These are the "do not force one model onto every
# company" rules made explicit.
MODEL_METHOD_WEIGHTS: dict[BusinessModel, dict[ValuationMethod, float]] = {
    BusinessModel.OPERATING: dict(DEFAULT_METHOD_WEIGHTS),
    # A bank's cash flows are not separable from its financing, so a standard
    # free-cash-flow DCF is meaningless. Value comes from book equity and the
    # spread between return on equity and the cost of equity.
    BusinessModel.BANK: {
        ValuationMethod.RESIDUAL_INCOME: 0.45,
        ValuationMethod.COMPARABLES: 0.30,
        ValuationMethod.HISTORICAL: 0.25,
    },
    # Insurers are valued on book and sustainable ROE much like banks, with
    # underwriting profitability as the quality overlay.
    BusinessModel.INSURER: {
        ValuationMethod.RESIDUAL_INCOME: 0.40,
        ValuationMethod.COMPARABLES: 0.30,
        ValuationMethod.HISTORICAL: 0.30,
    },
    # REIT earnings are depressed by non-cash depreciation; AFFO and net asset
    # value are the real measures.
    BusinessModel.REIT: {
        ValuationMethod.NAV_AFFO: 0.50,
        ValuationMethod.COMPARABLES: 0.25,
        ValuationMethod.HISTORICAL: 0.25,
    },
    # For a cyclical, trailing earnings describe where we are in the cycle, not
    # what the business earns through one. Normalised mid-cycle earnings lead.
    BusinessModel.CYCLICAL: {
        ValuationMethod.NORMALIZED_EARNINGS: 0.40,
        ValuationMethod.DCF: 0.20,
        ValuationMethod.COMPARABLES: 0.20,
        ValuationMethod.HISTORICAL: 0.20,
    },
    # With no profits there is nothing for a multiple to attach to, so the DCF
    # carries the estimate and revenue-based comparables provide the sanity
    # check. Confidence is structurally capped for these names.
    BusinessModel.UNPROFITABLE_GROWTH: {
        ValuationMethod.DCF: 0.60,
        ValuationMethod.COMPARABLES: 0.40,
    },
}


@dataclass(frozen=True)
class DiscountRateSpec:
    """How the discount rate is built.

    Capital asset pricing with a declared equity risk premium, a beta floor and
    cap to stop a noisy regression estimate from dominating, and a small-size
    premium. Every term is visible so the user can see what a change of
    assumption does.
    """

    risk_free_fallback: float = 0.042  # used when the live 10-year yield is unavailable
    equity_risk_premium: float = 0.050
    beta_floor: float = 0.60
    beta_cap: float = 2.00
    size_premium_small: float = 0.010  # applied below `size_premium_threshold`
    size_premium_threshold: float = 10_000_000_000.0  # $10bn market cap
    # Additional premium for companies whose fundamentals are hard to forecast,
    # applied on the unprofitable-growth path.
    uncertainty_premium: float = 0.015
    min_rate: float = 0.055
    max_rate: float = 0.160


@dataclass(frozen=True)
class ScenarioAssumptions:
    """Assumption set for one scenario of the discounted-cash-flow model.

    Expressed as adjustments to the company's own recent history rather than as
    absolute numbers, so that the same spec is meaningful for a utility and for a
    semiconductor company.
    """

    label: str
    # Multiplier on the starting revenue growth rate estimate.
    growth_multiplier: float
    # Absolute adjustment to the starting operating margin, in percentage points
    # expressed as a fraction (0.02 = +2pp).
    margin_adjustment: float
    # Rate at which growth fades toward terminal growth over the forecast.
    fade_strength: float
    # Adjustment to the terminal growth rate.
    terminal_growth_adjustment: float
    # Adjustment to the discount rate, in fraction terms.
    discount_rate_adjustment: float
    # Probability weight used when collapsing the three scenarios to a
    # midpoint. Deliberately not uniform: the base case should dominate.
    probability: float
    # Which end of the analyst range this scenario takes when the projection is
    # anchored to consensus. The spread between analysts is a better bear and
    # bull case than a multiplier chosen in advance, because it is a
    # disagreement that actually exists.
    consensus_side: str = "mid"  # low | mid | high


@dataclass(frozen=True)
class DcfSpec:
    forecast_years: int = 10
    terminal_growth: float = 0.025  # capped at long-run nominal GDP-ish growth
    terminal_growth_cap: float = 0.035
    # Effective tax rate bounds, applied to the company's own trailing rate.
    tax_rate_floor: float = 0.10
    tax_rate_cap: float = 0.35
    tax_rate_default: float = 0.21
    # Reinvestment is modelled via a sales-to-capital ratio, bounded so a single
    # odd year cannot produce absurd capital intensity.
    sales_to_capital_floor: float = 0.8
    sales_to_capital_cap: float = 6.0
    sales_to_capital_default: float = 2.0
    # Share-based compensation is a real cost to existing owners. The starting
    # margin is GAAP, so the cost is already charged; leaving this true simply
    # keeps it that way. Setting it false adds the expense back, which is the
    # non-GAAP view and systematically overvalues the companies that use it most.
    treat_sbc_as_expense: bool = True
    # Expected annual net dilution applied beyond buybacks, floored at zero.
    max_dilution_rate: float = 0.05
    # Terminal value may not exceed this share of total value without the
    # estimate being flagged as terminal-value dependent.
    terminal_share_warning: float = 0.80
    # How far the projection's first year may disagree with reported free cash
    # flow, as a fraction of revenue, before the projection is treated as
    # describing a different company. A sign flip alone is not enough: a
    # genuinely investing business can dip negative. A sign flip worth a tenth
    # of revenue is an assumption error.
    fcf_contradiction_threshold: float = 0.10
    # Anchoring the near-term projection to consensus needs enough analysts for
    # the consensus to mean anything.
    consensus_min_analysts: int = 5
    # How far above the company's current operating margin a consensus-implied
    # margin may sit before it is treated as an estimate error rather than an
    # expectation. Expressed in margin points.
    consensus_margin_cap_pp: float = 0.25
    discount: DiscountRateSpec = field(default_factory=DiscountRateSpec)
    scenarios: tuple[ScenarioAssumptions, ...] = (
        ScenarioAssumptions(
            label="bear",
            growth_multiplier=0.55,
            margin_adjustment=-0.025,
            fade_strength=1.35,
            terminal_growth_adjustment=-0.010,
            discount_rate_adjustment=+0.015,
            probability=0.25,
            consensus_side="low",
        ),
        ScenarioAssumptions(
            label="base",
            growth_multiplier=1.00,
            margin_adjustment=0.0,
            fade_strength=1.00,
            terminal_growth_adjustment=0.0,
            discount_rate_adjustment=0.0,
            probability=0.50,
        ),
        ScenarioAssumptions(
            label="bull",
            growth_multiplier=1.35,
            margin_adjustment=+0.025,
            fade_strength=0.75,
            terminal_growth_adjustment=+0.005,
            discount_rate_adjustment=-0.010,
            probability=0.25,
            consensus_side="high",
        ),
    )

    def __post_init__(self) -> None:
        p = sum(s.probability for s in self.scenarios)
        if abs(p - 1.0) > 1e-9:
            raise ValueError(f"scenario probabilities must sum to 1.0, got {p}")


@dataclass(frozen=True)
class ComparablesSpec:
    """Peer-multiple valuation."""

    min_peers: int = 6
    # Trim the peer multiple distribution before taking the central value, so one
    # distressed or one euphoric peer cannot set fair value.
    trim_fraction: float = 0.20
    # Multiples tried in order of preference for a normal operating company.
    preferred_multiples: tuple[str, ...] = ("ev_ebit", "ev_ebitda", "fcf_yield", "forward_pe")
    # Peers are penalised for differing in growth and profitability; a peer
    # multiple is only fair if the peer is actually comparable.
    quality_adjustment: bool = True
    max_adjustment: float = 0.30  # cap the size of that adjustment
    # Quantiles of the peer multiple distribution used to express the spread of
    # defensible answers. Where comparables is the only method that ran, these
    # are the honest bear and bull cases: the peer group itself disagrees, and
    # by this much.
    spread_quantiles: tuple[float, float] = (0.25, 0.75)
    # How far one multiple's answer may sit from the median of the others
    # before it is treated as arithmetic on a near-zero denominator rather than
    # a different opinion about value.
    max_multiple_ratio: float = 4.0


@dataclass(frozen=True)
class HistoricalSpec:
    """Valuation against the company's own history."""

    lookback_years: int = 5
    min_observations: int = 500  # roughly two years of daily history
    # Central tendency of the company's own multiple over the lookback, with the
    # extremes trimmed.
    trim_fraction: float = 0.10
    # A company whose business has changed materially should not be valued
    # against its own past multiple. Flag when growth or margin has shifted more
    # than this much versus the lookback average.
    regime_change_growth_delta: float = 0.15
    regime_change_margin_delta: float = 0.08


@dataclass(frozen=True)
class ClassificationSpec:
    """How price versus the fair-value range becomes a label.

    The thresholds scale with the *width* of the range: a stock 15% below a
    tight, confident range is a stronger statement than one 15% below a range
    spanning a factor of three. That is why classification reads the range and
    the confidence, never a bare multiple.
    """

    # Upside to the midpoint required for each label, as a fraction, before
    # width scaling.
    significantly_undervalued: float = 0.30
    undervalued: float = 0.12
    overvalued: float = -0.12
    significantly_overvalued: float = -0.30
    # Scale thresholds by range width relative to this reference width
    # (bull/bear spread over midpoint). A wider range demands more upside.
    reference_width: float = 0.60
    max_width_multiplier: float = 2.0
    # Below this confidence we decline to classify at all.
    min_confidence: float = 0.35


@dataclass(frozen=True)
class ValuationSpec:
    version: str
    method_weights: dict[BusinessModel, dict[ValuationMethod, float]]
    dcf: DcfSpec = field(default_factory=DcfSpec)
    comparables: ComparablesSpec = field(default_factory=ComparablesSpec)
    historical: HistoricalSpec = field(default_factory=HistoricalSpec)
    classification: ClassificationSpec = field(default_factory=ClassificationSpec)
    notes: str = ""

    def __post_init__(self) -> None:
        for model, weights in self.method_weights.items():
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"method weights for {model} must sum to 1.0, got {total}"
                )

    def weights_for(self, model: BusinessModel) -> dict[ValuationMethod, float]:
        return dict(self.method_weights.get(model, DEFAULT_METHOD_WEIGHTS))


VALUATION_V1 = ValuationSpec(
    version="value-2026.09.1",
    method_weights=MODEL_METHOD_WEIGHTS,
    notes=(
        "Initial specification. The 50/30/20 discounted-cash-flow / comparables / own-history "
        "blend from the product brief applies to ordinary operating companies; banks, insurers, "
        "REITs, cyclicals and unprofitable growth companies use the model-appropriate blends "
        "declared above. Weights renormalise when a method cannot run on the available data."
    ),
)

ACTIVE_VALUATION_SPEC = VALUATION_V1
VALUATION_SPECS: dict[str, ValuationSpec] = {VALUATION_V1.version: VALUATION_V1}


def get_valuation_spec(version: str | None = None) -> ValuationSpec:
    if version is None:
        return ACTIVE_VALUATION_SPEC
    try:
        return VALUATION_SPECS[version]
    except KeyError:
        raise KeyError(
            f"unknown valuation spec version {version!r}; known: {sorted(VALUATION_SPECS)}"
        ) from None
