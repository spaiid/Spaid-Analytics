"""Relative valuation: comparable companies, and a company against its own history.

Both methods answer "what would the market pay for this?" rather than "what is
this worth?", which is why neither carries the majority weight for an ordinary
company. They are the check on the discounted cash-flow model's assumptions, not
a substitute for them.

Each has a characteristic failure the implementation guards against.

**Comparables** fail when the peers are not comparable. A low-growth,
low-margin company sitting in a sector of fast-growing ones will look cheap
against the peer multiple forever, which is precisely the value trap the product
is meant to catch. So the peer multiple is adjusted for how far the subject's
growth and profitability sit from the peer group's, and the adjustment is
capped so it can never do more than modestly re-rate the estimate.

**Own history** fails when the business has changed. Valuing a company that has
transitioned from hardware to subscriptions against its own hardware-era
multiple is an error of category, not of degree. The implementation detects a
material shift in growth or margin versus the lookback average and reports it,
so the blending layer can down-weight or drop the method.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from spaid.config.valuation import ComparablesSpec, HistoricalSpec

log = logging.getLogger(__name__)

EPS = 1e-9

# Which multiple to use, and how to turn it back into a per-share value.
# `is_enterprise` says whether the multiple values the whole firm (so net debt
# must be subtracted to reach equity) or the equity directly.
MULTIPLE_DRIVERS: dict[str, dict] = {
    "ev_ebit": {"driver": "ebit", "is_enterprise": True, "label": "EV / EBIT"},
    "ev_ebitda": {"driver": "ebitda", "is_enterprise": True, "label": "EV / EBITDA"},
    "ev_revenue": {"driver": "revenue", "is_enterprise": True, "label": "EV / revenue"},
    "fcf_yield": {"driver": "free_cash_flow", "is_enterprise": False, "label": "Price / free cash flow"},
    "forward_pe": {"driver": "forward_earnings", "is_enterprise": False, "label": "Forward P/E"},
    "trailing_pe": {"driver": "net_income", "is_enterprise": False, "label": "Trailing P/E"},
    "price_to_book": {"driver": "equity", "is_enterprise": False, "label": "Price / book"},
}


@dataclass(frozen=True)
class RelativeInputs:
    """The subject company's own figures."""

    shares: float
    net_debt: float
    minority_interest: float = 0.0
    preferred_equity: float = 0.0
    revenue: float | None = None
    ebit: float | None = None
    ebitda: float | None = None
    net_income: float | None = None
    free_cash_flow: float | None = None
    forward_earnings: float | None = None
    equity: float | None = None
    # Used to adjust a peer multiple for genuine differences in the business.
    growth: float | None = None
    margin: float | None = None
    roe: float | None = None

    def driver(self, name: str) -> float | None:
        return getattr(self, name, None)


@dataclass
class RelativeResult:
    value_per_share: float | None
    multiple_used: str | None
    multiple_label: str | None
    peer_multiple: float | None
    subject_multiple: float | None
    peer_count: int = 0
    adjustment: float = 0.0
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


def _trimmed_median(values: np.ndarray, trim: float) -> float | None:
    """Central value with the tails removed, so one distressed peer cannot set it."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return None
    if clean.size >= 5 and trim > 0:
        lo, hi = np.quantile(clean, [trim, 1.0 - trim])
        clean = clean[(clean >= lo) & (clean <= hi)]
        if clean.size == 0:
            return None
    return float(np.median(clean))


def _equity_from_multiple(
    inputs: RelativeInputs, multiple: float, key: str
) -> float | None:
    """Convert a peer multiple into an equity value for the subject."""
    meta = MULTIPLE_DRIVERS[key]
    driver = inputs.driver(meta["driver"])
    if driver is None or not math.isfinite(driver):
        return None

    if key == "fcf_yield":
        # A yield, not a multiple: price = cash flow / yield.
        if multiple <= EPS or driver <= 0:
            return None
        return driver / multiple

    if driver <= 0:
        # A negative denominator makes the multiple meaningless. A loss-making
        # company cannot be valued on a price/earnings comparison, and pretending
        # otherwise produces a negative fair value.
        return None

    gross = multiple * driver
    if meta["is_enterprise"]:
        return gross - inputs.net_debt - inputs.minority_interest - inputs.preferred_equity
    return gross


def comparables_value(
    inputs: RelativeInputs,
    peer_multiples: dict[str, np.ndarray],
    *,
    spec: ComparablesSpec,
    subject_multiples: dict[str, float] | None = None,
    peer_growth: np.ndarray | None = None,
    peer_margin: np.ndarray | None = None,
    preferred: tuple[str, ...] | None = None,
) -> RelativeResult:
    """Value the subject at the peer group's multiple.

    `peer_multiples` maps a multiple key to the peer group's observed values.
    The first preferred multiple with enough usable peers and a positive driver
    for the subject wins.
    """
    warnings: list[str] = []
    order = preferred or spec.preferred_multiples

    for key in order:
        if key not in MULTIPLE_DRIVERS:
            continue
        values = peer_multiples.get(key)
        if values is None:
            continue
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean) & (clean > 0)]
        if clean.size < spec.min_peers:
            continue

        peer_multiple = _trimmed_median(clean, spec.trim_fraction)
        if peer_multiple is None or peer_multiple <= 0:
            continue

        adjustment = 0.0
        if spec.quality_adjustment:
            adjustment = _quality_adjustment(
                inputs, peer_growth, peer_margin, spec.max_adjustment
            )
        adjusted = peer_multiple * (1.0 + adjustment)

        equity = _equity_from_multiple(inputs, adjusted, key)
        if equity is None or inputs.shares <= 0:
            continue

        detail = (
            f"{MULTIPLE_DRIVERS[key]['label']} of {peer_multiple:.1f} across "
            f"{clean.size} peers"
        )
        if abs(adjustment) > 0.005:
            direction = "premium" if adjustment > 0 else "discount"
            detail += (
                f", adjusted to {adjusted:.1f} for a {abs(adjustment):.0%} {direction} "
                "reflecting this company's growth and margins versus the group"
            )

        return RelativeResult(
            value_per_share=equity / inputs.shares,
            multiple_used=key,
            multiple_label=MULTIPLE_DRIVERS[key]["label"],
            peer_multiple=peer_multiple,
            subject_multiple=(subject_multiples or {}).get(key),
            peer_count=int(clean.size),
            adjustment=adjustment,
            detail=detail,
            warnings=warnings,
        )

    warnings.append(
        "no multiple had both enough comparable peers and a positive figure for this company"
    )
    return RelativeResult(
        value_per_share=None,
        multiple_used=None,
        multiple_label=None,
        peer_multiple=None,
        subject_multiple=None,
        detail="comparable-company valuation not available",
        warnings=warnings,
    )


def _quality_adjustment(
    inputs: RelativeInputs,
    peer_growth: np.ndarray | None,
    peer_margin: np.ndarray | None,
    cap: float,
) -> float:
    """Premium or discount for being better or worse than the peer group.

    Without this, the cheapest company in every industry is simply the worst one,
    and the screen becomes a machine for finding declining businesses.
    """
    adjustment = 0.0

    if peer_growth is not None and inputs.growth is not None:
        clean = np.asarray(peer_growth, dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size >= 5:
            median = float(np.median(clean))
            spread = float(np.std(clean)) or 0.05
            z = (inputs.growth - median) / max(spread, 0.01)
            adjustment += 0.10 * max(min(z, 2.0), -2.0)

    if peer_margin is not None and inputs.margin is not None:
        clean = np.asarray(peer_margin, dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size >= 5:
            median = float(np.median(clean))
            spread = float(np.std(clean)) or 0.03
            z = (inputs.margin - median) / max(spread, 0.01)
            adjustment += 0.08 * max(min(z, 2.0), -2.0)

    return max(min(adjustment, cap), -cap)


@dataclass
class HistoricalResult:
    value_per_share: float | None
    multiple_used: str | None
    multiple_label: str | None
    historical_multiple: float | None
    current_multiple: float | None
    observations: int = 0
    regime_change: bool = False
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


def historical_value(
    inputs: RelativeInputs,
    history: dict[str, np.ndarray],
    *,
    spec: HistoricalSpec,
    current_growth: float | None = None,
    historical_growth: float | None = None,
    current_margin: float | None = None,
    historical_margin: float | None = None,
    preferred: tuple[str, ...] = ("ev_ebit", "ev_ebitda", "trailing_pe", "price_to_book"),
) -> HistoricalResult:
    """Value the subject at its own historical multiple.

    Reports a regime change rather than silently valuing a transformed business
    against the multiple its previous incarnation earned.
    """
    warnings: list[str] = []

    regime_change = False
    reasons: list[str] = []
    if current_growth is not None and historical_growth is not None:
        if abs(current_growth - historical_growth) > spec.regime_change_growth_delta:
            regime_change = True
            reasons.append(
                f"growth has moved from {historical_growth:.0%} to {current_growth:.0%}"
            )
    if current_margin is not None and historical_margin is not None:
        if abs(current_margin - historical_margin) > spec.regime_change_margin_delta:
            regime_change = True
            reasons.append(
                f"operating margin has moved from {historical_margin:.0%} to {current_margin:.0%}"
            )
    if regime_change:
        warnings.append(
            "the business has changed materially over the lookback period ("
            + "; ".join(reasons)
            + "), so its own past multiple is a weak guide"
        )

    for key in preferred:
        if key not in MULTIPLE_DRIVERS:
            continue
        values = history.get(key)
        if values is None:
            continue
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean) & (clean > 0)]
        if clean.size < spec.min_observations:
            continue

        typical = _trimmed_median(clean, spec.trim_fraction)
        if typical is None or typical <= 0:
            continue

        equity = _equity_from_multiple(inputs, typical, key)
        if equity is None or inputs.shares <= 0:
            continue

        return HistoricalResult(
            value_per_share=equity / inputs.shares,
            multiple_used=key,
            multiple_label=MULTIPLE_DRIVERS[key]["label"],
            historical_multiple=typical,
            current_multiple=None,
            observations=int(clean.size),
            regime_change=regime_change,
            detail=(
                f"its own median {MULTIPLE_DRIVERS[key]['label']} of {typical:.1f} over "
                f"the past {spec.lookback_years} years"
            ),
            warnings=warnings,
        )

    warnings.append("not enough valuation history to compare against")
    return HistoricalResult(
        value_per_share=None,
        multiple_used=None,
        multiple_label=None,
        historical_multiple=None,
        current_multiple=None,
        regime_change=regime_change,
        detail="historical valuation not available",
        warnings=warnings,
    )
