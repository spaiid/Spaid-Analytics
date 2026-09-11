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
# `is_yield` says the number is quoted the other way up -- cash flow over price
# rather than price over cash flow -- which inverts every operation that scales
# it. See `_apply_adjustment`.
MULTIPLE_DRIVERS: dict[str, dict] = {
    "ev_ebit": {"driver": "ebit", "is_enterprise": True, "label": "EV / EBIT"},
    "ev_ebitda": {"driver": "ebitda", "is_enterprise": True, "label": "EV / EBITDA"},
    "ev_revenue": {"driver": "revenue", "is_enterprise": True, "label": "EV / revenue"},
    "fcf_yield": {
        "driver": "free_cash_flow",
        "is_enterprise": False,
        "is_yield": True,
        "label": "Free cash flow yield",
    },
    "forward_pe": {"driver": "forward_earnings", "is_enterprise": False, "label": "Forward P/E"},
    "trailing_pe": {"driver": "net_income", "is_enterprise": False, "label": "Trailing P/E"},
    "price_to_book": {"driver": "equity", "is_enterprise": False, "label": "Price / book"},
}


def _apply_adjustment(peer_multiple: float, adjustment: float, key: str) -> float:
    """Scale a peer multiple by the quality adjustment, in the right direction.

    A premium means "this company deserves to be valued more richly than its
    peers". For a multiple that is *more*, so the multiple rises. For a yield it
    is *less*: a company the market should pay up for is one it accepts a lower
    cash-flow yield on. Scaling a yield the same way as a multiple does not just
    lose the premium, it reverses it -- the resulting value is wrong by a factor
    of (1 + adjustment) squared, which turned Kraft Heinz's 30% discount into a
    43% premium and published a $66.48 fair value for a $32.64 company.
    """
    if MULTIPLE_DRIVERS[key].get("is_yield"):
        return peer_multiple / (1.0 + adjustment)
    return peer_multiple * (1.0 + adjustment)


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
    # The same valuation done at the low and high quantiles of the peer
    # multiple distribution. The spread between them is what the peer group
    # actually disagrees by, which is the only honest range available when no
    # other method ran to disagree with.
    value_low: float | None = None
    value_high: float | None = None
    # Every multiple that contributed, so the blend can be shown rather than
    # asserted.
    components: tuple[dict, ...] = ()


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


def _describe_multiple(key: str, value: float) -> str:
    """A peer multiple as a reader would expect to see it.

    `fcf_yield` holds a yield rather than a multiple, so two decimal places of
    a number near 0.03 renders as "0.0" and looks like missing data.
    """
    if key == "fcf_yield":
        return f"{value:.1%}"
    return f"{value:.1f}"


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
    """Value the subject on every peer multiple that works, and blend them.

    `peer_multiples` maps a multiple key to the peer group's observed values.

    Every usable multiple contributes, because choosing one and discarding the
    rest hides the largest source of uncertainty in a relative valuation: which
    multiple you picked. This used to take the first preferred multiple with
    enough peers, which resolved to EV/EBIT for 345 of 366 operating companies
    and made the preference list a constant. For AMD the five usable multiples
    ran from $140 to $267 a share, and the one chosen sat near the bottom while
    the reported uncertainty -- the spread of peers within that single multiple
    -- described none of it.

    The estimate is the median across multiples, which is robust to one
    distorted denominator, and the range is the spread of the multiples that
    survived trimming. Where only one multiple is available the peer
    distribution's own quantiles supply the range instead, since there is no
    cross-multiple spread to measure.
    """
    warnings: list[str] = []
    order = preferred or spec.preferred_multiples

    adjustment = 0.0
    if spec.quality_adjustment:
        adjustment = _quality_adjustment(
            inputs, peer_growth, peer_margin, spec.max_adjustment
        )

    components: list[dict] = []
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

        adjusted = _apply_adjustment(peer_multiple, adjustment, key)
        equity = _equity_from_multiple(inputs, adjusted, key)
        if equity is None or inputs.shares <= 0:
            continue
        per_share = equity / inputs.shares
        if not math.isfinite(per_share) or per_share <= 0:
            continue

        # The same arithmetic at the edges of the peer distribution, kept so a
        # lone surviving multiple can still express an uncertainty.
        low_q, high_q = spec.spread_quantiles
        edges: list[float | None] = []
        for q in (low_q, high_q):
            edge_multiple = _apply_adjustment(
                float(np.quantile(clean, q)), adjustment, key
            )
            edge_equity = (
                _equity_from_multiple(inputs, edge_multiple, key)
                if edge_multiple > 0
                else None
            )
            edges.append(edge_equity / inputs.shares if edge_equity is not None else None)
        # A yield rises as value falls, so the quantiles arrive inverted.
        peer_low, peer_high = edges
        if peer_low is not None and peer_high is not None and peer_low > peer_high:
            peer_low, peer_high = peer_high, peer_low

        components.append(
            {
                "multiple": key,
                "label": MULTIPLE_DRIVERS[key]["label"],
                "peer_multiple": peer_multiple,
                "adjusted_multiple": adjusted,
                "peer_count": int(clean.size),
                "value_per_share": per_share,
                "peer_low": peer_low,
                "peer_high": peer_high,
            }
        )

    if not components:
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

    # One denominator close to zero -- Gilead's EBITDA net of an impairment was
    # $0.27bn against a $737 EV multiple -- produces a per-share figure orders
    # of magnitude from the others. The median already resists it, but the
    # reported range must not be blown open by it, so it is dropped outright.
    blended = float(np.median([c["value_per_share"] for c in components]))
    retained = [
        c
        for c in components
        if blended / spec.max_multiple_ratio <= c["value_per_share"] <= blended * spec.max_multiple_ratio
    ] or components
    if len(retained) < len(components):
        dropped = [c["label"] for c in components if c not in retained]
        warnings.append(
            f"{'; '.join(dropped)} implied a value far from every other multiple and was "
            "excluded, which usually means the figure it divides by is close to zero"
        )
    blended = float(np.median([c["value_per_share"] for c in retained]))

    per_share_values = [c["value_per_share"] for c in retained]
    if len(retained) > 1:
        value_low, value_high = min(per_share_values), max(per_share_values)
        detail = (
            f"median of {len(retained)} peer multiples ("
            + ", ".join(
                f"{c['label']} {_describe_multiple(c['multiple'], c['peer_multiple'])} "
                f"across {c['peer_count']}"
                for c in retained
            )
            + ")"
        )
    else:
        lone = retained[0]
        value_low, value_high = lone["peer_low"], lone["peer_high"]
        detail = (
            f"{lone['label']} of {_describe_multiple(lone['multiple'], lone['peer_multiple'])} "
            f"across {lone['peer_count']} peers"
        )

    if abs(adjustment) > 0.005:
        direction = "premium" if adjustment > 0 else "discount"
        detail += (
            f", each adjusted by a {abs(adjustment):.0%} {direction} "
            "reflecting this company's growth and margins versus the group"
        )

    # The multiple whose own answer is closest to the blend, so that every
    # consumer expecting a single representative multiple still gets an honest
    # one rather than a label invented for the blend.
    representative = min(retained, key=lambda c: abs(c["value_per_share"] - blended))

    return RelativeResult(
        value_per_share=blended,
        multiple_used=representative["multiple"],
        multiple_label=representative["label"],
        peer_multiple=representative["peer_multiple"],
        subject_multiple=(subject_multiples or {}).get(representative["multiple"]),
        peer_count=representative["peer_count"],
        adjustment=adjustment,
        detail=detail,
        warnings=warnings,
        value_low=value_low,
        value_high=value_high,
        components=tuple(retained),
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
                f"its own median {MULTIPLE_DRIVERS[key]['label']} of "
                f"{_describe_multiple(key, typical)} over "
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
