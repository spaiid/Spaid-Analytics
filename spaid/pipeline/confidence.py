"""The confidence score: how much to believe a recommendation.

The product brief is emphatic that a high score with low confidence must not be
presented as equivalent to a high score with high confidence, and this module is
where that separation is made real.

Confidence is deliberately *not* part of the opportunity score. Blending them
would destroy the information: a 70 could then mean "good company, well
understood" or "great company, barely understood", and the user could not tell
which. Keeping them orthogonal means the portfolio engine can size by one and
rank by the other.

Every component is a plain, explainable multiplier with a stated reason. Nothing
here is fitted, and an AI-generated narrative can never raise it: the brief
forbids that explicitly, and the arithmetic makes it impossible because no
component reads text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    label: str
    weight: float
    description: str


COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("data_completeness", "Data completeness", 0.25,
                  "How much of the scoring evidence was actually available."),
    ComponentSpec("data_freshness", "Data freshness", 0.15,
                  "How recently the underlying financials were filed."),
    ComponentSpec("valuation_agreement", "Valuation agreement", 0.20,
                  "Whether the valuation methods reach similar answers."),
    ComponentSpec("score_stability", "Score stability", 0.15,
                  "Whether the composite score has been steady or lurching."),
    ComponentSpec("business_predictability", "Business predictability", 0.15,
                  "How forecastable this kind of business is."),
    ComponentSpec("analyst_coverage", "Analyst coverage", 0.10,
                  "Whether forward estimates rest on enough independent opinions."),
)

COMPONENT_BY_KEY = {c.key: c for c in COMPONENTS}


@dataclass
class ConfidenceResult:
    score: float
    label: str
    components: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


def label_for(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Moderate"
    if score >= 0.35:
        return "Low"
    return "Very low"


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def assess(
    *,
    metric_coverage: float | None,
    data_age_days: float | None,
    valuation_confidence: float | None,
    method_dispersion: float | None,
    score_change_1m: float | None,
    business_model: str,
    n_analysts: int | None,
    n_missing_metrics: int | None = None,
) -> ConfidenceResult:
    """Blend the components into one 0-1 confidence with its reasoning."""
    components: list[dict] = []
    caveats: list[str] = []

    # --- data completeness -------------------------------------------------
    completeness = _clamp(metric_coverage if metric_coverage is not None else 0.5)
    detail = (
        f"{completeness:.0%} of the scoring evidence was available"
        if metric_coverage is not None
        else "coverage unknown"
    )
    if completeness < 0.6:
        caveats.append(
            f"only {completeness:.0%} of the scoring inputs were available for this company"
        )
    if n_missing_metrics:
        detail += f"; {n_missing_metrics} metrics could not be computed"
    components.append(_component("data_completeness", completeness, detail))

    # --- freshness ---------------------------------------------------------
    if data_age_days is None:
        freshness, detail = 0.5, "filing age unknown"
    elif data_age_days <= 45:
        freshness, detail = 1.0, f"financials filed {data_age_days:.0f} days ago"
    elif data_age_days <= 100:
        freshness, detail = 0.85, f"financials filed {data_age_days:.0f} days ago"
    elif data_age_days <= 140:
        freshness, detail = 0.6, f"financials are {data_age_days:.0f} days old"
    else:
        freshness = _clamp(0.4 - (data_age_days - 140) / 500)
        detail = f"financials are {data_age_days:.0f} days old, so a quarter may be missing"
        caveats.append(detail)
    components.append(_component("data_freshness", freshness, detail))

    # --- valuation agreement ----------------------------------------------
    if valuation_confidence is None:
        agreement, detail = 0.35, "no valuation could be produced"
        caveats.append("no fair-value estimate could be produced for this company")
    else:
        agreement = _clamp(valuation_confidence)
        if method_dispersion is not None:
            detail = (
                f"valuation methods disagree by {method_dispersion:.0%} around the base case"
            )
        else:
            detail = "based on the fair-value engine's own confidence"
    components.append(_component("valuation_agreement", agreement, detail))

    # --- score stability ---------------------------------------------------
    if score_change_1m is None:
        stability, detail = 0.6, "no score history yet"
    else:
        magnitude = abs(score_change_1m)
        stability = _clamp(1.0 - magnitude / 40.0)
        detail = f"the composite score moved {score_change_1m:+.1f} points over the past month"
        if magnitude > 20:
            caveats.append(
                f"the score has moved sharply ({score_change_1m:+.1f} points in a month), so "
                "the evidence behind it is shifting"
            )
    components.append(_component("score_stability", stability, detail))

    # --- business predictability -------------------------------------------
    predictability = {
        "operating": 0.85,
        "bank": 0.70,
        "insurer": 0.65,
        "reit": 0.80,
        "cyclical": 0.50,
        "unprofitable_growth": 0.30,
    }.get(business_model, 0.6)
    detail = {
        "operating": "an ordinary operating company, reasonably forecastable",
        "bank": "a bank: leverage and credit cycles make the outlook less predictable",
        "insurer": "an insurer: underwriting results are lumpy",
        "reit": "a property trust: rental income is comparatively stable",
        "cyclical": "a cyclical business, where earnings depend on where the cycle sits",
        "unprofitable_growth": "not yet profitable, so the outcome is genuinely uncertain",
    }.get(business_model, "business type unknown")
    if predictability <= 0.5:
        caveats.append(f"This is {detail}.")
    components.append(_component("business_predictability", predictability, detail))

    # --- analyst coverage --------------------------------------------------
    if n_analysts is None:
        coverage_score, detail = 0.5, "analyst coverage unknown"
    elif n_analysts >= 15:
        coverage_score, detail = 1.0, f"{n_analysts} analysts cover the company"
    elif n_analysts >= 8:
        coverage_score, detail = 0.8, f"{n_analysts} analysts cover the company"
    elif n_analysts >= 3:
        coverage_score, detail = 0.55, f"only {n_analysts} analysts cover the company"
    else:
        coverage_score = 0.3
        detail = f"only {n_analysts} analysts cover the company, so forward estimates are thin"
        caveats.append(detail)
    components.append(_component("analyst_coverage", coverage_score, detail))

    score = sum(c["score"] * c["weight"] for c in components)
    return ConfidenceResult(
        score=_clamp(score), label=label_for(score), components=components, caveats=caveats
    )


def _component(key: str, score: float, detail: str) -> dict:
    spec = COMPONENT_BY_KEY[key]
    return {
        "key": key,
        "label": spec.label,
        "score": _clamp(score),
        "weight": spec.weight,
        "detail": detail,
    }
