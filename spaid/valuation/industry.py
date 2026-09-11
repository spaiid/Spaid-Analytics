"""Valuation models for businesses the standard template does not fit.

Forcing one model onto every company is how an app ends up reporting that a bank
is worth nothing because its free cash flow is negative, or that a real-estate
trust earns a 2% return because depreciation swamps its accounting profit. Each
model here exists because a specific class of business breaks the general one.

**Banks.** A bank's borrowing *is* its raw material, so free cash flow to the
firm is not a meaningful concept and net debt is not a claim to subtract. Value
comes instead from book equity plus the present value of returns earned above
the cost of that equity: if a bank earns 14% on equity that costs 10%, the
excess is worth something, and if it earns 8% the market should pay less than
book. That is the residual-income model.

**Insurers.** Same structure, with underwriting profitability as the quality
overlay: a combined ratio below one means the business is paid to hold float.

**Real-estate trusts.** Property is carried at depreciated cost, so reported
earnings understate economics badly after a decade of appreciation, and book
value understates the assets. Adjusted funds from operations adds depreciation
back and subtracts the capital expenditure genuinely needed to maintain the
buildings; net asset value capitalises the property income at a market yield.

**Cyclicals.** Trailing earnings describe where the cycle is, not what the
business earns through one. Normalising the margin to its mid-cycle average and
applying it to current revenue gives a figure that does not call a miner cheap
at the top of the cycle and expensive at the bottom.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

EPS = 1e-9


@dataclass
class ModelResult:
    value_per_share: float | None
    method: str
    label: str
    detail: str = ""
    assumptions: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Banks and insurers: residual income
# ---------------------------------------------------------------------------


def residual_income_value(
    *,
    book_equity: float | None,
    shares: float | None,
    roe: float | None,
    cost_of_equity: float,
    growth: float = 0.02,
    fade_years: int = 10,
    terminal_growth: float = 0.02,
    intangibles: float | None = None,
    use_tangible_book: bool = False,
) -> ModelResult:
    """Book value plus the present value of returns above the cost of equity.

    The excess return fades to zero over `fade_years`: competition erodes an
    advantage in banking faster than in most industries, and assuming a bank
    earns 16% on equity in perpetuity is the single easiest way to overvalue one.
    """
    warnings: list[str] = []
    if not book_equity or not shares or shares <= 0:
        return ModelResult(None, "residual_income", "Residual income",
                           "no book equity or share count", warnings=["missing book equity"])
    if book_equity <= 0:
        return ModelResult(None, "residual_income", "Residual income",
                           "negative book equity", warnings=["book equity is negative"])
    if roe is None or not math.isfinite(roe):
        return ModelResult(None, "residual_income", "Residual income",
                           "no return on equity", warnings=["missing return on equity"])

    equity = book_equity
    if use_tangible_book and intangibles:
        equity = max(book_equity - intangibles, book_equity * 0.2)

    if roe > 0.40:
        warnings.append(f"return on equity of {roe:.0%} capped at 40% as implausibly persistent")
        roe = 0.40
    if roe < -0.30:
        roe = -0.30

    value = equity
    book = equity
    for t in range(1, fade_years + 1):
        # The excess return decays linearly to zero at the horizon.
        decay = 1.0 - (t - 1) / fade_years
        excess = (roe - cost_of_equity) * decay
        residual = book * excess
        value += residual / ((1.0 + cost_of_equity) ** t)
        book *= 1.0 + growth

    spread = roe - cost_of_equity
    detail = (
        f"book equity of {equity / 1e9:.1f}bn plus the present value of a "
        f"{spread:+.1%} return spread fading over {fade_years} years"
    )
    return ModelResult(
        value_per_share=value / shares,
        method="residual_income",
        label="Residual income",
        detail=detail,
        assumptions={
            "book_equity": equity,
            "return_on_equity": roe,
            "cost_of_equity": cost_of_equity,
            "spread": spread,
            "fade_years": fade_years,
            "book_growth": growth,
            "tangible_book_used": use_tangible_book,
        },
        warnings=warnings,
    )


def insurer_value(
    *,
    book_equity: float | None,
    shares: float | None,
    roe: float | None,
    cost_of_equity: float,
    premiums_earned: float | None = None,
    policy_benefits: float | None = None,
    growth: float = 0.03,
) -> ModelResult:
    """Residual income with the underwriting result as a quality overlay.

    An insurer whose claims exceed its premiums is being paid to take risk only
    if its investment income more than covers the gap, so the loss ratio adjusts
    the sustainable return on equity rather than the value directly.
    """
    adjusted_roe = roe
    warnings: list[str] = []
    loss_ratio = None

    if premiums_earned and policy_benefits and premiums_earned > EPS:
        loss_ratio = policy_benefits / premiums_earned
        if loss_ratio > 1.0:
            # Underwriting at a loss: haircut the sustainable return.
            penalty = min((loss_ratio - 1.0) * 0.5, 0.05)
            adjusted_roe = (roe or 0.0) - penalty
            warnings.append(
                f"claims are {loss_ratio:.0%} of premiums, so underwriting loses money; "
                "the sustainable return on equity has been reduced accordingly"
            )

    result = residual_income_value(
        book_equity=book_equity,
        shares=shares,
        roe=adjusted_roe,
        cost_of_equity=cost_of_equity,
        growth=growth,
    )
    result.method = "residual_income"
    result.label = "Residual income (insurer)"
    result.warnings.extend(warnings)
    if loss_ratio is not None:
        result.assumptions["loss_ratio"] = loss_ratio
    return result


# ---------------------------------------------------------------------------
# Real-estate trusts: adjusted funds from operations and net asset value
# ---------------------------------------------------------------------------


def reit_value(
    *,
    net_income: float | None,
    depreciation: float | None,
    capex: float | None,
    shares: float | None,
    net_debt: float | None,
    affo_multiple: float = 18.0,
    cap_rate: float = 0.055,
    peer_affo_multiple: float | None = None,
) -> ModelResult:
    """Blend an adjusted-funds-from-operations multiple with a net-asset value.

    Funds from operations adds depreciation back to earnings because buildings
    do not actually wear out on a straight line. The *adjusted* version then
    subtracts the capital expenditure that genuinely is needed to keep them
    rentable, which is the honest version of the measure.
    """
    warnings: list[str] = []
    if not shares or shares <= 0:
        return ModelResult(None, "nav_affo", "AFFO and net asset value",
                           "no share count", warnings=["missing share count"])
    if net_income is None or depreciation is None:
        return ModelResult(None, "nav_affo", "AFFO and net asset value",
                           "missing earnings or depreciation",
                           warnings=["cannot compute funds from operations"])

    ffo = net_income + depreciation
    maintenance_capex = capex if capex is not None else depreciation * 0.15
    affo = ffo - maintenance_capex
    if affo <= 0:
        return ModelResult(None, "nav_affo", "AFFO and net asset value",
                           "adjusted funds from operations is negative",
                           warnings=["negative adjusted funds from operations"])

    multiple = peer_affo_multiple if peer_affo_multiple and peer_affo_multiple > 0 else affo_multiple
    affo_equity = affo * multiple

    # Net asset value: capitalise the property income stream at a market yield,
    # then subtract the debt against it.
    nav_equity = None
    if net_debt is not None and cap_rate > EPS:
        gross_asset_value = ffo / cap_rate
        nav_equity = gross_asset_value - net_debt

    if nav_equity is not None and nav_equity > 0:
        equity = 0.6 * affo_equity + 0.4 * nav_equity
        detail = (
            f"adjusted funds from operations of {affo / 1e9:.2f}bn at {multiple:.0f}x, "
            f"blended with a net asset value at a {cap_rate:.1%} capitalisation rate"
        )
    else:
        equity = affo_equity
        detail = f"adjusted funds from operations of {affo / 1e9:.2f}bn at {multiple:.0f}x"
        if nav_equity is not None:
            warnings.append("net asset value came out negative, so only the AFFO multiple was used")

    return ModelResult(
        value_per_share=equity / shares,
        method="nav_affo",
        label="AFFO and net asset value",
        detail=detail,
        assumptions={
            "funds_from_operations": ffo,
            "adjusted_ffo": affo,
            "maintenance_capex": maintenance_capex,
            "affo_multiple": multiple,
            "cap_rate": cap_rate,
            "nav_equity": nav_equity,
        },
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Cyclicals: normalised mid-cycle earnings
# ---------------------------------------------------------------------------


def normalized_earnings_value(
    *,
    revenue: float | None,
    margin_history: np.ndarray | None,
    current_margin: float | None,
    tax_rate: float,
    shares: float | None,
    net_debt: float | None,
    exit_multiple: float = 12.0,
    peer_multiple: float | None = None,
) -> ModelResult:
    """Value a cyclical on the margin it earns through a cycle, not today's.

    A miner at the top of the cycle looks cheap on trailing earnings and a miner
    at the bottom looks ruinous; both readings are artefacts of where the
    commodity price happens to be. Using the median margin over as long a history
    as we have removes most of that.
    """
    warnings: list[str] = []
    if not shares or shares <= 0 or not revenue or revenue <= 0:
        return ModelResult(None, "normalized_earnings", "Normalised earnings",
                           "no revenue or share count",
                           warnings=["missing revenue or share count"])

    normalized_margin = None
    if margin_history is not None:
        clean = np.asarray(margin_history, dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size >= 8:
            normalized_margin = float(np.median(clean))

    if normalized_margin is None:
        if current_margin is None or not math.isfinite(current_margin):
            return ModelResult(None, "normalized_earnings", "Normalised earnings",
                               "no margin history",
                               warnings=["not enough margin history to normalise"])
        normalized_margin = current_margin
        warnings.append(
            "not enough history to establish a mid-cycle margin, so the current margin was used; "
            "for a cyclical business that is a weak assumption"
        )

    multiple = peer_multiple if peer_multiple and peer_multiple > 0 else exit_multiple
    normalized_ebit = revenue * normalized_margin
    if normalized_ebit <= 0:
        return ModelResult(None, "normalized_earnings", "Normalised earnings",
                           "mid-cycle earnings are negative",
                           warnings=["this business does not earn a profit through the cycle"])

    enterprise_value = normalized_ebit * multiple
    equity = enterprise_value - (net_debt or 0.0)

    position = ""
    if current_margin is not None and math.isfinite(current_margin):
        gap = current_margin - normalized_margin
        if gap > 0.03:
            position = " — currently earning above its mid-cycle margin"
        elif gap < -0.03:
            position = " — currently earning below its mid-cycle margin"

    return ModelResult(
        value_per_share=equity / shares,
        method="normalized_earnings",
        label="Normalised mid-cycle earnings",
        detail=(
            f"revenue at a mid-cycle margin of {normalized_margin:.1%} "
            f"capitalised at {multiple:.0f}x{position}"
        ),
        assumptions={
            "revenue": revenue,
            "normalized_margin": normalized_margin,
            "current_margin": current_margin,
            "exit_multiple": multiple,
            "normalized_ebit": normalized_ebit,
            "tax_rate": tax_rate,
            "net_debt": net_debt,
        },
        warnings=warnings,
    )
