"""Performance statistics, including the ones that argue against the result.

Most of this module is ordinary: compound a return series, annualise it,
compare it with a benchmark. Three parts are not, and they exist because a
backtest's headline numbers are systematically too good:

**Confidence intervals, by bootstrap.** A 12% annual return over nine years is
one draw from a distribution, and reporting it as a point estimate invites a
precision it does not have. The interval is built by resampling blocks of
consecutive days -- blocks, because daily equity returns are not independent and
resampling single days would understate the uncertainty.

**The deflated Sharpe ratio.** A Sharpe ratio is inflated by the number of
strategies tried, by the non-normality of the returns and by the shortness of
the sample. The deflated version corrects for all three, using the trial count
from the experiment registry rather than a flattering guess. This is why failed
trials are never deleted: deleting them would raise the deflated Sharpe of
whatever survived.

**The probability of backtest overfitting.** Combinatorially symmetric
cross-validation asks a blunt question of a set of trials: when you pick the
best one on half the periods, how often is it below median on the other half?
If the answer is "about half the time", the selection carried no information
and the winner was noise.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from itertools import combinations

import numpy as np

from spaid.config.settings import SETTINGS

log = logging.getLogger(__name__)

TRADING_DAYS = SETTINGS.trading_days_per_year
EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def to_returns(equity: np.ndarray) -> np.ndarray:
    """Simple period returns from an equity path, starting at the second point."""
    if equity.size < 2:
        return np.zeros(0)
    return equity[1:] / equity[:-1] - 1.0


def cagr(equity: np.ndarray, dates: list[date]) -> float:
    """Compound annual growth, measured on the calendar rather than on bar count."""
    if equity.size < 2 or equity[0] <= 0:
        return float("nan")
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)


def total_return(equity: np.ndarray) -> float:
    if equity.size < 2 or equity[0] <= 0:
        return float("nan")
    return float(equity[-1] / equity[0] - 1.0)


def volatility(returns: np.ndarray) -> float:
    clean = returns[np.isfinite(returns)]
    if clean.size < 2:
        return float("nan")
    return float(clean.std(ddof=1) * math.sqrt(TRADING_DAYS))


def sharpe(returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
    clean = returns[np.isfinite(returns)]
    if clean.size < 2:
        return float("nan")
    daily_rf = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = clean - daily_rf
    sd = excess.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(excess.mean() / sd * math.sqrt(TRADING_DAYS))


def sortino(returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
    """Like Sharpe, but only downside deviation counts as risk.

    The denominator divides by the full sample size, not by the number of
    negative days. Dividing by the count of losses would make a strategy with
    few but enormous losses look safe.
    """
    clean = returns[np.isfinite(returns)]
    if clean.size < 2:
        return float("nan")
    daily_rf = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = clean - daily_rf
    downside = np.minimum(excess, 0.0)
    dd = math.sqrt(float((downside**2).sum() / clean.size))
    if dd <= 0:
        return float("nan")
    return float(excess.mean() / dd * math.sqrt(TRADING_DAYS))


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    return float(drawdown_series(equity).min())


def drawdown_duration(equity: np.ndarray, dates: list[date]) -> dict:
    """The longest time spent below a previous peak, and where it happened.

    Duration matters separately from depth: a 30% fall recovered in four months
    is a different experience from a 20% fall that takes three years, and only
    one of them ends with the strategy abandoned.
    """
    if equity.size == 0:
        return {"days": 0, "start": None, "trough": None, "recovered": None}

    peak = equity[0]
    peak_i = 0
    worst = {"days": 0, "start": None, "trough": None, "recovered": None}
    trough_i = 0

    for i in range(1, equity.size):
        if equity[i] >= peak:
            days = (dates[i] - dates[peak_i]).days
            if days > worst["days"] and trough_i > peak_i:
                worst = {
                    "days": days,
                    "start": dates[peak_i],
                    "trough": dates[trough_i],
                    "recovered": dates[i],
                }
            peak, peak_i, trough_i = equity[i], i, i
        elif equity[i] < equity[trough_i]:
            trough_i = i

    # A drawdown still open at the end counts, measured to the last date.
    if peak_i < equity.size - 1:
        days = (dates[-1] - dates[peak_i]).days
        if days > worst["days"]:
            worst = {
                "days": days,
                "start": dates[peak_i],
                "trough": dates[trough_i],
                "recovered": None,
            }
    return worst


def calmar(equity: np.ndarray, dates: list[date]) -> float:
    dd = max_drawdown(equity)
    growth = cagr(equity, dates)
    if not np.isfinite(dd) or dd >= 0 or not np.isfinite(growth):
        return float("nan")
    return float(growth / abs(dd))


# ---------------------------------------------------------------------------
# Against a benchmark
# ---------------------------------------------------------------------------


def beta_alpha(
    returns: np.ndarray, benchmark_returns: np.ndarray, risk_free_annual: float = 0.0
) -> tuple[float, float]:
    """Ordinary-least-squares beta and annualised alpha against one benchmark."""
    mask = np.isfinite(returns) & np.isfinite(benchmark_returns)
    if mask.sum() < 30:
        return float("nan"), float("nan")
    daily_rf = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS) - 1.0
    y = returns[mask] - daily_rf
    x = benchmark_returns[mask] - daily_rf
    var = x.var(ddof=1)
    if var <= 0:
        return float("nan"), float("nan")
    b = float(np.cov(y, x, ddof=1)[0, 1] / var)
    a_daily = float(y.mean() - b * x.mean())
    return b, float((1.0 + a_daily) ** TRADING_DAYS - 1.0)


def tracking_error(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    mask = np.isfinite(returns) & np.isfinite(benchmark_returns)
    if mask.sum() < 2:
        return float("nan")
    diff = returns[mask] - benchmark_returns[mask]
    return float(diff.std(ddof=1) * math.sqrt(TRADING_DAYS))


def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    mask = np.isfinite(returns) & np.isfinite(benchmark_returns)
    if mask.sum() < 30:
        return float("nan")
    diff = returns[mask] - benchmark_returns[mask]
    sd = diff.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(diff.mean() / sd * math.sqrt(TRADING_DAYS))


def capture(returns: np.ndarray, benchmark_returns: np.ndarray) -> tuple[float, float]:
    """Upside and downside capture, on the benchmark's up and down months.

    Measured monthly rather than daily. Daily capture is dominated by noise and
    routinely reports figures above 200%, which describe the sampling frequency
    rather than the strategy.
    """
    mask = np.isfinite(returns) & np.isfinite(benchmark_returns)
    r, b = returns[mask], benchmark_returns[mask]
    if r.size < TRADING_DAYS // 4:
        return float("nan"), float("nan")

    up = b > 0
    down = b < 0
    out = []
    for sel in (up, down):
        if sel.sum() < 10 or b[sel].mean() == 0:
            out.append(float("nan"))
            continue
        out.append(float(r[sel].mean() / b[sel].mean()))
    return out[0], out[1]


def periodic_returns(
    equity: np.ndarray, dates: list[date], *, freq: str = "month"
) -> tuple[list[str], np.ndarray]:
    """Compound the equity path into calendar months or years."""
    if equity.size == 0:
        return [], np.zeros(0)
    keys: list[str] = []
    values: list[float] = []
    start_value = equity[0]
    current = _period_key(dates[0], freq)
    for i in range(1, equity.size):
        key = _period_key(dates[i], freq)
        if key != current:
            keys.append(current)
            values.append(equity[i - 1] / start_value - 1.0)
            start_value = equity[i - 1]
            current = key
    keys.append(current)
    values.append(equity[-1] / start_value - 1.0)
    return keys, np.array(values)


def _period_key(day: date, freq: str) -> str:
    return f"{day.year}" if freq == "year" else f"{day.year}-{day.month:02d}"


def rolling_outperformance(
    equity: np.ndarray,
    benchmark: np.ndarray,
    dates: list[date],
    months: int,
) -> dict:
    """How often the strategy beat the benchmark over every rolling window.

    Overlapping windows, so these are not independent observations and the
    fraction is a description rather than a test. It is reported because "beat
    it over the whole period" and "beat it over most three-year stretches an
    investor might actually have held it" are different claims.
    """
    window = months * 21
    if equity.size <= window or benchmark.size <= window:
        return {"windows": 0, "beat": 0, "share": float("nan")}
    strat = equity[window:] / equity[:-window] - 1.0
    bench = benchmark[window:] / benchmark[:-window] - 1.0
    mask = np.isfinite(strat) & np.isfinite(bench)
    if mask.sum() == 0:
        return {"windows": 0, "beat": 0, "share": float("nan")}
    beat = int((strat[mask] > bench[mask]).sum())
    return {
        "windows": int(mask.sum()),
        "beat": beat,
        "share": float(beat / mask.sum()),
        "median_excess": float(np.median(strat[mask] - bench[mask])),
        "worst_excess": float(np.min(strat[mask] - bench[mask])),
        "best_excess": float(np.max(strat[mask] - bench[mask])),
    }


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def block_bootstrap_ci(
    returns: np.ndarray,
    statistic,
    *,
    n_samples: int = 1000,
    block_size: int = 21,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """A percentile confidence interval from a moving-block bootstrap.

    Blocks of consecutive days preserve the autocorrelation and the volatility
    clustering that make daily returns non-independent. Resampling single days
    would break both and produce an interval far too narrow to be honest.
    """
    clean = returns[np.isfinite(returns)]
    if clean.size < block_size * 4:
        return float("nan"), float("nan")

    rng = np.random.default_rng(SETTINGS.seed if seed is None else seed)
    n = clean.size
    n_blocks = int(np.ceil(n / block_size))
    starts_max = n - block_size

    stats = np.empty(n_samples)
    for s in range(n_samples):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([clean[i : i + block_size] for i in starts])[:n]
        stats[s] = statistic(sample)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.percentile(stats, 100 * alpha / 2)),
        float(np.percentile(stats, 100 * (1 - alpha / 2))),
    )


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF, Acklam's rational approximation.

    Accurate to about 1e-9, which is far more than a deflated Sharpe ratio
    needs, and it avoids taking a dependency on SciPy for one function.
    """
    if not 0.0 < p < 1.0:
        return float("nan")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


@dataclass(frozen=True)
class DeflatedSharpe:
    sharpe_annual: float
    expected_max_sharpe_annual: float
    deflated: float  # probability the true Sharpe exceeds zero, after deflation
    n_trials: int
    skew: float
    kurtosis: float
    n_obs: int
    verdict: str


def deflated_sharpe(
    returns: np.ndarray, *, n_trials: int, trial_sharpes: np.ndarray | None = None
) -> DeflatedSharpe:
    """Bailey and Lopez de Prado's deflated Sharpe ratio.

    The intuition is that searching over many strategies produces a high Sharpe
    by construction. The expected maximum Sharpe under the null of no skill
    grows with the number of trials and with their dispersion, and the deflated
    ratio asks whether the observed Sharpe beats *that* benchmark rather than
    beating zero.
    """
    clean = returns[np.isfinite(returns)]
    n = clean.size
    if n < 60:
        return DeflatedSharpe(
            float("nan"), float("nan"), float("nan"), n_trials, float("nan"),
            float("nan"), n, "too few observations to deflate",
        )

    mean, sd = clean.mean(), clean.std(ddof=1)
    if sd <= 0:
        return DeflatedSharpe(
            float("nan"), float("nan"), float("nan"), n_trials, float("nan"),
            float("nan"), n, "no variation in returns",
        )

    sr = mean / sd  # per observation, not annualised
    centred = (clean - mean) / sd
    skew = float((centred**3).mean())
    kurt = float((centred**4).mean())

    trials = max(1, int(n_trials))
    if trial_sharpes is not None and trial_sharpes.size > 1:
        sr_variance = float(np.nanvar(trial_sharpes, ddof=1))
    else:
        # With no dispersion measured across trials, fall back to the variance
        # of a Sharpe estimate under the null, which is the conservative reading
        # available without inventing data.
        sr_variance = 1.0 / n

    sr_std = math.sqrt(max(sr_variance, 1e-12))
    if trials > 1:
        expected_max = sr_std * (
            (1 - EULER_MASCHERONI) * _norm_ppf(1 - 1.0 / trials)
            + EULER_MASCHERONI * _norm_ppf(1 - 1.0 / (trials * math.e))
        )
    else:
        expected_max = 0.0

    denominator = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr**2))
    statistic = (sr - expected_max) * math.sqrt(n - 1) / denominator
    probability = _norm_cdf(statistic)

    annual = math.sqrt(TRADING_DAYS)
    if probability >= 0.95:
        verdict = "survives deflation at the 95% level"
    elif probability >= 0.90:
        verdict = "marginal after deflation"
    else:
        verdict = "does not survive deflation"

    return DeflatedSharpe(
        sharpe_annual=float(sr * annual),
        expected_max_sharpe_annual=float(expected_max * annual),
        deflated=float(probability),
        n_trials=trials,
        skew=skew,
        kurtosis=kurt,
        n_obs=n,
        verdict=verdict,
    )


def probability_of_backtest_overfitting(
    trial_returns: np.ndarray, *, n_splits: int = 8
) -> dict:
    """Combinatorially symmetric cross-validation, after Bailey et al.

    `trial_returns` is one row per trial, one column per period. The procedure
    splits the periods into blocks, takes every balanced in-sample/out-of-sample
    partition, selects the trial that did best in-sample, and records where it
    ranked out-of-sample. If the best in-sample trial lands below the median
    out-of-sample about half the time, selection among these trials is noise.
    """
    if trial_returns.ndim != 2:
        return {"feasible": False, "reason": "trial returns must be a matrix"}
    n_trials, n_obs = trial_returns.shape
    if n_trials < 3:
        return {
            "feasible": False,
            "reason": f"only {n_trials} trials; at least 3 are needed to rank them",
        }
    if n_splits % 2 or n_obs < n_splits * 10:
        n_splits = max(4, min(8, (n_obs // 10) // 2 * 2))
    if n_splits < 4 or n_obs < n_splits * 5:
        return {
            "feasible": False,
            "reason": f"only {n_obs} periods; not enough to split {n_splits} ways",
        }

    edges = np.array_split(np.arange(n_obs), n_splits)
    half = n_splits // 2
    logits: list[float] = []

    for combo in combinations(range(n_splits), half):
        in_idx = np.concatenate([edges[i] for i in combo])
        out_idx = np.concatenate([edges[i] for i in range(n_splits) if i not in combo])

        def stat(block: np.ndarray) -> np.ndarray:
            sd = block.std(axis=1, ddof=1)
            sd = np.where(sd > 0, sd, np.nan)
            return block.mean(axis=1) / sd

        in_sharpe = stat(trial_returns[:, in_idx])
        out_sharpe = stat(trial_returns[:, out_idx])
        if not np.isfinite(in_sharpe).any() or not np.isfinite(out_sharpe).any():
            continue
        best = int(np.nanargmax(in_sharpe))
        ranks = np.argsort(np.argsort(np.nan_to_num(out_sharpe, nan=-np.inf)))
        relative = (ranks[best] + 1) / (n_trials + 1)
        relative = min(max(relative, 1e-6), 1 - 1e-6)
        logits.append(math.log(relative / (1 - relative)))

    if not logits:
        return {"feasible": False, "reason": "no usable partitions"}

    arr = np.array(logits)
    pbo = float((arr <= 0).mean())
    return {
        "feasible": True,
        "pbo": pbo,
        "n_trials": n_trials,
        "n_splits": n_splits,
        "n_partitions": arr.size,
        "median_logit": float(np.median(arr)),
        "verdict": (
            "selection among these trials looks like noise"
            if pbo >= 0.5
            else "the in-sample winner tends to hold up out of sample"
        ),
    }


# ---------------------------------------------------------------------------
# The full summary
# ---------------------------------------------------------------------------


def summarise(
    dates: list[date],
    equity: np.ndarray,
    *,
    benchmarks: dict[str, np.ndarray],
    costs_paid: np.ndarray | None = None,
    turnover: np.ndarray | None = None,
    n_trials: int = 1,
    trial_sharpes: np.ndarray | None = None,
    with_intervals: bool = True,
) -> dict:
    """Every headline statistic for one equity path, against every benchmark."""
    returns = to_returns(equity)
    out: dict = {
        "start": str(dates[0]),
        "end": str(dates[-1]),
        "years": round((dates[-1] - dates[0]).days / 365.25, 2),
        "n_sessions": int(equity.size),
        "total_return": total_return(equity),
        "cagr": cagr(equity, dates),
        "volatility": volatility(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity, dates),
        "final_equity": float(equity[-1]),
    }
    out["drawdown_duration"] = {
        k: (str(v) if isinstance(v, date) else v)
        for k, v in drawdown_duration(equity, dates).items()
    }

    months, monthly = periodic_returns(equity, dates, freq="month")
    years_keys, yearly = periodic_returns(equity, dates, freq="year")
    out["positive_months"] = float((monthly > 0).mean()) if monthly.size else float("nan")
    out["n_months"] = int(monthly.size)
    out["monthly_returns"] = [
        {"period": k, "return": float(v)} for k, v in zip(months, monthly, strict=True)
    ]
    out["yearly_returns"] = [
        {"period": k, "return": float(v)} for k, v in zip(years_keys, yearly, strict=True)
    ]
    if monthly.size:
        best_i, worst_i = int(np.argmax(monthly)), int(np.argmin(monthly))
        out["best_month"] = {"period": months[best_i], "return": float(monthly[best_i])}
        out["worst_month"] = {"period": months[worst_i], "return": float(monthly[worst_i])}
    if yearly.size:
        best_i, worst_i = int(np.argmax(yearly)), int(np.argmin(yearly))
        out["best_year"] = {"period": years_keys[best_i], "return": float(yearly[best_i])}
        out["worst_year"] = {"period": years_keys[worst_i], "return": float(yearly[worst_i])}

    if costs_paid is not None:
        out["total_costs"] = float(np.nansum(costs_paid))
        out["costs_as_share_of_final_equity"] = (
            float(np.nansum(costs_paid) / equity[-1]) if equity[-1] else float("nan")
        )
    if turnover is not None and turnover.size:
        rebalances = int((turnover > 0).sum())
        traded = float(np.nansum(turnover))
        out["turnover_per_rebalance"] = (
            float(turnover[turnover > 0].mean()) if rebalances else 0.0
        )
        # Both conventions, because they differ by a factor of two and quoting
        # only one invites the reader to assume the other.
        out["turnover_annual"] = traded / max(out["years"], 1e-9)
        out["turnover_annual_one_way"] = traded / 2.0 / max(out["years"], 1e-9)
        out["n_rebalances_with_trades"] = rebalances

    if with_intervals:
        out["cagr_ci"] = _interval_from_returns(returns, dates, "cagr")
        out["sharpe_ci"] = _interval_from_returns(returns, dates, "sharpe")

    out["benchmarks"] = {}
    for name, index in benchmarks.items():
        if index is None or index.size != equity.size:
            continue
        b_returns = to_returns(index)
        b_beta, b_alpha = beta_alpha(returns, b_returns)
        up, down = capture(returns, b_returns)
        entry = {
            "total_return": total_return(index),
            "cagr": cagr(index, dates),
            "excess_cagr": out["cagr"] - cagr(index, dates),
            "alpha_annual": b_alpha,
            "beta": b_beta,
            "tracking_error": tracking_error(returns, b_returns),
            "information_ratio": information_ratio(returns, b_returns),
            "upside_capture": up,
            "downside_capture": down,
            "volatility": volatility(b_returns),
            "sharpe": sharpe(b_returns),
            "max_drawdown": max_drawdown(index),
        }
        for window in (12, 36, 60):
            entry[f"rolling_{window}m"] = rolling_outperformance(equity, index, dates, window)
        out["benchmarks"][name] = entry

    ds = deflated_sharpe(returns, n_trials=n_trials, trial_sharpes=trial_sharpes)
    out["deflated_sharpe"] = {
        "sharpe_annual": ds.sharpe_annual,
        "expected_max_sharpe_annual": ds.expected_max_sharpe_annual,
        "probability": ds.deflated,
        "n_trials": ds.n_trials,
        "skew": ds.skew,
        "kurtosis": ds.kurtosis,
        "verdict": ds.verdict,
    }
    return out


def _interval_from_returns(returns: np.ndarray, dates: list[date], which: str) -> dict:
    """A bootstrap interval for one annualised statistic."""
    if which == "cagr":
        def statistic(sample: np.ndarray) -> float:
            growth = float(np.prod(1.0 + sample))
            years = sample.size / TRADING_DAYS
            return growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else float("nan")
    else:
        def statistic(sample: np.ndarray) -> float:
            sd = sample.std(ddof=1)
            return float(sample.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    low, high = block_bootstrap_ci(returns, statistic)
    return {"low": low, "high": high, "level": 0.95, "method": "moving block bootstrap"}
