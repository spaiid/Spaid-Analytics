"""Does the ranking carry information, separately from whether it made money?

A portfolio result is a single path. It can be good because the ordering was
informative, or because two holdings were extraordinary, or because the whole
thing was a bet on one factor that happened to work. These diagnostics exist to
tell those apart, and they are computed on the *whole cross-section* rather than
on the twenty names that were actually bought, because twenty names is far too
small a sample to learn anything from.

The headline is the rank information coefficient: within a single date, the
Spearman correlation between the score and the return that followed. Averaged
over dates it says whether the ordering was informative. Two details decide
whether it says it honestly:

* **It is computed within a date, never pooled.** Pooling dates would let the
  market's own ups and downs dominate the correlation, and the result would
  measure the calendar rather than the ranking.
* **Its t-statistic is computed on non-overlapping blocks.** Monthly
  observations of a twelve-month forward return share eleven twelfths of their
  window. Treating them as independent inflates the t-statistic by roughly the
  square root of the overlap, which is how a signal with no significance
  acquires a t-statistic of four.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from itertools import pairwise

import numpy as np
import polars as pl

from spaid.config.scoring import Category, ScoringSpec

log = logging.getLogger(__name__)

CATEGORY_COLUMNS: tuple[str, ...] = tuple(c.value for c in Category)

# Observations needed before an expanding quantile means anything. Below this
# the regime is left undefined rather than guessed from a handful of dates.
_MIN_REGIME_HISTORY = 10


def _expanding_bands(
    values: np.ndarray, *, low: float = 33.0, high: float = 67.0
) -> tuple[np.ndarray, np.ndarray]:
    """Tercile boundaries computed only from what had already happened.

    Taking `np.percentile` over the whole sample decides which dates count as
    low-volatility using the volatility of dates that had not happened yet. It
    is a subtle look-ahead, and it landed on the single strongest number in the
    validation artifacts -- a low-volatility information coefficient of +0.079
    with a t-statistic of 3.23, the only t above 2 anywhere in the run. A fifth
    of the development dates change bucket once the threshold can only see the
    past, so that figure was not reproducible out of sample.

    Returns NaN bands until `_MIN_REGIME_HISTORY` observations exist, which
    leaves those dates in the middle bucket.
    """
    lows = np.full(values.size, np.nan)
    highs = np.full(values.size, np.nan)
    for i in range(values.size):
        seen = values[: i + 1]
        seen = seen[np.isfinite(seen)]
        if seen.size >= _MIN_REGIME_HISTORY:
            lows[i] = np.percentile(seen, low)
            highs[i] = np.percentile(seen, high)
    return lows, highs


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties sharing the mean of the positions they occupy."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    sorted_a = a[order]
    i = 0
    while i < sorted_a.size:
        j = i
        while j + 1 < sorted_a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    rx = _ranks(x[mask])
    ry = _ranks(y[mask])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = math.sqrt(float((rx**2).sum() * (ry**2).sum()))
    return float(rx @ ry / denominator) if denominator > 0 else float("nan")


def ic_by_date(
    panel: pl.DataFrame, score_col: str, target_col: str, *, min_names: int = 20
) -> pl.DataFrame:
    """One rank correlation per decision date."""
    rows: list[tuple] = []
    for (d,), group in panel.group_by(["date"], maintain_order=True):
        x = group[score_col].to_numpy().astype(float)
        y = group[target_col].to_numpy().astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_names:
            continue
        rows.append((d, spearman(x[mask], y[mask]), int(mask.sum())))
    if not rows:
        return pl.DataFrame(schema={"date": pl.Date, "ic": pl.Float64, "n": pl.Int64})
    return pl.DataFrame(rows, schema=["date", "ic", "n"], orient="row").sort("date")


def ic_summary(ic: pl.DataFrame, *, horizon_months: int) -> dict:
    """Mean, volatility, hit rate and a t-statistic that respects the overlap.

    With monthly decision dates and an `h`-month forward return, consecutive
    observations overlap by `h - 1` months. The t-statistic is therefore
    computed on the mean of each non-overlapping block of `h` dates, which
    costs precision and buys honesty.
    """
    empty = {
        "ic_mean": float("nan"), "ic_median": float("nan"), "ic_std": float("nan"),
        "ic_t": float("nan"), "hit_rate": float("nan"), "n_dates": 0, "n_blocks": 0,
        "ic_ir": float("nan"),
    }
    if ic.is_empty():
        return empty
    values = ic["ic"].to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return empty

    blocks = np.array(
        [
            values[i : i + horizon_months].mean()
            for i in range(0, values.size, max(1, horizon_months))
        ]
    )
    if blocks.size > 2 and blocks.std(ddof=1) > 0:
        t = float(blocks.mean() / (blocks.std(ddof=1) / math.sqrt(blocks.size)))
    else:
        t = float("nan")

    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return {
        "ic_mean": float(values.mean()),
        "ic_median": float(np.median(values)),
        "ic_std": std,
        # The information ratio of the signal itself: mean IC over its own
        # volatility. A high average IC that swings between +0.3 and -0.3 is a
        # different object from a steady +0.03.
        "ic_ir": float(values.mean() / std) if std and np.isfinite(std) and std > 0 else float("nan"),
        "ic_t": t,
        "hit_rate": float((values > 0).mean()),
        "n_dates": int(values.size),
        "n_blocks": int(blocks.size),
    }


def ic_across_horizons(
    panel: pl.DataFrame, horizons: tuple[int, ...], *, score_col: str = "score"
) -> dict[int, dict]:
    """The information coefficient at each forward horizon, with its series."""
    out: dict[int, dict] = {}
    for months in horizons:
        target = f"fwd_excess_{months}m"
        if target not in panel.columns:
            continue
        series = ic_by_date(panel, score_col, target)
        summary = ic_summary(series, horizon_months=months)
        summary["series"] = [
            {"date": str(d), "ic": float(v), "n": int(n)}
            for d, v, n in series.iter_rows()
        ]
        out[months] = summary
    return out


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------


def bucket_returns(
    panel: pl.DataFrame,
    *,
    n_buckets: int,
    horizon_months: int,
    score_col: str = "score",
) -> pl.DataFrame:
    """Mean forward return by score bucket, assigned within each date.

    Bucketing within the date matters: assigning deciles across the pooled
    panel would put most of 2020 in one decile and most of 2022 in another, and
    the resulting "decile returns" would describe the years, not the scores.
    """
    target = f"fwd_excess_{horizon_months}m"
    raw_target = f"fwd_{horizon_months}m"
    if target not in panel.columns:
        return pl.DataFrame()

    ranked = panel.filter(
        pl.col(score_col).is_not_null() & pl.col(target).is_not_null()
    ).with_columns(
        (
            (pl.col(score_col).rank("average").over("date") - 0.5)
            / pl.len().over("date")
            * n_buckets
        )
        .floor()
        .clip(0, n_buckets - 1)
        .cast(pl.Int32)
        .alias("bucket")
    )
    if ranked.is_empty():
        return pl.DataFrame()

    return (
        ranked.group_by("bucket")
        .agg(
            pl.col(target).mean().alias("mean_excess"),
            pl.col(target).median().alias("median_excess"),
            pl.col(target).std().alias("std_excess"),
            pl.col(raw_target).mean().alias("mean_total")
            if raw_target in ranked.columns
            else pl.lit(None).alias("mean_total"),
            (pl.col(target) > 0).mean().alias("hit_rate"),
            pl.col(score_col).mean().alias("mean_score"),
            pl.len().alias("n"),
        )
        .sort("bucket")
    )


def monotonicity(buckets: pl.DataFrame) -> float:
    """Spearman correlation between bucket index and mean forward return.

    1.0 means every step up in score bought a higher return. A signal that
    works only at the top and bottom with nothing in between scores low here,
    which is the point: two extreme buckets can be produced by a handful of
    outliers, a monotone staircase cannot.
    """
    if buckets.is_empty() or buckets.height < 3:
        return float("nan")
    return spearman(
        buckets["bucket"].to_numpy().astype(float),
        buckets["mean_excess"].to_numpy().astype(float),
    )


def spread(buckets: pl.DataFrame) -> float:
    """Top bucket minus bottom bucket, in mean forward excess return."""
    if buckets.is_empty() or buckets.height < 2:
        return float("nan")
    ordered = buckets.sort("bucket")
    return float(ordered["mean_excess"][-1] - ordered["mean_excess"][0])


def band_returns(panel: pl.DataFrame, *, horizon_months: int) -> pl.DataFrame:
    """Forward returns by the score band the interface actually displays.

    The bands are absolute cutoffs, not relative ones, so this answers the
    question a user of the app would ask: when the screen said "Very
    attractive", what happened next?
    """
    target = f"fwd_excess_{horizon_months}m"
    if "band" not in panel.columns or target not in panel.columns:
        return pl.DataFrame()
    return (
        panel.filter(pl.col("band").is_not_null() & pl.col(target).is_not_null())
        .group_by("band")
        .agg(
            pl.col(target).mean().alias("mean_excess"),
            pl.col(target).median().alias("median_excess"),
            (pl.col(target) > 0).mean().alias("hit_rate"),
            pl.col("score").mean().alias("mean_score"),
            pl.len().alias("n"),
        )
        .sort("mean_score", descending=True)
    )


# ---------------------------------------------------------------------------
# Cuts
# ---------------------------------------------------------------------------


def ic_by_group(
    panel: pl.DataFrame,
    group_col: str,
    *,
    horizon_months: int,
    score_col: str = "score",
    min_dates: int = 12,
) -> pl.DataFrame:
    """The information coefficient computed separately inside each group.

    Within a sector the ranking is against sector peers only, which is a harder
    and more useful test than the whole-market one: a score that works only
    because it sorts technology above utilities is not selecting companies.
    """
    target = f"fwd_excess_{horizon_months}m"
    if group_col not in panel.columns or target not in panel.columns:
        return pl.DataFrame()

    rows: list[dict] = []
    for (group,), frame in panel.group_by([group_col], maintain_order=True):
        if group is None:
            continue
        series = ic_by_date(frame, score_col, target, min_names=10)
        if series.height < min_dates:
            continue
        summary = ic_summary(series, horizon_months=horizon_months)
        rows.append(
            {
                "group": str(group),
                "ic_mean": summary["ic_mean"],
                "ic_t": summary["ic_t"],
                "hit_rate": summary["hit_rate"],
                "n_dates": summary["n_dates"],
                "n_obs": int(frame.filter(pl.col(target).is_not_null()).height),
            }
        )
    return pl.DataFrame(rows).sort("ic_mean", descending=True) if rows else pl.DataFrame()


def add_size_buckets(panel: pl.DataFrame, *, n: int = 5) -> pl.DataFrame:
    """Quintiles of market capitalisation, assigned within each date."""
    if "market_cap" not in panel.columns:
        return panel.with_columns(pl.lit(None, dtype=pl.Utf8).alias("size_bucket"))
    labels = {0: "1 smallest", 1: "2", 2: "3", 3: "4", 4: "5 largest"}
    return panel.with_columns(
        (
            (pl.col("market_cap").rank("average").over("date") - 0.5)
            / pl.col("market_cap").is_not_null().sum().over("date")
            * n
        )
        .floor()
        .clip(0, n - 1)
        .cast(pl.Int32)
        .replace_strict(labels, default=None)
        .alias("size_bucket")
    )


def add_calendar_year(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.with_columns(pl.col("date").dt.year().cast(pl.Utf8).alias("calendar_year"))


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------


def classify_regimes(
    prices: pl.DataFrame,
    decision_dates: list[date],
    *,
    benchmark: str = "SPY",
) -> pl.DataFrame:
    """Label each decision date with the market environment it sat in.

    Every label is computed from data available *on* that date -- a trailing
    average, a drawdown from the running peak, a trailing volatility, the
    prevailing yield. None of them looks forward, because a regime label built
    with hindsight would sort the dates by what happened next and every
    "regime result" would be circular.
    """
    from spaid.providers.fred import rates as fred_rates

    market = (
        prices.filter(pl.col("ticker") == benchmark)
        .select(["date", "close_adj"])
        .sort("date")
        .with_columns(
            pl.col("close_adj").rolling_mean(200).alias("ma200"),
            pl.col("close_adj").cum_max().alias("peak"),
            (pl.col("close_adj") / pl.col("close_adj").shift(1) - 1.0).alias("ret"),
        )
        .with_columns(
            pl.col("ret").rolling_std(63).alias("vol63"),
            (pl.col("ma200") / pl.col("ma200").shift(63) - 1.0).alias("ma200_slope"),
            (pl.col("close_adj") / pl.col("peak") - 1.0).alias("from_peak"),
        )
    )

    at_dates = market.filter(pl.col("date").is_in(decision_dates))
    if at_dates.is_empty():
        return pl.DataFrame()

    vol = at_dates["vol63"].to_numpy()
    vol_low, vol_high = _expanding_bands(vol)

    out = at_dates.with_columns(
        pl.Series("_vol_low", vol_low), pl.Series("_vol_high", vol_high)
    ).with_columns(
        pl.when(pl.col("from_peak") <= -0.10)
        .then(pl.lit("bear"))
        .when((pl.col("close_adj") > pl.col("ma200")) & (pl.col("ma200_slope") > 0))
        .then(pl.lit("bull"))
        .otherwise(pl.lit("neutral"))
        .alias("trend_regime"),
        pl.when(pl.col("vol63") >= pl.col("_vol_high"))
        .then(pl.lit("high_volatility"))
        .when(pl.col("vol63") <= pl.col("_vol_low"))
        .then(pl.lit("low_volatility"))
        .otherwise(pl.lit("mid_volatility"))
        .alias("volatility_regime"),
    ).drop("_vol_low", "_vol_high")

    yields = fred_rates.load()
    if yields is not None and not yields.is_empty():
        joined = (
            out.sort("date")
            .join_asof(
                yields.select(["date", pl.col("value").alias("yield_10y")]).sort("date"),
                on="date",
                strategy="backward",
            )
        )
        rate = joined["yield_10y"].to_numpy()
        finite_rate = rate[np.isfinite(rate)]
        if finite_rate.size > 10:
            rate_low, rate_high = _expanding_bands(rate)
            out = joined.with_columns(
                pl.Series("_rate_low", rate_low), pl.Series("_rate_high", rate_high)
            ).with_columns(
                pl.when(pl.col("yield_10y") >= pl.col("_rate_high"))
                .then(pl.lit("high_rate"))
                .when(pl.col("yield_10y") <= pl.col("_rate_low"))
                .then(pl.lit("low_rate"))
                .otherwise(pl.lit("mid_rate"))
                .alias("rate_regime")
            )
        else:
            out = joined.with_columns(pl.lit(None, dtype=pl.Utf8).alias("rate_regime"))
    else:
        out = out.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("yield_10y"),
            pl.lit(None, dtype=pl.Utf8).alias("rate_regime"),
        )

    return out.select(
        ["date", "trend_regime", "volatility_regime", "rate_regime", "yield_10y",
         "from_peak", "vol63"]
    )


# ---------------------------------------------------------------------------
# Categories and ablations
# ---------------------------------------------------------------------------


def category_ic(
    panel: pl.DataFrame, *, horizons: tuple[int, ...]
) -> pl.DataFrame:
    """Each category scored on its own, as a predictor in its own right.

    Quality, growth, momentum and valuation are measured separately because the
    composite can look informative while being carried by one of them, and
    because a category with a negative information coefficient is evidence
    against its weight rather than noise to be averaged away.
    """
    rows: list[dict] = []
    for column in CATEGORY_COLUMNS:
        if column not in panel.columns:
            continue
        for months in horizons:
            target = f"fwd_excess_{months}m"
            if target not in panel.columns:
                continue
            series = ic_by_date(panel, column, target)
            summary = ic_summary(series, horizon_months=months)
            rows.append(
                {
                    "category": column,
                    "horizon_months": months,
                    "ic_mean": summary["ic_mean"],
                    "ic_t": summary["ic_t"],
                    "ic_ir": summary["ic_ir"],
                    "hit_rate": summary["hit_rate"],
                    "n_dates": summary["n_dates"],
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def recompose(panel: pl.DataFrame, spec: ScoringSpec, *, drop: str | None = None) -> pl.Series:
    """Rebuild the composite from the stored category scores, optionally dropping one.

    This is the same arithmetic the scoring pipeline performs -- a weighted mean
    over the categories that produced a score, renormalised -- so an ablation is
    a genuine "what if this category had never existed" rather than a different
    model fitted without it.
    """
    weights = {c.value: w for c, w in spec.category_weights.items()}
    contribution = pl.lit(0.0)
    total = pl.lit(0.0)
    for column, weight in weights.items():
        if column == drop or column not in panel.columns:
            continue
        present = pl.col(column).is_not_null()
        contribution = contribution + pl.when(present).then(pl.col(column) * weight).otherwise(0.0)
        total = total + pl.when(present).then(pl.lit(weight)).otherwise(0.0)
    composed = (
        panel.with_columns(contribution.alias("_c"), total.alias("_t"))
        .with_columns(
            pl.when(pl.col("_t") >= spec.min_total_coverage)
            .then(pl.col("_c") / pl.col("_t"))
            .otherwise(None)
            .alias("_score")
        )
    )
    return composed["_score"]


def ablations(
    panel: pl.DataFrame, spec: ScoringSpec, *, horizon_months: int = 3
) -> pl.DataFrame:
    """What the composite's information coefficient becomes without each category.

    A category whose removal *improves* the signal is carrying weight it has not
    earned. That is a finding to report, not to act on in this milestone: the
    weights stay frozen until the baseline conclusion exists.
    """
    target = f"fwd_excess_{horizon_months}m"
    if target not in panel.columns:
        return pl.DataFrame()

    base_series = ic_by_date(panel, "score", target)
    base = ic_summary(base_series, horizon_months=horizon_months)

    rows = [
        {
            "removed": "nothing (full composite)",
            "ic_mean": base["ic_mean"],
            "ic_t": base["ic_t"],
            "delta_ic": 0.0,
            "hit_rate": base["hit_rate"],
            "n_dates": base["n_dates"],
        }
    ]
    for column in CATEGORY_COLUMNS:
        if column not in panel.columns:
            continue
        rebuilt = panel.with_columns(recompose(panel, spec, drop=column).alias("_ablated"))
        series = ic_by_date(rebuilt, "_ablated", target)
        summary = ic_summary(series, horizon_months=horizon_months)
        rows.append(
            {
                "removed": column,
                "ic_mean": summary["ic_mean"],
                "ic_t": summary["ic_t"],
                "delta_ic": summary["ic_mean"] - base["ic_mean"],
                "hit_rate": summary["hit_rate"],
                "n_dates": summary["n_dates"],
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Turnover and persistence
# ---------------------------------------------------------------------------


def score_persistence(panel: pl.DataFrame, *, score_col: str = "score") -> dict:
    """How stable the ranking is from one month to the next.

    A score that reshuffles completely every month implies turnover the cost
    model will punish; one that never moves implies the metrics are stale
    rather than stable. Both are visible here.
    """
    dates = sorted(panel["date"].unique().to_list())
    if len(dates) < 3:
        return {"autocorrelation": float("nan"), "n_dates": len(dates)}

    correlations: list[float] = []
    previous: pl.DataFrame | None = None
    for d in dates:
        current = panel.filter(pl.col("date") == d).select(["security_id", score_col])
        if previous is not None:
            merged = previous.join(current, on="security_id", how="inner", suffix="_now")
            if merged.height >= 20:
                correlations.append(
                    spearman(
                        merged[score_col].to_numpy().astype(float),
                        merged[f"{score_col}_now"].to_numpy().astype(float),
                    )
                )
        previous = current

    values = np.array([c for c in correlations if np.isfinite(c)])
    return {
        "autocorrelation": float(values.mean()) if values.size else float("nan"),
        "n_pairs": int(values.size),
        "n_dates": len(dates),
    }


def holding_periods(holdings: list[dict]) -> dict:
    """How long a name stays in the portfolio once bought, in rebalances."""
    if not holdings:
        return {"mean_months": float("nan"), "median_months": float("nan"), "n_spells": 0}

    frame = pl.DataFrame(holdings).select(["rebalance_date", "security_id"]).sort(
        ["security_id", "rebalance_date"]
    )
    dates = sorted(frame["rebalance_date"].unique().to_list())
    position = {d: i for i, d in enumerate(dates)}

    spells: list[int] = []
    for (_,), group in frame.group_by(["security_id"], maintain_order=True):
        indices = sorted(position[d] for d in group["rebalance_date"].to_list())
        run = 1
        for previous, current in pairwise(indices):
            if current == previous + 1:
                run += 1
            else:
                spells.append(run)
                run = 1
        spells.append(run)

    arr = np.array(spells, dtype=float)
    return {
        "mean_months": float(arr.mean()),
        "median_months": float(np.median(arr)),
        "max_months": int(arr.max()),
        "n_spells": int(arr.size),
        "share_single_month": float((arr == 1).mean()),
    }
