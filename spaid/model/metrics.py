"""Signal diagnostics.

The headline number for a cross-sectional ranker is the **rank information
coefficient**: the Spearman correlation, computed within a single date, between
predicted rank and realised forward excess return. Averaged over dates it says
whether the ordering carries information.

Its t-statistic needs care. Consecutive dates share almost all of their target
window -- with a 21-day horizon, today's and tomorrow's targets overlap by 20
days -- so daily IC observations are anything but independent. Treating them as
independent inflates the t-stat by roughly sqrt(21). Every t-stat here is
therefore computed on non-overlapping blocks, and that choice is surfaced in
the UI rather than buried.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    if n < 5:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    sa = a[order]
    i = 0
    while i < sa.size:
        j = i
        while j + 1 < sa.size and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def ic_by_date(
    df: pl.DataFrame, pred_col: str = "score", target_col: str = "target"
) -> pl.DataFrame:
    """Per-date rank IC."""
    rows = []
    for (d,), g in df.group_by(["date"], maintain_order=True):
        p = g[pred_col].to_numpy()
        y = g[target_col].to_numpy()
        mask = np.isfinite(p) & np.isfinite(y)
        if mask.sum() < 20:
            continue
        rows.append((d, _spearman(p[mask], y[mask]), int(mask.sum())))
    if not rows:
        return pl.DataFrame(schema={"date": pl.Date, "ic": pl.Float64, "n": pl.Int64})
    return pl.DataFrame(rows, schema=["date", "ic", "n"], orient="row")


def ic_summary(ic: pl.DataFrame, horizon: int) -> dict:
    """Mean IC plus a t-stat computed on non-overlapping blocks."""
    if ic.is_empty():
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_t": float("nan"),
                "hit_rate": float("nan"), "n_dates": 0, "n_blocks": 0}

    vals = ic["ic"].to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_t": float("nan"),
                "hit_rate": float("nan"), "n_dates": 0, "n_blocks": 0}

    # one observation per non-overlapping horizon block
    blocks = np.array([vals[i : i + horizon].mean() for i in range(0, vals.size, horizon)])
    t = (
        float(blocks.mean() / (blocks.std(ddof=1) / np.sqrt(blocks.size)))
        if blocks.size > 2 and blocks.std(ddof=1) > 0
        else float("nan")
    )
    return {
        "ic_mean": float(vals.mean()),
        "ic_std": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
        "ic_t": t,
        "hit_rate": float((vals > 0).mean()),
        "n_dates": int(vals.size),
        "n_blocks": int(blocks.size),
    }


def feature_ic(
    panel: pl.DataFrame, feature_cols: list[str], target_col: str = "target", horizon: int = 21
) -> pl.DataFrame:
    """IC of every individual feature, so the UI can show what is pulling weight."""
    out = []
    sub = panel.filter(pl.col(target_col).is_not_null())
    for col in feature_cols:
        raw = col[2:] if col.startswith("z_") else col
        has_col = f"has_{raw}"
        cov = float(sub[has_col].mean()) if has_col in sub.columns else float("nan")
        # only score a feature where it is actually observed
        scoped = sub.filter(pl.col(has_col) > 0) if has_col in sub.columns else sub
        if scoped.height < 500:
            continue
        s = ic_summary(ic_by_date(scoped, col, target_col), horizon)
        out.append({"feature": raw, "coverage": cov, **s})
    return pl.DataFrame(out) if out else pl.DataFrame()


def decile_returns(
    df: pl.DataFrame, pred_col: str = "score", target_col: str = "target", n_bins: int = 10
) -> pl.DataFrame:
    """Mean forward excess return by predicted decile — the monotonicity check."""
    ranked = (
        df.filter(pl.col(target_col).is_not_null())
        .with_columns(
            (
                (pl.col(pred_col).rank("average").over("date") - 0.5)
                / pl.len().over("date")
                * n_bins
            )
            .floor()
            .clip(0, n_bins - 1)
            .cast(pl.Int32)
            .alias("decile")
        )
    )
    return (
        ranked.group_by("decile")
        .agg(
            pl.col(target_col).mean().alias("mean_excess"),
            pl.col(target_col).median().alias("median_excess"),
            pl.len().alias("n"),
        )
        .sort("decile")
    )


def ic_decay(
    panel: pl.DataFrame, prices: pl.DataFrame, pred_col: str, horizons: list[int]
) -> pl.DataFrame:
    """How long the signal stays informative as the holding period extends."""
    from spaid.data.universe import benchmark_ticker

    bench = benchmark_ticker()
    mkt = prices.filter(pl.col("ticker") == bench).select(["date", "close"]).sort("date")

    rows = []
    for h in horizons:
        m = mkt.with_columns(
            (pl.col("close").shift(-h) / pl.col("close") - 1.0).alias("m_fwd")
        ).select(["date", "m_fwd"])
        d = (
            panel.sort(["ticker", "date"])
            .with_columns(
                (pl.col("close").shift(-h).over("ticker") / pl.col("close") - 1.0).alias("f")
            )
            .join(m, on="date", how="left")
            .with_columns((pl.col("f") - pl.col("m_fwd")).alias("t_h"))
            .filter(pl.col("t_h").is_not_null())
        )
        s = ic_summary(ic_by_date(d, pred_col, "t_h"), h)
        rows.append({"horizon": h, "ic": s["ic_mean"], "ic_t": s["ic_t"]})
    return pl.DataFrame(rows)


def group_correlation(panel: pl.DataFrame, group_cols: list[str]) -> tuple[list[str], list[list[float]]]:
    """Correlation between group scores — near-duplicate groups are not independent evidence."""
    present = [c for c in group_cols if c in panel.columns]
    if len(present) < 2:
        return [], []
    mat = panel.select(present).drop_nulls().to_numpy()
    if mat.shape[0] < 100:
        return [], []
    c = np.corrcoef(mat, rowvar=False)
    names = [p.replace("grp_", "") for p in present]
    return names, [[float(x) for x in row] for row in c]
