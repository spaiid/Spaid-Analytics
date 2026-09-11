"""Turning 37 features into one coherent, explainable score.

The composite is deliberately additive:

    group_score_g = mean of the direction-adjusted z-scores in group g,
                    averaged only over the features actually observed
    composite     = sum_g  weight_g * group_score_g

Because it is a plain weighted sum, the attribution the dashboard shows is not
an approximation of the model -- it *is* the model. `weight_g * group_score_g`
for each group sums exactly to the composite, so "why is this name ranked here"
has an arithmetic answer rather than a post-hoc story.

Weights come from each group's information coefficient measured **on training
data only**, shrunk hard toward equal weighting. Shrinkage matters: with eight
groups and a decade of overlapping observations, unshrunk IC weights chase
noise and reliably underperform the equal-weight blend they came from.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from spaid.features.registry import FEATURES, groups as feature_groups, in_group
from spaid.model.metrics import ic_by_date

log = logging.getLogger(__name__)

SHRINK = 1.0          # scale on signed IC weights; shrinks toward "no view"
MIN_T = 2.0           # trailing |t| a group must clear before it earns any weight
MIN_IC_OBS = 250      # trailing IC dates required before a group is eligible
MIN_GROUP_COVERAGE = 0.34   # a group needs a third of its features to count


def group_column(group: str) -> str:
    return "grp_" + group.lower().replace(" ", "_")


@dataclass
class ScoreModel:
    """Fitted group weights plus the diagnostics behind them."""

    weights: dict[str, float] = field(default_factory=dict)
    group_ic: dict[str, float] = field(default_factory=dict)
    trained_through: str | None = None
    n_train: int = 0
    group_t: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "group_ic": self.group_ic,
            "group_t": self.group_t,
            "trained_through": self.trained_through,
            "n_train": self.n_train,
        }

    @property
    def active_groups(self) -> list[str]:
        return [c for c, w in self.weights.items() if abs(w) > 1e-9]


def add_group_scores(panel: pl.DataFrame) -> pl.DataFrame:
    """Average each group's observed z-scores, and record its coverage.

    Averaging only over observed features is the point. Filling a missing
    feature with zero and then averaging would pull a sparsely-covered name
    toward the middle of every group it is thin on, which reads as "average
    evidence" when the truth is "no evidence".
    """
    out = panel
    for g in feature_groups():
        feats = [f.name for f in in_group(g) if f"z_{f.name}" in panel.columns]
        if not feats:
            continue
        z_sum = pl.sum_horizontal([pl.col(f"z_{f}") * pl.col(f"has_{f}") for f in feats])
        n_obs = pl.sum_horizontal([pl.col(f"has_{f}") for f in feats])
        col = group_column(g)
        out = out.with_columns(
            [
                (n_obs / len(feats)).alias(f"cov_{col}"),
                pl.when(n_obs >= max(1.0, len(feats) * MIN_GROUP_COVERAGE))
                .then(z_sum / n_obs)
                .otherwise(None)
                .alias(col),
            ]
        )
    return out


def precompute_group_ic(panel: pl.DataFrame) -> pl.DataFrame:
    """Per-date IC for every group, computed once for the whole panel.

    Refitting weights each fold by re-scanning the entire expanding training
    window is quadratic and takes tens of minutes over 125 folds. The IC at a
    given date depends only on that date's cross-section, so it can be computed
    once and averaged per fold instead.

    This stays point-in-time: a fold only ever averages IC dates at or before
    its training cutoff, and that cutoff already sits a full purge-plus-embargo
    gap before the test window, so those targets resolved well before the test
    period opens.
    """
    gcols = [group_column(g) for g in feature_groups() if group_column(g) in panel.columns]
    scoped = panel.filter(pl.col("target").is_not_null())

    frames = []
    for c in gcols:
        sub = scoped.filter(pl.col(c).is_not_null())
        if sub.height < 1000:
            continue
        ic = ic_by_date(sub, c, "target")
        if ic.is_empty():
            continue
        frames.append(ic.select(["date", pl.col("ic").alias(c)]))

    if not frames:
        return pl.DataFrame(schema={"date": pl.Date})
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="date", how="full", coalesce=True)
    return out.sort("date")


def _block_t(vals: np.ndarray, horizon: int = 21) -> float:
    """t-statistic of a mean IC, on non-overlapping blocks.

    Daily IC observations share almost all of their target window, so treating
    them as independent overstates significance by roughly sqrt(horizon). This
    gate is the only thing standing between the model and noise-chasing, so it
    has to use the honest denominator.
    """
    vals = vals[np.isfinite(vals)]
    if vals.size < horizon * 3:
        return 0.0
    blocks = np.array([vals[i : i + horizon].mean() for i in range(0, vals.size, horizon)])
    sd = blocks.std(ddof=1)
    if blocks.size < 3 or sd <= 0:
        return 0.0
    return float(blocks.mean() / (sd / np.sqrt(blocks.size)))


def fit_weights_from_ic(
    ic_table: pl.DataFrame,
    train_end,
    *,
    shrink: float = SHRINK,
    n_train: int = 0,
    min_t: float = MIN_T,
    horizon: int = 21,
) -> ScoreModel:
    """Average precomputed IC up to `train_end` into signed, gated weights.

    A group only earns weight when its trailing IC is statistically
    distinguishable from zero. Without that gate the weights chase noise: group
    ICs here run around 0.005-0.02, small enough that their trailing estimates
    flip sign five to eleven times across the walk-forward, and a weight that
    keeps reversing is worse than no weight at all because it sells what it
    just bought.
    """
    gcols = [c for c in ic_table.columns if c != "date"]
    hist = ic_table.filter(pl.col("date") <= train_end)

    ics: dict[str, float] = {}
    tstats: dict[str, float] = {}
    for c in gcols:
        vals = hist[c].drop_nulls().to_numpy() if c in hist.columns else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size < MIN_IC_OBS:
            ics[c], tstats[c] = 0.0, 0.0
            continue
        t = _block_t(vals, horizon)
        tstats[c] = t
        # below the bar, the group is held flat rather than guessed at
        ics[c] = float(vals.mean()) if abs(t) >= min_t else 0.0

    model = _weights_from_ics(ics, gcols, shrink, str(train_end), n_train)
    model.group_t = tstats
    return model


def _weights_from_ics(
    ics: dict[str, float],
    gcols: list[str],
    shrink: float,
    trained_through: str | None,
    n_train: int,
) -> ScoreModel:
    """Signed, shrunk weights from trailing information coefficients.

    Weights are allowed to go **negative**, and that is the whole point.

    The registry encodes each feature's long-run academic direction -- cheap
    beats expensive, profitable beats unprofitable, low volatility beats high.
    Measured over 2016-2026 US large caps, several of those relationships ran
    the other way: high-volatility, high-beta, low-margin names outperformed for
    most of the sample. An equal-weighted composite over fixed academic signs is
    therefore guaranteed to lean the wrong way in a regime like this one.

    The honest response is neither to hard-code the textbook signs nor to flip
    them to match the sample -- flipping them is hindsight, and it bakes in a bet
    that the last decade repeats. Instead each fold estimates the sign *and*
    magnitude from its own trailing window, so the tilt is always a statement
    about what has been working up to that date, and it turns when the data
    turns. The cost is a lag at regime changes, which is real and is surfaced in
    the UI rather than hidden.

    Shrinkage pulls toward zero (no view), not toward equal weight, because with
    signs unknown "equal weight" is itself a directional assumption.
    """
    raw = {c: ics.get(c, 0.0) for c in gcols}
    scale = sum(abs(v) for v in raw.values())

    if scale <= 1e-12:
        # No trailing signal at all: hold no view rather than invent one.
        return ScoreModel(
            weights={c: 0.0 for c in gcols},
            group_ic={k: float(v) for k, v in raw.items()},
            trained_through=trained_through,
            n_train=n_train,
        )

    weights = {c: shrink * (raw[c] / scale) for c in gcols}
    gross = sum(abs(v) for v in weights.values())
    if gross > 0:
        weights = {k: v / gross for k, v in weights.items()}

    return ScoreModel(
        weights=weights,
        group_ic={k: float(v) for k, v in raw.items()},
        trained_through=trained_through,
        n_train=n_train,
    )


def fit_weights(train: pl.DataFrame, *, shrink: float = SHRINK) -> ScoreModel:
    """Estimate group weights from training-period IC only."""
    gcols = [group_column(g) for g in feature_groups() if group_column(g) in train.columns]
    scoped = train.filter(pl.col("target").is_not_null())

    ics: dict[str, float] = {}
    for c in gcols:
        sub = scoped.filter(pl.col(c).is_not_null())
        if sub.height < 2000:
            ics[c] = 0.0
            continue
        ic = ic_by_date(sub, c, "target")
        vals = ic["ic"].to_numpy() if not ic.is_empty() else np.array([])
        vals = vals[np.isfinite(vals)]
        ics[c] = float(vals.mean()) if vals.size else 0.0

    # Only groups with a positive training IC earn weight; the rest are
    # carried at their equal-weight floor rather than shorted, because a
    # negative in-sample IC on eight groups is usually noise.
    return _weights_from_ics(
        ics, gcols, shrink,
        str(scoped["date"].max()) if scoped.height else None,
        scoped.height,
    )


def apply_weights(df: pl.DataFrame, model: ScoreModel) -> pl.DataFrame:
    """Composite score, per-group contributions, and cross-group agreement.

    Groups missing for a name are dropped and the remaining weights are
    renormalised, so a name is never quietly credited with average evidence on
    a group it has no data for.
    """
    gcols = [c for c in model.weights if c in df.columns]
    if not gcols:
        raise ValueError("no group score columns present")

    # absolute, not net: with signed weights a long/short pair can net to zero
    # and dividing by that would blow the score up
    avail_w = pl.sum_horizontal(
        [pl.when(pl.col(c).is_not_null()).then(abs(model.weights[c])).otherwise(0.0) for c in gcols]
    )
    contrib = {
        c: (
            pl.when(pl.col(c).is_not_null())
            .then(pl.col(c) * model.weights[c] / pl.when(avail_w > 0).then(avail_w).otherwise(1.0))
            .otherwise(0.0)
        )
        for c in gcols
    }

    out = df.with_columns([expr.alias(f"contrib_{c}") for c, expr in contrib.items()])
    out = out.with_columns(
        [
            pl.sum_horizontal([pl.col(f"contrib_{c}") for c in gcols]).alias("score_raw"),
            avail_w.alias("weight_available"),
        ]
    )

    # Agreement: share of contributing groups pointing the same way as the
    # composite. Conviction should fall when the evidence is internally split.
    # Measured on CONTRIBUTIONS, not raw group scores: with signed weights a
    # group can score positively and still push a name down, and it is the push
    # that the user sees and that conviction should reflect.
    n_groups = pl.sum_horizontal(
        [(pl.col(c).is_not_null() & (pl.lit(abs(model.weights[c])) > 1e-9)).cast(pl.Float64)
         for c in gcols]
    )
    n_pos = pl.sum_horizontal(
        [pl.when(pl.col(c).is_not_null() & (pl.col(f"contrib_{c}") > 0)).then(1.0).otherwise(0.0)
         for c in gcols]
    )
    out = out.with_columns(
        pl.when(n_groups > 0)
        .then(
            pl.when(pl.col("score_raw") >= 0)
            .then(n_pos / n_groups)
            .otherwise(1.0 - n_pos / n_groups)
        )
        .otherwise(None)
        .alias("agreement")
    )

    # Standardise per date so scores are comparable across time.
    out = out.with_columns(
        (
            (pl.col("score_raw") - pl.col("score_raw").mean().over("date"))
            / (pl.col("score_raw").std().over("date") + 1e-9)
        ).alias("score")
    ).with_columns(
        (
            (pl.col("score").rank("average").over("date") - 0.5) / pl.len().over("date")
        ).alias("percentile")
    )
    return out


def rate(percentile: float | None, agreement: float | None) -> str:
    """Map percentile and agreement to a label.

    Conviction requires both a high rank *and* agreement across data points --
    a name the groups disagree about is held back to Accumulate no matter how
    high it ranks, which is what "coherent" has to mean in practice.
    """
    if percentile is None:
        return "hold"
    agree = agreement if agreement is not None else 0.5
    if percentile >= 0.90:
        return "buy" if agree >= 0.6 else "accumulate"
    if percentile >= 0.75:
        return "accumulate"
    if percentile <= 0.10:
        return "avoid" if agree >= 0.6 else "reduce"
    if percentile <= 0.25:
        return "reduce"
    return "hold"


def contribution_records(row: dict, model: ScoreModel) -> list[dict]:
    """Per-group contributions for one name, largest magnitude first."""
    out = []
    for g in feature_groups():
        c = group_column(g)
        key = f"contrib_{c}"
        if key not in row or row[key] is None:
            continue
        if row.get(c) is None:
            continue
        out.append(
            {
                "group": g,
                "label": g,
                "z": float(row[c]),
                "contribution": float(row[key]),
                "weight": float(model.weights.get(c, 0.0)),
                "coverage": float(row.get(f"cov_{c}") or 0.0),
            }
        )
    return sorted(out, key=lambda d: -abs(d["contribution"]))
