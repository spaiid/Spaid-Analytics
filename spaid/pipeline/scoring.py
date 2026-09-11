"""The composite opportunity score: peer normalisation, categories, 0-100.

The score is a weighted average of percentile ranks, and it is deliberately
simple arithmetic rather than a fitted model. That has one enormous practical
benefit: the explanation the dashboard shows is not an approximation of the
score, it *is* the score. "Why is this company ranked here" has an exact
arithmetic answer that adds up.

Three rules from the product brief are enforced here rather than assumed:

**Weights are policy, not output.** Quality 30, Growth 25, Momentum 25,
Valuation 20 come from the versioned spec and are never fitted to historical
returns. The information coefficient of each category is measured and reported
as a diagnostic, but it never feeds back into the weights.

**Peers, not the whole market.** Comparing a utility's operating margin with a
software company's says nothing. Profitability and valuation metrics rank inside
the industry, falling back to the sector when an industry is too thin to rank
against. Price behaviour ranks against the whole universe, because that is a
market-wide phenomenon, with separate explicit metrics carrying the sector
comparison.

**Missing is not average.** A metric with no data is dropped from its category
and the remaining weights renormalise, so a company is never credited with
average evidence for something we do not know. Below a coverage floor the
category declines to score at all, and below a total-coverage floor so does the
composite. That is why a thinly-covered company appears with a low confidence
rather than a confidently mediocre score.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import polars as pl

from spaid.config.scoring import (
    Category,
    MetricSpec,
    PeerBasis,
    ScoringSpec,
    score_band,
)
from spaid.storage.schema import (
    CATEGORY_SCORES,
    METRIC_SCORES,
    OPPORTUNITY_SCORES,
    coerce,
)

log = logging.getLogger(__name__)

# Peer-group fallback order. An industry too thin to rank against falls back to
# its sector, and a sector too thin falls back to the whole universe, so a
# metric is never ranked against three companies.
FALLBACK: dict[PeerBasis, PeerBasis | None] = {
    PeerBasis.INDUSTRY: PeerBasis.SECTOR,
    PeerBasis.SECTOR: PeerBasis.UNIVERSE,
    PeerBasis.SIZE_DECILE: PeerBasis.UNIVERSE,
    PeerBasis.UNIVERSE: None,
}


def add_peer_keys(panel: pl.DataFrame) -> pl.DataFrame:
    """Attach the grouping key for each peer basis.

    `panel` must already carry `sector` and `industry` from the companies table
    and `market_cap` from the metrics layer.
    """
    out = panel.with_columns(
        pl.lit("all").alias("peer_universe"),
        pl.col("sector").fill_null("Unclassified").alias("peer_sector"),
        (
            pl.col("sector").fill_null("Unclassified")
            + " / "
            + pl.col("industry").fill_null("Unclassified")
        ).alias("peer_industry"),
    )
    if "market_cap" in out.columns:
        out = out.with_columns(
            (
                (
                    pl.col("market_cap").rank("ordinal").over("date")
                    / pl.len().over("date")
                    * 10
                )
                .ceil()
                .clip(1, 10)
                .cast(pl.Int32)
                .cast(pl.Utf8)
            ).alias("peer_size_decile")
        )
    else:
        out = out.with_columns(pl.lit("1").alias("peer_size_decile"))
    return out


PEER_COLUMN: dict[PeerBasis, str] = {
    PeerBasis.UNIVERSE: "peer_universe",
    PeerBasis.SECTOR: "peer_sector",
    PeerBasis.INDUSTRY: "peer_industry",
    PeerBasis.SIZE_DECILE: "peer_size_decile",
}


def _applicable_mask(panel: pl.DataFrame, spec: MetricSpec) -> pl.Expr:
    """True where this metric is meaningful for the company's business model."""
    allowed = [m.value for m in spec.applies_to]
    return pl.col("business_model").is_in(allowed)


def score_metric(panel: pl.DataFrame, spec: MetricSpec) -> pl.DataFrame:
    """Percentile-rank one metric within its peer group, per date.

    Returns one row per company-date with the raw value preserved alongside the
    score, because the product brief requires both and because a percentile
    without the number behind it is not explainable.
    """
    if spec.key not in panel.columns:
        return pl.DataFrame()

    basis = spec.peer_basis

    # The chain of groupings to try, finest first. A company whose industry is
    # too thin is ranked against its whole sector -- against every company in
    # it, not only against the other companies that also fell back, which would
    # compare the four loneliest businesses in the index with each other.
    chain: list[PeerBasis] = [basis]
    cursor: PeerBasis | None = FALLBACK.get(basis)
    while cursor is not None:
        chain.append(cursor)
        cursor = FALLBACK.get(cursor)

    work = panel.select(
        [
            "company_id", "ticker", "date", "business_model", spec.key,
            *{PEER_COLUMN[b] for b in chain},
        ]
    ).rename({spec.key: "raw_value"})

    work = work.with_columns(
        _applicable_mask(panel, spec).alias("is_applicable"),
    ).with_columns(
        # A value that exists but is not meaningful for this business is
        # excluded from the ranking, not ranked badly.
        pl.when(pl.col("is_applicable"))
        .then(pl.col("raw_value"))
        .otherwise(None)
        .alias("usable_value")
    )

    lo, hi = spec.winsor, 1.0 - spec.winsor
    for level in chain:
        col = PEER_COLUMN[level]
        keys = ["date", col]
        # Peer-group size counts only companies with a usable observation:
        # ranking against a group of forty where thirty-five are null is
        # ranking against five.
        n = pl.col("usable_value").is_not_null().sum().over(keys)
        winsorized = pl.col("usable_value").clip(
            pl.col("usable_value").quantile(lo).over(keys),
            pl.col("usable_value").quantile(hi).over(keys),
        )
        rank = winsorized.rank("average").over(keys)
        pct = (rank - 0.5) / n
        if spec.direction < 0:
            pct = 1.0 - pct
        work = work.with_columns(
            n.cast(pl.Int32).alias(f"_n__{level.value}"),
            winsorized.alias(f"_win__{level.value}"),
            pl.when(n >= 2).then(pct).otherwise(None).alias(f"_pct__{level.value}"),
            pl.col("usable_value").median().over(keys).alias(f"_med__{level.value}"),
        )

    # Pick the finest level that clears the minimum peer count, falling back
    # through the chain and finally taking the broadest available.
    def pick(prefix: str, dtype: pl.DataType) -> pl.Expr:
        expr = pl.lit(None, dtype=dtype)
        for level in reversed(chain):
            expr = (
                pl.when(pl.col(f"_n__{level.value}") >= spec.min_peers)
                .then(pl.col(f"{prefix}__{level.value}").cast(dtype))
                .otherwise(expr)
            )
        return expr

    def pick_basis() -> pl.Expr:
        expr = pl.lit(chain[-1].value, dtype=pl.Utf8)
        for level in reversed(chain):
            expr = (
                pl.when(pl.col(f"_n__{level.value}") >= spec.min_peers)
                .then(pl.lit(level.value))
                .otherwise(expr)
            )
        return expr

    def pick_group() -> pl.Expr:
        expr = pl.col(PEER_COLUMN[chain[-1]])
        for level in reversed(chain):
            expr = (
                pl.when(pl.col(f"_n__{level.value}") >= spec.min_peers)
                .then(pl.col(PEER_COLUMN[level]))
                .otherwise(expr)
            )
        return expr

    work = work.with_columns(
        pick("_pct", pl.Float64).alias("percentile"),
        pick("_win", pl.Float64).alias("winsorized_value"),
        pick("_med", pl.Float64).alias("peer_median"),
        pick("_n", pl.Int32).alias("peer_count"),
        pick_basis().alias("peer_basis"),
        pick_group().alias("peer_group"),
    ).with_columns((pl.col("percentile") * 100.0).alias("score"))

    work = work.with_columns(
        pl.when(~pl.col("is_applicable"))
        .then(pl.lit("not_applicable"))
        .when(pl.col("raw_value").is_null())
        .then(pl.lit("missing"))
        .when(pl.col("percentile").is_null())
        .then(pl.lit("thin_peers"))
        .otherwise(pl.lit("scored"))
        .alias("status")
    ).with_columns((pl.col("status") != "scored").alias("is_missing"))

    return work.select(
        "company_id",
        "ticker",
        "date",
        pl.lit(spec.key).alias("metric"),
        pl.lit(spec.category.value).alias("category"),
        "raw_value",
        "winsorized_value",
        "percentile",
        "score",
        "peer_basis",
        "peer_group",
        "peer_count",
        "peer_median",
        "is_missing",
        "status",
        pl.lit(spec.weight).alias("weight"),
    )


def build_metric_scores(panel: pl.DataFrame, spec: ScoringSpec) -> pl.DataFrame:
    """Percentile-rank every metric in the spec."""
    panel = add_peer_keys(panel)
    pieces = [score_metric(panel, m) for m in spec.metrics]
    pieces = [p for p in pieces if not p.is_empty()]
    if not pieces:
        raise RuntimeError("no metrics could be scored; is the metric panel empty?")

    out = pl.concat(pieces, how="vertical_relaxed")
    out = out.with_columns(
        pl.lit(spec.version).alias("spec_version"),
        pl.lit(datetime.now(UTC)).alias("computed_at"),
    )
    scored = out.filter(pl.col("status") == "scored").height
    log.info(
        "metric scores: %d rows (%d scored, %.0f%% coverage) across %d metrics",
        out.height, scored, 100.0 * scored / max(out.height, 1), out["metric"].n_unique(),
    )
    return coerce(out, METRIC_SCORES)


def build_category_scores(metric_scores: pl.DataFrame, spec: ScoringSpec) -> pl.DataFrame:
    """Weighted average of each category's observed metrics.

    Coverage is the share of the category's *applicable* weight that was
    actually observed. Metrics a business model cannot support are excluded from
    both the numerator and the denominator, so a bank is not penalised for
    having no gross margin.
    """
    applicable = metric_scores.filter(pl.col("status") != "not_applicable")

    agg = (
        applicable.group_by(["company_id", "ticker", "date", "category"])
        .agg(
            (
                (pl.col("score") * pl.col("weight")).filter(pl.col("status") == "scored").sum()
            ).alias("_weighted_sum"),
            (pl.col("weight").filter(pl.col("status") == "scored").sum()).alias(
                "_observed_weight"
            ),
            pl.col("weight").sum().alias("_applicable_weight"),
            pl.len().alias("n_metrics"),
            (pl.col("status") != "scored").sum().cast(pl.Int32).alias("n_missing"),
        )
        .with_columns(
            (pl.col("_observed_weight") / pl.col("_applicable_weight")).alias("coverage")
        )
    )

    weights = {c.value: w for c, w in spec.category_weights.items()}
    out = agg.with_columns(
        pl.when(
            (pl.col("coverage") >= spec.min_category_coverage)
            & (pl.col("_observed_weight") > 0)
        )
        .then(pl.col("_weighted_sum") / pl.col("_observed_weight"))
        .otherwise(None)
        .alias("score"),
        pl.col("category").replace_strict(weights, default=0.0).alias("weight"),
    )
    out = out.with_columns(
        (pl.col("score") * pl.col("weight")).alias("contribution"),
        pl.lit(spec.version).alias("spec_version"),
        pl.lit(datetime.now(UTC)).alias("computed_at"),
        pl.col("n_metrics").cast(pl.Int32),
    ).drop("_weighted_sum", "_observed_weight", "_applicable_weight")

    log.info(
        "category scores: %d rows; %d categories declined for thin coverage",
        out.height, int(out["score"].is_null().sum()),
    )
    return coerce(out, CATEGORY_SCORES)


def build_opportunity_scores(
    category_scores: pl.DataFrame, spec: ScoringSpec
) -> pl.DataFrame:
    """Blend the categories into one 0-100 composite.

    Categories that declined to score are dropped and the remaining weights
    renormalise, which keeps the composite on the same 0-100 scale rather than
    silently shrinking it toward zero for every company with a data gap.
    """
    present = category_scores.filter(pl.col("score").is_not_null())

    totals = (
        present.group_by(["company_id", "ticker", "date"])
        .agg(
            pl.col("contribution").sum().alias("_contribution"),
            pl.col("weight").sum().alias("_weight"),
        )
        .with_columns(
            pl.when(pl.col("_weight") > 0)
            .then(pl.col("_contribution") / pl.col("_weight"))
            .otherwise(None)
            .alias("score"),
            pl.col("_weight").alias("coverage"),
        )
    )

    wide = (
        category_scores.select(["company_id", "date", "category", "score"])
        .pivot(on="category", index=["company_id", "date"], values="score")
    )
    for cat in Category:
        if cat.value not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(cat.value))

    out = totals.join(wide, on=["company_id", "date"], how="left")

    # A composite built on less than half the available evidence is not a
    # composite. It is reported as unscored, and the reason is visible.
    out = out.with_columns(
        pl.when(pl.col("coverage") >= spec.min_total_coverage)
        .then(pl.col("score"))
        .otherwise(None)
        .alias("score")
    )

    out = out.with_columns(
        pl.col("score").rank("ordinal", descending=True).over("date").cast(pl.Int32).alias("rank"),
        (
            (pl.col("score").rank("average").over("date") - 0.5)
            / pl.col("score").is_not_null().sum().over("date")
        ).alias("percentile"),
        pl.len().over("date").cast(pl.Int32).alias("universe_size"),
    ).with_columns(
        pl.col("score")
        .map_elements(score_band, return_dtype=pl.Utf8)
        .alias("band")
    )

    out = out.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("confidence"),
        pl.lit(None, dtype=pl.Utf8).alias("confidence_label"),
        pl.lit(spec.version).alias("spec_version"),
        pl.lit(datetime.now(UTC)).alias("computed_at"),
    )
    out = add_score_changes(out)

    log.info(
        "opportunity scores: %d rows, %d scored, median %.1f",
        out.height,
        int(out["score"].is_not_null().sum()),
        out["score"].median() or float("nan"),
    )
    return coerce(out, OPPORTUNITY_SCORES)


def add_score_changes(scores: pl.DataFrame) -> pl.DataFrame:
    """How much each company's score has moved recently.

    "What changed" is one of the questions the dashboard must answer, and a
    score that jumped ten points last week is a different object from one that
    has sat still for a year even when today's number is identical.
    """
    from datetime import timedelta

    out = scores.sort(["company_id", "date"])
    for label, days in (("1w", 7), ("1m", 30)):
        lagged = (
            out.select(["company_id", "date", "score"])
            .with_columns((pl.col("date") + timedelta(days=days)).alias("date"))
            .rename({"score": f"_score_{label}"})
            .sort("date")
        )
        out = (
            out.sort("date")
            .join_asof(lagged, on="date", by="company_id", strategy="backward")
            .with_columns(
                (pl.col("score") - pl.col(f"_score_{label}")).alias(f"score_change_{label}")
            )
            .drop(f"_score_{label}")
        )
    return out.sort(["company_id", "date"])


def explain(
    metric_scores: pl.DataFrame,
    category_scores: pl.DataFrame,
    company_id: str,
    date,
) -> dict:
    """The full arithmetic behind one company's score on one date.

    Everything the "complete explanation of how the score was calculated" view
    needs, assembled from stored values only: no recomputation, so what the user
    reads is exactly what the engine used.
    """
    m = metric_scores.filter(
        (pl.col("company_id") == company_id) & (pl.col("date") == date)
    )
    c = category_scores.filter(
        (pl.col("company_id") == company_id) & (pl.col("date") == date)
    )
    if m.is_empty():
        return {}

    categories = []
    for row in c.sort("weight", descending=True).iter_rows(named=True):
        metrics = (
            m.filter(pl.col("category") == row["category"])
            .sort(["status", "weight"], descending=[False, True])
            .to_dicts()
        )
        categories.append({**row, "metrics": metrics})

    return {
        "company_id": company_id,
        "date": str(date),
        "categories": categories,
        "spec_version": m["spec_version"][0],
    }
