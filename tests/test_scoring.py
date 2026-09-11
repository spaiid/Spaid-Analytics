"""Tests for peer normalisation and the composite score.

The properties that matter are not "does it return a number" but: does missing
data stay missing, do the declared weights actually govern the result, and does
a business model that cannot support a metric get excluded rather than ranked
badly on it.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from spaid.config.scoring import (
    ALL_METRICS,
    BusinessModel,
    Category,
    MetricSpec,
    PeerBasis,
    ScoringSpec,
    score_band,
)
from spaid.pipeline.scoring import (
    add_peer_keys,
    build_category_scores,
    build_metric_scores,
    build_opportunity_scores,
    score_metric,
)

D = date(2026, 9, 10)


def make_spec(metrics: tuple[MetricSpec, ...]) -> ScoringSpec:
    """A spec containing exactly `metrics`, padded so every category validates."""
    by_cat: dict[Category, list[MetricSpec]] = {c: [] for c in Category}
    for m in metrics:
        by_cat[m.category].append(m)

    filled: list[MetricSpec] = []
    for cat, items in by_cat.items():
        if not items:
            items = [
                MetricSpec(
                    f"_filler_{cat.value}", f"filler {cat.value}", cat, 1, 1.0,
                    PeerBasis.UNIVERSE, "test filler", min_peers=2,
                )
            ]
        total = sum(m.weight for m in items)
        filled.extend(
            MetricSpec(
                m.key, m.label, m.category, m.direction, m.weight / total,
                m.peer_basis, m.description, m.applies_to, m.winsor, m.min_peers, m.unit,
            )
            for m in items
        )
    return ScoringSpec(
        version="test-1",
        category_weights={
            Category.QUALITY: 0.30,
            Category.GROWTH: 0.25,
            Category.MOMENTUM: 0.25,
            Category.VALUATION: 0.20,
        },
        metrics=tuple(filled),
    )


def panel(rows: list[dict], *, fillers: bool = True) -> pl.DataFrame:
    """A test panel. Filler columns give every category something to score, so
    the composite is not suppressed by the total-coverage floor."""
    base = {
        "date": D,
        "sector": "Information Technology",
        "industry": "Software",
        "business_model": "operating",
        "market_cap": 1e11,
        "_filler_quality": 1.0,
        "_filler_growth": 1.0,
        "_filler_momentum": 1.0,
        "_filler_valuation": 1.0,
    }
    keys = ("_filler_quality", "_filler_growth", "_filler_momentum", "_filler_valuation")
    if not fillers:
        base = {k: v for k, v in base.items() if k not in keys}
    out = [{**base, **r} for r in rows]
    if fillers:
        for i, row in enumerate(out):
            for key in keys:
                row[key] = float(i)
    return pl.DataFrame(out)


class TestScoreBand:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (95.0, "Very attractive"),
            (68.0, "Very attractive"),
            (64.0, "Attractive"),
            (50.0, "Neutral"),
            (35.0, "Unattractive"),
            (10.0, "Very unattractive"),
            (None, "Not scored"),
        ],
    )
    def test_bands(self, score, expected):
        assert score_band(score) == expected

    def test_bands_are_monotone_and_cover_the_whole_range(self):
        """Every score from 0 to 100 must land in exactly one band."""
        from spaid.config.scoring import SCORE_BANDS

        thresholds = [t for t, _ in SCORE_BANDS]
        assert thresholds == sorted(thresholds, reverse=True)
        assert thresholds[-1] == 0.0
        labels = {score_band(float(s)) for s in range(0, 101)}
        assert labels == {label for _, label in SCORE_BANDS}


class TestPercentileRanking:
    def _metric(self, **kw) -> MetricSpec:
        defaults = dict(
            key="roe", label="Return on equity", category=Category.QUALITY,
            direction=1, weight=1.0, peer_basis=PeerBasis.UNIVERSE,
            description="", min_peers=2,
        )
        defaults.update(kw)
        return MetricSpec(**defaults)

    def test_highest_value_scores_highest_when_direction_is_positive(self):
        p = add_peer_keys(panel([
            {"company_id": f"C{i}", "ticker": f"T{i}", "roe": v}
            for i, v in enumerate([0.05, 0.15, 0.25, 0.35])
        ]))
        out = score_metric(p, self._metric()).sort("raw_value")
        assert out["score"].to_list() == pytest.approx([12.5, 37.5, 62.5, 87.5])

    def test_direction_inverts_the_ranking(self):
        """A -1 direction metric must score the *lowest* raw value highest."""
        p = add_peer_keys(panel([
            {"company_id": f"C{i}", "ticker": f"T{i}", "ev_ebit": v}
            for i, v in enumerate([8.0, 12.0, 20.0, 40.0])
        ]))
        spec = self._metric(key="ev_ebit", direction=-1, category=Category.VALUATION)
        out = score_metric(p, spec).sort("raw_value")
        assert out["score"].to_list() == pytest.approx([87.5, 62.5, 37.5, 12.5])

    def test_missing_value_is_not_scored_as_average(self):
        """The central failure mode: a null must not become a 50."""
        p = add_peer_keys(panel([
            {"company_id": "A", "ticker": "A", "roe": 0.10},
            {"company_id": "B", "ticker": "B", "roe": 0.20},
            {"company_id": "C", "ticker": "C", "roe": 0.30},
            {"company_id": "D", "ticker": "D", "roe": None},
        ]))
        out = score_metric(p, self._metric())
        missing = out.filter(pl.col("company_id") == "D")
        assert missing["score"][0] is None
        assert missing["status"][0] == "missing"
        assert missing["is_missing"][0] is True

    def test_missing_values_do_not_dilute_the_peer_count(self):
        p = add_peer_keys(panel([
            {"company_id": "A", "ticker": "A", "roe": 0.10},
            {"company_id": "B", "ticker": "B", "roe": 0.20},
            *[{"company_id": f"N{i}", "ticker": f"N{i}", "roe": None} for i in range(20)],
        ]))
        out = score_metric(p, self._metric())
        assert out["peer_count"].max() == 2

    def test_not_applicable_when_the_business_model_cannot_support_it(self):
        """A bank has no gross margin; that is not a low gross margin."""
        p = add_peer_keys(
            pl.DataFrame([
                {"company_id": "BANK", "ticker": "JPM", "date": D, "sector": "Financials",
                 "industry": "Diversified Banks", "business_model": "bank",
                 "market_cap": 1e11, "gross_margin": None},
                {"company_id": "SOFT", "ticker": "MSFT", "date": D,
                 "sector": "Information Technology", "industry": "Software",
                 "business_model": "operating", "market_cap": 3e12, "gross_margin": 0.70},
                {"company_id": "HW", "ticker": "AAPL", "date": D,
                 "sector": "Information Technology", "industry": "Hardware",
                 "business_model": "operating", "market_cap": 4e12, "gross_margin": 0.47},
            ])
        )
        spec = self._metric(
            key="gross_margin",
            applies_to=frozenset({BusinessModel.OPERATING, BusinessModel.CYCLICAL}),
        )
        out = score_metric(p, spec)
        bank = out.filter(pl.col("company_id") == "BANK")
        assert bank["status"][0] == "not_applicable"
        assert bank["score"][0] is None

    def test_winsorisation_caps_an_extreme_value(self):
        values = [0.1] * 20 + [500.0]
        p = add_peer_keys(panel([
            {"company_id": f"C{i}", "ticker": f"T{i}", "roe": v}
            for i, v in enumerate(values)
        ]))
        out = score_metric(p, self._metric(winsor=0.05))
        assert out["winsorized_value"].max() < 500.0

    def test_raw_value_is_preserved_alongside_the_score(self):
        """The brief requires storing both the raw value and the normalised score."""
        p = add_peer_keys(panel([
            {"company_id": "A", "ticker": "A", "roe": 0.123},
            {"company_id": "B", "ticker": "B", "roe": 0.456},
        ]))
        out = score_metric(p, self._metric())
        assert sorted(out["raw_value"].to_list()) == pytest.approx([0.123, 0.456])

    def test_industry_falls_back_to_sector_when_too_thin(self):
        rows = [
            {"company_id": "A", "ticker": "A", "roe": 0.1, "industry": "Tiny Niche"},
            {"company_id": "B", "ticker": "B", "roe": 0.2, "industry": "Software"},
            {"company_id": "C", "ticker": "C", "roe": 0.3, "industry": "Software"},
            {"company_id": "D", "ticker": "D", "roe": 0.4, "industry": "Software"},
        ]
        p = add_peer_keys(panel(rows))
        out = score_metric(p, self._metric(peer_basis=PeerBasis.INDUSTRY, min_peers=3))
        lonely = out.filter(pl.col("company_id") == "A")
        assert lonely["peer_basis"][0] == "sector"
        assert lonely["score"][0] is not None


class TestCategoryAndComposite:
    def _two_metric_spec(self) -> ScoringSpec:
        return make_spec((
            MetricSpec("roe", "ROE", Category.QUALITY, 1, 0.5, PeerBasis.UNIVERSE, "", min_peers=2),
            MetricSpec("roic", "ROIC", Category.QUALITY, 1, 0.5, PeerBasis.UNIVERSE, "", min_peers=2),
        ))

    def _panel(self, n: int = 4) -> pl.DataFrame:
        return panel([
            {
                "company_id": f"C{i}", "ticker": f"T{i}",
                "roe": 0.1 * (i + 1), "roic": 0.1 * (i + 1),
            }
            for i in range(n)
        ])

    def test_category_score_is_the_weighted_average(self):
        spec = self._two_metric_spec()
        ms = build_metric_scores(self._panel(), spec)
        cs = build_category_scores(ms, spec)
        quality = cs.filter(
            (pl.col("category") == "quality") & (pl.col("company_id") == "C3")
        )
        # Best on both metrics in a four-name universe: (1 - 0.5/4) * 100.
        assert quality["score"][0] == pytest.approx(87.5)
        assert quality["coverage"][0] == pytest.approx(1.0)

    def test_coverage_falls_when_a_metric_is_missing(self):
        spec = self._two_metric_spec()
        p = self._panel()
        p = p.with_columns(
            pl.when(pl.col("company_id") == "C0").then(None).otherwise(pl.col("roic")).alias("roic")
        )
        ms = build_metric_scores(p, spec)
        cs = build_category_scores(ms, spec)
        thin = cs.filter((pl.col("category") == "quality") & (pl.col("company_id") == "C0"))
        assert thin["coverage"][0] == pytest.approx(0.5)
        assert thin["n_missing"][0] == 1

    def test_category_declines_to_score_below_the_coverage_floor(self):
        spec = self._two_metric_spec()
        p = self._panel().with_columns(
            pl.when(pl.col("company_id") == "C0").then(None).otherwise(pl.col("roe")).alias("roe"),
            pl.when(pl.col("company_id") == "C0").then(None).otherwise(pl.col("roic")).alias("roic"),
        )
        ms = build_metric_scores(p, spec)
        cs = build_category_scores(ms, spec)
        thin = cs.filter((pl.col("category") == "quality") & (pl.col("company_id") == "C0"))
        assert thin["score"][0] is None

    def test_composite_respects_the_declared_category_weights(self):
        """The whole point of a fixed spec: the weights govern the arithmetic."""
        spec = make_spec((
            MetricSpec("roe", "ROE", Category.QUALITY, 1, 1.0, PeerBasis.UNIVERSE, "", min_peers=2),
            MetricSpec("mom_12_1", "Mom", Category.MOMENTUM, 1, 1.0, PeerBasis.UNIVERSE, "", min_peers=2),
        ))
        # No filler columns: only quality and momentum score, so the composite
        # is exactly 0.30 * quality + 0.25 * momentum renormalised.
        p = panel(
            [
                {"company_id": "BEST_Q", "ticker": "A", "roe": 0.9, "mom_12_1": -0.5},
                {"company_id": "BEST_M", "ticker": "B", "roe": -0.5, "mom_12_1": 0.9},
            ],
            fillers=False,
        )
        ms = build_metric_scores(p, spec)
        cs = build_category_scores(ms, spec)
        opp = build_opportunity_scores(cs, spec)
        best_q = opp.filter(pl.col("company_id") == "BEST_Q")["score"][0]
        best_m = opp.filter(pl.col("company_id") == "BEST_M")["score"][0]
        # Quality carries 0.30 and momentum 0.25, so with only those two
        # categories present the quality leader must finish ahead.
        assert best_q > best_m
        # And by exactly the amount the declared weights imply: the winner of
        # each metric scores 75, the loser 25, over a two-name universe.
        expected_q = (0.30 * 75.0 + 0.25 * 25.0) / 0.55
        expected_m = (0.30 * 25.0 + 0.25 * 75.0) / 0.55
        assert best_q == pytest.approx(expected_q)
        assert best_m == pytest.approx(expected_m)

    def test_composite_is_on_a_zero_to_one_hundred_scale(self):
        spec = self._two_metric_spec()
        ms = build_metric_scores(self._panel(10), spec)
        cs = build_category_scores(ms, spec)
        opp = build_opportunity_scores(cs, spec)
        scores = [s for s in opp["score"].to_list() if s is not None]
        assert scores
        assert min(scores) >= 0.0
        assert max(scores) <= 100.0

    def test_missing_category_renormalises_rather_than_dragging_the_score_down(self):
        """A company with no growth data must not score as if growth were zero."""
        spec = make_spec((
            MetricSpec("roe", "ROE", Category.QUALITY, 1, 1.0, PeerBasis.UNIVERSE, "", min_peers=2),
            MetricSpec(
                "revenue_growth_1y", "Growth", Category.GROWTH, 1, 1.0,
                PeerBasis.UNIVERSE, "", min_peers=2,
            ),
        ))
        p = panel([
            {"company_id": "FULL", "ticker": "A", "roe": 0.9, "revenue_growth_1y": 0.9},
            {"company_id": "PARTIAL", "ticker": "B", "roe": 0.9, "revenue_growth_1y": None},
            {"company_id": "LOW", "ticker": "C", "roe": 0.1, "revenue_growth_1y": 0.1},
        ])
        ms = build_metric_scores(p, spec)
        cs = build_category_scores(ms, spec)
        opp = build_opportunity_scores(cs, spec)
        partial = opp.filter(pl.col("company_id") == "PARTIAL")["score"][0]
        low = opp.filter(pl.col("company_id") == "LOW")["score"][0]
        assert partial > low
        assert partial > 50.0

    def test_rank_is_dense_and_ordered(self):
        spec = self._two_metric_spec()
        ms = build_metric_scores(self._panel(6), spec)
        cs = build_category_scores(ms, spec)
        opp = build_opportunity_scores(cs, spec).sort("rank")
        scores = opp["score"].to_list()
        assert scores == sorted(scores, reverse=True)


class TestRealSpecIntegrity:
    def test_every_metric_belongs_to_exactly_one_category(self):
        keys = [m.key for m in ALL_METRICS]
        assert len(keys) == len(set(keys))

    def test_category_weights_match_the_brief(self):
        from spaid.config.scoring import ACTIVE_SCORING_SPEC as spec

        assert spec.category_weights[Category.QUALITY] == pytest.approx(0.30)
        assert spec.category_weights[Category.GROWTH] == pytest.approx(0.25)
        assert spec.category_weights[Category.MOMENTUM] == pytest.approx(0.25)
        assert spec.category_weights[Category.VALUATION] == pytest.approx(0.20)

    def test_banks_are_excluded_from_operating_only_metrics(self):
        from spaid.config.scoring import ACTIVE_SCORING_SPEC as spec

        bank_metrics = {m.key for m in spec.applicable(BusinessModel.BANK)}
        assert "gross_margin" not in bank_metrics
        assert "ev_ebitda" not in bank_metrics
        assert "roe" in bank_metrics
        assert "price_to_book" in bank_metrics

    def test_a_bank_still_has_enough_weight_to_score_every_category(self):
        """Exclusions must not leave a category below its own coverage floor."""
        from spaid.config.scoring import ACTIVE_SCORING_SPEC as spec

        for model in BusinessModel:
            for cat in Category:
                applicable = [
                    m for m in spec.in_category(cat) if model in m.applies_to
                ]
                weight = sum(m.weight for m in applicable)
                assert weight >= spec.min_category_coverage, (
                    f"{model.value}/{cat.value} retains only {weight:.2f} of its metric weight, "
                    f"below the {spec.min_category_coverage} floor: it could never score"
                )
