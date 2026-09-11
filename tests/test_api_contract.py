"""Contract tests for the HTTP interface.

These run against the real store, so they double as an end-to-end check that the
pipeline produced something coherent. When the store is empty they skip rather
than fail: a fresh clone has no data, and a red test suite that only means "you
have not run the pipeline yet" trains people to ignore red test suites.

What they actually assert is the contract the frontend depends on: the fields
exist, the units are what the client expects, and the qualitative readings the
interface is forbidden from computing are present in the payload.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from spaid.api.server import app
from spaid.storage import store


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def has_scores() -> bool:
    scores = store.read("opportunity_scores")
    return scores is not None and not scores.is_empty()


def requires_data(has_scores: bool) -> None:
    if not has_scores:
        pytest.skip("no scores in the store; run `spaid run` first")


class TestStatus:
    def test_responds(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert "as_of" in body
        assert "universe_size" in body
        assert "freshness" in body

    def test_reports_the_active_spec_versions(self, client, has_scores):
        requires_data(has_scores)
        body = client.get("/api/status").json()
        assert body["scoring_version"]
        assert body["valuation_version"]

    def test_freshness_carries_a_human_readable_reason(self, client):
        f = client.get("/api/status").json()["freshness"]
        assert "is_stale" in f
        assert f.get("detail")


class TestOpportunities:
    def test_returns_a_ranked_list(self, client, has_scores):
        requires_data(has_scores)
        body = client.get("/api/opportunities?limit=50").json()
        rows = body["rows"]
        assert rows
        ranks = [r["rank"] for r in rows if r["rank"] is not None]
        assert ranks == sorted(ranks)

    def test_counts_describe_the_universe_not_the_page(self, client, has_scores):
        """A limit must not change what "412 of 500 scored" means."""
        requires_data(has_scores)
        small = client.get("/api/opportunities?limit=5").json()
        large = client.get("/api/opportunities?limit=500").json()
        assert small["scored_count"] == large["scored_count"]
        assert small["universe_size"] == large["universe_size"]
        assert small["sectors"] == large["sectors"]

    def test_every_row_carries_its_qualitative_reading(self, client, has_scores):
        """The interface must never have to decide what a score means."""
        requires_data(has_scores)
        rows = client.get("/api/opportunities?limit=100").json()["rows"]
        for r in rows:
            if r["score"] is not None:
                assert r["band"], f"{r['ticker']} has a score but no band"
            if r["valuation_class"] is not None:
                assert r["valuation_label"], f"{r['ticker']} has a class but no label"
            if r["confidence"] is not None:
                assert r["confidence_label"]

    def test_scores_are_on_a_zero_to_one_hundred_scale(self, client, has_scores):
        requires_data(has_scores)
        rows = client.get("/api/opportunities?limit=500").json()["rows"]
        scores = [r["score"] for r in rows if r["score"] is not None]
        assert scores
        assert 0.0 <= min(scores) <= max(scores) <= 100.0

    def test_upside_is_a_fraction_not_a_percentage(self, client, has_scores):
        """A frontend that multiplies by 100 must receive 0.12, not 12."""
        requires_data(has_scores)
        rows = client.get("/api/opportunities?limit=500").json()["rows"]
        upsides = [abs(r["upside"]) for r in rows if r["upside"] is not None]
        assert upsides
        assert max(upsides) < 20.0, "upside looks like a percentage, not a fraction"

    def test_unscored_companies_are_explained_rather_than_hidden(self, client, has_scores):
        requires_data(has_scores)
        body = client.get("/api/opportunities?limit=500").json()
        unscored = [r for r in body["rows"] if r["score"] is None]
        if unscored:
            assert body["notes"], "unscored companies present but no note explaining them"


class TestStockDetail:
    @pytest.fixture(scope="class")
    def ticker(self, client, has_scores) -> str:
        requires_data(has_scores)
        rows = client.get("/api/opportunities?limit=1").json()["rows"]
        return rows[0]["ticker"]

    def test_unknown_ticker_returns_404_with_a_reason(self, client):
        r = client.get("/api/stock/NOTATICKER")
        assert r.status_code == 404
        assert "detail" in r.json()

    def test_returns_the_full_analysis(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        assert body["ticker"] == ticker
        assert body["categories"]
        assert body["fair_value"] is not None
        assert body["confidence"] is not None

    def test_all_four_categories_are_present(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        keys = {c["key"] for c in body["categories"]}
        assert keys == {"quality", "growth", "momentum", "valuation"}

    def test_category_weights_match_the_specification(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        weights = {c["key"]: c["weight"] for c in body["categories"]}
        assert weights["quality"] == pytest.approx(0.30)
        assert weights["growth"] == pytest.approx(0.25)
        assert weights["momentum"] == pytest.approx(0.25)
        assert weights["valuation"] == pytest.approx(0.20)

    def test_the_score_arithmetic_adds_up(self, client, ticker):
        """The explanation must *be* the calculation, not a story about it."""
        body = client.get(f"/api/stock/{ticker}").json()
        if body["score"] is None:
            pytest.skip("company is unscored")
        scored = [c for c in body["categories"] if c["score"] is not None]
        contribution = sum(c["contribution"] for c in scored)
        weight = sum(c["weight"] for c in scored)
        assert contribution / weight == pytest.approx(body["score"], rel=1e-6)

    def test_every_metric_declares_its_status(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        for cat in body["categories"]:
            for m in cat["metrics"]:
                assert m["status"] in {"scored", "missing", "not_applicable", "thin_peers"}
                if m["status"] != "scored":
                    assert m["score"] is None, "an unscored metric must not carry a score"
                    assert m["status_detail"], "an unscored metric must explain itself"

    def test_scored_metrics_keep_their_raw_value_and_peer_context(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        scored = [
            m for cat in body["categories"] for m in cat["metrics"] if m["status"] == "scored"
        ]
        assert scored
        for m in scored:
            assert m["raw_value"] is not None
            assert m["display_value"] is not None
            assert m["peer_count"] is not None and m["peer_count"] >= 2
            assert m["peer_basis"] in {"universe", "sector", "industry", "size_decile"}

    def test_fair_value_is_a_range_never_a_bare_number(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        fv = body["fair_value"]
        if fv["base"] is None:
            pytest.skip("no valuation available for this company")
        assert fv["range_low"] is not None and fv["range_high"] is not None
        assert fv["range_low"] <= fv["base"] <= fv["range_high"]
        # The headline is the weighted blend, not the centre of the range: the
        # centre of an asymmetric range sits above the estimate the methods
        # actually produced. Both are served, and they are different numbers.
        assert fv["midpoint"] == pytest.approx(fv["base"])
        assert fv["range_midpoint"] == pytest.approx(
            (fv["range_low"] + fv["range_high"]) / 2
        )
        assert fv["classification_label"]

    def test_skipped_valuation_methods_say_why(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        for m in body["fair_value"]["methods"]:
            if not m["used"]:
                assert m["reason"], f"{m['label']} was skipped without a reason"

    def test_valuation_carries_caveats(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        assert isinstance(body["fair_value"]["caveats"], list)

    def test_confidence_is_decomposed(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        c = body["confidence"]
        assert 0.0 <= c["score"] <= 1.0
        assert c["label"]
        assert c["components"]
        assert sum(x["weight"] for x in c["components"]) == pytest.approx(1.0)

    def test_value_trap_signals_are_enumerated(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        trap = body["value_trap"]
        if trap is None:
            pytest.skip("no trap assessment")
        assert trap["classification_label"]
        assert trap["signals"]
        assert all("fired" in s and "label" in s for s in trap["signals"])

    def test_sources_and_freshness_are_reported(self, client, ticker):
        body = client.get(f"/api/stock/{ticker}").json()
        assert body["sources"]
        assert body["freshness"]["detail"]


class TestHealth:
    def test_lists_every_table(self, client):
        body = client.get("/api/health").json()
        names = {t["name"] for t in body["tables"]}
        assert {"companies", "prices", "observations", "fundamentals"} <= names

    def test_warnings_carry_a_severity_and_a_message(self, client):
        body = client.get("/api/health").json()
        for w in body["warnings"]:
            assert w["severity"] in {"info", "warning", "critical"}
            assert w["message"]


class TestSpec:
    def test_publishes_the_active_model_specification(self, client):
        body = client.get("/api/spec").json()
        assert body["scoring"]["version"]
        weights = body["scoring"]["category_weights"]
        assert sum(weights.values()) == pytest.approx(1.0)
        assert body["scoring"]["metrics"]
        assert body["valuation"]["method_weights"]["bank"]

    def test_bank_weights_exclude_discounted_cash_flow(self, client):
        body = client.get("/api/spec").json()
        assert "dcf" not in body["valuation"]["method_weights"]["bank"]


class TestPipelineStatus:
    def test_reports_idle_state(self, client):
        body = client.get("/api/pipeline/status").json()
        assert "running" in body
        assert body["running"] in (True, False)
