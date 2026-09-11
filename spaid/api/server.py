"""The HTTP service.

Thin by design: routing, validation and background-task bookkeeping only. Every
number comes from `spaid.api.service`, which reads stored tables, which were
written by the pipeline. No investment calculation happens in this file or
anywhere downstream of it.

The built frontend is served from the same origin, so there is no cross-origin
configuration to get wrong and no separate deployment to keep in step.
"""

from __future__ import annotations

import logging
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from spaid.api import schemas as S
from spaid.api import service

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"

# Bookkeeping for the one long-running job the interface can trigger. A lock
# rather than a queue: running two pipelines at once would have them writing the
# same parquet files.
_pipeline_lock = threading.Lock()
_pipeline_state: dict = {
    "running": False,
    "stage": None,
    "progress": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def _run_pipeline(*, full: bool) -> None:
    """Execute the pipeline, recording progress for the status endpoint."""
    from spaid.pipeline import ingest
    from spaid.pipeline.analyze import run_analysis

    stages = (
        [
            ("Refreshing the constituents list", lambda: ingest.refresh_universe(force=full)),
            ("Downloading prices", lambda: ingest.refresh_prices(force=full)),
            ("Reading SEC filings", lambda: ingest.refresh_observations(force=full)),
            ("Building point-in-time financials", lambda: ingest.refresh_fundamentals(force=full)),
            ("Fetching share counts and consensus", lambda: ingest.refresh_snapshot(force=full)),
            ("Fetching analyst estimates", lambda: ingest.refresh_estimates(force=full)),
            ("Scoring and valuing the universe", run_analysis),
        ]
        if full
        else [
            ("Downloading prices", lambda: ingest.refresh_prices(force=True)),
            ("Scoring and valuing the universe", run_analysis),
        ]
    )

    try:
        for i, (label, step) in enumerate(stages):
            _pipeline_state.update(
                {"stage": label, "progress": i / len(stages), "error": None}
            )
            log.info("pipeline: %s", label)
            step()
        _pipeline_state.update({"stage": "Complete", "progress": 1.0})
    except Exception as exc:
        log.exception("pipeline failed")
        _pipeline_state.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "stage": "Failed",
                "traceback": traceback.format_exc()[-2000:],
            }
        )
    finally:
        _pipeline_state.update(
            {"running": False, "finished_at": datetime.now(UTC).isoformat()}
        )
        if _pipeline_lock.locked():
            _pipeline_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log.info("Spaid Analytics API starting; frontend bundle at %s", WEB_DIST)
    yield


app = FastAPI(
    title="Spaid Analytics",
    description="Private stock-recommendation and portfolio-decision engine.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.exception_handler(FileNotFoundError)
async def missing_table_handler(request, exc: FileNotFoundError):
    """A table that has not been built yet is a state, not a crash."""
    return JSONResponse(
        status_code=409,
        content={
            "detail": (
                f"{exc}. Run the pipeline to build the missing data before using this view."
            )
        },
    )


@app.get("/api/status", response_model=S.StatusResponse)
def status() -> S.StatusResponse:
    return service.get_status(_pipeline_state)


@app.get("/api/opportunities", response_model=S.OpportunitiesResponse)
def opportunities(
    limit: int | None = Query(None, ge=1, le=1000),
) -> S.OpportunitiesResponse:
    """The ranked list.

    Filtering is left to the client: the whole ranked set is a few hundred rows,
    and sending it once means filters respond instantly without a round trip.
    """
    return service.get_opportunities(limit=limit)


@app.get("/api/stock/{ticker}", response_model=S.StockDetail)
def stock(ticker: str) -> S.StockDetail:
    detail = service.get_stock(ticker)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"{ticker.upper()} is not in the current universe.",
        )
    return detail


@app.get("/api/health", response_model=S.HealthResponse)
def health() -> S.HealthResponse:
    return service.get_health()


@app.post("/api/pipeline/run")
def run_pipeline(full: bool = Query(False)) -> dict:
    """Start a pipeline run in the background.

    `full` re-fetches everything including the SEC archive, which takes several
    minutes. The default refreshes prices and recomputes scores, which takes
    under a minute and is what a daily use actually needs.
    """
    if not _pipeline_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A pipeline run is already in progress.")
    _pipeline_state.update(
        {
            "running": True,
            "stage": "Starting",
            "progress": 0.0,
            "error": None,
            "started_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
        }
    )
    thread = threading.Thread(target=_run_pipeline, kwargs={"full": full}, daemon=True)
    thread.start()
    return {"started": True, "full": full}


@app.get("/api/pipeline/status")
def pipeline_status() -> dict:
    return {
        "running": _pipeline_state["running"],
        "stage": _pipeline_state["stage"],
        "progress": _pipeline_state["progress"],
        "error": _pipeline_state["error"],
        "started_at": _pipeline_state.get("started_at"),
        "finished_at": _pipeline_state.get("finished_at"),
    }


@app.get("/api/spec")
def spec() -> dict:
    """The active model specifications, so the interface can show what it is using."""
    from spaid.config.scoring import ACTIVE_SCORING_SPEC
    from spaid.config.valuation import ACTIVE_VALUATION_SPEC

    return {
        "scoring": {
            "version": ACTIVE_SCORING_SPEC.version,
            "category_weights": {
                k.value: v for k, v in ACTIVE_SCORING_SPEC.category_weights.items()
            },
            "notes": ACTIVE_SCORING_SPEC.notes,
            "metrics": [
                {
                    "key": m.key,
                    "label": m.label,
                    "category": m.category.value,
                    "weight": m.weight,
                    "direction": m.direction,
                    "peer_basis": m.peer_basis.value,
                    "description": m.description,
                }
                for m in ACTIVE_SCORING_SPEC.metrics
            ],
        },
        "valuation": {
            "version": ACTIVE_VALUATION_SPEC.version,
            "notes": ACTIVE_VALUATION_SPEC.notes,
            "method_weights": {
                model.value: {m.value: w for m, w in weights.items()}
                for model, weights in ACTIVE_VALUATION_SPEC.method_weights.items()
            },
        },
    }


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

if WEB_DIST.exists():
    app.mount(
        "/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets"
    )

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        """Serve the single-page app, letting the client router own the path."""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Unknown endpoint.")
        candidate = WEB_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")

else:

    @app.get("/")
    def no_frontend() -> dict:
        return {
            "detail": (
                "The frontend has not been built. Run `npm install && npm run build` in "
                "the web directory, or use the API directly at /docs."
            )
        }
