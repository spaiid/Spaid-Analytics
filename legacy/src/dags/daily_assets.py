from dagster import (
    asset, define_asset_job, ScheduleDefinition, DefaultScheduleStatus,
    AssetExecutionContext, Definitions
)
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import pandas_market_calendars as mcal

from ingest.sources.job_postings.fetch import fetch_job_postings
from ingest.sources.job_postings.normalize import normalize_job_postings
from features.job_postings import job_posting_features

# --- Trading day inference (NYSE / XNYS) ---
def current_trading_day_et() -> str:
    """
    Returns the active NYSE session date (YYYY-MM-DD):
    - If it's a trading day and we're before the open -> returns the previous session
    - If between open and close -> returns today
    - If after close -> returns today
    - If weekend/holiday -> returns the most recent prior session
    """
    now = datetime.now(ZoneInfo("America/New_York"))
    nyse = mcal.get_calendar("XNYS")

    # Build a small window around 'now' to cover previous and next sessions
    start = (now - timedelta(days=10)).date()
    end = (now + timedelta(days=2)).date()

    sched = nyse.schedule(start_date=start, end_date=end, tz="America/New_York")
    # Rows indexed by session date; columns: market_open, market_close (tz-aware)
    ts_now = pd.Timestamp(now)

    # Is today a session?
    if now.date() in sched.index:
        today_row = sched.loc[now.date()]
        if ts_now < today_row["market_open"]:
            # Before today's open -> use previous session
            prior_idx = sched.index.get_loc(now.date()) - 1
            if prior_idx >= 0:
                return sched.index[prior_idx].strftime("%Y-%m-%d")
        # During or after today's session -> use today
        return now.date().strftime("%Y-%m-%d")

    # Not a session day (weekend/holiday): pick the most recent session before now
    prior = sched[sched["market_open"] <= ts_now].tail(1)
    if not prior.empty:
        return prior.index[-1].strftime("%Y-%m-%d")

    # Fallback (very early window edge): first available in schedule
    return sched.index[0].strftime("%Y-%m-%d")

def dt_from(context: AssetExecutionContext) -> str:
    cfg = getattr(context, "op_config", None) or {}
    return cfg.get("dt") or current_trading_day_et()

@asset
def raw_job_postings(context: AssetExecutionContext):
    return fetch_job_postings(dt_from(context))

@asset(deps=[raw_job_postings])
def normalized_job_postings(context: AssetExecutionContext):
    return normalize_job_postings(dt_from(context))

@asset(deps=[normalized_job_postings])
def features_job_postings(context: AssetExecutionContext):
    return job_posting_features(dt_from(context))

daily_job = define_asset_job("daily_job", selection="*")

schedule = ScheduleDefinition(
    job=daily_job,
    cron_schedule="5 1 * * *",
    default_status=DefaultScheduleStatus.RUNNING,
)

defs = Definitions(
    assets=[raw_job_postings, normalized_job_postings, features_job_postings],
    jobs=[daily_job],
    schedules=[schedule],
)
