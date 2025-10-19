import polars as pl
from datetime import datetime, timezone
from signalgraph.schemas import Signal
from signalgraph.utils import stable_uid

def trends_to_signals(keyword: str, series: list[tuple[datetime, int]], entity: str):
    # series: [(ts, score)]
    rows = []
    for ts, score in series:
        uid = stable_uid("google_trends", entity, ts.isoformat(), "behavior.google_trends", keyword)
        rows.append(Signal(
            uid=uid, ts=ts, entity=entity, kind="behavior.google_trends",
            value=float(score), source="google_trends",
            meta={"keyword": keyword}, asof=ts.date().isoformat()
        ).model_dump())
    return pl.DataFrame(rows)
