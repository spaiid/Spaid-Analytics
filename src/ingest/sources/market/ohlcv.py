def ohlcv_to_signals(ticker: str, rows: list[dict]):
    # rows: [{"date":"2025-10-17","close":...,"volume":...}]
    sigs = []
    for r in rows:
        d = r["date"]
        ts = datetime.fromisoformat(d + "T21:00:00+00:00")
        sigs += [
            Signal(uid=stable_uid("polygon", ticker, ts.isoformat(), "market.price","close"),
                   ts=ts, entity=ticker, kind="market.price", value=float(r["close"]),
                   source="polygon", meta={"field":"close"}, asof=d).model_dump(),
            Signal(uid=stable_uid("polygon", ticker, ts.isoformat(), "market.volume","vol"),
                   ts=ts, entity=ticker, kind="market.volume", value=float(r["volume"]),
                   source="polygon", meta={}, asof=d).model_dump(),
        ]
    return pl.DataFrame(sigs)
