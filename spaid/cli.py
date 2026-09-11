"""Command-line entry point.

Everything the application does is reachable from here, so a scheduled job and a
person at a terminal use the same code path the interface does.

    spaid ingest          fetch prices, filings, estimates
    spaid analyze         score and value the universe
    spaid run             both, in order
    spaid serve           start the web interface
    spaid health          data-quality report
    spaid top             the ranked list, in the terminal
    spaid stock TICKER    one company's full analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # These libraries are chatty at INFO and say nothing useful.
    for noisy in ("httpx", "httpcore", "urllib3", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_ingest(args) -> int:
    from spaid.pipeline import ingest

    result = ingest.ingest_all(force=args.force, limit=args.limit)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_analyze(args) -> int:
    from spaid.pipeline.analyze import run_analysis

    result = run_analysis(weekly_years=args.years, daily_days=args.daily_days)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_run(args) -> int:
    rc = cmd_ingest(args)
    return rc or cmd_analyze(args)


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(
        "spaid.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def cmd_health(args) -> int:
    from spaid.pipeline import quality
    from spaid.storage import store

    rows = store.health()
    width = max(len(r["name"]) for r in rows) + 2
    print(f"{'TABLE'.ljust(width)}{'STATUS':10}{'ROWS':>12}{'ENTITIES':>10}  SPAN")
    for r in sorted(rows, key=lambda x: (x["layer"], x["name"])):
        rows_text = f"{r['rows']:,}" if r.get("rows") is not None else "-"
        entities = f"{r['entities']:,}" if r.get("entities") is not None else "-"
        print(
            f"{r['name'].ljust(width)}{r['status']:10}{rows_text:>12}{entities:>10}  "
            f"{r.get('span') or ''}"
        )

    warnings = quality.check_all()
    if warnings:
        print(f"\n{len(warnings)} data-quality notes:\n")
        for w in warnings:
            print(f"  [{w['severity']}] {w['message']}\n")
    else:
        print("\nNo data-quality warnings.")
    return 0


def cmd_top(args) -> int:
    from spaid.api import service

    data = service.get_opportunities(limit=args.limit)
    if not data.rows:
        print("No scores yet. Run `spaid run` first.", file=sys.stderr)
        return 1

    print(
        f"As of {data.as_of} - {data.scored_count} of {data.universe_size} scored "
        f"(spec {data.spec_version})\n"
    )
    header = (
        f"{'#':>4} {'TICKER':8}{'NAME':28}{'SCORE':>6} {'BAND':18}"
        f"{'VALUATION':28}{'UPSIDE':>8} {'CONF':10}"
    )
    print(header)
    print("-" * len(header))
    for r in data.rows:
        upside = f"{r.upside * 100:+.0f}%" if r.upside is not None else "-"
        print(
            f"{r.rank or 0:>4} {r.ticker:8}{(r.name or '')[:26]:28}"
            f"{r.score or 0:>6.1f} {(r.band or '')[:16]:18}"
            f"{(r.valuation_label or '')[:26]:28}{upside:>8} {(r.confidence_label or ''):10}"
        )
    if data.notes:
        print()
        for n in data.notes:
            print(f"note: {n}")
    return 0


def cmd_stock(args) -> int:
    from spaid.api import service

    detail = service.get_stock(args.ticker)
    if detail is None:
        print(f"{args.ticker.upper()} is not in the universe.", file=sys.stderr)
        return 1

    if args.json:
        print(detail.model_dump_json(indent=2))
        return 0

    print(f"\n{detail.ticker} - {detail.name}")
    print(f"{detail.sector} / {detail.industry}  ({detail.business_model})")
    print(f"As of {detail.as_of}\n")

    score = f"{detail.score:.1f}" if detail.score is not None else "not scored"
    print(f"Composite score : {score}  {detail.band or ''}")
    if detail.rank:
        print(f"Rank            : {detail.rank} of {detail.universe_size}")
    if detail.confidence:
        print(f"Confidence      : {detail.confidence.label} ({detail.confidence.score:.2f})")

    print("\nCategories")
    for c in detail.categories:
        s = f"{c.score:6.1f}" if c.score is not None else "     -"
        contribution = f"{c.contribution:5.1f}" if c.contribution is not None else "    -"
        print(
            f"  {c.label:10} {s}  weight {c.weight:.2f}  contributes {contribution}  "
            f"coverage {c.coverage:.0%}  ({c.n_missing}/{c.n_metrics} missing)"
        )

    fv = detail.fair_value
    if fv:
        print("\nFair value")
        print(f"  Price          : {fv.price:,.2f}")
        if fv.range_low is not None:
            print(f"  Range          : {fv.range_low:,.2f} to {fv.range_high:,.2f}")
            print(f"  Midpoint       : {fv.midpoint:,.2f}")
        if fv.upside is not None:
            print(f"  Upside to mid  : {fv.upside * 100:+.1f}%")
        print(f"  Classification : {fv.classification_label}")
        print(f"  Confidence     : {fv.confidence_label}")
        print("  Methods:")
        for m in fv.methods:
            if m.used:
                print(f"    - {m.label} (weight {m.weight:.0%}): {m.value_per_share:,.2f}")
            else:
                print(f"    - {m.label}: not used - {m.reason}")
        if fv.caveats:
            print("  Why this may be wrong:")
            for c in fv.caveats:
                print(f"    - {c}")

    if detail.value_trap:
        t = detail.value_trap
        print(f"\nValue situation : {t.classification_label} ({t.n_fired} warning signals)")
        for s in t.signals:
            if s.fired:
                print(f"    - {s.label}")

    if detail.strengths:
        print("\nStrengths")
        for s in detail.strengths[:5]:
            print(f"  + {s}")
    if detail.weaknesses:
        print("\nWeaknesses")
        for s in detail.weaknesses[:5]:
            print(f"  - {s}")
    if detail.risks:
        print("\nRisks")
        for s in detail.risks[:6]:
            print(f"  ! {s}")
    print()
    return 0


def cmd_explain(args) -> int:
    """Every metric behind one company's score, so the arithmetic can be checked."""
    from spaid.api import service

    detail = service.get_stock(args.ticker)
    if detail is None:
        print(f"{args.ticker.upper()} is not in the universe.", file=sys.stderr)
        return 1

    print(f"\n{detail.ticker} - how the score was calculated\n")
    running = 0.0
    total_weight = 0.0
    for c in detail.categories:
        print(f"{c.label}  (weight {c.weight:.0%}, coverage {c.coverage:.0%})")
        for m in c.metrics:
            if m.status == "scored":
                print(
                    f"   {m.label:36} {str(m.display_value):>10}  "
                    f"score {m.score:5.1f}  w {m.weight:.2f}  "
                    f"vs {m.peer_count} {m.peer_basis} peers (median {m.peer_median_display})"
                )
            else:
                print(f"   {m.label:36} {'-':>10}  {m.status}: {m.status_detail}")
        if c.score is not None:
            print(f"   -> category score {c.score:.1f} x weight {c.weight:.2f} = {c.contribution:.2f}")
            running += c.contribution
            total_weight += c.weight
        else:
            print("   -> not scored; excluded and remaining weights renormalised")
        print()

    if total_weight > 0:
        print(f"Composite = {running:.2f} / {total_weight:.2f} = {running / total_weight:.1f}")
        print(f"Reported  = {detail.score:.1f}" if detail.score is not None else "Reported  = -")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spaid", description="Spaid Analytics - stock analysis and portfolio decisions"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="fetch and normalise source data")
    p.add_argument("--force", action="store_true", help="refetch even if cached data is fresh")
    p.add_argument("--limit", type=int, help="only process N companies (for testing)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("analyze", help="compute metrics, scores and valuations")
    p.add_argument("--years", type=int, default=6, help="years of weekly history to compute")
    p.add_argument("--daily-days", type=int, default=30, help="recent days at daily resolution")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("run", help="ingest then analyze")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--years", type=int, default=6)
    p.add_argument("--daily-days", type=int, default=30)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="start the web interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("health", help="data-quality report")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("top", help="the ranked list")
    p.add_argument("-n", "--limit", type=int, default=25)
    p.set_defaults(func=cmd_top)

    p = sub.add_parser("stock", help="one company's analysis")
    p.add_argument("ticker")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stock)

    p = sub.add_parser("explain", help="every metric behind one company's score")
    p.add_argument("ticker")
    p.set_defaults(func=cmd_explain)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
