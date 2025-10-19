# tools/check_companies.py
import sys, json, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ingest.sources.job_postings.fetch import COMPANIES
from ingest.sources.job_postings.adapters import greenhouse, lever, workday, workday_cxs

ADAPTERS = {
    "greenhouse": greenhouse.fetch_company,
    "lever": lever.fetch_company,
    "workday": workday.fetch_company,
    "workday_cxs": workday_cxs.fetch_company,
}

def probe(company, verbose=False):
    key = company["key"]
    platform = company["platform"]
    fn = ADAPTERS.get(platform)
    start = time.time()
    if verbose:
        print(f"[start] {key:<18} {platform}")
    if not fn:
        return {"key": key, "platform": platform, "ok": False, "reason": "no_adapter", "secs": time.time()-start}
    try:
        rows = fn(company)  # just probe; we don't write to _data
        secs = time.time() - start
        if verbose:
            print(f"[ok]    {key:<18} {platform:<10} rows={len(rows):<5} t={secs:.2f}s")
        return {"key": key, "platform": platform, "ok": True, "rows": len(rows), "secs": secs}
    except Exception as e:
        secs = time.time() - start
        if verbose:
            print(f"[fail]  {key:<18} {platform:<10} {type(e).__name__}: {e} t={secs:.2f}s")
        return {"key": key, "platform": platform, "ok": False, "reason": str(e), "secs": secs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=list(ADAPTERS.keys()), help="Only test one platform")
    ap.add_argument("--keys", help="Comma-separated company keys to test (overrides --platform)")
    ap.add_argument("--limit", type=int, help="Limit number of companies")
    ap.add_argument("--concurrency", type=int, default=6, help="Parallel probes")
    ap.add_argument("--verbose", action="store_true", help="Log each call start/finish")
    args = ap.parse_args()

    # Select companies
    comps = COMPANIES
    if args.platform:
        comps = [c for c in comps if c["platform"] == args.platform]
    if args.keys:
        keys = {k.strip() for k in args.keys.split(",")}
        comps = [c for c in comps if c["key"] in keys]
    if args.limit:
        comps = comps[: args.limit]

    print(f"Probing {len(comps)} companies (concurrency={args.concurrency})...")
    results = {"ok": [], "fail": []}

    # Run in parallel (requests releases the GIL; be polite to endpoints overall)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(probe, c, args.verbose): c for c in comps}
        for fut in as_completed(futs):
            r = fut.result()
            (results["ok"] if r.get("ok") else results["fail"]).append(r)

    # Save & print summary
    out_path = ROOT / "_state" / "company_health.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok_n, fail_n = len(results["ok"]), len(results["fail"])
    print(f"\nOK: {ok_n} | FAIL: {fail_n}")
    if args.verbose:
        # Top performers by rows/second
        tops = sorted(results["ok"], key=lambda r: (r.get("rows", 0)/(r.get("secs") or 1)), reverse=True)[:10]
        if tops:
            print("\nTop throughput (rows/sec):")
            for r in tops:
                rate = r["rows"] / (r["secs"] or 1)
                print(f"  {r['key']:<18} {r['platform']:<10} rows={r['rows']:<5} t={r['secs']:.2f}s  rps={rate:.1f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
