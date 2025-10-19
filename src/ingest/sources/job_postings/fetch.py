from datetime import datetime, timezone
from typing import Dict, List
import json
import polars as pl

from src.schemas.raw_job_posting import RawJobPosting
from src.common.io import write_parquet_partition
from .adapters import greenhouse, lever, workday, workday_cxs

# near the top of fetch.py
from pathlib import Path
import json
STATE_HEALTH = Path("./_state/company_health.json")

def is_alive(key: str) -> bool:
    try:
        data = json.loads(STATE_HEALTH.read_text(encoding="utf-8"))
        oks = {r["key"] for r in data.get("ok", [])}
        return key in oks
    except Exception:
        return True  # include all if no health file yet


# --- Company registry ---
COMPANIES: list[dict] = [
    # Greenhouse
    {"key": "stripe",      "brand": "Stripe",      "domain": "stripe.com",      "platform": "greenhouse", "org": "stripe"},
    {"key": "coinbase",    "brand": "Coinbase",    "domain": "coinbase.com",    "platform": "greenhouse", "org": "coinbase"},
    {"key": "databricks",  "brand": "Databricks",  "domain": "databricks.com",  "platform": "greenhouse", "org": "databricks"},
    {"key": "brex",        "brand": "Brex",        "domain": "brex.com",        "platform": "greenhouse", "org": "brex"},
    {"key": "scaleai",     "brand": "Scale AI",    "domain": "scale.com",       "platform": "greenhouse", "org": "scaleai"},
    {"key": "figma",       "brand": "Figma",       "domain": "figma.com",       "platform": "greenhouse", "org": "figma"},
    {"key": "datadog",     "brand": "Datadog",     "domain": "datadoghq.com",   "platform": "greenhouse", "org": "datadog"},
    {"key": "toast",       "brand": "Toast",       "domain": "toasttab.com",    "platform": "greenhouse", "org": "toast"},
    {"key": "instacart",   "brand": "Instacart",   "domain": "instacart.com",   "platform": "greenhouse", "org": "instacart"},
    {"key": "airbnb",      "brand": "Airbnb",      "domain": "airbnb.com",      "platform": "greenhouse", "org": "airbnb"},
    {"key": "cloudflare",  "brand": "Cloudflare",  "domain": "cloudflare.com",  "platform": "greenhouse", "org": "cloudflare"},
    {"key": "robinhood",   "brand": "Robinhood",   "domain": "robinhood.com",   "platform": "greenhouse", "org": "robinhood"},
    {"key": "affirm",      "brand": "Affirm",      "domain": "affirm.com",      "platform": "greenhouse", "org": "affirm"},
    {"key": "asana",       "brand": "Asana",       "domain": "asana.com",       "platform": "greenhouse", "org": "asana"},
    {"key": "discord",     "brand": "Discord",     "domain": "discord.com",     "platform": "greenhouse", "org": "discord"},
    {"key": "lyft",        "brand": "Lyft",        "domain": "lyft.com",        "platform": "greenhouse", "org": "lyft"},
    {"key": "okta",        "brand": "Okta",        "domain": "okta.com",        "platform": "greenhouse", "org": "okta"},
    {"key": "twilio",      "brand": "Twilio",      "domain": "twilio.com",      "platform": "greenhouse", "org": "twilio"},
    {"key": "dropbox",     "brand": "Dropbox",     "domain": "dropbox.com",     "platform": "greenhouse", "org": "dropbox"},
    {"key": "benchling",   "brand": "Benchling",   "domain": "benchling.com",   "platform": "greenhouse", "org": "benchling"},

    # Lever (these two worked for you)
    {"key": "rover",       "brand": "Rover",       "domain": "rover.com",       "platform": "lever",       "org": "rover"},
    {"key": "angellist",   "brand": "AngelList",   "domain": "angel.co",        "platform": "lever",       "org": "angellist"},

    # --- Workday via POST (cxs) for F500/big-tech ---
    {"key": "nvidia",     "brand": "NVIDIA",    "domain": "nvidia.com",     "platform": "workday_cxs",
    "host": "nvidia.wd5.myworkdayjobs.com",    "tenant": "wday", "board": "NVIDIAExternalCareerSite"},

    {"key": "amd",        "brand": "AMD",       "domain": "amd.com",        "platform": "workday_cxs",
    "host": "amd.wd1.myworkdayjobs.com",       "tenant": "wday", "board": "External"},

    {"key": "microsoft",  "brand": "Microsoft", "domain": "microsoft.com",  "platform": "workday_cxs",
    "host": "microsoft.wd3.myworkdayjobs.com", "tenant": "wday", "board": "Microsoft"},

    {"key": "adobe",      "brand": "Adobe",     "domain": "adobe.com",      "platform": "workday_cxs",
    "host": "adobe.wd5.myworkdayjobs.com",     "tenant": "wday", "board": "AdobeExternal"},

    {"key": "salesforce", "brand": "Salesforce","domain": "salesforce.com", "platform": "workday_cxs",
    "host": "salesforce.wd1.myworkdayjobs.com","tenant": "wday", "board": "External"},

    {"key": "intel",      "brand": "Intel",     "domain": "intel.com",      "platform": "workday_cxs",
    "host": "intel.wd1.myworkdayjobs.com",     "tenant": "wday", "board": "External"},

    {"key": "boeing",     "brand": "Boeing",    "domain": "boeing.com",     "platform": "workday_cxs",
    "host": "boeing.wd1.myworkdayjobs.com",    "tenant": "wday", "board": "Boeing"},

    {"key": "lockheed",   "brand": "Lockheed Martin","domain": "lockheedmartin.com","platform": "workday_cxs",
    "host": "lockheedmartin.wd1.myworkdayjobs.com","tenant": "wday", "board": "External"},

    {"key": "jpmorgan",   "brand": "J.P. Morgan","domain": "jpmorganchase.com","platform": "workday_cxs",
    "host": "jpmchase.wd5.myworkdayjobs.com",  "tenant": "wday", "board": "JPMC"},

    {"key": "goldmansachs","brand": "Goldman Sachs","domain": "goldmansachs.com","platform": "workday_cxs",
    "host": "gs.wd1.myworkdayjobs.com",        "tenant": "wday", "board": "GS"},
]



ADAPTERS = {
    "greenhouse": greenhouse.fetch_company,
    "lever": lever.fetch_company,
    "workday": workday.fetch_company,
    "workday_cxs": workday_cxs.fetch_company
}

SCHEMA = {
    "source": pl.Utf8,
    "platform": pl.Utf8,
    "fetched_at": pl.Utf8,
    "dt": pl.Utf8,
    "brand": pl.Utf8,
    "domain": pl.Utf8,
    "posting_id": pl.Utf8,
    "title": pl.Utf8,
    "location": pl.Utf8,
    "url": pl.Utf8,
    "department": pl.Utf8,
    "posted_at": pl.Utf8,
    "updated_at": pl.Utf8,
    "raw": pl.Utf8,  # store JSON string to avoid nested-struct inference panics
    "company_key": pl.Utf8,
}



def _to_json_str(obj) -> str | None:
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return None

def fetch_job_postings(dt: str) -> str:
    all_rows: List[dict] = []

    for c in COMPANIES:
        if not is_alive(c["key"]):
            continue
        fn = ADAPTERS.get(c["platform"])
        if not fn:
            continue
        try:
            rows = fn(c)
            for r in rows:
                r["dt"] = dt
                # Normalize field presence/types
                r.setdefault("source", "job_postings")
                r.setdefault("platform", None)
                r.setdefault("brand", None)
                r.setdefault("domain", None)
                r["posting_id"] = None if r.get("posting_id") is None else str(r["posting_id"])
                # flatten raw to JSON string (critical: avoid nested dict dtype mismatches)
                r["raw"] = _to_json_str(r.get("raw"))
                r["company_key"] = c["key"]
                # Validate, but keep going on validation hiccups
                try:
                    all_rows.append(RawJobPosting(**r).model_dump())
                except Exception:
                    all_rows.append(r)
        except Exception as e:
            print(f"[fetch_job_postings] {c['key']} error: {e}")

    if not all_rows:
        return ""

    # Build with a fixed schema to keep Polars happy across adapters
    df = pl.from_dicts(all_rows, schema=SCHEMA, strict=False).with_columns(
        [pl.col(k).cast(t) for k, t in SCHEMA.items()]
    )

    return write_parquet_partition(df, "raw", "job_postings", dt)
