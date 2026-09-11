from datetime import datetime, timezone
from typing import Dict, Iterable, List
from .common import http_get_json, get_wm, set_wm

# Public API doc: https://developers.greenhouse.io/job-board.html
# Typical job board URL: https://boards-api.greenhouse.io/v1/boards/{org}/jobs

def fetch_company(company: Dict) -> List[Dict]:
    """
    company = {
      "key": "stripe",              # unique key in registry
      "brand": "Stripe",
      "domain": "stripe.com",
      "platform": "greenhouse",
      "org": "stripe"               # greenhouse org slug
    }
    """
    org = company["org"]
    url = f"https://boards-api.greenhouse.io/v1/boards/{org}/jobs"
    data = http_get_json(url, params={"content": "true"})  # include descriptions
    rows = []
    now = datetime.now(timezone.utc).isoformat()

    # Basic incremental (Greenhouse lacks updated_since on this endpoint; we dedup later by posting_id)
    for j in data.get("jobs", []):
        rows.append({
            "source": "job_postings",
            "platform": "greenhouse",
            "fetched_at": now,
            "brand": company["brand"],
            "domain": company["domain"],
            "posting_id": str(j.get("id")),
            "title": j.get("title"),
            "location": (j.get("location") or {}).get("name"),
            "url": j.get("absolute_url"),
            "department": (j.get("departments") or [{}])[0].get("name") if j.get("departments") else None,
            "posted_at": j.get("updated_at") or j.get("created_at"),
            "updated_at": j.get("updated_at"),
            "raw": j,
        })
    # No watermark to set (no delta API). We still set count for transparency.
    set_wm(company["key"], {"count": len(rows), "last_fetch": now})
    return rows
