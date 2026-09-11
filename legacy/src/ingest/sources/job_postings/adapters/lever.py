from datetime import datetime, timezone
from typing import Dict, List
from .common import http_get_json, get_wm, set_wm

# Public job site Feed: https://api.lever.co/v0/postings/{company}?mode=json
# Some orgs use "lever.co/{company}" slugs; the postings endpoint is consistent.

def fetch_company(company: Dict) -> List[Dict]:
    """
    company = {
      "key": "robinhood",
      "brand": "Robinhood",
      "domain": "robinhood.com",
      "platform": "lever",
      "org": "robinhood"
    }
    """
    org = company["org"]
    url = f"https://api.lever.co/v0/postings/{org}"
    data = http_get_json(url, params={"mode": "json"})
    now = datetime.now(timezone.utc).isoformat()

    rows = []
    for j in data:
        # Lever postings have id, createdAt/updatedAt in ms
        def ts_ms_to_iso(ms):
            try:
                return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc).isoformat()
            except Exception:
                return None
        rows.append({
            "source": "job_postings",
            "platform": "lever",
            "fetched_at": now,
            "brand": company["brand"],
            "domain": company["domain"],
            "posting_id": str(j.get("id") or j.get("lever_id") or j.get("_id")),
            "title": j.get("text") or j.get("title"),
            "location": (j.get("categories") or {}).get("location"),
            "url": j.get("hostedUrl") or j.get("applyUrl"),
            "department": (j.get("categories") or {}).get("team"),
            "posted_at": ts_ms_to_iso(j.get("createdAt")),
            "updated_at": ts_ms_to_iso(j.get("updatedAt")),
            "raw": j,
        })
    set_wm(company["key"], {"count": len(rows), "last_fetch": now})
    return rows
