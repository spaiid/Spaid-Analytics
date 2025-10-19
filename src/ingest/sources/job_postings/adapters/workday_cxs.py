from __future__ import annotations
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import time
import requests

# POST-based Workday "cxs" search API:
#   https://{host}/wday/cxs/{tenant}/{board}/jobs
# Body example:
#   {"appliedFacets":{}, "limit":20, "offset":0, "searchText":""}
#
# This adapter paginates until no more results. It is resilient to slight
# schema variations across tenants.

DEFAULT_LIMIT = 50
DEFAULT_TIMEOUT = 15.0
UA = {"User-Agent": "SignalForge/0.1 (+research; polite)"}

def _post_json(url: str, body: dict, timeout: float = DEFAULT_TIMEOUT) -> dict:
    r = requests.post(url, json=body, headers=UA, timeout=timeout)
    # Many tenants return 4xx/5xx intermittently; don't retry too aggressively here.
    r.raise_for_status()
    return r.json()

def _safe_iso(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    # Some fields are already ISO; others might be like "2024-09-02"
    try:
        # Try parse common Workday format "YYYY-MM-DD"
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return dt.isoformat()
        # Else, accept as-is
        return s
    except Exception:
        return None

def fetch_company(company: Dict) -> List[Dict]:
    """
    company = {
      "key": "nvidia",
      "brand": "NVIDIA",
      "domain": "nvidia.com",
      "platform": "workday_cxs",
      "host": "nvidia.wd5.myworkdayjobs.com",
      "tenant": "wday",                      # often literally "wday" for external
      "board": "NVIDIAExternalCareerSite",   # path segment
      # optional:
      # "query": "",                          # search text
      # "limit": 50,                          # page size (<= 50 is polite)
      # "sleep": 0.2,                         # between pages
      # "timeout": 15.0                       # request timeout
    }
    """
    host     = company["host"].rstrip("/")
    tenant   = company.get("tenant", "wday")
    board    = company["board"]
    query    = company.get("query", "")
    limit    = int(company.get("limit", DEFAULT_LIMIT))
    sleep    = float(company.get("sleep", 0.2))
    timeout  = float(company.get("timeout", DEFAULT_TIMEOUT))

    url = f"https://{host}/wday/cxs/{tenant}/{board}/jobs"

    rows: List[Dict[str, Any]] = []
    offset = 0
    total_seen = 0
    now = datetime.now(timezone.utc).isoformat()

    # Base body; we keep this minimal for breadth.
    base_body = {
        "appliedFacets": {},   # add facet filters if needed later
        "limit": limit,
        "offset": offset,
        "searchText": query,
    }

    while True:
        body = dict(base_body, offset=offset)
        data = _post_json(url, body, timeout=timeout)

        # Common shapes:
        # {"total":1234, "jobPostings":[{...}, {...}]}
        job_list = []
        if isinstance(data, dict):
            if isinstance(data.get("jobPostings"), list):
                job_list = data["jobPostings"]
            elif isinstance(data.get("data"), list):
                job_list = data["data"]

        if not job_list:
            break

        for j in job_list:
            # Broad normalization across flavors we’ve seen
            jid = (
                j.get("id")
                or j.get("jobId")
                or j.get("number")
                or j.get("externalPath")
                or j.get("title")
            )
            title = j.get("title") or j.get("jobTitle")
            loc = (
                j.get("locationsText")
                or j.get("location")
                or (j.get("locations") or [{}])[0].get("name")
                if j.get("locations") else None
            )
            url_abs = j.get("externalPath") or j.get("jobPostingUrl") or j.get("url")
            dept = j.get("category") or j.get("jobFamily") or j.get("businessUnit")
            posted = _safe_iso(j.get("postedOn") or j.get("startDate") or j.get("postedDate"))
            updated = _safe_iso(j.get("lastUpdated") or posted)

            rows.append({
                "source": "job_postings",
                "platform": "workday",
                "fetched_at": now,
                "brand": company["brand"],
                "domain": company["domain"],
                "posting_id": str(jid) if jid is not None else None,
                "title": title,
                "location": loc,
                "url": (f"https://{host}{url_abs}" if url_abs and str(url_abs).startswith("/") else url_abs),
                "department": dept,
                "posted_at": posted,
                "updated_at": updated,
                "raw": j,
            })

        total_seen += len(job_list)
        offset += limit
        # Some tenants cap results; stop if we reached reported total
        if isinstance(data.get("total"), int) and total_seen >= int(data["total"]):
            break

        # Safety: stop if page returned less than limit
        if len(job_list) < limit:
            break

        time.sleep(sleep)

    return rows
