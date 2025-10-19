from datetime import datetime, timezone
from typing import Dict, List, Any
from .common import http_get_json, get_wm, set_wm

# NOTE: Workday has many deployments. This adapter supports the common "finder" endpoint:
# e.g., https://{host}/wday/cxs/{tenant}/jobs (POST is typical, but some GET JSON are exposed).
# To keep things simple and GET-only, we use an alternate public "job board" JSON where available.
# For companies without this public JSON, you may need to implement a POST JSON with filters.

def fetch_company(company: Dict) -> List[Dict]:
    """
    company = {
      "key": "nvidia",
      "brand": "NVIDIA",
      "domain": "nvidia.com",
      "platform": "workday",
      "host": "nvidia.wd5.myworkdayjobs.com",
      "board_path": "NVIDIAExternalCareerSite"  # path segment after host
    }
    """
    host = company["host"]
    board = company["board_path"]
    # Public search API (JSON) many WD sites expose:
    # https://{host}/wday/cxs/{tenant}/{board}/jobs (POST). For GET fallback, some sites expose:
    # https://{host}/{board}/fs/job/locations?clientRequestID=... (and then drill).
    # We'll try a known "jobs" GET aggregation endpoint leveraged by the UI in many tenants:
    url = f"https://{host}/{board}/fs/searchJob/0/0"  # public-ish; returns jobs list segments
    try:
        data = http_get_json(url)
    except Exception:
        # Fallback: try a simpler listing endpoint some tenants keep:
        alt = f"https://{host}/{board}/fs/jobPostings"
        data = http_get_json(alt)

    now = datetime.now(timezone.utc).isoformat()
    rows: List[Dict[str, Any]] = []

    # Normalize a variety of structures into a common list
    jobs: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        if "jobs" in data and isinstance(data["jobs"], list):
            jobs = data["jobs"]
        elif "data" in data and isinstance(data["data"], list):
            jobs = data["data"]
    elif isinstance(data, list):
        jobs = data

    for j in jobs:
        jid = str(j.get("id") or j.get("jobId") or j.get("number") or j.get("externalPath") or j.get("title"))
        title = j.get("title") or j.get("jobTitle")
        loc = j.get("locationsText") or j.get("location") or (j.get("locations") or [{}])[0].get("name") if j.get("locations") else None
        url_abs = j.get("externalPath") or j.get("jobPostingUrl") or j.get("url")
        dept = j.get("category") or j.get("jobFamily") or j.get("businessUnit")
        posted = j.get("postedOn") or j.get("startDate") or j.get("postedDate")
        updated = j.get("lastUpdated") or posted

        rows.append({
            "source": "job_postings",
            "platform": "workday",
            "fetched_at": now,
            "brand": company["brand"],
            "domain": company["domain"],
            "posting_id": jid,
            "title": title,
            "location": loc,
            "url": (f"https://{host}{url_abs}" if url_abs and url_abs.startswith("/") else url_abs),
            "department": dept,
            "posted_at": posted,
            "updated_at": updated,
            "raw": j,
        })

    set_wm(company["key"], {"count": len(rows), "last_fetch": now})
    return rows
