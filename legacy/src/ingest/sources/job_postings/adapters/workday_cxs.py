from __future__ import annotations
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import time
import requests

DEFAULT_LIMIT = 50
DEFAULT_TIMEOUT = 15.0

def _headers(host: str) -> dict:
    base = f"https://{host}"
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SignalForge/0.1",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": base,
        "Referer": base + "/",
        "Content-Type": "application/json;charset=UTF-8",
    }

class WorkdayHTTPError(Exception):
    pass

def _post_json(url: str, body: dict, host: str, timeout: float) -> dict:
    r = requests.post(url, json=body, headers=_headers(host), timeout=timeout)
    if r.status_code in (400, 404, 422):
        raise WorkdayHTTPError(f"{r.status_code} {r.reason} for url: {url}")
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        raise WorkdayHTTPError(f"Non-JSON response from {url} (len={len(r.content)})")

def _get_json(url: str, host: str, params: dict, timeout: float) -> dict:
    r = requests.get(url, params=params, headers=_headers(host), timeout=timeout)
    if r.status_code in (400, 404, 422):
        raise WorkdayHTTPError(f"{r.status_code} {r.reason} for url: {r.url}")
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        raise WorkdayHTTPError(f"Non-JSON response from {r.url} (len={len(r.content)})")


def _safe_iso(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).isoformat()
        return s
    except Exception:
        return None

def _tenant_candidates(company: Dict) -> List[str]:
    cand = []
    if company.get("tenant"):
        cand.append(str(company["tenant"]))
    prefix = company["host"].split(".", 1)[0]
    if prefix not in cand:
        cand.append(prefix)
    if "wday" not in cand:
        cand.append("wday")
    return cand

# Known body shapes tenants accept; we’ll try in order.
def _body_templates(limit: int, offset: int, query: str) -> List[dict]:
    return [
        # Minimal (works for many)
        {"appliedFacets": {}, "limit": limit, "offset": offset, "searchText": query},
        # Add languages
        {
            "appliedFacets": {},
            "limit": limit, "offset": offset, "searchText": query,
            "siteLanguage": "en-US", "userSelectedLanguage": "en-US"
        },
        # Add languages + sort newest first (common at big tenants)
        {
            "appliedFacets": {},
            "limit": limit, "offset": offset, "searchText": query,
            "siteLanguage": "en-US", "userSelectedLanguage": "en-US",
            "sort": "Most recent"  # Workday accepts string labels on some sites
        },
        # Explicit facet keys (empty arrays) – some schemas require presence
        {
            "appliedFacets": {"locations": [], "timeType": [], "jobFamilyGroup": [], "workerSubType": []},
            "limit": limit, "offset": offset, "searchText": query,
            "siteLanguage": "en-US", "userSelectedLanguage": "en-US",
        },
    ]

def _try_page(host: str, tenant: str, board: str, limit: int, offset: int, query: str, timeout: float) -> dict:
    base = f"https://{host}/wday/cxs/{tenant}/{board}/jobs"

    last_err = None

    # 1) Try POST with several body templates (some tenants are picky)
    for body in _body_templates(limit, offset, query):
        try:
            return _post_json(base, body, host, timeout)
        except WorkdayHTTPError as e:
            last_err = e
            continue

    # 2) GET fallback (a few tenants only accept GET for search)
    try:
        return _get_json(base, host, {"limit": limit, "offset": offset, "searchText": query}, timeout)
    except WorkdayHTTPError as e:
        last_err = e

    if last_err:
        raise last_err
    raise WorkdayHTTPError("Unknown Workday error")

def fetch_company(company: Dict) -> List[Dict]:
    """
    Required fields in company:
      host:    e.g., "nvidia.wd5.myworkdayjobs.com"
      board:   e.g., "NVIDIAExternalCareerSite"
      brand, domain, key: used for normalization/metadata
    Optional:
      tenant, query, limit, sleep, timeout
    """
    host     = company["host"].rstrip("/")
    board    = company["board"]
    query    = company.get("query", "")
    limit    = int(company.get("limit", DEFAULT_LIMIT))
    sleep    = float(company.get("sleep", 0.25))  # be polite
    timeout  = float(company.get("timeout", DEFAULT_TIMEOUT))
    tenants  = _tenant_candidates(company)

    rows: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    total_seen = 0
    offset = 0
    chosen_tenant: Optional[str] = None

    while True:
        last_err = None
        for tenant in tenants if chosen_tenant is None else [chosen_tenant]:
            try:
                data = _try_page(host, tenant, board, limit, offset, query, timeout)
                chosen_tenant = tenant
                break
            except WorkdayHTTPError as e:
                last_err = e
                continue
            except requests.HTTPError as e:
                # Real server errors (403/500) – surface for visibility
                raise
        else:
            if last_err:
                raise last_err
            raise WorkdayHTTPError("No valid tenant endpoint")

        job_list = []
        if isinstance(data, dict):
            if isinstance(data.get("jobPostings"), list):
                job_list = data["jobPostings"]
            elif isinstance(data.get("data"), list):
                job_list = data["data"]

        if not job_list:
            break

        for j in job_list:
            jid = j.get("id") or j.get("jobId") or j.get("number") or j.get("externalPath") or j.get("title")
            title = j.get("title") or j.get("jobTitle")
            loc = (j.get("locationsText") or j.get("location") or
                   ((j.get("locations") or [{}])[0].get("name") if j.get("locations") else None))
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
        if isinstance(data.get("total"), int) and total_seen >= int(data["total"]):
            break
        if len(job_list) < limit:
            break
        time.sleep(sleep)

    return rows
