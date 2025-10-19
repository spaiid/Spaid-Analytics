import re
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from playwright.sync_api import sync_playwright

DEFAULT_TIMEOUT = 25_000  # ms
DEFAULT_LIMIT = 50

SEARCH_PAGES = (
    "/",  # many sites boot the search right on /
    "/en-US/careers", "/en-US/careers/job-search",
    "/careers", "/career", "/job-search",
)

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _tenant_candidates(host: str, explicit: Optional[str]) -> List[str]:
    cand = []
    if explicit:
        cand.append(explicit)
    prefix = host.split(".", 1)[0]
    if prefix not in cand:
        cand.append(prefix)
    if "wday" not in cand:
        cand.append("wday")
    return cand

def _learn_request(page, host: str, tenant_hint: str, board_hint: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Navigate to a likely careers page and wait for the site's own
    /wday/cxs/{tenant}/{board}/jobs request. Return its url and json body.
    """
    pattern = re.compile(r"/wday/cxs/([^/]+)/([^/]+)/jobs$")
    learned = None

    def on_request(req):
        nonlocal learned
        url = req.url
        m = pattern.search(url)
        if not m:
            return
        t_found, b_found = m.group(1), m.group(2)
        if tenant_hint and t_found != tenant_hint:
            return
        if board_hint and b_found != board_hint:
            return
        # Only POSTs carry the full schema; GETs can be misleading
        if req.method != "POST":
            return
        try:
            body = req.post_data_json
        except Exception:
            body = None
        if isinstance(body, dict):
            learned = {
                "url": url,
                "tenant": t_found,
                "board": b_found,
                "body": body,
            }

    page.on("request", on_request)

    for path in SEARCH_PAGES:
        try:
            page.goto(f"https://{host}{path}", wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT)
            # many sites lazy-load; poke the search UI to fire:
            page.wait_for_timeout(800)
            if learned:
                break
            # common trick: type a space then backspace into the search box to trigger
            for sel in ("input[type=search]", "input[role=searchbox]", "input[name=keyword]", "input#keyword"):
                try:
                    if page.locator(sel).first.is_visible():
                        box = page.locator(sel).first
                        box.click()
                        page.keyboard.type(" ")
                        page.keyboard.press("Backspace")
                        page.wait_for_timeout(600)
                        if learned:
                            break
                except Exception:
                    pass
            if learned:
                break
        except Exception:
            continue

    page.remove_listener("request", on_request)
    return learned

def _fetch_json_in_page(page, url: str, payload: dict) -> dict:
    js = """
    async (url, body) => {
      const resp = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json;charset=UTF-8',
          'Accept': 'application/json, text/plain, */*'
        },
        body: JSON.stringify(body),
        credentials: 'include'
      });
      if (!resp.ok) {
        return { __error__: resp.status + ' ' + resp.statusText, __url__: url, __status__: resp.status, __text__: await resp.text() };
      }
      return await resp.json();
    }
    """
    data = page.evaluate(js, url, payload)
    if isinstance(data, dict) and data.get("__error__"):
        raise RuntimeError(f"{data['__error__']} for url: {data['__url__']}")
    return data

def fetch_company(company: Dict[str, Any]) -> List[Dict[str, Any]]:
    host   = company["host"].rstrip("/")
    board  = company.get("board")
    tenant = company.get("tenant")
    query  = company.get("query", "")
    limit  = int(company.get("limit", DEFAULT_LIMIT))
    sleep_ms = int(company.get("sleep", 250))

    tenants = _tenant_candidates(host, tenant)
    results: List[Dict[str, Any]] = []
    now_iso = _now_iso()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) SignalForge/0.1")
        page = context.new_page()

        # Warm cookies
        page.goto(f"https://{host}/", wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT)

        learned = None
        for t in tenants:
            learned = _learn_request(page, host, t, board)
            if learned:
                break

        if not learned:
            context.close()
            browser.close()
            return results  # graceful: nothing learned → empty

        # Seed values from learned request
        url = learned["url"]
        payload = learned["body"] or {}
        # Respect their schema; just update pagination + search text if present in their schema.
        if "limit" in payload:
            payload["limit"] = limit
        if "offset" not in payload:
            payload["offset"] = 0
        if "searchText" in payload:
            payload["searchText"] = query

        total_seen = 0
        while True:
            data = _fetch_json_in_page(page, url, payload)

            # Allow for several shapes: jobPostings, data, items, positions
            job_list = []
            if isinstance(data, dict):
                for key in ("jobPostings", "data", "items", "positions"):
                    if isinstance(data.get(key), list):
                        job_list = data[key]
                        break

            if not job_list:
                break

            for j in job_list:
                title = j.get("title") or j.get("jobTitle")
                loc = (j.get("locationsText") or j.get("location") or
                       ((j.get("locations") or [{}])[0].get("name") if j.get("locations") else None))
                url_abs = j.get("externalPath") or j.get("jobPostingUrl") or j.get("url")
                jid = j.get("id") or j.get("jobId") or j.get("number") or j.get("externalPath") or j.get("title")
                dept = j.get("category") or j.get("jobFamily") or j.get("businessUnit")
                posted = j.get("postedOn") or j.get("startDate") or j.get("postedDate")
                updated = j.get("lastUpdated") or posted

                results.append({
                    "source": "job_postings",
                    "platform": "workday",
                    "fetched_at": now_iso,
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

            # Paginate using the site's own schema: bump 'offset' if present, else break
            if "offset" in payload and "limit" in payload:
                payload["offset"] = int(payload.get("offset", 0)) + int(payload.get("limit", limit))
            else:
                break

            # Stop if the payload or response advertised a total
            total = None
            for key in ("total", "totalHits", "totalResults"):
                if isinstance(data.get(key), int):
                    total = data[key]
                    break
            if isinstance(total, int) and total_seen >= total:
                break

            page.wait_for_timeout(sleep_ms)

        context.close()
        browser.close()

    return results
