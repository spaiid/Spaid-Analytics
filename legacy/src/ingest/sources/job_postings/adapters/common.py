import time, json, os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import requests

# --- Polite HTTP with simple retry ---
def http_get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, retries: int = 3, sleep: float = 0.6) -> Any:
    ua = {"User-Agent": "SignalForge/0.1 (+research; polite)"}
    if headers:
        ua.update(headers)
    last_exc = None
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, headers=ua, timeout=15)
            if r.status_code == 429:
                time.sleep(2.0)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(sleep)
    if last_exc:
        raise last_exc

# --- Watermark storage (per-company incremental fetch) ---
STATE_FILE = Path(os.environ.get("SF_STATE_PATH", "./_state/watermarks.json"))

def load_watermarks() -> Dict[str, Dict[str, Any]]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_watermarks(state: Dict[str, Dict[str, Any]]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)

def get_wm(company_key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    wm = load_watermarks()
    return wm.get(company_key, default or {})

def set_wm(company_key: str, value: Dict[str, Any]) -> None:
    wm = load_watermarks()
    wm[company_key] = value
    save_watermarks(wm)
