"""Runtime settings: where data lives, how we talk to providers, what the universe is.

Everything that varies by environment or preference lives here rather than being
buried in a module. Model *semantics* -- scoring weights, valuation weights --
are deliberately NOT here: they are versioned specs in `scoring.py` and
`valuation.py`, because changing them changes what a score means and that has to
be traceable in the research journal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SPAID_DATA", ROOT / "data"))

# Four storage layers, kept separate on purpose (mission: "Separate raw provider
# data, normalized canonical data, derived metrics, scores, ...").
RAW = DATA_DIR / "raw"  # provider payloads, as received
CURATED = DATA_DIR / "curated"  # canonical normalized tables
DERIVED = DATA_DIR / "derived"  # metrics, scores, valuations, recommendations
ARTIFACTS = DATA_DIR / "artifacts"  # run outputs, backtests, journal
CACHE = DATA_DIR / "cache"  # HTTP cache

for _d in (RAW, CURATED, DERIVED, ARTIFACTS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# SEC requires a descriptive User-Agent with contact details.
CONTACT = os.environ.get("SPAID_CONTACT", "SpaidAnalytics jspaid@holder.com")
USER_AGENT = CONTACT
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class UniverseConfig:
    """Which companies are eligible to be scored.

    `min_price` is compared against the *actually traded* price, never a
    split-adjusted one. Judging a 2015 price by today's share basis lets a future
    corporate action decide past eligibility, which is look-ahead bias.
    """

    name: str = "sp500"
    history_start: date = date(2013, 1, 1)
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0  # 20-day median
    benchmark: str = "SPY"
    momentum_benchmark: str = "SPMO"  # secondary benchmark required by the mission
    # Extra tickers to keep priced even when not in the universe (benchmarks,
    # sector proxies used for relative strength).
    extra_tickers: tuple[str, ...] = (
        "SPY",
        "SPMO",
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    )


# GICS sector -> the SPDR sector ETF used for relative-strength comparisons.
SECTOR_ETF: dict[str, str] = {
    "Materials": "XLB",
    "Communication Services": "XLC",
    "Energy": "XLE",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Consumer Staples": "XLP",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
}


@dataclass(frozen=True)
class HttpConfig:
    timeout: float = 30.0
    retries: int = 3
    cache_hours: float = 12.0


@dataclass(frozen=True)
class DataFreshness:
    """How old a dataset may be before the UI must flag it as stale.

    Staleness is a first-class product concept here: a recommendation built on
    three-week-old prices is a different object from one built on today's, and
    the user has to be able to see which they are looking at.
    """

    prices_hours: float = 24.0
    fundamentals_hours: float = 24.0 * 7
    universe_hours: float = 24.0 * 7
    estimates_hours: float = 24.0


@dataclass(frozen=True)
class Settings:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    freshness: DataFreshness = field(default_factory=DataFreshness)
    # Trading days per year, used for annualisation throughout.
    trading_days_per_year: int = 252
    seed: int = 7


SETTINGS = Settings()
