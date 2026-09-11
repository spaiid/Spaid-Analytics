"""S&P 500 constituents: companies, securities and index membership.

Wikipedia's constituents table is the only free, key-free source that carries
ticker, sector, sub-industry and SEC Central Index Key together, which is what
makes it possible to join price data to filings at all.

Two limitations are stated here rather than discovered later:

* **Membership is current, not historical.** The table lists who is in the index
  today and when they joined; it does not list who was dropped. `date_added`
  lets us avoid trading a company before the index picked it, which removes half
  the survivorship bias. The other half -- deletions -- cannot be recovered from
  this source, so it is recorded as a known gap and surfaced in data health
  rather than quietly ignored.
* **A company is not a ticker.** Alphabet files once but trades as two tickers.
  Treating them as two companies lets a twenty-name portfolio hold the same
  business twice and double-counts it in every sector limit. So securities are
  mapped to one company identifier, and exactly one share class is marked
  primary.
"""

from __future__ import annotations

import io
import logging
from datetime import UTC, date, datetime

import polars as pl

from spaid.config.settings import SETTINGS
from spaid.providers.http import fetch_text
from spaid.providers.sec.extract import company_id_for
from spaid.storage.schema import COMPANIES, SECURITIES, UNIVERSE_MEMBERSHIP, coerce

log = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SOURCE = "wikipedia_sp500"


def to_yahoo(symbol: str) -> str:
    """Wikipedia writes class shares with a dot; Yahoo uses a hyphen."""
    return symbol.strip().upper().replace(".", "-")


def _parse_date(value) -> date | None:
    if value is None:
        return None
    text = str(value).strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _share_class(ticker: str) -> str | None:
    """The class suffix, if the ticker carries one (BRK-B -> B)."""
    if "-" in ticker:
        suffix = ticker.rsplit("-", 1)[-1]
        if len(suffix) <= 2 and suffix.isalpha():
            return suffix
    return None


def fetch_constituents(*, cache_hours: float = 24.0) -> pl.DataFrame:
    """Raw constituents table: one row per listed security."""
    import pandas as pd

    from spaid.config.settings import BROWSER_UA

    html = fetch_text(
        WIKI_URL,
        headers={"User-Agent": BROWSER_UA},
        cache_hours=cache_hours,
        suffix=".html",
    )
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError("no tables parsed from the Wikipedia constituents page")

    raw = tables[0]
    required = {"Symbol", "Security", "GICS Sector", "CIK"}
    if not required.issubset(raw.columns):
        raise RuntimeError(f"unexpected constituents table columns: {list(raw.columns)}")

    n = len(raw)
    df = pl.DataFrame(
        {
            "ticker": [to_yahoo(s) for s in raw["Symbol"].astype(str)],
            "name": raw["Security"].astype(str).tolist(),
            "sector": raw["GICS Sector"].astype(str).tolist(),
            "industry": raw.get("GICS Sub-Industry", pd.Series([""] * n)).astype(str).tolist(),
            "cik": [int(c) for c in raw["CIK"]],
            "date_added": [
                _parse_date(x) for x in raw.get("Date added", pd.Series([None] * n))
            ],
            "headquarters": raw.get(
                "Headquarters Location", pd.Series([""] * n)
            ).astype(str).tolist(),
        }
    ).unique(subset=["ticker"], keep="first")

    log.info(
        "constituents: %d securities, %d distinct filers, %d sectors",
        df.height, df["cik"].n_unique(), df["sector"].n_unique(),
    )
    return df.sort("ticker")


# Explicit GICS sub-industry classification.
#
# Substring matching is too blunt here and produced real errors: "Asset
# Management & Custody Banks" contains the word "bank", so Ameriprise and
# BlackRock were valued on book equity and a return-on-equity spread like
# deposit-taking lenders. An asset manager's value is in its fee streams, not
# its balance sheet, and the residual-income model priced Ameriprise at a 74%
# discount to a market that was not obviously wrong.
#
# So the financial sector is enumerated rather than matched. The distinction is
# between businesses whose *balance sheet is the business* -- lenders, brokers
# carrying inventory, insurers holding float -- and fee-earning businesses that
# happen to be classified as financial.
FINANCIAL_MODELS: dict[str, str] = {
    # Balance-sheet businesses: value comes from book equity and the return
    # earned on it.
    "Diversified Banks": "bank",
    "Regional Banks": "bank",
    "Commercial & Residential Mortgage Finance": "bank",
    "Thrifts & Mortgage Finance": "bank",
    "Consumer Finance": "bank",  # card issuers and lenders carry credit risk
    "Investment Banking & Brokerage": "bank",  # balance-sheet intensive
    # Underwriters: value comes from book plus underwriting profitability.
    "Life & Health Insurance": "insurer",
    "Property & Casualty Insurance": "insurer",
    "Multi-line Insurance": "insurer",
    "Reinsurance": "insurer",
    # Fee businesses in a financial wrapper. No underwriting risk, no lending
    # book, so they are ordinary operating companies.
    "Asset Management & Custody Banks": "operating",
    "Insurance Brokers": "operating",
    "Financial Exchanges & Data": "operating",
    "Transaction & Payment Processing Services": "operating",
    "Multi-Sector Holdings": "operating",
}

# Sub-industries whose earnings swing with a commodity or a capital cycle, where
# trailing profits describe the cycle rather than the business.
CYCLICAL_KEYWORDS: tuple[str, ...] = (
    "oil", "gas", "coal", "consumable fuels", "metals", "mining", "gold",
    "copper", "steel", "aluminum", "commodity chemicals", "fertilizers",
    "paper", "forest", "construction materials", "containers", "packaging",
    "airlines", "automobile manufacturers", "automotive parts", "auto parts",
    "homebuilding", "marine", "trucking",
)


def classify_business_model(sector: str | None, industry: str | None) -> str:
    """Assign the analytical template a company needs, from its GICS labels.

    This is the first pass, based only on what the business *is*. A second pass
    in the metrics pipeline can override it to `unprofitable_growth` when the
    financials say so, because that is a state a company enters and leaves
    rather than a permanent classification.
    """
    s = (sector or "").strip()
    i = (industry or "").strip()
    lower = i.lower()

    if s == "Financials":
        # Fall back to operating rather than guessing: mispricing a fee business
        # as a lender is a large, silent error, and an unknown sub-industry is
        # more likely to be a fee business than a bank.
        return FINANCIAL_MODELS.get(i, "operating")

    if s == "Real Estate":
        # Only landlords are trusts. Real-estate services firms and data
        # providers are ordinary operating companies in the same sector.
        return "reit" if "reit" in lower else "operating"

    if s in {"Energy", "Materials"} or any(k in lower for k in CYCLICAL_KEYWORDS):
        return "cyclical"

    return "operating"


def build_identity(
    constituents: pl.DataFrame,
    *,
    liquidity: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split the constituents table into companies, securities and membership.

    `liquidity` optionally supplies a `ticker` / `dollar_volume` frame used to
    pick which share class of a dual-listed company is the primary one. Without
    it the alphabetically first ticker is chosen, which is deterministic but
    arbitrary, and that choice is logged.
    """
    collected_at = datetime.now(UTC)
    today = datetime.now(UTC).date()

    df = constituents.with_columns(
        pl.col("cik")
        .map_elements(company_id_for, return_dtype=pl.Utf8)
        .alias("company_id"),
        pl.col("ticker")
        .map_elements(_share_class, return_dtype=pl.Utf8)
        .alias("share_class"),
    )

    # ---- primary share class -------------------------------------------------
    if liquidity is not None and not liquidity.is_empty():
        df = df.join(liquidity.select(["ticker", "dollar_volume"]), on="ticker", how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("dollar_volume"))

    df = df.with_columns(
        pl.col("dollar_volume").fill_null(0.0).alias("_liq"),
    ).with_columns(
        (
            pl.struct(["_liq", "ticker"]).rank("ordinal", descending=True).over("company_id") == 1
        ).alias("is_primary")
    )

    multi = (
        df.group_by("company_id")
        .agg(pl.len().alias("n"), pl.col("ticker").sort())
        .filter(pl.col("n") > 1)
    )
    for row in multi.iter_rows(named=True):
        chosen = (
            df.filter((pl.col("company_id") == row["company_id"]) & pl.col("is_primary"))
            ["ticker"].to_list()
        )
        log.info(
            "multi-class filer %s: %s -> primary %s",
            row["company_id"], row["ticker"], chosen,
        )

    # ---- companies -----------------------------------------------------------
    companies = (
        df.filter(pl.col("is_primary"))
        .select(
            pl.col("company_id"),
            pl.col("cik"),
            pl.col("name"),
            pl.col("sector"),
            pl.col("industry"),
            pl.lit("US").alias("country"),
            pl.lit(None, dtype=pl.Utf8).alias("sic"),
            pl.struct(["sector", "industry"])
            .map_elements(
                lambda s: classify_business_model(s["sector"], s["industry"]),
                return_dtype=pl.Utf8,
            )
            .alias("business_model"),
            pl.lit(None, dtype=pl.Utf8).alias("fiscal_year_end"),
            pl.col("date_added").alias("first_seen"),
            pl.lit(today).alias("last_seen"),
            pl.lit(collected_at).alias("collected_at"),
        )
        .unique(subset=["company_id"], keep="first")
        .sort("company_id")
    )

    # ---- securities ----------------------------------------------------------
    securities = (
        df.select(
            pl.col("company_id"),
            pl.col("ticker"),
            pl.col("share_class"),
            pl.col("is_primary"),
            pl.col("date_added").alias("first_date"),
            pl.lit(None, dtype=pl.Date).alias("last_date"),
            pl.lit("listed").alias("status"),
            pl.lit(None, dtype=pl.Utf8).alias("successor_company_id"),
            pl.lit(collected_at).alias("collected_at"),
        )
        .unique(subset=["company_id", "ticker"], keep="first")
        .sort(["company_id", "ticker"])
    )

    # ---- index membership ----------------------------------------------------
    membership = (
        df.filter(pl.col("is_primary"))
        .select(
            pl.lit(SETTINGS.universe.name).alias("index_name"),
            pl.col("company_id"),
            pl.col("ticker"),
            pl.col("date_added"),
            pl.lit(None, dtype=pl.Date).alias("date_removed"),
            pl.lit(SOURCE).alias("source"),
            pl.lit(collected_at).alias("collected_at"),
        )
        .unique(subset=["index_name", "company_id", "date_added"], keep="first")
        .sort("company_id")
    )

    companies = coerce(companies, COMPANIES)
    securities = coerce(securities, SECURITIES)
    membership = coerce(membership, UNIVERSE_MEMBERSHIP)

    log.info(
        "identity: %d companies, %d securities (%d non-primary classes collapsed)",
        companies.height, securities.height, securities.height - companies.height,
    )
    counts = companies.group_by("business_model").agg(pl.len().alias("n")).sort("n", descending=True)
    log.info("business models: %s", dict(counts.iter_rows()))
    return companies, securities, membership
