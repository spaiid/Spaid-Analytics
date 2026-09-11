"""The historical security master and universe-membership model.

This is the layer that decides what a backtest is *allowed* to believe. It
answers four questions for every security that has ever been in the index, and
it answers them separately so that a gap in one cannot be hidden by the others:

1. **Who is this?** A stable identifier that does not move when the ticker does,
   plus the ticker history, because tickers are recycled onto unrelated
   companies and joining prices on ticker alone eventually buys the wrong one.
2. **When was it in the index?** Entry and exit, each with the observation
   bracket it was derived from and a precision label, so a decision date inside
   an ambiguous window can be refused rather than guessed.
3. **What happened to it?** Corporate actions, delisting date, delisting reason,
   and the final return -- with `unavailable` as a first-class value. An unknown
   delisting return is never written as zero, because zero is the specific claim
   that holders lost everything.
4. **Can we actually trade it?** Whether price history exists for the period it
   was a member. A member with no prices is a hole in the universe, and the
   count of those holes is the honest size of the survivorship bias.

The output of the fourth question is what currently keeps the whole backtest
labelled exploratory, and it is measured here rather than asserted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, date, datetime

import polars as pl

from spaid.config.settings import SETTINGS
from spaid.providers.sec.extract import company_id_for
from spaid.providers.wikipedia import history as wiki_history
from spaid.providers.wikipedia.universe import classify_business_model
from spaid.storage import store
from spaid.storage.schema import (
    CORPORATE_ACTIONS,
    SECURITY_MASTER,
    TICKER_HISTORY,
    UNIVERSE_MEMBERSHIP,
    UNIVERSE_VERSIONS,
    coerce,
)

log = logging.getLogger(__name__)

SOURCE = wiki_history.SOURCE

# The first month for which a snapshot carries CIK. Before it the table gives a
# ticker and a company name only, which is not enough to identify a filer with
# the confidence a point-in-time join needs.
HISTORY_START = date(2014, 6, 30)


def security_id_for(cik: int | None, ticker: str, first_seen: date) -> str:
    """A stable, provider-independent identifier.

    Preference order is deliberate. The Central Index Key is assigned by a
    regulator, never reused, and survives renames, relistings and ticker
    changes. Only when there is no CIK at all does the identifier fall back to
    the ticker, and it pins the first date alongside it so that a ticker later
    reassigned to a different company becomes a different security rather than
    a continuation of the old one.
    """
    if cik:
        return company_id_for(int(cik))
    return f"TKR{ticker.upper()}:{first_seen.isoformat()}"


# ---------------------------------------------------------------------------
# Price coverage
# ---------------------------------------------------------------------------


def _calendar_bounds(prices: pl.DataFrame) -> tuple[date, date]:
    """The first and last complete session, taken from the benchmark's series."""
    bench = prices.filter(pl.col("ticker") == SETTINGS.universe.benchmark)
    if bench.is_empty():
        return prices["date"].min(), prices["date"].max()
    return bench["date"].min(), bench["date"].max()


def _price_coverage(prices: pl.DataFrame) -> pl.DataFrame:
    """First and last priced session per ticker, and the data's own last date."""
    if prices is None or prices.is_empty():
        return pl.DataFrame(
            schema={"ticker": pl.Utf8, "price_first": pl.Date, "price_last": pl.Date,
                    "n_bars": pl.Int32}
        )
    return (
        prices.group_by("ticker")
        .agg(
            pl.col("date").min().alias("price_first"),
            pl.col("date").max().alias("price_last"),
            pl.len().cast(pl.Int32).alias("n_bars"),
        )
        .sort("ticker")
    )


def _classify_coverage(
    price_first: date | None,
    price_last: date | None,
    member_from: date,
    member_to: date | None,
    data_start: date,
    data_end: date,
) -> str:
    """Whether prices cover the part of this spell that falls inside our data.

    The question is not "do we have the company's whole history" -- Abbott has
    been in the index since 1957 and no backtest here reaches back that far.
    It is "over the window this application can evaluate at all, can we price
    this membership". So the spell is clipped to the price table's own span
    before being judged.

    `partial` is its own category rather than being rounded up to `full`,
    because a security priced for half of its membership produces a portfolio
    that holds it for half the time and then silently stops, which looks like a
    decision and is not.
    """
    if price_first is None or price_last is None:
        return "none"
    start = max(member_from, data_start)
    end = min(member_to or data_end, data_end)
    if start > end:
        # The spell ended before our price history began; nothing to cover.
        return "none"
    return "full" if price_first <= start and price_last >= end else "partial"


# ---------------------------------------------------------------------------
# Corporate actions
# ---------------------------------------------------------------------------


def _price_actions(prices: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    """Splits and dividends, read off the price series that already carries them."""
    if prices is None or prices.is_empty():
        return pl.DataFrame()

    events = prices.join(ids, on="ticker", how="inner")

    splits = (
        events.filter(
            pl.col("split_ratio").is_not_null()
            & (pl.col("split_ratio") > 0)
            & (pl.col("split_ratio") != 1.0)
        )
        .select(
            pl.col("security_id"),
            pl.col("ticker"),
            pl.lit("split").alias("action_type"),
            pl.col("date").alias("ex_date"),
            pl.col("date").alias("effective_date"),
            pl.col("split_ratio").alias("ratio"),
            pl.lit(None, dtype=pl.Float64).alias("cash_amount"),
            pl.lit(None, dtype=pl.Utf8).alias("counterparty_security_id"),
            (
                pl.lit("Split ")
                + pl.col("split_ratio").round(4).cast(pl.Utf8)
                + pl.lit(":1")
            ).alias("description"),
        )
    )

    dividends = (
        events.filter(pl.col("dividend").is_not_null() & (pl.col("dividend") > 0))
        .select(
            pl.col("security_id"),
            pl.col("ticker"),
            pl.lit("dividend").alias("action_type"),
            pl.col("date").alias("ex_date"),
            pl.col("date").alias("effective_date"),
            pl.lit(None, dtype=pl.Float64).alias("ratio"),
            pl.col("dividend").alias("cash_amount"),
            pl.lit(None, dtype=pl.Utf8).alias("counterparty_security_id"),
            (pl.lit("Cash dividend ") + pl.col("dividend").cast(pl.Utf8)).alias("description"),
        )
    )
    return pl.concat([splits, dividends], how="vertical_relaxed")


def _membership_actions(membership: pl.DataFrame) -> pl.DataFrame:
    """Index entries and exits, as dated events rather than as row attributes."""
    entries = membership.select(
        pl.col("security_id"),
        pl.col("ticker"),
        pl.lit("index_entry").alias("action_type"),
        pl.col("entry_date").alias("ex_date"),
        pl.col("entry_date").alias("effective_date"),
        pl.lit(None, dtype=pl.Float64).alias("ratio"),
        pl.lit(None, dtype=pl.Float64).alias("cash_amount"),
        pl.lit(None, dtype=pl.Utf8).alias("counterparty_security_id"),
        (pl.lit("Joined ") + pl.col("index_name") + pl.lit(" (") + pl.col("entry_precision") + pl.lit(")")).alias("description"),
    ).filter(pl.col("ex_date").is_not_null())

    exits = (
        membership.filter(pl.col("exit_date").is_not_null())
        .select(
            pl.col("security_id"),
            pl.col("ticker"),
            pl.lit("index_exit").alias("action_type"),
            pl.col("exit_date").alias("ex_date"),
            pl.col("exit_date").alias("effective_date"),
            pl.lit(None, dtype=pl.Float64).alias("ratio"),
            pl.lit(None, dtype=pl.Float64).alias("cash_amount"),
            pl.lit(None, dtype=pl.Utf8).alias("counterparty_security_id"),
            (
                pl.lit("Left ")
                + pl.col("index_name")
                + pl.lit("; reason ")
                + pl.col("exit_reason").fill_null("unknown")
            ).alias("description"),
        )
    )
    return pl.concat([entries, exits], how="vertical_relaxed")


def _collapse_share_classes(membership: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep one membership row per company per spell, not one per share class.

    Alphabet is in the index once, not twice, and a twenty-name portfolio that
    holds GOOG and GOOGL holds nineteen businesses while believing it holds
    twenty -- and carries double the intended exposure to one of them. The
    non-primary classes are not discarded: they are returned so the ticker
    history records that the company also traded under them.

    Which class is primary is decided by evidence, in order: the one we hold
    prices for, then the one observed in more snapshots, then alphabetically so
    that the choice is at least deterministic.
    """
    # Ranking a struct orders by its fields in turn, ascending. Coverage is
    # encoded so that 0 is best, and the snapshot count is negated so that more
    # observations sort first.
    ranked = membership.with_columns(
        pl.col("price_coverage")
        .replace_strict({"full": 0, "partial": 1, "none": 2}, default=3)
        .alias("_coverage_rank"),
        (-pl.col("n_snapshots")).alias("_snapshot_rank"),
    ).with_columns(
        pl.struct(["_coverage_rank", "_snapshot_rank", "ticker"])
        .rank("ordinal")
        .over(["index_name", "security_id", "entry_observed_by"])
        .alias("_class_rank")
    )

    primary = ranked.filter(pl.col("_class_rank") == 1).drop(
        "_coverage_rank", "_snapshot_rank", "_class_rank"
    )
    secondary = ranked.filter(pl.col("_class_rank") > 1).drop(
        "_coverage_rank", "_snapshot_rank", "_class_rank"
    )
    if secondary.height:
        log.info(
            "collapsed %d secondary share classes into their primary listing (%s)",
            secondary.height,
            ", ".join(sorted(secondary["ticker"].unique().to_list())[:8]),
        )
    return primary, secondary


def _action_id(row: dict) -> str:
    key = f"{row['security_id']}|{row['action_type']}|{row['ex_date']}|{row.get('ratio')}|{row.get('cash_amount')}"
    return hashlib.blake2b(key.encode(), digest_size=12).hexdigest()


def _ticker_changes(
    snaps: dict[date, pl.DataFrame], primary_tickers: set[str]
) -> pl.DataFrame:
    """Ticker spells per company, from consecutive snapshots.

    A CIK appearing under a new ticker is a rename; a ticker appearing under a
    new CIK is a reassignment. Both are recorded, because conflating them is
    what lets a backtest hold a company it never bought.
    """
    rows: list[dict] = []
    seen: dict[int, dict] = {}  # cik -> open spell
    for as_of in sorted(snaps):
        frame = snaps[as_of]
        for row in frame.iter_rows(named=True):
            cik = row.get("cik")
            if not cik:
                continue
            ticker = row["ticker"]
            if ticker not in primary_tickers:
                continue
            spell = seen.get(cik)
            if spell is None:
                seen[cik] = {"cik": cik, "ticker": ticker, "start": as_of, "end": None,
                             "reason": "initial"}
            elif spell["ticker"] != ticker:
                spell["end"] = as_of
                rows.append(dict(spell))
                seen[cik] = {"cik": cik, "ticker": ticker, "start": as_of, "end": None,
                             "reason": "rename"}
    rows.extend(seen.values())
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def build(
    *,
    index_name: str | None = None,
    start: date = HISTORY_START,
    end: date | None = None,
    persist: bool = True,
) -> dict:
    """Reconstruct the historical universe and write the security master.

    Returns a summary that is deliberately dominated by what is *missing*: the
    number of securities with no prices, the fraction of member-months that
    cannot be traded, and the resulting data-quality verdict.
    """
    started = datetime.now(UTC)
    index_name = index_name or SETTINGS.universe.name
    collected_at = datetime.now(UTC)

    prices = store.read("prices")
    has_prices = prices is not None and not prices.is_empty()
    # The evaluable window is the benchmark's own calendar, not the maximum
    # across every ticker. Taking the maximum lets one security that happens to
    # carry a partial bar for the session in progress push the window past the
    # last complete session, at which point every *other* security looks like
    # it stops early and the whole universe reads as partially covered.
    data_start, data_end = _calendar_bounds(prices) if has_prices else (date.today(), date.today())
    end = end or data_end

    log.info("reconstructing %s membership from %s to %s", index_name, start, end)
    snaps = wiki_history.snapshots(start, end)
    if not snaps:
        raise RuntimeError("no membership snapshots could be retrieved")
    spans = wiki_history.spans_frame(wiki_history.build_spans(snaps))

    # ---- identity ---------------------------------------------------------
    spans = spans.with_columns(
        pl.struct(["cik", "ticker", "entry_observed_by"])
        .map_elements(
            lambda s: security_id_for(s["cik"], s["ticker"], s["entry_observed_by"]),
            return_dtype=pl.Utf8,
        )
        .alias("security_id"),
        pl.when(pl.col("cik").is_not_null())
        .then(
            pl.col("cik").map_elements(
                lambda c: company_id_for(int(c)), return_dtype=pl.Utf8
            )
        )
        .otherwise(None)
        .alias("company_id"),
    )

    # ---- membership -------------------------------------------------------
    # The dates the backtester acts on are the conservative edges of each
    # bracket. Entry is the first date the company was *certainly* a member, so
    # nothing is bought on the strength of an entry that may not have happened
    # yet. Exit is the first date it was certainly absent, so a holding is
    # carried through the window in which it was dropped rather than being sold
    # just before the fall that usually caused the removal. Both choices cost
    # the strategy return; that is why they are the right way round.
    membership = spans.select(
        pl.lit(index_name).alias("index_name"),
        pl.col("security_id"),
        pl.col("company_id"),
        pl.col("ticker"),
        pl.coalesce(pl.col("entry_date"), pl.col("entry_observed_by")).alias("entry_date"),
        pl.col("exit_observed_by").alias("exit_date"),
        pl.col("entry_observed_after"),
        pl.col("entry_observed_by"),
        pl.col("exit_observed_after"),
        pl.col("exit_observed_by"),
        pl.when(pl.col("entry_date").is_not_null())
        .then(pl.lit("exact"))
        .otherwise(pl.lit("bracketed"))
        .alias("entry_precision"),
        pl.when(pl.col("still_member"))
        .then(pl.lit("still_member"))
        .otherwise(pl.lit("bracketed"))
        .alias("exit_precision"),
        # The source no longer publishes reasons for index changes, and an exit
        # is not evidence of any particular fate. Left unknown.
        pl.lit(None, dtype=pl.Utf8).alias("exit_reason"),
        pl.lit(None, dtype=pl.Boolean).alias("tradable"),
        pl.lit(None, dtype=pl.Utf8).alias("universe_version"),
        pl.lit(SOURCE).alias("source"),
        pl.lit(collected_at).alias("collected_at"),
        # carried for the master, dropped before the table is written
        pl.col("name"),
        pl.col("sector"),
        pl.col("industry"),
        pl.col("cik"),
        pl.col("still_member"),
        pl.col("n_snapshots"),
    )

    # ---- price coverage ---------------------------------------------------
    coverage = _price_coverage(prices)
    membership = membership.join(coverage, on="ticker", how="left").with_columns(
        pl.struct(["price_first", "price_last", "entry_date", "exit_date"])
        .map_elements(
            lambda s: _classify_coverage(
                s["price_first"], s["price_last"], s["entry_date"], s["exit_date"],
                data_start, data_end,
            ),
            return_dtype=pl.Utf8,
        )
        .alias("price_coverage")
    ).with_columns((pl.col("price_coverage") == "full").alias("tradable"))

    # ---- one company, one membership row ----------------------------------
    membership, secondary_classes = _collapse_share_classes(membership)

    # ---- security master --------------------------------------------------
    master = (
        membership.sort(["security_id", "entry_observed_by"])
        .group_by("security_id")
        .agg(
            pl.col("company_id").last(),
            pl.col("cik").last(),
            pl.col("name").last(),
            pl.col("sector").last(),
            pl.col("industry").last(),
            pl.col("ticker").last().alias("current_ticker"),
            pl.col("entry_date").min().alias("first_date"),
            pl.col("exit_date").max().alias("_last_exit"),
            pl.col("still_member").any().alias("_still"),
            pl.col("price_first").min(),
            pl.col("price_last").max(),
            pl.col("price_coverage").last(),
        )
    )

    # A security whose price series stops well before the data does has been
    # delisted; one whose prices run to the end has not. Two weeks of slack
    # absorbs a stale final bar without turning a live company into a delisting.
    master = master.with_columns(
        pl.when(pl.col("_still"))
        .then(None)
        .otherwise(pl.col("_last_exit"))
        .alias("last_date"),
        pl.when(
            (~pl.col("_still"))
            & pl.col("price_last").is_not_null()
            & (pl.col("price_last") < pl.lit(data_end) - pl.duration(days=14))
        )
        .then(pl.col("price_last"))
        .otherwise(None)
        .alias("delisting_date"),
    ).with_columns(
        pl.when(pl.col("_still"))
        .then(pl.lit("listed"))
        .when(pl.col("delisting_date").is_not_null())
        .then(pl.lit("delisted"))
        # Left the index but we have no evidence it stopped trading. Saying
        # "unknown" is the accurate statement; saying "listed" would assert
        # something we did not check.
        .otherwise(pl.lit("unknown"))
        .alias("status"),
        # Why it went is genuinely unknown: the source stopped publishing
        # reasons, and an index exit is not itself a fate.
        pl.when(pl.col("_still"))
        .then(None)
        .otherwise(pl.lit("unknown"))
        .alias("delisting_reason"),
        # Never zero. A null here means "we do not know what holders received",
        # which is a different claim from "holders received nothing", and the
        # backtest refuses to hold a security whose ending it cannot price.
        pl.lit(None, dtype=pl.Float64).alias("delisting_return"),
        pl.when(pl.col("_still"))
        .then(None)
        .otherwise(pl.lit("unavailable"))
        .alias("delisting_return_status"),
        pl.lit(None, dtype=pl.Float64).alias("final_value_per_share"),
        pl.lit(None, dtype=pl.Utf8).alias("successor_security_id"),
        pl.lit(None, dtype=pl.Utf8).alias("share_class"),
        pl.lit(True).alias("is_primary"),
        pl.struct(["sector", "industry"])
        .map_elements(
            lambda s: classify_business_model(s["sector"], s["industry"]),
            return_dtype=pl.Utf8,
        )
        .alias("business_model"),
        pl.lit(SOURCE).alias("source"),
    ).with_columns(
        pl.when(pl.col("price_coverage") == "none")
        .then(pl.lit("no_prices"))
        .when(pl.col("price_coverage") == "partial")
        .then(pl.lit("partial_prices"))
        .when(pl.col("company_id").is_null())
        .then(pl.lit("unresolved_identity"))
        .otherwise(pl.lit("complete"))
        .alias("data_quality"),
        pl.struct(["price_coverage", "_still"])
        .map_elements(
            lambda s: json.dumps(
                {
                    "identity": "sec_cik via wikipedia snapshot",
                    "membership": SOURCE,
                    "prices": "yahoo" if s["price_coverage"] != "none" else "unavailable",
                    "delisting": "not sourced",
                }
            ),
            return_dtype=pl.Utf8,
        )
        .alias("provenance"),
        pl.lit(collected_at).alias("collected_at"),
    )
    master = coerce(master, SECURITY_MASTER)

    # ---- ticker history ---------------------------------------------------
    # Only the primary listing contributes rename events. Without that filter a
    # dual-class company looks like it renames itself every month, because the
    # two classes take turns appearing first in the table.
    primary_tickers = set(membership["ticker"].unique().to_list())
    changes = _ticker_changes(snaps, primary_tickers)
    spells: list[pl.DataFrame] = []
    if not changes.is_empty():
        spells.append(
            changes.with_columns(
                pl.col("cik")
                .map_elements(lambda c: company_id_for(int(c)), return_dtype=pl.Utf8)
                .alias("security_id"),
                pl.col("start").alias("start_date"),
                pl.col("end").alias("end_date"),
                pl.lit(SOURCE).alias("source"),
                # The snapshot grid locates a rename within a month, so the date
                # is an observation, not the company's own announcement.
                pl.lit("inferred").alias("confidence"),
                pl.lit(collected_at).alias("collected_at"),
            )
        )
    if secondary_classes.height:
        spells.append(
            secondary_classes.select(
                pl.col("security_id"),
                pl.col("ticker"),
                pl.col("entry_observed_by").alias("start_date"),
                pl.col("exit_observed_by").alias("end_date"),
                pl.lit("share_class").alias("reason"),
                pl.lit(SOURCE).alias("source"),
                pl.lit("confirmed").alias("confidence"),
                pl.lit(collected_at).alias("collected_at"),
            )
        )
    if spells:
        ticker_history = coerce(
            pl.concat([coerce(s, TICKER_HISTORY) for s in spells], how="vertical_relaxed"),
            TICKER_HISTORY,
        ).unique(subset=list(TICKER_HISTORY.primary_key), keep="last")
    else:
        ticker_history = pl.DataFrame(schema=dict(TICKER_HISTORY.schema))

    # ---- corporate actions ------------------------------------------------
    ids = membership.select(["ticker", "security_id"]).unique(subset=["ticker"], keep="last")
    pieces = [p for p in (_price_actions(prices, ids), _membership_actions(membership)) if not p.is_empty()]
    if pieces:
        actions = pl.concat(pieces, how="vertical_relaxed")
        actions = actions.with_columns(
            pl.struct(actions.columns)
            .map_elements(_action_id, return_dtype=pl.Utf8)
            .alias("action_id"),
            pl.lit("yahoo/wikipedia").alias("source"),
            pl.lit("confirmed").alias("confidence"),
            pl.lit(collected_at).alias("collected_at"),
        )
        actions = coerce(actions, CORPORATE_ACTIONS).unique(subset=["action_id"], keep="first")
    else:
        actions = pl.DataFrame(schema=dict(CORPORATE_ACTIONS.schema))

    # ---- version and quality ----------------------------------------------
    n_exits = int(membership.filter(~pl.col("still_member")).height)
    n_tradable = int(membership.filter(pl.col("tradable")).height)
    n_untradable = membership.height - n_tradable
    removed_no_prices = int(
        membership.filter((~pl.col("still_member")) & (pl.col("price_coverage") == "none")).height
    )

    quality = {
        "snapshots": len(snaps),
        "snapshot_grid": "month_end",
        "securities": int(master.height),
        "spells": int(membership.height),
        "exits_observed": n_exits,
        "spells_tradable": n_tradable,
        "spells_without_prices": n_untradable,
        "removed_securities_without_prices": removed_no_prices,
        "delisting_returns_known": 0,
        "identity_unresolved": int(master.filter(pl.col("company_id").is_null()).height),
        "entry_precision_exact": int(
            membership.filter(pl.col("entry_precision") == "exact").height
        ),
        "exit_reason_known": 0,
    }

    payload = json.dumps(
        {
            "index": index_name,
            "start": str(start),
            "end": str(end),
            "snapshot_revisions": sorted({int(df["revid"][0]) for df in snaps.values()}),
        },
        sort_keys=True,
    )
    checksum = hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()
    universe_version = f"sp500-revhist-{end:%Y%m}-{checksum[:8]}"

    membership = membership.with_columns(pl.lit(universe_version).alias("universe_version"))
    membership_out = coerce(membership, UNIVERSE_MEMBERSHIP)

    version_row = coerce(
        pl.DataFrame(
            [
                {
                    "universe_version": universe_version,
                    "index_name": index_name,
                    "built_at": collected_at,
                    "method": "revision_history",
                    "source": SOURCE,
                    "grid": "month_end",
                    "n_snapshots": len(snaps),
                    "n_securities": master.height,
                    "n_spells": membership.height,
                    "n_exits": n_exits,
                    "n_tradable": n_tradable,
                    "coverage_start": min(snaps),
                    "coverage_end": max(snaps),
                    "checksum": checksum,
                    "quality": json.dumps(quality, sort_keys=True),
                    "notes": (
                        "Reconstructed from month-end revisions of the Wikipedia constituents "
                        "page. Membership is observational: entry and exit dates are the "
                        "conservative edges of a one-month bracket. Exit reasons and delisting "
                        "returns are not available from this source and are recorded as unknown."
                    ),
                }
            ]
        ),
        UNIVERSE_VERSIONS,
    )

    if persist:
        store.write(master, "security_master", source=SOURCE)
        store.write(ticker_history, "ticker_history", source=SOURCE)
        store.write(actions, "corporate_actions", source="yahoo/wikipedia")
        store.write(membership_out, "universe_membership", source=SOURCE)
        store.upsert(version_row, "universe_versions", source=SOURCE)

    summary = {
        "universe_version": universe_version,
        "snapshots": len(snaps),
        "coverage_start": str(min(snaps)),
        "coverage_end": str(max(snaps)),
        "securities": int(master.height),
        "membership_spells": int(membership.height),
        "observed_exits": n_exits,
        "tradable_spells": n_tradable,
        "untradable_spells": n_untradable,
        "removed_without_prices": removed_no_prices,
        "corporate_actions": int(actions.height),
        "ticker_spells": int(ticker_history.height),
        "elapsed_seconds": round((datetime.now(UTC) - started).total_seconds(), 1),
        **{f"quality_{k}": v for k, v in quality.items()},
    }
    log.info("security master: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------


def members_on(as_of: date, *, index_name: str | None = None, tradable_only: bool = False) -> pl.DataFrame:
    """Who was in the index on one date, as the reconstruction sees it.

    `tradable_only` additionally requires that we hold prices covering the
    spell. The difference between the two counts on any date is the size of the
    hole a backtest has to declare.
    """
    index_name = index_name or SETTINGS.universe.name
    membership = store.read("universe_membership", required=True)
    out = membership.filter(
        (pl.col("index_name") == index_name)
        & (pl.col("entry_date") <= as_of)
        & (pl.col("exit_date").is_null() | (pl.col("exit_date") > as_of))
    )
    if tradable_only:
        out = out.filter(pl.col("tradable"))
    return out


def coverage_report(*, index_name: str | None = None) -> dict:
    """What the universe reconstruction can and cannot support, as numbers.

    Everything the validation status depends on is computed here, so that the
    interface reads a measurement rather than repeating a claim from a docstring.
    """
    index_name = index_name or SETTINGS.universe.name
    membership = store.read("universe_membership")
    master = store.read("security_master")
    versions = store.read("universe_versions")

    if membership is None or membership.is_empty():
        return {
            "built": False,
            "reason": "The historical universe has not been reconstructed yet.",
        }

    current = membership.filter(pl.col("universe_version") == "current-list").height
    historical = membership.filter(pl.col("universe_version") != "current-list")
    if historical.is_empty():
        return {
            "built": False,
            "reason": (
                "Only the current constituents list is stored; membership has no history, so "
                "every historical ranking would be drawn from today's survivors."
            ),
            "current_rows": current,
        }

    # Price coverage lives on the security master, not on the membership row:
    # it is a property of the security, not of any one spell in the index.
    coverage = (
        master.select(["security_id", "price_coverage", "delisting_return_status"])
        if master is not None and not master.is_empty()
        else pl.DataFrame(
            schema={
                "security_id": pl.Utf8,
                "price_coverage": pl.Utf8,
                "delisting_return_status": pl.Utf8,
            }
        )
    )
    historical = historical.join(coverage, on="security_id", how="left")
    removed = historical.filter(pl.col("exit_date").is_not_null())
    removed_priced = removed.filter(pl.col("price_coverage") != "none")

    version = None
    if versions is not None and not versions.is_empty():
        version = versions.sort("built_at").tail(1).to_dicts()[0]

    delisting_known = 0
    if master is not None and not master.is_empty():
        delisting_known = int(
            master.filter(pl.col("delisting_return_status") == "observed").height
        )

    # Member-months, not securities, is the honest denominator for the
    # survivorship hole: one company missing for nine years is a larger gap
    # than nine companies missing for a month.
    member_months = int(historical["n_snapshots"].sum()) if "n_snapshots" in historical.columns else None
    return {
        "built": True,
        "universe_version": version["universe_version"] if version else None,
        "coverage_start": str(version["coverage_start"]) if version else None,
        "coverage_end": str(version["coverage_end"]) if version else None,
        "snapshots": int(version["n_snapshots"]) if version else None,
        "securities": int(historical["security_id"].n_unique()),
        "spells": int(historical.height),
        "member_months": member_months,
        "exits_observed": int(removed.height),
        "removed_securities": int(removed["security_id"].n_unique()),
        "removed_with_prices": int(removed_priced["security_id"].n_unique()),
        "removed_without_prices": int(
            removed.filter(pl.col("price_coverage") == "none")["security_id"].n_unique()
        ),
        "delisting_returns_known": delisting_known,
        "tradable_spells": int(historical.filter(pl.col("tradable")).height),
        "untradable_spells": int(historical.filter(~pl.col("tradable").fill_null(False)).height),
        "current_list_rows": current,
    }
