"""Point-in-time canonical financials, assembled from raw observations.

The rule this module exists to enforce: a financial value may be used on or
after the date it became public, never from the date the period ended. Apple's
June quarter ends on a Saturday in late June and is not public until the 10-Q is
filed at the end of July. Keying on the period end hands the model five weeks of
foresight on every fundamental input, which is more than enough to manufacture a
backtest nobody could have traded.

So everything here is built *as of a filing date*. At each date on which new
facts arrived, we ask: given only what had been filed by then, what is this
company's trailing-twelve-month revenue, and what does its balance sheet look
like? The answer is stamped with that date and never revised.

Two mechanical problems make this harder than it sounds.

**Year-to-date reporting.** Income-statement tags usually arrive as discrete
quarters, but cash-flow tags arrive cumulatively: a filer reports operating cash
flow over 90, 180, 270 and 360-day windows that all share a fiscal-year start.
Reading those as four quarters triple-counts the year. Facts sharing a start are
therefore differenced -- nine months minus six months is the third quarter --
which turns either reporting style into the same set of atomic periods.

**Restatements.** A company may refile a quarter with a different number. On any
given date the right value is the most recently *filed* version available by
then, which is what a reader would have seen. Earlier versions are not deleted;
they stay in the observations table, and the difference between them is a signal
in its own right.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import polars as pl

from spaid.providers.sec.concepts import (
    ALL_CONCEPTS,
    Aggregation,
    ConceptSpec,
    PeriodType,
)
from spaid.storage.schema import FUNDAMENTALS, coerce

log = logging.getLogger(__name__)

# A trailing-twelve-month window must cover roughly a year. Retailers and other
# 52/53-week filers land between 364 and 371 days; the band is wide enough for
# them and tight enough to reject a three-quarter stub or a five-quarter overlap.
TTM_MIN_DAYS = 350
TTM_MAX_DAYS = 385

# Consecutive periods may abut either inclusively (previous end == next start) or
# exclusively (previous end == next start - 1). Both conventions appear in the
# wild, so the chain walk tolerates a few days of slack.
CHAIN_TOLERANCE_DAYS = 4

# An atomic period shorter or longer than this is not a quarter, half or year.
MIN_PERIOD_DAYS = 20
MAX_PERIOD_DAYS = 400

# No trailing-twelve-month figure should need more than this many pieces.
MAX_TTM_PIECES = 8


@dataclass(frozen=True)
class Fact:
    """One observation, reduced to what the assembly algorithms need."""

    end: date
    filed: date
    available_at: date
    value: float
    accession: str | None
    tag_rank: int  # position in the concept's tag preference list
    start: date | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None

    @property
    def duration_days(self) -> int | None:
        return (self.end - self.start).days if self.start else None


@dataclass(frozen=True)
class AtomicPeriod:
    """A non-overlapping slice of time with a value attached."""

    start: date
    end: date
    value: float
    accessions: tuple[str, ...]
    available_at: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days


@dataclass(frozen=True)
class PitValue:
    """A value together with the date it became knowable."""

    available_at: date
    period_end: date
    value: float
    n_sources: int
    accessions: tuple[str, ...]
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    is_derived: bool = False


def _better(candidate: Fact, incumbent: Fact) -> bool:
    """Should `candidate` replace `incumbent` as the known value for a period?

    A more recent filing always wins -- that is the restatement rule. Within the
    same filing date, the more specific tag wins, which is how a filer reporting
    both a precise revenue-recognition tag and a generic `Revenues` total gets
    read the way it was meant.
    """
    if candidate.filed != incumbent.filed:
        return candidate.filed > incumbent.filed
    return candidate.tag_rank < incumbent.tag_rank


def decompose(known: dict[tuple[date | None, date], Fact]) -> list[AtomicPeriod]:
    """Turn a set of duration facts into non-overlapping atomic periods.

    Facts sharing a start date are treated as cumulative cuts of the same window
    and differenced. Facts with unique starts pass through unchanged.
    """
    by_start: dict[date, list[Fact]] = defaultdict(list)
    for (start, _end), fact in known.items():
        if start is not None:
            by_start[start].append(fact)

    atomics: list[AtomicPeriod] = []
    for start, facts in by_start.items():
        facts = sorted(facts, key=lambda f: f.end)
        prev_end: date | None = None
        prev_value = 0.0
        prev_accession: str | None = None
        prev_available: date | None = None

        for f in facts:
            if prev_end is None:
                atomics.append(
                    AtomicPeriod(
                        start=start,
                        end=f.end,
                        value=f.value,
                        accessions=(f.accession,) if f.accession else (),
                        available_at=f.available_at,
                    )
                )
            else:
                # The incremental slice between two cumulative cuts. It is only
                # knowable once *both* cuts have been filed.
                sources = tuple(a for a in (f.accession, prev_accession) if a)
                atomics.append(
                    AtomicPeriod(
                        start=prev_end,
                        end=f.end,
                        value=f.value - prev_value,
                        accessions=sources,
                        available_at=max(f.available_at, prev_available or f.available_at),
                    )
                )
            prev_end, prev_value = f.end, f.value
            prev_accession, prev_available = f.accession, f.available_at

    return [a for a in atomics if MIN_PERIOD_DAYS <= a.days <= MAX_PERIOD_DAYS]


def _index_by_end(atomics: Sequence[AtomicPeriod]) -> dict[date, AtomicPeriod]:
    """Index periods by their end date, preferring the shortest.

    A filer's third-quarter 10-Q reports both the discrete quarter and the
    nine-month cumulative total, and both end on the same day. Preferring the
    shorter one keeps the chain walk building from quarters, which is what makes
    it land on a clean twelve months.
    """
    best: dict[date, AtomicPeriod] = {}
    for a in atomics:
        current = best.get(a.end)
        if current is None or a.days < current.days:
            best[a.end] = a
    return best


def _find_preceding(
    by_end: dict[date, AtomicPeriod], cursor: date
) -> AtomicPeriod | None:
    """The period ending at (or just before) `cursor`."""
    exact = by_end.get(cursor)
    if exact is not None:
        return exact
    from datetime import timedelta

    for back in range(1, CHAIN_TOLERANCE_DAYS + 1):
        found = by_end.get(cursor - timedelta(days=back))
        if found is not None:
            return found
    return None


def latest_level(known: dict[tuple[date | None, date], Fact]) -> PitValue | None:
    """The most recently reported value, read as a level rather than a flow.

    Used for quantities reported over a window that are not additive across
    windows. A weighted-average diluted share count is the clearest case: four
    quarters of it describe the same few billion shares four times over, so
    summing them reports a company four times its real size, which then divides
    into every per-share and market-capitalisation figure downstream.

    Crucially this does *not* decompose year-to-date cuts. Differencing a
    nine-month average share count against a six-month one is meaningless
    arithmetic on a non-additive quantity, and produces small negative numbers.
    The shortest window ending latest is the current quarter's figure, which is
    what we want.
    """
    candidates = [f for f in known.values() if f.start is not None and f.value > 0]
    if not candidates:
        return None
    latest = max(
        candidates,
        key=lambda f: (f.end, -(f.duration_days or 10**6), -f.tag_rank),
    )
    return PitValue(
        available_at=latest.available_at,
        period_end=latest.end,
        value=latest.value,
        n_sources=1,
        accessions=(latest.accession,) if latest.accession else (),
        fiscal_year=latest.fiscal_year,
        fiscal_period=latest.fiscal_period,
        is_derived=False,
    )


def assemble_ttm(
    atomics: Sequence[AtomicPeriod], target_end: date | None = None
) -> PitValue | None:
    """Sum backwards from `target_end` until roughly a year is covered.

    Returns None rather than a partial figure when the chain breaks. A
    three-quarter total presented as trailing-twelve-month revenue is worse than
    no number at all: it looks plausible, it is 25% low, and nothing downstream
    can tell.
    """
    if not atomics:
        return None

    by_end = _index_by_end(atomics)
    if not by_end:
        return None
    target = target_end if target_end is not None else max(by_end)
    if target not in by_end:
        return None

    total = 0.0
    accessions: list[str] = []
    available: date | None = None
    cursor = target
    pieces = 0

    while pieces < MAX_TTM_PIECES:
        piece = _find_preceding(by_end, cursor) if pieces else by_end[target]
        if piece is None:
            return None
        total += piece.value
        accessions.extend(piece.accessions)
        available = piece.available_at if available is None else max(available, piece.available_at)
        cursor = piece.start
        pieces += 1
        covered = (target - cursor).days
        if covered >= TTM_MIN_DAYS:
            break

    covered = (target - cursor).days
    if not (TTM_MIN_DAYS <= covered <= TTM_MAX_DAYS):
        return None
    assert available is not None

    return PitValue(
        available_at=available,
        period_end=target,
        value=total,
        n_sources=pieces,
        accessions=tuple(dict.fromkeys(accessions)),
        is_derived=pieces > 1,
    )


def ttm_series(
    facts: Sequence[Fact], aggregation: Aggregation = Aggregation.SUM
) -> list[PitValue]:
    """Trailing values, one per date on which the answer changed.

    Walks filing dates in order, accumulating what was known by each, so the
    emitted value at date D uses only filings submitted on or before D.
    `aggregation` decides whether four quarters add up (revenue) or whether the
    most recent one stands alone (share count).
    """
    if not facts:
        return []

    ordered = sorted(facts, key=lambda f: (f.available_at, f.filed))
    known: dict[tuple[date | None, date], Fact] = {}
    out: list[PitValue] = []
    last_signature: tuple[date, float] | None = None

    i = 0
    n = len(ordered)
    while i < n:
        as_of = ordered[i].available_at
        while i < n and ordered[i].available_at == as_of:
            f = ordered[i]
            if f.start is None:
                i += 1
                continue  # instants are handled separately
            key = (f.start, f.end)
            incumbent = known.get(key)
            if incumbent is None or _better(f, incumbent):
                known[key] = f
            i += 1

        if aggregation is Aggregation.LAST:
            result = latest_level(known)
        else:
            result = assemble_ttm(decompose(known))
        if result is None:
            continue
        signature = (result.period_end, result.value)
        if signature == last_signature:
            continue
        last_signature = signature
        # The value is knowable from `as_of`, which is when the final piece
        # arrived; the pieces themselves may have been filed earlier.
        latest = known.get((None, result.period_end))
        out.append(
            PitValue(
                available_at=as_of,
                period_end=result.period_end,
                value=result.value,
                n_sources=result.n_sources,
                accessions=result.accessions,
                fiscal_year=latest.fiscal_year if latest else None,
                fiscal_period=latest.fiscal_period if latest else None,
                is_derived=result.is_derived,
            )
        )
    return out


def instant_series(facts: Sequence[Fact]) -> list[PitValue]:
    """Balance-sheet levels, one per date on which the latest known value changed.

    "Latest known" means the most recent period end among facts filed by that
    date -- not the largest period end in the file, which is how a mis-dated
    fact freezes a stale number in place for years.
    """
    if not facts:
        return []

    ordered = sorted(facts, key=lambda f: (f.available_at, f.filed))
    known: dict[date, Fact] = {}
    out: list[PitValue] = []
    last_signature: tuple[date, float] | None = None

    i = 0
    n = len(ordered)
    while i < n:
        as_of = ordered[i].available_at
        while i < n and ordered[i].available_at == as_of:
            f = ordered[i]
            if f.start is not None:
                i += 1
                continue
            incumbent = known.get(f.end)
            if incumbent is None or _better(f, incumbent):
                known[f.end] = f
            i += 1

        if not known:
            continue
        latest_end = max(known)
        fact = known[latest_end]
        signature = (latest_end, fact.value)
        if signature == last_signature:
            continue
        last_signature = signature
        out.append(
            PitValue(
                available_at=as_of,
                period_end=latest_end,
                value=fact.value,
                n_sources=1,
                accessions=(fact.accession,) if fact.accession else (),
                fiscal_year=fact.fiscal_year,
                fiscal_period=fact.fiscal_period,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Frame-level assembly
# ---------------------------------------------------------------------------

_UNIT_FOR = {
    "USD": "USD",
    "shares": "shares",
    "USD/shares": "USD/shares",
    "pure": "pure",
}


# A value this many times away from its neighbours is a reporting error, not a
# business event. Real companies do not grow revenue a hundredfold in a quarter,
# and the errors that do occur are order-of-magnitude scale mistakes.
SCALE_OUTLIER_FACTOR = 50.0
# Below this many comparable facts there is nothing to compare against, so the
# guard stays out of the way rather than rejecting sparse data.
SCALE_MIN_NEIGHBOURS = 6


def reject_scale_outliers(facts: list[Fact]) -> tuple[list[Fact], list[Fact]]:
    """Drop facts whose magnitude is implausible next to their neighbours.

    Filers occasionally submit a value with the wrong scale. FedEx's fiscal-2026
    net income was filed once correctly as $4.433bn and again, by a different
    filing agent, as $4,433 -- and because a later filing normally supersedes an
    earlier one, the broken value won and the company appeared to earn four
    thousand dollars. Waters' diluted share count arrived as 98 billion against a
    real figure near 60 million.

    Neither is a restatement. Both are off by a clean factor of a thousand or
    more, which is the signature of a scale error, and no amount of downstream
    care recovers from one: every margin, multiple and per-share figure built on
    it is wrong while still looking like a number.

    Comparison is against facts of a similar period length, because a quarterly
    figure and an annual one legitimately differ by four.
    """
    if len(facts) < SCALE_MIN_NEIGHBOURS:
        return facts, []

    def bucket(f: Fact) -> str:
        if f.start is None:
            return "instant"
        days = (f.end - f.start).days
        if days <= 120:
            return "quarter"
        if days <= 250:
            return "half"
        return "year"

    by_bucket: dict[str, list[Fact]] = defaultdict(list)
    for f in facts:
        by_bucket[bucket(f)].append(f)

    kept: list[Fact] = []
    rejected: list[Fact] = []
    for group in by_bucket.values():
        magnitudes = [abs(f.value) for f in group if f.value]
        if len(group) < SCALE_MIN_NEIGHBOURS or not magnitudes:
            kept.extend(group)
            continue
        magnitudes.sort()
        median = magnitudes[len(magnitudes) // 2]
        if median <= 0:
            kept.extend(group)
            continue
        for f in group:
            magnitude = abs(f.value)
            # A genuine zero is meaningful (no buybacks this quarter); only
            # non-zero values are checked against the scale of their peers.
            if magnitude > 0 and (
                magnitude > median * SCALE_OUTLIER_FACTOR
                or magnitude < median / SCALE_OUTLIER_FACTOR
            ):
                rejected.append(f)
            else:
                kept.append(f)
    return kept, rejected


def _facts_for_concept(rows: list[dict], spec: ConceptSpec) -> list[Fact]:
    """Collect and rank the observations that can serve one concept."""
    rank_of = {tag: i for i, tag in enumerate(spec.tags)}
    want_unit = _UNIT_FOR[spec.unit.value]
    out: list[Fact] = []
    for r in rows:
        rank = rank_of.get(r["tag"])
        if rank is None or r["unit"] != want_unit:
            continue
        out.append(
            Fact(
                end=r["period_end"],
                filed=r["filed"],
                available_at=r["available_at"],
                value=r["value"],
                accession=r["accession"],
                tag_rank=rank,
                start=r["period_start"],
                fiscal_year=r["fiscal_year"],
                fiscal_period=r["fiscal_period"],
            )
        )
    return out


@dataclass
class BuildStats:
    companies: int = 0
    concepts_built: int = 0
    concepts_empty: int = 0
    scale_outliers: int = 0
    per_concept: dict[str, int] = field(default_factory=dict)


def build_fundamentals(
    observations: pl.DataFrame,
    *,
    concepts: Sequence[ConceptSpec] | None = None,
    tickers: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Derive point-in-time trailing and instant financials from observations."""
    specs = list(concepts or ALL_CONCEPTS)
    collected_at = datetime.now(UTC)
    stats = BuildStats()
    records: list[dict] = []

    needed = {"company_id", "tag", "unit", "value", "period_start", "period_end",
              "filed", "available_at", "accession", "fiscal_year", "fiscal_period"}
    missing = needed - set(observations.columns)
    if missing:
        raise ValueError(f"observations frame is missing columns: {sorted(missing)}")

    obs = observations.select(sorted(needed))
    for (company_id,), group in obs.group_by(["company_id"], maintain_order=True):
        rows = group.to_dicts()
        stats.companies += 1
        ticker = (tickers or {}).get(company_id)

        for spec in specs:
            facts = _facts_for_concept(rows, spec)
            facts, dropped = reject_scale_outliers(facts)
            if dropped:
                stats.scale_outliers += len(dropped)
                log.debug(
                    "%s/%s: dropped %d facts with implausible magnitude (e.g. %s)",
                    company_id, spec.key, len(dropped),
                    f"{dropped[0].value:,.0f} on {dropped[0].end}",
                )
            if not facts:
                stats.concepts_empty += 1
                continue
            if spec.period_type is PeriodType.DURATION:
                series = ttm_series(facts, spec.aggregation)
                basis = "ttm" if spec.aggregation is Aggregation.SUM else "quarter"
            else:
                series = instant_series(facts)
                basis = "instant"
            if not series:
                stats.concepts_empty += 1
                continue
            stats.concepts_built += 1
            stats.per_concept[spec.key] = stats.per_concept.get(spec.key, 0) + 1

            for pv in series:
                records.append(
                    {
                        "company_id": company_id,
                        "ticker": ticker,
                        "concept": spec.key,
                        "basis": basis,
                        "value": pv.value,
                        "period_end": pv.period_end,
                        "available_at": pv.available_at,
                        "fiscal_year": pv.fiscal_year,
                        "fiscal_period": pv.fiscal_period,
                        "n_sources": pv.n_sources,
                        "source_accessions": ",".join(pv.accessions) or None,
                        "is_derived": pv.is_derived,
                        "collected_at": collected_at,
                    }
                )

    if not records:
        raise RuntimeError("no fundamentals could be derived from the observations")

    df = coerce(pl.DataFrame(records, infer_schema_length=None), FUNDAMENTALS)
    # One row per (company, concept, basis, available_at): if two filings on the
    # same session both move a concept, the later one is the state of knowledge.
    df = df.unique(
        subset=["company_id", "concept", "basis", "available_at"], keep="last"
    ).sort(["company_id", "concept", "available_at"])

    log.info(
        "fundamentals: %d rows, %d companies, %d concepts; rejected %d facts with "
        "implausible magnitude",
        df.height, df["company_id"].n_unique(), df["concept"].n_unique(),
        stats.scale_outliers,
    )
    return df
