"""Canonical table schemas.

Every table the application persists is declared here with its columns, dtypes,
primary key and a schema version. Three reasons this is worth the ceremony:

* A renamed or retyped column becomes a loud failure at read time instead of a
  silent column of nulls feeding a recommendation.
* The mission's data-integrity requirements ("every financial observation should
  include fields equivalent to ...") are checkable rather than aspirational, and
  `OBSERVATIONS` below is the table that carries them.
* Migrations have something to migrate between.

The layering is deliberate and is enforced by which module may write which table:

    raw/        provider payloads, untouched
    curated/    canonical normalized data (companies, securities, prices,
                observations, fundamentals, estimates)
    derived/    things we computed (metrics, scores, valuations, recommendations)
    artifacts/  run outputs, backtests, the research journal
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TableSpec:
    name: str
    layer: str  # raw | curated | derived | artifacts
    schema: dict[str, pl.DataType]
    primary_key: tuple[str, ...]
    # Columns that must never be null. Nullability is a modelling statement:
    # a null `value` in an observation is a bug, a null `fiscal_period` is not.
    required: tuple[str, ...] = ()
    entity_col: str | None = "company_id"
    date_col: str | None = None
    description: str = ""

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.schema)

    def polars_schema(self) -> pl.Schema:
        return pl.Schema(self.schema)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
# Tickers are reused and reassigned; a ticker is a label, not an identity. The
# SEC's Central Index Key is the stable identifier for a filer, so `company_id`
# is derived from it ("CIK0000320193") and every other table joins on that.

COMPANIES = TableSpec(
    name="companies",
    layer="curated",
    schema={
        "company_id": pl.Utf8,
        "cik": pl.Int64,
        "name": pl.Utf8,
        "sector": pl.Utf8,
        "industry": pl.Utf8,
        "country": pl.Utf8,
        "sic": pl.Utf8,
        "business_model": pl.Utf8,  # spaid.config.scoring.BusinessModel
        "fiscal_year_end": pl.Utf8,  # "MM-DD"
        "first_seen": pl.Date,
        "last_seen": pl.Date,
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id",),
    required=("company_id", "cik", "name"),
    date_col=None,
    description="One row per filing entity, keyed by SEC Central Index Key.",
)

SECURITIES = TableSpec(
    name="securities",
    layer="curated",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "share_class": pl.Utf8,
        "is_primary": pl.Boolean,  # the class we price and trade
        "first_date": pl.Date,
        "last_date": pl.Date,  # null while still listed
        "status": pl.Utf8,  # listed | delisted | acquired | merged | bankrupt
        "successor_company_id": pl.Utf8,
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "ticker"),
    required=("company_id", "ticker"),
    description=(
        "Ticker-to-company mapping through time. Dual-class issuers have one row per class; "
        "only the primary class enters the scored universe so the same company cannot occupy "
        "two portfolio slots."
    ),
)

SECURITY_MASTER = TableSpec(
    name="security_master",
    layer="curated",
    schema={
        # Provider-independent and stable for the life of the business. Derived
        # from the SEC Central Index Key where the company files, and otherwise
        # from the first ticker and the date it was first observed -- never from
        # a provider's own internal id, which would make the whole history
        # hostage to one vendor.
        "security_id": pl.Utf8,
        "company_id": pl.Utf8,  # null where the security could not be resolved to a filer
        "cik": pl.Int64,
        "name": pl.Utf8,
        # Classification travels with the security rather than with the
        # companies table, because that table holds today's constituents and a
        # security removed from the index in 2017 has no row there -- yet it
        # still needs a peer group to be ranked inside on a 2017 decision date.
        "sector": pl.Utf8,
        "industry": pl.Utf8,
        "business_model": pl.Utf8,  # spaid.config.scoring.BusinessModel
        "current_ticker": pl.Utf8,  # the most recent ticker, for display only
        "share_class": pl.Utf8,
        "is_primary": pl.Boolean,
        "first_date": pl.Date,  # first evidence of the security existing
        "last_date": pl.Date,  # last evidence; null while still listed
        "status": pl.Utf8,  # listed | delisted | acquired | merged | bankrupt | unknown
        "delisting_date": pl.Date,
        # Why it stopped trading. `unknown` is a real and common value here and
        # is never upgraded to a guess: an index exit is not by itself evidence
        # of an acquisition, a bankruptcy or anything else.
        "delisting_reason": pl.Utf8,
        # The return earned on the final day, including whatever holders
        # received. Deliberately nullable, and deliberately never defaulted to
        # zero: a zero delisting return is the claim that holders were wiped
        # out, which is a very different statement from "we do not know".
        "delisting_return": pl.Float64,
        "delisting_return_status": pl.Utf8,  # observed | estimated | unavailable
        "final_value_per_share": pl.Float64,
        "successor_security_id": pl.Utf8,
        # Whether we hold prices for this security, and over what span. A
        # security with no price history cannot be held in a backtest, and the
        # count of these is the size of the survivorship hole.
        "price_coverage": pl.Utf8,  # full | partial | none
        "price_first": pl.Date,
        "price_last": pl.Date,
        "source": pl.Utf8,
        "provenance": pl.Utf8,  # json: which source supplied which field
        # What is wrong with this row, in one word, so that a caller can filter
        # rather than having to re-derive the judgement.
        "data_quality": pl.Utf8,  # complete | no_prices | partial_prices | unresolved_identity
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("security_id",),
    required=("security_id", "current_ticker", "status", "data_quality"),
    entity_col="security_id",
    description=(
        "Provider-independent security master: one row per tradable security ever seen, "
        "including those that have since been delisted, with the identity, the corporate "
        "fate, the price coverage we actually hold and the provenance of each."
    ),
)

TICKER_HISTORY = TableSpec(
    name="ticker_history",
    layer="curated",
    schema={
        "security_id": pl.Utf8,
        "ticker": pl.Utf8,
        "start_date": pl.Date,
        "end_date": pl.Date,  # null while current
        "reason": pl.Utf8,  # initial | rename | reassignment | share_class
        "source": pl.Utf8,
        "confidence": pl.Utf8,  # confirmed | inferred
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("security_id", "ticker", "start_date"),
    required=("security_id", "ticker", "start_date"),
    entity_col="security_id",
    date_col="start_date",
    description=(
        "Which ticker a security traded under, and when. Tickers are reassigned to "
        "unrelated companies after a delisting, so a backtest that joins prices on ticker "
        "alone will silently buy a different business."
    ),
)

CORPORATE_ACTIONS = TableSpec(
    name="corporate_actions",
    layer="curated",
    schema={
        "action_id": pl.Utf8,
        "security_id": pl.Utf8,
        "ticker": pl.Utf8,
        # split | dividend | ticker_change | merger | acquisition | spinoff |
        # delisting | bankruptcy | index_entry | index_exit
        "action_type": pl.Utf8,
        "ex_date": pl.Date,
        "effective_date": pl.Date,
        "ratio": pl.Float64,  # split ratio, or exchange ratio for a stock merger
        "cash_amount": pl.Float64,  # per share
        "counterparty_security_id": pl.Utf8,
        "description": pl.Utf8,
        "source": pl.Utf8,
        "confidence": pl.Utf8,  # confirmed | inferred
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("action_id",),
    required=("action_id", "security_id", "action_type"),
    entity_col="security_id",
    date_col="ex_date",
    description=(
        "Splits, dividends, ticker changes, mergers and delistings, each with the source "
        "that asserted it and whether it was confirmed or inferred."
    ),
)

UNIVERSE_MEMBERSHIP = TableSpec(
    name="universe_membership",
    layer="curated",
    schema={
        "index_name": pl.Utf8,
        "security_id": pl.Utf8,
        "company_id": pl.Utf8,  # null where identity could not be resolved
        "ticker": pl.Utf8,
        # The dates the backtester actually uses. Where the source states an
        # exact date they are exact; where membership was reconstructed from
        # periodic observations they are the conservative edges of the bracket
        # below -- a company is treated as a member only over the window in
        # which it was certainly one.
        "entry_date": pl.Date,
        "exit_date": pl.Date,  # null while still a member
        # The observation bracket. `entry_observed_after` is the last date the
        # company was seen absent, `entry_observed_by` the first date it was
        # seen present; the true entry lies between them. Storing both is what
        # allows the backtest to state its own precision instead of implying
        # one it does not have.
        "entry_observed_after": pl.Date,
        "entry_observed_by": pl.Date,
        "exit_observed_after": pl.Date,
        "exit_observed_by": pl.Date,
        "entry_precision": pl.Utf8,  # exact | bracketed | unknown
        "exit_precision": pl.Utf8,  # exact | bracketed | unknown | still_member
        "exit_reason": pl.Utf8,  # unknown unless a source actually says
        # Whether this membership row can be acted on: a member we hold no
        # prices for is a hole in the universe, not a tradable name.
        "tradable": pl.Boolean,
        "universe_version": pl.Utf8,
        "source": pl.Utf8,
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("index_name", "security_id", "entry_observed_by"),
    required=("index_name", "security_id", "ticker", "entry_observed_by"),
    entity_col="security_id",
    date_col="entry_date",
    description=(
        "Index membership through time, one row per continuous spell. Exits are what make "
        "a survivorship-free backtest possible; where they cannot be recovered the row "
        "says so through `exit_precision` rather than reading as 'still a member'."
    ),
)

UNIVERSE_VERSIONS = TableSpec(
    name="universe_versions",
    layer="artifacts",
    schema={
        "universe_version": pl.Utf8,
        "index_name": pl.Utf8,
        "built_at": pl.Datetime("us", "UTC"),
        "method": pl.Utf8,  # current_snapshot | revision_history
        "source": pl.Utf8,
        "grid": pl.Utf8,  # the observation cadence, e.g. "month_end"
        "n_snapshots": pl.Int32,
        "n_securities": pl.Int32,
        "n_spells": pl.Int32,
        "n_exits": pl.Int32,
        "n_tradable": pl.Int32,
        "coverage_start": pl.Date,
        "coverage_end": pl.Date,
        "checksum": pl.Utf8,
        "quality": pl.Utf8,  # json: the measured gaps
        "notes": pl.Utf8,
    },
    primary_key=("universe_version",),
    required=("universe_version", "built_at", "method"),
    entity_col=None,
    description=(
        "One row per reconstruction of the historical universe. A backtest records which "
        "version it ran against, so a result can be reproduced even after the "
        "reconstruction improves."
    ),
)

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

PRICES = TableSpec(
    name="prices",
    layer="curated",
    schema={
        "ticker": pl.Utf8,
        "date": pl.Date,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        # Both series are stated on TODAY's share basis, and they differ only by
        # dividends. The provider restates its whole history for splits, so
        # Nvidia's June 2015 close reads $0.56 in `close_raw` where the stock
        # actually traded at $22.38 -- a forty-fold difference, being the splits
        # since. Two consequences that have to be carried by every consumer:
        #
        #   * A price screen must not be applied to `close_raw` directly, or a
        #     split that happens in 2024 retroactively makes a company
        #     ineligible in 2015. Multiply by `split_factor` first to recover
        #     the price the stock genuinely traded at.
        #   * A simulator must NOT adjust its share counts on an ex-split date.
        #     The series has no discontinuity there to compensate for, so doing
        #     it multiplies the position by the split ratio out of nothing.
        #
        # `close_adj` additionally reinvests dividends, which makes it the
        # total-return series and the right one for measuring returns.
        "close_raw": pl.Float64,
        "close_adj": pl.Float64,
        "volume": pl.Float64,
        "dollar_volume": pl.Float64,  # from close_raw, never the adjusted close
        "split_ratio": pl.Float64,  # split occurring on this date, else 1.0
        # Per share, on today's share basis, matching the close series.
        "dividend": pl.Float64,
        # Product of every split strictly after this date. It bridges between the
        # two bases in both directions: multiply a share count reported on this
        # date's basis by it to restate onto today's, and multiply `close_raw` by
        # it to recover the price the stock actually traded at on the day.
        "split_factor": pl.Float64,
        "source": pl.Utf8,
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("ticker", "date"),
    required=("ticker", "date", "close_raw", "close_adj"),
    entity_col="ticker",
    date_col="date",
    description=(
        "Daily bars on today's share basis: `close_raw` excludes dividends, `close_adj` "
        "includes them, and `split_factor` converts either back to the as-traded price."
    ),
)

# ---------------------------------------------------------------------------
# Financial observations — the mission's core data-integrity table
# ---------------------------------------------------------------------------
# One row per (company, tag, period, filing). Nothing is aggregated, nothing is
# discarded: restatements sit alongside the original, tagged with a revision
# number, so "what did we know on date X" is always answerable.

OBSERVATIONS = TableSpec(
    name="observations",
    layer="curated",
    schema={
        "company_id": pl.Utf8,  # stable company identifier
        "cik": pl.Int64,
        "ticker": pl.Utf8,  # convenience only; never a join key
        "taxonomy": pl.Utf8,  # us-gaap | dei | ifrs-full | srt
        "tag": pl.Utf8,  # the exact source tag, e.g. NetIncomeLoss
        "concept": pl.Utf8,  # our canonical name, e.g. net_income
        "value": pl.Float64,  # raw value, exactly as filed
        "unit": pl.Utf8,  # USD | shares | USD/shares | pure
        "period_start": pl.Date,  # null for instants
        "period_end": pl.Date,  # the period the result describes
        "period_type": pl.Utf8,  # duration | instant
        "duration_days": pl.Int32,
        "fiscal_year": pl.Int32,
        "fiscal_period": pl.Utf8,  # FY | Q1 | Q2 | Q3 | Q4
        "form": pl.Utf8,  # 10-K | 10-Q | 8-K | 20-F ...
        "accession": pl.Utf8,  # the filing this value came from
        "filed": pl.Date,  # when the filing was submitted
        # When the application could practically have acted on it. Distinct from
        # `filed` because a filing submitted after the close is not actionable
        # until the next session; the ingestion layer sets it explicitly.
        "available_at": pl.Date,
        "revision": pl.Int32,  # 0 = first disclosure of this period, 1+ = restatement
        "is_restatement": pl.Boolean,
        "source": pl.Utf8,
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "tag", "period_end", "period_start", "accession"),
    required=(
        "company_id",
        "cik",
        "tag",
        "concept",
        "value",
        "unit",
        "period_end",
        "period_type",
        "filed",
        "available_at",
        "source",
    ),
    date_col="filed",
    description=(
        "Raw financial observations with full provenance. Carries the period the result "
        "describes, the date it was reported, and the date we could have known it -- three "
        "different dates that must never be collapsed into one."
    ),
)

# Derived from OBSERVATIONS: canonical trailing-twelve-month flows and
# balance-sheet instants, each still stamped with when it became knowable.
FUNDAMENTALS = TableSpec(
    name="fundamentals",
    layer="curated",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "concept": pl.Utf8,
        "basis": pl.Utf8,  # ttm | instant | quarter | annual
        "value": pl.Float64,
        "period_end": pl.Date,
        "available_at": pl.Date,
        "fiscal_year": pl.Int32,
        "fiscal_period": pl.Utf8,
        # How many distinct filings the value was assembled from, and the latest
        # of their filing dates. A TTM figure is only knowable once its last
        # constituent quarter was filed.
        "n_sources": pl.Int32,
        "source_accessions": pl.Utf8,
        "is_derived": pl.Boolean,  # true when inferred (e.g. Q4 = FY - Q1..Q3)
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "concept", "basis", "available_at"),
    required=("company_id", "concept", "basis", "value", "period_end", "available_at"),
    date_col="available_at",
    description="Point-in-time canonical financials: TTM flows and balance-sheet instants.",
)

ESTIMATES = TableSpec(
    name="estimates",
    layer="curated",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "as_of": pl.Date,  # snapshot date; estimates have no vintage history from free sources
        "metric": pl.Utf8,  # eps | revenue
        "period": pl.Utf8,  # 0q | +1q | 0y | +1y
        "period_end_estimate": pl.Date,
        "consensus": pl.Float64,
        "low": pl.Float64,
        "high": pl.Float64,
        "n_analysts": pl.Int32,
        "year_ago": pl.Float64,
        "growth": pl.Float64,
        "up_7d": pl.Int32,
        "up_30d": pl.Int32,
        "down_7d": pl.Int32,
        "down_30d": pl.Int32,
        "source": pl.Utf8,
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "as_of", "metric", "period"),
    required=("company_id", "as_of", "metric", "period"),
    date_col="as_of",
    description=(
        "Analyst consensus and revision counts. Free sources publish only the current "
        "snapshot, so history accumulates forward from first collection and the backtester "
        "must treat estimates as unavailable before that date rather than back-filling them."
    ),
)

EARNINGS_EVENTS = TableSpec(
    name="earnings_events",
    layer="curated",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "event_date": pl.Date,
        "is_future": pl.Boolean,
        "eps_estimate": pl.Float64,
        "eps_actual": pl.Float64,
        # A fraction, not percentage points: 0.05 is a 5% beat. Every ratio in
        # this schema is a fraction, and the one exception cost a metric that
        # displayed a 5% average surprise as 512%.
        "surprise": pl.Float64,
        "source": pl.Utf8,
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "event_date"),
    required=("company_id", "event_date"),
    date_col="event_date",
    description="Reported and scheduled earnings dates, for surprise history and event risk.",
)

# ---------------------------------------------------------------------------
# Derived layer
# ---------------------------------------------------------------------------

METRIC_SCORES = TableSpec(
    name="metric_scores",
    layer="derived",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "date": pl.Date,
        "metric": pl.Utf8,
        "category": pl.Utf8,
        "raw_value": pl.Float64,  # kept alongside the score, per the mission
        "winsorized_value": pl.Float64,
        "percentile": pl.Float64,  # 0-1 within the peer group, direction-adjusted
        "score": pl.Float64,  # 0-100
        "peer_basis": pl.Utf8,
        "peer_group": pl.Utf8,
        "peer_count": pl.Int32,
        "peer_median": pl.Float64,
        "is_missing": pl.Boolean,  # true means absent, never "average"
        # Why a metric did not score. The mission requires missing data to be
        # distinguishable from genuinely poor performance, and "this business
        # has no gross margin" is a third thing again.
        "status": pl.Utf8,  # scored | missing | not_applicable | thin_peers
        "weight": pl.Float64,
        "spec_version": pl.Utf8,
        "computed_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "date", "metric"),
    required=("company_id", "date", "metric", "category"),
    date_col="date",
    description="Per-metric normalized scores with the raw value and peer context retained.",
)

CATEGORY_SCORES = TableSpec(
    name="category_scores",
    layer="derived",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "date": pl.Date,
        "category": pl.Utf8,
        "score": pl.Float64,  # 0-100
        "coverage": pl.Float64,  # share of category weight actually observed
        "weight": pl.Float64,  # category weight from the spec
        "contribution": pl.Float64,  # points contributed to the composite
        "n_metrics": pl.Int32,
        "n_missing": pl.Int32,
        "spec_version": pl.Utf8,
        "computed_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "date", "category"),
    required=("company_id", "date", "category"),
    date_col="date",
)

OPPORTUNITY_SCORES = TableSpec(
    name="opportunity_scores",
    layer="derived",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "date": pl.Date,
        "score": pl.Float64,  # 0-100 composite
        "rank": pl.Int32,
        "percentile": pl.Float64,
        "band": pl.Utf8,
        "quality": pl.Float64,
        "growth": pl.Float64,
        "momentum": pl.Float64,
        "valuation": pl.Float64,
        "coverage": pl.Float64,
        "confidence": pl.Float64,
        "confidence_label": pl.Utf8,
        "score_change_1w": pl.Float64,
        "score_change_1m": pl.Float64,
        "universe_size": pl.Int32,
        "spec_version": pl.Utf8,
        "computed_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "date"),
    required=("company_id", "date"),
    date_col="date",
)

VALUATIONS = TableSpec(
    name="valuations",
    layer="derived",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "date": pl.Date,
        "price": pl.Float64,
        "bear": pl.Float64,
        "base": pl.Float64,
        "bull": pl.Float64,
        "range_low": pl.Float64,
        "range_high": pl.Float64,
        # The headline estimate: the weight-blended base case.
        "midpoint": pl.Float64,
        # The arithmetic centre of the bear-bull range, which is a different and
        # systematically higher number. Reported, never decided on.
        "range_midpoint": pl.Float64,
        "upside": pl.Float64,  # to the blended estimate, as a fraction
        "classification": pl.Utf8,
        "confidence": pl.Float64,
        "confidence_label": pl.Utf8,
        "business_model": pl.Utf8,
        "methods_used": pl.Utf8,  # json: {method: {weight, value, ...}}
        "method_dispersion": pl.Float64,  # disagreement between methods
        "assumptions": pl.Utf8,  # json
        "sensitivities": pl.Utf8,  # json
        "caveats": pl.Utf8,  # json list: why this estimate may be wrong
        "spec_version": pl.Utf8,
        "computed_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "date"),
    required=("company_id", "date", "price"),
    date_col="date",
)

RISK = TableSpec(
    name="risk",
    layer="derived",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "date": pl.Date,
        "volatility": pl.Float64,
        "beta": pl.Float64,
        "downside_beta": pl.Float64,
        "downside_deviation": pl.Float64,
        "max_drawdown_1y": pl.Float64,
        "max_drawdown_5y": pl.Float64,
        "dollar_volume_median": pl.Float64,
        "amihud": pl.Float64,
        "spread_bps_est": pl.Float64,
        "days_to_earnings": pl.Int32,
        "earnings_move_avg": pl.Float64,
        "distress_score": pl.Float64,  # Altman Z or the financial-model equivalent
        "distress_label": pl.Utf8,
        "piotroski": pl.Int32,
        "risk_score": pl.Float64,  # 0-100, higher = riskier
        "spec_version": pl.Utf8,
        "computed_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "date"),
    required=("company_id", "date"),
    date_col="date",
)

VALUE_TRAP = TableSpec(
    name="value_trap",
    layer="derived",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "date": pl.Date,
        "trap_score": pl.Float64,  # 0-100, higher = more trap-like
        "classification": pl.Utf8,
        "signals": pl.Utf8,  # json: which signals fired, with values
        "n_signals": pl.Int32,
        "spec_version": pl.Utf8,
        "computed_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "date"),
    required=("company_id", "date"),
    date_col="date",
)

RECOMMENDATIONS = TableSpec(
    name="recommendations",
    layer="derived",
    schema={
        "recommendation_id": pl.Utf8,
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "date": pl.Date,
        "action": pl.Utf8,
        "target_weight": pl.Float64,
        "target_value": pl.Float64,
        "trade_value": pl.Float64,
        "trade_shares": pl.Float64,
        "price": pl.Float64,
        "score": pl.Float64,
        "confidence": pl.Float64,
        "valuation_class": pl.Utf8,
        "upside": pl.Float64,
        "horizon_months": pl.Int32,
        "reasons": pl.Utf8,  # json list
        "risks": pl.Utf8,  # json list
        "invalidation": pl.Utf8,  # json list
        "catalysts": pl.Utf8,  # json list
        "portfolio_impact": pl.Utf8,  # json
        "estimated_cost": pl.Float64,
        "estimated_tax": pl.Float64,
        "spec_versions": pl.Utf8,  # json: scoring/valuation/portfolio versions
        "computed_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("recommendation_id",),
    required=("recommendation_id", "company_id", "date", "action"),
    date_col="date",
)

# ---------------------------------------------------------------------------
# Portfolio state (user-entered, not derived)
# ---------------------------------------------------------------------------

POSITIONS = TableSpec(
    name="positions",
    layer="curated",
    schema={
        "account_id": pl.Utf8,
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "shares": pl.Float64,
        "cost_basis_per_share": pl.Float64,
        "opened_at": pl.Date,
        "last_trade_at": pl.Date,
        "updated_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("account_id", "company_id"),
    required=("account_id", "company_id", "shares"),
)

ACCOUNTS = TableSpec(
    name="accounts",
    layer="curated",
    schema={
        "account_id": pl.Utf8,
        "name": pl.Utf8,
        "account_type": pl.Utf8,
        "cash": pl.Float64,
        "risk_tolerance": pl.Utf8,
        "max_positions": pl.Int32,
        "max_position_weight": pl.Float64,
        "max_sector_weight": pl.Float64,
        "min_cash_weight": pl.Float64,
        "realized_gains_ytd": pl.Float64,
        "updated_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("account_id",),
    required=("account_id", "account_type"),
    entity_col="account_id",
)

# ---------------------------------------------------------------------------
# Research journal — append-only, never rewritten
# ---------------------------------------------------------------------------

JOURNAL = TableSpec(
    name="journal",
    layer="artifacts",
    schema={
        "entry_id": pl.Utf8,
        "created_at": pl.Datetime("us", "UTC"),
        "as_of_date": pl.Date,
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "action": pl.Utf8,
        "price": pl.Float64,
        "score": pl.Float64,
        "confidence": pl.Float64,
        "fair_value_low": pl.Float64,
        "fair_value_mid": pl.Float64,
        "fair_value_high": pl.Float64,
        "valuation_class": pl.Utf8,
        "horizon_months": pl.Int32,
        "target_weight": pl.Float64,
        "reasons": pl.Utf8,
        "risks": pl.Utf8,
        "invalidation": pl.Utf8,
        # A complete snapshot of the inputs available at the moment of the call,
        # so the recommendation can be re-derived later without touching data
        # that did not exist yet.
        "snapshot": pl.Utf8,  # json
        "scoring_version": pl.Utf8,
        "valuation_version": pl.Utf8,
        "portfolio_version": pl.Utf8,
        "mode": pl.Utf8,  # research | backtest | paper | live
    },
    primary_key=("entry_id",),
    required=("entry_id", "created_at", "as_of_date", "company_id", "action", "mode"),
    date_col="as_of_date",
    description=(
        "Append-only record of every recommendation and the evidence behind it. Rows are "
        "never updated: outcomes are written to `journal_outcomes` so the original call "
        "cannot be retroactively improved."
    ),
)

JOURNAL_OUTCOMES = TableSpec(
    name="journal_outcomes",
    layer="artifacts",
    schema={
        "entry_id": pl.Utf8,
        "evaluated_at": pl.Datetime("us", "UTC"),
        "horizon_days": pl.Int32,
        "price_then": pl.Float64,
        "price_now": pl.Float64,
        "total_return": pl.Float64,
        "benchmark_return": pl.Float64,
        "excess_return": pl.Float64,
        "max_drawdown_since": pl.Float64,
        "invalidated": pl.Boolean,
        "invalidation_reason": pl.Utf8,
        "status": pl.Utf8,  # open | closed | invalidated
    },
    primary_key=("entry_id", "horizon_days"),
    required=("entry_id", "evaluated_at", "horizon_days"),
    entity_col=None,
)


# A current-moment snapshot from a market-data provider. It exists because some
# facts are simply not in the filings: dual-class issuers report share counts
# per class, and the SEC's aggregate interface carries only undimensioned facts,
# so Visa and Berkshire have no consolidated share count there at all. It also
# carries the forward-looking fields (consensus forward earnings, analyst
# targets) and the microstructure fields (bid/ask) that filings never contain.
#
# There is no history here and there never will be: this is what the provider
# says today. The backtester must not read it.
SECURITY_SNAPSHOT = TableSpec(
    name="security_snapshot",
    layer="curated",
    schema={
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "as_of": pl.Date,
        "price": pl.Float64,
        "shares_outstanding": pl.Float64,
        "market_cap": pl.Float64,
        "enterprise_value": pl.Float64,
        "forward_eps": pl.Float64,
        "forward_pe": pl.Float64,
        "trailing_eps": pl.Float64,
        "beta": pl.Float64,
        "dividend_yield": pl.Float64,
        "target_mean_price": pl.Float64,
        "n_analysts": pl.Int32,
        "bid": pl.Float64,
        "ask": pl.Float64,
        "spread_bps": pl.Float64,
        "float_shares": pl.Float64,
        "source": pl.Utf8,
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("company_id", "as_of"),
    required=("company_id", "ticker", "as_of"),
    date_col="as_of",
    description=(
        "Current provider snapshot: share count, forward consensus and quote data that the "
        "filings do not contain. Point-in-time only for today; never read by the backtester."
    ),
)


# ---------------------------------------------------------------------------
# Validation and backtesting
# ---------------------------------------------------------------------------
# A backtest result is only worth as much as its reproducibility. Every run
# therefore records four coordinates -- which strategy, which data, which
# universe, which code -- and every one of them is a checksum or a commit, not
# a description. A number that cannot be re-derived is an anecdote.
#
# Failed and disappointing runs are stored exactly like successful ones. That
# is the point: the count of trials is what turns a good Sharpe ratio into a
# selection effect, and deleting the failures is how a research process lies to
# itself without anyone intending to.

STRATEGY_VERSIONS = TableSpec(
    name="strategy_versions",
    layer="artifacts",
    schema={
        "strategy_version": pl.Utf8,
        "name": pl.Utf8,
        "frozen_at": pl.Datetime("us", "UTC"),
        "scoring_version": pl.Utf8,
        # The complete configuration this version pins: scoring spec, metric
        # weights, eligibility rules, portfolio construction, execution and
        # costs. Serialised canonically so the checksum is stable across runs.
        "config": pl.Utf8,  # json
        "config_checksum": pl.Utf8,
        "code_commit": pl.Utf8,
        "status": pl.Utf8,  # frozen | draft | superseded
        "notes": pl.Utf8,
    },
    primary_key=("strategy_version",),
    required=("strategy_version", "frozen_at", "config", "config_checksum", "status"),
    entity_col=None,
    description=(
        "A frozen strategy definition and its checksum. Freezing is what makes the first "
        "test an honest baseline rather than the best of several attempts."
    ),
)

BACKTEST_RUNS = TableSpec(
    name="backtest_runs",
    layer="artifacts",
    schema={
        "run_id": pl.Utf8,
        "created_at": pl.Datetime("us", "UTC"),
        # Why this run exists, written before it ran. A trial whose purpose is
        # recorded afterwards is a trial whose purpose is its result.
        "purpose": pl.Utf8,
        "trial_type": pl.Utf8,  # primary | variant | robustness | ablation | factor | holdout
        "variant_label": pl.Utf8,
        "strategy_version": pl.Utf8,
        "config_checksum": pl.Utf8,
        "code_commit": pl.Utf8,
        "data_version": pl.Utf8,
        "universe_version": pl.Utf8,
        "period_label": pl.Utf8,  # development | validation | holdout | full | paper
        "period_start": pl.Date,
        "period_end": pl.Date,
        "params": pl.Utf8,  # json
        "costs": pl.Utf8,  # json
        # The honest label for what this result can support, from
        # spaid.config.validation.ValidationStatus. Never "validated" while the
        # universe or the execution assumptions are materially incomplete.
        "validity": pl.Utf8,
        "status": pl.Utf8,  # completed | failed
        "n_rebalances": pl.Int32,
        "results": pl.Utf8,  # json: the headline metrics
        "error": pl.Utf8,
        "elapsed_seconds": pl.Float64,
    },
    primary_key=("run_id",),
    required=("run_id", "created_at", "purpose", "strategy_version", "status"),
    entity_col=None,
    date_col="period_start",
    description="One row per backtest execution, with every coordinate needed to reproduce it.",
)

BACKTEST_POSITIONS = TableSpec(
    name="backtest_positions",
    layer="artifacts",
    schema={
        "run_id": pl.Utf8,
        "rebalance_date": pl.Date,
        "security_id": pl.Utf8,
        "company_id": pl.Utf8,
        "ticker": pl.Utf8,
        "sector": pl.Utf8,
        "score": pl.Float64,
        "rank": pl.Int32,
        "target_weight": pl.Float64,
        "held_weight": pl.Float64,  # after execution, including the delay
        "shares": pl.Float64,
        "price": pl.Float64,
        "value": pl.Float64,
    },
    primary_key=("run_id", "rebalance_date", "security_id"),
    required=("run_id", "rebalance_date", "security_id", "ticker"),
    entity_col="security_id",
    date_col="rebalance_date",
)

BACKTEST_TRADES = TableSpec(
    name="backtest_trades",
    layer="artifacts",
    schema={
        "run_id": pl.Utf8,
        "trade_id": pl.Utf8,
        # The date the decision was made and the date it was executed are
        # separate columns because the gap between them is a modelled
        # assumption, not an implementation detail.
        "decision_date": pl.Date,
        "trade_date": pl.Date,
        "security_id": pl.Utf8,
        "ticker": pl.Utf8,
        "side": pl.Utf8,  # buy | sell
        "shares": pl.Float64,
        "price": pl.Float64,
        "gross_value": pl.Float64,
        "spread_cost": pl.Float64,
        "slippage_cost": pl.Float64,
        "commission": pl.Float64,
        "total_cost": pl.Float64,
        # rebalance | entry | exit | delisting | index_exit | liquidation
        "reason": pl.Utf8,
    },
    primary_key=("trade_id",),
    required=("run_id", "trade_id", "trade_date", "security_id", "side"),
    entity_col="security_id",
    date_col="trade_date",
)

BACKTEST_DAILY = TableSpec(
    name="backtest_daily",
    layer="artifacts",
    schema={
        "run_id": pl.Utf8,
        "date": pl.Date,
        # Gross and net are carried separately all the way through, because the
        # difference between them is the whole question of whether a strategy
        # survives its own turnover.
        "gross_equity": pl.Float64,
        "net_equity": pl.Float64,
        "cash": pl.Float64,
        "invested": pl.Float64,
        "n_positions": pl.Int32,
        "gross_return": pl.Float64,
        "net_return": pl.Float64,
        "costs_paid": pl.Float64,
        "drawdown": pl.Float64,
        "is_rebalance": pl.Boolean,
        # Traded notional on both sides as a fraction of equity, so a full
        # replacement reads 2.0. This is the quantity costs are charged on.
        "turnover": pl.Float64,
    },
    primary_key=("run_id", "date"),
    required=("run_id", "date", "net_equity"),
    entity_col="run_id",
    date_col="date",
)

BENCHMARK_VALUES = TableSpec(
    name="benchmark_values",
    layer="artifacts",
    schema={
        "benchmark": pl.Utf8,
        "date": pl.Date,
        "close_adj": pl.Float64,
        "total_return_index": pl.Float64,  # rebased to 1.0 at the series start
        "daily_return": pl.Float64,
        "source": pl.Utf8,
        "collected_at": pl.Datetime("us", "UTC"),
    },
    primary_key=("benchmark", "date"),
    required=("benchmark", "date", "close_adj"),
    entity_col="benchmark",
    date_col="date",
    description=(
        "Benchmark total-return series, stored rather than recomputed so that a historical "
        "comparison does not move when the price table is refreshed."
    ),
)

PERFORMANCE_METRICS = TableSpec(
    name="performance_metrics",
    layer="artifacts",
    schema={
        "run_id": pl.Utf8,
        "scope": pl.Utf8,  # gross | net
        "benchmark": pl.Utf8,  # null for absolute metrics
        "period_label": pl.Utf8,  # full | development | validation | holdout | a year
        "metric": pl.Utf8,
        "value": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "n_obs": pl.Int32,
        "detail": pl.Utf8,  # json, for metrics that carry more than one number
    },
    primary_key=("run_id", "scope", "benchmark", "period_label", "metric"),
    required=("run_id", "scope", "period_label", "metric"),
    entity_col="run_id",
)

SIGNAL_DIAGNOSTICS = TableSpec(
    name="signal_diagnostics",
    layer="artifacts",
    schema={
        "run_id": pl.Utf8,
        # ic | ic_series | decile | quintile | band | sector | size | year |
        # regime | category | ablation | turnover | holding_period
        "kind": pl.Utf8,
        "horizon_months": pl.Int32,
        "bucket": pl.Utf8,  # the decile, sector, year or regime this row describes
        "metric": pl.Utf8,
        "value": pl.Float64,
        "n_obs": pl.Int32,
        "date": pl.Date,  # only for time series
        "detail": pl.Utf8,  # json
    },
    primary_key=("run_id", "kind", "horizon_months", "bucket", "metric", "date"),
    required=("run_id", "kind", "metric"),
    entity_col="run_id",
    date_col="date",
    description=(
        "Whether the ranking itself carries information, separately from whether one "
        "portfolio built on it made money. A portfolio can succeed on two lucky holdings; "
        "an information coefficient cannot."
    ),
)

ROBUSTNESS_TESTS = TableSpec(
    name="robustness_tests",
    layer="artifacts",
    schema={
        "run_id": pl.Utf8,  # the robustness run
        "base_run_id": pl.Utf8,  # what it is compared against
        "test_name": pl.Utf8,
        "parameter": pl.Utf8,
        "parameter_value": pl.Utf8,
        "metric": pl.Utf8,
        "base_value": pl.Float64,
        "value": pl.Float64,
        "delta": pl.Float64,
        # survived | weakened | failed -- against a threshold declared before
        # the test ran, not chosen after seeing it.
        "verdict": pl.Utf8,
    },
    primary_key=("run_id", "test_name", "parameter_value", "metric"),
    required=("run_id", "base_run_id", "test_name", "metric"),
    entity_col="run_id",
)

EXPERIMENTS = TableSpec(
    name="experiments",
    layer="artifacts",
    schema={
        "trial_id": pl.Utf8,
        "created_at": pl.Datetime("us", "UTC"),
        "purpose": pl.Utf8,
        "hypothesis": pl.Utf8,
        "strategy_version": pl.Utf8,
        "config_checksum": pl.Utf8,
        "code_commit": pl.Utf8,
        "data_version": pl.Utf8,
        "universe_version": pl.Utf8,
        "period_label": pl.Utf8,
        "period_start": pl.Date,
        "period_end": pl.Date,
        "params": pl.Utf8,  # json
        "costs": pl.Utf8,  # json
        "run_id": pl.Utf8,
        "outcome": pl.Utf8,  # completed | failed | inconclusive
        "results": pl.Utf8,  # json
        "notes": pl.Utf8,
    },
    primary_key=("trial_id",),
    required=("trial_id", "created_at", "purpose", "strategy_version", "outcome"),
    entity_col=None,
    date_col="period_start",
    description=(
        "Append-only registry of every trial, including the ones that failed. The number of "
        "trials is an input to the deflated Sharpe ratio, so discarding the failures would "
        "inflate the significance of whatever survived."
    ),
)

HOLDOUT_AUDIT = TableSpec(
    name="holdout_audit",
    layer="artifacts",
    schema={
        "audit_id": pl.Utf8,
        "requested_at": pl.Datetime("us", "UTC"),
        "reason": pl.Utf8,
        "strategy_version": pl.Utf8,
        "config_checksum": pl.Utf8,
        "holdout_start": pl.Date,
        "holdout_end": pl.Date,
        "run_id": pl.Utf8,
        "results": pl.Utf8,  # json
    },
    primary_key=("audit_id",),
    required=("audit_id", "requested_at", "reason", "strategy_version"),
    entity_col=None,
    description=(
        "Every evaluation of the final holdout, recorded permanently. A holdout that has "
        "been looked at twice is no longer a holdout, and this table is how that stays "
        "visible rather than being forgotten."
    ),
)


ALL_TABLES: dict[str, TableSpec] = {
    t.name: t
    for t in (
        COMPANIES,
        SECURITIES,
        SECURITY_MASTER,
        TICKER_HISTORY,
        CORPORATE_ACTIONS,
        UNIVERSE_MEMBERSHIP,
        UNIVERSE_VERSIONS,
        PRICES,
        OBSERVATIONS,
        FUNDAMENTALS,
        ESTIMATES,
        EARNINGS_EVENTS,
        SECURITY_SNAPSHOT,
        METRIC_SCORES,
        CATEGORY_SCORES,
        OPPORTUNITY_SCORES,
        VALUATIONS,
        RISK,
        VALUE_TRAP,
        RECOMMENDATIONS,
        POSITIONS,
        ACCOUNTS,
        JOURNAL,
        JOURNAL_OUTCOMES,
        STRATEGY_VERSIONS,
        BACKTEST_RUNS,
        BACKTEST_POSITIONS,
        BACKTEST_TRADES,
        BACKTEST_DAILY,
        BENCHMARK_VALUES,
        PERFORMANCE_METRICS,
        SIGNAL_DIAGNOSTICS,
        ROBUSTNESS_TESTS,
        EXPERIMENTS,
        HOLDOUT_AUDIT,
    )
}


class SchemaError(ValueError):
    """Raised when a frame does not match its declared schema."""


def validate(df: pl.DataFrame, spec: TableSpec, *, strict: bool = True) -> pl.DataFrame:
    """Check `df` against `spec` and return it with columns in declared order.

    `strict` also enforces the primary key and the non-null requirements. It is
    on by default because a duplicated primary key in a financial table means
    some company is about to be counted twice.
    """
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{spec.name}: missing columns {missing}")

    extra = [c for c in df.columns if c not in spec.schema]
    if extra:
        raise SchemaError(f"{spec.name}: unexpected columns {extra}")

    wrong: list[str] = []
    for col, dtype in spec.schema.items():
        actual = df.schema[col]
        if actual != dtype:
            wrong.append(f"{col}: expected {dtype}, got {actual}")
    if wrong:
        raise SchemaError(f"{spec.name}: dtype mismatch -- " + "; ".join(wrong))

    out = df.select(list(spec.columns))

    if not strict or out.is_empty():
        return out

    for col in spec.required:
        n_null = out[col].null_count()
        if n_null:
            raise SchemaError(f"{spec.name}: column {col!r} has {n_null} nulls but is required")

    pk = list(spec.primary_key)
    n_dupes = out.height - out.select(pk).unique().height
    if n_dupes:
        sample = (
            out.group_by(pk)
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
            .sort("n", descending=True)
            .head(3)
            .to_dicts()
        )
        raise SchemaError(
            f"{spec.name}: {n_dupes} rows violate primary key {pk}; examples {sample}"
        )
    return out


def coerce(df: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    """Add missing columns as nulls and cast to the declared dtypes.

    Used at the boundary where a provider hands us a partial frame. It never
    invents values -- a column the provider did not supply becomes null, which is
    honest, rather than zero, which is not.
    """
    exprs = []
    for col, dtype in spec.schema.items():
        if col in df.columns:
            exprs.append(pl.col(col).cast(dtype, strict=False).alias(col))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(col))
    return df.with_columns(exprs).select(list(spec.columns))
