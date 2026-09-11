"""Strategy Version 1: the ranking as it exists today, frozen and checksummed.

The question this milestone answers is whether the *existing* score predicts
anything. That only means something if the score cannot move while the question
is being answered, so this module takes the live scoring specification, pins it
together with the portfolio rules, execution assumptions and cost model, hashes
the whole thing, and stores it. Every backtest records that checksum. If the
scoring weights are edited, the checksum changes and old results stop claiming
to describe the current strategy.

One honest complication is recorded in the frozen definition rather than hidden
in a code path. Seven of the forty-three metrics are derived from analyst
estimates, and the estimate source publishes only today's snapshot -- there is
no history to join to a 2017 decision date. Back-filling today's values would
hand the backtest nine years of foresight, so at historical dates those metrics
are *absent*. The scoring spec already knows what to do with an absent metric:
drop it and renormalise the surviving weights. That is not a special case
invented for the backtest, it is the same rule that handles a bank with no
gross margin.

The consequence is stated plainly here and carried into every report: what is
tested is Strategy Version 1 running on the evidence that genuinely existed at
the time, which is a slightly different object from Strategy Version 1 running
today. Both are named, and the difference is measured rather than assumed away.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import polars as pl

from spaid.backtest import provenance
from spaid.config.scoring import ACTIVE_SCORING_SPEC, Category, ScoringSpec
from spaid.config.validation import (
    ACTIVE_VALIDATION_SPEC,
    CostModel,
    ExecutionSpec,
    ValidationSpec,
)
from spaid.storage import store
from spaid.storage.schema import STRATEGY_VERSIONS, coerce

log = logging.getLogger(__name__)


# Metrics that cannot be computed at a historical date, and why. Listed
# explicitly rather than discovered at runtime, so that the frozen strategy
# states its own limitation and the checksum covers it.
POINT_IN_TIME_UNAVAILABLE: dict[str, str] = {
    "eps_revision_3m": "analyst estimate revisions have no vintage history",
    "revenue_revision_3m": "analyst estimate revisions have no vintage history",
    "forward_eps_growth": "consensus forward earnings are a current snapshot only",
    "forward_pe": "consensus forward earnings are a current snapshot only",
    "peg": "derived from forward earnings, which are a current snapshot only",
    "earnings_surprise": "the surprise history is fetched as a snapshot, not by vintage",
    "post_earnings_drift": "requires the reported-earnings calendar, fetched as a snapshot",
}


@dataclass(frozen=True)
class EligibilitySpec:
    """Which securities may be held on a decision date.

    Every rule is evaluated against what was knowable then. `min_price` is
    compared against the traded price, never the split-adjusted one, for the
    same reason it is elsewhere in this codebase: judging a 2015 price on
    today's share basis lets a later split decide past eligibility.
    """

    require_index_member: bool = True
    require_priced: bool = True
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0
    # A company needs enough price history for momentum to mean anything.
    min_price_history_days: int = 252
    # And enough evidence for the composite to be emitted at all; below the
    # spec's own coverage floor it is unscored, and unscored is not ranked.
    require_score: bool = True


@dataclass(frozen=True)
class PortfolioRules:
    """One predefined portfolio construction. All variants are reported."""

    label: str
    description: str
    top_n: int | None = None
    top_fraction: float | None = None  # 0.10 for the top decile
    weighting: str = "equal"  # equal | score_proportional
    max_sector_weight: float | None = None
    max_position_weight: float | None = None

    def size_for(self, n_eligible: int) -> int:
        if self.top_n is not None:
            return min(self.top_n, n_eligible)
        if self.top_fraction is not None:
            return max(1, round(n_eligible * self.top_fraction))
        raise ValueError(f"{self.label}: neither top_n nor top_fraction is set")


VARIANTS: dict[str, PortfolioRules] = {
    "top10": PortfolioRules(
        label="top10",
        description="The ten highest-scoring eligible companies, equally weighted.",
        top_n=10,
    ),
    "top20": PortfolioRules(
        label="top20",
        description="The twenty highest-scoring eligible companies, equally weighted.",
        top_n=20,
    ),
    "top_decile": PortfolioRules(
        label="top_decile",
        description="The highest-scoring tenth of the eligible universe, equally weighted.",
        top_fraction=0.10,
    ),
}


@dataclass(frozen=True)
class StrategyVersion:
    """A complete, frozen strategy: what is ranked, what is held, what it costs."""

    version: str
    name: str
    scoring: ScoringSpec
    eligibility: EligibilitySpec
    variants: tuple[str, ...]
    primary_variant: str
    execution: ExecutionSpec
    costs: CostModel
    rebalance: str
    # Benchmarks the result is judged against. SPY is the market; SPMO is the
    # one that matters more, because a momentum-tilted score that cannot beat a
    # momentum index is describing a factor rather than adding to it.
    benchmarks: tuple[str, ...]
    unavailable_point_in_time: dict[str, str] = field(
        default_factory=lambda: dict(POINT_IN_TIME_UNAVAILABLE)
    )
    notes: str = ""

    # -- configuration ---------------------------------------------------
    def config(self) -> dict:
        """The canonical configuration this version pins.

        Everything that could change the result is in here, and nothing that
        cannot. The checksum is taken over this, so a cosmetic edit to a
        docstring does not invalidate a year of stored runs while a changed
        metric weight does.
        """
        return {
            "version": self.version,
            "name": self.name,
            "scoring": {
                "version": self.scoring.version,
                "category_weights": {
                    c.value: w for c, w in sorted(
                        self.scoring.category_weights.items(), key=lambda kv: kv[0].value
                    )
                },
                "min_category_coverage": self.scoring.min_category_coverage,
                "min_total_coverage": self.scoring.min_total_coverage,
                "min_universe_size": self.scoring.min_universe_size,
                "metrics": [
                    {
                        "key": m.key,
                        "category": m.category.value,
                        "weight": m.weight,
                        "direction": m.direction,
                        "peer_basis": m.peer_basis.value,
                        "applies_to": sorted(b.value for b in m.applies_to),
                        "winsor": m.winsor,
                        "min_peers": m.min_peers,
                    }
                    for m in sorted(self.scoring.metrics, key=lambda m: m.key)
                ],
            },
            "eligibility": asdict(self.eligibility),
            "variants": list(self.variants),
            "primary_variant": self.primary_variant,
            "execution": asdict(self.execution),
            "costs": self.costs.to_dict(),
            "rebalance": self.rebalance,
            "benchmarks": list(self.benchmarks),
            "unavailable_point_in_time": dict(sorted(self.unavailable_point_in_time.items())),
        }

    @property
    def checksum(self) -> str:
        return provenance.checksum(self.config())

    def rules(self, variant: str) -> PortfolioRules:
        if variant not in self.variants:
            raise KeyError(
                f"{variant!r} is not a declared variant of {self.version}; "
                f"declared: {list(self.variants)}"
            )
        return VARIANTS[variant]

    # -- what the missing metrics cost -----------------------------------
    def available_metrics(self) -> tuple[str, ...]:
        """Metric keys that can be computed at a historical decision date."""
        return tuple(
            m.key for m in self.scoring.metrics
            if m.key not in self.unavailable_point_in_time
        )

    def category_coverage_point_in_time(self) -> dict[str, float]:
        """The share of each category's weight that survives point-in-time.

        Reported because it is the size of the compromise. Growth loses the most
        -- estimate revisions and forward growth are a third of it -- and a
        reader deserves to know that before comparing the backtested score with
        the one on the dashboard.
        """
        out: dict[str, float] = {}
        for category in Category:
            metrics = self.scoring.in_category(category)
            total = sum(m.weight for m in metrics)
            kept = sum(
                m.weight for m in metrics if m.key not in self.unavailable_point_in_time
            )
            out[category.value] = round(kept / total, 4) if total else 0.0
        return out


def _build_v1(validation: ValidationSpec | None = None) -> StrategyVersion:
    validation = validation or ACTIVE_VALIDATION_SPEC
    return StrategyVersion(
        version="strategy-v1",
        name="Strategy Version 1 — the composite opportunity score, unchanged",
        scoring=ACTIVE_SCORING_SPEC,
        eligibility=EligibilitySpec(),
        variants=validation.portfolio_variants,
        primary_variant=validation.primary_variant,
        execution=validation.execution,
        costs=validation.costs,
        rebalance=validation.rebalance,
        benchmarks=("SPY", "SPMO"),
        notes=(
            "The scoring specification exactly as it stands, frozen before any backtest was "
            "run and not adjusted afterwards. The portfolio rules, the execution delay, the "
            "cost model and the choice of the top-20 basket as primary were all declared in "
            "spaid.config.validation before the first result existed. Seven estimate-derived "
            "metrics are unavailable at historical dates and are dropped by the spec's own "
            "renormalisation rule rather than back-filled from today's snapshot."
        ),
    )


STRATEGY_V1 = _build_v1()
STRATEGIES: dict[str, StrategyVersion] = {STRATEGY_V1.version: STRATEGY_V1}
ACTIVE_STRATEGY = STRATEGY_V1


def get_strategy(version: str | None = None) -> StrategyVersion:
    if version is None:
        return ACTIVE_STRATEGY
    try:
        return STRATEGIES[version]
    except KeyError:
        raise KeyError(
            f"unknown strategy version {version!r}; known: {sorted(STRATEGIES)}"
        ) from None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def freeze(strategy: StrategyVersion | None = None, *, persist: bool = True) -> dict:
    """Record a strategy version and its checksum, refusing to redefine one.

    A version already on disk with a different checksum means the code changed
    under a name that was supposed to be fixed. That is an error, not an
    update: the whole point of freezing is that `strategy-v1` means one thing
    forever.
    """
    strategy = strategy or ACTIVE_STRATEGY
    existing = store.read("strategy_versions")
    checksum = strategy.checksum

    if existing is not None and not existing.is_empty():
        prior = existing.filter(pl.col("strategy_version") == strategy.version)
        if not prior.is_empty():
            stored = prior["config_checksum"][0]
            if stored != checksum:
                raise ValueError(
                    f"{strategy.version} is already frozen with checksum {stored}, but the "
                    f"current configuration hashes to {checksum}. A frozen version cannot be "
                    "redefined -- publish the change as a new version so that results "
                    "recorded under the old one keep meaning what they said."
                )
            log.info("%s already frozen (%s)", strategy.version, checksum[:12])
            return {
                "strategy_version": strategy.version,
                "config_checksum": checksum,
                "frozen_at": str(prior["frozen_at"][0]),
                "already_frozen": True,
            }

    row = coerce(
        pl.DataFrame(
            [
                {
                    "strategy_version": strategy.version,
                    "name": strategy.name,
                    "frozen_at": datetime.now(UTC),
                    "scoring_version": strategy.scoring.version,
                    "config": json.dumps(strategy.config(), sort_keys=True),
                    "config_checksum": checksum,
                    "code_commit": provenance.code_commit(),
                    "status": "frozen",
                    "notes": strategy.notes,
                }
            ]
        ),
        STRATEGY_VERSIONS,
    )
    if persist:
        store.upsert(row, "strategy_versions", source="spaid")
    log.info(
        "froze %s: checksum %s, scoring spec %s, %d metrics (%d available point-in-time)",
        strategy.version, checksum[:12], strategy.scoring.version,
        len(strategy.scoring.metrics), len(strategy.available_metrics()),
    )
    return {
        "strategy_version": strategy.version,
        "config_checksum": checksum,
        "scoring_version": strategy.scoring.version,
        "code_commit": provenance.code_commit(),
        "metrics_total": len(strategy.scoring.metrics),
        "metrics_point_in_time": len(strategy.available_metrics()),
        "category_coverage_point_in_time": strategy.category_coverage_point_in_time(),
        "already_frozen": False,
    }


def frozen_record(version: str) -> dict | None:
    """The stored definition of a version, for the interface to display."""
    existing = store.read("strategy_versions")
    if existing is None or existing.is_empty():
        return None
    row = existing.filter(pl.col("strategy_version") == version)
    return row.to_dicts()[0] if not row.is_empty() else None
