"""The validation protocol: periods, costs, gates and honest status labels.

This module is written *before* any result exists, and that is the whole point.
Every threshold a strategy has to clear, every period it is tested over, and
every word the interface is allowed to use about it are declared here first, so
that a disappointing result cannot be rescued by moving a boundary afterwards.

Three commitments are encoded rather than intended:

**Chronological splits, never shuffled.** Financial observations are ordered in
time and serially correlated; a random split trains on the future. Development,
validation and holdout are three consecutive calendar periods, separated by a
purge wide enough that a twelve-month forward return measured in one period
cannot reach into the next.

**The holdout is inaccessible by default.** Its boundaries live here, but
evaluating it needs an explicit command and writes a permanent audit record.
A holdout that gets peeked at during iteration is not a holdout, and the only
defence against peeking is making it cost something.

**"Validated" is a word with prerequisites.** The data gates below are hard:
while the historical universe is survivorship-biased or delisting returns are
unavailable, no result -- however good -- may be labelled anything better than
exploratory. The gate is on the *evidence*, not on the number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum


class ValidationStatus(StrEnum):
    """What a result is allowed to claim.

    Ordered from weakest to strongest. The interface shows the label verbatim;
    there is no summarising into "good" or "bad", because the distinction
    between "we could not test this" and "we tested it and it failed" is the
    most important one on the page.
    """

    NOT_TESTABLE = "not_testable"
    EXPLORATORY_ONLY = "exploratory_only"
    IN_SAMPLE = "in_sample"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    HOLDOUT_PASSED = "holdout_passed"
    HOLDOUT_FAILED = "holdout_failed"
    PAPER_TRACKING = "paper_tracking"
    LIVE_TRACKING = "live_tracking"


STATUS_LABELS: dict[ValidationStatus, str] = {
    ValidationStatus.NOT_TESTABLE: "Not testable",
    ValidationStatus.EXPLORATORY_ONLY: "Exploratory only",
    ValidationStatus.IN_SAMPLE: "In-sample",
    ValidationStatus.VALIDATION_PASSED: "Validation passed",
    ValidationStatus.VALIDATION_FAILED: "Validation failed",
    ValidationStatus.HOLDOUT_PASSED: "Holdout passed",
    ValidationStatus.HOLDOUT_FAILED: "Holdout failed",
    ValidationStatus.PAPER_TRACKING: "Paper tracking",
    ValidationStatus.LIVE_TRACKING: "Live tracking",
}


class Conclusion(StrEnum):
    """The verdict on whether the score predicts anything."""

    EVIDENCE_OF_ABILITY = "evidence_of_predictive_ability"
    NO_EVIDENCE = "no_meaningful_evidence"
    MIXED = "evidence_is_mixed"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID = "backtest_invalid"


CONCLUSION_LABELS: dict[Conclusion, str] = {
    Conclusion.EVIDENCE_OF_ABILITY: "Evidence of predictive ranking ability",
    Conclusion.NO_EVIDENCE: "No meaningful evidence of predictive ability",
    Conclusion.MIXED: "Evidence is mixed",
    Conclusion.INSUFFICIENT_DATA: "Insufficient data to determine",
    Conclusion.INVALID: "Backtest is invalid because of data limitations",
}


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Period:
    label: str
    start: date
    end: date
    purpose: str

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    @property
    def years(self) -> float:
        return (self.end - self.start).days / 365.25


@dataclass(frozen=True)
class PeriodSpec:
    """The chronological split, and the separation between its parts.

    The start date is not a preference. It is the latest of four binding data
    constraints, each measured rather than assumed:

    * daily prices begin 2013-01-02, and twelve-month momentum needs a year of
      them before the first decision date;
    * point-in-time fundamentals begin with the 2013 filings, and the
      three-year growth metrics need three years of them;
    * SPMO, the momentum benchmark the mission requires, has no history before
      2015-10-12;
    * the metrics that need five years of financial history do not exist until
      2018, and are carried as missing before then rather than back-filled.

    Extending backwards past this point would lengthen the result without
    improving the evidence, which is the definition of a worse test.
    """

    development: Period
    validation: Period
    holdout: Period
    paper_starts: date | None

    # A forward return measured over twelve months from the last development
    # date reaches a year into the validation period. Purging removes those
    # observations; the embargo adds separation on top, because adjacent
    # observations remain correlated even once the label windows stop touching.
    purge_months: int = 12
    embargo_days: int = 21

    @property
    def full(self) -> Period:
        return Period(
            label="full",
            start=self.development.start,
            end=self.holdout.end,
            purpose="Every date for which the strategy can be evaluated at all.",
        )

    @property
    def iterable(self) -> Period:
        """Development plus validation: everything iteration may touch."""
        return Period(
            label="iterable",
            start=self.development.start,
            end=self.validation.end,
            purpose="The periods available during model iteration.",
        )

    def label_for(self, day: date) -> str:
        for period in (self.development, self.validation, self.holdout):
            if period.contains(day):
                return period.label
        if self.paper_starts is not None and day >= self.paper_starts:
            return "paper"
        return "outside"


PERIODS = PeriodSpec(
    development=Period(
        label="development",
        start=date(2016, 1, 1),
        end=date(2020, 12, 31),
        purpose=(
            "Where the mechanics are built and inspected. Any number read from here has "
            "been seen by the person building the system and is in-sample by definition."
        ),
    ),
    validation=Period(
        label="validation",
        start=date(2021, 1, 1),
        end=date(2024, 12, 31),
        purpose=(
            "The first out-of-sample test. Read once per strategy version; each reading is "
            "recorded as a trial because reading it repeatedly turns it into training data."
        ),
    ),
    holdout=Period(
        label="holdout",
        start=date(2025, 1, 1),
        end=date(2026, 8, 31),
        purpose=(
            "Untouched. Evaluating it requires an explicit command and writes a permanent "
            "audit record, so the number of times it has been seen is always knowable."
        ),
    ),
    # No recommendation has been recorded yet, so there is no paper period. It
    # begins on the first day the journal receives a live entry, and until then
    # the interface says "not started" rather than showing an empty chart.
    paper_starts=None,
)


# ---------------------------------------------------------------------------
# Execution and costs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """What trading is assumed to cost, per side.

    Large-capitalisation US equities in normal conditions, which is what the
    S&P 500 is. The defaults are deliberately not optimistic: a retail investor
    paying no commission still crosses a spread and still moves the price a
    little, and the robustness battery doubles all of it to check the
    conclusion does not depend on the assumption.
    """

    half_spread_bps: float = 1.5
    slippage_bps: float = 2.5
    commission_bps: float = 0.5
    # A floor for names that are thinner than the index median. The backtest
    # applies the larger of this and the modelled cost.
    min_cost_bps: float = 1.0

    @property
    def one_way_bps(self) -> float:
        return max(
            self.min_cost_bps,
            self.half_spread_bps + self.slippage_bps + self.commission_bps,
        )

    def scaled(self, factor: float) -> CostModel:
        return CostModel(
            half_spread_bps=self.half_spread_bps * factor,
            slippage_bps=self.slippage_bps * factor,
            commission_bps=self.commission_bps * factor,
            min_cost_bps=self.min_cost_bps * factor,
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "one_way_bps": self.one_way_bps}


@dataclass(frozen=True)
class ExecutionSpec:
    """How a decision becomes a fill.

    The delay is the single most important honesty setting in a backtest. A
    score computed from the close of day *t* cannot be traded at that same
    close: the earliest realistic execution is the next session. One day is the
    default and the robustness battery tests two and three.
    """

    delay_days: int = 1
    price_field: str = "close_adj"  # total-return basis; returns, never levels
    # Cash earns nothing. Modelling a money-market yield would help the strategy
    # in exactly the periods it holds cash, so the conservative choice is zero.
    cash_return_annual: float = 0.0


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataGate:
    """A precondition on the *evidence*, not on the result.

    Each of these describes something that, while unmet, makes a good backtest
    result uninformative rather than encouraging. They are checked before any
    number is shown and they cap the status the run may claim.
    """

    key: str
    label: str
    requirement: str
    caps_status_at: ValidationStatus


DATA_GATES: tuple[DataGate, ...] = (
    DataGate(
        key="historical_membership",
        label="Survivorship-free universe",
        requirement=(
            "Index membership must be reconstructed through time, with exits, so that a "
            "historical decision date ranks the companies that were actually in the index "
            "then rather than the ones that survived to today."
        ),
        caps_status_at=ValidationStatus.EXPLORATORY_ONLY,
    ),
    DataGate(
        key="delisted_prices",
        label="Prices for removed companies",
        requirement=(
            "Price history must be available for companies that have since been delisted. "
            "Without it, a portfolio cannot hold the names that failed, and the result is "
            "biased upward by an amount that cannot be estimated from the data."
        ),
        caps_status_at=ValidationStatus.EXPLORATORY_ONLY,
    ),
    DataGate(
        key="delisting_returns",
        label="Delisting returns",
        requirement=(
            "The final return of a delisted holding must be known or explicitly modelled. "
            "Treating an unknown delisting as a zero return silently assumes a total loss; "
            "treating it as a sale at the last price silently assumes a full recovery."
        ),
        caps_status_at=ValidationStatus.EXPLORATORY_ONLY,
    ),
    DataGate(
        key="point_in_time_fundamentals",
        label="Point-in-time fundamentals",
        requirement=(
            "Every financial input must be joined on the date it became knowable, not the "
            "period it describes."
        ),
        caps_status_at=ValidationStatus.NOT_TESTABLE,
    ),
    DataGate(
        key="no_estimate_lookahead",
        label="No analyst-estimate look-ahead",
        requirement=(
            "Analyst estimates have no vintage history, so they must be unavailable at "
            "historical dates rather than back-filled from today's snapshot."
        ),
        caps_status_at=ValidationStatus.NOT_TESTABLE,
    ),
    DataGate(
        key="execution_delay",
        label="Realistic execution",
        requirement=(
            "Trades must execute at least one session after the decision, at prices that "
            "include spread and slippage."
        ),
        caps_status_at=ValidationStatus.NOT_TESTABLE,
    ),
    DataGate(
        key="benchmark_coverage",
        label="Benchmark coverage",
        requirement="SPY and SPMO total returns must cover the whole test period.",
        caps_status_at=ValidationStatus.EXPLORATORY_ONLY,
    ),
)


@dataclass(frozen=True)
class ResultGate:
    """A threshold the result must clear, declared before the test is run."""

    key: str
    label: str
    metric: str
    minimum: float
    rationale: str


RESULT_GATES: tuple[ResultGate, ...] = (
    ResultGate(
        key="ic_positive",
        label="Rank information coefficient is positive",
        metric="ic_mean_3m",
        minimum=0.01,
        rationale=(
            "The lowest bar there is: the ordering must correlate positively with what "
            "happens next. Below this the score is not ranking anything."
        ),
    ),
    ResultGate(
        key="ic_significant",
        label="Information coefficient is statistically distinguishable from zero",
        metric="ic_t_3m",
        minimum=2.0,
        rationale=(
            "Computed on non-overlapping blocks. Overlapping monthly observations of a "
            "three-month forward return are not independent, and treating them as such "
            "inflates the t-statistic by roughly the square root of the overlap."
        ),
    ),
    ResultGate(
        key="decile_spread",
        label="Top decile beats bottom decile",
        metric="decile_spread_3m",
        minimum=0.0,
        rationale=(
            "A ranking that carries information should separate its extremes. This is "
            "weaker evidence than the IC because two deciles are a small sample, but it "
            "fails loudly when the relationship is driven by the middle of the range."
        ),
    ),
    ResultGate(
        key="monotonic",
        label="Forward returns rise with the score",
        metric="decile_monotonicity",
        minimum=0.5,
        rationale=(
            "Spearman correlation between decile index and mean forward return. A signal "
            "that works only at the extremes and is flat in between is more likely to be a "
            "few outliers than a relationship."
        ),
    ),
    ResultGate(
        key="net_excess",
        label="Net of costs, the portfolio beats SPY",
        metric="excess_cagr_net_spy",
        minimum=0.0,
        rationale=(
            "Gross outperformance that turnover consumes is not an edge. Every conclusion "
            "is drawn from the net series."
        ),
    ),
    ResultGate(
        key="information_ratio",
        label="Excess return is large relative to its own noise",
        metric="information_ratio_net_spy",
        minimum=0.3,
        rationale=(
            "Beating the benchmark by a hair with enormous tracking error is not evidence "
            "of skill. 0.3 is a modest bar chosen to be clearable by a real but small edge."
        ),
    ),
)


@dataclass(frozen=True)
class RobustnessThresholds:
    """How much a result may move under a robustness test before it counts as fragile."""

    # Fraction of the base excess return that may be lost before the test is
    # "weakened", and the point at which it has "failed".
    weakened_below: float = 0.5
    failed_below: float = 0.0
    # A valid signal should not depend on the rebalance falling on a particular
    # day, so date shifts are held to a tighter standard.
    date_shift_weakened_below: float = 0.7


@dataclass(frozen=True)
class ValidationSpec:
    """The complete, versioned validation protocol."""

    version: str
    periods: PeriodSpec
    costs: CostModel
    execution: ExecutionSpec
    data_gates: tuple[DataGate, ...]
    result_gates: tuple[ResultGate, ...]
    robustness: RobustnessThresholds
    # Forward-return horizons, in months, over which the ranking is measured.
    ic_horizons: tuple[int, ...] = (1, 3, 6, 12)
    # The portfolio variants tested. All of them are reported; the primary one
    # is declared here so that the answer cannot be whichever won.
    portfolio_variants: tuple[str, ...] = ("top10", "top20", "top_decile")
    primary_variant: str = "top20"
    rebalance: str = "monthly"
    notes: str = ""

    def gate(self, key: str) -> ResultGate:
        for gate in self.result_gates:
            if gate.key == key:
                return gate
        raise KeyError(key)


VALIDATION_V1 = ValidationSpec(
    version="validation-2026.09.1",
    periods=PERIODS,
    costs=CostModel(),
    execution=ExecutionSpec(),
    data_gates=DATA_GATES,
    result_gates=RESULT_GATES,
    robustness=RobustnessThresholds(),
    notes=(
        "The first validation protocol, written before Strategy Version 1 was tested. The "
        "period boundaries, the cost model, the execution delay, the portfolio variants and "
        "every threshold were fixed in advance; the primary portfolio is the top-20 "
        "equal-weight basket regardless of which variant performs best."
    ),
)

ACTIVE_VALIDATION_SPEC = VALIDATION_V1
VALIDATION_SPECS: dict[str, ValidationSpec] = {VALIDATION_V1.version: VALIDATION_V1}


def get_validation_spec(version: str | None = None) -> ValidationSpec:
    if version is None:
        return ACTIVE_VALIDATION_SPEC
    try:
        return VALIDATION_SPECS[version]
    except KeyError:
        raise KeyError(
            f"unknown validation spec {version!r}; known: {sorted(VALIDATION_SPECS)}"
        ) from None


# ---------------------------------------------------------------------------
# Robustness battery
# ---------------------------------------------------------------------------
# Declared as data so that the set of tests run is fixed in advance and visible,
# rather than being whatever the person running it thought to try.


@dataclass(frozen=True)
class RobustnessTest:
    key: str
    label: str
    question: str
    # Parameter overrides passed to the backtest, one run per entry.
    variants: tuple[tuple[str, dict], ...] = field(default_factory=tuple)


ROBUSTNESS_BATTERY: tuple[RobustnessTest, ...] = (
    RobustnessTest(
        key="double_costs",
        label="Double the trading costs",
        question="Does the edge survive if execution is twice as expensive as assumed?",
        variants=(("2x", {"cost_factor": 2.0}), ("4x", {"cost_factor": 4.0})),
    ),
    RobustnessTest(
        key="execution_delay",
        label="Delay execution further",
        question="Does the edge depend on trading immediately after the signal?",
        variants=(
            ("2 days", {"delay_days": 2}),
            ("3 days", {"delay_days": 3}),
            ("5 days", {"delay_days": 5}),
        ),
    ),
    RobustnessTest(
        key="rebalance_shift",
        label="Shift the rebalance date",
        question="Does the result depend on rebalancing on a particular day of the month?",
        variants=(
            ("-5 days", {"rebalance_offset": -5}),
            ("-1 day", {"rebalance_offset": -1}),
            ("+1 day", {"rebalance_offset": 1}),
            ("+5 days", {"rebalance_offset": 5}),
        ),
    ),
    RobustnessTest(
        key="drop_winners",
        label="Remove the biggest winners",
        question="Was the result produced by a handful of companies?",
        variants=(
            ("drop top 1", {"drop_top_contributors": 1}),
            ("drop top 3", {"drop_top_contributors": 3}),
            ("drop top 5", {"drop_top_contributors": 5}),
        ),
    ),
    RobustnessTest(
        key="sector_cap",
        label="Cap sector concentration",
        question="Does the result survive when it cannot concentrate in one sector?",
        variants=(("30%", {"max_sector_weight": 0.30}), ("20%", {"max_sector_weight": 0.20})),
    ),
    RobustnessTest(
        key="weighting",
        label="Score-proportional weighting",
        question="Is equal weighting doing the work, or the ranking?",
        variants=(("score-proportional", {"weighting": "score_proportional"}),),
    ),
    RobustnessTest(
        key="threshold",
        label="Nearby portfolio sizes",
        question="Is the result specific to holding exactly twenty names?",
        variants=(
            ("15 names", {"top_n": 15}),
            ("25 names", {"top_n": 25}),
            ("30 names", {"top_n": 30}),
        ),
    ),
)


REGIMES: tuple[tuple[str, str], ...] = (
    ("bull", "Benchmark above its 200-day average and rising"),
    ("bear", "Benchmark more than 10% below its running peak"),
    ("high_volatility", "Benchmark realised volatility in the top third of the period"),
    ("low_volatility", "Benchmark realised volatility in the bottom third of the period"),
    ("high_rate", "Ten-year Treasury yield in the top third of the period"),
    ("low_rate", "Ten-year Treasury yield in the bottom third of the period"),
)
