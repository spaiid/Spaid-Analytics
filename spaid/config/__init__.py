"""Configuration: runtime settings plus the versioned model specifications.

`settings` holds things that vary by environment. `scoring`, `valuation` and
`portfolio` hold things that define what the numbers *mean*, and are versioned
so a recommendation made months ago can still be reproduced exactly.
"""

from spaid.config.portfolio import (
    ACTIVE_PORTFOLIO_SPEC,
    RISK_PROFILES,
    AccountType,
    ActionType,
    CostSpec,
    PortfolioSpec,
    RiskTolerance,
    TaxSpec,
    TurnoverSpec,
)
from spaid.config.scoring import (
    ACTIVE_SCORING_SPEC,
    ALL_METRICS,
    METRIC_BY_KEY,
    BusinessModel,
    Category,
    MetricSpec,
    PeerBasis,
    ScoringSpec,
    get_scoring_spec,
    score_band,
)
from spaid.config.settings import (
    ARTIFACTS,
    BROWSER_UA,
    CACHE,
    CURATED,
    DATA_DIR,
    DERIVED,
    RAW,
    ROOT,
    SECTOR_ETF,
    SETTINGS,
    USER_AGENT,
    Settings,
)
from spaid.config.valuation import (
    ACTIVE_VALUATION_SPEC,
    Scenario,
    ValuationClass,
    ValuationMethod,
    ValuationSpec,
    get_valuation_spec,
)

__all__ = [
    "ACTIVE_PORTFOLIO_SPEC",
    "ACTIVE_SCORING_SPEC",
    "ACTIVE_VALUATION_SPEC",
    "ALL_METRICS",
    "ARTIFACTS",
    "BROWSER_UA",
    "CACHE",
    "CURATED",
    "DATA_DIR",
    "DERIVED",
    "METRIC_BY_KEY",
    "RAW",
    "RISK_PROFILES",
    "ROOT",
    "SECTOR_ETF",
    "SETTINGS",
    "USER_AGENT",
    "AccountType",
    "ActionType",
    "BusinessModel",
    "Category",
    "CostSpec",
    "MetricSpec",
    "PeerBasis",
    "PortfolioSpec",
    "RiskTolerance",
    "Scenario",
    "ScoringSpec",
    "Settings",
    "TaxSpec",
    "TurnoverSpec",
    "ValuationClass",
    "ValuationMethod",
    "ValuationSpec",
    "get_scoring_spec",
    "get_valuation_spec",
    "score_band",
]
