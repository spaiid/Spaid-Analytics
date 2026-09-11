from __future__ import annotations
from typing import Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

SignalType = Literal[
    "market.price", "market.volume", "market.volatility"
]

class Signal(BaseModel):
    uid: str
    ts: datetime             # precise timestamp (UTC)
    entity: str              # ticker, e.g., NVDA
    kind: SignalType
    value: float
    confidence: float = Field(ge=0.0, le=1.0, default=0.95)

    source: str              # e.g., "yfinance"
    meta: Dict[str, Any] = {}

    asof: Optional[str] = None  # trading day "YYYY-MM-DD"
