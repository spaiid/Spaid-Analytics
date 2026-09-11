"""Predecessor registrants: carrying filing history across a reorganisation.

When a company reorganises into a new holding entity it receives a new Central
Index Key, and its entire filing history stays behind under the old one. On the
S&P 500 today that affects ExxonMobil and BlackRock, each of which has a single
filing under its current registrant and a decade of history under its previous
one. Without linking them, two of the largest companies in the index score on
one quarter of data, or not at all.

The distinction this module exists to preserve is between a **reorganisation**
and a **spin-off**, because they are opposite cases:

* In a reorganisation the same business continues under a new legal wrapper, so
  the predecessor's history *is* this company's history and must be carried
  forward.
* In a spin-off a piece of a larger company becomes independent. FedEx Freight's
  history is not FedEx's, and attaching the parent's revenue and margins to it
  would be straightforwardly false -- the resulting "company" never existed.

So only reorganisations are listed here. Spin-offs are deliberately absent and
surface instead as a limited-history warning, which is the honest outcome: a
company with two quarters of public financials genuinely cannot be assessed on
five-year growth, and the confidence system should say so rather than borrow
numbers from a parent.

Every entry is a judgement that should be re-verified when the constituents
change. `verified` records when it last was.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Predecessor:
    successor_cik: int
    predecessor_cik: int
    successor_name: str
    predecessor_name: str
    reason: str
    verified: str  # ISO date on which this link was last checked


# Reorganisations only. A spin-off must never be added here.
PREDECESSORS: tuple[Predecessor, ...] = (
    Predecessor(
        successor_cik=2115436,
        predecessor_cik=34088,
        successor_name="ExxonMobil Holdings Corp",
        predecessor_name="Exxon Mobil Corp",
        reason=(
            "Holding-company reorganisation: the operating business is unchanged and the "
            "new registrant succeeded to the old one's reporting obligations."
        ),
        verified="2026-09-10",
    ),
    Predecessor(
        successor_cik=2012383,
        predecessor_cik=1364742,
        successor_name="BlackRock, Inc.",
        predecessor_name="BlackRock Inc.",
        reason=(
            "Holding-company reorganisation completed in 2024; the asset-management "
            "business and its financial history continue unchanged."
        ),
        verified="2026-09-10",
    ),
)

BY_SUCCESSOR: dict[int, Predecessor] = {p.successor_cik: p for p in PREDECESSORS}


def predecessor_for(cik: int) -> Predecessor | None:
    return BY_SUCCESSOR.get(int(cik))


def all_ciks_for(cik: int) -> list[int]:
    """The registrant's own identifier plus any predecessor whose history it inherits."""
    link = predecessor_for(cik)
    return [int(cik)] if link is None else [int(cik), link.predecessor_cik]
