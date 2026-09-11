"""Purged, embargoed walk-forward splits.

The subtle failure in most equity backtests is not that the split is in the
wrong place -- it is that the *label* straddles it. A row dated `t` carries a
target measured over `(t, t + horizon]`. If `t` sits in the training set and
`t + horizon` reaches past the split date, the model has been trained on the
very period it is about to be scored on.

So the training set ends `horizon + embargo` trading days before the first test
date. The `horizon` part is the purge and is mandatory -- without it the leak is
guaranteed, not merely possible. The `embargo` is extra separation to blunt the
serial correlation that overlapping windows leave behind.

Each fold trains on everything available up to its cutoff (expanding, not
rolling) because with ~13 years of history throwing away old data costs more
than the staleness it saves.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import polars as pl

from spaid.config import SETTINGS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: date
    train_end: date      # inclusive; already purged and embargoed
    test_start: date     # inclusive
    test_end: date       # inclusive
    n_train: int
    n_test: int

    @property
    def gap_days(self) -> int:
        return (self.test_start - self.train_end).days


def make_folds(
    panel: pl.DataFrame,
    *,
    horizon: int | None = None,
    embargo: int | None = None,
    min_train_days: int | None = None,
    refit_every: int | None = None,
) -> list[Fold]:
    """Expanding walk-forward folds over the panel's trading calendar."""
    cfg = SETTINGS.model
    horizon = horizon if horizon is not None else cfg.horizon_days
    embargo = embargo if embargo is not None else cfg.embargo_days
    min_train_days = min_train_days if min_train_days is not None else cfg.min_train_days
    refit_every = refit_every if refit_every is not None else cfg.refit_every_days

    if embargo < horizon:
        raise ValueError(
            f"embargo ({embargo}) < horizon ({horizon}): training labels would "
            "overlap the test window"
        )

    dates = panel.select("date").unique().sort("date")["date"].to_list()
    n = len(dates)
    gap = horizon + embargo

    folds: list[Fold] = []
    start_i = min_train_days + gap
    if start_i >= n:
        raise ValueError(
            f"only {n} trading days available; need more than {start_i} "
            f"({min_train_days} training + {gap} purge/embargo)"
        )

    counts = dict(
        panel.group_by("date").agg(pl.len().alias("n")).iter_rows()
    )

    for i, test_start_i in enumerate(range(start_i, n, refit_every)):
        test_end_i = min(test_start_i + refit_every - 1, n - 1)
        train_end_i = test_start_i - gap - 1
        if train_end_i < min_train_days - 1:
            continue

        train_dates = dates[: train_end_i + 1]
        test_dates = dates[test_start_i : test_end_i + 1]
        if not train_dates or not test_dates:
            continue

        folds.append(
            Fold(
                index=i,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                n_train=sum(counts.get(d, 0) for d in train_dates),
                n_test=sum(counts.get(d, 0) for d in test_dates),
            )
        )

    log.info(
        "walk-forward: %d folds, %d-day purge+embargo gap, first test %s, last test %s",
        len(folds), gap, folds[0].test_start if folds else "-", folds[-1].test_end if folds else "-",
    )
    return folds


def split(panel: pl.DataFrame, fold: Fold) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Materialise one fold. Training rows with a null target are dropped."""
    train = panel.filter(
        (pl.col("date") <= fold.train_end) & pl.col("target").is_not_null()
    )
    test = panel.filter(
        (pl.col("date") >= fold.test_start) & (pl.col("date") <= fold.test_end)
    )
    return train, test


def assert_no_leakage(train: pl.DataFrame, test: pl.DataFrame, horizon: int) -> None:
    """Fail loudly if any training label reaches into the test window.

    Cheap enough to run on every fold, and it is the one invariant that, if it
    ever breaks, invalidates every number the app reports.
    """
    if train.is_empty() or test.is_empty():
        return
    last_train = train["date"].max()
    first_test = test["date"].min()
    gap = (first_test - last_train).days
    if gap <= horizon:
        raise AssertionError(
            f"leakage: last training date {last_train} + {horizon}-day target "
            f"window reaches test start {first_test} (calendar gap {gap} days)"
        )
