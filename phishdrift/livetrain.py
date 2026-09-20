"""Training on live data, and the 2x2 that asks whether it closes the gap.

The project has two questions, and the four-cell decomposition in ``gap.py``
only answers the first:

1. How far does benchmark performance fall on live data?
2. **Does training on live data close that gap?**

Question 2 needs a model the benchmark never touched. This module builds one
from the accumulated daily snapshots and scores the 2x2 that the sibling solar
project uses:

                        |  benchmark test  |  live holdout
    --------------------+------------------+----------------
    benchmark-trained   |        A         |       B
    live-trained        |        C         |       D

* **A - B** is the gap: what the benchmark overstates.
* **D - B** is the answer to question 2: how much of that gap live training
  recovers. If it is near zero, the loss is not a training-data problem, and
  collecting more live data will not fix it.
* **C** is the control that keeps the comparison honest. A live-trained model
  that also scores well on the benchmark test has learned something general; one
  that scores well only on live data may have learned this month's campaigns.

Temporal splitting, not random
------------------------------
Live rows are split by snapshot date: earlier days train, later days test. A
random split would let a model memorise a phishing campaign from Tuesday and be
tested on the same campaign from Thursday, which is precisely the leakage the
rest of this project exists to measure. Splitting on time is the only protocol
that answers "would this have worked if deployed".

Domains are additionally held disjoint across the boundary. A campaign that
spans the cutoff would otherwise leak across it even though the dates differ;
where a domain appears on both sides, its later rows are dropped, so the
training side keeps its history and the test side stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import evaluate
from .benchmark import Split
from .model import TrainedModel, train as train_model

# A live-trained model is only worth fitting once there is enough history for a
# temporal split to mean anything. These are deliberately conservative: below
# them the answer to question 2 would be noise presented as a result.
MIN_TRAIN_DAYS = 10
MIN_HOLDOUT_DAYS = 4
MIN_ROWS_PER_SIDE = 400
MIN_POSITIVES_PER_SIDE = 100


@dataclass
class Readiness:
    """Whether the accumulated live record can yet support question 2."""

    ready: bool
    reason: str
    n_days: int
    n_rows: int
    days_needed: int = 0

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "n_days": self.n_days,
            "n_rows": self.n_rows,
            "days_needed": self.days_needed,
            "thresholds": {
                "min_train_days": MIN_TRAIN_DAYS,
                "min_holdout_days": MIN_HOLDOUT_DAYS,
                "min_rows_per_side": MIN_ROWS_PER_SIDE,
                "min_positives_per_side": MIN_POSITIVES_PER_SIDE,
            },
        }


def headline_rows(live: pd.DataFrame) -> pd.DataFrame:
    """The live subset used for question 2: OpenPhish vs realistic benign.

    Templated benign rows are excluded here. They exist to isolate a
    construction artifact in ``gap.py``; training on them would teach a live
    model the very template whose effect we are trying to remove.
    """
    keep = (
        ((live.y == 1) & (live.source == "openphish"))
        | ((live.y == 0) & (live.source == "commoncrawl"))
    )
    return live[keep].copy()


def assess(live: pd.DataFrame) -> Readiness:
    """Report whether there is yet enough live history, and what is missing."""
    rows = headline_rows(live)
    if rows.empty:
        return Readiness(False, "no live snapshots yet", 0, 0,
                         MIN_TRAIN_DAYS + MIN_HOLDOUT_DAYS)

    days = sorted(rows.snapshot_date.unique())
    need = MIN_TRAIN_DAYS + MIN_HOLDOUT_DAYS
    if len(days) < need:
        return Readiness(
            False,
            f"{len(days)} of {need} days collected; a temporal split needs "
            f"{MIN_TRAIN_DAYS} training days and {MIN_HOLDOUT_DAYS} holdout days",
            len(days), len(rows), need - len(days),
        )

    split = temporal_split(rows)
    if split is None:
        return Readiness(False, "temporal split left too few rows on one side",
                         len(days), len(rows))

    for name, part in (("train", split.train), ("holdout", split.test)):
        if len(part) < MIN_ROWS_PER_SIDE:
            return Readiness(False, f"{name} side has {len(part)} rows, "
                                    f"need {MIN_ROWS_PER_SIDE}",
                             len(days), len(rows))
        if int(part.y.sum()) < MIN_POSITIVES_PER_SIDE:
            return Readiness(False, f"{name} side has {int(part.y.sum())} phishing "
                                    f"rows, need {MIN_POSITIVES_PER_SIDE}",
                             len(days), len(rows))

    return Readiness(True, "sufficient live history", len(days), len(rows))


def temporal_split(rows: pd.DataFrame, holdout_frac: float = 0.3,
                   val_frac: float = 0.2) -> Split | None:
    """Split live rows by date: earliest days train, latest days test.

    ``val_frac`` is taken from the *end of the training period*, not at random,
    so threshold selection also happens on data strictly older than the holdout.
    Selecting a threshold on randomly scattered days would leak future
    information into the operating point.
    """
    days = sorted(rows.snapshot_date.unique())
    if len(days) < 3:
        return None

    n_holdout = max(1, int(round(len(days) * holdout_frac)))
    train_days = days[:-n_holdout]
    holdout_days = days[-n_holdout:]
    if len(train_days) < 2:
        return None

    n_val = max(1, int(round(len(train_days) * val_frac)))
    val_days = train_days[-n_val:]
    fit_days = train_days[:-n_val]
    if not fit_days:
        return None

    train = rows[rows.snapshot_date.isin(fit_days)].copy()
    val = rows[rows.snapshot_date.isin(val_days)].copy()
    test = rows[rows.snapshot_date.isin(holdout_days)].copy()

    # Drop later rows for any domain already seen earlier, so a campaign
    # straddling the cutoff cannot leak across it.
    seen = set(train.domain)
    val = val[~val.domain.isin(seen)]
    seen |= set(val.domain)
    test = test[~test.domain.isin(seen)]

    if train.empty or val.empty or test.empty:
        return None
    if val.y.nunique() < 2 or test.y.nunique() < 2 or train.y.nunique() < 2:
        return None

    return Split(name="live_temporal", train=train, val=val, test=test)


def train_live_model(live: pd.DataFrame, n_estimators: int = 300,
                     seed: int = 20260920) -> tuple[TrainedModel | None, Split | None]:
    """Fit a model on live snapshots alone, with thresholds frozen on live validation."""
    rows = headline_rows(live)
    split = temporal_split(rows)
    if split is None:
        return None, None

    model = train_model(split, n_estimators=n_estimators, seed=seed)
    model.metadata["benchmark"] = "live"
    model.metadata["live_span"] = {
        "train": [str(split.train.snapshot_date.min()), str(split.train.snapshot_date.max())],
        "val": [str(split.val.snapshot_date.min()), str(split.val.snapshot_date.max())],
        "holdout": [str(split.test.snapshot_date.min()), str(split.test.snapshot_date.max())],
    }
    return model, split


def run_2x2(benchmark_model: TrainedModel,
            benchmark_test: pd.DataFrame,
            live: pd.DataFrame,
            n_estimators: int = 300,
            n_resamples: int = 1000,
            seed: int = 20260920) -> dict:
    """The training-source x test-set 2x2, and the answer to question 2.

    Returns a document with a ``ready`` flag. When the live record is too short
    the structure is still returned, carrying what is missing and how many more
    days are needed, so the report states the question as open rather than
    omitting it.
    """
    readiness = assess(live)
    out: dict = {"readiness": readiness.to_dict()}
    if not readiness.ready:
        out["question"] = (
            "Does training on live data close the benchmark-to-operational gap? "
            "Not yet answerable: " + readiness.reason + "."
        )
        return out

    live_model, split = train_live_model(live, n_estimators=n_estimators, seed=seed)
    if live_model is None:
        out["readiness"]["ready"] = False
        out["readiness"]["reason"] = "temporal split failed"
        return out

    holdout = split.test
    cells: dict[str, dict] = {}

    def score(tag: str, model: TrainedModel, frame: pd.DataFrame, desc: str) -> None:
        y = frame.y.to_numpy()
        prob = model.predict_proba(frame.url)
        thr = model.thresholds["tss"]
        s = evaluate.score_at(y, prob, thr).to_dict()
        lo, hi = evaluate.cluster_bootstrap_ci(
            y, prob, thr, groups=frame.domain.to_numpy(), n_resamples=n_resamples
        )
        s["tss_ci95"] = [round(lo, 6), round(hi, 6)]
        s["description"] = desc
        cells[tag] = s

    score("A_benchmark_trained_on_benchmark", benchmark_model, benchmark_test,
          "Benchmark-trained model on its own held-out benchmark test.")
    score("B_benchmark_trained_on_live", benchmark_model, holdout,
          "Benchmark-trained model on the live holdout — the deployed reality.")
    score("C_live_trained_on_benchmark", live_model, benchmark_test,
          "Live-trained model on the benchmark test — does live training "
          "generalise, or only fit this month's campaigns?")
    score("D_live_trained_on_live", live_model, holdout,
          "Live-trained model on the live holdout — the best achievable with "
          "operational data alone.")

    gap = cells["A_benchmark_trained_on_benchmark"]["tss"] - cells["B_benchmark_trained_on_live"]["tss"]
    recovery = cells["D_live_trained_on_live"]["tss"] - cells["B_benchmark_trained_on_live"]["tss"]

    point, lo, hi = evaluate.paired_difference_ci(
        holdout.y.to_numpy(), live_model.predict_proba(holdout.url),
        live_model.thresholds["tss"],
        holdout.y.to_numpy(), benchmark_model.predict_proba(holdout.url),
        benchmark_model.thresholds["tss"],
        groups_a=holdout.domain.to_numpy(), groups_b=holdout.domain.to_numpy(),
        n_resamples=n_resamples,
    )

    out.update({
        "cells": cells,
        "gap_A_minus_B": round(gap, 6),
        "recovery_D_minus_B": round(recovery, 6),
        "recovery_ci95": [round(lo, 6), round(hi, 6)],
        "recovery_significant": bool(lo > 0),
        "recovery_fraction_of_gap": round(recovery / gap, 6) if gap else None,
        "live_model": live_model.metadata,
        "question": (
            "Does training on live data close the benchmark-to-operational gap?"
        ),
        "answer": (
            f"Live training recovers {recovery:+.3f} TSS over the benchmark-trained "
            f"model on the same holdout (95% CI {lo:.3f}..{hi:.3f}), against a gap of "
            f"{gap:.3f}. "
            + (
                "The interval excludes zero, so live training measurably helps."
                if lo > 0 else
                "The interval includes zero, so on this much live data we cannot "
                "claim live training helps — which, if it holds as the record "
                "grows, means the loss is not a training-data problem and "
                "collecting more live data will not fix it."
            )
        ),
    })
    return out
