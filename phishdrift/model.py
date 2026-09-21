"""Model training, threshold freezing, and artifact persistence.

A trained artifact carries its frozen thresholds with it. This is deliberate:
the thresholds are part of the deployed model, not a parameter the evaluation
script gets to choose later. Once written, an artifact scored against live data
uses the thresholds committed at training time, with no retuning -- the same
discipline the Helios preregistration applies to solar forecasts.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from benchgap import evaluate
from .benchmark import BENIGN_TEMPLATE, Split
from .features import FEATURE_NAMES, feature_matrix

MODEL_VERSION = 1


def _git_commit() -> str:
    """Record the code state that produced an artifact, for reproducibility."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


@dataclass
class TrainedModel:
    """A classifier plus the operating points frozen at training time."""

    estimator: object
    feature_names: tuple[str, ...]
    thresholds: dict[str, float]
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, urls) -> np.ndarray:
        """P(phishing) for an iterable of raw URL strings."""
        X = feature_matrix(urls)
        return self.estimator.predict_proba(X)[:, 1]

    def save(self, path: Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        path.with_suffix(".meta.json").write_text(
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )
        return path

    @staticmethod
    def load(path: Path) -> "TrainedModel":
        import joblib

        return joblib.load(Path(path))


def train(split: Split, n_estimators: int = 300, max_depth: int | None = None,
          seed: int = 20260920, n_jobs: int = -1) -> TrainedModel:
    """Fit on ``split.train`` and freeze both operating points on ``split.val``.

    Two thresholds are stored:

    ``f1``   -- maximises F1 on validation. This is the field default and the
                one whose operational collapse is the pre-registered hypothesis.
    ``tss``  -- maximises Youden's J on validation. Base-rate invariant.

    Both come from the *validation* partition. ``split.test`` is not touched
    here, and live data does not exist at training time by construction.
    """
    X_train = feature_matrix(split.train["url"])
    y_train = split.train["y"].to_numpy()

    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=n_jobs,
    )
    estimator.fit(X_train, y_train)

    X_val = feature_matrix(split.val["url"])
    y_val = split.val["y"].to_numpy()
    prob_val = estimator.predict_proba(X_val)[:, 1]

    thresholds = {
        "f1": evaluate.select_threshold(y_val, prob_val, objective="f1"),
        "tss": evaluate.select_threshold(y_val, prob_val, objective="tss"),
    }

    importances = sorted(
        zip(FEATURE_NAMES, estimator.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "split_name": split.name,
        "split_sizes": split.sizes(),
        "estimator": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": seed,
        },
        "n_features": len(FEATURE_NAMES),
        "thresholds": {k: round(v, 6) for k, v in thresholds.items()},
        "top_features": [{"name": n, "importance": round(float(i), 6)}
                         for n, i in importances[:15]],
    }

    return TrainedModel(
        estimator=estimator,
        feature_names=FEATURE_NAMES,
        thresholds=thresholds,
        metadata=metadata,
    )


class TemplateRule:
    """Zero-parameter baseline: ``URL does not match the benign string template``.

    Exposed through the same ``predict_proba`` interface as a trained model so
    it can be pushed through the identical evaluation path. It returns hard 0/1
    "probabilities", which is honest -- it is a rule, not a calibrated model,
    and its Brier score should be read with that in mind.

    Its purpose is to establish how much of a benchmark score is available
    without any learning at all. A learned model that fails to beat it has
    demonstrated nothing about the task.
    """

    thresholds = {"f1": 0.5, "tss": 0.5}
    feature_names = ()
    metadata = {"estimator": "TemplateRule", "parameters": 0}

    @staticmethod
    def predict_proba(urls) -> np.ndarray:
        import pandas as pd

        s = pd.Series(list(urls), dtype=str)
        return (~s.str.match(BENIGN_TEMPLATE)).to_numpy(dtype=float)
