"""Benchmark corpus loading, split construction, and construction audits.

We consume only the ``URL`` and ``label`` columns of PhiUSIIL. Its other 54
columns are deliberately discarded for two reasons:

* Roughly half (``LineOfCode``, ``HasTitle``, ``HasFavicon``, ``NoOfJS``,
  ``HasPasswordField``, ...) are derived from the *rendered page*. A URL pulled
  from a live phishing feed is typically dead within hours, so those features
  cannot be computed at operational time. Training on them would make the
  benchmark-versus-live comparison impossible by construction.
* ``URLSimilarityIndex`` is computed relative to the dataset's own legitimate
  set, so it encodes test-set membership. It is a leakage feature.

Recomputing our own lexical features from the raw URL (``features.py``) means
benchmark rows and live rows traverse one identical code path.

Label convention
----------------
PhiUSIIL ships ``label == 1`` for *legitimate*. We invert it on load, so that
throughout this project ``y == 1`` means **phishing** -- the positive class is
the event being detected, matching the convention used for TSS, recall and
precision everywhere else in the literature. ``load()`` is the only place this
inversion happens.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import registrable_domain

PHIUSIIL_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
PHIUSIIL_MEMBER = "PhiUSIIL_Phishing_URL_Dataset.csv"

# The string template that PhiUSIIL's legitimate class was evidently generated
# from. Discovered by inspection, not fitted; see `template_audit`.
BENIGN_TEMPLATE = re.compile(r"^https://www\.[^/?#]+/?$")


def download(dest: Path) -> Path:
    """Fetch the PhiUSIIL archive to ``dest`` unless it is already present."""
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    with requests.get(PHIUSIIL_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return dest


def load(archive: Path) -> pd.DataFrame:
    """Load PhiUSIIL as ``[url, y, domain]`` with ``y == 1`` meaning phishing."""
    with zipfile.ZipFile(archive) as z:
        raw = pd.read_csv(io.BytesIO(z.read(PHIUSIIL_MEMBER)), usecols=["URL", "label"])

    df = pd.DataFrame({
        "url": raw["URL"].astype(str),
        # Inversion happens here and nowhere else. See module docstring.
        "y": (raw["label"] == 0).astype(np.int8),
    })
    df["domain"] = [registrable_domain(u) for u in df["url"]]
    return df


# --------------------------------------------------------------------------
# Construction audit
# --------------------------------------------------------------------------

@dataclass
class TemplateAudit:
    """Result of asking how much of a benchmark is explained by string shape."""

    n: int
    tp: int
    fp: int
    fn: int
    tn: int
    benign_template_share: float
    phish_template_share: float

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def false_alarm_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def tss(self) -> float:
        return self.recall - self.false_alarm_rate

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "benign_template_share": round(self.benign_template_share, 6),
            "phish_template_share": round(self.phish_template_share, 6),
            "recall": round(self.recall, 6),
            "false_alarm_rate": round(self.false_alarm_rate, 6),
            "precision": round(self.precision, 6),
            "accuracy": round(self.accuracy, 6),
            "tss": round(self.tss, 6),
        }


def template_audit(df: pd.DataFrame) -> TemplateAudit:
    """Score the zero-parameter rule ``URL does not match BENIGN_TEMPLATE``.

    This is the project's control for *dataset construction artifacts*. If a
    hand-written regex over surface string shape scores comparably to published
    machine-learning results on the same corpus, then those results are not
    evidence of phishing detection skill -- they are evidence that the two
    classes were collected by different pipelines and are separable on
    formatting alone.

    It carries no learned parameters, so it cannot overfit and needs no split.
    """
    matches_template = df["url"].str.match(BENIGN_TEMPLATE)
    pred = (~matches_template).astype(np.int8)   # not the benign template -> call it phishing
    y = df["y"].to_numpy()

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    benign = matches_template[y == 0]
    phish = matches_template[y == 1]

    return TemplateAudit(
        n=len(df),
        tp=tp, fp=fp, fn=fn, tn=tn,
        benign_template_share=float(benign.mean()) if len(benign) else 0.0,
        phish_template_share=float(phish.mean()) if len(phish) else 0.0,
    )


def structural_degeneracy(df: pd.DataFrame) -> dict:
    """Per-class rates of the four surface properties the template is built from.

    A healthy corpus has these overlapping between classes. Rates of exactly
    0.0 or 1.0 on one side mean the property is a class label in disguise.
    """
    from urllib.parse import urlsplit

    parsed = [urlsplit(u if "://" in u else "http://" + u) for u in df["url"]]
    frame = pd.DataFrame({
        "y": df["y"].to_numpy(),
        "is_https": [p.scheme == "https" for p in parsed],
        "has_www": [(p.hostname or "").startswith("www.") for p in parsed],
        "has_path": [len((p.path or "").rstrip("/")) > 0 for p in parsed],
        "has_query": [bool(p.query) for p in parsed],
    })
    out = {}
    for col in ("is_https", "has_www", "has_path", "has_query"):
        grouped = frame.groupby("y")[col].mean()
        out[col] = {
            "benign": round(float(grouped.get(0, float("nan"))), 6),
            "phishing": round(float(grouped.get(1, float("nan"))), 6),
        }
    return out


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

@dataclass
class Split:
    """A train/validation/test partition, carrying its own provenance."""

    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "train_base_rate": round(float(self.train.y.mean()), 6),
            "test_base_rate": round(float(self.test.y.mean()), 6),
        }


def random_split(df: pd.DataFrame, seed: int = 20260920,
                 fractions=(0.70, 0.15, 0.15)) -> Split:
    """Stratified i.i.d. row split -- the protocol the published literature uses.

    This is cell 1 of the study. It is *not* the honest protocol; it is the
    protocol whose optimism we are quantifying. URLs from one domain land on
    both sides of the partition, so a model can memorise domains it will be
    tested on.
    """
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, group in df.groupby("y"):            # stratify on class
        idx = rng.permutation(group.index.to_numpy())
        n_train = int(len(idx) * fractions[0])
        n_val = int(len(idx) * fractions[1])
        parts["train"].append(idx[:n_train])
        parts["val"].append(idx[n_train:n_train + n_val])
        parts["test"].append(idx[n_train + n_val:])
    return Split(
        name="random",
        train=df.loc[np.concatenate(parts["train"])].copy(),
        val=df.loc[np.concatenate(parts["val"])].copy(),
        test=df.loc[np.concatenate(parts["test"])].copy(),
    )


def domain_disjoint_split(df: pd.DataFrame, seed: int = 20260920,
                          fractions=(0.70, 0.15, 0.15)) -> Split:
    """Split on registrable domains, so no eTLD+1 appears in two partitions.

    This is cell 2. It is the same data and the same era as cell 1 -- only the
    partitioning changes -- so the difference between them isolates how much of
    the benchmark score comes from memorising domains rather than learning
    phishing. It is the direct analogue of clustering the bootstrap over solar
    active regions instead of over individual magnetogram windows.

    Domains are allocated whole, shuffled, largest-first-free: because domain
    sizes are heavily skewed, we assign in random order and let the partition
    fill to its quota, which keeps partition sizes close to the requested
    fractions without ever splitting a domain.
    """
    rng = np.random.default_rng(seed)
    domains = df["domain"].to_numpy()
    unique = np.unique(domains)
    rng.shuffle(unique)

    counts = df.groupby("domain").size()
    target_train = len(df) * fractions[0]
    target_val = len(df) * fractions[1]

    assign: dict[str, str] = {}
    n_train = n_val = 0
    for dom in unique:
        size = int(counts[dom])
        if n_train < target_train:
            assign[dom] = "train"
            n_train += size
        elif n_val < target_val:
            assign[dom] = "val"
            n_val += size
        else:
            assign[dom] = "test"

    bucket = df["domain"].map(assign)
    split = Split(
        name="domain_disjoint",
        train=df[bucket == "train"].copy(),
        val=df[bucket == "val"].copy(),
        test=df[bucket == "test"].copy(),
    )

    # Fail closed: the entire point of this split is the disjointness guarantee,
    # so verify it rather than trusting the construction above.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(getattr(split, a).domain) & set(getattr(split, b).domain)
        if overlap:
            raise AssertionError(
                f"domain_disjoint_split leaked {len(overlap)} domains between "
                f"{a} and {b}; example: {sorted(overlap)[:3]}"
            )
    return split
