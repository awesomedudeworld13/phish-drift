"""Benchmark corpus loading, split construction, and construction audits.

Two public phishing-URL corpora are registered in ``BENCHMARKS``, and running
both is what makes the project's central claim testable. A single collapsing
benchmark shows only that *one* corpus is flawed; a second corpus, audited by
exactly the same rule, distinguishes "this dataset is broken" from "the field
is broken". In this case it came out the first way, which is the more useful
result -- the healthy corpus becomes the control against which the genuine
operational gap can be measured, with the construction artifact removed.

From every corpus we take **only the raw URL string and the label**, and
recompute our own lexical features (``features.py``). Both corpora ship dozens
of precomputed columns; we discard all of them, for two reasons:

* Many (``LineOfCode``, ``HasTitle``, ``HasFavicon``, ``NoOfJS``,
  ``HasPasswordField``, ...) are derived from the *rendered page*. A URL pulled
  from a live phishing feed is typically dead within hours, so those features
  cannot be computed at operational time. Training on them would make the
  benchmark-versus-live comparison impossible by construction.
* PhiUSIIL's ``URLSimilarityIndex`` is computed relative to that dataset's own
  legitimate set, so it encodes test-set membership. It is a leakage feature.

Recomputing from the raw URL also means the two corpora and the live feeds all
traverse one identical code path, so a difference between them is a property of
the data rather than of the pipeline.

Label convention
----------------
The corpora disagree: PhiUSIIL ships ``label == 1`` for *legitimate*, while
Hannousse ships a ``status`` string. Each loader normalises to ``y == 1``
meaning **phishing** -- the positive class is the event being detected, matching
the convention used for TSS, recall and precision everywhere else. The per-corpus
loader is the only place this normalisation happens.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import registrable_domain

# The string template that PhiUSIIL's legitimate class was evidently generated
# from. Discovered by inspection of PhiUSIIL, not fitted; see `template_audit`.
# It is applied unchanged to every corpus, which is what makes the audit a
# comparison rather than a bespoke accusation.
BENIGN_TEMPLATE = re.compile(r"^https://www\.[^/?#]+/?$")


@dataclass(frozen=True)
class BenchmarkSource:
    """A public phishing-URL corpus we can load as ``[url, y, domain]``.

    Two corpora are registered, and the contrast between them is the point.
    Running one benchmark can only show that *a* benchmark is flawed; running a
    second, healthy one distinguishes "this corpus is broken" from "the field
    is broken", and supplies a control on which the operational gap can be
    measured without a construction artifact confounding it.
    """

    key: str
    title: str
    citation: str
    url: str
    filename: str
    min_bytes: int
    _loader: str                      # module-level function that parses it
    licence: str = ""
    sha256: str = ""                  # upstream-published hash, if any
    vendored: str = ""                # repo-relative path to a committed copy

    def resolve(self, dest_dir: Path) -> Path:
        """Return a local path to the corpus, preferring a committed copy.

        A vendored file is used when present because it makes the analysis
        reproducible without a live third party: `data/benchmarks/` holds the
        exact bytes every published number was computed from, verified against
        the hash the publisher advertises. Vendoring is only done where the
        licence permits redistribution.
        """
        if self.vendored:
            path = Path(__file__).resolve().parent.parent / self.vendored
            if path.exists():
                return path
        return self.download(dest_dir)

    def download(self, dest_dir: Path) -> Path:
        import requests

        dest = Path(dest_dir) / self.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size >= self.min_bytes:
            return dest

        # Mendeley rejects the default python-requests User-Agent with a 403,
        # so every source is fetched under an identifying one. Downloading to a
        # temporary path and renaming on success keeps a truncated or
        # error-page response from being cached as if it were the dataset --
        # which would then quietly train a model on HTML.
        headers = {"User-Agent": (
            "phish-drift/1.0 (science-fair research project; "
            "benchmark evaluation; contact via GitHub issues)"
        )}
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(self.url, stream=True, timeout=300, headers=headers) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        if tmp.stat().st_size < self.min_bytes:
            size = tmp.stat().st_size
            tmp.unlink()
            raise RuntimeError(
                f"{self.key}: download was {size:,} bytes, expected at least "
                f"{self.min_bytes:,}; the source may have moved or be rate-limiting"
            )
        tmp.replace(dest)
        return dest

    def load(self, dest_dir: Path) -> pd.DataFrame:
        frame = globals()[self._loader](self.resolve(dest_dir))
        frame["domain"] = [registrable_domain(u) for u in frame["url"]]
        return frame.reset_index(drop=True)


def _load_phiusiil(path: Path) -> pd.DataFrame:
    """PhiUSIIL ships ``label == 1`` for legitimate; we invert to y==1 phishing."""
    with zipfile.ZipFile(path) as z:
        raw = pd.read_csv(io.BytesIO(z.read("PhiUSIIL_Phishing_URL_Dataset.csv")),
                          usecols=["URL", "label"])
    return pd.DataFrame({
        "url": raw["URL"].astype(str),
        "y": (raw["label"] == 0).astype(np.int8),
    })


def _load_hannousse(path: Path) -> pd.DataFrame:
    """Hannousse & Yahiouche ship a ``status`` column of 'phishing'/'legitimate'.

    Like PhiUSIIL, this corpus carries dozens of precomputed features we ignore
    -- many of them page-derived and so uncomputable on live URLs. We take the
    raw ``url`` column and recompute our own features, exactly as for PhiUSIIL,
    so the two corpora remain comparable to each other and to live data.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        raw = pd.read_csv(fh, usecols=["url", "status"])
    return pd.DataFrame({
        "url": raw["url"].astype(str),
        "y": (raw["status"].str.strip().str.lower() == "phishing").astype(np.int8),
    })


def _load_kaitholikkal(path: Path) -> pd.DataFrame:
    """Kaitholikkal & Arthi ship ``url,type`` with type in {legitimate, phishing}."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        raw = pd.read_csv(fh, usecols=["url", "type"], on_bad_lines="skip")
    raw = raw.dropna(subset=["url", "type"])
    return pd.DataFrame({
        "url": raw["url"].astype(str),
        "y": (raw["type"].str.strip().str.lower() != "legitimate").astype(np.int8),
    })


def _load_faizann(path: Path) -> pd.DataFrame:
    """A widely-copied community corpus labelled ``good`` / ``bad``.

    Provenance is weaker than the other three: it is a GitHub repository rather
    than a peer-reviewed release, with no accompanying paper. It is included
    precisely because it is one of the most-copied phishing URL datasets in
    circulation -- it appears in a large number of derivative tutorials and
    notebooks -- so what it teaches a model is worth measuring regardless of
    where it was published. Its status is stated rather than glossed.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        raw = pd.read_csv(fh, usecols=["url", "label"], on_bad_lines="skip")
    raw = raw.dropna(subset=["url", "label"])
    return pd.DataFrame({
        "url": raw["url"].astype(str),
        "y": (raw["label"].str.strip().str.lower() == "bad").astype(np.int8),
    })


BENCHMARKS: dict[str, BenchmarkSource] = {
    "phiusiil": BenchmarkSource(
        key="phiusiil",
        title="PhiUSIIL Phishing URL Dataset",
        citation=(
            "Prasad, A. & Chandra, S. (2024). PhiUSIIL: A diverse security "
            "profile empowered phishing URL detection framework. "
            "UCI Machine Learning Repository, id 967."
        ),
        url="https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip",
        filename="phiusiil.zip",
        min_bytes=1_000_000,
        _loader="_load_phiusiil",
    ),
    "hannousse": BenchmarkSource(
        key="hannousse",
        title="Web page phishing detection (Hannousse & Yahiouche)",
        citation=(
            "Hannousse, A. & Yahiouche, S. (2021). Towards benchmark datasets "
            "for machine learning based website phishing detection: An "
            "experimental study. Engineering Applications of Artificial "
            "Intelligence, 104. Mendeley Data, doi:10.17632/c2gw7fy2j4.3"
        ),
        url=(
            "https://data.mendeley.com/public-files/datasets/c2gw7fy2j4/files/"
            "575316f4-ee1d-453e-a04f-7b950915b61b/file_downloaded"
        ),
        filename="hannousse.csv",
        min_bytes=1_000_000,
        _loader="_load_hannousse",
        licence="CC BY 4.0",
        # Published by Mendeley Data for dataset_B_05_2020.csv. Our committed
        # copy was verified against this before being added; `verify_vendored`
        # re-checks it, and CI runs that check on every push.
        sha256="21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c",
        # Mendeley sits behind bot protection that rejects the TLS fingerprint
        # of Python's HTTP stack with a 403 while serving curl normally, so an
        # unattended CI download is not dependable. CC BY 4.0 permits
        # redistribution with attribution, so the exact bytes are committed.
        vendored="data/benchmarks/hannousse_dataset_B_05_2020.csv.gz",
    ),
    "kaitholikkal": BenchmarkSource(
        key="kaitholikkal",
        title="Phishing URL dataset (Kaitholikkal & Arthi)",
        citation=(
            "Kaitholikkal, J. K. S. & Arthi, B. (2024). Phishing URL dataset. "
            "Mendeley Data, doi:10.17632/vfszbj9b36.1. Legitimate URLs drawn "
            "from the Majestic Million; phishing URLs from PhishTank."
        ),
        url=(
            "https://data.mendeley.com/public-files/datasets/vfszbj9b36/files/"
            "f0de314f-ea72-4385-9faa-f06593bb0a2d/file_downloaded"
        ),
        filename="kaitholikkal.csv",
        min_bytes=10_000_000,
        _loader="_load_kaitholikkal",
        licence="CC BY 4.0",
        sha256="accb2dfbfd3329a8b5cb1b85dcad90314a660c272be755cb45d0f6865014b466",
        vendored="data/benchmarks/kaitholikkal_url_dataset.csv.gz",
    ),
    "faizann": BenchmarkSource(
        key="faizann",
        title="Malicious URL corpus (faizann24, community)",
        citation=(
            "faizann24, 'Using machine learning to detect malicious URLs', "
            "GitHub, data/data.csv. Community dataset with no accompanying "
            "paper; included for its wide reuse, not its provenance."
        ),
        url=(
            "https://raw.githubusercontent.com/faizann24/"
            "Using-machine-learning-to-detect-malicious-URLs/master/data/data.csv"
        ),
        filename="faizann.csv",
        min_bytes=10_000_000,
        _loader="_load_faizann",
        licence="unspecified",
        # GitHub raw serves Python's HTTP stack normally, so unlike the Mendeley
        # sources this one needs no committed copy. It is also of unspecified
        # licence, so redistributing it would not be appropriate in any case.
    ),
}


def verify_vendored(key: str) -> dict:
    """Check a committed corpus still matches the hash its publisher advertises.

    Guards against the quiet failure modes of vendoring: a corrupted file, a
    bad merge, or someone regenerating the copy from a different source
    version. Returns a report rather than raising, so a caller can decide
    whether a mismatch is fatal.
    """
    source = BENCHMARKS[key]
    if not (source.vendored and source.sha256):
        return {"key": key, "checked": False, "reason": "no vendored copy or no published hash"}

    path = Path(__file__).resolve().parent.parent / source.vendored
    if not path.exists():
        return {"key": key, "checked": False, "reason": f"missing {source.vendored}"}

    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    got = digest.hexdigest()
    return {
        "key": key,
        "checked": True,
        "expected": source.sha256,
        "actual": got,
        "match": got == source.sha256,
        "licence": source.licence,
    }

DEFAULT_BENCHMARK = "phiusiil"


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

    ``separation`` is ``|P(property | benign) - P(property | phishing)|``, which
    is exactly the TSS of using that single property as the entire classifier.
    That makes it directly comparable to every other number in this project: a
    separation of 0.94 means one boolean beats most published models.

    Corpora fail in two distinct ways and conflating them would misreport both:

    ``degenerate``       extreme on the benign side (exactly 0.0 or 1.0) and
                         materially different on the other. The danger is the
                         *conjunction*: because no benign row deviates, a rule
                         combining several such properties identifies the benign
                         class with perfect precision. This is PhiUSIIL, where
                         individual separations are only 0.27-0.59 but the four
                         together yield a template rule scoring TSS 0.9897.
    ``high_separation``  one property alone achieves separation >= 0.70. No
                         conjunction needed and no exact extreme required -- a
                         corpus at 0.9999 vs 0.0621 is not "nearly clean", it is
                         solved by a single boolean. Testing only for exact
                         extremes misses this entirely.
    ``elevated``         separation >= 0.35. Plausibly a real-world signal
                         (legitimate sites genuinely do use ``www`` more), but
                         worth stating rather than burying.
    ``constant``         the same extreme on *both* sides. The corpus destroyed
                         the information rather than leaking it -- harmless for
                         leakage, but it cannot teach a model a signal that real
                         deployments have (a corpus that strips URL schemes can
                         never teach anything about ``https``).
    ``ok``              overlapping, as a healthy corpus should be.
    """
    # Parsing must be as forgiving as feature extraction: live and third-party
    # corpora contain genuinely malformed URLs (bracketed hosts, bad encoding),
    # and raw urlsplit raises on them. Dropping those rows here but keeping
    # them during feature extraction would make this audit describe a different
    # corpus than the one the model is trained on.
    from .features import _safe_split

    parsed = [_safe_split(u if "://" in u else "http://" + u) for u in df["url"]]
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
        benign = float(grouped.get(0, float("nan")))
        phishing = float(grouped.get(1, float("nan")))

        separation = abs(benign - phishing)
        # A tolerance rather than exact equality: a property present in 0.01%
        # of one class is constant in substance, and calling it "ok" because it
        # is not exactly zero would hide that the corpus carries no signal there.
        both_absent = max(benign, phishing) <= 0.01
        both_present = min(benign, phishing) >= 0.99

        if both_absent or both_present:
            verdict = "constant"
        elif benign in (0.0, 1.0) and separation >= 0.05:
            verdict = "degenerate"
        elif separation >= 0.70:
            verdict = "high_separation"
        elif separation >= 0.35:
            verdict = "elevated"
        else:
            verdict = "ok"

        out[col] = {
            "benign": round(benign, 6),
            "phishing": round(phishing, 6),
            "separation": round(separation, 6),
            "verdict": verdict,
        }
    return out


def properties_with_verdict(rates: dict, *verdicts: str) -> list[str]:
    """Names of properties whose verdict is any of ``verdicts``."""
    return [name for name, v in rates.items() if v.get("verdict") in verdicts]


def degenerate_properties(rates: dict) -> list[str]:
    """Properties that are a class label in disguise, by either pathology."""
    return properties_with_verdict(rates, "degenerate", "high_separation")


def constant_properties(rates: dict) -> list[str]:
    """Properties the corpus has flattened away on both classes."""
    return properties_with_verdict(rates, "constant")


def max_single_property_tss(rates: dict) -> float:
    """Best TSS achievable from one surface boolean alone.

    A compact summary of how much of a corpus is solved before any learning: it
    is the strongest single-feature baseline, and any model that does not
    comfortably beat it has demonstrated nothing about the task.
    """
    return round(max((v["separation"] for v in rates.values()), default=0.0), 6)


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
