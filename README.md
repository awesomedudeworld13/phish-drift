# phish-drift

**How much of a phishing-detection benchmark score is detection skill, and how
much is an artifact of how the dataset was built?**

For PhiUSIIL — a 2024 URL phishing benchmark with 235,795 samples, hosted by the
UCI Machine Learning Repository — the answer is that essentially none of it
survives contact with live data, and the reason is not drift.

---

## The three findings

### 1. A regular expression beats the published machine learning

Every one of PhiUSIIL's 134,850 legitimate URLs matches the string template
`https://www.<domain>` — https, `www.`-prefixed, no path, no query. Not 99%.
All of them. The phishing class, drawn from live feeds, matches it 1.03% of the
time.

So a rule with **zero learned parameters** — *"does this URL fail to match that
template? then call it phishing"* — scores:

| n | accuracy | precision | recall | false alarm | **TSS** |
|---|---|---|---|---|---|
| 235,795 | 0.9956 | 1.0000 | 0.9897 | 0.0000 | **0.9897** |

Published models on this corpus report around 99.2% accuracy. The regex gets
99.56%. Those models are not detecting phishing; they are detecting which
harvesting pipeline produced the string.

| surface property | legitimate | phishing |
|---|---|---|
| `is_https` | **1.0000** | 0.4874 |
| `has_www` | **1.0000** | 0.4135 |
| `has_path` | **0.0000** | 0.2719 |
| `has_query` | **0.0000** | 0.0602 |

A rate of exactly 0 or 1 on one side means the property is not a feature. It is
the class label, written in a different alphabet.

### 2. Models learn the artifact, and it is worth nothing operationally

A RandomForest trained on our own 52 lexical features ranks
`is_https`, `n_slashes`, `path_length`, `has_www` as its top features — the same
surface properties the regex uses. Scored against live OpenPhish URLs and real
benign URLs from Common Crawl, at a threshold frozen on benchmark validation
data:

| cell | what changes | n | **TSS** |
|---|---|---|---|
| 1 · benchmark, random split | the published protocol | 35,371 | **0.996** |
| 2 · benchmark, domain-disjoint split | no eTLD+1 shared with training | 35,368 | **0.995** |
| 3 · live, benign in the benchmark's surface form | era changes, construction held | 600 | **0.987** |
| 4 · live, realistic benign | construction released | 568 | **0.000** |

TSS 0.000 is not degradation. The model labels **every** URL phishing — 300 true
positives, 268 false positives, zero true negatives. It has no discriminative
power whatsoever on live data.

### 3. The collapse is construction, not drift

Because each step changes exactly one thing, the drop can be attributed rather
than merely observed:

| step | ΔTSS | share of total |
|---|---|---|
| split leakage (cell 1 → 2) | 0.0001 | 0.0% |
| temporal / adversarial drift (cell 2 → 3) | 0.0087 | 0.9% |
| **benign construction artifact (cell 3 → 4)** | **0.9867** | **99.1%** |
| **total (cell 1 → 4)** | **0.9955** | 95% CI 0.994 – 0.997 |

The usual explanations fail. Domain memorisation across a random split costs
0.0001. Two years of attackers adapting costs 0.0087. What costs 0.9867 is
letting the benign class stop being a string template.

---

## Why this is the right way to measure it

Cell 3 is the load-bearing idea. A two-point benchmark-versus-live comparison
cannot tell a construction artifact apart from genuine drift — both look like
"the score fell." Holding the benign surface form fixed while changing the era,
then changing it back, separates them.

Other commitments, all made before live collection began
([PREREGISTRATION.md](PREREGISTRATION.md)):

- **One feature function for both sides.** Benchmark URLs and live URLs go
  through identical code. We use only PhiUSIIL's raw `URL` and `label` columns;
  its other 54 features are discarded because roughly half require fetching the
  page (impossible for live phishing URLs, which die within hours) and
  `URLSimilarityIndex` is computed against the dataset's own legitimate set,
  making it a leakage feature.
- **Thresholds frozen on benchmark validation data**, never on the data being
  scored, at two operating points (F1 and TSS objectives).
- **Confidence intervals clustered on registrable domain.** URLs sharing a host
  are not independent observations; i.i.d. resampling would make intervals
  several times too narrow.
- **TSS as the headline, not accuracy.** At an imbalanced base rate a constant
  forecast scores well on accuracy and has TSS exactly 0 by construction, so the
  zero line *is* the no-skill baseline.

## What we got wrong, and are reporting anyway

Our pre-registration committed to discounting the headline if a model given only
path-shape features could separate our own live classes above **TSS 0.30**. It
reaches **0.529**. That ceiling was breached and the result stands in
[RESULTS.md §6](RESULTS.md).

What it means, precisely: we cannot claim our OpenPhish and Common Crawl streams
are structurally equivalent, so we withhold any claim about how much genuine
phishing signal lives in path structure.

What it does not mean is that the collapse is our artifact, and the measured
direction is why. In the live corpus **92.2% of benign URLs carry a path against
61.7% of phishing URLs** — benign URLs are the structurally deeper ones, the
opposite of what the benchmark teaches. A model that learned "a path means
phishing" is not merely uninformed here, it is anti-correlated with reality,
which is exactly why its false-alarm rate is 1.000. Re-sampling our benign URLs
to be shallower would move them toward the benchmark's shape and flatter the
model, not penalise it. The reported collapse is a lower bound.

The same collapse reproduces on a different positive class: URLhaus malware
distribution URLs, same model, TSS 0.000 (n = 2,268).

---

## Reproducing

```bash
pip install -r requirements.txt

python -m phishdrift.cli audit     # finding 1 alone; ~30 s, no training
python -m phishdrift.cli train     # fit both models, freeze thresholds; ~3 min
python -m phishdrift.cli collect   # append one live snapshot
python -m phishdrift.cli report    # all cells -> results.json + RESULTS.md

python tests/test_features.py      # the invariants the study depends on
```

`RESULTS.md` is generated, never hand-edited. CI rebuilds every number from the
benchmark archive and the committed snapshots each week, and fails the build if
the construction-audit TSS moves from 0.9897.

## The live record

`.github/workflows/collect.yml` appends one immutable snapshot per UTC day to
`data/live/`: ~300 OpenPhish phishing URLs, ~300 Common Crawl benign URLs with
real paths sampled from Tranco top-50,000 domains, and the same domains rendered
in the benchmark's `https://www.<domain>` template form. Snapshots are committed
before they are ever scored, so the record is prospective by construction.

Verdicts for the hypotheses that depend on accumulated live data are scheduled
for **2026-12-15**, a date committed in advance so the analysis cannot stop at a
convenient moment.

## Layout

```
phishdrift/
  features.py     52 URL-lexical features; the one contract both sides share
  benchmark.py    PhiUSIIL loading, the two splits, the construction audit
  collect.py      daily live snapshots from OpenPhish / Common Crawl / Tranco
  model.py        training, threshold freezing, the zero-parameter baseline
  evaluate.py     TSS, Brier, threshold selection, domain-clustered bootstrap
  gap.py          the four cells and the attribution
docs/             a static results page (GitHub Pages)
tests/            invariant checks, run on every push
```

## Data sources

| | |
|---|---|
| Benchmark | [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset), UCI ML Repository (id 967) |
| Live phishing | [OpenPhish community feed](https://github.com/openphish/public_feed) |
| Live malware (robustness) | [URLhaus](https://urlhaus.abuse.ch/), abuse.ch |
| Benign URLs | [Common Crawl index](https://index.commoncrawl.org/) |
| Benign domain ranking | [Tranco](https://tranco-list.eu/) |

## Scope and limits

- **One benchmark.** The construction artifact documented here is PhiUSIIL's.
  Whether other phishing corpora share it is an open question and the obvious
  next test; a second benchmark would turn this from a case study into a claim
  about the field.
- **The benign stream is quasi-static.** Common Crawl publishes monthly, so
  benign URLs are fresh to within weeks while phishing URLs are fresh to within
  hours. Benign web structure moves far more slowly than phishing
  infrastructure, but the asymmetry is real and is recorded per row.
- **No claim of malice.** Corpus construction artifacts are common and usually
  accidental. The finding is about what a benchmark score can be taken to mean,
  not about the people who assembled it.

---

Part of a multi-domain study of the gap between benchmark and operational
machine-learning performance. Sibling projects measure the same question for
solar-flare forecasting (natural temporal drift) and ground-level ozone
exceedance (seasonal drift); this one covers adversarial drift and, as it turned
out, something rather more basic.
