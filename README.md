# phish-drift

**How much of a phishing-detection benchmark score is detection skill, and how
much is an artifact of how the dataset was built?**

Two public phishing-URL corpora, one identical pipeline, scored against live
phishing feeds. The answer turns out to be *entirely different for each* — and
you can tell which kind you are holding with a regular expression, before
training anything.

---

## The headline

| corpus | n | template-rule TSS | benign matching template | degenerate properties | benchmark TSS | **live TSS** | total gap |
|---|---|---|---|---|---|---|---|
| PhiUSIIL (UCI, 2024) | 235,795 | **0.9897** | 100.00% | `is_https` `has_www` `has_path` `has_query` | 0.996 | **0.000** | 0.996 |
| Hannousse & Yahiouche (2021) | 11,430 | **0.1608** | 16.47% | none | 0.807 | **0.298** | 0.508 |

The "template rule" is a single regular expression with **zero learned
parameters**, applied unchanged to both corpora:

> *Does this URL fail to match `^https://www\.[^/?#]+/?$`? Then call it phishing.*

On PhiUSIIL that rule scores accuracy 0.9956, precision 1.0000, TSS 0.9897 —
**better than the ~99.2% accuracy published models report on the same data.**
On Hannousse the identical rule is near-worthless.

## Why PhiUSIIL behaves that way

Every one of its 134,850 legitimate URLs matches that template: https,
`www.`-prefixed, no path, no query. Not 99%. All of them.

| surface property | legitimate | phishing |
|---|---|---|
| `is_https` | **1.0000** | 0.4874 |
| `has_www` | **1.0000** | 0.4135 |
| `has_path` | **0.0000** | 0.2719 |
| `has_query` | **0.0000** | 0.0602 |

A rate of exactly 0 or 1 on one side means the property is not a feature. It is
the class label, written in a different alphabet. Hannousse's equivalents all
overlap (0.445/0.333, 0.668/0.205, 0.613/0.802, 0.033/0.237) — nothing
degenerate.

Models trained on each corpus reflect this. On PhiUSIIL the RandomForest's top
features are `is_https`, `n_slashes`, `path_length`, `has_www` — the template.
On Hannousse they are `has_www`, `subdomain_length`, `digit_ratio`,
`suspicious_keyword_count` — that last one is an actual phishing signal.

## What happens on live data

Each model is scored at a threshold frozen on its own benchmark's validation
split, against live OpenPhish URLs and real benign URLs drawn from Common Crawl:

| cell | what changes | PhiUSIIL | Hannousse |
|---|---|---|---|
| 1 · benchmark, random split | the published protocol | 0.996 | 0.807 |
| 2 · benchmark, domain-disjoint split | no eTLD+1 shared with training | 0.995 | 0.719 |
| 3 · live, benign in template form | *(see caveat below)* | 0.987 | 0.907 |
| 4 · live, realistic benign | operational reality | **0.000** | **0.298** |

PhiUSIIL's 0.000 is not degradation. The model labels **every** URL phishing —
300 true positives, 268 false positives, zero true negatives. Even with a
hindsight-optimal threshold it reaches only 0.116, so this is not a threshold
problem. The same collapse reproduces on a different positive class entirely
(URLhaus malware URLs, TSS 0.000, n = 2,268).

Hannousse loses about half its skill and **keeps the rest**. That is what an
honest benchmark-to-operational gap looks like.

## Attribution, and where it legitimately applies

For PhiUSIIL each step changes exactly one thing, so the drop can be attributed
rather than merely observed:

| step | ΔTSS | share |
|---|---|---|
| split leakage (cell 1 → 2) | 0.0001 | 0.0% |
| temporal / adversarial drift (cell 2 → 3) | 0.0087 | 0.9% |
| **benign construction artifact (cell 3 → 4)** | **0.9867** | **99.1%** |
| **total (cell 1 → 4)** | **0.9955** | 95% CI 0.994 – 0.997 |

Domain memorisation across a random split costs 0.0001. Years of attackers
adapting cost 0.0087. Letting the benign class stop being a string template
costs 0.9867.

**The caveat, and it matters.** Cell 3 only means "hold construction fixed while
the era changes" for a corpus whose benign class *is* the template. Hannousse's
is not — its benign URLs carry paths, http and no `www` — so rendering live
benign URLs in template form *changes* its construction instead of holding it,
and its cell 3 comes out *above* its cell 2. The three-step decomposition is
therefore computed and published **only for PhiUSIIL**; for Hannousse the code
withholds the middle two steps automatically and reports only split leakage
(0.088 — real domain memorisation, which PhiUSIIL's template swamped) and the
total gap (0.508). Neither of those depends on cell 3. The guard is in
`gap.py`, keyed off each corpus's measured `benign_template_share`, not
hard-coded per dataset.

## What this actually shows

Not "benchmarks overstate performance" — that would be a single-corpus result
and a weak one. Two corpora, one pipeline, show that **the size of the
benchmark-to-operational gap varies enormously with how a corpus was assembled**,
from a total collapse to a survivable halving. And the cheapest possible
diagnostic — a regex over surface form, no training, no labels, seconds to run —
predicts which you are holding.

That makes the deliverable a check anyone can run on their own corpus before
trusting a number, rather than an accusation against one dataset.

---

## Method commitments

All made before live collection began ([PREREGISTRATION.md](PREREGISTRATION.md)):

- **One feature function for every corpus and for live data.** We take only the
  raw `url` and label columns from each benchmark and recompute our own 52
  lexical features, so benchmark rows and live rows traverse identical code.
  Both corpora ship precomputed features we discard: roughly half require
  fetching the page (impossible for live phishing URLs, dead within hours), and
  PhiUSIIL's `URLSimilarityIndex` is computed against its own legitimate set,
  making it a leakage feature.
- **Thresholds frozen on benchmark validation data**, never on the data being
  scored, at two operating points (F1 and TSS objectives).
- **Confidence intervals clustered on registrable domain.** URLs sharing a host
  are not independent; i.i.d. resampling would make intervals several times too
  narrow.
- **TSS as the headline, not accuracy.** At an imbalanced base rate a constant
  forecast scores well on accuracy and has TSS exactly 0 by construction, so the
  zero line *is* the no-skill baseline.
- **The audit rule is applied unchanged to every corpus.** It was derived by
  inspecting PhiUSIIL, but it is never re-tuned per dataset — which is what
  makes the cross-corpus numbers comparable.

## What we got wrong, and are reporting anyway

Our pre-registration committed to discounting the headline if a model given only
path-shape features could separate our own live classes above **TSS 0.30**. It
reaches **0.529**. That ceiling was breached and it stands in
[RESULTS.md §6](RESULTS.md) and on the dashboard, flagged.

What it means precisely: we cannot claim our OpenPhish and Common Crawl streams
are structurally equivalent, so we withhold any claim about how much genuine
phishing signal lives in path structure.

What it does not mean is that the collapse is our artifact, and the measured
direction is why. In the live corpus **92.2% of benign URLs carry a path against
61.7% of phishing URLs** — benign URLs are the structurally deeper ones, the
opposite of what PhiUSIIL teaches. A model that learned "a path means phishing"
is not merely uninformed here, it is anti-correlated with reality, which is
exactly why its false-alarm rate is 1.000. Re-sampling our benign URLs to be
shallower would move them toward the benchmark's shape and flatter the model,
not penalise it. The reported collapse is a lower bound.

---

## Reproducing

```bash
pip install -r requirements.txt

python -m phishdrift.cli audit                    # the regex finding, both corpora, ~1 min
python -m phishdrift.cli train                    # fit 4 models, freeze thresholds; ~4 min
python -m phishdrift.cli collect                  # append one live snapshot
python -m phishdrift.cli report                   # all cells -> results/ + RESULTS.md

python -m phishdrift.cli audit --benchmark hannousse   # one corpus only
python tests/test_features.py                     # the invariants the study depends on
```

`RESULTS.md` is generated, never hand-edited. CI rebuilds every number weekly
from the corpora and the committed snapshots, and fails the build if either
corpus's construction-audit TSS moves.

## Data provenance

| | |
|---|---|
| PhiUSIIL | [UCI ML Repository id 967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) — downloaded at runtime |
| Hannousse & Yahiouche | [Mendeley Data 10.17632/c2gw7fy2j4.3](https://data.mendeley.com/datasets/c2gw7fy2j4/3), CC BY 4.0 — **committed** to `data/benchmarks/` |
| Live phishing | [OpenPhish community feed](https://github.com/openphish/public_feed) |
| Live malware (robustness) | [URLhaus](https://urlhaus.abuse.ch/), abuse.ch |
| Benign URLs | [Common Crawl index](https://index.commoncrawl.org/) |
| Benign domain ranking | [Tranco](https://tranco-list.eu/) |

Mendeley sits behind bot protection that rejects Python's TLS fingerprint with a
403 while serving `curl` normally, so an unattended CI download is not
dependable. CC BY 4.0 permits redistribution with attribution, so the exact
bytes are committed — and verified against the SHA-256 Mendeley publishes
(`21093e29…`) on every CI run, so the committed copy is provably the upstream
file rather than merely claimed to be.

## The live record

`.github/workflows/collect.yml` appends one immutable snapshot per UTC day to
`data/live/`: ~300 OpenPhish phishing URLs, ~300 Common Crawl benign URLs with
real paths sampled from Tranco top-50,000 domains, and the same domains rendered
in template form. Snapshots are committed before they are ever scored, so the
record is prospective by construction.

Verdicts for hypotheses depending on accumulated live data are scheduled for
**2026-12-15**, committed in advance so the analysis cannot stop at a convenient
moment.

## Layout

```
phishdrift/
  features.py     52 URL-lexical features; the one contract every corpus shares
  benchmark.py    the corpus registry, the two splits, the construction audit
  collect.py      daily live snapshots from OpenPhish / Common Crawl / Tranco
  model.py        training, threshold freezing, the zero-parameter baseline
  evaluate.py     TSS, Brier, threshold selection, domain-clustered bootstrap
  gap.py          the four cells, the attribution, and its applicability guard
data/benchmarks/  committed corpora, hash-verified against upstream
docs/             a static results page (GitHub Pages)
```

## Scope and limits

- **Two corpora is enough to show variance, not enough to estimate a rate.** We
  can say the gap differs enormously between datasets; we cannot say what
  fraction of published phishing benchmarks are degenerate.
- **The benign stream is quasi-static.** Common Crawl publishes monthly, so
  benign URLs are fresh to within weeks while phishing URLs are fresh to within
  hours. Benign web structure moves far more slowly than phishing
  infrastructure, but the asymmetry is real and recorded per row.
- **Hannousse's live cells rest on one snapshot** at time of writing (n = 568).
  That grows by ~600 rows daily.
- **No claim of malice.** Corpus construction artifacts are common and usually
  accidental. The finding is about what a benchmark score can be taken to mean,
  not about the people who assembled it.

---

Part of a multi-domain study of the gap between benchmark and operational
machine-learning performance. Sibling projects measure the same question for
solar-flare forecasting (natural temporal drift) and
[ground-level ozone](https://github.com/awesomedudeworld13/ozone-drift)
(seasonal drift, the control); this one covers adversarial drift and, as it
turned out, something rather more basic.
