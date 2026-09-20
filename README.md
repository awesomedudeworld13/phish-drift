# phish-drift

This project asks two questions about machine-learning phishing detection:

1. **How far does benchmark performance fall on live data?**
2. **Does training on live data close that gap?**

Four public phishing-URL corpora, one identical pipeline, scored against live
phishing feeds collected daily. The answer to (1) turns out to be *entirely
different for each corpus* — and cheap, label-free checks predict which kind you
are holding before any model is trained. Question (2) is **open**: it needs a
model trained on live snapshots, and the live record is still accumulating
(see [§5](#2-does-training-on-live-data-close-the-gap)).

---

## 1. How far does performance fall?

| corpus | n | template rule | best single property | degenerate | constant | benchmark TSS | **live TSS** | gap |
|---|---|---|---|---|---|---|---|---|
| PhiUSIIL (UCI, 2024) | 235,795 | **0.9897** | 0.5865 | all four | — | 0.996 | **0.000** | 0.996 |
| Kaitholikkal & Arthi (2024) | 450,176 | 0.1357 | **0.9378** | `is_https` `has_www` | — | 0.993 | **0.072** | 0.922 |
| faizann24 (community) | 420,464 | 0.0000 | 0.0600 | — | `is_https` | 0.891 | **0.322** | 0.569 |
| Hannousse & Yahiouche (2021) | 11,430 | 0.1608 | 0.4625 | — | — | 0.807 | **0.298** | 0.508 |

Two **zero-parameter baselines**, because they catch different pathologies and a
corpus can pass one while failing the other:

- **Template rule** — the conjunction *"does this URL fail to match
  `^https://www\.[^/?#]+/?$`? then call it phishing"*. Scores high when **no**
  benign row deviates, so combining surface properties identifies the benign
  class with perfect precision.
- **Best single property** — the largest
  `|P(prop|benign) − P(prop|phishing)|`, which is exactly the TSS of that one
  boolean used as the entire classifier.

**Neither alone is sufficient.** The template rule scores 0.9897 on PhiUSIIL but
only 0.1357 on Kaitholikkal — which would clear the latter entirely, while
`is_https` alone (0.9999 benign vs 0.0621 phishing) solves it at 0.9378.
Conversely PhiUSIIL's best single property is just 0.5865; its pathology exists
only in the conjunction. We found this because the third corpus broke the
first detector — see [PREREGISTRATION.md addendum 2](PREREGISTRATION.md).

On PhiUSIIL the template rule scores accuracy 0.9956, precision 1.0000 —
**better than the ~99.2% accuracy published models report on the same data.**

### Why PhiUSIIL behaves that way

Every one of its 134,850 legitimate URLs matches that template: https,
`www.`-prefixed, no path, no query. Not 99%. All of them.

| surface property | legitimate | phishing |
|---|---|---|
| `is_https` | **1.0000** | 0.4874 |
| `has_www` | **1.0000** | 0.4135 |
| `has_path` | **0.0000** | 0.2719 |
| `has_query` | **0.0000** | 0.0602 |

A rate of exactly 0 or 1 on one side means the property is not a feature. It is
the class label, written in a different alphabet.

The other three fail differently, or not at all. **Kaitholikkal** is not exactly
extreme — 0.9999 and 0.9980 — so the template rule clears it, but those single
booleans separate the classes at 0.9378 and 0.8676 on their own. Its legitimate
URLs come from the Majestic Million and its phishing from PhishTank, so the
benign side is normalised to https+www while phishing keeps its original scheme.
**faizann24** strips schemes entirely (0.22% of its URLs carry one), so
`is_https` is 0.0000 vs 0.0001 — *constant*, not degenerate. That destroys the
information rather than leaking it, and means the corpus cannot teach a model a
signal real deployments have. **Hannousse** overlaps healthily throughout
(0.445/0.333, 0.668/0.205, 0.613/0.802, 0.033/0.237).

Models trained on each corpus reflect this. On PhiUSIIL the RandomForest's top
features are `is_https`, `n_slashes`, `path_length`, `has_www` — the template.
On Hannousse they are `has_www`, `subdomain_length`, `digit_ratio`,
`suspicious_keyword_count`; on faizann24, `tld_is_common`, `tld_length`,
`path_length`, `suspicious_keyword_count`. Those last ones are actual phishing
signals.

### What happens on live data

Each model is scored at a threshold frozen on its own benchmark's validation
split, against live OpenPhish URLs and real benign URLs drawn from Common Crawl:

| cell | what changes | PhiUSIIL | Kaitholikkal | faizann24 | Hannousse |
|---|---|---|---|---|---|
| 1 · benchmark, random split | the published protocol | 0.996 | 0.993 | 0.891 | 0.807 |
| 2 · benchmark, domain-disjoint split | no eTLD+1 shared with training | 0.995 | 0.994 | 0.802 | 0.719 |
| 3 · live, benign in template form | *(see caveat below)* | 0.987 | 0.847 | −0.133 | 0.907 |
| 4 · live, realistic benign | operational reality | **0.000** | **0.072** | **0.322** | **0.298** |

PhiUSIIL's 0.000 is not degradation. The model labels **every** URL phishing —
300 true positives, 268 false positives, zero true negatives. Even with a
hindsight-optimal threshold it reaches only 0.116, so this is not a threshold
problem. The same collapse reproduces on a different positive class entirely
(URLhaus malware URLs, TSS 0.000, n = 2,268).

The two corpora that fail an audit collapse (0.000 and 0.072). The two that pass
lose roughly half and **keep the rest** (0.322 and 0.298). That is what an honest
benchmark-to-operational gap looks like.

### Attribution, and where it legitimately applies

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
the era changes" for a corpus whose benign class *is* the template. Only
PhiUSIIL's is (100% match). The other three match at 13–16%, so rendering live
benign URLs in template form *changes* their construction instead of holding it
— which is why Hannousse's cell 3 lands *above* its cell 2, and why faizann24's
goes negative. The three-step decomposition is therefore computed and published
**only for PhiUSIIL**. For the rest the code withholds the middle two steps
automatically and states why, reporting only split leakage and the total gap,
neither of which depends on cell 3.

That guard matters in both directions: split leakage is 0.0001 on PhiUSIIL,
where surface form swamps everything, but **0.088 on Hannousse and 0.089 on
faizann24** — real domain memorisation that the broken corpus had hidden. The
guard lives in `gap.py`, keyed off each corpus's measured
`benign_template_share`, not hard-coded per dataset.

## 2. Does training on live data close the gap?

**Open — and reported as open rather than omitted.**

Every model above was fitted on a benchmark corpus, so none of them can say
whether the loss is a *training-data* problem or something training cannot fix.
Answering that needs a model the benchmark never touched, scored in a
training-source × test-set 2×2:

|  | benchmark test | live holdout |
|---|---|---|
| **benchmark-trained** | A | B |
| **live-trained** | C | D |

`A − B` is the gap. **`D − B` is the answer** — how much live training recovers.
`C` is the control: a live-trained model that also scores well on the benchmark
has learned something general, not this month's campaigns.

`livetrain.py` builds this automatically once the record is long enough. It
needs 10 training days and 4 holdout days, split **temporally** (earliest days
train, latest test) because a random split over live data would let a model
memorise a campaign from one day and be tested on it days later — exactly the
leakage this project exists to measure. As of the latest run: **1 day collected,
13 remaining.** Predictions are committed in
[PREREGISTRATION.md addendum 3](PREREGISTRATION.md), written before any
live-trained model exists.

## What this shows so far

Not "benchmarks overstate performance" — that would be a single-corpus result
and a weak one. Four corpora, one pipeline, show that **the size of the
benchmark-to-operational gap varies enormously with how a corpus was assembled**,
from total collapse to a survivable halving. And the cheapest possible
diagnostics — surface-form checks, no training, no labels, seconds to run —
predict which you are holding.

**With a limit we state rather than bury.** The audits predict *whether* a
corpus will collapse: the two that fail an audit have by far the two largest
gaps. They do **not** predict *how much* a clean corpus will drift — faizann24
audits cleaner than Hannousse yet has the slightly larger gap (0.569 vs 0.508).
That hypothesis was registered in advance and came out **partial**; it is
recorded as such.

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

- **Four corpora is enough to show variance, not enough to estimate a rate.** We
  can say construction quality differs enormously between datasets and that
  problems are not rare; we cannot say what fraction of published phishing
  benchmarks are affected, and no such figure is claimed.
- **Question 2 is unanswered.** Until the live record supports a temporal split,
  this repository measures the gap but says nothing about whether live training
  closes it. Do not read the collapse as "collect more live data".
- **faizann24 has weaker provenance** than the other three: a community GitHub
  repository, no accompanying paper, unspecified licence. Included for its wide
  reuse, flagged wherever reported, and not redistributed.
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
