# Pre-registration

Committed to git **before** live data collection began. The point of writing
this down first is that a result is only evidence if the prediction could have
been wrong. Everything below is falsifiable, and the verdicts section at the
bottom is filled in afterwards — including for predictions that fail.

Any change to this file after the first live snapshot exists must be an
*addition* recorded with its date, never an edit to a prediction.

---

## Research question

When a machine-learning model reports a high score on a public phishing-URL
benchmark, how much of that score is detection skill, and how much is an
artifact of how the benchmark corpus was assembled?

## Corpus and code state at registration

- **Benchmark:** PhiUSIIL Phishing URL Dataset (UCI ML Repository, id 967),
  235,795 URLs, 100,945 phishing / 134,850 legitimate.
  Only the `URL` and `label` columns are used; see `benchmark.py` for why the
  other 54 are discarded.
- **Live phishing:** OpenPhish community feed, collected daily from
  2026-09-20 onward.
- **Live benign:** Common Crawl index, sampled from Tranco top-50,000 domains
  (`commoncrawl`), and the same domains rendered as `https://www.<domain>`
  (`tranco_templated`).
- **Seed:** 20260920, fixed across all splits and bootstraps.
- **Model:** RandomForestClassifier, 300 trees, `min_samples_leaf=2`,
  `class_weight="balanced_subsample"`.
- **Features:** the 52 URL-lexical features in `features.FEATURE_NAMES`, frozen
  before collection. Keyword, brand, TLD and shortener lists are hard-coded
  constants and were **not** tuned against any test split or live sample.

## Frozen decisions

1. **Thresholds are selected on the benchmark validation partition only**, at
   two operating points (`f1`, `tss`), and then frozen. Live data is never used
   to choose a threshold.
2. **Confidence intervals cluster on registrable domain** (1,000 resamples,
   95% percentile). URLs sharing a host are not independent observations.
3. **TSS is the headline metric.** Accuracy is reported only to demonstrate its
   inadequacy: at an imbalanced base rate a constant forecast scores well on
   accuracy and has TSS exactly 0 by construction.
4. **Deduplication keeps the first appearance of a URL.** OpenPhish rotates a
   fixed-size window, so a long-lived campaign would otherwise be counted once
   per day it survives.
5. **Every daily snapshot is immutable** once written. A rerun does not
   overwrite an existing day.

---

## Hypotheses

**H1 — Construction artifact.**
A zero-parameter rule that tests only whether a URL matches the string template
`^https://www\.[^/?#]+/?$` will score above **TSS 0.90** on the full PhiUSIIL
corpus.

> *Verified before live collection, at registration time: TSS **0.9897**,
> accuracy 0.9956, precision 1.0000. 100.00% of the legitimate class matches the
> template; 1.03% of the phishing class does. Recorded here as a registered
> quantity rather than a prediction, because it is a property of a static
> published file and needed no new data.*

**H2 — Models learn the artifact.**
A RandomForest trained on the benchmark will place `is_https`, `has_www`, and
path-presence features among its five most important, i.e. it will rely on the
same surface properties the template rule uses.

> *Verified at training time, before live collection: the top five features are
> `is_https`, `n_slashes`, `path_length`, `has_www`, `subdomain_length` for both
> the random-split and the domain-disjoint model.*

**H3 — Split leakage is present but secondary.**
Moving from a random i.i.d. split to a domain-disjoint split will reduce
benchmark TSS by **more than 0.005 and less than 0.10**. Prediction: the
benchmark is separable on surface form alone, so domain memorisation is not the
main mechanism and this step should be small.

**H4 — Operational collapse.**
The domain-disjoint model, scored at its frozen TSS threshold against live
phishing versus *realistic* benign URLs (Common Crawl, with paths), will score
**below TSS 0.35**, versus above 0.90 on the benchmark.

**H5 — The collapse is mostly construction, not drift.**
Of the total drop from cell 1 to cell 4, the `benign_construction_artifact`
step (cell 3 → cell 4, which changes only the benign surface form and holds the
era fixed) will account for **more than half**.

**H6 — The template rule does not transfer.**
The same zero-parameter rule that scores TSS > 0.90 on the benchmark will score
**below TSS 0.20** on live data with realistic benign URLs.

## Predictions that would falsify the thesis

Stated explicitly so they cannot be quietly reinterpreted later:

- If H4 fails and live TSS lands above 0.70, the benchmark largely transfers and
  the project's central claim is wrong.
- If H5 fails and the construction step is small while the temporal step is
  large, then the collapse is ordinary concept drift and PhiUSIIL's degenerate
  benign class is a curiosity rather than the cause.
- If the sampling-confound diagnostic (`gap.sampling_confound_diagnostic`)
  returns **peak TSS above 0.30**, our own benign and phishing collection
  pipelines are structurally distinguishable, and the headline numbers are
  measuring our collection procedure rather than the world. This would require
  discounting or withdrawing the result, and will be reported either way.

## Reporting commitment

All cells that can be computed will be reported, including null and
contradicting results. If a hypothesis fails it stays in this file with its
verdict marked FAILED; it is not removed or rewritten. If additional benchmarks
are added later, every benchmark attempted is reported, not only those showing
an effect.

---

## Verdicts

*Filled in after the live record reaches the pre-committed reporting date.
Empty entries are pending, not omitted.*

| Hypothesis | Predicted | Observed | Verdict |
|---|---|---|---|
| H1 template rule TSS > 0.90 | > 0.90 | 0.9897 | **HELD** (registration-time) |
| H2 artifact features dominate | top-5 contains is_https, has_www, path | confirmed | **HELD** (training-time) |
| H3 leakage step in (0.005, 0.10) | small | _pending_ | _pending_ |
| H4 live TSS < 0.35 | < 0.35 | _pending_ | _pending_ |
| H5 construction > half of total drop | > 50% | _pending_ | _pending_ |
| H6 template rule live TSS < 0.20 | < 0.20 | _pending_ | _pending_ |
| Confound diagnostic < 0.30 | < 0.30 | _pending_ | _pending_ |

**Reporting date for H3–H6:** the live record is scored and these verdicts
filled in no earlier than **2026-12-15**, giving roughly three months of daily
collection. The date is committed here so the analysis cannot be stopped at a
moment that happens to look favourable.

---

# Addendum — 2026-09-20: second benchmark

**This is an addition, not a revision.** Nothing above has been edited. It is
recorded separately because of when it happened, which changes what it is worth.

## Disclosure of ordering

The second corpus was added **after** the PhiUSIIL results in H1–H2 were
observed. It is therefore **not a pre-registered prediction** and must not be
presented as one. The motivation was a specific weakness in the original design,
raised as a limitation in the first version of the README: a single corpus can
only show that *one* dataset is flawed, and cannot distinguish that from the
field being uniformly flawed. Only a second corpus separates those.

We state the ordering plainly because the finding it produced happens to be
favourable to the project's framing, and a reader is entitled to discount it
accordingly.

## What was fixed in place

- **Corpus:** Hannousse, A. & Yahiouche, S. (2021), *Web page phishing
  detection*, Mendeley Data doi:10.17632/c2gw7fy2j4.3, `dataset_B_05_2020.csv`,
  11,430 URLs, balanced 50/50. CC BY 4.0.
- **Committed bytes.** The copy in `data/benchmarks/` was verified against the
  SHA-256 Mendeley publishes (`21093e29…`) before being added, and CI re-checks
  it on every push. The corpus cannot be silently swapped for a more favourable
  version.
- **The audit rule is not re-tuned.** `BENIGN_TEMPLATE` was derived by
  inspecting PhiUSIIL and is applied to Hannousse **unchanged**. Fitting a
  bespoke rule per corpus would make the cross-corpus comparison meaningless,
  and would guarantee a high score on any dataset by construction.
- **Everything else is held constant**: the same 52 features, the same seed
  (20260920), the same splits, the same frozen-threshold discipline, the same
  live snapshots.

## Registered before running the cells

These *were* written down before the four-cell report was run on Hannousse
(the audit in A1 had already been computed; the live cells had not):

**A1 — Not all corpora are degenerate.** The template rule will score
**below TSS 0.50** on Hannousse.
> *Observed before the cells were run: **0.1608**, with no surface property
> degenerate on either side. **HELD**.*

**A2 — The healthy corpus retains operational skill.** Hannousse's live TSS
(cell 4) will be **above 0.15**, versus PhiUSIIL's 0.000.

**A3 — Split leakage is larger where the construction artifact is smaller.**
PhiUSIIL's cells 1→2 step is 0.0001 because surface form swamps everything
else; on a corpus without that shortcut, domain memorisation should become
visible, and the step will exceed **0.02**.

## Correction to the published method

Running the second corpus exposed an error in the original decomposition, which
is recorded here rather than quietly patched.

Cell 3 was defined as "live feeds, benign held in the benchmark's own surface
form", and the three-step attribution assumes that. That assumption is only true
for a corpus whose benign class *is* the template. Hannousse's is not, so for it
cell 3 *changes* construction rather than holding it — and its cell 3 duly came
out above its cell 2, which would have published a nonsensical negative "drift"
step had it been reported as one.

`gap.py` now computes each corpus's `benign_template_share` and publishes the
full three-step attribution only when that share is ≥ 0.90, withholding the
middle two steps otherwise with a stated reason. Split leakage and the total gap
are unaffected, as neither depends on cell 3. The guard is data-driven, not
hard-coded per dataset, so a future corpus is handled on its measured properties.

## Verdicts — addendum

| Hypothesis | Predicted | Observed | Verdict |
|---|---|---|---|
| A1 template rule < 0.50 on a second corpus | < 0.50 | 0.1608 | **HELD** |
| A2 healthy corpus live TSS > 0.15 | > 0.15 | 0.298 | **HELD** |
| A3 split leakage > 0.02 there | > 0.02 | 0.088 | **HELD** |

All three are single-corpus observations at n = 11,430 with live cells resting
on one daily snapshot (n = 568). They are reported now and re-scored at the
2026-12-15 date committed above.

---

# Addendum 2 — 2026-09-20: third and fourth corpora

Again an addition, not a revision. Again added after seeing earlier results, and
again disclosed as such.

## Corpora

- **Kaitholikkal, J. K. S. & Arthi, B. (2024)**, *Phishing URL dataset*,
  Mendeley Data doi:10.17632/vfszbj9b36.1, 450,176 URLs. CC BY 4.0, vendored and
  hash-verified (`accb2dfb…`). Legitimate URLs from the Majestic Million,
  phishing from PhishTank.
- **faizann24**, *Using machine learning to detect malicious URLs*, GitHub,
  420,464 URLs. **Weaker provenance than the other three** — a community
  repository with no accompanying paper and an unspecified licence. Included
  because it is among the most-copied phishing URL datasets in circulation, so
  what it teaches is worth measuring; not redistributed, only fetched. Its
  status is stated wherever it is reported.

## A method error the third corpus exposed

This is the substantive part of this addendum, and it goes against the earlier
work rather than extending it.

The original audit tested whether a surface property was **exactly** 0.0 or 1.0
on the benign side. Kaitholikkal's `is_https` is **0.9999 benign vs 0.0621
phishing**, and `has_www` is 0.9980 vs 0.1304. Under the original rule both were
reported as `ok`. They are not remotely ok: a separation of 0.9378 means that
single boolean, used as the entire classifier, achieves TSS 0.9378 — better than
almost anything published on the corpus.

So the original detector would have **passed a corpus that one boolean solves**,
purely because a rate was 0.9999 rather than 1.0000.

The audit now reports two independent zero-parameter baselines:

| baseline | what it catches | PhiUSIIL | Kaitholikkal |
|---|---|---|---|
| template rule (conjunction) | no benign row deviates → perfect precision | **0.9897** | 0.1357 |
| best single property | one boolean nearly solves it | 0.5865 | **0.9378** |

Each corpus fails a *different* one. Neither baseline alone is sufficient, and
had the study stopped at two corpora the second pathology would never have been
found. This is recorded as a correction because the earlier single-detector
result was incomplete, not merely extensible.

`separation` is defined as `|P(prop|benign) − P(prop|phishing)|`, which is
exactly the TSS of that one property as a classifier. Verdict thresholds
(`degenerate`, `high_separation` ≥ 0.70, `elevated` ≥ 0.35, `constant`) are fixed
in code and applied identically to every corpus.

A second, smaller fix: `structural_degeneracy` parsed URLs with raw `urlsplit`
while feature extraction used a defensive parser, so the audit crashed on
malformed URLs that the model handles. It now shares the same parser, so the
audit describes the same corpus the model is trained on.

## Registered before running the cells

**B1 — Construction problems are common but not universal.** Across all four
corpora, at least one will be clean on both baselines (max < 0.35) and at least
two will fail at least one baseline (≥ 0.70).
> *Observed at audit time: PhiUSIIL fails the template rule (0.9897),
> Kaitholikkal fails single-property (0.9378), Hannousse is clean on both
> (0.1608 / 0.4625), faizann24 is clean on both but has `is_https` constant.
> **HELD**.*

**B2 — Audit severity predicts operational collapse.** Ranking the four corpora
by `max(template_rule, best_single_property)` will produce the same ordering as
ranking them by total benchmark-to-live gap, for the corpora where the gap is
defined.

**B3 — The constant-property corpus underperforms on the benchmark itself.**
faizann24, having destroyed the scheme signal, will score a *lower* cell-1
benchmark TSS than the corpora that leak it.

## Verdicts — addendum 2

| Hypothesis | Predicted | Observed | Verdict |
|---|---|---|---|
| B1 common but not universal | ≥1 clean, ≥2 failing | 2 clean, 2 failing | **HELD** |
| B2 audit severity predicts gap ordering | same ordering | top 2 match, bottom 2 invert | **PARTIAL** |
| B3 constant-property corpus scores lower on cell 1 | lower | lower than both leaking corpora, higher than the clean one | **PARTIAL** |

**B2 in detail — this one only half worked and the failure is reported as
such.** Ranking by audit severity `max(template_rule, best_single_property)`
gives PhiUSIIL (0.9897) > Kaitholikkal (0.9378) > Hannousse (0.4625) >
faizann24 (0.0600). Ranking by total benchmark-to-live gap gives PhiUSIIL
(0.996) > Kaitholikkal (0.922) > faizann24 (0.569) > Hannousse (0.508).

The two corpora that fail an audit do have the two largest gaps, and by a wide
margin — that part held. But the ordering **inverts** between the two corpora
that pass: faizann24 audits cleaner than Hannousse yet has the slightly larger
gap (0.569 vs 0.508, a difference small enough that one snapshot of live data
cannot resolve it). So the audits predict *whether* a corpus will collapse, not
*how much* it will drift once it is clean. The stronger claim — that audit
severity is a graded predictor of gap size — is **not** supported by these four
corpora and is not made.

**B3 in detail.** faizann24's cell-1 benchmark TSS is 0.891, below both corpora
that leak surface form (PhiUSIIL 0.996, Kaitholikkal 0.993), which is the
predicted direction. But it is *above* Hannousse's 0.807, so "destroying the
scheme signal lowers benchmark score" is true only relative to corpora that leak
it. Recorded as partial rather than held.

Four corpora is enough to show that construction quality varies and that it is
not rare. It is **not** enough to estimate what fraction of published phishing
benchmarks are affected, and no such figure is claimed.

---

# Addendum 3 — 2026-09-20: the second research question

A structural omission, recorded because it changes what the repository claims to
answer.

The study has always had two questions:

1. How far does benchmark performance fall on live data?
2. **Does training on live data close that gap?**

Everything above answers only question 1. Every model in the four cells was
fitted on a benchmark corpus, so none of them can say whether the loss is a
training-data problem or something training cannot fix. Without question 2, a
reader could reasonably conclude "collect more live data" is the remedy, which
the evidence so far does not support either way.

`livetrain.py` now builds a model from the accumulated live snapshots alone and
scores the training-source × test-set 2×2:

|  | benchmark test | live holdout |
|---|---|---|
| benchmark-trained | A | B |
| live-trained | C | D |

- **A − B** is the gap.
- **D − B** answers question 2: how much live training recovers.
- **C** is the control — a live-trained model that also scores well on the
  benchmark has learned something general, not this month's campaigns.

**Frozen before any live-trained model exists:**

- Live rows split **temporally**, earliest days training, latest days holdout.
  A random split over live data would let a model memorise a campaign from one
  day and be tested on the same campaign days later — the exact leakage this
  project exists to measure.
- The validation slice used to freeze thresholds is taken from the **end of the
  training period**, not at random, so the operating point never sees data
  newer than the holdout.
- Domains are held disjoint across the cutoff; where one appears on both sides,
  its later rows are dropped.
- Templated benign rows are **excluded** from live training. They exist to
  isolate a construction artifact; training on them would teach a live model
  the very template whose effect is being removed.
- Minimum data before the question is answered at all: 10 training days, 4
  holdout days, ≥400 rows and ≥100 phishing rows per side. Below that the
  report states the question as open and names how many days remain.

**C1 — Live training does not fully close the gap.** Recovery (D − B) will be
less than the gap (A − B) for every corpus.

**C2 — Live training helps more where the gap is construction than where it is
drift.** Recovery as a fraction of the gap will be larger for PhiUSIIL and
Kaitholikkal (which collapse) than for Hannousse (which does not).

Status at registration: **1 of 14 days collected, 13 remaining.** No
live-trained model has been fitted. These predictions are recorded before any
exists, which is the only reason they count.
