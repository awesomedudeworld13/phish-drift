# Pre-registration: testing-new (retrospective live corpus, RQ2, collection fixes)

**Committed on 2026-09-22 (CDT) BEFORE the retrospective corpus was built,
before any model below was trained, and before the branch collector ran.**
This adds to `PREREGISTRATION.md` on main and does not replace it. Nothing below
may be edited once the first retrospective score exists; corrections go in the
dated "Deviations" section at the bottom.

## Why

Research question 2 (does training on live data close the gap?) is still open
on main: it needs 14 days of live collection and has 3. Two facts change that:

1. **OpenPhish's community feed is a git repository**
   (`github.com/openphish/public_feed`) whose history keeps every past version
   of `feed.txt`, twice a day, from 2024-10-12 on. The first commit that
   contains a URL dates it, independently of us: 378,062 unique phishing URLs
   over ~23 months. Main's collector saves one 300-URL snapshot a day and misses
   the rest.
2. **Common Crawl publishes one crawl per month** over the same span
   (CC-MAIN-2024-42 … CC-MAIN-2026-39), so benign URLs can be dated to the month
   with the collector's own sampling method.

The collection audit on main is **BREACHED** (path-shape-only peak TSS 0.516
against a 0.30 ceiling), so every new cell is also run on a path-matched benign
sample.

## Retrospective corpus (fixed now)

- **Span:** 2024-10-13 to 2026-09-19 UTC. Day 1 of the history (2024-10-12)
  is excluded because its URLs have unknown true first-seen dates. The corpus
  stops the day before main's prospective collection started (2026-09-20), so
  the two never overlap.
- **Phishing:** each URL's first-seen date is the committer date of the first
  history commit that contains it. Per calendar month, a seeded random sample
  of **1,500** (all of them if fewer). Seed 20260920.
- **Benign:** for each month, that month's Common Crawl crawl, queried
  exactly as `collect.fetch_commoncrawl_urls` does (random domains from the
  Tranco top-50,000, up to 25 successful-HTML URLs per domain,
  robots/sitemap excluded). Target pool **2,500 per month**. Each benign URL's
  date is a seeded uniform draw inside its crawl window.
  - `realistic` = seeded random **1,500** from the pool.
  - `path_matched` = **1,500** drawn from the pool to match that month's
    phishing sample on path-depth bucket {0, 1, 2, 3+} × has_query. Sampled
    with replacement within a bucket only if the pool runs short; the count of
    such draws is reported.
- Features, feature code and the domain rule (`features.registrable_domain`)
  are main's, unchanged.

## Analyses and hypotheses

Every score is TSS at a threshold frozen on validation data older than the
test data, with 95 % CIs from 1,000 domain-clustered bootstrap resamples
(main's `benchgap.evaluate`). Domains are kept disjoint across every temporal
boundary, as `livetrain.temporal_split` does. The learner is main's frozen
RandomForest unless stated.

### P1: the RQ2 2x2 on the retrospective corpus

Split by date: **fit** 2024-10-13 to 2025-10-31, **validation** 2025-11-01 to
2025-12-31, **test** 2026-01-01 to 2026-09-19. The 2x2 is main's `run_2x2`
cell definitions, run for each of the four benchmarks' domain-disjoint models.

- **H7 (primary).** Recovery D − B > 0 with the CI excluding 0, **for all four
  benchmarks**. *Predicted:* D between 0.60 and 0.90; D − B ≥ +0.30 for every
  benchmark.
- **H8.** The live-trained model does not inherit the benchmark: C (live model
  on each benchmark's test) is **below** A for every benchmark. *Predicted:*
  C < A − 0.10 for at least three of the four.

### P2: decay, and whether refreshing helps

- **Fixed model:** trained on 2024-10-13 to 2024-12-17 (validation
  2024-12-18 to 2024-12-31) and scored on each month 2025-01 … 2026-09.
- **Refreshed model:** for each test month *m*, trained on months m−3 to m−1
  (validation = the last 14 days of m−1) and scored on month *m*.
- **H9.** The fixed model's monthly TSS declines: the OLS slope over months
  elapsed is < 0 with the bootstrap CI excluding 0. *Predicted:* −0.005 to −0.03
  TSS per month.
- **H10.** Averaged over the same test months, refreshed TSS exceeds fixed TSS
  (paired, CI excluding 0). *Predicted:* +0.02 to +0.10.

### P3: stronger learner (reported, never swapped silently)

`HistGradientBoostingClassifier` (max_iter 300, learning_rate 0.1,
class_weight balanced, seed 20260920) on the same features and splits as P1.
- **H11.** |D_HGB − D_RF| < 0.05. RF stays the primary learner whatever the
  result.

### P4: the audit breach

- **H12.** On `path_matched` benign, main's path-shape-only diagnostic
  (`gap.sampling_confound_diagnostic`) on the P1 test months scores **< 0.30**.
- **H13.** P1's recovery D − B survives path matching: it stays > 0 with the CI
  excluding 0 for all four benchmarks. *Predicted:* D falls by 0.05–0.25 against
  `realistic` benign.

### P5: prospective confirmation (branch collector)

From 2026-09-23 the branch collects every OpenPhish feed update since its last
run, plus 600 realistic benign URLs a day, into `testing_new/live/`. Main's
collector is untouched. **Final model F** = the P1 recipe refit on the whole
retrospective corpus, frozen now.
- **L1.** At the 30-day (2026-10-23) and 60-day (2026-11-22) checks, F's TSS on
  branch-collected data minus each benchmark model's TSS on the same rows is
  > 0 with the CI excluding 0, for all four benchmarks.

## What would falsify the "live training closes the gap" claim

- H7 fails for any benchmark: live training does not reliably recover skill.
- H13 fails: the recovery was the collection confound, not phishing signal.
- L1 fails while H7 held: the retrospective result doesn't survive contact
  with the future.

All results are reported, including failures, in `testing_new/RESULTS.md`.

## Deviations

(none yet)
