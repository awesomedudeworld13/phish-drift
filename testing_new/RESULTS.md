# testing-new: retrospective results (scored once, 2026-09-23 UTC)

From `python -m phishdrift.retro_eval`; raw numbers in `retro_results.json`,
corpus in `retro/corpus.csv.gz` (108,000 rows; `retro/stats.json` per month).
Design: `PREREGISTRATION.md` (committed first) plus deviations D1 (Common
Crawl read over its CDN) and D2 (P2 validation windows).

**Corpus:** 36,000 dated OpenPhish URLs (1,500 per month, Oct 2024 to Sep 2026,
from the feed's own git history) against Common Crawl benign URLs from each
month's crawl. There are two benign samples: `realistic` and `path_matched`.

## P1: does training on live data close the gap? (realistic benign)

Live model (RF) trained on 2024-10-13 to 2025-10-31, validated Nov–Dec 2025,
tested 2026-01-01 to 2026-09-19 (19,215 URLs, 6,555 phishing; domains disjoint
from training). **D (live-trained on live test) = 0.731 (0.704–0.755).**

| benchmark | A: bench on bench | B: bench on live | C: live on bench | gap A−B | **recovery D−B (95% CI)** |
|---|---|---|---|---|---|
| PhiUSIIL | 0.995 | 0.001 | 0.860 | 0.995 | **+0.731 (0.704–0.754)** |
| Hannousse | 0.719 | 0.403 | 0.410 | 0.315 | **+0.328 (0.286–0.369)** |
| Kaitholikkal | 0.994 | 0.388 | 0.792 | 0.607 | **+0.344 (0.294–0.391)** |
| faizann24 | 0.802 | 0.148 | 0.062 | 0.654 | **+0.583 (0.543–0.623)** |

## P4: the same with path-matched benign (the audit-breach fix)

**D = 0.654 (0.624–0.683).** Recoveries: PhiUSIIL +0.651, Hannousse +0.260,
Kaitholikkal +0.311, faizann +0.485. **Every CI excludes zero.**

Collection audit (path shape alone), same test months:
- `realistic` benign: **0.584** (breached, as on main)
- `path_matched` benign: **0.274** (under the 0.30 ceiling)

## P2 (corrected, D2): decay and refreshing

Fixed model trained Oct–Nov 2024, then scored on each month from Jan 2025 to
Sep 2026. Refreshed model retrained every month on the 3 months before.
- **Fixed model slope:** −0.0042 TSS per month (95% CI −0.0064 to −0.0019).
  Over 21 months that's about −0.09, from ~0.79 to ~0.60–0.65.
- **Refreshed − fixed, averaged over the months:** **+0.045 (0.034 to 0.058).**

## P3: stronger learner

HGB D = 0.761 (0.737–0.783) vs RF 0.731. RF stays the primary learner as
pre-registered; HGB is reported alongside.

## Verdicts

| | Prediction | Result | Verdict |
|---|---|---|---|
| **H7 (primary)** | D−B > 0, CI excl. 0, all 4; D 0.60–0.90; D−B ≥ 0.30 | all 4 excl. 0; D 0.731; smallest D−B +0.328 | **Held** |
| H8 | C < A for all 4; C < A−0.10 for ≥3 | all 4, all by > 0.13 | **Held** |
| H9 (D2) | slope < 0, CI excl. 0; −0.005 to −0.03/month | −0.0042 (CI excl. 0) | **Held on direction**, magnitude just under the predicted band |
| H10 (D2) | refreshed > fixed, CI excl. 0; +0.02 to +0.10 | +0.045 (0.034–0.058) | **Held** |
| H11 | \|D_HGB − D_RF\| < 0.05 | 0.030 | **Held** |
| H12 | path-matched audit < 0.30 | 0.274 | **Held** |
| H13 | recovery survives path matching, all 4; D falls 0.05–0.25 | all 4 excl. 0; D fell 0.077 | **Held** |

## What it means

For phishing, the answer to research question 2 is **yes, clearly**. A model
trained on dated live data recovers most of the benchmark-to-operational gap
on every benchmark: +0.33 to +0.73 TSS on data from 2026 it never saw.

This isn't an artifact of our own collection. With benign URLs matched to
phishing on path shape, the audit passes (0.274) and the recovery still holds
for every benchmark.

It also decays. A model frozen in late 2024 loses about 0.004 TSS a month, and
retraining monthly on the last three months is worth about +0.045. That's the
adversarial-drift signature this domain was chosen for, and the contrast with
ozone (where recency mattered through thresholds) and solar (where the gap
closed mostly because the benchmark became realistic) is the cross-domain
result.

Minor: benign URLs for Oct 2024 are dated inside the Oct crawl window
(10-03 to 10-16), so a few fall before the phishing span's 10-13 start. They
sit in P1's training period and don't touch any test window.

**Prospective check (L1):** `python -m phishdrift.retro_eval --prospective`
scores final model F against the four benchmark models on the branch's
daily-collected data. Check on 2026-10-23 and 2026-11-22.
