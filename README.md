# phish-drift

**Question: when a phishing detector scores 99% on a public dataset, is it actually good at catching phishing?**

For one popular 2024 dataset, no. **A three-line regular expression — no machine learning at all — beats the published results.** It isn't detecting phishing. It's detecting a formatting quirk in how the dataset was assembled.

---

## The short version

We tested four public phishing-URL datasets the same way, then checked each trained model against **live phishing URLs collected daily**.

Two of the four are broken in ways that make them nearly useless, and the failures are invisible unless you go looking:

| Dataset | Can a regex solve it? | Can one yes/no check solve it? | Score on the dataset | **Score on live URLs** |
|---|---|---|---|---|
| PhiUSIIL (2024) | **yes, 0.99** | 0.59 | 0.996 | **0.000** |
| Kaitholikkal & Arthi (2024) | no, 0.14 | **yes, 0.94** | 0.993 | **0.072** |
| faizann24 (community) | no, 0.00 | no, 0.06 | 0.891 | **0.322** |
| Hannousse & Yahiouche (2021) | no, 0.16 | no, 0.46 | 0.807 | **0.298** |

The two datasets that fail a cheap check collapse completely on real data. The two that pass lose about half their skill and **keep the rest** — which is what a normal, honest result looks like.

The useful takeaway isn't "benchmarks lie." It's that **you can tell which kind of dataset you're holding in about ten seconds, before training anything.**

---

## Background: what the numbers mean

**TSS** (True Skill Statistic) is the score used throughout:

> TSS = (fraction of real phishing you caught) − (fraction of safe sites you falsely flagged)

**TSS is 0 for a model with no skill**, and 1 for a perfect one. We use it instead of accuracy because accuracy is misleading whenever one outcome is rare — a model that calls everything safe can score 95% accuracy while catching zero phishing.

**"Live URLs"** means real phishing links reported by [OpenPhish](https://github.com/openphish/public_feed), collected automatically every day, paired with real safe URLs pulled from [Common Crawl](https://index.commoncrawl.org/) (a public archive of the actual web).

---

## Finding 1: a regex beats the published models

PhiUSIIL contains 235,795 URLs and is hosted by the UC Irvine Machine Learning Repository, a standard source for this kind of data. Published models report around **99.2% accuracy** on it.

Here is the problem. Every single one of its 134,850 "safe" URLs looks like this:

```
https://www.example.com
```

That is: `https`, a `www.` prefix, and nothing after the domain. No page path, no question marks. Not most of them — **all 134,850 of them.** The phishing URLs, pulled from a live feed, match that shape only 1% of the time.

So you can "solve" the dataset with a rule that involves no learning whatsoever:

> *Does this URL fail to match `https://www.something`? Then call it phishing.*

| URLs | Accuracy | Of the ones it flagged, how many were phishing | TSS |
|---|---|---|---|
| 235,795 | **99.56%** | **100%** | **0.9897** |

**99.56% accuracy, zero false alarms, from one line of text-matching.** That beats the machine-learning papers published on the same data.

Those models were never learning about phishing. They were learning that the safe examples were collected by a script that wrote them all in the same format.

Here is the same thing in table form. Each row is a simple yes/no property of a URL:

| Property | Safe URLs | Phishing URLs |
|---|---|---|
| uses `https` | **100.00%** | 48.74% |
| starts with `www.` | **100.00%** | 41.35% |
| has a page path | **0.00%** | 27.19% |
| has a query string | **0.00%** | 6.02% |

When a property is **exactly** 0% or 100% on one side, it has stopped being a useful clue. It *is* the answer key, written in a different alphabet.

---

## Finding 2: one cheap check isn't enough

This is where a third dataset changed our method.

The Kaitholikkal dataset **passes** the regex test (0.14 — basically nothing). We would have declared it healthy. But look at it one property at a time:

| Property | Safe URLs | Phishing URLs | Gap |
|---|---|---|---|
| uses `https` | 99.99% | 6.21% | **0.94** |
| starts with `www.` | 99.80% | 13.04% | **0.87** |

That gap number is exactly the TSS you'd get by using **that single yes/no property as your entire detector**. Checking only "is it https?" scores **0.94** on this dataset. One boolean. Nothing else.

Why did our first test miss it? Because we had checked whether a property was *exactly* 100%. Kaitholikkal's is 99.99%. Close enough to slip through, nowhere near close enough to be meaningful.

The cause is visible in how it was built: safe URLs came from the Majestic Million (a ranked list of popular sites, normalized to `https://www.`), while phishing URLs came from PhishTank with their original addresses intact. The web address format gives away the source.

So there are **two different ways a dataset can be broken**, and each of our checks catches only one:

| Check | What it catches | PhiUSIIL | Kaitholikkal |
|---|---|---|---|
| The regex | *no* safe URL deviates, so combining several properties gives perfect precision | **0.9897** | 0.1357 |
| Best single property | one yes/no check nearly solves it by itself | 0.5865 | **0.9378** |

Each dataset fails the check the other one passes. **Run only one and you'll clear a broken dataset.** We found this because the third dataset broke our first method — which is exactly what adding more test cases is for.

---

## Finding 3: what this costs in practice

We trained a standard model (a random forest) on each dataset using 52 features computed from the URL text alone, then scored it against live phishing feeds.

| Test | What changes | PhiUSIIL | Kaitholikkal | faizann24 | Hannousse |
|---|---|---|---|---|---|
| 1. Shuffle rows randomly | what most papers do | 0.996 | 0.993 | 0.891 | 0.807 |
| 2. Keep each website wholly in training or testing | removes memorization | 0.995 | 0.994 | 0.802 | 0.719 |
| 3. Live phishing, live safe URLs | real deployment | **0.000** | **0.072** | **0.322** | **0.298** |

**PhiUSIIL's 0.000 is not "degraded performance."** The model labels *every* URL phishing — 300 correct catches, 268 false alarms, and zero safe URLs correctly identified. Even if we let it cheat by picking the best possible cutoff after seeing the answers, it only reaches 0.116. It has essentially no ability to tell the two apart.

The same collapse happens with a completely different threat type: malware-distribution URLs from URLhaus, same model, TSS 0.000 across 2,268 URLs.

### Where the skill actually went

For PhiUSIIL we can pin down the cause, because each step above changes exactly one thing:

| Step | Cost in TSS | Share |
|---|---|---|
| Websites appearing in both training and testing | 0.0001 | 0.0% |
| Two years of attackers adapting their tactics | 0.0087 | 0.9% |
| **The safe URLs no longer being a fixed template** | **0.9867** | **99.1%** |
| **Total** | **0.9955** | 95% confidence: 0.994–0.997 |

Memorization costs essentially nothing. Years of adversaries evolving costs almost nothing. **Letting the safe examples look like the real web costs everything.**

**One important caveat.** That three-way breakdown only makes sense for a dataset whose safe URLs *are* the template — which is PhiUSIIL alone. For the other three, rewriting live safe URLs into that template *changes* their format instead of holding it steady, so the middle steps would measure nothing real. The code detects this automatically (from each dataset's own measured template rate) and refuses to report those steps, giving a reason. It still reports the memorization step and the total, since neither depends on it.

That guard cuts both ways: memorization costs 0.0001 on PhiUSIIL, where the formatting quirk drowns everything, but **0.088 on Hannousse and 0.089 on faizann24** — real memorization the broken dataset had been hiding.

---

## Does retraining on live data fix it?

**Still open, and we say so rather than leaving the question out.**

Every model above was trained on a static dataset, so none of them can tell you whether the problem is *the training data* or something training can't fix. Answering that needs a model trained on live data, compared four ways:

|  | tested on the dataset | tested on live URLs |
|---|---|---|
| **trained on the dataset** | A | B |
| **trained on live data** | C | D |

`A − B` is the gap. **`D − B` is the answer.** `C` is the sanity check: a live-trained model that also does well on the dataset has learned something general, rather than just memorizing this month's phishing campaigns.

This builds itself automatically once enough days accumulate. It needs 10 days to train and 4 to test, split by date — **training on Monday and testing on Wednesday would let the model memorize a campaign that ran all week**, which is the exact mistake this project is about. As of the latest run: **1 day collected, 13 to go.** Our predictions are written down in [PREREGISTRATION.md](PREREGISTRATION.md) *before* any live-trained model exists, which is the only thing that makes them count.

The companion ozone and solar projects have already answered this question, and both found that retraining on newer data does **not** close the gap.

---

## Where this is weak

**We broke one of our own rules, and we're reporting it.** Before collecting anything, we committed to discounting our results if a model using *only* the shape of the URL path could separate our live phishing from our live safe URLs above 0.30. It reaches **0.529**. That's over the line, and it's flagged in [RESULTS.md §7](RESULTS.md) and on the dashboard rather than buried.

What it means: we can't claim our two collection sources are structurally equivalent, so we make no claim about how much genuine phishing signal lives in URL path structure.

What it does **not** mean is that the collapse is our fault — and the direction of the problem is why. In our live data, **92.2% of safe URLs have a page path versus 61.7% of phishing URLs.** Safe URLs are the *more* complex ones, which is the opposite of what PhiUSIIL teaches. A model that learned "a path means phishing" isn't just uninformed here — it's backwards, which is precisely why it false-alarms on everything. If we made our safe URLs simpler, we'd be moving them *toward* the broken dataset's shape and flattering the model, not punishing it. So the collapse we report is a floor, not a ceiling.

Other honest limits:

- **Four datasets shows that quality varies; it does not measure how common the problem is.** We can't tell you what fraction of published phishing datasets are affected, and we don't claim a number.
- **The live results rest on a small sample so far** (568 URLs). It grows by about 600 a day.
- **faizann24 has weaker credentials** than the other three — a community GitHub repository, no accompanying paper, no stated license. We included it because it's one of the most-copied phishing datasets around, so what it teaches is worth measuring. We flag its status wherever it appears, and we don't redistribute it.
- **No accusation is intended.** Dataset construction artifacts are common and almost always accidental. This is about what a score can be taken to mean, not about the people who assembled the data.

---

## How we kept ourselves honest

- **One feature function, used everywhere.** We take only the raw URL text and label from each dataset and compute our own 52 features, so dataset URLs and live URLs go through identical code. Both datasets ship extra pre-computed features we throw away: about half need the actual web page (impossible for live phishing links, which die within hours), and PhiUSIIL's `URLSimilarityIndex` is computed by comparing against its own safe URLs, which means it leaks the answer.
- **The warning cutoff is frozen before testing.** A model gives a probability; you pick a cutoff above which you raise an alarm. Picking that cutoff on the same data you're about to report makes any model look better than it is.
- **Error bars group whole websites together.** Many URLs from one hacked site aren't independent observations — a phishing kit spread across 200 pages of one domain is closer to *one* event than 200. Treating them as 200 would make our error bars several times too narrow.
- **The same audit rule is applied to every dataset, unchanged.** We found the regex by examining PhiUSIIL, but we never re-tune it per dataset. Tailoring it each time would guarantee a high score everywhere and make the comparison meaningless.

---

## The bigger picture

This is one of several test cases in a study asking whether a benchmark score overstates real performance, and whether that depends on **what kind of change** the model faces between testing and deployment:

| Domain | What changes | Inflation from shuffling | Real-world drift | Total |
|---|---|---|---|---|
| [Ozone](https://github.com/awesomedudeworld13/ozone-drift) | seasons and weather — nothing adversarial | +0.215 | **−0.077** | **+0.138** |
| [Solar flares](https://github.com/solarflarepredictor-cmd/SolarFlarePredictor) | the Sun's 11-year cycle | +0.118 | +0.107 | **+0.225** |
| **Phishing** (this repo) | attackers actively adapting, plus broken datasets | +0.0001 to +0.088 | +0.009 | **+0.508 to +0.996** |

Phishing is the extreme case: the only domain where someone is actively working against the model, and the only one where we found datasets that don't measure the task at all. Ozone is the control at the other end. All three now use a character-for-character identical copy of the same scoring code, which is what makes putting them in one table legitimate.

---

## Running it yourself

```bash
pip install -r requirements.txt

python -m phishdrift.cli audit      # the regex finding, all four datasets, ~1 min
python -m phishdrift.cli train      # train the models, freeze cutoffs; ~4 min
python -m phishdrift.cli collect    # grab one day of live URLs
python -m phishdrift.cli report     # everything -> RESULTS.md

python -m phishdrift.cli audit --benchmark hannousse   # just one dataset
python tests/test_features.py                          # checks the method itself
```

`RESULTS.md` is generated, never hand-edited. Automated checks rebuild every number weekly and fail the build if any dataset's audit score shifts.

### The daily collection

A scheduled job adds one permanent snapshot per day to `data/live/`: about 300 phishing URLs from OpenPhish, 300 real safe URLs with real page paths from Common Crawl, and the same sites rewritten into PhiUSIIL's template format for comparison. Snapshots are committed **before** they're ever scored, so the record can't be quietly adjusted afterward.

### Where the data comes from

| | |
|---|---|
| PhiUSIIL | [UC Irvine ML Repository #967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) — downloaded on demand |
| Hannousse & Yahiouche | [Mendeley Data](https://data.mendeley.com/datasets/c2gw7fy2j4/3), CC BY 4.0 — **stored in this repo** |
| Kaitholikkal & Arthi | [Mendeley Data](https://data.mendeley.com/datasets/vfszbj9b36/1), CC BY 4.0 — **stored in this repo** |
| faizann24 | GitHub community dataset — downloaded on demand, not redistributed |
| Live phishing | [OpenPhish community feed](https://github.com/openphish/public_feed) |
| Live malware (cross-check) | [URLhaus](https://urlhaus.abuse.ch/) |
| Live safe URLs | [Common Crawl](https://index.commoncrawl.org/), sampled from [Tranco](https://tranco-list.eu/) rankings |

The two Mendeley datasets are stored here rather than downloaded because Mendeley's bot protection rejects Python's network requests while allowing browsers through — so an automated download isn't dependable. Their license permits redistribution with credit. We verify our stored copies against the fingerprints Mendeley publishes on every automated run, so they're provably the original files rather than merely claimed to be.
