# phish-drift

**Question: when a phishing detector scores 99% on a public dataset, is it actually good at catching phishing?**

For one popular 2024 dataset, no. A three-line regular expression, with no machine learning in it anywhere, beats the published results. What that regex detects is not phishing. It is a formatting quirk left behind by whatever script assembled the dataset.

---

## The short version

We tested four public phishing-URL datasets the same way, then checked each trained model against live phishing URLs collected daily.

Two of the four are broken badly enough to be nearly useless, and the failures stay invisible unless you go looking for them:

| Dataset | Can a regex solve it? | Can one yes/no check solve it? | Score on the dataset | Score on live URLs |
|---|---|---|---|---|
| PhiUSIIL (2024) | **yes, 0.99** | 0.59 | 0.996 | **0.000** |
| Kaitholikkal & Arthi (2024) | no, 0.14 | **yes, 0.94** | 0.993 | **0.072** |
| faizann24 (community) | no, 0.00 | no, 0.06 | 0.891 | **0.322** |
| Hannousse & Yahiouche (2021) | no, 0.16 | no, 0.46 | 0.807 | **0.298** |

The two datasets that fail a cheap check collapse completely on real data. The two that pass lose about half their skill and keep the rest, which is what a normal result looks like.

The useful takeaway is not that benchmarks lie. It is that you can tell which kind of dataset you are holding in about ten seconds, before training anything.

---

## Background: what the numbers mean

**TSS** (True Skill Statistic) is the score used throughout:

> TSS = (fraction of real phishing you caught) − (fraction of safe sites you falsely flagged)

TSS is 0 for a model with no skill and 1 for a perfect one. We use it instead of accuracy because accuracy is misleading whenever one outcome is rare. A model that calls everything safe can score 95% accuracy while catching no phishing at all.

"Live URLs" means real phishing links reported by [OpenPhish](https://github.com/openphish/public_feed), collected automatically every day, paired with real safe URLs pulled from [Common Crawl](https://index.commoncrawl.org/), a public archive of the actual web.

---

## Finding 1: a regex beats the published models

PhiUSIIL contains 235,795 URLs and is hosted by the UC Irvine Machine Learning Repository, a standard source for this kind of data. Published models report around 99.2% accuracy on it.

Here is the problem. Every single one of its 134,850 "safe" URLs looks like this:

```
https://www.example.com
```

That is `https`, a `www.` prefix, and nothing after the domain. No page path, no question marks. Not most of them. All 134,850. The phishing URLs, which were pulled from a live feed, match that shape only 1% of the time.

So the dataset can be solved by a rule that involves no learning whatsoever:

> *Does this URL fail to match `https://www.something`? Then call it phishing.*

| URLs | Accuracy | Of the ones it flagged, how many were phishing | TSS |
|---|---|---|---|
| 235,795 | **99.56%** | **100%** | **0.9897** |

That is 99.56% accuracy with zero false alarms, out of one line of text matching, and it beats the machine-learning papers published on the same data.

Those models were never learning about phishing. They were learning that the safe examples had all been written by one script in one format.

The same point in table form. Each row is a simple yes/no property of a URL:

| Property | Safe URLs | Phishing URLs |
|---|---|---|
| uses `https` | **100.00%** | 48.74% |
| starts with `www.` | **100.00%** | 41.35% |
| has a page path | **0.00%** | 27.19% |
| has a query string | **0.00%** | 6.02% |

When a property sits at exactly 0% or 100% on one side, it has stopped being a clue about phishing. It is the answer key written in a different alphabet.

---

## Finding 2: one cheap check isn't enough

A third dataset is what forced us to change our method.

The Kaitholikkal dataset passes the regex test, scoring 0.14, which is essentially nothing. On that evidence alone we would have called it healthy. Looking at it one property at a time tells a different story:

| Property | Safe URLs | Phishing URLs | Gap |
|---|---|---|---|
| uses `https` | 99.99% | 6.21% | **0.94** |
| starts with `www.` | 99.80% | 13.04% | **0.87** |

That gap number is exactly the TSS you would get by using that single yes/no property as your entire detector. Checking only "is it https?" scores 0.94 on this dataset. One boolean, nothing else.

Our first test missed it because we had checked whether a property was *exactly* 100%. Kaitholikkal's is 99.99%: close enough to slip past the check, nowhere near close enough to be meaningful.

The cause is visible in how the dataset was built. Safe URLs came from the Majestic Million, a ranked list of popular sites normalized to `https://www.`, while phishing URLs came from PhishTank with their original addresses intact. The address format gives away which source a row came from.

So there are two different ways a dataset can be broken, and each of our checks catches only one of them:

| Check | What it catches | PhiUSIIL | Kaitholikkal |
|---|---|---|---|
| The regex | *no* safe URL deviates, so combining several properties gives perfect precision | **0.9897** | 0.1357 |
| Best single property | one yes/no check nearly solves it by itself | 0.5865 | **0.9378** |

Each dataset fails the check the other one passes, so running only one of them will clear a broken dataset. We found this because the third dataset broke our first method, which is what adding test cases is for.

---

## Finding 3: what this costs in practice

We trained a standard model, a random forest, on each dataset using 52 features computed from the URL text alone, then scored it against live phishing feeds.

| Test | What changes | PhiUSIIL | Kaitholikkal | faizann24 | Hannousse |
|---|---|---|---|---|---|
| 1. Shuffle rows randomly | what most papers do | 0.996 | 0.993 | 0.891 | 0.807 |
| 2. Keep each website wholly in training or testing | removes memorization | 0.995 | 0.994 | 0.802 | 0.719 |
| 3. Live phishing, live safe URLs | real deployment | **0.000** | **0.072** | **0.322** | **0.298** |

PhiUSIIL's 0.000 is not degraded performance. The model labels every URL phishing: 300 correct catches, 268 false alarms, and no safe URLs correctly identified. Even when we let it cheat by picking the best possible cutoff after seeing the answers, it only reaches 0.116. It has essentially no ability to tell the two classes apart.

The same collapse shows up with a completely different threat type. Malware-distribution URLs from URLhaus, same model, TSS 0.000 across 2,268 URLs.

### Where the skill actually went

For PhiUSIIL we can pin down the cause, because each step above changes exactly one thing:

| Step | Cost in TSS | Share |
|---|---|---|
| Websites appearing in both training and testing | 0.0001 | 0.0% |
| Two years of attackers adapting their tactics | 0.0087 | 0.9% |
| **The safe URLs no longer being a fixed template** | **0.9867** | **99.1%** |
| **Total** | **0.9955** | 95% confidence: 0.994–0.997 |

Memorization costs essentially nothing. Two years of adversaries evolving costs almost nothing. Letting the safe examples look like the real web costs everything.

One important caveat. That three-way breakdown only makes sense for a dataset whose safe URLs *are* the template, and PhiUSIIL is the only one of the four that qualifies. For the other three, rewriting live safe URLs into that template changes their format instead of holding it steady, so the middle steps would measure nothing real. The code detects this on its own, using each dataset's measured template rate, and refuses to report those steps while giving a reason. It still reports the memorization step and the total, since neither of those depends on the template.

That guard cuts both ways. Memorization costs 0.0001 on PhiUSIIL, where the formatting quirk drowns out everything else, but it costs 0.088 on Hannousse and 0.089 on faizann24. That is real memorization the broken dataset had been hiding.

---

## Does retraining on live data fix it?

Still open, and we say so rather than leaving the question out.

Every model above was trained on a static dataset, so none of them can tell you whether the problem is the training data or something training cannot fix. Answering that needs a model trained on live data, compared four ways:

|  | tested on the dataset | tested on live URLs |
|---|---|---|
| **trained on the dataset** | A | B |
| **trained on live data** | C | D |

`A − B` is the gap. `D − B` is the answer. `C` is the sanity check: a live-trained model that also does well on the dataset has learned something general, rather than memorizing this month's phishing campaigns.

This builds itself once enough days accumulate. It needs 10 days to train on and 4 to test on, split by date. Training on Monday and testing on Wednesday would let the model memorize a campaign that ran all week, which is the exact mistake this project is about. As of the latest run, 1 day is collected and 13 remain. Our predictions are written down in [PREREGISTRATION.md](PREREGISTRATION.md) before any live-trained model exists, which is the only thing that makes them count.

The companion projects don't settle it either. Solar found that retraining didn't close the gap in solar cycle 24. Ozone first reported that retraining made things worse, but that turned out to be the warning cutoff, not the model: with the cutoff taken out, the two models scored the same. An exploratory run on the `testing-new` branch, using dated historical feed data, suggests that for phishing the answer is yes; it is not pre-registered here and is reported there.

---

## Where this is weak

We broke one of our own rules, and we are reporting it. Before collecting anything, we committed to discounting our results if a model using only the shape of the URL path could separate our live phishing from our live safe URLs above 0.30. It reaches 0.529. That is over the line, and it is flagged in [RESULTS.md §7](RESULTS.md) and on the dashboard rather than buried.

What it means is that we cannot claim our two collection sources are structurally equivalent, so we make no claim about how much genuine phishing signal lives in URL path structure.

What it does not mean is that the collapse is our fault, and the direction of the problem is the reason why. In our live data, 92.2% of safe URLs have a page path against 61.7% of phishing URLs. The safe URLs are the more complex ones, which is the opposite of what PhiUSIIL teaches. A model that learned "a path means phishing" is not merely uninformed on our data, it is backwards, and that is why it false-alarms on everything. Making our safe URLs simpler would move them toward the broken dataset's shape and flatter the model rather than punish it. The collapse we report is therefore a floor, not a ceiling.

Other honest limits:

- **Four datasets shows that quality varies. It does not measure how common the problem is.** We cannot tell you what fraction of published phishing datasets are affected, and we do not claim a number.
- **The live results rest on a small sample so far**, 568 URLs, growing by about 600 a day.
- **faizann24 has weaker credentials than the other three.** It is a community GitHub repository with no accompanying paper and no stated license. We included it because it is one of the most-copied phishing datasets around, so what it teaches is worth measuring. We flag its status wherever it appears, and we do not redistribute it.
- **No accusation is intended.** Dataset construction artifacts are common and almost always accidental. This is about what a score can be taken to mean, not about the people who assembled the data.

---

## How we kept ourselves honest

- **One feature function, used everywhere.** We take only the raw URL text and label from each dataset and compute our own 52 features, so dataset URLs and live URLs go through identical code. Both datasets ship extra pre-computed features that we throw away. About half of those need the actual web page, which is impossible for live phishing links since they die within hours, and PhiUSIIL's `URLSimilarityIndex` is computed by comparing against its own safe URLs, so it leaks the answer.
- **The warning cutoff is frozen before testing.** A model gives a probability, and you pick a cutoff above which you raise an alarm. Picking that cutoff on the same data you are about to report on makes any model look better than it is.
- **Error bars group whole websites together.** Many URLs from one hacked site are not independent observations. A phishing kit spread across 200 pages of one domain is closer to a single event than to 200 of them, and counting it as 200 would make our error bars several times too narrow.
- **The same audit rule is applied to every dataset, unchanged.** We found the regex by examining PhiUSIIL, but we never re-tune it per dataset. Tailoring it each time would guarantee a high score everywhere and make the comparison meaningless.

---

## The bigger picture

This is one of several test cases in a study asking whether a benchmark score overstates real performance, and whether that depends on what kind of change the model faces between testing and deployment:

| Domain | What changes between testing and the real world | Inflation from shuffling | Real-world drift | Total |
|---|---|---|---|---|
| [Ozone / smog](https://github.com/awesomedudeworld13/ozone-drift) | seasons and weather, nothing adversarial | +0.215 | **−0.077** | **+0.138** |
| [Solar flares](https://github.com/solarflarepredictor-cmd/SolarFlarePredictor) | the Sun's 11-year cycle | +0.118 | +0.107 | **+0.225** |
| [Geomagnetic storms](https://github.com/solarflarepredictor-cmd/SolarFlarePredictor) | multi-day space-weather disturbances | +0.325 | +0.071 | **+0.396** |
| **Phishing URLs** (this repo) | attackers adapting, plus broken datasets | +0.0001 to +0.088 | +0.009 | **+0.508 to +0.996** |

Phishing is the extreme case. It is the only domain where someone is actively working against the model, and the only one where we found datasets that do not measure the task at all. Ozone sits at the other end as the control. All four domains score with the same code, the [benchgap](https://github.com/awesomedudeworld13/benchgap) package, which is what makes putting them in one table legitimate.

Live dashboards: [phishing](https://awesomedudeworld13.github.io/phish-drift/) · [ozone](https://awesomedudeworld13.github.io/ozone-drift/)

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

A scheduled job adds one permanent snapshot per day to `data/live/`: about 300 phishing URLs from OpenPhish, 300 real safe URLs with real page paths from Common Crawl, and the same sites rewritten into PhiUSIIL's template format for comparison. Snapshots are committed before they are ever scored, so the record cannot be quietly adjusted afterward. They are committed **encrypted**, because OpenPhish's terms forbid republishing its URLs; [DATA_HANDLING.md](DATA_HANDLING.md) explains what was done and why.

### Where the data comes from

| | |
|---|---|
| PhiUSIIL | [UC Irvine ML Repository #967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset), downloaded on demand |
| Hannousse & Yahiouche | [Mendeley Data](https://data.mendeley.com/datasets/c2gw7fy2j4/3), CC BY 4.0, **stored in this repo** |
| Kaitholikkal & Arthi | [Mendeley Data](https://data.mendeley.com/datasets/vfszbj9b36/1), CC BY 4.0, **stored in this repo** |
| faizann24 | GitHub community dataset, downloaded on demand, not redistributed |
| Live phishing | [OpenPhish community feed](https://github.com/openphish/public_feed) |
| Live malware (cross-check) | [URLhaus](https://urlhaus.abuse.ch/) |
| Live safe URLs | [Common Crawl](https://index.commoncrawl.org/), sampled from [Tranco](https://tranco-list.eu/) rankings |

The two Mendeley datasets are stored here rather than downloaded because Mendeley's bot protection rejects Python's network requests while letting browsers through, so an automated download is not dependable. Their license permits redistribution with credit. We verify our stored copies against the fingerprints Mendeley publishes on every automated run, so they are provably the original files rather than merely claimed to be.
