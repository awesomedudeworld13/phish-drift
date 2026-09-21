
> **When a machine-learning model scores well on a test set, does that score still hold up once the model is used for real?**

The short answer is that it depends — enormously — on what kind of change the model faces between testing and deployment. In one case a model that scores **0.996** on a published dataset scores **0.000** on live data. In another, the score barely moves.

---

## The study

Four forecasting problems, picked so each faces a **different kind of change** between test data and reality. All four are scored with the same code, which is what makes them comparable.

| Domain | What changes | Inflation from sloppy testing | Real-world drift | **Total gap** |
|---|---|---|---|---|
| [**Ozone / smog**](https://github.com/awesomedudeworld13/ozone-drift) | seasons and weather — nothing adversarial | +0.215 | **−0.077** | **+0.138** |
| [**Solar flares**](https://github.com/solarflarepredictor-cmd/SolarFlarePredictor) | the Sun's 11-year cycle | +0.118 | +0.107 | **+0.225** |
| [**Geomagnetic storms**](https://github.com/solarflarepredictor-cmd/SolarFlarePredictor) | multi-day space-weather disturbances | +0.325 | +0.071 | **+0.396** |
| [**Phishing URLs**](https://github.com/awesomedudeworld13/phish-drift) | attackers actively adapting, plus broken datasets | +0.0001 to +0.088 | +0.009 | **+0.508 to +0.996** |

That ordering — ozone < flares < storms < phishing — was **written down before any of it was measured**, and it held.

### What the numbers mean

The score is **TSS**: the fraction of real events you catch, minus the fraction of quiet periods you falsely alarm on. **TSS is 0 for a model with no skill** and 1 for a perfect one.

Accuracy is deliberately avoided. When the thing you're predicting is rare — smog days are 2% of days — a model that always says "nothing will happen" scores **98% accurate** and is completely useless. Its TSS is 0, which is the honest description.

---

## Three things worth clicking

### A regular expression beat the published machine learning

[**phish-drift**](https://github.com/awesomedudeworld13/phish-drift) · [live dashboard](https://awesomedudeworld13.github.io/phish-drift/)

A widely-used 2024 phishing dataset has 235,795 URLs. Every single one of its 134,850 "safe" URLs is formatted `https://www.something` — no path, no query string. Not most. All of them.

So a one-line text match — *"doesn't look like that? call it phishing"* — scores **99.56% accuracy with zero false alarms**, beating the machine-learning papers published on the same data. Those models never learned about phishing. They learned which script wrote the URL.

Trained on it and pointed at live phishing feeds, the model labels **every** URL phishing. TSS 0.000.

Four datasets tested, two broken, in two *different* ways that need two different checks to catch.

### Sloppy testing, not a changing world

[**ozone-drift**](https://github.com/awesomedudeworld13/ozone-drift) · [live dashboard](https://awesomedudeworld13.github.io/ozone-drift/)

Forecasting whether Houston smog will break the federal health limit tomorrow. Almost the entire apparent drop-off came from one habit: shuffling days at random before splitting into training and test.

Weather doesn't reset at midnight, and a smog day hits all 17 monitoring stations at once — so shuffling lets a model recognise a day it has nearly already seen. Fix the testing and the score *improves* on years it had never seen.

This is the control case, and a small result here is the useful one: it shows the conditions under which a benchmark score **can** be trusted.

### A forecaster that runs in public and grades itself

[**SolarFlarePredictor**](https://github.com/solarflarepredictor-cmd/SolarFlarePredictor)

Reads live NASA and NOAA satellite data and estimates the chance of a solar flare in the next 12/24/48 hours. Issues one forecast a day, writes it to a timestamped file **before** the outcome is known, and checks itself afterward. Nothing can be quietly revised after the fact.

---

## The tool

[**benchgap**](https://github.com/awesomedudeworld13/benchgap) · [PyPI](https://pypi.org/project/benchgap/)

```bash
pip install benchgap
```

The scoring code all four studies share, extracted into one package so every number they report is genuinely the same quantity. Depends on numpy and nothing else.

It exists mainly to get one thing right that's routinely got wrong: **confidence intervals on model evaluations are usually far too narrow**, because the rows being resampled aren't independent. 200 URLs on one hacked website aren't 200 observations — they're closer to one. Neither are 17 stations recording the same smog event, or 300 snapshots of the same sunspot group.

---

## How this work is kept honest

- **Predictions are written down before the results exist.** When one fails, it stays in the file marked FAILED — one did, and it's still there.
- **Decision thresholds are frozen on separate data**, never on the numbers being reported.
- **Results files are generated by script, never hand-edited**, and rebuilt automatically.
- **Every dataset gets audited before it's trusted** — including our own data collection, which breached one of its own pre-registered limits and is reported that way rather than buried.

---

<sub>Also: <a href="https://github.com/awesomedudeworld13/defi-threat-monitor">defi-threat-monitor</a> · <a href="https://github.com/awesomedudeworld13/tinyeconomy">tinyeconomy</a></sub>
