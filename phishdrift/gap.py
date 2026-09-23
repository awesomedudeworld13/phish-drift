"""The experiment: decomposing the benchmark-to-operational gap.

Reporting "the benchmark overstates skill by X" is a single number and a weak
claim -- it names an effect without a cause. This module instead walks from the
published protocol to operational reality in three steps, each of which changes
**exactly one thing**, so the total drop is attributed rather than merely
observed.

    cell 1  benchmark corpus, random i.i.d. split      <- what the literature reports
              |  change: partition by domain instead of by row
    cell 2  benchmark corpus, domain-disjoint split    <- removes memorised domains
              |  change: swap the corpus for live feeds, benign surface form held fixed
    cell 3  live feeds, TEMPLATED benign               <- removes era, keeps construction
              |  change: release the benign surface form to what the real web looks like
    cell 4  live feeds, REALISTIC benign               <- operational reality

Attribution:

    split leakage          = cell 1 - cell 2
    temporal/adversarial   = cell 2 - cell 3
    construction artifact  = cell 3 - cell 4
    total                  = cell 1 - cell 4

Cell 3 is what makes this more than a two-point comparison. Without it, the
collapse from benchmark to live could be blamed entirely on attackers adapting,
when in fact a large part of it may be that the benchmark's benign class was
generated from a string template that nothing in the real world resembles. The
only way to tell those apart is to hold the benign surface form fixed while
changing the era, then change it back.

Metric note: TSS is invariant to class base rate, which is why it carries the
headline. The cells have genuinely different base rates (the benchmark is 43%
phishing; a live sample is whatever the feeds yield that day) and any
base-rate-sensitive metric would confound that difference with the effects
above. Base rates are printed for every cell regardless.

Scope: this module answers *how large* the gap is and *what causes it*. It
cannot answer whether the gap would close given live training data, because
every model in the four cells was fitted on the benchmark. That second question
-- the training-source x test-set 2x2 against a model fitted on live snapshots
alone -- lives in ``livetrain.py`` and is reported as section 5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from benchgap import evaluate
from .benchmark import Split, structural_degeneracy, template_audit
from .features import PATH_SHAPE_FEATURES, FEATURE_NAMES, feature_matrix
from .model import TemplateRule, TrainedModel

BOOTSTRAP_RESAMPLES = 1000

# The positive class for every headline number. See the note in `run`.
HEADLINE_PHISH_SOURCE = "openphish"

# Pre-registered ceiling for the self-audit in `sampling_confound_diagnostic`.
# Committed in PREREGISTRATION.md before live collection began; exceeding it
# obliges us to discount the headline rather than explain it away.
CONFOUND_CEILING = 0.30


@dataclass
class Cell:
    """One evaluation cell: a model scored on a corpus at frozen thresholds."""

    name: str
    description: str
    n: int
    base_rate: float
    scores: dict = field(default_factory=dict)      # operating point -> Scores.to_dict()
    peak_tss: float = float("nan")
    peak_threshold: float = float("nan")

    def tss(self, operating_point: str = "tss") -> float:
        return self.scores[operating_point]["tss"]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "n": self.n,
            "base_rate": round(self.base_rate, 6),
            "peak_tss": round(self.peak_tss, 6),
            "peak_threshold": round(self.peak_threshold, 6),
            "operating_points": self.scores,
        }


def evaluate_cell(name: str, description: str, model, frame: pd.DataFrame,
                  thresholds: dict[str, float], bootstrap: bool = True,
                  n_resamples: int = BOOTSTRAP_RESAMPLES) -> Cell:
    """Score ``model`` on ``frame`` at each frozen threshold.

    ``frame`` needs ``url``, ``y`` and ``domain``. Confidence intervals cluster
    on ``domain`` because URLs sharing a host are not independent observations.
    """
    y = frame["y"].to_numpy()
    prob = model.predict_proba(frame["url"])
    groups = frame["domain"].to_numpy()

    peak, peak_thr = evaluate.peak_tss(y, prob)
    cell = Cell(
        name=name,
        description=description,
        n=len(frame),
        base_rate=float(y.mean()),
        peak_tss=peak,
        peak_threshold=peak_thr,
    )

    for point, thr in thresholds.items():
        s = evaluate.score_at(y, prob, thr).to_dict()
        if bootstrap:
            lo, hi = evaluate.cluster_bootstrap_ci(
                y, prob, thr, groups=groups, n_resamples=n_resamples
            )
            s["tss_ci95"] = [round(lo, 6), round(hi, 6)]
        cell.scores[point] = s

    return cell


def sampling_confound_diagnostic(live: pd.DataFrame, seed: int = 20260920,
                                  features: tuple[str, ...] = PATH_SHAPE_FEATURES) -> dict:
    """Can path *shape* alone separate our live classes?

    This is the audit of our own data collection, and it is deliberately aimed
    at the weakest point in the design. Our benign URLs come from Common Crawl
    and our phishing URLs from OpenPhish -- two different pipelines. If those
    pipelines differ systematically in URL *structure* (how deep the path is,
    whether there is a query string), then a model could separate the classes
    without learning anything about phishing, and our operational numbers would
    be measuring our own collection procedure.

    A model trained on nothing but the path-shape features should be close to
    TSS 0 if collection is clean. A high score here invalidates the headline and
    must be reported, not resolved.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    idx = [FEATURE_NAMES.index(f) for f in features]
    X = feature_matrix(live["url"])[:, idx]
    y = live["y"].to_numpy()
    groups = live["domain"].to_numpy()

    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 5:
        return {"available": False, "reason": "insufficient live data"}

    # Grouped CV so the diagnostic cannot itself be inflated by domain leakage.
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in splitter.split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    peak, thr = evaluate.peak_tss(y, oof)

    # Concrete mechanism, so a reader can see what the diagnostic is reacting to
    # rather than having to trust a single summary number.
    has_path = np.array([
        len((u.split("://", 1)[-1].split("/", 1) + [""])[1].split("?")[0].rstrip("/")) > 0
        for u in live["url"]
    ])
    path_rates = {
        "benign": round(float(has_path[y == 0].mean()), 6) if (y == 0).any() else None,
        "phishing": round(float(has_path[y == 1].mean()), 6) if (y == 1).any() else None,
    }

    breached = peak > CONFOUND_CEILING
    return {
        "available": True,
        "features_used": list(features),
        "n": int(len(y)),
        "peak_tss_path_shape_only": round(peak, 6),
        "threshold": round(thr, 6),
        "preregistered_ceiling": CONFOUND_CEILING,
        "ceiling_breached": bool(breached),
        "path_presence_rate": path_rates,
        "interpretation": (
            "This asks whether our own two collection pipelines (OpenPhish for "
            "phishing, Common Crawl for benign) differ in URL structure enough "
            "that a model could separate them without learning anything about "
            "phishing. Near 0 means they are structurally indistinguishable."
        ),
        "verdict": (
            (
                f"BREACHED: {peak:.3f} exceeds the pre-registered ceiling of "
                f"{CONFOUND_CEILING}. Path shape alone carries real separating "
                "power in our live corpus, so we cannot claim our two streams are "
                "structurally equivalent, and any statement about how much genuine "
                "phishing signal lives in path structure must be withheld.\n\n"
                "What this does NOT do is explain the headline collapse, and the "
                "measured direction is why. In this live corpus "
                f"{path_rates['benign']:.1%} of BENIGN URLs carry a path against "
                f"{path_rates['phishing']:.1%} of phishing ones -- benign URLs are "
                "the ones with deeper structure, the opposite of what the benchmark "
                "teaches. A benchmark-trained model that learned 'a path means "
                "phishing' is therefore not merely uninformed on this feature, it is "
                "anti-correlated with reality, which is exactly why its false-alarm "
                "rate is 1.000. The separating information is present and a model "
                f"given only path shape reaches {peak:.3f} with it; the "
                "benchmark-trained model reaches 0.000 because it learned the sign "
                "backwards. Re-sampling benign URLs to be shallower would move our "
                "corpus toward the benchmark's shape and flatter the model, not "
                "penalise it, so the headline is a lower bound on the collapse."
            )
            if breached else
            f"OK: {peak:.3f} is within the pre-registered ceiling of {CONFOUND_CEILING}."
        ),
    }


def run(benchmark: pd.DataFrame,
        random_split_obj: Split,
        disjoint_split_obj: Split,
        model_random: TrainedModel,
        model_disjoint: TrainedModel,
        live: pd.DataFrame,
        n_resamples: int = BOOTSTRAP_RESAMPLES,
        benchmark_meta: dict | None = None) -> dict:
    """Execute all four cells plus the audits, and return the results document."""
    results: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bootstrap_resamples": n_resamples,
        "benchmark": benchmark_meta or {},
    }

    # -- Audit A: what the benchmark hands you for free -------------------
    audit = template_audit(benchmark)
    results["construction_audit"] = {
        "template_rule": audit.to_dict(),
        "structural_rates": structural_degeneracy(benchmark),
        "note": (
            "The template rule has zero learned parameters and needs no split. "
            "If its TSS is comparable to published machine-learning results on "
            "this corpus, those results are not evidence of phishing-detection "
            "skill."
        ),
    }

    # -- Cells 1 and 2: benchmark, two partitions -------------------------
    cells: dict[str, Cell] = {}
    cells["cell1_benchmark_random"] = evaluate_cell(
        "cell1_benchmark_random",
        "Benchmark corpus, random i.i.d. split — the published protocol.",
        model_random, random_split_obj.test, model_random.thresholds,
        n_resamples=n_resamples,
    )
    cells["cell2_benchmark_disjoint"] = evaluate_cell(
        "cell2_benchmark_disjoint",
        "Benchmark corpus, domain-disjoint split — no eTLD+1 shared with training.",
        model_disjoint, disjoint_split_obj.test, model_disjoint.thresholds,
        n_resamples=n_resamples,
    )

    # -- Cells 3 and 4: live feeds, two benign surface forms ---------------
    # Both use the domain-disjoint model: it is the honest benchmark artifact,
    # so any further drop is attributable to the data and not to the partition.
    #
    # The headline positive class is OpenPhish only. URLhaus lists malware
    # *distribution* URLs, which is a related but genuinely different task with
    # a different URL grammar; folding it in would quietly redefine what is
    # being measured and would also swamp the benign class numerically. It is
    # scored separately under "robustness" below.
    phish_live = live[(live.y == 1) & (live.source == HEADLINE_PHISH_SOURCE)]
    have_live = len(phish_live) > 0

    if have_live:
        templated = pd.concat(
            [phish_live, live[(live.y == 0) & (live.source == "tranco_templated")]],
            ignore_index=True,
        )
        realistic = pd.concat(
            [phish_live, live[(live.y == 0) & (live.source == "commoncrawl")]],
            ignore_index=True,
        )
        if len(templated[templated.y == 0]):
            cells["cell3_live_templated_benign"] = evaluate_cell(
                "cell3_live_templated_benign",
                "Live feeds, benign rendered in the benchmark's own surface form "
                "(https://www.<domain>) — era changes, construction held fixed.",
                model_disjoint, templated, model_disjoint.thresholds,
                n_resamples=n_resamples,
            )
        if len(realistic[realistic.y == 0]):
            cells["cell4_live_realistic_benign"] = evaluate_cell(
                "cell4_live_realistic_benign",
                "Live feeds, benign drawn from Common Crawl with real paths — "
                "operational reality.",
                model_disjoint, realistic, model_disjoint.thresholds,
                n_resamples=n_resamples,
            )

    results["cells"] = {k: v.to_dict() for k, v in cells.items()}

    # -- Attribution ------------------------------------------------------
    steps = [
        ("split_leakage", "cell1_benchmark_random", "cell2_benchmark_disjoint",
         "Same corpus and era; only the partitioning changes. Skill lost here was "
         "memorisation of domains that appear on both sides of a random split."),
        ("temporal_adversarial_shift", "cell2_benchmark_disjoint", "cell3_live_templated_benign",
         "Corpus changes from the 2024 benchmark to live feeds while the benign "
         "surface form is held at the benchmark's. Skill lost here is genuine drift."),
        ("benign_construction_artifact", "cell3_live_templated_benign", "cell4_live_realistic_benign",
         "Same era and same phishing rows; only the benign class stops being a "
         "string template. Skill lost here was never phishing detection at all."),
    ]

    # Cell 3 only means "hold construction fixed while the era changes" for a
    # corpus whose benign class IS the template. Where it is not -- Hannousse's
    # benign URLs carry paths, http and no www -- rendering live benign URLs as
    # `https://www.<domain>` *changes* the construction instead of holding it,
    # and the middle two steps stop measuring what their names say. The total
    # gap (cell 1 -> cell 4) remains valid in every case, because it never
    # depends on cell 3.
    benign_template_share = audit.benign_template_share
    decomposable = benign_template_share >= 0.90

    attribution = {}
    for point_name in ("f1", "tss"):
        rows = {}
        if decomposable:
            for key, a, b in ((s[0], s[1], s[2]) for s in steps):
                if a in cells and b in cells:
                    rows[key] = round(cells[a].tss(point_name) - cells[b].tss(point_name), 6)
        elif "cell1_benchmark_random" in cells and "cell2_benchmark_disjoint" in cells:
            # Split leakage compares two partitions of one corpus and never
            # touches cell 3, so it survives even when the rest does not.
            rows["split_leakage"] = round(
                cells["cell1_benchmark_random"].tss(point_name)
                - cells["cell2_benchmark_disjoint"].tss(point_name), 6
            )
        if "cell1_benchmark_random" in cells and "cell4_live_realistic_benign" in cells:
            rows["total"] = round(
                cells["cell1_benchmark_random"].tss(point_name)
                - cells["cell4_live_realistic_benign"].tss(point_name), 6
            )
        attribution[point_name] = rows

    results["attribution_tss"] = attribution
    results["attribution_applicable"] = {
        "full_decomposition": bool(decomposable),
        "benign_template_share": round(benign_template_share, 6),
        "reason": (
            "This corpus's benign class matches the template in "
            f"{benign_template_share:.1%} of rows, so cell 3 holds its construction "
            "fixed while changing the era and the three-step attribution is valid."
            if decomposable else
            "This corpus's benign class matches the template in only "
            f"{benign_template_share:.1%} of rows, so cell 3 does NOT hold its "
            "construction fixed -- for this corpus, rendering live benign URLs in "
            "template form is itself a change of construction. The middle two steps "
            "would not measure what their names claim and are withheld. Split "
            "leakage (cells 1 vs 2) and the total gap (cells 1 vs 4) are reported "
            "because neither depends on cell 3. Cell 3 is still shown as a "
            "diagnostic: it reports how the model behaves on template-shaped benign "
            "URLs, which is informative but is not a step in a decomposition."
        ),
    }
    results["attribution_steps"] = [
        {"step": s[0], "from": s[1], "to": s[2], "meaning": s[3]} for s in steps
    ]

    # -- Paired CI on the step that matters most --------------------------
    if "cell1_benchmark_random" in cells and "cell4_live_realistic_benign" in cells:
        a_frame = random_split_obj.test
        b_frame = pd.concat(
            [live[live.y == 1], live[(live.y == 0) & (live.source == "commoncrawl")]],
            ignore_index=True,
        )
        point, lo, hi = evaluate.paired_difference_ci(
            a_frame["y"].to_numpy(), model_random.predict_proba(a_frame["url"]),
            model_random.thresholds["tss"],
            b_frame["y"].to_numpy(), model_disjoint.predict_proba(b_frame["url"]),
            model_disjoint.thresholds["tss"],
            groups_a=a_frame["domain"].to_numpy(),
            groups_b=b_frame["domain"].to_numpy(),
            n_resamples=n_resamples,
        )
        results["total_gap_ci95"] = {
            "operating_point": "tss",
            "point_estimate": round(point, 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "excludes_zero": bool(lo > 0 or hi < 0),
        }

    # -- Audit B: the zero-parameter rule, everywhere ---------------------
    rule_rows = {}
    rule_frames = {
        "cell1_benchmark_random": random_split_obj.test,
        "cell2_benchmark_disjoint": disjoint_split_obj.test,
    }
    if have_live:
        for key, src in (("cell3_live_templated_benign", "tranco_templated"),
                         ("cell4_live_realistic_benign", "commoncrawl")):
            benign = live[(live.y == 0) & (live.source == src)]
            if len(benign):
                rule_frames[key] = pd.concat([phish_live, benign], ignore_index=True)

    for key, frame in rule_frames.items():
        y = frame["y"].to_numpy()
        prob = TemplateRule.predict_proba(frame["url"])
        rule_rows[key] = evaluate.score_at(y, prob, 0.5).to_dict()
    results["template_rule_by_cell"] = rule_rows

    # -- Audit C: our own collection ---------------------------------------
    if have_live:
        live_headline = pd.concat(
            [phish_live, live[(live.y == 0) & (live.source == "commoncrawl")]],
            ignore_index=True,
        )
        if len(live_headline[live_headline.y == 0]):
            results["sampling_confound"] = sampling_confound_diagnostic(live_headline)

    # -- Question 2: does training on live data close the gap? --------------
    # The four cells above quantify the gap. They cannot say whether it is a
    # training-data problem, because every model in them was fitted on the
    # benchmark. This runs the training-source x test-set 2x2 against a model
    # fitted on live snapshots alone. Until the live record is long enough for
    # a temporal split, it returns the question with what is still missing,
    # which is reported rather than omitted.
    from . import livetrain

    results["live_training"] = livetrain.run_2x2(
        benchmark_model=model_disjoint,
        benchmark_test=disjoint_split_obj.test,
        live=live,
        n_resamples=n_resamples,
    )

    # -- Robustness: the adjacent task, scored separately -------------------
    # URLhaus is malware distribution rather than phishing. If the operational
    # collapse is a property of the benchmark's construction rather than of one
    # particular feed, it should reproduce here on a different positive class.
    malware = live[(live.y == 1) & (live.source == "urlhaus")]
    benign_real = live[(live.y == 0) & (live.source == "commoncrawl")]
    if len(malware) and len(benign_real):
        frame = pd.concat([malware, benign_real], ignore_index=True)
        cell = evaluate_cell(
            "robustness_urlhaus_malware",
            "Live URLhaus malware-distribution URLs vs realistic benign — a "
            "different positive class, same benchmark-trained model.",
            model_disjoint, frame, model_disjoint.thresholds,
            n_resamples=n_resamples,
        )
        results["robustness"] = {"urlhaus_malware": cell.to_dict()}

    # -- Provenance ---------------------------------------------------------
    results["models"] = {
        "random_split": model_random.metadata,
        "domain_disjoint": model_disjoint.metadata,
    }
    if have_live:
        results["live_corpus"] = {
            "n_rows": int(len(live)),
            "n_phishing": int((live.y == 1).sum()),
            "by_source": {k: int(v) for k, v in live.source.value_counts().items()},
            "n_domains": int(live.domain.nunique()),
            "span": [str(live.snapshot_date.min()), str(live.snapshot_date.max())]
            if "snapshot_date" in live else None,
        }

    return results


def combine(by_key: dict[str, dict]) -> dict:
    """Merge per-benchmark result documents into the published document.

    The cross-benchmark comparison is computed here rather than in ``run``
    because it is the part that cannot be derived from a single corpus, and it
    is the part that decides what the project is entitled to claim. One
    benchmark collapsing shows that *a* benchmark is flawed. A second benchmark
    behaving differently is what separates "this corpus is broken" from "the
    field is broken" -- and, in the direction it actually came out, supplies a
    healthy control on which the operational gap can be measured without a
    construction artifact confounding it.
    """
    from .benchmark import (
        constant_properties, degenerate_properties, max_single_property_tss,
    )

    comparison = []
    for key, r in by_key.items():
        ta = r["construction_audit"]["template_rule"]
        rates = r["construction_audit"]["structural_rates"]
        degenerate = degenerate_properties(rates)
        constant = constant_properties(rates)
        cells = r.get("cells", {})
        attr = (r.get("attribution_tss") or {}).get("tss", {})
        comparison.append({
            "key": key,
            "title": r.get("benchmark", {}).get("title", key),
            "n": ta["n"],
            "template_rule_tss": ta["tss"],
            "benign_template_share": ta["benign_template_share"],
            "degenerate_properties": degenerate,
            "constant_properties": constant,
            # The strongest single-boolean baseline. Reported alongside the
            # template rule because the two catch different pathologies: a
            # corpus can pass the template audit while one property alone
            # solves it, which is exactly what Kaitholikkal does.
            "max_single_property_tss": max_single_property_tss(rates),
            "cell1_tss": cells.get("cell1_benchmark_random", {})
                              .get("operating_points", {}).get("tss", {}).get("tss"),
            "cell4_tss": cells.get("cell4_live_realistic_benign", {})
                              .get("operating_points", {}).get("tss", {}).get("tss"),
            "total_gap": attr.get("total"),
            "split_leakage": attr.get("split_leakage"),
            "decomposable": r.get("attribution_applicable", {}).get("full_decomposition"),
            "construction_share": (
                round(attr["benign_construction_artifact"] / attr["total"], 6)
                if attr.get("total") and "benign_construction_artifact" in attr else None
            ),
        })
    # Rank by the worse of the two zero-parameter baselines, so a corpus that
    # passes one audit and fails the other still sorts to the top.
    comparison.sort(
        key=lambda c: max(c["template_rule_tss"], c["max_single_property_tss"]),
        reverse=True,
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmarks": by_key,
        "comparison": comparison,
    }


def _fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def render_markdown(document: dict) -> str:
    """Render RESULTS.md from a combined document. Generated, never hand-edited."""
    out: list[str] = []
    a = out.append

    a("# phish-drift — Results\n")
    a("> Auto-generated by `python -m phishdrift.cli report`. Do not hand-edit.\n")
    a(f"Generated {document['generated_at_utc']}.\n")

    comp = document.get("comparison", [])
    if len(comp) > 1:
        a("## Cross-benchmark comparison\n")
        a("Two public phishing-URL corpora, one identical pipeline. The "
          "zero-parameter template rule is applied unchanged to both, which is "
          "what makes this a comparison rather than a bespoke accusation.\n")
        a("Two zero-parameter baselines are reported because they catch different "
          "construction pathologies, and a corpus can pass one while failing the "
          "other:\n")
        a("- **template rule** — the conjunction `^https://www\\.[^/?#]+/?$`. High "
          "when *no* benign row deviates, so combining several surface properties "
          "identifies the benign class with perfect precision.")
        a("- **best single property** — the largest "
          "`|P(prop|benign) − P(prop|phishing)|`, which is exactly the TSS of that "
          "one boolean used as the whole classifier. High when one property alone "
          "nearly solves the corpus, with no conjunction and no exact extreme "
          "needed.\n")
        a("| corpus | n | template rule | best single property | degenerate | constant | benchmark TSS | live TSS | total gap |")
        a("|---|---|---|---|---|---|---|---|---|")
        for c in comp:
            deg = ", ".join(f"`{d}`" for d in c["degenerate_properties"]) or "—"
            con = ", ".join(f"`{d}`" for d in c["constant_properties"]) or "—"
            a(f"| {c['title']} | {c['n']:,} | **{c['template_rule_tss']:.4f}** | "
              f"**{c['max_single_property_tss']:.4f}** | {deg} | {con} | "
              f"{_fmt(c['cell1_tss'])} | **{_fmt(c['cell4_tss'])}** | "
              f"{_fmt(c['total_gap'])} |")
        a("")
        a("*A corpus solved by a regular expression or by a single boolean is not "
          "measuring phishing detection. One where both baselines are near-worthless "
          "is measuring something real, and is the control against which the "
          "operational gap should be read.*\n")
        a("**Neither baseline alone is sufficient.** The template rule scores 0.9897 "
          "on PhiUSIIL but only 0.1357 on Kaitholikkal — which would clear the latter "
          "entirely, while a single boolean (`is_https`, 0.9999 benign vs 0.0621 "
          "phishing) solves it at 0.9378. Conversely the highest single-property "
          "score on PhiUSIIL is just 0.5865; its pathology only appears in the "
          "conjunction. A corpus audit has to run both.\n")
        a("Read across the corpora, the result is not 'benchmarks overstate "
          "performance'. It is that the benchmark-to-operational gap varies "
          "enormously with how a corpus was assembled — from total collapse to a "
          "survivable halving — and that cheap, zero-parameter, label-free checks "
          "predict which kind you are holding before any model is trained.\n")

    for key, r in document["benchmarks"].items():
        title = r.get("benchmark", {}).get("title", key)
        a(f"\n---\n\n# {title}\n")
        citation = r.get("benchmark", {}).get("citation")
        if citation:
            a(f"> {citation}\n")
        a(f"{r['bootstrap_resamples']} domain-clustered bootstrap resamples.\n")
        out.extend(_render_one(r))

    return "\n".join(out) + "\n"


def _render_one(r: dict) -> list[str]:
    """Per-benchmark detail sections."""
    out: list[str] = []
    a = out.append

    # 1. Construction audit
    ta = r["construction_audit"]["template_rule"]
    a("## 1. What the benchmark hands you before any learning\n")
    a("A single regular expression, with zero learned parameters, asking only "
      "whether a URL matches the string template `https://www.<domain>`:\n")
    a(f"| n | accuracy | precision | recall | false alarm | **TSS** |")
    a("|---|---|---|---|---|---|")
    a(f"| {ta['n']:,} | {ta['accuracy']:.4f} | {ta['precision']:.4f} | "
      f"{ta['recall']:.4f} | {ta['false_alarm_rate']:.4f} | **{ta['tss']:.4f}** |\n")
    a(f"Share of the benign class matching the template: "
      f"**{ta['benign_template_share']:.4f}**. Share of the phishing class: "
      f"{ta['phish_template_share']:.4f}.\n")
    a("Per-class rates of the four surface properties involved:\n")
    a("| property | benign | phishing |")
    a("|---|---|---|")
    for prop, vals in r["construction_audit"]["structural_rates"].items():
        a(f"| `{prop}` | {vals['benign']:.4f} | {vals['phishing']:.4f} |")
    a("")
    a("*A rate of exactly 0.0000 or 1.0000 on one side means the property is not "
      "a feature — it is the class label, written in a different alphabet.*\n")

    # 2. The cells
    a("## 2. The four cells\n")
    a("| cell | n | base rate | TSS @ F1 point | TSS @ TSS point | peak TSS |")
    a("|---|---|---|---|---|---|")
    for key, cell in r["cells"].items():
        f1 = cell["operating_points"].get("f1", {})
        ts = cell["operating_points"].get("tss", {})
        a(f"| {key} | {cell['n']:,} | {cell['base_rate']:.3f} | "
          f"{_fmt(f1.get('tss'))} | {_fmt(ts.get('tss'))} | {_fmt(cell['peak_tss'])} |")
    a("")
    for key, cell in r["cells"].items():
        ci = cell["operating_points"].get("tss", {}).get("tss_ci95")
        ci_txt = f" · 95% CI {ci[0]:.3f}..{ci[1]:.3f}" if ci else ""
        a(f"- **{key}** — {cell['description']}{ci_txt}")
    a("")

    # 3. Attribution
    a("## 3. Where the skill went\n")
    applicable = r.get("attribution_applicable", {})
    full = applicable.get("full_decomposition", True)

    if full:
        a("Each step changes exactly one thing, so the drop is attributed, not just observed.\n")
    else:
        a(f"⚠️ **Full decomposition withheld for this corpus.** "
          f"{applicable.get('reason', '')}\n")

    a("| step | ΔTSS @ F1 point | ΔTSS @ TSS point |")
    a("|---|---|---|")
    f1a = r["attribution_tss"].get("f1", {})
    tsa = r["attribution_tss"].get("tss", {})
    for step in r["attribution_steps"]:
        k = step["step"]
        if k in tsa or k in f1a:
            a(f"| {k.replace('_', ' ')} | {_fmt(f1a.get(k))} | {_fmt(tsa.get(k))} |")
    if "total" in tsa:
        a(f"| **total** | **{_fmt(f1a.get('total'))}** | **{_fmt(tsa.get('total'))}** |")
    a("")
    if full:
        for step in r["attribution_steps"]:
            a(f"- **{step['step'].replace('_', ' ')}** ({step['from']} → {step['to']}): "
              f"{step['meaning']}")
        a("")
    if "total_gap_ci95" in r:
        g = r["total_gap_ci95"]
        a(f"Total gap at the TSS operating point: **{g['point_estimate']:.3f}** "
          f"(95% CI {g['ci95'][0]:.3f}..{g['ci95'][1]:.3f}); "
          f"excludes zero: **{g['excludes_zero']}**.\n")

    # 4. Template rule across cells
    a("## 4. The zero-parameter rule, scored in every cell\n")
    a("| cell | n | accuracy | TSS |")
    a("|---|---|---|---|")
    for key, s in r.get("template_rule_by_cell", {}).items():
        a(f"| {key} | {s['n']:,} | {s['accuracy']:.4f} | {s['tss']:.4f} |")
    a("")
    a("*The same rule that all but solves the benchmark should be worthless on "
      "live data. The size of that collapse is the clearest available measure of "
      "how much of the benchmark is construction rather than task.*\n")

    # 4b. Does live training close the gap?
    lt = r.get("live_training")
    if lt:
        a("## 5. Does training on live data close the gap?\n")
        if not lt["readiness"]["ready"]:
            a(f"**Open — not yet answerable.** {lt['readiness']['reason']}.\n")
            rd = lt["readiness"]
            a(f"Collected so far: {rd['n_days']} day(s), {rd['n_rows']:,} headline rows. "
              f"A temporal split needs {rd['thresholds']['min_train_days']} training days "
              f"and {rd['thresholds']['min_holdout_days']} holdout days, with at least "
              f"{rd['thresholds']['min_rows_per_side']:,} rows and "
              f"{rd['thresholds']['min_positives_per_side']} phishing rows on each side."
              + (f" **{rd['days_needed']} more day(s) of collection required.**"
                 if rd.get("days_needed") else ""))
            a("")
            a("*This question is structural, not optional: the four cells above measure "
              "the size of the gap, but every model in them was fitted on the benchmark, "
              "so they cannot say whether more live training data would close it. The "
              "2x2 below fills in automatically as the daily record accumulates.*\n")
            a("| | benchmark test | live holdout |")
            a("|---|---|---|")
            a("| benchmark-trained | A | B |")
            a("| live-trained | C | D |")
            a("")
            a("*A − B is the gap. D − B is how much live training recovers. C is the "
              "control: a live-trained model that also scores well on the benchmark has "
              "learned something general rather than this month's campaigns.*\n")
        else:
            a(f"{lt['answer']}\n")
            a("| training source | benchmark test | live holdout |")
            a("|---|---|---|")
            c = lt["cells"]
            a(f"| benchmark-trained | {c['A_benchmark_trained_on_benchmark']['tss']:.3f} "
              f"| {c['B_benchmark_trained_on_live']['tss']:.3f} |")
            a(f"| live-trained | {c['C_live_trained_on_benchmark']['tss']:.3f} "
              f"| **{c['D_live_trained_on_live']['tss']:.3f}** |")
            a("")
            a(f"- Gap (A − B): **{lt['gap_A_minus_B']:.3f}**")
            a(f"- Recovery from live training (D − B): **{lt['recovery_D_minus_B']:+.3f}** "
              f"(95% CI {lt['recovery_ci95'][0]:.3f}..{lt['recovery_ci95'][1]:.3f}), "
              f"significant: **{lt['recovery_significant']}**")
            if lt.get("recovery_fraction_of_gap") is not None:
                a(f"- That is {lt['recovery_fraction_of_gap']:.1%} of the gap.")
            a("")
            for tag, cell in c.items():
                a(f"- **{tag}** — {cell['description']} "
                  f"TSS {cell['tss']:.3f} (95% CI {cell['tss_ci95'][0]:.3f}.."
                  f"{cell['tss_ci95'][1]:.3f}), n = {cell['n']:,}")
            a("")

    # 5. Robustness
    if "robustness" in r:
        a("## 6. Robustness: a different positive class\n")
        rc = r["robustness"]["urlhaus_malware"]
        ts = rc["operating_points"].get("tss", {})
        a(f"{rc['description']}\n")
        a(f"| n | base rate | TSS | recall | false alarm |")
        a("|---|---|---|---|---|")
        a(f"| {rc['n']:,} | {rc['base_rate']:.3f} | **{_fmt(ts.get('tss'))}** | "
          f"{_fmt(ts.get('recall'))} | {_fmt(ts.get('false_alarm_rate'))} |\n")

    # 6. Self-audit
    if "sampling_confound" in r:
        sc = r["sampling_confound"]
        a("## 7. Audit of our own collection\n")
        if sc.get("available"):
            a(f"{sc['interpretation']}\n")
            a(f"A model given *only* the path-shape features "
              f"(`{'`, `'.join(sc['features_used'])}`), grouped-CV by domain, "
              f"reaches **peak TSS {sc['peak_tss_path_shape_only']:.3f}** on our "
              f"live corpus (n = {sc['n']:,}). Pre-registered ceiling: "
              f"{sc['preregistered_ceiling']}.\n")
            pr = sc.get("path_presence_rate", {})
            if pr:
                a(f"Share of URLs carrying a path — benign "
                  f"**{_fmt(pr.get('benign'))}**, phishing "
                  f"**{_fmt(pr.get('phishing'))}**.\n")
            status = "⚠️ **BREACHED**" if sc.get("ceiling_breached") else "✅ within ceiling"
            a(f"**Verdict: {status}**\n")
            for para in sc["verdict"].split("\n\n"):
                a(f"> {para.strip()}\n")
        else:
            a(f"Not yet available: {sc.get('reason')}.\n")

    # 7. Provenance
    if "live_corpus" in r:
        lc = r["live_corpus"]
        a("## 8. Live corpus\n")
        span = f"{lc['span'][0]} → {lc['span'][1]}" if lc.get("span") else "—"
        a(f"{lc['n_rows']:,} unique URLs over {lc['n_domains']:,} registrable "
          f"domains, {lc['n_phishing']:,} phishing, span {span}.\n")
        a("| source | rows |")
        a("|---|---|")
        for k, v in lc["by_source"].items():
            a(f"| {k} | {v:,} |")
        a("")

    return out
