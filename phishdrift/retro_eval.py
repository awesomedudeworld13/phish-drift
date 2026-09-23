"""Score the testing-new pre-registration (P1-P4) on the retrospective corpus.

    python -m phishdrift.retro_eval        # -> testing_new/retro_results.json

Everything here reuses main's frozen pieces: features, RandomForest recipe
(model.train), benchmark domain-disjoint models and test splits, and
benchgap.evaluate's domain-clustered bootstrap.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from benchgap import evaluate
from . import benchmark as bm, sealed
from .benchmark import Split
from .features import feature_matrix, registrable_domain
from .gap import sampling_confound_diagnostic
from .model import TrainedModel, train as train_rf

SEED = 20260920
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "testing_new" / "retro" / "corpus.csv.gz.enc"
OUT = ROOT / "testing_new" / "retro_results.json"
N_RES = 1000

P1 = {"fit": ("2024-10-13", "2025-10-31"), "val": ("2025-11-01", "2025-12-31"),
      "test": ("2026-01-01", "2026-09-19")}
P2_FIXED = {"fit": ("2024-10-13", "2024-12-17"), "val": ("2024-12-18", "2024-12-31")}


def load_corpus(variant: str) -> pd.DataFrame:
    f = sealed.read_csv(CORPUS)
    f = f[(f.variant == "both") | (f.variant == variant)].copy()
    f["domain"] = [registrable_domain(u) for u in f.url]
    return f.reset_index(drop=True)


def _between(f, span):
    return f[(f.snapshot_date >= span[0]) & (f.snapshot_date <= span[1])]


def date_split(f: pd.DataFrame, fit, val, test, name="retro") -> Split:
    """Explicit date windows, domains disjoint across each boundary (as livetrain.temporal_split)."""
    tr, va, te = _between(f, fit).copy(), _between(f, val).copy(), _between(f, test).copy()
    seen = set(tr.domain)
    va = va[~va.domain.isin(seen)]
    seen |= set(va.domain)
    te = te[~te.domain.isin(seen)]
    return Split(name=name, train=tr, val=va, test=te)


def train_hgb(split: Split) -> TrainedModel:
    est = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         class_weight="balanced", random_state=SEED)
    est.fit(feature_matrix(split.train.url), split.train.y.to_numpy())
    pv = est.predict_proba(feature_matrix(split.val.url))[:, 1]
    thr = {"f1": evaluate.select_threshold(split.val.y.to_numpy(), pv, "f1"),
           "tss": evaluate.select_threshold(split.val.y.to_numpy(), pv, "tss")}
    return TrainedModel(estimator=est, feature_names=(), thresholds=thr,
                        metadata={"estimator": "HistGradientBoostingClassifier"})


def cell(model, frame) -> dict:
    y, p = frame.y.to_numpy(), model.predict_proba(frame.url)
    thr = model.thresholds["tss"]
    s = evaluate.score_at(y, p, thr).to_dict()
    lo, hi = evaluate.cluster_bootstrap_ci(y, p, thr, groups=frame.domain.to_numpy(), n_resamples=N_RES)
    return {"tss": round(s["tss"], 4), "ci95": [round(lo, 4), round(hi, 4)], "n": int(len(y)),
            "positives": int(y.sum())}


def recovery(live_model, bench_model, test) -> dict:
    y = test.y.to_numpy()
    pt, lo, hi = evaluate.paired_difference_ci(
        y, live_model.predict_proba(test.url), live_model.thresholds["tss"],
        y, bench_model.predict_proba(test.url), bench_model.thresholds["tss"],
        groups_a=test.domain.to_numpy(), groups_b=test.domain.to_numpy(), n_resamples=N_RES)
    return {"value": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)], "excludes_zero": bool(lo > 0)}


def benchmarks():
    """{key: (domain-disjoint benchmark model, its domain-disjoint test frame)}."""
    out = {}
    for key, source in bm.BENCHMARKS.items():
        frame = source.load(ROOT / "data")
        test = bm.domain_disjoint_split(frame, seed=SEED).test
        out[key] = (TrainedModel.load(ROOT / "models" / f"{key}_rf_domain_disjoint.joblib"), test)
    return out


def p1(f: pd.DataFrame, benches: dict, learner=train_rf) -> dict:
    split = date_split(f, **P1)
    live = learner(split)
    res = {"split_sizes": {k: {"n": len(v), "phishing": int(v.y.sum())}
                           for k, v in (("fit", split.train), ("val", split.val), ("test", split.test))},
           "D_live_on_live": cell(live, split.test), "benchmarks": {}}
    for key, (bmod, btest) in benches.items():
        a, b, c = cell(bmod, btest), cell(bmod, split.test), cell(live, btest)
        res["benchmarks"][key] = {"A_bench_on_bench": a, "B_bench_on_live": b, "C_live_on_bench": c,
                                  "gap_A_minus_B": round(a["tss"] - b["tss"], 4),
                                  "recovery_D_minus_B": recovery(live, bmod, split.test)}
    return res, live, split


def _tss(y, pred):
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return np.nan
    return pred[pos].mean() - pred[neg].mean()


P2_FIXED_D2 = {"fit": ("2024-10-13", "2024-11-30"), "val": ("2024-12-01", "2024-12-31")}


def _month_span(m: str) -> tuple[str, str]:
    return f"{m}-01", pd.Period(m).end_time.date().isoformat()


def p2(f: pd.DataFrame, corrected: bool = False) -> dict:
    """corrected=True is deviation D2: validation = the whole previous month (the pre-registered
    'last 14 days' windows often held zero benign rows, since benign dates sit inside each
    month's crawl window)."""
    fx = P2_FIXED_D2 if corrected else P2_FIXED
    fixed = train_rf(date_split(f, fx["fit"], fx["val"], ("9999", "9999")))
    test_months = sorted(m for m in f.month.unique() if m >= "2025-01")
    per, seen_fixed = [], set(_between(f, (fx["fit"][0], fx["val"][1])).domain)
    for i, m in enumerate(test_months):
        prev = [str(p)[:7] for p in pd.period_range(end=pd.Period(m) - 1, periods=3, freq="M")]
        fit_end = (pd.Period(prev[-1]).end_time - pd.Timedelta(days=14)).date().isoformat()
        val_start = (pd.Period(prev[-1]).end_time - pd.Timedelta(days=13)).date().isoformat()
        if corrected:                                      # fit m-3..m-2, validate on all of m-1
            sp = date_split(f, (_month_span(prev[0])[0], _month_span(prev[1])[1]), _month_span(prev[2]),
                            _month_span(m))
        else:
            sp = date_split(f, (f"{prev[0]}-01", fit_end), (val_start, pd.Period(prev[-1]).end_time.date().isoformat()),
                            (f"{m}-01", pd.Period(m).end_time.date().isoformat()))
        refreshed = train_rf(sp)
        te = sp.test[~sp.test.domain.isin(seen_fixed)]      # disjoint from BOTH models' training domains
        per.append({"month": m, "elapsed": i + 1, "y": te.y.to_numpy(), "g": te.domain.to_numpy(),
                    "fixed": fixed.predict_proba(te.url) >= fixed.thresholds["tss"],
                    "refreshed": refreshed.predict_proba(te.url) >= refreshed.thresholds["tss"]})
        print(f"  P2 {m}: n={len(te)}", flush=True)
    rng = np.random.default_rng(SEED)
    x = np.array([r["elapsed"] for r in per], dtype=float)

    def stats(idx_by_month):
        fx = np.array([_tss(r["y"][ix], r["fixed"][ix]) for r, ix in zip(per, idx_by_month)])
        rf = np.array([_tss(r["y"][ix], r["refreshed"][ix]) for r, ix in zip(per, idx_by_month)])
        ok = ~np.isnan(fx) & ~np.isnan(rf)
        return np.polyfit(x[ok], fx[ok], 1)[0], float(np.mean(rf[ok] - fx[ok])), fx, rf

    full = [np.arange(len(r["y"])) for r in per]
    slope, adv, fx, rf = stats(full)
    groups = [{d: np.where(r["g"] == d)[0] for d in np.unique(r["g"])} for r in per]
    bs, ba = [], []
    for _ in range(N_RES):
        idx = []
        for gm in groups:
            keys = list(gm)
            pick = rng.integers(0, len(keys), len(keys))
            idx.append(np.concatenate([gm[keys[k]] for k in pick]))
        s, a, _, _ = stats(idx)
        bs.append(s)
        ba.append(a)
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"months": [{"month": r["month"], "n": int(len(r["y"])), "fixed_tss": round(float(a), 4),
                        "refreshed_tss": round(float(b), 4)} for r, a, b in zip(per, fx, rf)],
            "H9_fixed_slope_per_month": {"value": round(float(slope), 5), "ci95": ci(bs)},
            "H10_refresh_advantage": {"value": round(adv, 4), "ci95": ci(ba)}}


def final_model() -> tuple[TrainedModel, set]:
    """F: the P1 recipe refit on the whole realistic corpus (validation = its last 7 weeks)."""
    f = load_corpus("realistic")
    split = date_split(f, ("2024-10-13", "2026-07-31"), ("2026-08-01", "2026-09-19"), ("9999", "9999"))
    return train_rf(split), set(split.train.domain) | set(split.val.domain)


def score_prospective() -> dict:
    """L1: F vs each benchmark model on branch-collected rows (testing_new/live/), F's domains excluded."""
    frames = [sealed.read_csv(p) for p in sealed.data_files(ROOT / "testing_new" / "live")]
    if not frames:
        return {"ready": False, "reason": "no branch snapshots yet"}
    live = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"])
    live["domain"] = [registrable_domain(u) for u in live.url]
    F, seen = final_model()
    live = live[~live.domain.isin(seen)]
    out = {"n": int(len(live)), "phishing": int(live.y.sum()),
           "span": [str(live.first_seen_utc.min())[:10], str(live.first_seen_utc.max())[:10]],
           "F_on_live": cell(F, live), "benchmarks": {}}
    for key, (bmod, _) in benchmarks().items():
        out["benchmarks"][key] = {"bench_on_live": cell(bmod, live), "L1_F_minus_bench": recovery(F, bmod, live)}
    (ROOT / "testing_new" / "live_score.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    benches = benchmarks()
    out = {"preregistration": "testing_new/PREREGISTRATION.md"}
    real = load_corpus("realistic")
    print("P1 (RF, realistic) ...", flush=True)
    out["P1_realistic_rf"], live_rf, split = p1(real, benches)
    print("P3 (HGB) ...", flush=True)
    hgb = train_hgb(split)
    d_h = cell(hgb, split.test)
    out["P3_hgb"] = {"D_live_on_live_hgb": d_h,
                     "D_rf_minus_hgb": round(out["P1_realistic_rf"]["D_live_on_live"]["tss"] - d_h["tss"], 4)}
    print("P4 (path-matched) ...", flush=True)
    matched = load_corpus("path_matched")
    out["P4_path_matched_rf"], _, msplit = p1(matched, benches)
    out["P4_audit_path_matched"] = sampling_confound_diagnostic(msplit.test)
    out["P4_audit_realistic_same_months"] = sampling_confound_diagnostic(split.test)
    print("P2 (decay) ...", flush=True)
    out["P2_decay_realistic_rf"] = p2(real)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    import sys
    if "--p2-corrected" in sys.argv:
        res = json.loads(OUT.read_text(encoding="utf-8"))
        res["P2_decay_corrected_D2"] = p2(load_corpus("realistic"), corrected=True)
        OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        print(json.dumps(res["P2_decay_corrected_D2"], indent=2))
    elif "--prospective" in sys.argv:
        print(json.dumps(score_prospective(), indent=2))
    else:
        main()
