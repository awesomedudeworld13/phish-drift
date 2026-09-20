"""Command line entry points.

    python -m phishdrift.cli audit     # construction audit only (no training, ~30 s)
    python -m phishdrift.cli train     # fit both models, freeze thresholds
    python -m phishdrift.cli collect   # append one daily live snapshot
    python -m phishdrift.cli report    # run all cells, write results.json + RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import benchmark as bm
from . import collect as live_collect
from . import gap
from .model import TrainedModel, train as train_model

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data" / "phiusiil.zip"
MODEL_RANDOM = ROOT / "models" / "rf_random_split.joblib"
MODEL_DISJOINT = ROOT / "models" / "rf_domain_disjoint.joblib"
RESULTS_JSON = ROOT / "results.json"
RESULTS_MD = ROOT / "RESULTS.md"
DOCS_JSON = ROOT / "docs" / "results.json"
LIVE_DIR = ROOT / "data" / "live"

SEED = 20260920


def _load_benchmark():
    print(f"[benchmark] ensuring {ARCHIVE.name} ...", flush=True)
    bm.download(ARCHIVE)
    frame = bm.load(ARCHIVE)
    print(f"[benchmark] {len(frame):,} URLs, "
          f"{frame.y.sum():,} phishing ({frame.y.mean():.1%}), "
          f"{frame.domain.nunique():,} registrable domains", flush=True)
    return frame


def cmd_audit(args) -> int:
    frame = _load_benchmark()
    audit = bm.template_audit(frame)
    rates = bm.structural_degeneracy(frame)

    print("\n=== Construction audit: the zero-parameter template rule ===")
    print(f"  rule        : URL does NOT match ^https://www\\.[^/?#]+/?$  ->  phishing")
    print(f"  n           : {audit.n:,}")
    print(f"  accuracy    : {audit.accuracy:.4f}")
    print(f"  precision   : {audit.precision:.4f}")
    print(f"  recall      : {audit.recall:.4f}")
    print(f"  false alarm : {audit.false_alarm_rate:.4f}")
    print(f"  TSS         : {audit.tss:.4f}")
    print(f"\n  benign matching the template   : {audit.benign_template_share:.4f}")
    print(f"  phishing matching the template : {audit.phish_template_share:.4f}")

    print("\n=== Per-class surface-property rates ===")
    print(f"  {'property':<12} {'benign':>10} {'phishing':>10}")
    for prop, vals in rates.items():
        print(f"  {prop:<12} {vals['benign']:>10.4f} {vals['phishing']:>10.4f}")
    print("\n  A rate of exactly 0.0000 or 1.0000 on one side means the property")
    print("  is not a feature — it is the class label in a different alphabet.")
    return 0


def cmd_train(args) -> int:
    frame = _load_benchmark()

    print("\n[split] random i.i.d. (the published protocol) ...", flush=True)
    split_random = bm.random_split(frame, seed=SEED)
    print(f"        {split_random.sizes()}", flush=True)

    print("[split] domain-disjoint (no eTLD+1 shared) ...", flush=True)
    split_disjoint = bm.domain_disjoint_split(frame, seed=SEED)
    print(f"        {split_disjoint.sizes()}", flush=True)

    for split, path in ((split_random, MODEL_RANDOM), (split_disjoint, MODEL_DISJOINT)):
        print(f"\n[train] {split.name} ...", flush=True)
        model = train_model(split, n_estimators=args.n_estimators, seed=SEED)
        model.save(path)
        print(f"        thresholds {model.metadata['thresholds']}")
        print(f"        top features: "
              f"{', '.join(f['name'] for f in model.metadata['top_features'][:6])}")
        print(f"        saved -> {path.relative_to(ROOT)}")
    return 0


def cmd_collect(args) -> int:
    print("[collect] building snapshot ...", flush=True)
    snap = live_collect.build_snapshot(
        n_benign_realistic=args.benign,
        n_benign_templated=args.benign,
        cache_dir=ROOT / "data" / "cache",
    )
    path = live_collect.write_snapshot(snap, LIVE_DIR)
    print(f"[collect] {snap.date}: {json.dumps(snap.stats)}")
    print(f"[collect] -> {path.relative_to(ROOT)}")
    return 0


def cmd_report(args) -> int:
    if not MODEL_RANDOM.exists() or not MODEL_DISJOINT.exists():
        print("error: models not found. Run `python -m phishdrift.cli train` first.",
              file=sys.stderr)
        return 1

    frame = _load_benchmark()
    split_random = bm.random_split(frame, seed=SEED)
    split_disjoint = bm.domain_disjoint_split(frame, seed=SEED)

    model_random = TrainedModel.load(MODEL_RANDOM)
    model_disjoint = TrainedModel.load(MODEL_DISJOINT)

    live = live_collect.load_live(LIVE_DIR)
    print(f"[live] {len(live):,} unique URLs across "
          f"{live.snapshot_date.nunique() if len(live) else 0} snapshots", flush=True)
    if live.empty:
        print("[live] no snapshots yet — cells 3 and 4 will be omitted.", flush=True)

    print("[report] running cells ...", flush=True)
    results = gap.run(
        benchmark=frame,
        random_split_obj=split_random,
        disjoint_split_obj=split_disjoint,
        model_random=model_random,
        model_disjoint=model_disjoint,
        live=live,
        n_resamples=args.resamples,
    )

    gap.write(results, RESULTS_JSON, RESULTS_MD)
    DOCS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"[report] -> {RESULTS_JSON.name}, {RESULTS_MD.name}, docs/results.json")
    for key, cell in results["cells"].items():
        ts = cell["operating_points"].get("tss", {})
        print(f"         {key:<34} TSS {ts.get('tss', float('nan')):.3f}  (n={cell['n']:,})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="phishdrift",
        description="Measure how much of a phishing benchmark's score survives deployment.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit", help="construction audit only; no training")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("train", help="fit both models and freeze thresholds")
    p.add_argument("--n-estimators", type=int, default=300)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("collect", help="append one daily live snapshot")
    p.add_argument("--benign", type=int, default=300,
                   help="target benign URLs per variant per day")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("report", help="run all cells and write results")
    p.add_argument("--resamples", type=int, default=1000,
                   help="bootstrap resamples (lower for a fast local run)")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
