"""Command line entry points.

    python -m phishdrift.cli audit                 # construction audit, both corpora
    python -m phishdrift.cli train  --benchmark X  # fit both models for one corpus
    python -m phishdrift.cli collect               # append one daily live snapshot
    python -m phishdrift.cli report                # run all cells for every corpus

``--benchmark`` accepts any key in ``benchmark.BENCHMARKS`` or ``all``.
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
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_MD = ROOT / "RESULTS.md"
DOCS_JSON = ROOT / "docs" / "results.json"
LIVE_DIR = DATA / "live"

SEED = 20260920


def _model_paths(key: str) -> tuple[Path, Path]:
    return (MODELS / f"{key}_rf_random_split.joblib",
            MODELS / f"{key}_rf_domain_disjoint.joblib")


def _resolve(selection: str) -> list[str]:
    if selection == "all":
        return list(bm.BENCHMARKS)
    if selection not in bm.BENCHMARKS:
        raise SystemExit(
            f"unknown benchmark {selection!r}; "
            f"choose from {', '.join(bm.BENCHMARKS)} or 'all'"
        )
    return [selection]


def _load(key: str):
    source = bm.BENCHMARKS[key]
    print(f"[{key}] ensuring {source.filename} ...", flush=True)
    frame = source.load(DATA)
    print(f"[{key}] {len(frame):,} URLs, {frame.y.sum():,} phishing "
          f"({frame.y.mean():.1%}), {frame.domain.nunique():,} registrable domains",
          flush=True)
    return source, frame


# --------------------------------------------------------------------------

def cmd_audit(args) -> int:
    for key in _resolve(args.benchmark):
        source, frame = _load(key)
        audit = bm.template_audit(frame)
        rates = bm.structural_degeneracy(frame)

        print(f"\n=== {source.title} ===")
        print(f"  rule        : URL does NOT match ^https://www\\.[^/?#]+/?$  ->  phishing")
        print(f"  n           : {audit.n:,}")
        print(f"  accuracy    : {audit.accuracy:.4f}")
        print(f"  precision   : {audit.precision:.4f}")
        print(f"  recall      : {audit.recall:.4f}")
        print(f"  false alarm : {audit.false_alarm_rate:.4f}")
        print(f"  TSS         : {audit.tss:.4f}")
        print(f"  benign matching the template   : {audit.benign_template_share:.4f}")
        print(f"  phishing matching the template : {audit.phish_template_share:.4f}")
        print(f"\n  {'property':<12} {'benign':>10} {'phishing':>10}")
        for prop, vals in rates.items():
            flag = "  <-- degenerate" if vals["benign"] in (0.0, 1.0) else ""
            print(f"  {prop:<12} {vals['benign']:>10.4f} {vals['phishing']:>10.4f}{flag}")

    print("\n  A rate of exactly 0.0000 or 1.0000 on one side means the property")
    print("  is not a feature — it is the class label in a different alphabet.")
    return 0


def cmd_train(args) -> int:
    for key in _resolve(args.benchmark):
        _, frame = _load(key)

        split_random = bm.random_split(frame, seed=SEED)
        split_disjoint = bm.domain_disjoint_split(frame, seed=SEED)
        print(f"[{key}] random   {split_random.sizes()}")
        print(f"[{key}] disjoint {split_disjoint.sizes()}")

        path_random, path_disjoint = _model_paths(key)
        for split, path in ((split_random, path_random), (split_disjoint, path_disjoint)):
            print(f"\n[{key}] training {split.name} ...", flush=True)
            model = train_model(split, n_estimators=args.n_estimators, seed=SEED)
            model.metadata["benchmark"] = key
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
        cache_dir=DATA / "cache",
    )
    path = live_collect.write_snapshot(snap, LIVE_DIR)
    print(f"[collect] {snap.date}: {json.dumps(snap.stats)}")
    print(f"[collect] -> {path.relative_to(ROOT)}")
    return 0


def cmd_report(args) -> int:
    live = live_collect.load_live(LIVE_DIR)
    n_snapshots = live.snapshot_date.nunique() if len(live) else 0
    print(f"[live] {len(live):,} unique URLs across {n_snapshots} snapshots", flush=True)
    if live.empty:
        print("[live] no snapshots yet — cells 3 and 4 will be omitted.", flush=True)

    by_key: dict[str, dict] = {}
    for key in _resolve(args.benchmark):
        path_random, path_disjoint = _model_paths(key)
        if not path_random.exists() or not path_disjoint.exists():
            print(f"error: models for {key!r} not found. Run "
                  f"`python -m phishdrift.cli train --benchmark {key}` first.",
                  file=sys.stderr)
            return 1

        source, frame = _load(key)
        print(f"[{key}] running cells ...", flush=True)
        results = gap.run(
            benchmark=frame,
            random_split_obj=bm.random_split(frame, seed=SEED),
            disjoint_split_obj=bm.domain_disjoint_split(frame, seed=SEED),
            model_random=TrainedModel.load(path_random),
            model_disjoint=TrainedModel.load(path_disjoint),
            live=live,
            n_resamples=args.resamples,
            benchmark_meta={"key": key, "title": source.title,
                            "citation": source.citation},
        )
        by_key[key] = results

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{key}.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")

        for cell_key, cell in results["cells"].items():
            ts = cell["operating_points"].get("tss", {})
            print(f"         {cell_key:<32} TSS {ts.get('tss', float('nan')):+.3f} "
                  f"(n={cell['n']:,})")

    document = gap.combine(by_key)
    RESULTS_MD.write_text(gap.render_markdown(document), encoding="utf-8")
    DOCS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_JSON.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"\n[report] -> results/*.json, RESULTS.md, docs/results.json")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="phishdrift",
        description="Measure how much of a phishing benchmark's score survives deployment.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_benchmark_flag(p):
        p.add_argument("--benchmark", default="all",
                       help=f"{', '.join(bm.BENCHMARKS)} or 'all' (default: all)")

    p = sub.add_parser("audit", help="construction audit only; no training")
    add_benchmark_flag(p)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("train", help="fit both models and freeze thresholds")
    add_benchmark_flag(p)
    p.add_argument("--n-estimators", type=int, default=300)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("collect", help="append one daily live snapshot")
    p.add_argument("--benign", type=int, default=300,
                   help="target benign URLs per variant per day")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("report", help="run all cells and write results")
    add_benchmark_flag(p)
    p.add_argument("--resamples", type=int, default=1000,
                   help="bootstrap resamples (lower for a fast local run)")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
