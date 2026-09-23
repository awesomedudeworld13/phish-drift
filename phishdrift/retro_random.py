"""Exploratory, NOT pre-registered: P1 with benign URLs from random crawl blocks.

Why: the pre-registered benign samples come from the Tranco top-50,000, so every
benign URL is a popular site. Domain-only features separate them from phishing
at peak TSS ~0.6-0.66 even after path matching, which means the P1 recovery can
be carried by "popular site or not" -- the same kind of construction artifact
this project flags in Kaitholikkal (Majestic Million). This variant draws benign
URLs from uniformly random CDX blocks of each month's crawl instead.

The pre-registered corpus and results are untouched. Phishing rows are the same.

    python -m phishdrift.retro_random          # resumable; caches pools per month
"""

from __future__ import annotations

import gzip
import json
import random
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from . import cc_cdn
from .retro import OUT, SEED, crawl_for_month, months

N_RANDOM = 500
POOLS = OUT / "pools_random"
CORPUS_RANDOM = OUT / "corpus_random_benign.csv.gz"


def pool(month: str, crawl: dict) -> list[str]:
    path = POOLS / f"{month}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(SEED + 7 + int(month.replace("-", "")))
    urls = cc_cdn.random_urls(crawl["id"], N_RANDOM, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(urls), encoding="utf-8")
    print(f"{month}: {len(urls)} random-crawl benign", flush=True)
    return urls


def build() -> pd.DataFrame:
    rows = []
    for month, crawl in ((m, crawl_for_month()[m]) for m in months()):
        rng = random.Random(SEED + 11 + int(month.replace("-", "")))
        c0 = date.fromisoformat(crawl["from"])
        span = (date.fromisoformat(crawl["to"]) - c0).days
        for u in pool(month, crawl):
            d = datetime.combine(c0 + timedelta(days=rng.randint(0, span)), datetime.min.time(), timezone.utc)
            rows.append({"url": u, "y": 0, "source": f"commoncrawl_random_{crawl['id']}",
                         "first_seen_utc": d.isoformat(timespec="seconds"), "variant": "crawl_random"})
    frame = pd.DataFrame(rows)
    frame["snapshot_date"] = frame.first_seen_utc.str[:10]
    frame["month"] = frame.snapshot_date.str[:7]
    with gzip.open(CORPUS_RANDOM, "wt", encoding="utf-8", newline="") as fh:
        frame.to_csv(fh, index=False)
    return frame


DOMAIN_FEATURES = (
    "domain_length", "subdomain_length", "entropy_domain", "domain_digit_ratio", "domain_hyphen_count",
    "domain_vowel_ratio", "domain_max_consonant_run", "domain_max_repeat_run", "domain_digit_group_count",
    "domain_token_count", "domain_longest_token", "n_subdomains", "has_www", "is_ip_domain", "has_punycode",
    "tld_length", "tld_is_common", "tld_is_abused", "is_shortener", "is_https")
RESULTS = OUT.parent / "retro_random_results.json"


def evaluate_variant() -> dict:
    """P1 on random-crawl benign, plus both audits on every benign sample's test months."""
    from .features import PATH_SHAPE_FEATURES, registrable_domain
    from .gap import sampling_confound_diagnostic
    from .retro_eval import CORPUS, P1, benchmarks, cell, date_split, load_corpus, p1

    rnd = pd.read_csv(CORPUS_RANDOM)
    from . import sealed
    ph = sealed.read_csv(CORPUS)
    f = pd.concat([ph[ph.variant == "both"], rnd], ignore_index=True)
    f["domain"] = [registrable_domain(u) for u in f.url]
    benches = benchmarks()
    out = {"note": "EXPLORATORY, not pre-registered. Benign = random Common Crawl blocks, not Tranco top-50k."}
    out["P1_crawl_random_rf"], _, split = p1(f, benches)

    # The operational question: a live model trained the pre-registered way (popular-site
    # benign) scored on random-crawl benign it never saw, domains disjoint from its training.
    real = load_corpus("realistic")
    _, live_real, rsplit = p1(real, {})
    seen = set(rsplit.train.domain) | set(rsplit.val.domain)
    out["tranco_trained_on_crawl_random_test"] = cell(live_real, split.test[~split.test.domain.isin(seen)])

    audits = {}
    for name, frame in (("realistic", real), ("path_matched", load_corpus("path_matched")), ("crawl_random", f)):
        t = date_split(frame, **P1).test
        audits[name] = {feat: sampling_confound_diagnostic(t, features=fs)["peak_tss_path_shape_only"]
                        for feat, fs in (("path_shape", PATH_SHAPE_FEATURES), ("domain_only", DOMAIN_FEATURES))}
    out["audits_peak_tss"] = audits
    RESULTS.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    if "--evaluate" in sys.argv:
        print(json.dumps(evaluate_variant(), indent=2, default=str))
    else:
        f = build()
        print(len(f), "rows ->", CORPUS_RANDOM)
