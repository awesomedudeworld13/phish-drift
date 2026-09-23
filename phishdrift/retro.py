"""Retrospective live corpus (testing-new pre-registration).

Phishing: OpenPhish's community feed is a git repository whose history keeps
every past ``feed.txt``. A URL's first-seen date is the committer date of the
first commit containing it -- dated by OpenPhish, not by us.

Benign: for each month, that month's Common Crawl crawl, sampled like
``collect.fetch_commoncrawl_urls`` (same records, read over the CDN; see D1). Two samples per month: ``realistic``
(random) and ``path_matched`` (matched to that month's phishing sample on
path-depth bucket x has_query, the audit-breach fix).

    python -m phishdrift.retro            # resumable; caches each month's benign pool

Writes testing_new/retro/corpus.csv.gz.enc, encrypted (url, y, source, first_seen_utc,
snapshot_date, month, variant) and testing_new/retro/stats.json.
"""

from __future__ import annotations

import gzip
import json
import random
import subprocess
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from . import cc_cdn, collect

SEED = 20260920
SPAN = (date(2024, 10, 13), date(2026, 9, 19))
N_PHISH = 1500
N_BENIGN = 1500
POOL = 2500
OUT = Path("testing_new/retro")
CACHE = Path("data/cache")
OPENPHISH_GIT = "https://github.com/openphish/public_feed.git"


def openphish_first_seen(repo: Path = CACHE / "openphish_public_feed") -> dict[str, str]:
    """{url: first-seen ISO timestamp} over the whole feed history."""
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", OPENPHISH_GIT, str(repo)], check=True)
    else:
        subprocess.run(["git", "-C", str(repo), "pull", "-q"], check=True)
    log = subprocess.run(["git", "-C", str(repo), "log", "--reverse", "--format=%H %cI", "--", "feed.txt"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    cat = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    seen: dict[str, str] = {}
    for line in filter(None, log):
        sha, ts = line.split()
        cat.stdin.write(f"{sha}:feed.txt\n".encode())
        cat.stdin.flush()
        hdr = cat.stdout.readline().split()
        if len(hdr) < 3:                                  # "missing" -> file absent at that commit
            continue
        body = cat.stdout.read(int(hdr[2]))
        cat.stdout.read(1)
        when = datetime.fromisoformat(ts).astimezone(timezone.utc).isoformat(timespec="seconds")
        for u in body.decode(errors="replace").splitlines():
            u = u.strip()
            if u.startswith("http") and u not in seen:
                seen[u] = when
    cat.stdin.close()
    return seen


def months():
    m = SPAN[0].replace(day=1)
    while m <= SPAN[1]:
        yield m.strftime("%Y-%m")
        m = (m + timedelta(days=32)).replace(day=1)


# Crawl windows as listed by index.commoncrawl.org/collinfo.json on 2026-09-22
# (embedded because the index server was down during the build; see D1).
CRAWLS = """CC-MAIN-2026-39 2026-09-04 2026-09-17
CC-MAIN-2026-34 2026-08-07 2026-08-20
CC-MAIN-2026-30 2026-07-10 2026-07-23
CC-MAIN-2026-25 2026-06-05 2026-06-18
CC-MAIN-2026-21 2026-05-08 2026-05-21
CC-MAIN-2026-17 2026-04-10 2026-04-23
CC-MAIN-2026-12 2026-03-05 2026-03-17
CC-MAIN-2026-08 2026-02-06 2026-02-19
CC-MAIN-2026-04 2026-01-12 2026-01-25
CC-MAIN-2025-51 2025-12-04 2025-12-17
CC-MAIN-2025-47 2025-11-06 2025-11-19
CC-MAIN-2025-43 2025-10-05 2025-10-19
CC-MAIN-2025-38 2025-09-05 2025-09-18
CC-MAIN-2025-33 2025-08-02 2025-08-15
CC-MAIN-2025-30 2025-07-07 2025-07-20
CC-MAIN-2025-26 2025-06-12 2025-06-25
CC-MAIN-2025-21 2025-05-12 2025-05-25
CC-MAIN-2025-18 2025-04-17 2025-05-01
CC-MAIN-2025-13 2025-03-15 2025-03-28
CC-MAIN-2025-08 2025-02-06 2025-02-19
CC-MAIN-2025-05 2025-01-12 2025-01-26
CC-MAIN-2024-51 2024-12-01 2024-12-15
CC-MAIN-2024-46 2024-11-01 2024-11-15
CC-MAIN-2024-42 2024-10-03 2024-10-16"""


def crawl_for_month() -> dict[str, dict]:
    """{YYYY-MM: {"id", "from", "to"}} -- the crawl that started in that month."""
    out = {}
    for line in CRAWLS.splitlines():
        cid, f, t = line.split()
        out.setdefault(f[:7], {"id": cid, "from": f, "to": t})
    return out


def shape_bucket(url: str) -> tuple[int, int]:
    """(path-depth bucket 0/1/2/3+, has_query) -- the matching key for path_matched."""
    p = urlsplit(url)
    depth = len([s for s in p.path.split("/") if s])
    return min(depth, 3), int(bool(p.query))


def benign_pool(month: str, crawl: dict, domains: list[str]) -> list[str]:
    path = OUT / "pools" / f"{month}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(SEED + int(month.replace("-", "")))
    ds = domains[:]
    rng.shuffle(ds)
    urls: list[str] = []
    start = 0
    while len(urls) < POOL and start < len(ds):         # batches of 40 domains, like the collector
        urls += cc_cdn.fetch_commoncrawl_urls(ds[start:start + 40], per_domain=25,
                                              collection=crawl["id"], max_domains=40)
        start += 40
        print(f"  {month}: {len(urls)} benign after {start} domains", flush=True)
        if start >= 400:                                 # hard cap on queries per month
            break
    urls = list(dict.fromkeys(urls))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(urls), encoding="utf-8")
    return urls


def path_matched(phish: list[str], pool: list[str], rng: random.Random) -> tuple[list[str], int]:
    """Benign sample whose (depth bucket, has_query) mix equals the phishing sample's."""
    by = defaultdict(list)
    for u in pool:
        by[shape_bucket(u)].append(u)
    for v in by.values():
        rng.shuffle(v)
    need = defaultdict(int)
    for u in phish:
        need[shape_bucket(u)] += 1
    total = sum(need.values())
    out, replaced = [], 0
    for key, k in need.items():
        k = round(k * N_BENIGN / total)
        have = by.get(key, [])
        if not have:
            continue
        take = have[:k]
        if len(take) < k:                                 # pool short: resample within the bucket
            replaced += k - len(take)
            take += [rng.choice(have) for _ in range(k - len(take))]
        out += take
    return out, replaced


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("OpenPhish history ...", flush=True)
    seen = openphish_first_seen()
    by_month = defaultdict(list)
    for u, ts in seen.items():
        d = date.fromisoformat(ts[:10])
        if SPAN[0] <= d <= SPAN[1]:
            by_month[ts[:7]].append((u, ts))
    crawls = crawl_for_month()
    domains = collect.fetch_tranco_domains(50_000, cache=CACHE / "tranco.json")
    rows, stats = [], {"span": [str(SPAN[0]), str(SPAN[1])], "months": {}}
    for month in months():
        rng = random.Random(SEED + int(month.replace("-", "")))
        ph = sorted(by_month[month])
        rng.shuffle(ph)
        ph = ph[:N_PHISH]
        crawl = crawls[month]
        pool = benign_pool(month, crawl, domains)
        realistic = rng.sample(pool, min(N_BENIGN, len(pool)))
        matched, replaced = path_matched([u for u, _ in ph], pool, rng)
        c0 = date.fromisoformat(crawl["from"])
        span_days = (date.fromisoformat(crawl["to"]) - c0).days

        def bdate():
            return datetime.combine(c0 + timedelta(days=rng.randint(0, span_days)),
                                    datetime.min.time(), timezone.utc).isoformat(timespec="seconds")
        for u, ts in ph:
            rows.append({"url": u, "y": 1, "source": "openphish_history", "first_seen_utc": ts,
                         "variant": "both"})
        for variant, urls in (("realistic", realistic), ("path_matched", matched)):
            for u in urls:
                rows.append({"url": u, "y": 0, "source": f"commoncrawl_{crawl['id']}",
                             "first_seen_utc": bdate(), "variant": variant})
        stats["months"][month] = {"phishing": len(ph), "phishing_available": len(by_month[month]),
                                  "crawl": crawl["id"], "benign_pool": len(pool),
                                  "realistic": len(realistic), "path_matched": len(matched),
                                  "path_matched_resampled": replaced}
        print(f"{month}: {stats['months'][month]}", flush=True)
    frame = pd.DataFrame(rows)
    frame["snapshot_date"] = frame["first_seen_utc"].str[:10]
    frame["month"] = frame["snapshot_date"].str[:7]
    from . import sealed
    sealed.write_csv(frame, OUT / "corpus.csv.gz")       # encrypted: see DATA_HANDLING.md
    stats["rows"] = len(frame)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    build()
