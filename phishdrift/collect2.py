"""testing-new daily collector: every OpenPhish URL, not one snapshot.

Main's collector (``collect.py``) saves the feed as it stands at 06:20 UTC, so
URLs that enter and leave the ~300-URL window between runs are never seen.
This one reads the feed's git history instead and keeps every URL whose
first-seen commit falls after the previous run's cutoff (~500/day), plus
600 realistic Common Crawl benign URLs. It writes to testing_new/live/, never
to data/live/, so main's prospective record is untouched.

    python -m phishdrift.collect2
"""

from __future__ import annotations

import gzip
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import cc_cdn, collect

LIVE = Path("testing_new/live")
REPO = Path("data/cache/openphish_recent")
START = "2026-09-23T00:00:00+00:00"            # pre-registered start of the branch record
N_BENIGN = 600


def _first_seen_recent() -> dict[str, str]:
    """{url: first-seen UTC ISO}, from a shallow history that starts well before START."""
    if not REPO.exists():
        subprocess.run(["git", "clone", "-q", "--shallow-since=2026-09-01", collect_git(), str(REPO)], check=True)
    else:
        subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "--shallow-since=2026-09-01", "origin"], check=True)
        subprocess.run(["git", "-C", str(REPO), "reset", "-q", "--hard", "origin/HEAD"], check=True)
    log = subprocess.run(["git", "-C", str(REPO), "log", "--reverse", "--format=%H %cI", "--", "feed.txt"],
                         capture_output=True, text=True, check=True).stdout.split()
    seen: dict[str, str] = {}
    for sha, ts in zip(log[::2], log[1::2]):
        body = subprocess.run(["git", "-C", str(REPO), "show", f"{sha}:feed.txt"],
                              capture_output=True, text=True).stdout
        when = datetime.fromisoformat(ts).astimezone(timezone.utc).isoformat(timespec="seconds")
        for u in body.splitlines():
            u = u.strip()
            if u.startswith("http") and u not in seen:
                seen[u] = when
    return seen


def collect_git() -> str:
    from .retro import OPENPHISH_GIT
    return OPENPHISH_GIT


def _latest_crawl() -> str:
    """Newest crawl id from the CDN's crawl list (the index server's collinfo can be down)."""
    import requests
    r = requests.get("https://data.commoncrawl.org/crawl-data/index.html", timeout=60,
                     headers=collect._HEADERS)
    import re
    return sorted(set(re.findall(r"CC-MAIN-\d{4}-\d{2}", r.text)))[-1]


def last_cutoff() -> str:
    cuts = [json.loads(p.read_text(encoding="utf-8"))["phish_through"] for p in LIVE.glob("*.stats.json")]
    return max(cuts, default=START)


def run() -> Path | None:
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")
    from . import sealed
    path = LIVE / f"{day}.csv.gz{sealed.SUFFIX}"
    if path.exists() or (LIVE / f"{day}.csv.gz").exists():   # immutable, one file per UTC day
        print(f"{path} exists; nothing to do")
        return None
    cutoff = last_cutoff()
    seen = _first_seen_recent()
    phish = sorted((u, ts) for u, ts in seen.items() if cutoff < ts <= now.isoformat())
    through = max((ts for _, ts in phish), default=cutoff)
    rng = random.Random(int(day.replace("-", "")))
    domains = collect.fetch_tranco_domains(50_000, cache=Path("data/cache/tranco.json"))
    rng.shuffle(domains)
    crawl = _latest_crawl()
    benign: list[str] = []
    for start in range(0, 400, 40):
        if len(benign) >= N_BENIGN:
            break
        benign += cc_cdn.fetch_commoncrawl_urls(domains[start:start + 40], per_domain=25,
                                                collection=crawl, max_domains=40)
    benign = list(dict.fromkeys(benign))[:N_BENIGN]
    stamp = now.isoformat(timespec="seconds")
    rows = ([{"url": u, "y": 1, "source": "openphish_git", "first_seen_utc": ts} for u, ts in phish]
            + [{"url": u, "y": 0, "source": "commoncrawl", "first_seen_utc": stamp} for u in benign])
    frame = pd.DataFrame(rows).drop_duplicates(subset=["url"])
    LIVE.mkdir(parents=True, exist_ok=True)
    sealed.write_csv(frame, path)                       # encrypted: see DATA_HANDLING.md
    stats = {"date": day, "collected_utc": stamp, "phish_after": cutoff, "phish_through": through,
             "openphish_git": len(phish), "benign_realistic": len(benign), "rows": len(frame)}
    (LIVE / f"{day}.stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(stats)
    return path


if __name__ == "__main__":
    run()
