"""Daily live-data collection.

Runs unattended in CI (``.github/workflows/collect.yml``) and appends one
immutable, git-tracked snapshot per UTC day to ``data/live/``.

Streams
-------
**Phishing (positive class)** -- the OpenPhish community feed, refreshed every
12 hours. The feed *is* the label; no annotation step exists, so there is no
annotator bias to argue about. URLhaus is collected in parallel and tagged
separately: it lists *malware distribution* URLs, which is a related but
different task, so it is retained for a robustness check and excluded from the
headline task by default.

**Benign (negative class)** -- two variants, and collecting both is the point:

``realistic``  Real URLs with real paths, drawn from the Common Crawl index for
               domains sampled out of the Tranco top-1M. This is what benign
               traffic actually looks like.
``templated``  The same Tranco domains rendered as ``https://www.<domain>``,
               which reproduces the surface form of PhiUSIIL's benign class.

The benchmark's benign class is structurally degenerate (see
``benchmark.template_audit``): every legitimate URL in it is pathless,
queryless, https and www-prefixed. Scoring a benchmark-trained model against
``templated`` benign URLs holds that degeneracy fixed; scoring it against
``realistic`` ones removes it. The difference between the two isolates how much
of the operational collapse is caused by the benchmark's construction rather
than by genuine drift -- which is a question we cannot answer with one benign
stream, however carefully chosen.

Known limitation
----------------
The phishing stream is genuinely live and daily. The benign stream is quasi-
static: Common Crawl publishes monthly, so benign URLs are fresh to within
weeks, not hours. Benign web structure moves far more slowly than phishing
infrastructure does, so this is an acceptable asymmetry, but it is an asymmetry
and it is recorded in every snapshot's ``source`` column rather than smoothed
over.
"""

from __future__ import annotations

import gzip
import io
import json
import random
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OPENPHISH_FEED = "https://raw.githubusercontent.com/openphish/public_feed/main/feed.txt"
URLHAUS_RECENT = "https://urlhaus.abuse.ch/downloads/csv_recent/"
TRANCO_API = "https://tranco-list.eu/api/lists/date/{date}"
TRANCO_LATEST = "https://tranco-list.eu/top-1m.csv.zip"
CC_COLLINFO = "https://index.commoncrawl.org/collinfo.json"
CC_INDEX = "https://index.commoncrawl.org/{collection}-index"

USER_AGENT = (
    "phish-drift/1.0 (science-fair research project; "
    "benchmark-vs-operational evaluation; contact via GitHub issues)"
)
_HEADERS = {"User-Agent": USER_AGENT}

# Politeness delay between Common Crawl index queries. Their index is a free
# public service and we are a cron job; going faster buys nothing.
CC_DELAY_SECONDS = 1.5


@dataclass
class Snapshot:
    date: str
    frame: pd.DataFrame
    stats: dict


def _get(url: str, **kwargs) -> requests.Response:
    r = requests.get(url, headers=_HEADERS, timeout=kwargs.pop("timeout", 90), **kwargs)
    r.raise_for_status()
    return r


# --------------------------------------------------------------------------
# Positive class
# --------------------------------------------------------------------------

def fetch_openphish() -> list[str]:
    """Current OpenPhish community feed (~300 URLs, rotated every 12 h)."""
    text = _get(OPENPHISH_FEED).text
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("http")]


def fetch_urlhaus(limit: int = 2000) -> list[str]:
    """Recent URLhaus malware-distribution URLs. Tagged separately; not the headline task."""
    text = _get(URLHAUS_RECENT).text
    urls = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # id,dateadded,url,url_status,... -- quoted CSV
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 2 and parts[2].startswith("http"):
            urls.append(parts[2])
        if len(urls) >= limit:
            break
    return urls


# --------------------------------------------------------------------------
# Negative class
# --------------------------------------------------------------------------

def fetch_tranco_domains(n: int = 50_000, cache: Path | None = None) -> list[str]:
    """Top-``n`` domains from the current daily Tranco list.

    Tranco is used rather than a raw Alexa/Majestic list because it is built to
    be reproducible and citable: each daily list has a permanent id, so a
    reviewer can reconstruct exactly the ranking we sampled from.
    """
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text(encoding="utf-8"))[:n]

    blob = _get(TRANCO_LATEST, timeout=180).content
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        member = z.namelist()[0]
        frame = pd.read_csv(io.BytesIO(z.read(member)), names=["rank", "domain"], nrows=n)
    domains = frame["domain"].astype(str).tolist()

    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(domains), encoding="utf-8")
    return domains


def latest_cc_collection() -> str:
    return _get(CC_COLLINFO).json()[0]["id"]


def fetch_commoncrawl_urls(domains: list[str], per_domain: int = 25,
                           collection: str | None = None,
                           max_domains: int = 30) -> list[str]:
    """Real benign URLs *with paths*, from the Common Crawl index.

    One index query per domain, so we query few domains and take several URLs
    from each rather than the reverse. Only successful HTML captures are kept;
    ``robots.txt`` and other non-page captures are filtered out, since they are
    not what a URL classifier would ever be shown.
    """
    collection = collection or latest_cc_collection()
    endpoint = CC_INDEX.format(collection=collection)
    out: list[str] = []

    for domain in domains[:max_domains]:
        try:
            r = requests.get(
                endpoint,
                params={"url": domain, "matchType": "domain",
                        "output": "json", "limit": per_domain * 4},
                headers=_HEADERS, timeout=90,
            )
            if r.status_code != 200:
                continue
            kept = 0
            for line in r.text.splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                url = rec.get("url", "")
                if (rec.get("status") == "200"
                        and "html" in (rec.get("mime") or "")
                        and not url.rstrip("/").endswith(("robots.txt", "sitemap.xml"))):
                    out.append(url)
                    kept += 1
                    if kept >= per_domain:
                        break
        except requests.RequestException:
            continue
        time.sleep(CC_DELAY_SECONDS)

    return out


def templated_benign(domains: list[str], n: int) -> list[str]:
    """``https://www.<domain>`` -- PhiUSIIL's benign surface form, reproduced."""
    return [f"https://www.{d}" for d in domains[:n]]


# --------------------------------------------------------------------------
# Snapshot assembly
# --------------------------------------------------------------------------

def build_snapshot(n_benign_realistic: int = 300,
                   n_benign_templated: int = 300,
                   tranco_pool: int = 50_000,
                   seed: int | None = None,
                   cache_dir: Path = Path("data/cache")) -> Snapshot:
    """Assemble one day's labelled live sample.

    Benign domains are drawn at random from the Tranco pool rather than from the
    head of the ranking: the top few hundred sites are atypically long-lived and
    well-formed, and sampling only those would make the benign class easier than
    the real web in a way that flatters the model.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Seed from the date so a re-run on the same day reproduces that day's draw,
    # while different days draw independently.
    rng = random.Random(seed if seed is not None else int(date.replace("-", "")))

    rows: list[dict] = []
    stats: dict = {"date": date}

    phish = fetch_openphish()
    rows += [{"url": u, "y": 1, "source": "openphish"} for u in phish]
    stats["openphish"] = len(phish)

    try:
        malware = fetch_urlhaus()
        rows += [{"url": u, "y": 1, "source": "urlhaus"} for u in malware]
        stats["urlhaus"] = len(malware)
    except requests.RequestException as exc:
        stats["urlhaus_error"] = str(exc)[:200]

    domains = fetch_tranco_domains(tranco_pool, cache=cache_dir / "tranco.json")
    rng.shuffle(domains)

    realistic = fetch_commoncrawl_urls(domains, per_domain=25, max_domains=40)
    realistic = realistic[:n_benign_realistic]
    rows += [{"url": u, "y": 0, "source": "commoncrawl"} for u in realistic]
    stats["benign_realistic"] = len(realistic)

    templated = templated_benign(domains[n_benign_realistic:], n_benign_templated)
    rows += [{"url": u, "y": 0, "source": "tranco_templated"} for u in templated]
    stats["benign_templated"] = len(templated)

    frame = pd.DataFrame(rows)
    frame["first_seen_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame = frame.drop_duplicates(subset=["url"]).reset_index(drop=True)
    stats["rows"] = len(frame)

    return Snapshot(date=date, frame=frame, stats=stats)


def write_snapshot(snapshot: Snapshot, live_dir: Path = Path("data/live")) -> Path:
    """Persist one day's snapshot. Never overwrites a day that already exists.

    Immutability matters here: these files are the prospective record. A rerun
    that silently replaced yesterday's sample would destroy the guarantee that
    every row was collected before it was ever scored.
    """
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    path = live_dir / f"{snapshot.date}.csv.gz"
    if path.exists():
        return path
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        snapshot.frame.to_csv(fh, index=False)
    (live_dir / f"{snapshot.date}.stats.json").write_text(
        json.dumps(snapshot.stats, indent=2), encoding="utf-8"
    )
    return path


def load_live(live_dir: Path = Path("data/live"),
              sources: tuple[str, ...] | None = None,
              deduplicate: bool = True) -> pd.DataFrame:
    """Concatenate every daily snapshot into one frame.

    ``deduplicate`` keeps the *first* appearance of each URL. OpenPhish rotates
    a fixed-size window, so a URL reported on three consecutive days would
    otherwise be counted three times, over-weighting long-lived campaigns.
    """
    live_dir = Path(live_dir)
    files = sorted(live_dir.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame(columns=["url", "y", "source", "first_seen_utc", "snapshot_date"])

    frames = []
    for f in files:
        frame = pd.read_csv(f)
        frame["snapshot_date"] = f.name.removesuffix(".csv.gz")
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    if sources is not None:
        out = out[out["source"].isin(sources)]
    if deduplicate:
        out = out.sort_values("snapshot_date").drop_duplicates(subset=["url"], keep="first")

    from .features import registrable_domain
    out["domain"] = [registrable_domain(u) for u in out["url"]]
    return out.reset_index(drop=True)
