"""Common Crawl index lookups via the static CDN (data.commoncrawl.org).

Same records as ``collect.fetch_commoncrawl_urls`` (the index-server API), read
from the crawl's sorted CDX shards instead: binary-search the crawl's
``cluster.idx`` with HTTP range requests to find the compressed block where a
domain's SURT prefix starts, fetch that block (and the next, if the domain
runs over the edge), and take the first ``per_domain * 4`` matching lines, the
same window the API query uses. Used when index.commoncrawl.org is
unavailable (testing-new deviation D1).
"""

from __future__ import annotations

import json
import time
import zlib
from functools import lru_cache

import requests

from .collect import _HEADERS

BASE = "https://data.commoncrawl.org/cc-index/collections/{c}/indexes/"
_S = requests.Session()
_S.headers.update(_HEADERS)


def surt_prefix(domain: str) -> str:
    return ",".join(reversed(domain.lower().strip(".").split(".")))


DELAY = 0.5          # seconds between requests: one polite client, never a burst
_last = [0.0]


class Blocked(RuntimeError):
    """The CDN refused us (403/429/503) even after backing off -- stop, don't save partial data."""


def _get(url: str, start: int, end: int) -> bytes:
    wait = 30
    for attempt in range(5):
        pause = _last[0] + DELAY - time.time()
        if pause > 0:
            time.sleep(pause)
        _last[0] = time.time()
        try:
            r = _S.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
        except requests.RequestException:
            time.sleep(wait)
            continue
        if r.status_code in (200, 206):
            return r.content
        if r.status_code in (403, 429, 503):
            time.sleep(wait)
            wait *= 2
            continue
        break
    raise Blocked(f"range fetch failed ({url} {start}-{end})")


@lru_cache(maxsize=64)
def _size(url: str) -> int:
    return int(_S.head(url, timeout=60).headers["Content-Length"])


def _line_at(url: str, pos: int) -> tuple[str, int]:
    """First complete line starting at or after byte `pos` -> (line, its start offset)."""
    chunk = _get(url, pos, pos + 4095)
    i = 0 if pos == 0 else chunk.find(b"\n") + 1
    j = chunk.find(b"\n", i)
    return chunk[i:j].decode("utf-8", "replace"), pos + i


_CLUSTER: dict = {}


def _cluster(collection: str) -> tuple[list[str], list[str]]:
    """The crawl's cluster.idx, downloaded once (one request) and kept in memory: (keys, lines)."""
    if collection in _CLUSTER:
        return _CLUSTER[collection]
    from pathlib import Path
    path = Path("data/cache/cc") / f"{collection}.cluster.idx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        for attempt in range(4):
            r = _S.get(BASE.format(c=collection) + "cluster.idx", stream=True, timeout=120)
            if r.status_code == 200:
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.replace(path)
                break
            time.sleep(60 * (attempt + 1))
        else:
            raise Blocked(f"cluster.idx download refused for {collection}")
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if "	" in ln]
    _CLUSTER.clear()                                      # one crawl resident at a time
    _CLUSTER[collection] = ([ln.split(" ", 1)[0] for ln in lines], lines)
    return _CLUSTER[collection]


def _block_start(collection: str, prefix: str) -> list[str]:
    """cluster.idx entries from the last block whose key < prefix, onward (a few)."""
    import bisect
    keys, lines = _cluster(collection)
    k = max(0, bisect.bisect_left(keys, prefix) - 1)
    return lines[k:k + 3]


def domain_records(collection: str, domain: str, limit: int) -> list[dict]:
    prefix = surt_prefix(domain)
    out = []
    for entry in _block_start(collection, prefix):
        _key, shard, off, length = entry.split("\t")[:4]
        raw = _get(BASE.format(c=collection) + shard, int(off), int(off) + int(length) - 1)
        for line in zlib.decompress(raw, 16 + zlib.MAX_WBITS).decode("utf-8", "replace").splitlines():
            surt = line.split(" ", 1)[0]
            if not (surt.startswith(prefix + ")") or surt.startswith(prefix + ",")):
                if surt > prefix + "~":                     # sorted: past this domain
                    return out
                continue
            try:
                out.append(json.loads(line.split(" ", 2)[2]))
            except (IndexError, json.JSONDecodeError):
                continue
            if len(out) >= limit:
                return out
    return out


def fetch_commoncrawl_urls(domains: list[str], per_domain: int = 25, collection: str | None = None,
                           max_domains: int = 30) -> list[str]:
    """Drop-in for collect.fetch_commoncrawl_urls, over the CDN."""
    out: list[str] = []
    for domain in domains[:max_domains]:
        recs = domain_records(collection, domain, per_domain * 4)   # Blocked propagates: no partial pools
        kept = 0
        for rec in recs:
            url = rec.get("url", "")
            if (rec.get("status") == "200" and "html" in (rec.get("mime") or "")
                    and not url.rstrip("/").endswith(("robots.txt", "sitemap.xml"))):
                out.append(url)
                kept += 1
                if kept >= per_domain:
                    break
    return out


if __name__ == "__main__":
    urls = fetch_commoncrawl_urls(["wikipedia.org", "example.com"], collection="CC-MAIN-2025-05", max_domains=2)
    assert urls and all("wikipedia.org" in u or "example.com" in u for u in urls), urls[:5]
    print(len(urls), urls[:3])
