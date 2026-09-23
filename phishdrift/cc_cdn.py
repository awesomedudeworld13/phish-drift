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
import zlib
from functools import lru_cache

import requests

from .collect import _HEADERS

BASE = "https://data.commoncrawl.org/cc-index/collections/{c}/indexes/"
_S = requests.Session()
_S.headers.update(_HEADERS)


def surt_prefix(domain: str) -> str:
    return ",".join(reversed(domain.lower().strip(".").split(".")))


def _get(url: str, start: int, end: int) -> bytes:
    for attempt in range(4):
        try:
            r = _S.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
            if r.status_code in (200, 206):
                return r.content
        except requests.RequestException:
            pass
    raise RuntimeError(f"range fetch failed: {url} {start}-{end}")


@lru_cache(maxsize=64)
def _size(url: str) -> int:
    return int(_S.head(url, timeout=60).headers["Content-Length"])


def _line_at(url: str, pos: int) -> tuple[str, int]:
    """First complete line starting at or after byte `pos` -> (line, its start offset)."""
    chunk = _get(url, pos, pos + 4095)
    i = 0 if pos == 0 else chunk.find(b"\n") + 1
    j = chunk.find(b"\n", i)
    return chunk[i:j].decode("utf-8", "replace"), pos + i


def _block_start(collection: str, prefix: str) -> list[str]:
    """cluster.idx lines from the last block whose key < prefix, onward (a few)."""
    url = BASE.format(c=collection) + "cluster.idx"
    lo, hi = 0, _size(url)
    while hi - lo > 8192:                                  # binary search on byte offset
        mid = (lo + hi) // 2
        line, _ = _line_at(url, mid)
        if line.split(" ", 1)[0] < prefix:
            lo = mid
        else:
            hi = mid
    chunk = _get(url, max(0, lo - 4096), hi + 65536).decode("utf-8", "replace").split("\n")[1:-1]
    keys = [ln for ln in chunk if "\t" in ln]
    k = max((i for i, ln in enumerate(keys) if ln.split(" ", 1)[0] < prefix), default=0)
    return keys[k:k + 3]


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
        try:
            recs = domain_records(collection, domain, per_domain * 4)
        except RuntimeError:
            continue
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
