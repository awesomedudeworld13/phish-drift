"""URL-lexical feature extraction.

This module is the single most important contract in the project. The whole
study rests on one claim: the benchmark URLs and the live URLs are turned into
feature vectors by *the same code*, so any difference in model performance is a
property of the data, not of the pipeline.

Three invariants are enforced by ``tests/test_features.py`` and must not be
relaxed:

1. **No network access.** Every feature is computed from the URL string alone.
   Live phishing URLs die within hours of being reported, so any feature that
   requires fetching the page is uncomputable at operational time. This is why
   we do not use PhiUSIIL's own 54-column feature set (see ``benchmark.py``).

2. **No vocabulary fitted on data.** Keyword lists, brand lists and TLD buckets
   are hard-coded constants below. A feature encoding fitted on the training
   set (target encoding, one-hot over observed TLDs, tf-idf) would silently
   change dimensionality or meaning between benchmark and live data, which
   would confound the very gap we are trying to measure.

3. **Fixed dimensionality and column order.** ``FEATURE_NAMES`` is the schema.
   ``extract()`` returns exactly these keys in this order, for any input.

Feature families: length/structure, character composition, domain shape,
scheme/port, and a small set of documented lexical red flags.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlsplit, parse_qsl

import tldextract

# tldextract needs the Public Suffix List to split "suhanir7.github.io" into
# subdomain="suhanir7", domain="github", suffix="io" correctly. We pin to the
# bundled snapshot rather than letting it fetch a fresh PSL at runtime: an
# evolving suffix list would make feature extraction non-reproducible, and a
# CI runner with no network would silently fall back to different behaviour.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

_VOWELS = set("aeiou")
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_HEX_ESCAPE = re.compile(r"%[0-9a-fA-F]{2}")
_DIGIT_RUN = re.compile(r"\d+")

# Lexical red flags. Chosen a priori from the phishing literature and frozen
# before any live data was collected (see PREREGISTRATION.md). They are NOT
# tuned against the test split or the live feed.
_SUSPICIOUS_KEYWORDS = frozenset({
    "account", "admin", "alert", "auth", "bank", "billing", "confirm",
    "credential", "customer", "ebayisapi", "expire", "invoice", "limited",
    "live", "lock", "login", "logon", "mail", "manage", "password", "pay",
    "recover", "register", "secure", "security", "signin", "submit",
    "support", "suspend", "unlock", "update", "user", "verify", "wallet",
    "webscr", "websrc",
})

# Brand tokens are checked separately from generic keywords because a brand
# name appearing OUTSIDE its own registrable domain is the classic phishing
# signature ("paypal.attacker.com"), whereas a generic keyword is only weakly
# suspicious anywhere.
_BRAND_TOKENS = frozenset({
    "adobe", "amazon", "americanexpress", "apple", "att", "bankofamerica",
    "binance", "bradesco", "chase", "citi", "coinbase", "correos", "dhl",
    "dropbox", "ebay", "facebook", "fedex", "google", "hsbc", "icloud",
    "instagram", "itau", "linkedin", "metamask", "microsoft", "netflix",
    "office365", "outlook", "paypal", "roblox", "santander", "steam",
    "telegram", "tiktok", "usps", "wellsfargo", "whatsapp", "yahoo",
})

# Frozen TLD bucket. Encoding the TLD as a fixed membership test keeps the
# feature schema identical across benchmark and live even when the live feed
# starts serving TLDs the benchmark never contained -- which it does, and which
# is itself part of the distribution shift we are measuring.
_COMMON_TLDS = frozenset({
    "com", "org", "net", "edu", "gov", "uk", "de", "jp", "fr", "au", "ru",
    "br", "it", "cn", "es", "nl", "ca", "in", "co", "io",
})

# TLDs repeatedly flagged as abuse-heavy in registry transparency reports.
# Frozen a priori; see PREREGISTRATION.md.
_ABUSED_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "buzz", "click", "link",
    "work", "live", "icu", "cyou", "rest", "fit", "gdn", "loan", "men",
    "date", "racing", "stream", "download", "review", "country", "kim",
})

# URL shorteners collapse the entire lexical signal into an opaque token, so a
# model must be able to tell that it is looking at one. Frozen a priori.
_SHORTENERS = frozenset({
    "bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc",
    "rebrand.ly", "s.id", "t.ly", "short.io", "lnkd.in",
})

_SCRIPT_EXTS = frozenset({"php", "asp", "aspx", "jsp", "cgi", "pl", "py"})
_ARCHIVE_EXTS = frozenset({"zip", "rar", "7z", "gz", "tar", "exe", "msi", "apk", "scr"})


FEATURE_NAMES: tuple[str, ...] = (
    # -- length and structure -------------------------------------------
    "url_length",
    "domain_length",
    "subdomain_length",
    "path_length",
    "query_length",
    "fragment_length",
    "path_depth",
    "n_query_params",
    # -- character composition over the whole URL ------------------------
    "digit_ratio",
    "letter_ratio",
    "special_ratio",
    "upper_ratio",
    "entropy_url",
    "n_dots",
    "n_hyphens",
    "n_underscores",
    "n_slashes",
    "n_equals",
    "n_ampersands",
    "n_at",
    "n_tilde",
    "n_plus",
    "n_commas",
    "n_hex_escapes",
    # -- registrable-domain shape ----------------------------------------
    "entropy_domain",
    "domain_digit_ratio",
    "domain_hyphen_count",
    "domain_vowel_ratio",
    "domain_max_consonant_run",
    "domain_max_repeat_run",
    "domain_digit_group_count",
    "domain_token_count",
    "domain_longest_token",
    "n_subdomains",
    "has_www",
    "is_ip_domain",
    "has_punycode",
    "tld_length",
    "tld_is_common",
    "tld_is_abused",
    "is_shortener",
    # -- scheme and authority --------------------------------------------
    "is_https",
    "has_port",
    "has_nonstandard_port",
    "has_userinfo",
    # -- documented lexical red flags -------------------------------------
    "has_at_symbol",
    "double_slash_in_path",
    "suspicious_keyword_count",
    "brand_token_count",
    "brand_outside_domain",
    "has_script_ext",
    "has_archive_ext",
)


def _entropy(s: str) -> float:
    """Shannon entropy of the character distribution, in bits."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _max_run(s: str, predicate) -> int:
    """Length of the longest run of characters satisfying ``predicate``."""
    best = run = 0
    for ch in s:
        run = run + 1 if predicate(ch) else 0
        if run > best:
            best = run
    return best


def _max_repeat_run(s: str) -> int:
    """Length of the longest run of one repeated character ("aaaa" -> 4)."""
    best = run = 0
    prev = None
    for ch in s:
        run = run + 1 if ch == prev else 1
        prev = ch
        if run > best:
            best = run
    return best


def _safe_split(url: str):
    """urlsplit that never raises.

    Live feeds carry genuinely malformed URLs -- bracketed hosts, stray
    control characters, bad percent-encoding. Dropping those rows would be a
    silent, non-random filter applied to the live side only, which is exactly
    the kind of hidden cleanup this project exists to criticise. So we parse
    defensively and keep the row.
    """
    try:
        return urlsplit(url)
    except ValueError:
        return urlsplit("")


def extract(url: str) -> dict[str, float]:
    """Return the fixed-schema lexical feature vector for one URL.

    Accepts a raw URL with or without a scheme. Never raises, never touches the
    network. Keys are exactly ``FEATURE_NAMES``, in order.
    """
    url = (url or "").strip()
    # A bare "example.com/path" has no scheme; urlsplit would read the whole
    # thing as a path and the host features would all be empty. Normalising to
    # http:// is the standard fix, and we record the original scheme presence
    # via is_https below rather than losing it.
    had_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url))
    parsed = _safe_split(url if had_scheme else "http://" + url)

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    ext = _EXTRACT(host)
    registrable = ext.domain or ""
    suffix = ext.suffix or ""
    subdomain = ext.subdomain or ""
    reg_domain = ".".join(p for p in (registrable, suffix) if p)

    lowered = url.lower()
    path_segments = [seg for seg in path.split("/") if seg]

    n_chars = len(url) or 1
    n_digits = sum(ch.isdigit() for ch in url)
    n_letters = sum(ch.isalpha() for ch in url)
    n_upper = sum(ch.isupper() for ch in url)

    dom_chars = len(registrable) or 1
    dom_letters = [ch for ch in registrable if ch.isalpha()]

    # Tokenise on the separators an attacker actually uses to build lookalike
    # strings, so "secure-paypal.login" yields three comparable tokens.
    tokens = [t for t in re.split(r"[^a-z0-9]+", lowered) if t]
    domain_tokens = [t for t in re.split(r"[^a-z0-9]+", reg_domain) if t]

    # A brand token is only a red flag when it appears somewhere OTHER than the
    # registrable domain it legitimately belongs to.
    brand_hits = sum(1 for t in tokens if t in _BRAND_TOKENS)
    brand_in_reg = any(t in _BRAND_TOKENS for t in domain_tokens)
    outside_tokens = [t for t in re.split(r"[^a-z0-9]+", (subdomain + " " + path + " " + query).lower()) if t]
    brand_outside = sum(1 for t in outside_tokens if t in _BRAND_TOKENS)

    try:
        port = parsed.port
    except ValueError:
        port = None

    return {
        "url_length": float(len(url)),
        "domain_length": float(len(host)),
        "subdomain_length": float(len(subdomain)),
        "path_length": float(len(path)),
        "query_length": float(len(query)),
        "fragment_length": float(len(fragment)),
        "path_depth": float(len(path_segments)),
        "n_query_params": float(len(parse_qsl(query, keep_blank_values=True))),

        "digit_ratio": n_digits / n_chars,
        "letter_ratio": n_letters / n_chars,
        "special_ratio": (n_chars - n_digits - n_letters) / n_chars,
        "upper_ratio": n_upper / n_chars,
        "entropy_url": _entropy(url),
        "n_dots": float(url.count(".")),
        "n_hyphens": float(url.count("-")),
        "n_underscores": float(url.count("_")),
        "n_slashes": float(url.count("/")),
        "n_equals": float(url.count("=")),
        "n_ampersands": float(url.count("&")),
        "n_at": float(url.count("@")),
        "n_tilde": float(url.count("~")),
        "n_plus": float(url.count("+")),
        "n_commas": float(url.count(",")),
        "n_hex_escapes": float(len(_HEX_ESCAPE.findall(url))),

        "entropy_domain": _entropy(registrable),
        "domain_digit_ratio": sum(ch.isdigit() for ch in registrable) / dom_chars,
        "domain_hyphen_count": float(registrable.count("-")),
        "domain_vowel_ratio": (
            sum(ch in _VOWELS for ch in dom_letters) / len(dom_letters) if dom_letters else 0.0
        ),
        "domain_max_consonant_run": float(
            _max_run(registrable, lambda c: c.isalpha() and c not in _VOWELS)
        ),
        "domain_max_repeat_run": float(_max_repeat_run(registrable)),
        "domain_digit_group_count": float(len(_DIGIT_RUN.findall(registrable))),
        "domain_token_count": float(len(domain_tokens)),
        "domain_longest_token": float(max((len(t) for t in domain_tokens), default=0)),
        "n_subdomains": float(len([p for p in subdomain.split(".") if p])),
        "has_www": float(subdomain == "www" or subdomain.startswith("www.")),
        "is_ip_domain": float(bool(_IPV4.match(host))),
        "has_punycode": float("xn--" in host),
        "tld_length": float(len(suffix)),
        "tld_is_common": float(suffix in _COMMON_TLDS),
        "tld_is_abused": float(suffix.split(".")[-1] in _ABUSED_TLDS if suffix else False),
        "is_shortener": float(reg_domain in _SHORTENERS),

        "is_https": float(parsed.scheme == "https"),
        "has_port": float(port is not None),
        "has_nonstandard_port": float(port is not None and port not in (80, 443)),
        "has_userinfo": float(bool(parsed.username)),

        "has_at_symbol": float("@" in url),
        "double_slash_in_path": float("//" in path),
        "suspicious_keyword_count": float(sum(1 for t in tokens if t in _SUSPICIOUS_KEYWORDS)),
        "brand_token_count": float(brand_hits),
        "brand_outside_domain": float(brand_outside if not brand_in_reg else 0),
        "has_script_ext": float(
            bool(path_segments) and path_segments[-1].rsplit(".", 1)[-1].lower() in _SCRIPT_EXTS
        ),
        "has_archive_ext": float(
            bool(path_segments) and path_segments[-1].rsplit(".", 1)[-1].lower() in _ARCHIVE_EXTS
        ),
    }


def registrable_domain(url: str) -> str:
    """eTLD+1 for a URL, used to build domain-disjoint splits.

    Correctness matters here more than anywhere else in the project: a naive
    ``host.split(".")[-2:]`` maps "suhanir7.github.io" to "github.io" and
    "bbc.co.uk" to "co.uk", which would put thousands of unrelated URLs into
    one pseudo-domain and quietly break the disjointness guarantee that cell 2
    of the study depends on. The Public Suffix List is the only correct source.

    Note we use the PSL's *public* section only (tldextract's default), so
    "a.github.io" and "b.github.io" both collapse to "github.io". That is
    deliberate. Enabling private suffixes would treat them as two independent
    domains and let the model learn "github.io is phishy" from the training
    split and cash that in on the test split -- precisely the leakage cell 2
    exists to eliminate. Collapsing over-groups shared hosting platforms, which
    can only make the disjoint split stricter, never looser.
    """
    url = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        url = "http://" + url
    host = (_safe_split(url).hostname or "").lower()
    if _IPV4.match(host):
        return host
    ext = _EXTRACT(host)
    parts = [p for p in (ext.domain, ext.suffix) if p]
    return ".".join(parts) if parts else host


# Path-derived features, named here so evaluate.py can run the sampling-confound
# diagnostic: if a model trained on ONLY these can separate live phishing from
# live benign, our benign sampling differs structurally from our phishing
# sampling and the headline result is an artifact. See RESULTS.md section 4.
PATH_SHAPE_FEATURES: tuple[str, ...] = (
    "path_length",
    "query_length",
    "fragment_length",
    "path_depth",
    "n_query_params",
    "n_slashes",
    "n_equals",
    "n_ampersands",
)


def feature_matrix(urls):
    """Vectorise an iterable of URLs into a (n, len(FEATURE_NAMES)) float array."""
    import numpy as np

    rows = [extract(u) for u in urls]
    return np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64
    )
