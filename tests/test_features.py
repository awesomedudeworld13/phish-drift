"""Checks on the invariants the study depends on.

Run with `python -m pytest tests/ -q`, or directly: `python tests/test_features.py`.

These are not exhaustive unit tests. Each one guards a specific way the
experiment could become invalid without any visible error:

* a feature that changes dimensionality or meaning between benchmark and live
  data would confound the gap with a pipeline difference;
* a feature that touches the network would be uncomputable on dead live URLs;
* a broken eTLD+1 would silently destroy the domain-disjoint guarantee that
  cell 2 rests on;
* a threshold selected on the wrong partition would reintroduce the optimism
  the whole project exists to measure.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishdrift import evaluate                                    # noqa: E402
from phishdrift.benchmark import (                                 # noqa: E402
    BENCHMARKS, BENIGN_TEMPLATE, Split, domain_disjoint_split, random_split,
    template_audit, verify_vendored,
)
from phishdrift.features import (                                  # noqa: E402
    FEATURE_NAMES, extract, feature_matrix, registrable_domain,
)


def test_schema_is_fixed_and_ordered():
    """Every URL yields exactly FEATURE_NAMES, in order, whatever the input."""
    inputs = [
        "https://www.example.com",
        "http://192.168.0.1:8080/a/b/c.php?x=1&y=2#frag",
        "bit.ly/abc",
        "",
        "not a url at all",
        "http://xn--80ak6aa92e.com/pay",
        "https://user:pw@host.example.co.uk:8443//double//slash/",
    ]
    for url in inputs:
        row = extract(url)
        assert tuple(row.keys()) == FEATURE_NAMES, f"schema drift on {url!r}"
        assert all(isinstance(v, float) for v in row.values()), f"non-float on {url!r}"
        assert all(np.isfinite(v) for v in row.values()), f"non-finite on {url!r}"
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "duplicate feature name"
    assert feature_matrix(inputs).shape == (len(inputs), len(FEATURE_NAMES))


def test_extraction_is_deterministic_and_offline():
    """No network, no hidden state: same input, same output, always."""
    import socket

    url = "https://secure-paypal.login.example.tk/verify?id=99"
    first = extract(url)

    # Any socket use inside extraction is a hard failure -- live phishing URLs
    # are dead by the time we score them, so a network-dependent feature would
    # silently return different values for benchmark and live rows.
    real_socket = socket.socket

    def forbidden(*a, **k):
        raise AssertionError("feature extraction attempted a network connection")

    socket.socket = forbidden
    try:
        second = extract(url)
    finally:
        socket.socket = real_socket

    assert first == second


def test_registrable_domain_uses_public_suffix_list():
    """Naive splitting breaks exactly the cases that matter for disjointness."""
    cases = {
        "https://www.bbc.co.uk/news": "bbc.co.uk",       # multi-part suffix
        "http://a.b.c.example.com/x": "example.com",     # deep subdomains
        "http://suhanir7.github.io/Clone": "github.io",  # shared hosting collapses
        "http://other.github.io/thing": "github.io",     # ... to the same unit
        "http://192.168.1.1/login": "192.168.1.1",       # bare IP
    }
    for url, expected in cases.items():
        assert registrable_domain(url) == expected, url


def test_domain_disjoint_split_shares_no_domain():
    """The guarantee cell 2 depends on, checked on synthetic data with heavy reuse."""
    rows = []
    for d in range(200):
        for p in range(np.random.randint(1, 8)):
            rows.append({
                "url": f"https://host{d}.example{d}.com/page{p}",
                "y": d % 2,
            })
    frame = pd.DataFrame(rows)
    frame["domain"] = [registrable_domain(u) for u in frame.url]

    split = domain_disjoint_split(frame, seed=1)
    assert not (set(split.train.domain) & set(split.test.domain))
    assert not (set(split.train.domain) & set(split.val.domain))
    assert not (set(split.val.domain) & set(split.test.domain))
    assert len(split.train) + len(split.val) + len(split.test) == len(frame)


def test_random_split_does_share_domains():
    """The contrast is only meaningful if the random split really does leak."""
    rows = [{"url": f"https://host.example{d % 20}.com/p{p}", "y": d % 2}
            for d in range(200) for p in range(5)]
    frame = pd.DataFrame(rows)
    frame["domain"] = [registrable_domain(u) for u in frame.url]

    split = random_split(frame, seed=1)
    shared = set(split.train.domain) & set(split.test.domain)
    assert shared, "random split shared no domains; the cell 1 vs 2 contrast is vacuous"


def test_template_rule_matches_hand_computation():
    """The headline audit number must be reproducible by hand."""
    frame = pd.DataFrame({
        "url": [
            "https://www.good.com",        # benign, matches template
            "https://www.good2.com/",      # benign, trailing slash still matches
            "http://evil.com/login.php",   # phishing, no match
            "https://www.evil2.com",       # phishing that DOES match the template
        ],
        "y": [0, 0, 1, 1],
    })
    audit = template_audit(frame)
    assert audit.tp == 1 and audit.fp == 0 and audit.fn == 1 and audit.tn == 2
    assert audit.precision == 1.0
    assert abs(audit.tss - 0.5) < 1e-9
    assert audit.benign_template_share == 1.0
    assert audit.phish_template_share == 0.5


def test_benign_template_regex_boundaries():
    assert BENIGN_TEMPLATE.match("https://www.a.com")
    assert BENIGN_TEMPLATE.match("https://www.a.com/")
    assert not BENIGN_TEMPLATE.match("http://www.a.com")      # scheme
    assert not BENIGN_TEMPLATE.match("https://a.com")         # no www
    assert not BENIGN_TEMPLATE.match("https://www.a.com/x")   # path
    assert not BENIGN_TEMPLATE.match("https://www.a.com?q=1") # query


def test_vendored_corpora_match_their_published_hash():
    """A committed corpus must still be the bytes its publisher advertises.

    Vendoring buys reproducibility but introduces a silent failure mode: a
    corrupted file, a bad merge, or a copy regenerated from a different
    upstream version would change every published number with nothing visibly
    wrong. This is the check that makes the committed copy trustworthy.
    """
    checked_any = False
    for key in BENCHMARKS:
        report = verify_vendored(key)
        if not report["checked"]:
            continue
        checked_any = True
        assert report["match"], (
            f"{key}: vendored copy hash {report['actual'][:16]}... does not match "
            f"published {report['expected'][:16]}..."
        )
    assert checked_any, "no vendored corpus was verified; expected at least one"


def test_every_benchmark_normalises_labels_to_phishing_positive():
    """Corpora disagree on label polarity; the loaders must not.

    PhiUSIIL uses label==1 for legitimate, Hannousse a status string. Getting
    this backwards for one corpus would invert its every metric while still
    producing entirely plausible-looking numbers.
    """
    for key, source in BENCHMARKS.items():
        assert source.key == key, f"registry key {key!r} != source.key {source.key!r}"
        assert source._loader in vars(sys.modules["phishdrift.benchmark"]), \
            f"{key}: loader {source._loader!r} not defined"
        # A vendored corpus can be loaded offline, so assert on the real thing.
        if source.vendored:
            frame = source.load(Path("data"))
            assert set(frame.columns) >= {"url", "y", "domain"}
            assert frame.y.isin([0, 1]).all()
            # Phishing URLs are longer than legitimate ones in every published
            # phishing corpus; a polarity flip would reverse this.
            phish_len = frame[frame.y == 1].url.str.len().mean()
            legit_len = frame[frame.y == 0].url.str.len().mean()
            assert phish_len > legit_len, (
                f"{key}: phishing URLs ({phish_len:.0f} chars) are not longer than "
                f"legitimate ({legit_len:.0f}); labels may be inverted"
            )


def test_tss_is_zero_for_constant_forecasts():
    """The property that makes TSS the honest headline metric at low base rates."""
    y = np.array([1] * 5 + [0] * 995)
    for constant in (0.0, 0.5, 1.0):
        prob = np.full(len(y), constant)
        for thr in (0.0, 0.25, 0.75, 1.0):
            s = evaluate.score_at(y, prob, thr)
            assert abs(s.tss) < 1e-12, f"constant {constant} at thr {thr} -> TSS {s.tss}"
            # ... while accuracy looks excellent for the same useless forecast.
        assert evaluate.score_at(y, prob, 1.01).accuracy >= 0.99


def test_threshold_selection_optimises_the_named_objective():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 4000)
    prob = np.clip(y * 0.45 + rng.normal(0.3, 0.2, 4000), 0, 1)

    thr_f1 = evaluate.select_threshold(y, prob, "f1")
    thr_tss = evaluate.select_threshold(y, prob, "tss")

    assert evaluate.score_at(y, prob, thr_f1).f1 >= evaluate.score_at(y, prob, thr_tss).f1
    assert evaluate.score_at(y, prob, thr_tss).tss >= evaluate.score_at(y, prob, thr_f1).tss


def test_cluster_bootstrap_is_wider_than_iid():
    """Clustering must not be cosmetic: correlated rows have to widen the interval."""
    rng = np.random.default_rng(3)
    # 40 domains, 50 near-identical rows each -- effective n is 40, not 2000.
    # A quarter of the domains are scored on the wrong side of the threshold, so
    # the metric is not saturated and there is real variance for the two
    # resampling schemes to disagree about. Whether such a domain is drawn is
    # one coin flip under clustering and fifty under i.i.d. resampling, which is
    # exactly the dependence that makes i.i.d. intervals too narrow.
    y, prob, groups = [], [], []
    for d in range(40):
        label = d % 2
        correct = (d % 4) != 3
        centre = (0.65 if label else 0.35) if correct else (0.35 if label else 0.65)
        for _ in range(50):
            y.append(label)
            prob.append(np.clip(centre + rng.normal(0, 0.02), 0, 1))
            groups.append(f"domain{d}.com")
    y, prob, groups = np.array(y), np.array(prob), np.array(groups)

    lo_c, hi_c = evaluate.cluster_bootstrap_ci(y, prob, 0.5, groups=groups, n_resamples=300)
    lo_i, hi_i = evaluate.cluster_bootstrap_ci(y, prob, 0.5, groups=None, n_resamples=300)
    assert (hi_c - lo_c) > (hi_i - lo_i), "clustered CI was not wider than i.i.d."


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"\n{'all checks passed' if not failures else f'{failures} FAILED'}")
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_all())
