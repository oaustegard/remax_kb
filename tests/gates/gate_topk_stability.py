#!/usr/bin/env python3
"""GATE — top-k selection is stable at the k-th boundary.

``remax_kb/_hamming.py:top_k`` promises "stable ties (lower index first)". A
selection built as ``argpartition(d, k-1)[:k]`` followed by a stable sort does
NOT deliver that: ``argpartition`` is unstable, so when several distances tie at
the kth value it may keep a higher-indexed element inside the partition and drop
a lower-indexed one with the same distance. Sorting inside the partition cannot
bring the dropped element back. remax fixed exactly this in PR #32
("fix(search): stable tie-break at the kth boundary in top-k") by widening the
candidate set to ``d <= pivot`` before sorting; remax_kb re-introduced the
un-widened form with a comment claiming it was "the same recipe as remax".

The wrong conclusion this gate blocks: *shipping a reader whose result order is
documented as deterministic and index-stable when it is actually an artifact of
numpy's introselect pivot choice* — which means two readers (or two numpy
builds) can return different documents for the same query and same .kb, and
the cross-reader parity gate has no way to tell you why.

ANCHOR: ``np.argsort(distances, kind="stable")[:k]``. Numpy's own stable sort —
an independent implementation of the exact contract the docstring states,
written by neither remax nor remax_kb. Not a golden, not a previous run.

KNOWN-BAD: the un-widened implementation, kept in this file and nowhere else,
run on a distance vector with a deliberate tie straddling the k-th boundary.

    PYTHONDONTWRITEBYTECODE=1 python3 tests/gates/gate_topk_stability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate import Gate  # noqa: E402

from remax_kb._hamming import hamming_scan, top_k  # noqa: E402


def reference(d: np.ndarray, k: int) -> np.ndarray:
    """The anchor: numpy's own stable argsort, truncated."""
    k = min(int(k), d.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    return np.argsort(d, kind="stable")[:k]


def unwidened_top_k(d: np.ndarray, k: int) -> np.ndarray:
    """KNOWN-BAD: the shipped-then-reverted recipe. argpartition, no widening."""
    k = min(int(k), d.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    cut = np.argpartition(d, k - 1)[:k]
    return cut[np.argsort(d[cut], kind="stable")]


# ---- fixtures ---------------------------------------------------------- #

# EVERY case here runs at a realistic corpus size, and that is load-bearing.
# numpy's introselect falls back to an insertion sort below an internal size
# threshold, and insertion sort happens to be stable — so at n<=200 the
# un-widened implementation agrees with np.argsort on EVERY fixture in this
# file, including an all-equal vector. A version of this gate written at toy
# size had two known-bads that silently failed to fire and certified nothing.
# Measured on numpy 2.4.4: n=200 -> stable (known-bad dead), n>=500 -> unstable.
CORPUS_N = 4000


def boundary_tie(n: int = CORPUS_N, k: int = 8) -> np.ndarray:
    """A distance vector whose k-th boundary is a tie, by construction.

    Indices 0..k-2 are uniquely smallest. Every remaining index shares one
    single distance value, so the element that lands in slot k-1 is decided
    purely by tie-break: the contract says index k-1, and nothing else.

    ``n`` must be a realistic corpus size — see CORPUS_N.
    """
    d = np.full(n, 40, dtype=np.int32)
    d[: k - 1] = np.arange(k - 1, dtype=np.int32)
    return d


def hamming_like(seed: int, n: int = CORPUS_N, lo: int = 100, hi: int = 110) -> np.ndarray:
    """Integer distances over a narrow range — what a real Hamming scan yields."""
    return np.random.default_rng(seed).integers(lo, hi, size=n).astype(np.int32)


def real_scan(seed: int, n: int = CORPUS_N, b: int = 96) -> np.ndarray:
    """Distances from an actual packed-code scan, not a synthetic vector."""
    rng = np.random.default_rng(seed)
    codes = np.ascontiguousarray(rng.integers(0, 256, size=(n, b), dtype=np.uint8))
    query = rng.integers(0, 256, size=b, dtype=np.uint8)
    return hamming_scan(codes, query)


K_VALUES = (1, 5, 8, 25, 50, 200)


def main() -> int:
    g = Gate("top-k stability at the k-th boundary (_hamming.top_k vs np.argsort)")

    g.note(f"numpy {np.__version__}")
    try:
        from remax.packing import stable_top_k  # noqa: F401
        import remax_kb._hamming as _h

        g.check(_h.top_k.__module__ == "remax_kb._hamming"
                and "stable_top_k" in _h.top_k.__code__.co_names,
                "top_k delegates selection to remax.packing.stable_top_k",
                "no duplicate selection algorithm is maintained here")
    except ImportError as exc:  # pragma: no cover - remax is a hard dependency
        g.check(False, "remax.packing importable", str(exc))

    # ---- arm 1: the constructed boundary tie ---------------------------- #
    for k in (2, 8, 32):
        d = boundary_tie(k=k)
        got, want = top_k(d, k), reference(d, k)
        g.check(np.array_equal(got, want),
                f"constructed k-th-boundary tie, k={k}",
                f"got={got.tolist()} want={want.tolist()}")

    # ---- arm 2: realistic tie-dense integer distances -------------------- #
    for seed in range(6):
        d = hamming_like(seed)
        for k in K_VALUES:
            got, want = top_k(d, k), reference(d, k)
            g.check(np.array_equal(got, want),
                    f"tie-dense distances (seed={seed}, k={k})",
                    f"first mismatch at "
                    f"{int(np.flatnonzero(got != want)[0]) if got.shape == want.shape and not np.array_equal(got, want) else -1}")

    # ---- arm 3: distances from a real packed scan ----------------------- #
    for seed in range(3):
        d = real_scan(seed)
        for k in (10, 25, 100):
            got, want = top_k(d, k), reference(d, k)
            g.check(np.array_equal(got, want),
                    f"real hamming_scan distances (seed={seed}, k={k})",
                    f"distinct distance values={len(np.unique(d))}, "
                    f"n={d.shape[0]}")

    # ---- arm 4: degenerate shapes --------------------------------------- #
    d = np.array([3, 1, 2, 1, 0], dtype=np.int32)
    g.check(np.array_equal(top_k(d, 0), np.empty(0, dtype=np.intp)),
            "degenerate shape: k=0 returns empty", f"{top_k(d, 0).tolist()}")
    g.check(np.array_equal(top_k(d, 10), reference(d, 10)),
            "degenerate shape: k > n clamps to a full stable order",
            f"got={top_k(d, 10).tolist()} want={reference(d, 10).tolist()}")
    g.check(np.array_equal(top_k(np.full(CORPUS_N, 7, dtype=np.int32), 4),
                           np.arange(4)),
            "degenerate shape: all distances identical -> first k indices",
            f"the maximally-tied case at n={CORPUS_N}; every tie is at the boundary")

    # ---- known-bads ------------------------------------------------------ #
    # (a) the constructed boundary tie, at the configuration the gate runs in.
    bad_boundary = []
    for k in (2, 8, 32):
        d = boundary_tie(k=k)
        if not np.array_equal(unwidened_top_k(d, k), reference(d, k)):
            bad_boundary.append(k)
    g.known_bad(
        "un-widened argpartition is rejected on a constructed k-th-boundary tie",
        rejected=bool(bad_boundary),
        detail=f"k values where it disagrees with np.argsort: {bad_boundary}",
        covers=("constructed k-th-boundary tie",),
    )

    # (b) the same defect on realistic and on real-scan distances — this is the
    #     one that matters, because it is the shape production traffic has.
    n_bad = n_tot = 0
    example = ""
    for seed in range(6):
        d = hamming_like(seed)
        for k in K_VALUES:
            n_tot += 1
            if not np.array_equal(unwidened_top_k(d, k), reference(d, k)):
                n_bad += 1
                if not example:
                    example = (f"seed={seed} k={k}: "
                               f"{unwidened_top_k(d, k)[:5].tolist()} vs "
                               f"{reference(d, k)[:5].tolist()}")
    g.known_bad(
        "un-widened argpartition is rejected on tie-dense distances",
        rejected=n_bad > 0,
        detail=f"{n_bad}/{n_tot} (seed, k) combinations disagree; {example}",
        covers=("tie-dense distances",),
    )

    n_bad = n_tot = 0
    for seed in range(3):
        d = real_scan(seed)
        for k in (10, 25, 100):
            n_tot += 1
            if not np.array_equal(unwidened_top_k(d, k), reference(d, k)):
                n_bad += 1
    g.known_bad(
        "un-widened argpartition is rejected on real hamming_scan output",
        rejected=n_bad > 0,
        detail=f"{n_bad}/{n_tot} (seed, k) combinations disagree",
        covers=("real hamming_scan distances",),
    )

    # (c) a wholly-tied vector: the extreme the docstring's promise rests on.
    d = np.full(CORPUS_N, 7, dtype=np.int32)
    g.known_bad(
        "un-widened argpartition is rejected when every distance ties",
        rejected=not np.array_equal(unwidened_top_k(d, 4), np.arange(4)),
        detail=f"un-widened -> {unwidened_top_k(d, 4).tolist()}, "
               f"contract -> {np.arange(4).tolist()}",
        covers=("degenerate shape: all distances identical",),
    )

    # ---- coverage -------------------------------------------------------- #
    g.coverage(
        "This gate checks ONLY the selection order for a given distance vector. "
        "It says nothing about whether the distances themselves are right — "
        "hamming_scan correctness is tests/test_hamming.py's job, anchored on "
        "the per-byte popcount LUT. Confirmed by mutation: `mutate.py --target "
        "remax_kb/_hamming.py -- <this gate>` kills 16/23 and every survivor is "
        "in the popcount path (the uint64-view guard, the LUT). Under `-- "
        "pytest tests/test_hamming.py` those die and only 2 survive: the LUT's "
        "256 length, and the numpy<2 fallback branch, which never executes on a "
        "numpy>=2 interpreter."
    )
    g.coverage(
        "read_v2's dense arm does NOT go through _hamming.top_k: it has its own "
        "selection inside _dense_search, and v2 fusion re-ranks afterwards. A "
        "tie-instability re-introduced there would not turn this gate red. Only "
        "read.py (v1) and direct _hamming.top_k callers are covered."
    )
    g.coverage(
        "The JS reader (js/kb-reader.js) is not exercised here at all. If it "
        "selects top-k with a different tie rule, the two readers disagree and "
        "this gate stays green — that claim belongs to gate_cross_reader.py."
    )
    g.coverage(
        f"The k<=0 and k>n wrapper paths are NOT exercised by any known-bad "
        f"(the harness lists them as unreached). They are remax_kb-local "
        f"behaviour layered on top of remax.packing.stable_top_k, which raises "
        f"on k<=0 rather than returning empty; a regression there is caught "
        f"only by the anchor, not demonstrated by a rejected bad case."
    )
    g.coverage(
        f"Every fixture runs at n={CORPUS_N} because numpy's introselect is "
        f"stable below an internal size threshold: at n=200 the un-widened "
        f"known-bad agrees with np.argsort on every case in this file and the "
        f"gate would report PASS having rejected nothing. If this file is ever "
        f"shrunk for speed, the known-bads go quietly dead."
    )
    g.coverage(
        "The known-bad is ONE way the selection can go wrong (an unstable "
        "partition). A selection that is stable but wrong in some other way — "
        "an off-by-one in k, a descending sort — is caught by the anchor only "
        "because np.argsort disagrees, and no known-bad here demonstrates that."
    )
    return g.report()


if __name__ == "__main__":
    raise SystemExit(main())
