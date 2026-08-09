"""Vectorized Hamming-distance scan over a packed (N, B) uint8 corpus.

Three tiers, fastest first:

1. **Native** — ``remax.packing.hamming_distances``, which dispatches to
   ``remax._native``: a ``__builtin_popcountll`` C kernel compiled at import,
   ctypes-loaded, with its own graceful fallback. It streams the corpus once
   instead of materialising an ``(N, B)`` XOR intermediate.

   Measured here over tier 2, 256 B/row (d=512, k=4), single-threaded:
   **3.2x at n=50k, 4.8x at n=500k**. Note that ``remax._native``'s own
   docstring quotes 25-35x — that is against *remax's* NumPy fallback, which
   is the LUT gather at tier 3. Against tier 2 the gap is smaller, and 3-5x is
   the number to expect from this change.
2. **NumPy popcount** — XOR against the query, then ``np.bitwise_count``
   (numpy >= 2.0) over a uint64 view of the XOR when the row width is a
   multiple of 8, so the ufunc runs over 8x fewer elements (remax_kb#15).
3. **LUT gather** — the original per-byte 256-entry table, so the
   ``numpy>=1.24`` floor still works.

Tier 2 sits between the other two deliberately. remax's *own* NumPy fallback is
the LUT gather, so delegating unconditionally would be a **regression** on any
machine without a working compiler: ``_native.AVAILABLE`` is False there and
tier 3 is what remax would drop to. Testing ``AVAILABLE`` here keeps the
compiled kernel where it exists and the better NumPy kernel where it does not.

This wiring was missing until 2026-08-09. ``remax`` has always been a hard
dependency of this module (``stable_top_k``, below) and the C kernel has
shipped in it since v0.2.0 — the pinned version — but the scan reimplemented
the NumPy path inline and never called it. The related measurement — that
``.sum(axis=1)`` over a narrow inner axis, not the popcount, is what costs in
tier 2 — is in ``oaustegard/experiments`` -> ``lowbit-scan-crossover/``.
"""
from __future__ import annotations

import numpy as np
from remax.packing import hamming_distances as _remax_hamming_distances
from remax.packing import stable_top_k

try:                                   # pragma: no cover - toolchain dependent
    from remax._native import AVAILABLE as _NATIVE_AVAILABLE
except ImportError:                    # older or stripped remax install
    _NATIVE_AVAILABLE = False

POPCOUNT_LUT = np.array(
    [bin(b).count("1") for b in range(256)], dtype=np.uint16
)

# np.bitwise_count landed in numpy 2.0; the package floor is numpy>=1.24.
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")


def _popcount_rows(xor: np.ndarray) -> np.ndarray:
    """Sum set bits per row of a contiguous (N, B) uint8 XOR array -> (N,) int32.

    Uses the hardware-popcount fast path when available, viewing the row as
    uint64 (8x fewer elements) whenever B is a multiple of 8. Summing over the
    whole row makes the uint64 regrouping byte-order-independent, so the result
    is bit-for-bit identical to the per-byte count.
    """
    if _HAS_BITWISE_COUNT:
        if xor.shape[1] % 8 == 0:
            xor = xor.view(np.uint64)
        return np.bitwise_count(xor).sum(axis=1, dtype=np.int32)
    return POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int32)


def hamming_scan(
    codes: np.ndarray,
    query: np.ndarray,
    *,
    out: np.ndarray | None = None,
    threads: int | str | None = None,
) -> np.ndarray:
    """Return (N,) int32 Hamming distances from each row of ``codes`` to ``query``.

    Args:
        codes: (N, B) uint8, contiguous.
        query: (B,) uint8.
        out: optional (N,) int32 destination, so a caller looping over queries
            allocates one buffer instead of one per query. Native tier only —
            the NumPy tiers ignore it and return a fresh array, because their
            reduction allocates regardless.
        threads: forwarded to ``remax.packing.hamming_distances``. ``None``
            keeps remax's process default, which is 1, so the default behaviour
            is unchanged single-threaded. Native tier only.

    The result is identical across all three tiers: summing popcounts over a
    whole row is order-independent, so the uint64 regrouping and the C kernel
    agree with the per-byte LUT bit for bit. ``tests/test_hamming.py`` pins
    that.
    """
    if codes.ndim != 2 or codes.dtype != np.uint8:
        raise ValueError(
            f"codes must be 2-D uint8, got shape={codes.shape} dtype={codes.dtype}"
        )
    if query.ndim != 1 or query.dtype != np.uint8:
        raise ValueError(
            f"query must be 1-D uint8, got shape={query.shape} dtype={query.dtype}"
        )
    if codes.shape[1] != query.shape[0]:
        raise ValueError(
            f"row width mismatch: codes has {codes.shape[1]} bytes per row, "
            f"query has {query.shape[0]}"
        )
    if _NATIVE_AVAILABLE:
        # Streams the corpus once; no (N, B) intermediate at all.
        return _remax_hamming_distances(codes, query, out=out, threads=threads)
    # np.bitwise_xor over a broadcast query yields a C-contiguous (N, B) uint8
    # array, so the uint64 view inside _popcount_rows is always safe.
    xor = np.bitwise_xor(codes, query[None, :])
    return _popcount_rows(xor)


def top_k(distances: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest distances, ascending. Stable ties (lower index first).

    Delegates the selection to :func:`remax.packing.stable_top_k`, which is the
    one implementation of this algorithm that remax_kb should carry. That
    function widens the ``argpartition`` result to ``dists <= pivot`` before the
    stable sort, so a lower-indexed element tied at the kth distance cannot be
    stranded outside the partition (remax PR #32). Hamming distances are
    integer-valued over a narrow range, so ties AT the kth boundary are the
    common case here, not a corner case: a bare ``argpartition(d, k-1)[:k]``
    followed by a stable sort disagrees with
    ``np.argsort(d, kind="stable")[:k]`` on essentially every realistic scan.

    ``k <= 0`` returns an empty index array (remax's ``stable_top_k`` raises
    instead); ``k`` above ``len(distances)`` is clamped.

    Gated by ``tests/gates/gate_topk_stability.py``.
    """
    k = min(int(k), distances.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    return stable_top_k(distances, k)
