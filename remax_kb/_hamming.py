"""Vectorized Hamming-distance scan over a packed (N, B) uint8 corpus.

Fast path: XOR the query row against the corpus, then count set bits with
``np.bitwise_count`` (numpy >= 2.0, maps to a hardware POPCNT/VPOPCNT). When the
row width is a multiple of 8 bytes the XOR is reinterpreted as uint64 first, so
``bitwise_count`` runs over 8x fewer elements. This is ~10x faster than the old
256-entry popcount LUT and beats a BLAS float-cosine scan at every corpus size
while keeping the codes bit-packed (see remax_kb#15).

Fallback (numpy < 2.0, no ``bitwise_count``): the original per-byte LUT gather,
so the ``numpy>=1.24`` floor still works — just without the speedup.
"""
from __future__ import annotations

import numpy as np
from remax.packing import stable_top_k

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


def hamming_scan(codes: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return (N,) int32 Hamming distances from each row of ``codes`` to ``query``.

    Args:
        codes: (N, B) uint8, contiguous.
        query: (B,) uint8.
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
