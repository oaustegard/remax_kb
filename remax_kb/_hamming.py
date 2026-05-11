"""Vectorized Hamming-distance scan over a packed (N, B) uint8 corpus.

Single fast path: XOR the query row against the corpus, look up popcount
per byte through a 256-entry LUT, sum across bytes. For N up to ~100k
and rows of a few hundred bytes this is sub-millisecond on a laptop —
plenty fast for the proof-of-concept scope (the .kb fits in RAM).
"""
from __future__ import annotations

import numpy as np

POPCOUNT_LUT = np.array(
    [bin(b).count("1") for b in range(256)], dtype=np.uint16
)


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
    xor = np.bitwise_xor(codes, query[None, :])
    # POPCOUNT_LUT[xor] -> (N, B) uint16. Sum on axis=1 fits in int32 for any
    # row width below 2**16 bytes — vastly larger than any realistic .kb row.
    return POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int32)


def top_k(distances: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest distances, ascending. Stable ties (lower index first)."""
    k = min(int(k), distances.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    # np.argpartition for the cut, then sort the slice — same recipe as remax.
    cut = np.argpartition(distances, k - 1)[:k]
    return cut[np.argsort(distances[cut], kind="stable")]
