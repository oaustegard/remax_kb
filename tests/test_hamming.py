"""Unit tests for the Hamming-distance scan kernel (remax_kb/_hamming.py).

These are deliberately remax-free so the popcount fast path is exercised in CI
even when the embedder stack isn't installed. The fast path (uint64 view +
np.bitwise_count, numpy>=2.0) must return results bit-for-bit identical to the
reference per-byte popcount LUT, for any row width — including widths that are
not a multiple of 8 bytes (e.g. dim*k not a multiple of 64). See issue #15.
"""
from __future__ import annotations

import numpy as np
import pytest

from remax_kb._hamming import _popcount_rows, hamming_scan, top_k

# Reference popcount LUT — the pre-optimization implementation, frozen.
_REF_LUT = np.array([bin(b).count("1") for b in range(256)], dtype=np.uint16)


def _ref_scan(codes: np.ndarray, query: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(codes, query[None, :])
    return _REF_LUT[xor].sum(axis=1, dtype=np.int32)


# (dim, k) pairs from realistic remax configs; B = ceil(dim*k / 8) bytes.
# Includes widths divisible by 8 (256, 96, 64) and not (38, 13, 7).
_WIDTHS = [(512, 4), (768, 1), (512, 1), (300, 1), (256, 3), (1024, 2), (100, 1), (56, 1)]


@pytest.mark.parametrize("dim,k", _WIDTHS)
def test_scan_matches_reference_lut(dim: int, k: int) -> None:
    bits = dim * k
    b = (bits + 7) // 8
    rng = np.random.default_rng(bits)
    codes = np.ascontiguousarray(rng.integers(0, 256, size=(2000, b), dtype=np.uint8))
    query = rng.integers(0, 256, size=b, dtype=np.uint8)

    got = hamming_scan(codes, query)
    ref = _ref_scan(codes, query)

    assert got.dtype == np.int32
    assert np.array_equal(got, ref)
    # distances are bounded by the bit width
    assert got.min() >= 0 and got.max() <= bits


@pytest.mark.parametrize("dim,k", _WIDTHS)
def test_topk_matches_reference(dim: int, k: int) -> None:
    bits = dim * k
    b = (bits + 7) // 8
    rng = np.random.default_rng(bits + 1)
    codes = np.ascontiguousarray(rng.integers(0, 256, size=(5000, b), dtype=np.uint8))
    query = rng.integers(0, 256, size=b, dtype=np.uint8)

    ref = _ref_scan(codes, query)
    got = hamming_scan(codes, query)
    # top_k over the optimized distances == top_k over the reference distances
    assert np.array_equal(top_k(got, 25), top_k(ref, 25))


def test_identical_row_is_distance_zero() -> None:
    rng = np.random.default_rng(0)
    codes = np.ascontiguousarray(rng.integers(0, 256, size=(100, 32), dtype=np.uint8))
    # querying with an exact corpus row must yield distance 0 at that index
    for i in (0, 37, 99):
        dists = hamming_scan(codes, codes[i])
        assert dists[i] == 0
        assert top_k(dists, 1)[0] == i


def test_popcount_rows_direct() -> None:
    # full-ones row XORs to all bits set; popcount == bit width
    xor = np.full((4, 16), 0xFF, dtype=np.uint8)
    assert np.array_equal(_popcount_rows(xor), np.full(4, 16 * 8, dtype=np.int32))
    # zero row -> distance 0
    assert np.array_equal(_popcount_rows(np.zeros((3, 9), dtype=np.uint8)), np.zeros(3, np.int32))


def test_validation_guards() -> None:
    good = np.zeros((4, 8), dtype=np.uint8)
    with pytest.raises(ValueError):
        hamming_scan(good.astype(np.uint16), np.zeros(8, dtype=np.uint8))
    with pytest.raises(ValueError):
        hamming_scan(good, np.zeros(8, dtype=np.uint16))
    with pytest.raises(ValueError):
        hamming_scan(good, np.zeros(7, dtype=np.uint8))  # width mismatch


def test_top_k_edge_cases() -> None:
    dists = np.array([3, 1, 2, 1, 0], dtype=np.int32)
    assert np.array_equal(top_k(dists, 0), np.empty(0, dtype=np.intp))
    # k larger than N clamps; ties broken by lower index first (stable)
    assert np.array_equal(top_k(dists, 10), np.array([4, 1, 3, 2, 0]))
