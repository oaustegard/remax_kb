"""Unit tests for the Hamming-distance scan kernel (remax_kb/_hamming.py).

Every kernel tier must return results bit-for-bit identical to the reference
per-byte popcount LUT, for any row width — including widths that are not a
multiple of 8 bytes (e.g. dim*k not a multiple of 64). See issue #15.

These need `remax` (the module has always imported `stable_top_k` from it) but
not the embedder stack, so they run in CI without models. The tier tests at the
bottom force each kernel explicitly rather than testing whichever one this
machine offers.
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


# --------------------------------------------------------------------------- #
# Kernel tiers (remax_kb#15 follow-up, 2026-08-09)
#
# hamming_scan picks one of three kernels: remax's compiled _native scan, the
# uint64 np.bitwise_count path, or the per-byte LUT. The tests above only
# exercise whichever tier this machine happens to offer. These force each one,
# because the whole point of the ladder is that the choice is invisible in the
# result and visible only in the clock.
# --------------------------------------------------------------------------- #

from remax_kb import _hamming as _h  # noqa: E402


def _codes(n=1500, b=32, seed=7):
    rng = np.random.default_rng(seed)
    return (np.ascontiguousarray(rng.integers(0, 256, size=(n, b), dtype=np.uint8)),
            rng.integers(0, 256, size=b, dtype=np.uint8))


def test_native_and_numpy_tiers_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    codes, query = _codes()
    ref = _ref_scan(codes, query)

    monkeypatch.setattr(_h, "_NATIVE_AVAILABLE", False)
    numpy_tier = hamming_scan(codes, query)
    monkeypatch.undo()

    np.testing.assert_array_equal(numpy_tier, ref)
    np.testing.assert_array_equal(hamming_scan(codes, query), ref)
    assert numpy_tier.dtype == np.int32


def test_lut_tier_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy < 2.0 floor: no bitwise_count, no native."""
    codes, query = _codes()
    monkeypatch.setattr(_h, "_NATIVE_AVAILABLE", False)
    monkeypatch.setattr(_h, "_HAS_BITWISE_COUNT", False)
    np.testing.assert_array_equal(hamming_scan(codes, query), _ref_scan(codes, query))


def test_native_tier_is_actually_reached() -> None:
    """The regression this fixes: remax was a hard dependency and the compiled
    kernel shipped in it, but the scan never called it."""
    if not _h._NATIVE_AVAILABLE:
        pytest.skip("no compiler in this environment; native tier unavailable")
    codes, query = _codes()
    seen: list[int] = []
    real = _h._remax_hamming_distances

    def spy(c, q, **kw):
        seen.append(len(c))
        return real(c, q, **kw)

    orig, _h._remax_hamming_distances = _h._remax_hamming_distances, spy
    try:
        got = hamming_scan(codes, query)
    finally:
        _h._remax_hamming_distances = orig
    assert seen == [len(codes)], "hamming_scan did not delegate to remax"
    np.testing.assert_array_equal(got, _ref_scan(codes, query))


def test_out_buffer_is_reused_and_fully_overwritten() -> None:
    if not _h._NATIVE_AVAILABLE:
        pytest.skip("out= is a native-tier passthrough")
    codes, q1 = _codes(seed=1)
    _, q2 = _codes(seed=2)
    buf = np.empty(len(codes), dtype=np.int32)

    r1 = hamming_scan(codes, q1, out=buf)
    assert r1 is buf
    r2 = hamming_scan(codes, q2, out=buf)
    # No stale values from the previous query survive.
    np.testing.assert_array_equal(r2, _ref_scan(codes, q2))


def test_threads_do_not_change_the_result() -> None:
    if not _h._NATIVE_AVAILABLE:
        pytest.skip("threads= is a native-tier passthrough")
    codes, query = _codes(n=40_000)
    ref = _ref_scan(codes, query)
    for threads in (1, 2, "auto"):
        np.testing.assert_array_equal(hamming_scan(codes, query, threads=threads), ref)
