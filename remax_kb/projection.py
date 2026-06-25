"""Portable seed-only projection for the stacked-SimHash binarizer.

The Haar projection (``remax.haar_rotation``) is built with numpy's
``PCG64 + Ziggurat + LAPACK-QR`` pipeline, which is **not** reproducible in a
non-numpy reader. So a Haar ``.kbi`` must *ship* its rotation matrices — two
independent QR implementations produce different orthogonal matrices, and
hashing the corpus with matrix A while hashing the query with matrix B collapses
recall to chance (~50% of code bits flip). int8 shrank that sidecar 4×; this
removes it entirely.

A **Rademacher** projection uses ±1 hyperplane entries drawn from ``splitmix64``,
a tiny integer PRNG that any language reproduces bit-for-bit (no floats, no
LAPACK). Producer and consumer regenerate the *identical* matrix from
``(dim, k, seed)`` — nothing is shipped, and the cross-language mismatch
catastrophe becomes structurally impossible.

Measured cost (experiments/kb-k-sweep, Gemini corpus, dim=768): ~2 recall@10
points vs Haar at matched ``k``, bought back with one extra stack
(Rademacher ``k=3`` ≈ Haar ``k=2``). For corpora below ~12k chunks the seed-only
index is also smaller *total* than Haar + int8 sidecar.

NORMATIVE cross-language algorithm — every reader MUST reproduce this exactly:

    constants (uint64):
        GOLDEN = 0x9E3779B97F4A7C15
        M1     = 0xBF58476D1CE4E5B9
        M2     = 0x94D049BB133111EB
    entry e at flat C-order index i in the (k, dim, dim) tensor
    (i = (j*dim + row)*dim + col) is:
        z = (seed + (i+1) * GOLDEN)  mod 2**64      # i-th splitmix64 draw
        z = ((z XOR (z >> 30)) * M1) mod 2**64
        z = ((z XOR (z >> 27)) * M2) mod 2**64
        z =  (z XOR (z >> 31))
        entry = -1.0 if (z >> 63) & 1 else +1.0

All arithmetic is unsigned 64-bit modular. JS uses BigInt masked with
``(1n<<64n)-1n`` after each step; numpy uses native uint64 wrap-around.
"""
from __future__ import annotations

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S63 = np.uint64(63)


def _splitmix64_stream(seed: int, n: int) -> np.ndarray:
    """The first ``n`` outputs of a splitmix64 stream seeded by ``seed``."""
    i = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + i * _GOLDEN          # mod 2**64 (uint64 wrap)
        z = (z ^ (z >> _S30)) * _M1
        z = (z ^ (z >> _S27)) * _M2
        z = z ^ (z >> _S31)
    return z


def rademacher_planes(dim: int, k: int, seed: int = 0) -> np.ndarray:
    """``(k, dim, dim)`` float32 of ±1, deterministic from ``(dim, k, seed)``.

    Bit-identical to the ``splitmix64`` reference above on any platform — this is
    the whole point: a reader regenerates these planes from the manifest's
    ``(dim, k, seed)`` instead of loading a shipped matrix.
    """
    n = k * dim * dim
    z = _splitmix64_stream(seed, n)
    bit = (z >> _S63).astype(np.int8)              # top bit: 0 or 1
    planes = (1.0 - 2.0 * bit).astype(np.float32)  # 0 -> +1.0, 1 -> -1.0
    return planes.reshape(k, dim, dim)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _fwht_int(a: np.ndarray) -> np.ndarray:
    """In-place-style integer Walsh-Hadamard along axis 1 (len must be 2**m).

    Pure int64 add/subtract — exact and bit-identical on any platform, and the
    magnitudes stay tiny (≈ few thousand at rounds=3), well inside both int64
    and JS's 2**53 exact-integer range.
    """
    a = a.copy()
    n = a.shape[1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[:, i:i + h].copy()
            y = a[:, i + h:i + h * 2].copy()
            a[:, i:i + h] = x + y
            a[:, i + h:i + h * 2] = x - y
        h *= 2
    return a


def srht_matrix(dim: int, k: int, seed: int = 0, rounds: int = 3) -> np.ndarray:
    """``(k, dim, dim)`` float32 SRHT projection, deterministic from the seed.

    Each stack is ``R`` rounds of (seed-driven ±1 diagonal `D`, then a
    Walsh-Hadamard transform `H`) on a ``dim→pad`` zero-padded space, taken back
    to ``dim``. ``H·D`` is *exactly* orthogonal, so the result is a structured
    orthogonal projection — recovering ~all of Haar's recall edge over plain
    Rademacher (experiments/kb-k-sweep Part 9) while shipping nothing.

    Materialized as an integer matrix (FWHT applied to the identity → exact,
    trivially portable) and then per-output-column L2-normalized into float32 to
    keep the runtime ``x @ M`` matmul well-conditioned. Reproduced bit-for-bit by
    the JS reader (``srhtMatrix`` in ``js/kb-reader.js``) — proven by round-trip.

    NORMATIVE — readers MUST reproduce exactly:
      pad = next power of two ≥ dim
      sign[stack j, round r, pos p] = top-bit→±1 of splitmix64 draw at flat index
                                      ((j*rounds + r)*pad + p)   (see §projection)
      build row d of stack j by transforming the padded basis vector e_d:
          row = e_d (length pad);  for r in 0..rounds-1: row = FWHT(row * sign[j,r])
          M_int[j, d, e] = row[e]      for e in 0..dim-1
      column-normalize: M[j, :, e] = float32( M_int[j, :, e] / ||M_int[j, :, e]|| )
    """
    pad = _next_pow2(dim)
    z = _splitmix64_stream(seed, k * rounds * pad)
    signs = (1 - 2 * (z >> _S63).astype(np.int64)).reshape(k, rounds, pad)
    eye = np.eye(dim, dtype=np.int64)
    mats = np.empty((k, dim, dim), dtype=np.float32)
    for j in range(k):
        Y = np.zeros((dim, pad), dtype=np.int64)
        Y[:, :dim] = eye
        for r in range(rounds):
            Y = _fwht_int(Y * signs[j, r])
        R = Y[:, :dim].astype(np.float64)
        nrm = np.sqrt((R * R).sum(axis=0, keepdims=True))
        nrm[nrm == 0] = 1.0
        mats[j] = (R / nrm).astype(np.float32)
    return mats
