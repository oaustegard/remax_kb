"""int8 quantization of the binarizer's Haar rotation matrices.

The shipped ``binarizer/rotations`` sidecar (``k`` ``dim``×``dim`` matrices)
exists so non-NumPy readers can encode a query into the same sign-space as the
corpus codes, since they can't reproduce NumPy's Haar QR. It is
corpus-independent (``k·dim²`` elements) and therefore dominates a *small*
``.kbi``. The matrices feed only a sign test (``x·Q ≥ 0``), so f32 precision is
overkill: int8 with a per-output-column scale shrinks the sidecar 4× and, on a
real corpus, flips ~0.24 % of code bits with no measurable recall loss
(experiments/kb-k-sweep in oaustegard/claude-workspace).

Canonical layout — every reader (Python, JS, …) MUST agree on this:

- ``binarizer/rotations.i8``      — ``int8[k, dim, dim]``, C-order (row-major),
                                     identical element order to ``rotations.f32``.
- ``binarizer/rotations.scale.f32`` — ``float32[k, dim]`` little-endian, one
                                     scale per (stack ``j``, output column ``e``).
- dequantized ``Q[j, d, e] = i8[j, d, e] * scale[j, e]``.

The column axis ``e`` is the output of the projection (the hyperplane index);
quantizing per-column keeps the per-hyperplane decision boundary tight.
"""
from __future__ import annotations

import numpy as np

I8_MAX = 127.0


def quantize_int8(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize ``(k, dim, dim)`` f32 rotations to per-column int8.

    Returns ``(codes_i8, scale_f32)`` where ``codes_i8`` is ``int8[k, dim, dim]``
    and ``scale_f32`` is ``float32[k, dim]`` (per stack, per output column).
    """
    R = np.asarray(rotations, dtype=np.float32)
    if R.ndim != 3 or R.shape[1] != R.shape[2]:
        raise ValueError(f"expected (k, dim, dim), got {R.shape}")
    # max abs over the input-row axis (axis=1) → one scale per output column.
    col_max = np.abs(R).max(axis=1)                      # (k, dim)
    scale = (col_max / I8_MAX).astype(np.float32)
    scale[scale == 0.0] = 1.0                            # avoid divide-by-zero
    codes = np.round(R / scale[:, None, :]).astype(np.int8)  # round-half-to-even
    return codes, scale


def dequantize_int8(codes_i8: np.ndarray, scale_f32: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_int8` → ``(k, dim, dim)`` f32."""
    codes = np.asarray(codes_i8, dtype=np.int8).astype(np.float32)
    scale = np.asarray(scale_f32, dtype=np.float32)
    return (codes * scale[:, None, :]).astype(np.float32)
