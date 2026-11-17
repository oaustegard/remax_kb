"""remex codec plumbing shared by the v1 and v2 packers and readers.

The rotation is part of a remex encoding the way the seed is: codes written
under one rotation decode to noise under another. remex < 0.7 built every
quantizer with a Haar rotation and remax_kb did not record it; remex >= 1.0
defaults to a randomized Hadamard transform (``"rht"``) for even ``d``. So
writers now record the rotation in ``binarizer.rotation``, and a remex
artifact without that field was, as a matter of fact, encoded with Haar.
Do not track :data:`LEGACY_ROTATION` to remex's current default.
"""
from __future__ import annotations

import numpy as np

#: Rotation of every remex artifact written before ``binarizer.rotation``
#: existed (remax_kb <= 0.5, remex <= 0.6). A decoding fact, not a default.
LEGACY_ROTATION = "haar"


def stored_rotation(binarizer: dict) -> str:
    """Rotation a remex binarizer block was encoded with."""
    return binarizer.get("rotation") or LEGACY_ROTATION


def quantizer(dim: int, bits: int, seed: int, rotation: str | None = None):
    """A ``remex.Quantizer``. ``rotation=None`` takes remex's own default,
    which is what new artifacts use; read the resolved value back from
    ``.rotation`` and record it."""
    from remex import Quantizer

    return Quantizer(d=dim, bits=bits, seed=seed, rotation=rotation)


def decode_rows(packed: np.ndarray, n: int, dim: int, bits: int, seed: int,
                rotation: str) -> np.ndarray:
    """Decode ``n`` bit-packed remex rows to ``(n, dim)`` float32.

    Norms are not stored (the codec requires an L2-normalizing embedder), so
    they are reconstructed as 1.0; remex's renorm then returns unit rows and
    ``rows @ q`` is a cosine estimate.
    """
    from remex import CompressedVectors, unpack

    flat = np.ascontiguousarray(packed).ravel()
    idx = unpack(flat, bits, n * dim).reshape(n, dim).astype(np.uint8)
    comp = CompressedVectors(
        idx, np.ones(n, dtype=np.float32), dim, bits,
        rotation=rotation, seed=seed,
    )
    return quantizer(dim, bits, seed, rotation).decode(comp).astype(np.float32)


def version() -> str:
    try:
        import remex

        return "remex-" + getattr(remex, "__version__", "0.0.0")
    except Exception:
        return "remex-unknown"
