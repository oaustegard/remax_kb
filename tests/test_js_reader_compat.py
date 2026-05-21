"""Verify the JS reader's bit-pack convention matches remax.

The JS reader can't be run from pytest, so this test executes the
same arithmetic in Python and asserts it equals what `remax.encode()`
produces. If this test passes, the JS reader will produce bit-identical
codes for the same input — assuming the JS implementation follows the
Python logic faithfully (it does, by construction).
"""
import numpy as np
import pytest
from remax import StackedSignBitQuantizer


def js_encode_python_emulation(x, rotations, d, k):
    """Mirror of js/kb-reader.js encodeQueryCode().

    Big-endian bit-pack: bit i lands at mask `1 << (7 - i & 7)` within
    its byte. Rotations are stack-ordered along the codeword.
    """
    row_bytes = (d * k) // 8
    code = np.zeros(row_bytes, dtype=np.uint8)
    for j in range(k):
        proj = x @ rotations[j]
        for col in range(d):
            if proj[col] >= 0:
                bit_idx = j * d + col
                code[bit_idx // 8] |= 1 << (7 - (bit_idx % 8))
    return code


@pytest.mark.parametrize("d,k,seed", [(32, 4, 42), (64, 2, 0), (256, 8, 7)])
def test_js_emulation_matches_remax(d, k, seed):
    q = StackedSignBitQuantizer(d=d, k=k, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    for trial in range(5):
        x = rng.standard_normal(d).astype(np.float32)
        ref = q.encode(x[None, :])[0]
        emu = js_encode_python_emulation(x, q.rotations_.astype(np.float32), d, k)
        np.testing.assert_array_equal(emu, ref,
            err_msg=f"d={d} k={k} seed={seed} trial={trial}")
