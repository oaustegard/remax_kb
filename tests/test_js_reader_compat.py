"""Verify the JS reader's bit-pack convention matches remax.

These tests execute the JS reader's arithmetic *re-implemented in Python* and
assert it equals what `remax.encode()` produces. That covers the convention
(bit order, stack order, dequant layout) across many (d, k, seed) settings —
but it cannot catch a divergence between the Python transcription and the
actual JavaScript, because a transcription is not evidence about its original.
The old claim here — "the JS implementation follows the Python logic
faithfully (it does, by construction)" — was exactly the assumption under test.

The JS reader IS executed now: `tests/gates/gate_cross_reader.py` runs
`js/kb-reader.js` in Node over a committed fixture and asserts byte-identical
query codes and matching top-k against `remax_kb.read_v2`. It covers one
configuration deeply; these tests cover many configurations shallowly. Keep
both.
"""
import numpy as np
import pytest
from remax import StackedSignBitQuantizer


def js_encode_python_emulation(x, rotations, d, k):
    """Mirror of js/kb-reader.js encodeQueryCode().

    Big-endian bit-pack: bit i lands at mask `1 << (7 - i & 7)` within
    its byte. Rotations are stack-ordered along the codeword.

    STRICT ``> 0``, mirroring the reader after the 2026-08 sign-convention fix
    and matching ``remax.packing.encode_signs`` ("x > 0 → bit 1"). ``>= 0``
    here would silently re-introduce the divergence this file is meant to
    detect — see ``test_exact_zero_projection_packs_like_remax``.
    """
    row_bytes = (d * k) // 8
    code = np.zeros(row_bytes, dtype=np.uint8)
    for j in range(k):
        proj = x @ rotations[j]
        for col in range(d):
            if proj[col] > 0:
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


def test_exact_zero_projection_packs_like_remax():
    """Issue #20 mechanism (a): the sign convention AT exactly 0.0.

    Constructed, not sampled. With ±1 rademacher planes and ``x = e_0 - e_1``,
    every output column where the two plane entries agree projects to exactly
    0.0 — in float32, in float64, under any summation order. So this test
    isolates the *convention* from the float-summation-order mechanism (b),
    which no amount of sampling can separate. ``>= 0`` fails it; ``> 0`` passes.

    The Node-executed counterpart lives in tests/gates/gate_cross_reader.py
    (`exact_zero_sign_parity`), which drives the real JS through the same
    construction and carries a known-bad that restores ``>= 0``.
    """
    from remax_kb.projection import rademacher_planes

    d, k, seed = 32, 2, 12345
    planes = rademacher_planes(d, k, seed).astype(np.float32)
    x = np.zeros(d, dtype=np.float32)
    x[0], x[1] = 1.0, -1.0

    proj = np.concatenate([x @ planes[j] for j in range(k)])
    n_zero = int((proj == 0.0).sum())
    assert 0 < n_zero < d * k, (
        f"probe is vacuous: {n_zero} exact zeros out of {d * k}")

    q = StackedSignBitQuantizer(d=d, k=k, seed=seed)
    q.rotations_ = planes.astype(q.dtype)
    for v in (x, -x):
        ref = q.encode(np.asarray(v)[None, :])[0]
        emu = js_encode_python_emulation(v, planes, d, k)
        np.testing.assert_array_equal(emu, ref)


def js_dequant_int8_emulation(codes_i8, scale, d, k):
    """Mirror of the int8 dequant the JS reader must perform on load:
    rot[j, row, col] = i8[j, row, col] * scale[j, col]  (per-output-column).
    Returns a (k, d, d) f32 array in the same layout as rotations.f32.
    """
    rot = np.empty((k, d, d), dtype=np.float32)
    for j in range(k):
        for col in range(d):
            s = scale[j, col]
            for row in range(d):
                rot[j, row, col] = codes_i8[j, row, col] * s
    return rot


@pytest.mark.xfail(
    reason="issue #20 mechanism (b) ONLY: float summation order. The emulation "
    "projects one stack at a time (x @ rotations[j]) while remax does a single "
    "matmul against the pre-flattened (d, k*d) matrix, so BLAS may accumulate "
    "in a different order and land on the opposite side of zero for a NEAR-zero "
    "projected coordinate. Mechanism (a), the `>= 0` vs `> 0` sign convention "
    "at EXACTLY 0.0, was closed 2026-08 and is now pinned by "
    "test_exact_zero_projection_packs_like_remax plus the exact-zero probe in "
    "tests/gates/gate_cross_reader.py — so this marker no longer covers it. "
    "Kept strict=False because these dequantized-int8 planes have never "
    "actually produced a near-zero coordinate at these (d, k, seed): the "
    "parametrizations xpass, on the arithmetic staying lucky rather than on any "
    "guarantee. Making it strict would assert a divergence that is not there; "
    "removing it would assert an agreement nothing in the code enforces.",
    strict=False,
)
@pytest.mark.parametrize("d,k,seed", [(32, 4, 42), (64, 2, 0), (256, 8, 7)])
def test_js_int8_dequant_then_encode_matches_packer(d, k, seed):
    """JS path for an int8 .kbi: dequant the shipped int8 rotations, then encode
    the query. Must equal a code produced from the packer's dequantized
    rotations — i.e. the corpus and the JS-encoded query share one sign-space.
    """
    from remax_kb.rotations import quantize_int8, dequantize_int8
    q = StackedSignBitQuantizer(d=d, k=k, seed=seed)
    codes_i8, scale = quantize_int8(q.rotations_.astype(np.float32))
    deq_ref = dequantize_int8(codes_i8, scale)            # packer-side
    deq_js = js_dequant_int8_emulation(codes_i8, scale, d, k)  # JS-side
    np.testing.assert_array_equal(deq_js, deq_ref)

    rng = np.random.default_rng(seed + 2000)
    q_ref = StackedSignBitQuantizer(d=d, k=k, seed=seed)
    q_ref.rotations_ = deq_ref.astype(q_ref.dtype)
    for trial in range(5):
        x = rng.standard_normal(d).astype(np.float32)
        ref = q_ref.encode(x[None, :])[0]
        emu = js_encode_python_emulation(x, deq_js, d, k)
        np.testing.assert_array_equal(emu, ref,
            err_msg=f"int8 d={d} k={k} seed={seed} trial={trial}")


def js_rademacher_emulation(dim, k, seed):
    """Mirror of js/kb-reader.js rademacherPlanes() using Python big-ints with
    explicit 64-bit masking — proves the JS BigInt transcription matches."""
    MASK = (1 << 64) - 1
    GOLDEN = 0x9E3779B97F4A7C15
    M1, M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
    n = k * dim * dim
    out = np.empty(n, dtype=np.float32)
    s = seed & MASK
    for i in range(n):
        z = (s + (i + 1) * GOLDEN) & MASK
        z = ((z ^ (z >> 30)) * M1) & MASK
        z = ((z ^ (z >> 27)) * M2) & MASK
        z = (z ^ (z >> 31)) & MASK
        out[i] = -1.0 if (z >> 63) & 1 else 1.0
    return out.reshape(k, dim, dim)


@pytest.mark.parametrize("dim,k,seed", [(8, 2, 0), (16, 3, 7), (64, 2, 42)])
def test_js_rademacher_matches_python(dim, k, seed):
    from remax_kb.projection import rademacher_planes
    np.testing.assert_array_equal(
        js_rademacher_emulation(dim, k, seed), rademacher_planes(dim, k, seed))


def js_srht_emulation(dim, k, seed, rounds):
    """Mirror of js/kb-reader.js srhtMatrix() — integer FWHT + float32 column norm."""
    import numpy as np
    MASK=(1<<64)-1; GOLDEN=0x9E3779B97F4A7C15; M1=0xBF58476D1CE4E5B9; M2=0x94D049BB133111EB
    pad=1
    while pad<dim: pad<<=1
    nsign=k*rounds*pad; sign=np.empty(nsign,dtype=np.int64); s=seed&MASK
    for i in range(nsign):
        z=(s+(i+1)*GOLDEN)&MASK; z=((z^(z>>30))*M1)&MASK; z=((z^(z>>27))*M2)&MASK; z=(z^(z>>31))&MASK
        sign[i]=-1 if (z>>63)&1 else 1
    def fwht(a):
        h=1
        while h<pad:
            for i in range(0,pad,h*2):
                for j in range(i,i+h):
                    x=a[j]; y=a[j+h]; a[j]=x+y; a[j+h]=x-y
            h*=2
    out=np.empty(k*dim*dim,dtype=np.float32)
    for jj in range(k):
        R=np.zeros((dim,dim),dtype=np.float64)
        for d in range(dim):
            row=np.zeros(pad,dtype=np.float64); row[d]=1.0
            for r in range(rounds):
                off=(jj*rounds+r)*pad
                for p in range(pad): row[p]*=sign[off+p]
                fwht(row)
            R[d]=row[:dim]
        for e in range(dim):
            nrm=np.sqrt((R[:,e]**2).sum()) or 1.0
            for d in range(dim): out[jj*dim*dim+d*dim+e]=np.float32(R[d,e]/nrm)
    return out.reshape(k,dim,dim)


@pytest.mark.parametrize("dim,k,seed,rounds", [(8,2,0,2),(16,2,7,3),(64,2,42,3)])
def test_js_srht_matches_python(dim,k,seed,rounds):
    from remax_kb.projection import srht_matrix
    np.testing.assert_array_equal(js_srht_emulation(dim,k,seed,rounds),
                                  srht_matrix(dim,k,seed,rounds))
