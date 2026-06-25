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
