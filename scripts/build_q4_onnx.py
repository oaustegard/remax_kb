"""Build the int4 (q4) Jina v5-nano retrieval ONNX from the fp32 export.

    python scripts/build_q4_onnx.py model.onnx model.q4.onnx

Recipe (deterministic):
  1. MatMulNBits 4-bit blockwise (block_size=32, symmetric) on the linear
     weights. This alone does NOT shrink the model below int8 — it leaves
     EuroBERT's large multilingual embedding table (a ``Gather`` initializer,
     ~400 MB fp32) untouched, so naive int4 is ~465 MB > int8's ~212 MB.
  2. int8 ``quantize_dynamic`` mop-up over the leftover graph, which quantizes
     that embedding table. Final: ~170 MB, ~5x smaller than fp32, runs on the
     CPU EP with retrieval parity to fp32 (see JinaQ4ONNXEmbedder docstring).

Requires: onnx, onnxruntime, onnx_ir (for MatMulNBitsQuantizer). Note: 3-bit is
unsupported (ORT asserts bits in {2,4,8}); 2-bit runs but halves recall.
"""
from __future__ import annotations

import sys
from pathlib import Path


def build_q4(src: str | Path, dst: str | Path, *, bits: int = 4,
             block_size: int = 32) -> Path:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    src, dst = Path(src), Path(dst)
    tmp = dst.with_suffix(".matmul.onnx")
    model = onnx.load(str(src))
    q = MatMulNBitsQuantizer(model, bits=bits, block_size=block_size, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(str(tmp), use_external_data_format=False)
    # int8 mop-up: catches the embedding Gather + any leftover fp32 weights.
    quantize_dynamic(str(tmp), str(dst), weight_type=QuantType.QUInt8)
    tmp.unlink(missing_ok=True)
    return dst


def _main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    out = build_q4(sys.argv[1], sys.argv[2])
    mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
