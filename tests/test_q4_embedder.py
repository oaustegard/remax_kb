"""JinaQ4ONNXEmbedder smoke + fp32-parity test.

Opt-in (heavy deps + model files): set $REMAX_KB_Q4_ONNX_PATH to a locally-built
``model.q4.onnx`` (``python scripts/build_q4_onnx.py model.onnx model.q4.onnx``)
and $REMAX_KB_TOKENIZER_PATH to the tokenizer. If $REMAX_KB_ONNX_PATH (fp32) is
also set, asserts the q4 query vectors stay ~aligned with fp32 (the parity claim
behind shipping q4 as a smaller drop-in runtime).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

Q4 = os.environ.get("REMAX_KB_Q4_ONNX_PATH")
TOK = os.environ.get("REMAX_KB_TOKENIZER_PATH")
FP32 = os.environ.get("REMAX_KB_ONNX_PATH")

if not (Q4 and TOK):
    pytest.skip(
        "set REMAX_KB_Q4_ONNX_PATH + REMAX_KB_TOKENIZER_PATH to run "
        "(build q4 via scripts/build_q4_onnx.py)",
        allow_module_level=True,
    )

from remax_kb.embedders import JinaQ4ONNXEmbedder, JinaONNXEmbedder  # noqa: E402

SENTS = [
    "Retrieval-augmented generation grounds answers in fetched documents.",
    "The mitochondria is the powerhouse of the cell.",
    "ATProto federates social data across personal data servers.",
]


def test_q4_shapes_and_norm():
    emb = JinaQ4ONNXEmbedder(model_path=Q4, tokenizer_path=TOK)
    v = emb.encode(SENTS, prompt="document")
    assert v.shape == (len(SENTS), 768)
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), norms


def test_q4_fingerprint_matches_fp32():
    # same retrieval semantics -> manifest-validation keys unchanged
    assert JinaQ4ONNXEmbedder().fingerprint() == JinaONNXEmbedder().fingerprint()


@pytest.mark.skipif(not FP32, reason="set REMAX_KB_ONNX_PATH (fp32) for parity check")
def test_q4_parity_with_fp32():
    q4 = JinaQ4ONNXEmbedder(model_path=Q4, tokenizer_path=TOK)
    fp = JinaONNXEmbedder(model_path=FP32, tokenizer_path=TOK)
    a = q4.encode(SENTS, prompt="query")
    b = fp.encode(SENTS, prompt="query")
    cos = (a * b).sum(axis=1)  # both L2-normalized
    assert cos.mean() > 0.95, f"q4 drifted from fp32: per-sent cos={cos}"
