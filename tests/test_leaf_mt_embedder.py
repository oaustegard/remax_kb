"""LeafMTONNXEmbedder smoke + pooling-parity test.

Opt-in (needs the model files). Either let the embedder fetch and SHA-verify
them from the pinned HF revision, or point at a local copy:

    REMAX_KB_LEAF_MT_ONNX_PATH=.../model_quantized.onnx \
    REMAX_KB_LEAF_MT_TOKENIZER_PATH=.../tokenizer.json \
        pytest tests/test_leaf_mt_embedder.py

The parity test is the load-bearing one. This embedder reads the graph's own
``sentence_embedding`` output instead of pooling by hand, which is only correct
if that output really is the upstream mean-pool + 384->1024 Dense head. The test
recomputes the manual path from ``last_hidden_state`` and asserts they agree, so
a future upstream export that changes the graph's pooling fails here rather than
silently producing a ``.kb`` whose vectors do not match its ``pooling`` field.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

ONNX = os.environ.get("REMAX_KB_LEAF_MT_ONNX_PATH")
TOK = os.environ.get("REMAX_KB_LEAF_MT_TOKENIZER_PATH")
NETWORK = os.environ.get("REMAX_KB_ALLOW_DOWNLOAD") == "1"

if not ((ONNX and TOK) or NETWORK):
    pytest.skip(
        "set REMAX_KB_LEAF_MT_ONNX_PATH + REMAX_KB_LEAF_MT_TOKENIZER_PATH to a "
        "local copy, or REMAX_KB_ALLOW_DOWNLOAD=1 to fetch the pinned revision",
        allow_module_level=True,
    )

from remax_kb.embedders import LEAF_MT_FULL_DIM, LeafMTONNXEmbedder  # noqa: E402

SENTS = [
    "Retrieval-augmented generation grounds answers in fetched documents.",
    "Recover the password for an encrypted archive.",
    "List every file under a directory, including hidden ones.",
]


def _embedder() -> LeafMTONNXEmbedder:
    kw = {}
    if ONNX:
        kw["model_path"] = Path(ONNX)
    if TOK:
        kw["tokenizer_path"] = Path(TOK)
    return LeafMTONNXEmbedder(**kw)


def test_shapes_and_norm():
    v = _embedder().encode(SENTS, prompt="document")
    assert v.shape == (len(SENTS), LEAF_MT_FULL_DIM)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)


def test_empty_input_returns_empty_matrix():
    v = _embedder().encode([], prompt="query")
    assert v.shape == (0, LEAF_MT_FULL_DIM)


def test_unknown_prompt_rejected():
    with pytest.raises(ValueError):
        _embedder().encode(SENTS, prompt="passage")


def test_query_prefix_changes_the_vector():
    """The prefix is load-bearing upstream; assert it is actually applied."""
    emb = _embedder()
    q = emb.encode(SENTS[:1], prompt="query")
    d = emb.encode(SENTS[:1], prompt="document")
    assert float(q[0] @ d[0]) < 0.999


def test_batching_preserves_order_and_near_preserves_value():
    """Length-sorted batching must scatter back to input order.

    Values are asserted by cosine, not elementwise: the int8 graph is
    padding-sensitive, so a row's vector shifts slightly with the batch it was
    padded against (~0.99 cosine, measured). That is a property of the
    quantized transformer, not of the pooling — a manual masked mean over
    ``last_hidden_state`` drifts by the same amount. See the class docstring:
    it means a ``.kb``'s bit-identity is conditional on ``batch_size``.
    """
    kw = {k: v for k, v in (("model_path", Path(ONNX) if ONNX else None),
                            ("tokenizer_path", Path(TOK) if TOK else None))
          if v is not None}
    one = LeafMTONNXEmbedder(**kw, batch_size=16).encode(SENTS, prompt="document")
    small = LeafMTONNXEmbedder(**kw, batch_size=1).encode(SENTS, prompt="document")
    cos = (one * small).sum(axis=1)  # both L2-normalized
    assert cos.min() > 0.98, f"batching changed vectors more than expected: {cos}"
    # Order must be exact even though values are not: the highest-cosine
    # pairing has to be the diagonal, or the scatter-back is wrong.
    assert np.argmax(one @ small.T, axis=1).tolist() == list(range(len(SENTS)))


def test_graph_pooling_is_mean_plus_dense():
    """`sentence_embedding` must equal a manual masked mean + Dense head.

    Guards the shortcut this embedder takes. If upstream re-exports with, say,
    CLS pooling, the vectors would change silently and every existing .kb built
    with this embedder would become incomparable — this fails instead.
    """
    import onnxruntime as ort
    from tokenizers import Tokenizer

    emb = _embedder()
    emb._load()
    sess: ort.InferenceSession = emb._session
    tok: Tokenizer = emb._tokenizer
    names = {o.name for o in sess.get_outputs()}
    assert {"last_hidden_state", "sentence_embedding"} <= names

    encoded = tok.encode_batch(list(SENTS))
    max_len = max(len(e.ids) for e in encoded)
    ids = np.zeros((len(encoded), max_len), dtype=np.int64)
    mask = np.zeros((len(encoded), max_len), dtype=np.int64)
    for i, e in enumerate(encoded):
        L = len(e.ids)
        ids[i, :L] = e.ids
        mask[i, :L] = e.attention_mask
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in {i.name for i in sess.get_inputs()}:
        feed["token_type_ids"] = np.zeros_like(ids)

    hidden, pooled = sess.run(["last_hidden_state", "sentence_embedding"], feed)
    m = mask.astype(np.float32)[..., None]
    mean = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)

    # Dense is linear, so a least-squares fit recovers it from this batch and
    # must reproduce `sentence_embedding` — no safetensors read required.
    w, *_ = np.linalg.lstsq(mean, pooled, rcond=None)
    recon = mean @ w
    def unit(x):
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)
    cos = (unit(recon) * unit(pooled)).sum(axis=1)
    assert cos.min() > 0.99, f"graph pooling is not mean+linear: {cos}"
