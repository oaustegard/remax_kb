"""LFM25Embedder unit tests with the transformers load mocked.

These run in CI with no torch, no transformers, and no network: the test
injects fake tokenizer/model objects into the embedder, which makes
``_load()`` a no-op (it early-returns once ``_model`` is set) and keeps
``encode()`` on its duck-typed path. The mocking style mirrors
``tests/test_gemini.py``.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from remax_kb.embedders import (
    LFM25_FULL_DIM,
    LFM25_MODEL_ID,
    LFM25_REVISION,
    LFM25Embedder,
    _check_transformers_version,
)


DIM = LFM25_FULL_DIM


def _vec_for(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic per-text vector, so batching/order can be asserted."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    # Deliberately un-normalized (norm != 1) so the L2 assertion is meaningful.
    return (rng.standard_normal(dim) * 7.5).astype(np.float32)


class _FakeBatch(dict):
    """Stands in for a transformers ``BatchEncoding``."""

    def __init__(self, texts: list[str]):
        super().__init__(texts=texts)
        self.moved_to: str | None = None

    def to(self, device):
        self.moved_to = device
        return self


class _FakeTensor:
    """3-D (N, S, D) fake ``last_hidden_state`` exposing just the torch
    surface ``encode()`` touches: ``[:, 0]``, detach/cpu/float/numpy."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __getitem__(self, key):
        return _FakeTensor(self._arr[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._arr


class _FakeOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class _FakeTokenizer:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return _FakeBatch(list(texts))


class _FakeModel:
    """Returns a (N, S, D) hidden state whose CLS row (position 0) is the
    deterministic per-text vector; other positions are junk, so a wrong
    pooling choice would be caught."""

    def __init__(self, seq_len: int = 3):
        self.batches: list[list[str]] = []
        self._seq_len = seq_len

    def __call__(self, *, texts):
        self.batches.append(list(texts))
        n, s = len(texts), self._seq_len
        arr = np.full((n, s, DIM), -999.0, dtype=np.float32)
        for i, t in enumerate(texts):
            arr[i, 0] = _vec_for(t)
        return _FakeOutput(_FakeTensor(arr))


def _mocked(**kwargs) -> tuple[LFM25Embedder, _FakeTokenizer, _FakeModel]:
    emb = LFM25Embedder(**kwargs)
    tok, model = _FakeTokenizer(), _FakeModel()
    emb._tokenizer = tok
    emb._model = model          # makes _load() a no-op → no torch import
    emb._torch = None           # → contextlib.nullcontext()
    emb._device = None
    return emb, tok, model


# --------------------------------------------------------------------- #
# protocol surface
# --------------------------------------------------------------------- #


def test_protocol_attributes():
    emb = LFM25Embedder()
    assert emb.model_id == "LiquidAI/LFM2.5-Embedding-350M" == LFM25_MODEL_ID
    assert emb.model_revision == LFM25_REVISION
    assert emb.task_adapter == "retrieval"
    assert emb.pooling == "cls"
    assert emb.full_dim == 1024
    assert emb.normalize_l2 is True
    assert emb.release_url is None
    assert emb.release_sha256 is None
    assert emb.prompts == {"query": "query: ", "document": "document: "}


def test_fingerprint_matches_manifest_keys():
    from remax_kb.manifest import Binarizer, CorpusInfo, Embedder, Manifest, Prompts

    emb = LFM25Embedder()
    fp = emb.fingerprint()
    assert fp == {
        "model_id": LFM25_MODEL_ID,
        "task_adapter": "retrieval",
        "pooling": "cls",
        "full_dim": 1024,
    }
    # The reader validates a manifest against exactly these four keys.
    m = Manifest(
        spec_version="1",
        embedder=Embedder(
            model_id=emb.model_id,
            model_revision=emb.model_revision,
            task_adapter=emb.task_adapter,
            pooling=emb.pooling,
            normalize_l2=emb.normalize_l2,
            full_dim=emb.full_dim,
            release_url=emb.release_url,
            release_sha256=emb.release_sha256,
        ),
        prompts=Prompts(**emb.prompts),
        binarizer=Binarizer.from_mean(
            remax_version="0", dim=64, k=4, seed=1,
            mean_vector=np.zeros(emb.full_dim, dtype=np.float32),
        ),
        corpus=CorpusInfo(chunk_count=0, build_hash="x", built_at="now"),
    )
    m.validate_static()
    m.validate_against_embedder(fp)


def test_import_is_torch_free():
    """Constructing the embedder must not import torch/transformers."""
    import subprocess
    import sys

    code = (
        "import sys, remax_kb.embedders as e; "
        "e.LFM25Embedder(); "
        "assert 'torch' not in sys.modules, 'torch imported at construct time'; "
        "assert 'transformers' not in sys.modules, 'transformers imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------------- #
# encode()
# --------------------------------------------------------------------- #


def test_prompt_prefixes_are_applied():
    emb, tok, _ = _mocked()
    emb.encode(["alpha"], prompt="document")
    emb.encode(["alpha"], prompt="query")
    assert tok.calls[0]["texts"] == ["document: alpha"]
    assert tok.calls[1]["texts"] == ["query: alpha"]
    # Trailing space is part of the prefix, not stripped.
    assert tok.calls[0]["texts"][0].startswith("document: ")
    assert tok.calls[1]["texts"][0].startswith("query: ")


def test_tokenizer_truncation_settings():
    emb, tok, _ = _mocked()
    emb.encode(["x"], prompt="document")
    call = tok.calls[0]
    assert call["truncation"] is True
    assert call["padding"] is True
    assert call["max_length"] == 512
    assert call["return_tensors"] == "pt"


def test_output_shape_dtype_and_l2_normalized():
    emb, _, _ = _mocked()
    vecs = emb.encode(["a", "bb", "ccc"], prompt="document")
    assert vecs.shape == (3, 1024)
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_cls_pooling_takes_position_zero():
    """Non-CLS positions are -999 in the fake; a mean/last pool would show it."""
    emb, _, _ = _mocked()
    vecs = emb.encode(["only"], prompt="document")
    want = _vec_for("document: only")
    want = want / np.linalg.norm(want)
    np.testing.assert_allclose(vecs[0], want, atol=1e-5)


def test_zero_norm_guard():
    emb, _, model = _mocked()

    class _ZeroModel(_FakeModel):
        def __call__(self, *, texts):
            self.batches.append(list(texts))
            arr = np.zeros((len(texts), 3, DIM), dtype=np.float32)
            return _FakeOutput(_FakeTensor(arr))

    emb._model = _ZeroModel()
    vecs = emb.encode(["a", "b"], prompt="document")
    assert vecs.shape == (2, 1024)
    assert np.all(np.isfinite(vecs))
    assert not np.any(np.isnan(vecs))


def test_empty_input():
    emb, tok, model = _mocked()
    vecs = emb.encode([], prompt="document")
    assert vecs.shape == (0, 1024)
    assert vecs.dtype == np.float32
    # No model/tokenizer traffic at all for empty input.
    assert tok.calls == [] and model.batches == []


def test_unknown_prompt_raises():
    emb, _, _ = _mocked()
    with pytest.raises(ValueError, match="unknown prompt"):
        emb.encode(["x"], prompt="classification")
    with pytest.raises(ValueError, match="unknown prompt"):
        emb.encode(["x"], prompt="passage")


def test_unknown_prompt_checked_before_load():
    """Rejection must not require a loaded model (no download on a typo)."""
    emb = LFM25Embedder()
    with pytest.raises(ValueError, match="unknown prompt"):
        emb.encode(["x"], prompt="nope")


# --------------------------------------------------------------------- #
# mini-batching + length-sort scatter
# --------------------------------------------------------------------- #


def test_batching_covers_all_rows_in_input_order():
    """With batch_size=3 and length-scrambled inputs, every row must come
    back in *input* order despite the internal length-sort."""
    emb, tok, model = _mocked(batch_size=3)
    texts = [
        "wwww" * 9,   # 0: longest
        "x",          # 1: shortest
        "yy" * 5,     # 2
        "zzz" * 3,    # 3
        "q" * 20,     # 4
        "rr",         # 5
        "s" * 14,     # 6
        "t",          # 7
    ]
    vecs = emb.encode(texts, prompt="document")
    assert vecs.shape == (len(texts), 1024)

    # Every text embedded exactly once, batches capped at batch_size.
    seen = [t for b in model.batches for t in b]
    assert sorted(seen) == sorted(f"document: {t}" for t in texts)
    assert len(model.batches) == 3  # ceil(8/3)
    assert all(len(b) <= 3 for b in model.batches)

    # Batches are length-sorted (the point of the sort: homogeneous padding).
    lengths = [len(t) for b in model.batches for t in b]
    assert lengths == sorted(lengths)

    # Scatter-back is correct: row i is the embedding of prefixed texts[i].
    for i, t in enumerate(texts):
        want = _vec_for(f"document: {t}")
        want = want / np.linalg.norm(want)
        np.testing.assert_allclose(vecs[i], want, atol=1e-5, err_msg=f"row {i}")


def test_single_batch_when_input_fits():
    emb, _, model = _mocked(batch_size=8)
    emb.encode(["a", "b", "c"], prompt="query")
    assert len(model.batches) == 1


def test_default_batch_size_is_small():
    """One-shot encode of a large corpus OOMs (bench/RESULTS_q4_official_vs_ours.md)."""
    emb = LFM25Embedder()
    assert emb._batch_size == 8
    emb2, _, model = _mocked()
    emb2.encode([f"doc {i}" for i in range(20)], prompt="document")
    assert len(model.batches) == 3  # ceil(20/8)
    assert max(len(b) for b in model.batches) == 8


def test_duplicate_texts_all_get_rows():
    emb, _, model = _mocked(batch_size=2)
    vecs = emb.encode(["same", "same", "same"], prompt="document")
    assert vecs.shape == (3, 1024)
    np.testing.assert_allclose(vecs[0], vecs[1], atol=1e-6)
    np.testing.assert_allclose(vecs[1], vecs[2], atol=1e-6)


# --------------------------------------------------------------------- #
# transformers pin
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("version", ["5.12.0", "5.13.2", "6.0.0", "5.12"])
def test_transformers_too_new_raises(version):
    with pytest.raises(RuntimeError, match="seq_idx"):
        _check_transformers_version(version)


@pytest.mark.parametrize("version", ["4.57.6", "5.11.9", "4.44.0", "5.0.1"])
def test_transformers_version_ok(version):
    _check_transformers_version(version)  # must not raise


def test_unparseable_transformers_version_does_not_block():
    _check_transformers_version("dev")  # must not raise


# --------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------- #


def test_cli_builds_lfm25():
    import argparse

    from remax_kb.cli import _build_embedder

    emb = _build_embedder("lfm25", argparse.Namespace())
    assert isinstance(emb, LFM25Embedder)
    assert emb.model_id == LFM25_MODEL_ID
