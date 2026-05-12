"""GeminiEmbedder unit tests with the HTTP layer mocked.

These tests run in CI with no API key — they monkeypatch ``httpx.Client``
to short-circuit the API call. The live counterpart is
``tests/test_gemini_live.py``, gated on ``$GEMINI_API_KEY``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("httpx")
pytest.importorskip("remax")

from remax_kb import KB, pack
from remax_kb.embedders import GeminiEmbedder
from remax_kb.pack import Chunk


DIM = 32
K = 4
SEED = 11
OUTPUT_DIM = 64


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | str):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if isinstance(payload, dict) else payload
        self.request = None

    def json(self):
        return self._payload

    def raise_for_status(self):
        import httpx
        if 400 <= self.status_code:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=self
            )


class _FakeClient:
    """Records request bodies; returns deterministic per-text vectors."""

    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url: str, *, json: dict) -> _FakeResponse:
        self.calls.append((url, json))
        # Deterministic mock: hash each text to a per-row vector so
        # pack/query roundtrip works (self-search returns the same text).
        import hashlib

        rng_state: list[list[float]] = []
        for req in json["requests"]:
            text = req["content"]["parts"][0]["text"]
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(req["outputDimensionality"]).tolist()
            rng_state.append(v)
        return _FakeResponse(
            200,
            {"embeddings": [{"values": v} for v in rng_state]},
        )


@pytest.fixture
def patched_httpx(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    return _FakeClient


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiEmbedder()


def test_api_key_from_arg(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    emb = GeminiEmbedder(api_key="x", output_dim=OUTPUT_DIM)
    assert emb._api_key == "x"
    assert emb.full_dim == OUTPUT_DIM
    assert emb.model_id == f"google/{emb.model}"


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    emb = GeminiEmbedder()
    assert emb._api_key == "env-key"


def test_fingerprint_shape(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    emb = GeminiEmbedder(output_dim=OUTPUT_DIM)
    fp = emb.fingerprint()
    assert fp["model_id"] == emb.model_id
    assert fp["task_adapter"] == "retrieval"
    assert fp["pooling"] == "native"
    assert fp["full_dim"] == OUTPUT_DIM
    assert fp["task_type_doc"] == "RETRIEVAL_DOCUMENT"
    assert fp["task_type_query"] == "RETRIEVAL_QUERY"


def test_release_url_is_none(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    emb = GeminiEmbedder()
    assert emb.release_url is None
    assert emb.release_sha256 is None


def test_encode_shape_and_normalization(monkeypatch, patched_httpx):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    emb = GeminiEmbedder(output_dim=OUTPUT_DIM)
    vecs = emb.encode(["hello world", "another doc"], prompt="document")
    assert vecs.shape == (2, OUTPUT_DIM)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_task_type_routing(monkeypatch, patched_httpx):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    emb = GeminiEmbedder(output_dim=OUTPUT_DIM)
    # Run two calls so we have call history to inspect on the patched client.
    captured: list[_FakeClient] = []

    real_client_cls = _FakeClient

    def make_client(*args, **kwargs):
        c = real_client_cls(*args, **kwargs)
        captured.append(c)
        return c

    monkeypatch.setattr("httpx.Client", make_client)

    emb.encode(["doc"], prompt="document")
    emb.encode(["q"], prompt="query")
    assert len(captured) == 2
    doc_call = captured[0].calls[0][1]
    q_call = captured[1].calls[0][1]
    assert doc_call["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert q_call["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


def test_unknown_prompt_raises(monkeypatch, patched_httpx):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    emb = GeminiEmbedder(output_dim=OUTPUT_DIM)
    with pytest.raises(ValueError, match="prompt"):
        emb.encode(["x"], prompt="classification")


def test_batching_at_limit(monkeypatch, patched_httpx):
    """When the input exceeds batch_limit, we should issue multiple
    requests in order."""
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    emb = GeminiEmbedder(output_dim=OUTPUT_DIM, batch_limit=2)
    captured: list[_FakeClient] = []

    def make_client(*args, **kwargs):
        c = _FakeClient(*args, **kwargs)
        captured.append(c)
        return c

    monkeypatch.setattr("httpx.Client", make_client)
    vecs = emb.encode(["a", "b", "c", "d", "e"], prompt="document")
    assert vecs.shape == (5, OUTPUT_DIM)
    # One client per encode() call; this one batched internally.
    assert len(captured[0].calls) == 3  # ceil(5/2) = 3


def test_retry_on_transient_error(monkeypatch):
    """429 then 200 should succeed without raising."""
    import httpx

    class FlakyClient:
        attempts = 0

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, *, json):
            FlakyClient.attempts += 1
            if FlakyClient.attempts == 1:
                return _FakeResponse(429, {"error": "rate limited"})
            return _FakeResponse(
                200,
                {"embeddings": [{"values": [1.0] * OUTPUT_DIM}]},
            )

    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(httpx, "Client", FlakyClient)
    # Make the sleep a no-op so the test runs fast.
    monkeypatch.setattr(GeminiEmbedder, "_sleep_backoff", staticmethod(lambda a: None))
    emb = GeminiEmbedder(output_dim=OUTPUT_DIM, max_retries=3)
    vecs = emb.encode(["hello"], prompt="document")
    assert vecs.shape == (1, OUTPUT_DIM)
    assert FlakyClient.attempts == 2


def test_pack_with_gemini_then_open(monkeypatch, patched_httpx, tmp_path: Path):
    """Pack a tiny corpus with a mocked Gemini embedder, then open the
    .kb and verify the manifest carries no release_url (API-backed)."""
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    emb = GeminiEmbedder(output_dim=OUTPUT_DIM)
    chunks = [
        Chunk(id=f"c-{i}", text=f"chunk text {i} with words", meta={"row": i})
        for i in range(10)
    ]
    out = tmp_path / "gemini-mock.kb"
    pack(chunks, out, embedder=emb, dim=DIM, k=K, seed=SEED, source_description="gemini-mock")
    kb = KB.open(out)
    assert len(kb) == 10
    assert kb.manifest.embedder.model_id == emb.model_id
    assert kb.manifest.embedder.release_url is None
    assert kb.manifest.embedder.release_sha256 is None
    assert kb.manifest.embedder.full_dim == OUTPUT_DIM
    # Roundtrip search: query with same embedder, verify fingerprint match passes.
    hits = kb.search("chunk text 3 with words", embedder=emb, k=3)
    assert len(hits) == 3
