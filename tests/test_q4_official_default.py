"""Regression: the q4 embedder defaults to the OFFICIAL upstream q4 (remax_kb#23).

Network-free — asserts wiring only (constants + class hierarchy). The heavy
encode/parity checks live in test_q4_embedder.py (opt-in via env model paths).
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from remax_kb import embedders as E  # noqa: E402


def test_default_q4_sources_official_split_onnx():
    # The default class carries no single-file release_url — the official q4 is a
    # two-file (graph + external-data) asset, downloaded in code, pinned by sha.
    assert E.JinaQ4ONNXEmbedder.release_url is None
    assert E.JINA_V5_NANO_OFFICIAL_Q4_REPO.endswith("text-nano-retrieval")
    assert len(E.JINA_V5_NANO_OFFICIAL_Q4_REVISION) == 40  # pinned commit
    for url in (E.JINA_V5_NANO_OFFICIAL_Q4_ONNX_URL, E.JINA_V5_NANO_OFFICIAL_Q4_DATA_URL):
        assert url.startswith("https://huggingface.co/")
        assert E.JINA_V5_NANO_OFFICIAL_Q4_REVISION in url
    for sha in (E.JINA_V5_NANO_OFFICIAL_Q4_ONNX_SHA256, E.JINA_V5_NANO_OFFICIAL_Q4_DATA_SHA256):
        assert len(sha) == 64


def test_ours_q4_is_explicit_optin_only():
    # Our earlier build is reachable only via the example-only subclass, which
    # still points at the mirror release for provenance/reproducibility.
    assert issubclass(E.JinaOursQ4ONNXEmbedder, E.JinaQ4ONNXEmbedder)
    assert E.JinaOursQ4ONNXEmbedder.release_url == E.JINA_V5_NANO_Q4_ONNX_URL
    assert E.JinaOursQ4ONNXEmbedder.release_sha256 == E.JINA_V5_NANO_Q4_ONNX_SHA256


def test_q4_fingerprint_unchanged():
    # Same retrieval semantics as fp32 -> .kb packed against fp32 still queries.
    assert E.JinaQ4ONNXEmbedder().fingerprint() == E.JinaONNXEmbedder().fingerprint()
    assert E.JinaOursQ4ONNXEmbedder().fingerprint() == E.JinaONNXEmbedder().fingerprint()
