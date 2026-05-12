"""Live GeminiEmbedder tests — gated on ``$GEMINI_API_KEY``.

Skipped silently in CI. Run locally with::

    GEMINI_API_KEY=... pytest tests/test_gemini_live.py -q

Hits ``generativelanguage.googleapis.com`` for real. One pack+query
roundtrip plus a single embedding sanity check.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("httpx")
pytest.importorskip("remax")

API_KEY = os.environ.get("GEMINI_API_KEY")
pytestmark = pytest.mark.skipif(
    not API_KEY, reason="GEMINI_API_KEY not set; skipping live API tests"
)


def test_live_embedding_shape():
    from remax_kb.embedders import GeminiEmbedder

    emb = GeminiEmbedder(output_dim=768)
    vecs = emb.encode(["hello world"], prompt="document")
    assert vecs.shape == (1, 768)
    assert vecs.dtype == np.float32
    norm = float(np.linalg.norm(vecs[0]))
    assert abs(norm - 1.0) < 1e-3, f"expected unit norm, got {norm}"


def test_live_pack_and_query(tmp_path: Path):
    """Pack 4 chunks, query for one — the query should land in top-2."""
    from remax_kb import KB, pack
    from remax_kb.embedders import GeminiEmbedder
    from remax_kb.pack import Chunk

    emb = GeminiEmbedder(output_dim=768)
    chunks = [
        Chunk(id="a", text="Cats are small carnivorous mammals.", meta={}),
        Chunk(id="b", text="Dogs are domesticated descendants of wolves.", meta={}),
        Chunk(id="c", text="Photosynthesis converts light energy into chemical energy.", meta={}),
        Chunk(id="d", text="The Roman Empire fell in 476 AD.", meta={}),
    ]
    out = tmp_path / "live.kb"
    pack(chunks, out, embedder=emb, dim=256, k=8, seed=0, source_description="gemini-live")
    kb = KB.open(out)
    hits = kb.search("when did Rome fall", embedder=emb, k=2)
    top_ids = [h[1]["id"] for h in hits]
    assert "d" in top_ids, f"expected Roman Empire chunk in top-2, got {top_ids}"
