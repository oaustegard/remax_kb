"""Compaction policy surface on the v2 writer.

Incremental sync appends and tombstones; it never reclaims bytes and it
reuses the *frozen* corpus mean. Over many edits the tombstone ratio
climbs and the mean drifts from the live distribution, eroding code
quality. ``compact()`` rebuilds from live rows and recomputes the mean;
``should_compact()`` decides when that's worth the full re-embed.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

from remax_kb.pack import Chunk
from remax_kb.pack_v2 import KBWriter
from remax_kb.read_v2 import KB


class DeterministicEmbedder:
    full_dim = 64
    normalize_l2 = True
    release_url = None
    release_sha256 = None
    model_revision = "test"
    prompts = {"query": "Query: ", "document": "Document: "}

    def __init__(self):
        self.encoded_texts: list[str] = []

    def fingerprint(self):
        return {"model_id": "test/mock", "task_adapter": "retrieval",
                "pooling": "native", "full_dim": self.full_dim}

    def encode(self, texts, *, prompt):
        self.encoded_texts.extend(texts)
        out = np.zeros((len(texts), self.full_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.full_dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-12)
        return out


def _corpus(n=5):
    return [
        Chunk(id=f"post-{i:03d}#chunk-001", text=f"Distinct sentence number {i} about ravens and retrieval.",
              meta={"source": f"post-{i:03d}"})
        for i in range(n)
    ]


def _build(out: Path, emb, chunks):
    w = KBWriter.create(name="test", output_dir=out, embedder=emb, dim=32, k=4, seed=0)
    w.add_chunks(chunks)
    w.commit()
    return w


def test_counts_and_tombstone_ratio():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        corpus = _corpus(5)
        _build(out, DeterministicEmbedder(), corpus)

        w = KBWriter.open(name="test", output_dir=out, embedder=DeterministicEmbedder())
        w.delete_chunks([corpus[0].id])
        w.commit()

        assert w.total_rows == 5
        assert w.live_count == 4
        assert w.tombstone_count == 1
        assert abs(w.tombstone_ratio - 0.2) < 1e-9


def test_should_compact_threshold():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        corpus = _corpus(5)
        _build(out, DeterministicEmbedder(), corpus)

        w = KBWriter.open(name="test", output_dir=out, embedder=DeterministicEmbedder())
        w.delete_chunks([corpus[0].id])
        w.commit()  # ratio == 0.2

        assert w.should_compact(max_tombstone_ratio=0.1) is True
        assert w.should_compact(max_tombstone_ratio=0.5) is False


def test_compact_drops_tombstones_and_reembeds_live():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        corpus = _corpus(5)
        _build(out, DeterministicEmbedder(), corpus)

        emb = DeterministicEmbedder()
        w = KBWriter.open(name="test", output_dir=out, embedder=emb)
        w.delete_chunks([corpus[0].id, corpus[1].id])
        w.commit()
        assert w.total_rows == 5 and w.live_count == 3

        emb.encoded_texts.clear()
        w.compact()

        # Only live rows remain; tombstones gone.
        assert w.total_rows == 3
        assert w.live_count == 3
        assert w.tombstone_ratio == 0.0
        # Compaction re-embeds exactly the live chunks.
        assert len(emb.encoded_texts) == 3

        kb = KB.open(out / "test.kbi")
        assert kb.live_count == 3
        assert kb.manifest["chunks"]["total_rows"] == 3
        # A surviving chunk is still retrievable post-compaction.
        hits = kb.search(corpus[4].text, embedder=DeterministicEmbedder(), k=3)
        assert hits[0].chunk_id == corpus[4].id


def test_empty_writer_does_not_request_compaction():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        w = KBWriter.create(name="test", output_dir=out, embedder=DeterministicEmbedder(),
                            dim=32, k=4, seed=0)
        assert w.total_rows == 0
        assert w.tombstone_ratio == 0.0
        assert w.should_compact() is False
