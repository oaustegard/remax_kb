"""Content-addressed incremental sync for the v2 writer.

`KBWriter.sync(chunks)` reconciles the live corpus to *exactly* the given
chunk set, identifying the delta by chunk id + content hash so that only
new or changed chunks are re-embedded on the following commit. Unchanged
chunks are skipped entirely — that is the whole point: embedding is the
expensive step, and a corpus edit usually touches a handful of chunks.
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
    """Maps each unique text to a stable random unit vector via hash seeding."""

    model_id = "test/mock-deterministic-v0"
    model_revision = "test"
    task_adapter = "retrieval"
    pooling = "native"
    full_dim = 64
    normalize_l2 = True
    release_url = None
    release_sha256 = None
    prompts = {"query": "Query: ", "document": "Document: "}

    def fingerprint(self):
        return {
            "model_id": self.model_id,
            "task_adapter": self.task_adapter,
            "pooling": self.pooling,
            "full_dim": self.full_dim,
        }

    def encode(self, texts, *, prompt):
        out = np.zeros((len(texts), self.full_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.full_dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            out[i] = v
        return out


def _corpus():
    return [
        Chunk(id="post-001#chunk-001", text="The raven flies at dawn and returns at dusk.",
              meta={"source": "post-001", "title": "Ravens"}),
        Chunk(id="post-001#chunk-002", text="Hamming distance counts mismatched bits between two strings.",
              meta={"source": "post-001", "title": "Ravens"}),
        Chunk(id="post-002#chunk-001", text="BM25 ranks documents by length-normalized term frequency.",
              meta={"source": "post-002", "title": "Retrieval"}),
        Chunk(id="post-002#chunk-002", text="Centered SimHash projects vectors to bits while preserving cosine.",
              meta={"source": "post-002", "title": "Retrieval"}),
        Chunk(id="post-003#chunk-001", text="Federalist 10 argues that a large republic mitigates factions.",
              meta={"source": "federalist-10", "title": "Federalist 10"}),
    ]


class CountingEmbedder(DeterministicEmbedder):
    """DeterministicEmbedder that records every text it is asked to encode."""

    def __init__(self):
        self.encoded_texts: list[str] = []
        self.encode_calls = 0

    def encode(self, texts, *, prompt):
        self.encode_calls += 1
        self.encoded_texts.extend(texts)
        return super().encode(texts, prompt=prompt)


def _writer(out: Path, embedder, **kw):
    return KBWriter.create(
        name="test", output_dir=out, embedder=embedder, dim=32, k=4, seed=0, **kw
    )


def test_sync_onto_empty_is_all_adds():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        emb = CountingEmbedder()
        w = _writer(out, emb)
        corpus = _corpus()

        stats = w.sync(corpus)

        assert stats.added == len(corpus)
        assert stats.updated == 0
        assert stats.deleted == 0
        assert stats.unchanged == 0
        assert stats.embedded == len(corpus)

        w.commit()
        kb = KB.open(out / "test.kbi")
        assert kb.live_count == len(corpus)


def test_sync_identical_corpus_embeds_nothing():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        corpus = _corpus()

        w = _writer(out, DeterministicEmbedder())
        w.add_chunks(corpus)
        w.commit()

        emb = CountingEmbedder()
        w2 = KBWriter.open(name="test", output_dir=out, embedder=emb)
        stats = w2.sync(corpus)

        assert stats.unchanged == len(corpus)
        assert stats.added == 0
        assert stats.updated == 0
        assert stats.deleted == 0
        assert stats.embedded == 0

        w2.commit()  # no-op commit
        assert emb.encode_calls == 0
        assert emb.encoded_texts == []

        kb = KB.open(out / "test.kbi")
        assert kb.live_count == len(corpus)


def test_sync_changed_chunk_reembeds_only_that_one():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        corpus = _corpus()

        w = _writer(out, DeterministicEmbedder())
        w.add_chunks(corpus)
        w.commit()

        # Same id, new text => "changed"
        edited = list(corpus)
        new_text = "BM25 ranks documents by saturated, length-normalized term frequency."
        edited[2] = Chunk(id=corpus[2].id, text=new_text, meta=corpus[2].meta)

        emb = CountingEmbedder()
        w2 = KBWriter.open(name="test", output_dir=out, embedder=emb)
        stats = w2.sync(edited)

        assert stats.updated == 1
        assert stats.added == 0
        assert stats.deleted == 0
        assert stats.unchanged == len(corpus) - 1
        assert stats.embedded == 1

        w2.commit()
        assert emb.encoded_texts == [new_text], emb.encoded_texts

        kb = KB.open(out / "test.kbi")
        assert kb.live_count == len(corpus)  # one tombstoned, one re-added
        hits = kb.search(new_text, embedder=DeterministicEmbedder(), k=3)
        assert hits[0].chunk_id == corpus[2].id


def test_sync_handles_adds_and_deletes_together():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        corpus = _corpus()

        w = _writer(out, DeterministicEmbedder())
        w.add_chunks(corpus)
        w.commit()

        # Drop the last chunk, introduce a brand new one.
        desired = corpus[:-1] + [
            Chunk(id="post-009#chunk-001",
                  text="HTTP Range requests fetch byte slices lazily from cold storage.",
                  meta={"source": "post-009", "title": "Range"}),
        ]

        emb = CountingEmbedder()
        w2 = KBWriter.open(name="test", output_dir=out, embedder=emb)
        stats = w2.sync(desired)

        assert stats.added == 1
        assert stats.deleted == 1
        assert stats.updated == 0
        assert stats.unchanged == len(corpus) - 1
        assert stats.embedded == 1

        w2.commit()
        assert len(emb.encoded_texts) == 1  # only the new chunk

        kb = KB.open(out / "test.kbi")
        assert kb.live_count == len(corpus)  # -1 +1
        ids = {h.chunk_id for h in kb.search("range byte slices", embedder=DeterministicEmbedder(), k=5)}
        assert "post-009#chunk-001" in ids
        gone = {h.chunk_id for h in kb.search(corpus[-1].text, embedder=DeterministicEmbedder(), k=5)}
        assert corpus[-1].id not in gone
