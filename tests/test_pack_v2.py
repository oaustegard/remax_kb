"""End-to-end test for v2 split-index pack + read.

Uses a deterministic mock embedder (random projections seeded by text
hash) so we don't depend on Gemini / ONNX network availability.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

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
            # Seed from (prompt, text) so query/doc encode similarly for same text
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


def test_pack_and_read_roundtrip():
    """Pack a small corpus and verify the .kbi structure + retrieval."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        embedder = DeterministicEmbedder()

        writer = KBWriter.create(
            name="test",
            output_dir=out,
            embedder=embedder,
            dim=32, k=4, seed=42,
        )
        chunks = _corpus()
        writer.add_chunks(chunks)
        writer.commit()

        # .kbi exists, .kbc/ exists
        kbi = out / "test.kbi"
        kbc = out / "test.kbc"
        assert kbi.is_file()
        assert kbc.is_dir()
        shards = list(kbc.glob("shard-*.bin"))
        assert len(shards) == 1, f"expected one shard, got {shards}"

        # Open and verify metadata
        kb = KB.open(kbi)
        assert kb.manifest["spec_version"] == "2"
        assert kb.manifest["kind"] == "split-index"
        assert kb.manifest["chunks"]["live_count"] == len(chunks)
        assert kb.manifest["chunks"]["total_rows"] == len(chunks)
        assert "lexical" in kb.manifest
        assert kb.manifest["lexical"]["kind"] == "bm25s"

        # Dense search for an exact-text query — embedding is deterministic
        # given identical input, so the chunk whose text matches should be
        # the closest hit.
        target = chunks[2]
        hits = kb.search(target.text, embedder=embedder, k=3)
        assert len(hits) > 0
        assert hits[0].chunk_id == target.id, (
            f"expected {target.id} as top hit; got {[h.chunk_id for h in hits]}"
        )

        # Fetch fills text + meta and verifies sha256
        fetched = kb.fetch(hits[:2])
        for hit in fetched:
            assert hit.text is not None
            assert hit.verified is True

        # BM25 lexical hit on a distinctive term
        lex_hits = kb.search("federalist", embedder=embedder, k=3)
        assert any(h.chunk_id == "post-003#chunk-001" for h in lex_hits[:2])


def test_append_via_open():
    """After commit, re-open and add more chunks."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        embedder = DeterministicEmbedder()
        writer = KBWriter.create(name="test", output_dir=out, embedder=embedder, dim=32, k=4, seed=0)
        writer.add_chunks(_corpus()[:3])
        writer.commit()

        # Reopen
        writer2 = KBWriter.open(name="test", output_dir=out, embedder=embedder)
        more = [
            Chunk(id="post-004#chunk-001", text="Range requests fetch byte slices over HTTP.",
                  meta={"source": "post-004", "title": "Range"}),
        ]
        writer2.add_chunks(more)
        writer2.commit()

        kb = KB.open(out / "test.kbi")
        assert kb.live_count == 4
        hits = kb.search("range requests http", embedder=embedder, k=3)
        ids = [h.chunk_id for h in hits[:2]]
        assert "post-004#chunk-001" in ids, f"new chunk not retrieved: {ids}"


def test_delete_tombstone():
    """Deletes apply a tombstone; live_count drops."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        embedder = DeterministicEmbedder()
        writer = KBWriter.create(name="test", output_dir=out, embedder=embedder, dim=32, k=4, seed=0)
        writer.add_chunks(_corpus())
        writer.commit()

        writer = KBWriter.open(name="test", output_dir=out, embedder=embedder)
        writer.delete_chunks(["post-001#chunk-001"])
        writer.commit()

        kb = KB.open(out / "test.kbi")
        assert kb.live_count == 4
        assert kb.manifest["chunks"]["total_rows"] == 5  # tombstoned row still counted
        hits = kb.search("raven", embedder=embedder, k=5)
        chunk_ids = [h.chunk_id for h in hits]
        assert "post-001#chunk-001" not in chunk_ids


def test_dense_only_no_bm25():
    """A .kbi without bm25/ entries should still be queryable, dense-only."""
    # Hard to test the missing-bm25/ path without a custom writer hook.
    # Use the file-level approach: pack normally, then surgically drop bm25/.
    import io, zipfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        embedder = DeterministicEmbedder()
        writer = KBWriter.create(name="test", output_dir=out, embedder=embedder, dim=32, k=4, seed=0)
        writer.add_chunks(_corpus())
        writer.commit()

        kbi = out / "test.kbi"
        with zipfile.ZipFile(kbi, "r") as zin:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zout:
                for item in zin.namelist():
                    if item.startswith("bm25/"):
                        continue
                    zout.writestr(item, zin.read(item))
        kbi.write_bytes(buf.getvalue())

        kb = KB.open(kbi)
        hits = kb.search("hamming distance bits", embedder=embedder, k=3)
        assert len(hits) > 0
        # All should have None bm25_score
        assert all(h.bm25_score is None for h in hits)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-x"]))
