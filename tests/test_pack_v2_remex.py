"""v2 split-index pack + read with the remex (multi-bit Lloyd-Max) codec.

Mirrors test_pack_v2 but with codec='remex': structure/manifest, dense + BM25
retrieval, mutation (append, delete), no rotation sidecar, no centering, and
re-open determinism. Deterministic mock embedder — no network/ONNX.
"""
from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from remax_kb.pack import Chunk
from remax_kb.pack_v2 import KBWriter
from remax_kb.read_v2 import KB

pytest.importorskip("remex")


class DeterministicEmbedder:
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
        return {"model_id": self.model_id, "task_adapter": self.task_adapter,
                "pooling": self.pooling, "full_dim": self.full_dim}

    def encode(self, texts, *, prompt):
        out = np.zeros((len(texts), self.full_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
            v = np.random.default_rng(seed).standard_normal(self.full_dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-12)
        return out


def _corpus():
    return [
        Chunk(id="post-001#chunk-001", text="The raven flies at dawn and returns at dusk.", meta={"source": "post-001"}),
        Chunk(id="post-001#chunk-002", text="Hamming distance counts mismatched bits between two strings.", meta={"source": "post-001"}),
        Chunk(id="post-002#chunk-001", text="BM25 ranks documents by length-normalized term frequency.", meta={"source": "post-002"}),
        Chunk(id="post-002#chunk-002", text="Centered SimHash projects vectors to bits while preserving cosine.", meta={"source": "post-002"}),
        Chunk(id="post-003#chunk-001", text="Federalist 10 argues that a large republic mitigates factions.", meta={"source": "federalist-10"}),
    ]


def test_remex_v2_roundtrip_and_manifest():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        emb = DeterministicEmbedder()
        w = KBWriter.create(name="t", output_dir=out, embedder=emb, dim=32, seed=42, codec="remex", bits=4)
        w.add_chunks(_corpus()); w.commit()

        kb = KB.open(out / "t.kbi")
        b = kb.manifest["binarizer"]
        assert b["kind"] == "remex-lloyd-max" and b["bits"] == 4
        # row width is dim*bits/8, and no rotation sidecar ships
        with zipfile.ZipFile(out / "t.kbi") as zf:
            names = set(zf.namelist())
            assert len(zf.read("vectors.bin")) == len(_corpus()) * (32 * 4 // 8)
            assert not any(n.startswith("binarizer/rotations") for n in names)
        # remex does not center: stored mean is zeros
        import base64
        mean = np.frombuffer(base64.b64decode(b["mean_vector_b64"]), dtype="<f4")
        assert np.allclose(mean, 0.0)

        target = _corpus()[2]
        hits = kb.search(target.text, embedder=emb, k=3)
        assert hits and hits[0].chunk_id == target.id
        # BM25 still fuses
        lex = kb.search("federalist", embedder=emb, k=3)
        assert any(h.chunk_id == "post-003#chunk-001" for h in lex[:2])


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_remex_v2_bit_widths(bits):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        emb = DeterministicEmbedder()
        w = KBWriter.create(name="t", output_dir=out, embedder=emb, dim=32, seed=0, codec="remex", bits=bits)
        w.add_chunks(_corpus()); w.commit()
        kb = KB.open(out / "t.kbi")
        t = _corpus()[0]
        assert kb.search(t.text, embedder=emb, k=1)[0].chunk_id == t.id


def test_remex_v2_append_and_delete():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        emb = DeterministicEmbedder()
        w = KBWriter.create(name="t", output_dir=out, embedder=emb, dim=32, seed=0, codec="remex", bits=4)
        w.add_chunks(_corpus()[:3]); w.commit()

        # re-open restores codec from manifest; mutate
        w2 = KBWriter.open(name="t", output_dir=out, embedder=emb)
        assert w2._codec == "remex" and w2._bits == 4
        w2.add_chunks([Chunk(id="post-004#chunk-001", text="Range requests fetch byte slices over HTTP.", meta={})])
        w2.commit()
        kb = KB.open(out / "t.kbi")
        assert kb.live_count == 4
        assert "post-004#chunk-001" in [h.chunk_id for h in kb.search("range requests http", embedder=emb, k=3)[:2]]

        w3 = KBWriter.open(name="t", output_dir=out, embedder=emb)
        w3.delete_chunks(["post-001#chunk-001"]); w3.commit()
        kb = KB.open(out / "t.kbi")
        assert kb.live_count == 3
        assert "post-001#chunk-001" not in [h.chunk_id for h in kb.search("raven", embedder=emb, k=5)]


def test_remex_v2_refuses_non_normalizing_embedder():
    class NonNorm(DeterministicEmbedder):
        normalize_l2 = False
    with tempfile.TemporaryDirectory() as d:
        w = KBWriter.create(name="t", output_dir=Path(d), embedder=NonNorm(), dim=32, seed=0, codec="remex", bits=4)
        w.add_chunks(_corpus())
        with pytest.raises(ValueError, match="L2-normalizing"):
            w.commit()
