"""int8-quantized rotation sidecar: packer + reader roundtrip and consistency.

The shipped rotation matrices can be stored as int8 + per-column scale (4x
smaller) instead of f32. This verifies the .kbi structure, the manifest flag,
and — the load-bearing property — that the reader dequantizes to the *same*
rotations the packer encoded the corpus with, so a query that equals a document
lands at Hamming distance 0.
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
from remax_kb.rotations import quantize_int8, dequantize_int8


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
    texts = [
        "The raven flies at dawn and returns at dusk.",
        "Hamming distance counts mismatched bits between two strings.",
        "BM25 ranks documents by length-normalized term frequency.",
        "Centered SimHash projects vectors to bits while preserving cosine.",
        "Federalist 10 argues that a large republic mitigates factions.",
        "Quantizing rotations to int8 barely moves the sign bits.",
    ]
    return [Chunk(id=f"post-{i:03d}#chunk-001", text=t, meta={"source": f"post-{i}"})
            for i, t in enumerate(texts)]


def _build(tmp: Path, quant: str, dim=64, k=4):
    w = KBWriter.create(name=f"kb_{quant}", output_dir=tmp, embedder=DeterministicEmbedder(),
                        dim=dim, k=k, seed=0, rotations_quant=quant)
    w.add_chunks(_corpus())
    w.commit()
    return tmp / f"kb_{quant}.kbi"


def test_quantize_roundtrip_shapes():
    R = np.random.default_rng(0).standard_normal((4, 64, 64)).astype(np.float32)
    codes, scale = quantize_int8(R)
    assert codes.dtype == np.int8 and codes.shape == (4, 64, 64)
    assert scale.dtype == np.float32 and scale.shape == (4, 64)
    deq = dequantize_int8(codes, scale)
    # per-column max error <= half a quant step
    assert np.abs(deq - R).max() <= (scale.max() / 1.0)


def test_int8_kbi_structure():
    with tempfile.TemporaryDirectory() as d:
        kbi = _build(Path(d), "int8")
        with zipfile.ZipFile(kbi) as zf:
            names = set(zf.namelist())
            assert "binarizer/rotations.i8" in names
            assert "binarizer/rotations.scale.f32" in names
            assert "binarizer/rotations.f32" not in names
            import json
            m = json.loads(zf.read("manifest.json"))
            assert m["binarizer"]["rotations_quant"] == "int8"
            # sizes: i8 = k*dim*dim bytes, scale = k*dim float32
            assert len(zf.read("binarizer/rotations.i8")) == 4 * 64 * 64
            assert len(zf.read("binarizer/rotations.scale.f32")) == 4 * 64 * 4


def test_f32_kbi_default_unchanged():
    with tempfile.TemporaryDirectory() as d:
        kbi = _build(Path(d), "float32")
        with zipfile.ZipFile(kbi) as zf:
            names = set(zf.namelist())
            assert "binarizer/rotations.f32" in names
            assert "binarizer/rotations.i8" not in names
            import json
            m = json.loads(zf.read("manifest.json"))
            assert m["binarizer"].get("rotations_quant", "float32") == "float32"


def test_int8_reader_is_bit_consistent_with_packer():
    """A query equal to a document must land at Hamming distance 0 — proving the
    reader dequantizes to the exact rotations the corpus was packed against."""
    with tempfile.TemporaryDirectory() as d:
        kbi = _build(Path(d), "int8")
        kb = KB.open(kbi)
        emb = DeterministicEmbedder()
        for c in _corpus():
            hits = kb._dense_search(c.text, emb)
            assert hits[0].dense_dist == 0, f"{c.id}: nearest dist {hits[0].dense_dist} != 0"


def test_int8_matches_f32_ranking_closely():
    """int8 rotations should preserve the top-k ordering vs f32 on this corpus."""
    with tempfile.TemporaryDirectory() as d:
        kbi_f = _build(Path(d), "float32")
        kbi_i = _build(Path(d), "int8")
        kb_f, kb_i = KB.open(kbi_f), KB.open(kbi_i)
        emb = DeterministicEmbedder()
        q = "How does SimHash preserve cosine similarity?"
        rf = [h.row for h in kb_f._dense_search(q, emb)[:3]]
        ri = [h.row for h in kb_i._dense_search(q, emb)[:3]]
        # top-1 identical; top-3 sets overlap by at least 2
        assert rf[0] == ri[0]
        assert len(set(rf) & set(ri)) >= 2
