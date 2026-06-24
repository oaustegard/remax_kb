"""Seed-only Rademacher projection: packer + reader, no shipped rotation.

A 'rademacher' .kbi carries no rotation sidecar — both packer and reader
regenerate the ±1 planes from (dim, k, seed) via splitmix64. This checks the
.kbi structure, the manifest flag, the load-bearing bit-consistency property
(query == document → Hamming 0, proving the reader rebuilds the same planes the
corpus was packed against), and determinism of the generator.
"""
from __future__ import annotations

import hashlib
import tempfile
import zipfile
import json
from pathlib import Path

import numpy as np
import pytest

from remax_kb.pack import Chunk
from remax_kb.pack_v2 import KBWriter
from remax_kb.read_v2 import KB
from remax_kb.projection import rademacher_planes


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
        "Rademacher planes need no shipped rotation matrix.",
        "Federalist 10 argues a large republic mitigates factions.",
    ]
    return [Chunk(id=f"post-{i:03d}#chunk-001", text=t, meta={"source": f"post-{i}"})
            for i, t in enumerate(texts)]


def _build(tmp: Path, projection: str, dim=64, k=4):
    w = KBWriter.create(name=f"kb_{projection}", output_dir=tmp,
                        embedder=DeterministicEmbedder(), dim=dim, k=k, seed=0,
                        projection=projection)
    w.add_chunks(_corpus())
    w.commit()
    return tmp / f"kb_{projection}.kbi"


def test_rademacher_planes_are_pm1_and_deterministic():
    a = rademacher_planes(64, 4, 0)
    b = rademacher_planes(64, 4, 0)
    assert a.shape == (4, 64, 64) and a.dtype == np.float32
    assert set(np.unique(a).tolist()) <= {-1.0, 1.0}
    np.testing.assert_array_equal(a, b)                       # deterministic
    assert not np.array_equal(a, rademacher_planes(64, 4, 1))  # seed matters


def test_rademacher_kbi_ships_no_rotation_sidecar():
    with tempfile.TemporaryDirectory() as d:
        kbi = _build(Path(d), "rademacher")
        with zipfile.ZipFile(kbi) as zf:
            names = set(zf.namelist())
            assert not any(n.startswith("binarizer/") for n in names), \
                f"rademacher .kbi should ship no rotation entry, got {names}"
            m = json.loads(zf.read("manifest.json"))
            assert m["binarizer"]["projection"] == "rademacher"
            assert m["binarizer"]["rotations_quant"] == "none"


def test_haar_default_still_ships_rotations():
    with tempfile.TemporaryDirectory() as d:
        kbi = _build(Path(d), "haar")
        with zipfile.ZipFile(kbi) as zf:
            assert "binarizer/rotations.f32" in set(zf.namelist())
            m = json.loads(zf.read("manifest.json"))
            assert m["binarizer"]["projection"] == "haar"


def test_rademacher_reader_is_bit_consistent_with_packer():
    """query == document ⇒ Hamming 0: the reader regenerates the same planes."""
    with tempfile.TemporaryDirectory() as d:
        kb = KB.open(_build(Path(d), "rademacher"))
        emb = DeterministicEmbedder()
        for c in _corpus():
            hits = kb._dense_search(c.text, emb)
            assert hits[0].dense_dist == 0, f"{c.id}: nearest {hits[0].dense_dist} != 0"


def test_rademacher_retrieval_runs_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        kb = KB.open(_build(Path(d), "rademacher"))
        hits = kb.search_and_fetch("How does SimHash preserve cosine?",
                                   embedder=DeterministicEmbedder(), k=3)
        assert len(hits) == 3
        assert all(h.verified for h in hits)
