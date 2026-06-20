"""v1 .kb → v2 .kbi/.kbc migration tests.

The central guarantee: migration reuses v1's dense codes verbatim (no
re-embedding), so the v2 ``vectors.bin`` is byte-identical to the v1
``vectors.bin``, and the carried-over mean / (dim, k, seed) / embedder
fingerprint all match. BM25 + the chunk store are built fresh.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("remax")
pytest.importorskip("bm25s")

from remax_kb import detect_format, migrate_v1_to_v2
from remax_kb.pack import Chunk, pack
from remax_kb.read import KB as KBv1
from remax_kb.read_v2 import KB as KBv2


FULL_DIM = 64


class DeterministicEmbedder:
    model_id = "test/mock-deterministic-v0"
    model_revision = "rev-7"
    task_adapter = "retrieval"
    pooling = "native"
    full_dim = FULL_DIM
    normalize_l2 = True
    release_url = None
    release_sha256 = None
    prompts = {"query": "Query: ", "document": "Document: "}

    def fingerprint(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_adapter": self.task_adapter,
            "pooling": self.pooling,
            "full_dim": self.full_dim,
        }

    def encode(self, texts, *, prompt):
        out = np.zeros((len(texts), self.full_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:4], "little")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.full_dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            out[i] = v
        return out


def _corpus() -> list[Chunk]:
    return [
        Chunk(id="post-001#chunk-001", text="The raven flies at dawn and returns at dusk.",
              meta={"source": "post-001"}),
        Chunk(id="post-001#chunk-002", text="Hamming distance counts mismatched bits between two strings.",
              meta={"source": "post-001"}),
        Chunk(id="post-002#chunk-001", text="BM25 ranks documents by length-normalized term frequency.",
              meta={"source": "post-002"}),
        Chunk(id="post-002#chunk-002", text="Centered SimHash projects vectors to bits while preserving cosine.",
              meta={"source": "post-002"}),
        Chunk(id="post-003#chunk-001", text="Federalist 10 argues that a large republic mitigates factions.",
              meta={"source": "federalist-10"}),
    ]


def _build_v1(tmp_path: Path) -> Path:
    out = tmp_path / "src.kb"
    pack(
        _corpus(),
        out,
        embedder=DeterministicEmbedder(),
        dim=32, k=4, seed=42,
        source_description="legacy v1 corpus",
    )
    return out


def _zip_entry(path: Path, name: str) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        return zf.read(name)


def test_detect_format(tmp_path: Path):
    v1 = _build_v1(tmp_path)
    assert detect_format(v1) == "1"
    kbi, _ = migrate_v1_to_v2(v1, tmp_path / "out")
    assert detect_format(kbi) == "2"


def test_migrate_preserves_codes_and_identity(tmp_path: Path):
    v1_path = _build_v1(tmp_path)
    v1 = KBv1.open(v1_path)

    kbi, kbc = migrate_v1_to_v2(v1_path, tmp_path / "out", name="migrated")
    assert kbi.is_file()
    assert kbc.is_dir()
    assert (kbc / "shard-0000.bin").is_file()

    # Dense codes are reused verbatim — byte-identical vectors.bin.
    assert _zip_entry(kbi, "vectors.bin") == _zip_entry(v1_path, "vectors.bin")

    kb = KBv2.open(kbi)
    m = kb.manifest
    assert m["spec_version"] == "2"
    assert m["kind"] == "split-index"
    assert m["chunks"]["live_count"] == len(_corpus())
    assert m["chunks"]["total_rows"] == len(_corpus())

    # Carried-over identity.
    assert m["embedder"]["model_id"] == DeterministicEmbedder.model_id
    assert m["embedder"]["full_dim"] == FULL_DIM
    assert m["binarizer"]["dim"] == v1.manifest.binarizer.dim
    assert m["binarizer"]["k"] == v1.manifest.binarizer.k
    assert m["binarizer"]["seed"] == v1.manifest.binarizer.seed
    assert m["source"] == "legacy v1 corpus"

    # Mean vector survives the round trip.
    import base64
    v2_mean = np.frombuffer(base64.b64decode(m["binarizer"]["mean_vector_b64"]), dtype="<f4")
    np.testing.assert_allclose(v2_mean, np.asarray(v1.manifest.binarizer.mean_vector))

    # Fresh BM25 index present.
    assert "lexical" in m
    assert m["lexical"]["kind"] == "bm25s"


def test_migrated_kb_is_queryable(tmp_path: Path):
    v1_path = _build_v1(tmp_path)
    kbi, _ = migrate_v1_to_v2(v1_path, tmp_path / "out")

    kb = KBv2.open(kbi)
    embedder = DeterministicEmbedder()

    target = _corpus()[2]
    hits = kb.search_and_fetch(target.text, embedder=embedder, k=3)
    assert hits[0].chunk_id == target.id
    assert hits[0].text == target.text
    assert hits[0].verified is True

    # Lexical retrieval works on a distinctive term.
    lex = kb.search(target.text, embedder=embedder, k=3)
    assert any(h.chunk_id == target.id for h in lex)


def test_migrate_refuses_existing_output(tmp_path: Path):
    v1_path = _build_v1(tmp_path)
    out = tmp_path / "out"
    migrate_v1_to_v2(v1_path, out, name="x")
    with pytest.raises(FileExistsError):
        migrate_v1_to_v2(v1_path, out, name="x")
