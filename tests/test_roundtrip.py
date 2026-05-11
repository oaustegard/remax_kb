"""Roundtrip tests: pack → read with a synthetic stub embedder.

Validates that the pure-format mechanics work end-to-end without ever
loading torch or downloading the jina model:
  (a) packed codes match an independent re-encode of the same vectors
  (b) self-distance is zero (every chunk is its own nearest neighbour)
  (c) build_hash verifies on read
  (d) chunk_count and the manifest fingerprint round-trip
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from remax_kb import KB, pack
from remax_kb.manifest import BINARIZER_KIND, SPEC_VERSION
from remax_kb.pack import Chunk

# remax is required even for the roundtrip path (StackedSignBitQuantizer).
remax = pytest.importorskip("remax")


FULL_DIM = 64        # tiny — fast tests, no downloads
DIM = 32             # truncation target
K = 4                # stack count
SEED = 1234
N_CHUNKS = 100


class StubEmbedder:
    """Deterministic-from-text random embedder. Pure numpy, no model."""

    model_id = "stub/test-embedder"
    model_revision = "0000000000000000000000000000000000000000"
    task_adapter = "retrieval"
    pooling = "stub"
    full_dim = FULL_DIM
    normalize_l2 = True
    release_url = "https://example.invalid/stub.onnx"
    release_sha256 = "0" * 64
    prompts = {"query": "Q: ", "document": "D: "}

    def fingerprint(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_adapter": self.task_adapter,
            "pooling": self.pooling,
            "full_dim": self.full_dim,
        }

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
        prefix = self.prompts[prompt]
        out = np.empty((len(texts), self.full_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256((prefix + t).encode("utf-8")).digest()
            # Seed a per-text RNG; produces a deterministic vector.
            seed = int.from_bytes(digest[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.full_dim).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            out[i] = v
        return out


def _synthetic_chunks(n: int) -> list[Chunk]:
    rng = np.random.default_rng(0)
    chunks = []
    for i in range(n):
        # Distinct text per chunk so the stub embedder produces distinct vectors.
        text = f"chunk number {i} with random tokens " + " ".join(
            f"tok{int(t)}" for t in rng.integers(0, 10_000, size=10)
        )
        chunks.append(
            Chunk(id=f"synthetic#{i:04d}", text=text, meta={"row": i})
        )
    return chunks


@pytest.fixture
def packed_kb(tmp_path: Path) -> Path:
    out = tmp_path / "synthetic.kb"
    pack(
        _synthetic_chunks(N_CHUNKS),
        out,
        embedder=StubEmbedder(),
        dim=DIM,
        k=K,
        seed=SEED,
        source_description="synthetic stub-embedder roundtrip",
    )
    return out


def test_kb_is_valid_zip_with_required_entries(packed_kb: Path):
    with zipfile.ZipFile(packed_kb, "r") as zf:
        names = set(zf.namelist())
    assert {"manifest.json", "vectors.bin", "chunks.jsonl"} <= names


def test_open_validates_and_loads(packed_kb: Path):
    kb = KB.open(packed_kb)
    assert len(kb) == N_CHUNKS
    m = kb.manifest
    assert m.spec_version == SPEC_VERSION
    assert m.binarizer.kind == BINARIZER_KIND
    assert m.binarizer.dim == DIM
    assert m.binarizer.k == K
    assert m.binarizer.seed == SEED
    assert m.embedder.full_dim == FULL_DIM
    assert m.embedder.model_id == StubEmbedder.model_id
    assert m.binarizer.mean_vector.shape == (FULL_DIM,)


def test_codes_match_independent_reencode(packed_kb: Path):
    """Re-embed every chunk via the stub, center, truncate, encode — bytes
    must equal what's in vectors.bin."""
    from remax import StackedSignBitQuantizer

    kb = KB.open(packed_kb)
    emb = StubEmbedder()
    chunks = kb.chunks
    vecs = emb.encode([c["text"] for c in chunks], prompt="document")
    centered = vecs - kb.manifest.binarizer.mean_vector
    truncated = np.ascontiguousarray(centered[:, :DIM])
    q = StackedSignBitQuantizer(d=DIM, k=K, seed=SEED)
    expected = q.encode(truncated)
    assert kb.codes.shape == expected.shape
    np.testing.assert_array_equal(kb.codes, expected)


def test_self_search_returns_zero_distance(packed_kb: Path):
    """Querying with each chunk's own text should return that chunk first
    with hamming distance zero."""
    kb = KB.open(packed_kb)
    emb = StubEmbedder()
    # Sample a few rows; running all 100 is overkill.
    for i in (0, 7, 42, 99):
        chunk = kb.chunks[i]
        # Warning: the document prompt was used at pack time. Reusing it
        # here gives an exact code match; using the query prompt would
        # not. We're testing the binarizer roundtrip, not the asymmetry.
        emb_with_doc_prompt = StubEmbedder()
        emb_with_doc_prompt.prompts = {"query": "D: ", "document": "D: "}
        hits = kb.search(chunk["text"], embedder=emb_with_doc_prompt, k=3)
        top_dist, top_chunk = hits[0]
        assert top_chunk["id"] == chunk["id"], (
            f"row {i}: expected own id, got {top_chunk['id']} at hamming={top_dist}"
        )
        assert top_dist == 0, f"row {i}: expected hamming=0, got {top_dist}"


def test_build_hash_verifies_on_read(packed_kb: Path, tmp_path: Path):
    """Tamper with vectors.bin and re-zip; KB.open must refuse."""
    bad = tmp_path / "tampered.kb"
    with zipfile.ZipFile(packed_kb, "r") as src, zipfile.ZipFile(
        bad, "w", compression=zipfile.ZIP_STORED
    ) as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "vectors.bin":
                # Flip one bit in the first byte.
                data = bytes([data[0] ^ 0x01]) + data[1:]
            dst.writestr(name, data)

    with pytest.raises(ValueError, match="build_hash"):
        KB.open(bad)


def test_fingerprint_mismatch_refused(packed_kb: Path):
    """A different model_id on the embedder must cause search to refuse."""
    kb = KB.open(packed_kb)

    class WrongModel(StubEmbedder):
        model_id = "stub/different-model"

    with pytest.raises(ValueError, match="model_id"):
        kb.search("anything", embedder=WrongModel(), k=1)


def test_manifest_round_trip_is_byte_stable():
    """to_json then from_json then to_json must be idempotent."""
    from remax_kb.manifest import (
        Binarizer,
        CorpusInfo,
        Embedder,
        Manifest,
        Prompts,
    )

    rng = np.random.default_rng(7)
    mean = rng.standard_normal(FULL_DIM).astype(np.float32)
    m1 = Manifest(
        spec_version=SPEC_VERSION,
        embedder=Embedder(
            model_id="stub/test-embedder",
            model_revision="abc",
            release_url="https://example.invalid/x.onnx",
            release_sha256="0" * 64,
            task_adapter="retrieval",
            pooling="stub",
            normalize_l2=True,
            full_dim=FULL_DIM,
        ),
        prompts=Prompts(query="Q: ", document="D: "),
        binarizer=Binarizer.from_mean(
            remax_version="0.0.0",
            dim=DIM,
            k=K,
            seed=SEED,
            mean_vector=mean,
        ),
        corpus=CorpusInfo(
            chunk_count=1,
            build_hash="0" * 64,
            built_at="2026-05-11T00:00:00+00:00",
            source="",
        ),
    )
    j1 = m1.to_json()
    m2 = Manifest.from_json(j1)
    j2 = m2.to_json()
    assert j1 == j2
    np.testing.assert_array_equal(m1.binarizer.mean_vector, m2.binarizer.mean_vector)
