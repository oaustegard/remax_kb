"""remex codec (multi-bit Lloyd-Max) round-trip on the v1 .kb path.

Pure-format mechanics with a synthetic stub embedder — no torch, no downloads:
  (a) manifest records the remex kind + bits, bytes_per_row = dim*bits//8
  (b) pack is deterministic (same bytes on re-pack)
  (c) self-retrieval lands each chunk on itself at rank 1 (high fidelity)
  (d) remex does not center (stored mean is all zeros)
  (e) the codec refuses a non-L2-normalizing embedder
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from remax_kb import KB, pack
from remax_kb.manifest import Manifest, REMEX_KIND
from remax_kb.pack import Chunk

pytest.importorskip("remax")
pytest.importorskip("remex")

FULL_DIM = 64
DIM = 32
SEED = 7


class TinyEmbedder:
    """Deterministic-from-text unit vectors, prompt-agnostic so a chunk's
    query encoding equals its document encoding (well-defined self-retrieval)."""

    model_id = "stub/remex-test"
    model_revision = "0" * 40
    task_adapter = "retrieval"
    pooling = "stub"
    full_dim = FULL_DIM
    normalize_l2 = True
    release_url = "https://example.invalid/stub.onnx"
    release_sha256 = "0" * 64
    prompts = {"query": "", "document": ""}

    def fingerprint(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "task_adapter": self.task_adapter,
                "pooling": self.pooling, "full_dim": self.full_dim}

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
        out = np.empty((len(texts), self.full_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.full_dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) or 1.0)
        return out


class NonNormEmbedder(TinyEmbedder):
    normalize_l2 = False


def _chunks(n: int) -> list[Chunk]:
    rng = np.random.default_rng(0)
    return [Chunk(id=f"c#{i:04d}",
                  text=f"chunk {i} " + " ".join(f"t{int(x)}" for x in rng.integers(0, 9999, 8)),
                  meta={"row": i}) for i in range(n)]


def _pack(tmp_path, bits=4, n=120) -> Path:
    out = tmp_path / f"remex_{bits}.kb"
    pack(_chunks(n), out, embedder=TinyEmbedder(), dim=DIM, codec="remex", bits=bits, seed=SEED)
    return out


def test_manifest_records_remex_codec(tmp_path):
    kb = KB.open(_pack(tmp_path, bits=4))
    b = kb.manifest.binarizer
    assert b.kind == REMEX_KIND
    assert b.bits == 4
    assert kb.manifest.bytes_per_row() == DIM * 4 // 8  # honest bit-packed size


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_roundtrip_all_bit_widths(tmp_path, bits):
    kb = KB.open(_pack(tmp_path, bits=bits))
    assert len(kb) == 120
    hits = kb.search(kb.chunks[3]["text"], embedder=TinyEmbedder(), k=3)
    assert hits and isinstance(hits[0][0], int)


def test_pack_is_deterministic(tmp_path):
    a = _pack(tmp_path / "a", bits=4)
    b = _pack(tmp_path / "b", bits=4)
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        assert za.read("vectors.bin") == zb.read("vectors.bin")


def test_self_retrieval_rank1(tmp_path):
    kb = KB.open(_pack(tmp_path, bits=4))
    emb = TinyEmbedder()
    hit1 = 0
    sample = list(range(0, 120, 5))
    for i in sample:
        top = kb.search(kb.chunks[i]["text"], embedder=emb, k=1)[0][1]["id"]
        hit1 += (top == kb.chunks[i]["id"])
    # 4-bit Lloyd-Max keeps a cos=1 self-match well clear of random ~0 others.
    assert hit1 / len(sample) >= 0.95


def test_remex_does_not_center(tmp_path):
    kb = KB.open(_pack(tmp_path, bits=4))
    assert np.allclose(kb.manifest.binarizer.mean_vector, 0.0)


def test_refuses_non_normalizing_embedder(tmp_path):
    with pytest.raises(ValueError, match="L2-normalizing"):
        pack(_chunks(8), tmp_path / "x.kb", embedder=NonNormEmbedder(),
             dim=DIM, codec="remex", bits=4, seed=SEED)
