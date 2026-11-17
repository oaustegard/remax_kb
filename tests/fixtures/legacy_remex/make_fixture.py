"""Regenerate the legacy remex fixtures. Run ONLY against remax_kb 0.5.0
(commit b6f358c) with remex==0.6.0 installed, from that checkout:

    python tests/fixtures/legacy_remex/make_fixture.py OUT_DIR

It writes a v1 ``legacy_v1.kb``, a v2 ``legacy_v2.kbi`` (+ chunk store) and
``expected_rows.npy``: the corpus rows as remex 0.6.0 decoded them. Those
artifacts carry no ``binarizer.rotation`` field and were encoded with Haar,
which is what test_remex_legacy.py checks the current reader still decodes.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from remax_kb import KB as KBv1, pack
from remax_kb.pack import Chunk
from remax_kb.pack_v2 import KBWriter
from remax_kb.read_v2 import KB as KBv2


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


TEXTS = [
    "The raven flies at dawn and returns at dusk.",
    "Hamming distance counts mismatched bits between two strings.",
    "BM25 ranks documents by length-normalized term frequency.",
    "Centered SimHash projects vectors to bits while preserving cosine.",
    "Federalist 10 argues that a large republic mitigates factions.",
    "Range requests fetch byte slices over HTTP.",
]


def chunks():
    return [Chunk(id=f"c{i}", text=t, meta={}) for i, t in enumerate(TEXTS)]


def main(out: Path) -> None:
    import remex
    assert remex.__version__ == "0.6.0", remex.__version__
    out.mkdir(parents=True, exist_ok=True)
    emb = DeterministicEmbedder()
    pack(chunks(), out / "legacy_v1.kb", embedder=emb, dim=32, seed=7,
         codec="remex", bits=4)
    w = KBWriter.create(name="legacy_v2", output_dir=out, embedder=emb, dim=32,
                        seed=7, codec="remex", bits=4)
    w.add_chunks(chunks()); w.commit()
    v1 = KBv1.open(out / "legacy_v1.kb")._remex_doc_matrix()
    v2 = KBv2.open(out / "legacy_v2.kbi")._remex_doc_matrix()
    assert np.array_equal(v1, v2)
    np.save(out / "expected_rows.npy", v1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
