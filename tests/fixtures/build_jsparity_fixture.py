#!/usr/bin/env python3
"""Build the committed cross-reader parity fixture.

Produces ``tests/fixtures/jsparity/`` — a real ``.kbi`` + ``.kbc/`` written by
the real :class:`remax_kb.pack_v2.KBWriter`, small enough to commit, plus a
``queries.json`` carrying the query embeddings so ``js/kb-reader.js`` can be
driven from Node without an embedder.

Deterministic: the embedder maps text → a hash-seeded unit vector, so
re-running this reproduces byte-identical artifacts. Re-run it (and commit the
result) only when the on-disk format itself changes.

    python3 tests/fixtures/build_jsparity_fixture.py

The corpus is developer-doc shaped on purpose: identifier tokens
(``response_model``, ``get_user``, ``item_id``) are exactly what the BM25
query tokenizer used to shred, so the same fixture exercises both the lexical
arm and the dense arm.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent / "jsparity"
NAME = "jsparity"
DIM = 32
K = 4
SEED = 42
FULL_DIM = 64


class DeterministicEmbedder:
    """Hash-seeded unit vectors. No model, no network, stable across runs."""

    model_id = "test/jsparity-deterministic-v0"
    model_revision = "test"
    task_adapter = "retrieval"
    pooling = "native"
    full_dim = FULL_DIM
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
            rng = np.random.default_rng(int.from_bytes(h[:4], "little"))
            v = rng.standard_normal(self.full_dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            out[i] = v
        return out


CORPUS = [
    ("doc-fastapi#001",
     "Use response_model to declare the shape of the response. FastAPI will "
     "filter the returned data to the declared response_model."),
    ("doc-fastapi#002",
     "def get_user(user_id: int): the path parameter item_id is declared in "
     "the path and passed to the function."),
    ("doc-fastapi#003",
     "Raise HTTPException with a status_code of 404 when the item is not found."),
    ("doc-unicode#001",
     "Un café serveur naïve résumé: Unicode identifiers survive tokenization "
     "intact when the pattern is Unicode aware."),
    ("doc-hamming#001",
     "Hamming distance counts mismatched bits between two packed binary codes."),
    ("doc-bm25#001",
     "BM25 ranks documents by length normalized term frequency and inverse "
     "document frequency."),
    ("doc-simhash#001",
     "Centered SimHash projects vectors onto random planes and keeps only the "
     "sign of each projection."),
    ("doc-retired#001",
     "This chunk is deleted before commit so the fixture carries a tombstone "
     "row and exercises the live row mapping."),
]

# Queries the gate drives both readers with. Identifier-shaped ones are the
# regression cases for the BM25 tokenizer fix; the prose ones exercise fusion.
QUERIES = [
    "response_model",
    "get_user",
    "item_id path parameter",
    "status_code 404 HTTPException",
    "café résumé",
    "hamming distance between packed codes",
    "how does SimHash keep the sign of a projection",
]


def build() -> None:
    from remax_kb.pack import Chunk
    from remax_kb.pack_v2 import KBWriter

    if FIXTURE_DIR.exists():
        shutil.rmtree(FIXTURE_DIR)
    FIXTURE_DIR.mkdir(parents=True)

    embedder = DeterministicEmbedder()
    # projection="haar" is PINNED, not inherited. haar stopped being the
    # writer's default in 2026-08 (srht ships no sidecar), and this fixture
    # exists precisely to keep the shipped-rotation path under test: the
    # cross-reader gate reads binarizer/rotations.f32 out of it to measure the
    # sign margin, and the open-validation gate strips the sidecar to check the
    # JS reader's refusal. Rebuilding it without this argument would silently
    # delete both. The seed-only path is covered by the srht fixture
    # gate_cross_reader builds at runtime.
    writer = KBWriter.create(
        name=NAME, output_dir=FIXTURE_DIR, embedder=embedder,
        dim=DIM, k=K, seed=SEED, projection="haar",
    )
    writer.add_chunks([
        Chunk(id=cid, text=text, meta={"source": cid.split("#")[0]})
        for cid, text in CORPUS
    ])
    writer.commit()

    # Tombstone one row so the fixture exercises the live→absolute row mapping
    # both readers build (JS `_rowOfLive`, Python `row_of_live`).
    writer2 = KBWriter.open(name=NAME, output_dir=FIXTURE_DIR, embedder=embedder)
    writer2.delete_chunks(["doc-retired#001"])
    writer2.commit()

    qvecs = embedder.encode(QUERIES, prompt="query")
    (FIXTURE_DIR / "queries.json").write_text(json.dumps({
        "note": "generated by tests/fixtures/build_jsparity_fixture.py",
        "embedder_fingerprint": embedder.fingerprint(),
        "queries": [
            {"text": q, "embedding": [float(x) for x in qvecs[i]]}
            for i, q in enumerate(QUERIES)
        ],
    }, indent=1) + "\n")

    print(f"wrote {FIXTURE_DIR}")
    for p in sorted(FIXTURE_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(FIXTURE_DIR)}  {p.stat().st_size} B")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    build()
