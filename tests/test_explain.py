"""Score provenance on returned hits (``Hit.explain``).

The fused score alone is unattributable: ``fused=0.0163`` says nothing about
which leg retrieved the row or at what rank. These tests pin the provenance
keys that ``KB.search`` now attaches, and the out-of-vocabulary diagnostic
that makes a silently dense-only hybrid search visible.
"""

import hashlib
from typing import ClassVar

import numpy as np
import pytest

from remax_kb.pack import Chunk
from remax_kb.pack_v2 import KBWriter
from remax_kb.read_v2 import KB


class DeterministicEmbedder:
    """Maps each unique text to a stable random unit vector via hash seeding.

    Same stub as ``test_pack_v2`` — no model download, no network.
    """

    model_id = "test/mock-deterministic-v0"
    model_revision = "test"
    task_adapter = "retrieval"
    pooling = "native"
    full_dim = 64
    normalize_l2 = True
    release_url = None
    release_sha256 = None
    prompts: ClassVar[dict] = {"query": "Query: ", "document": "Document: "}

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


CORPUS = [
    ("doc-0", "the migration ran clean against the postgres replica"),
    ("doc-1", "reciprocal rank fusion combines two ranked candidate lists"),
    ("doc-2", "hamming distance over bit-packed vectors is a popcount scan"),
    ("doc-3", "the tokenizer splits identifiers like response_model apart"),
]


@pytest.fixture
def kb(tmp_path):
    emb = DeterministicEmbedder()
    w = KBWriter.create(
        name="explain", output_dir=tmp_path, embedder=emb,
        dim=32, k=4, seed=42,
    )
    w.add_chunks([Chunk(id=cid, text=text, meta={}) for cid, text in CORPUS])
    w.commit()
    return KB.open(tmp_path / "explain.kbi"), emb


def test_rrf_hit_carries_rank_provenance(kb):
    k, emb = kb
    hits = k.search("reciprocal rank fusion", embedder=emb, k=4)
    assert hits
    for h in hits:
        e = h.explain
        assert e["mode"] == "rrf"
        assert e["rrf_c"] == 60
        assert set(e["legs"]) <= {"dense", "lex"}
        # Every listed leg contributed a rank; every absent leg did not.
        assert ("dense" in e["legs"]) == (e["dense_rank"] is not None)
        assert ("lex" in e["legs"]) == (e["lex_rank"] is not None)


def test_rrf_contributions_sum_to_fused(kb):
    k, emb = kb
    hits = k.search("popcount scan", embedder=emb, k=4)
    for h in hits:
        total = h.explain["rrf_dense"] + h.explain["rrf_lex"]
        assert h.fused == pytest.approx(total)
        # And each contribution is exactly 1/(c+rank) for its recorded rank.
        for leg, key in (("dense_rank", "rrf_dense"), ("lex_rank", "rrf_lex")):
            rank = h.explain[leg]
            expected = 0.0 if rank is None else 1.0 / (60 + rank)
            assert h.explain[key] == pytest.approx(expected)


def test_weighted_mode_records_alpha_and_contributions(kb):
    k, emb = kb
    hits = k.search("bit-packed vectors", embedder=emb, k=4, alpha=0.7)
    assert hits
    for h in hits:
        e = h.explain
        assert e["mode"] == "weighted"
        assert e["alpha"] == 0.7
        assert h.fused == pytest.approx(e["weighted_dense"] + e["weighted_lex"])


def test_oov_query_is_visible_in_explain(kb):
    k, emb = kb
    # None of these tokens appear in CORPUS, so the lexical arm scores zero and
    # hybrid degrades to dense-only. That degradation is the thing being made
    # visible; without it the caller sees ordinary-looking results.
    q = "zzyzx quuxbar frobnicate"
    diag = k.explain_query(q)
    assert diag["in_vocab"] == []
    assert set(diag["oov"]) == set(diag["tokens"])
    assert diag["vocab_size"] > 0

    hits = k.search(q, embedder=emb, k=2)
    for h in hits:
        assert h.explain["lex_rank"] is None
        assert h.explain["query"]["oov"]


def test_in_vocab_query_reaches_the_lexical_leg(kb):
    k, emb = kb
    diag = k.explain_query("postgres replica")
    assert set(diag["in_vocab"]) == {"postgres", "replica"}
    assert diag["oov"] == []

    hits = k.search("postgres replica", embedder=emb, k=4)
    assert any(h.explain["lex_rank"] is not None for h in hits)


def test_query_explain_is_shared_not_copied_per_hit(kb):
    k, emb = kb
    hits = k.search("migration replica", embedder=emb, k=4)
    assert len(hits) > 1
    first = hits[0].explain["query"]
    assert all(h.explain["query"] is first for h in hits)


def test_pool_counts_recorded(kb):
    k, emb = kb
    hits = k.search("popcount", embedder=emb, k=2)
    q = hits[0].explain["query"]
    assert q["dense_candidates"] >= q["dense_kept_after_floor"]
    assert q["lex_candidates"] is not None
    assert hits[0].explain["over_fetch"] >= 2
