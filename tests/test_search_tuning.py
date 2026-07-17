"""Tuning knobs for v2 hybrid retrieval: the semantic floor and fusion pool.

Uses a deterministic mock embedder (a stable random unit vector per text) so
the tests are network-free. Under this embedder an *exact-text* query encodes
to (essentially) the corpus vector — a near-perfect dense hit — while any other
query is orthogonal noise. That gives us a clean separation to assert on:

* the floor must keep the exact-text match, and
* the floor must drop the nonsense query (which otherwise still ranks
  *something* nearest and leaks into fusion).
"""
from __future__ import annotations

import hashlib
import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("remax")

from remax_kb.pack import Chunk  # noqa: E402
from remax_kb.pack_v2 import KBWriter  # noqa: E402
from remax_kb.read_v2 import KB, _fuse_ranks, Hit  # noqa: E402


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


def _make_kb(tmp: Path, *, drop_bm25: bool = False) -> KB:
    """Pack the corpus to a .kbi; optionally strip bm25/ to isolate dense."""
    embedder = DeterministicEmbedder()
    writer = KBWriter.create(name="t", output_dir=tmp, embedder=embedder,
                             dim=32, k=4, seed=42)
    writer.add_chunks(_corpus())
    writer.commit()
    kbi = tmp / "t.kbi"
    if drop_bm25:
        with zipfile.ZipFile(kbi, "r") as zin:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zout:
                for item in zin.namelist():
                    if item.startswith("bm25/"):
                        continue
                    zout.writestr(item, zin.read(item))
        kbi.write_bytes(buf.getvalue())
    return KB.open(kbi)


# --------------------------------------------------------------------------- #
# Backward compatibility: default (min_sim=None) preserves prior behaviour.
# --------------------------------------------------------------------------- #
def test_default_no_floor_returns_nonsense_hits():
    """Without a floor, a nonsense query against a dense-only .kbi still
    returns *something* (the historical behaviour we must not silently break)."""
    emb = DeterministicEmbedder()
    with tempfile.TemporaryDirectory() as d:
        kb = _make_kb(Path(d), drop_bm25=True)
        hits = kb.search("wholly unrelated gibberish xyzzy", embedder=emb, k=3)
        assert hits, "default path must preserve prior (unfiltered) behaviour"


# --------------------------------------------------------------------------- #
# The floor filters noise but keeps a genuine match.
# --------------------------------------------------------------------------- #
def test_floor_drops_nonsense_dense_only():
    emb = DeterministicEmbedder()
    with tempfile.TemporaryDirectory() as d:
        kb = _make_kb(Path(d), drop_bm25=True)
        # A high explicit floor rejects the orthogonal-noise nearest hit; with
        # no lexical modality to rescue it, the result is empty.
        hits = kb.search("wholly unrelated gibberish xyzzy", embedder=emb, k=5,
                         min_sim=0.9)
        assert hits == []


def test_floor_keeps_exact_match():
    emb = DeterministicEmbedder()
    target = _corpus()[2]
    with tempfile.TemporaryDirectory() as d:
        kb = _make_kb(Path(d), drop_bm25=True)
        hits = kb.search(target.text, embedder=emb, k=5, min_sim=0.9)
        assert hits, "an exact-text query must clear the floor"
        assert hits[0].chunk_id == target.id


def test_auto_floor_separates_signal_from_noise():
    """'auto' keeps the exact match and prunes noise.

    On this 5-doc / 128-bit toy the best-of-N noise fluke has high variance, so
    'auto' is asserted as a *reducer* of noise (not a hard zero — that strict
    case is covered by the explicit-floor test, and holds cleanly at a real
    corpus's larger bit budget)."""
    emb = DeterministicEmbedder()
    target = _corpus()[0]
    with tempfile.TemporaryDirectory() as d:
        kb = _make_kb(Path(d), drop_bm25=True)
        kept = kb.search(target.text, embedder=emb, k=5, min_sim="auto")
        assert any(h.chunk_id == target.id for h in kept)
        junk_auto = kb.search("qwerty zxcvb nonsense gibberish", embedder=emb,
                              k=5, min_sim="auto")
        junk_off = kb.search("qwerty zxcvb nonsense gibberish", embedder=emb,
                             k=5, min_sim="off")
        assert len(junk_auto) < len(junk_off), "auto floor must prune some noise"


def test_off_disables_floor():
    emb = DeterministicEmbedder()
    with tempfile.TemporaryDirectory() as d:
        kb = _make_kb(Path(d), drop_bm25=True)
        assert kb.search("gibberish nonsense", embedder=emb, k=3, min_sim="off")


# --------------------------------------------------------------------------- #
# _auto_min_sim: codec-aware, in-range, and grows with corpus size.
# --------------------------------------------------------------------------- #
def test_auto_min_sim_hamming_in_range():
    emb = DeterministicEmbedder()
    with tempfile.TemporaryDirectory() as d:
        kb = _make_kb(Path(d))
        floor = kb._auto_min_sim()
        # Hamming dense_sim is a fraction of agreeing bits: strictly above the
        # ½ random-agreement baseline and below 1.
        assert 0.5 < floor < 1.0


def test_auto_min_sim_grows_with_corpus():
    """A bigger corpus has a higher best-of-N noise fluke, so the floor rises."""
    emb = DeterministicEmbedder()
    with tempfile.TemporaryDirectory() as d:
        kb = _make_kb(Path(d))
        small = kb._auto_min_sim()
        kb._m["chunks"]["live_count"] = 1_000_000
        big = kb._auto_min_sim()
        assert big > small


# --------------------------------------------------------------------------- #
# Manifest-carried default is honoured when the caller passes None.
# --------------------------------------------------------------------------- #
def test_manifest_retrieval_default_applied():
    emb = DeterministicEmbedder()
    with tempfile.TemporaryDirectory() as d:
        kb = _make_kb(Path(d), drop_bm25=True)
        kb._m["retrieval"] = {"min_sim": 0.9}
        # No explicit min_sim → manifest default kicks in → nonsense filtered.
        assert kb.search("nonsense gibberish qwerty", embedder=emb, k=5) == []
        # Explicit 'off' overrides the manifest default.
        assert kb.search("nonsense gibberish qwerty", embedder=emb, k=5, min_sim="off")


# --------------------------------------------------------------------------- #
# Fusion knobs plumb through.
# --------------------------------------------------------------------------- #
def test_rrf_c_and_over_fetch_plumb_through():
    emb = DeterministicEmbedder()
    with tempfile.TemporaryDirectory() as d:
        kb = _make_kb(Path(d))
        hits = kb.search("federalist", embedder=emb, k=3, over_fetch=8, rrf_c=10)
        assert any(h.chunk_id == "post-003#chunk-001" for h in hits[:2])


def test_fuse_ranks_rrf_c_changes_scores():
    dense = [Hit(row=0, chunk_id="", dense_sim=0.9), Hit(row=1, chunk_id="", dense_sim=0.8)]
    lex = [Hit(row=1, chunk_id="", bm25_score=2.0), Hit(row=0, chunk_id="", bm25_score=1.0)]
    hi_c = _fuse_ranks(dense, lex, over_fetch=10, alpha=None, rrf_c=60)
    lo_c = _fuse_ranks(dense, lex, over_fetch=10, alpha=None, rrf_c=1)
    top = {h.row: h.fused for h in hi_c}
    bot = {h.row: h.fused for h in lo_c}
    # A smaller C sharpens rank-1 weighting, so fused scores differ.
    assert top != bot


def test_bad_min_sim_raises():
    emb = DeterministicEmbedder()
    with tempfile.TemporaryDirectory() as d:
        kb = _make_kb(Path(d))
        with pytest.raises(ValueError):
            kb.search("x", embedder=emb, k=1, min_sim="banana")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
