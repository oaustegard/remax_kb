"""Seed-only SRHT projection: structure, no sidecar, reader/packer consistency."""
from __future__ import annotations
import hashlib, tempfile, zipfile, json
from pathlib import Path
import numpy as np
import pytest
from remax_kb.pack import Chunk
from remax_kb.pack_v2 import KBWriter
from remax_kb.read_v2 import KB
from remax_kb.projection import srht_matrix


class DeterministicEmbedder:
    model_id="test/mock"; model_revision="t"; task_adapter="retrieval"; pooling="native"
    full_dim=64; normalize_l2=True; release_url=None; release_sha256=None
    prompts={"query":"Query: ","document":"Document: "}
    def fingerprint(self): return {"model_id":self.model_id,"task_adapter":self.task_adapter,
                                   "pooling":self.pooling,"full_dim":self.full_dim}
    def encode(self, texts, *, prompt):
        out=np.zeros((len(texts),self.full_dim),dtype=np.float32)
        for i,t in enumerate(texts):
            v=np.random.default_rng(int.from_bytes(hashlib.sha256(t.encode()).digest()[:4],"little")).standard_normal(self.full_dim).astype(np.float32)
            out[i]=v/(np.linalg.norm(v)+1e-12)
        return out

def _corpus():
    texts=["The raven flies at dawn.","Hamming distance counts mismatched bits.",
           "BM25 ranks by term frequency.","SimHash preserves cosine in bits.",
           "Hadamard transforms are exactly orthogonal.","Federalist 10 on factions."]
    return [Chunk(id=f"p-{i:03d}#chunk-001",text=t,meta={"source":f"p{i}"}) for i,t in enumerate(texts)]

def _build(tmp,dim=64,k=4,rounds=3):
    w=KBWriter.create(name="kb_srht",output_dir=tmp,embedder=DeterministicEmbedder(),
                      dim=dim,k=k,seed=0,projection="srht",srht_rounds=rounds)
    w.add_chunks(_corpus()); w.commit(); return tmp/"kb_srht.kbi"

def test_srht_matrix_deterministic_and_orthogonalish():
    a=srht_matrix(64,2,0,3); b=srht_matrix(64,2,0,3)
    assert a.shape==(2,64,64) and a.dtype==np.float32
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,srht_matrix(64,2,1,3))            # seed matters
    assert not np.array_equal(a,srht_matrix(64,2,0,2))            # rounds matter
    # columns ~unit norm
    np.testing.assert_allclose(np.linalg.norm(a[0],axis=0),1.0,atol=1e-5)

def test_srht_kbi_ships_no_sidecar():
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(_build(Path(d))) as zf:
            assert not any(n.startswith("binarizer/") for n in zf.namelist())
            m=json.loads(zf.read("manifest.json"))
            assert m["binarizer"]["projection"]=="srht"
            assert m["binarizer"]["srht_rounds"]==3
            assert m["binarizer"]["rotations_quant"]=="none"

def test_srht_reader_bit_consistent_with_packer():
    with tempfile.TemporaryDirectory() as d:
        kb=KB.open(_build(Path(d))); emb=DeterministicEmbedder()
        for c in _corpus():
            assert kb._dense_search(c.text,emb)[0].dense_dist==0

def test_srht_rounds_preserved_on_reopen():
    with tempfile.TemporaryDirectory() as d:
        _build(Path(d),rounds=2)
        w=KBWriter.open(name="kb_srht",output_dir=Path(d),embedder=DeterministicEmbedder())
        assert w._srht_rounds==2 and w._projection=="srht"


def test_legacy_manifest_without_projection_reopens_as_haar():
    """A v2 manifest that omits `binarizer.projection` must reopen as HAAR.

    The writer's default moved to srht in 2026-08, and the temptation is to
    track `KBWriter._open`'s fallback to the constructor default. That would be
    a corruption, not a default: an artifact predating the field was packed
    against Haar planes, and re-committing it under srht would append rows in a
    statistically independent sign-space — ~50% of bits flipped between the old
    and new rows of one index, with no error raised anywhere.

    Built by stripping the field from a real haar artifact rather than by
    hand-writing a manifest, so the rest of the artifact stays self-consistent.
    """
    import shutil

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        emb = DeterministicEmbedder()
        w = KBWriter.create(name="legacy", output_dir=tmp, embedder=emb,
                            dim=64, k=4, seed=0, projection="haar")
        w.add_chunks(_corpus())
        w.commit()
        kbi = tmp / "legacy.kbi"

        # Rewrite the .kbi with `projection` (and srht_rounds) removed.
        with zipfile.ZipFile(kbi) as zf:
            entries = {n: zf.read(n) for n in zf.namelist()}
        m = json.loads(entries["manifest.json"])
        assert m["binarizer"].pop("projection") == "haar"
        m["binarizer"].pop("srht_rounds", None)
        entries["manifest.json"] = json.dumps(m).encode()
        with zipfile.ZipFile(kbi, "w", zipfile.ZIP_STORED) as zf:
            for n, data in entries.items():
                zf.writestr(n, data)

        w2 = KBWriter.open(name="legacy", output_dir=tmp, embedder=emb)
        assert w2._projection == "haar", (
            "a manifest with no `projection` must be treated as haar, not as "
            "whatever the current writer default happens to be")
        assert w2._rotations_quant == "float32"

        # And a re-commit puts the field back, explicitly, still haar.
        w2.add_chunks([Chunk(id="p-999#chunk-001", text="one more row",
                             meta={"source": "p999"})])
        w2.commit()
        with zipfile.ZipFile(kbi) as zf:
            b = json.loads(zf.read("manifest.json"))["binarizer"]
            assert b["projection"] == "haar"
            assert "binarizer/rotations.f32" in zf.namelist()
        shutil.rmtree(tmp / "legacy.kbc", ignore_errors=True)
