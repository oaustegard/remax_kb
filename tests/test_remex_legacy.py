"""remex rotation compatibility.

remax_kb <= 0.5 packed remex codes with remex 0.6.0, whose quantizer was
always Haar, and recorded no rotation. remex >= 1.0 defaults to "rht", so a
reader that let remex pick the rotation refused those files ("rotation
mismatch: haar vs rht"). The fixtures in fixtures/legacy_remex were written by
remax_kb 0.5.0 + remex 0.6.0 (make_fixture.py); expected_rows.npy is how
remex 0.6.0 decoded them.
"""
from __future__ import annotations

import shutil
import zipfile
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("remex")
pytest.importorskip("remax")

from remax_kb import KB as KBv1, pack
from remax_kb.pack_v2 import KBWriter
from remax_kb.read_v2 import KB as KBv2

FIX = Path(__file__).parent / "fixtures" / "legacy_remex"

import sys
sys.path.insert(0, str(FIX))
from make_fixture import DeterministicEmbedder, TEXTS, chunks  # noqa: E402


def _unit(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _assert_same_directions(rows):
    # remex >= 0.7 renormalizes decoded rows; 0.6.0 did not. Directions match.
    expected = np.load(FIX / "expected_rows.npy")
    cos = np.sum(_unit(rows) * _unit(expected), axis=1)
    assert cos.min() > 0.9999, cos


def _manifest(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("manifest.json"))


def test_legacy_fixtures_have_no_rotation_field():
    assert "rotation" not in _manifest(FIX / "legacy_v1.kb")["binarizer"]
    assert "rotation" not in _manifest(FIX / "legacy_v2.kbi")["binarizer"]


def test_legacy_v1_decodes_as_haar():
    kb = KBv1.open(FIX / "legacy_v1.kb")
    _assert_same_directions(kb._remex_doc_matrix())
    emb = DeterministicEmbedder()
    for i, t in enumerate(TEXTS):
        assert kb.search(t, embedder=emb, k=1)[0][1]["id"] == f"c{i}"


def test_legacy_v2_decodes_as_haar(tmp_path):
    for p in FIX.glob("legacy_v2*"):
        (shutil.copytree if p.is_dir() else shutil.copy)(p, tmp_path / p.name)
    kb = KBv2.open(tmp_path / "legacy_v2.kbi")
    _assert_same_directions(kb._remex_doc_matrix())
    emb = DeterministicEmbedder()
    for i, t in enumerate(TEXTS):
        assert kb.search(t, embedder=emb, k=1)[0].chunk_id == f"c{i}"


def test_legacy_v2_recommit_keeps_haar(tmp_path):
    for p in FIX.glob("legacy_v2*"):
        (shutil.copytree if p.is_dir() else shutil.copy)(p, tmp_path / p.name)
    emb = DeterministicEmbedder()
    w = KBWriter.open(name="legacy_v2", output_dir=tmp_path, embedder=emb)
    from remax_kb.pack import Chunk
    w.add_chunks([Chunk(id="new", text="A late addition about ravens.", meta={})])
    w.commit()
    assert _manifest(tmp_path / "legacy_v2.kbi")["binarizer"]["rotation"] == "haar"
    kb = KBv2.open(tmp_path / "legacy_v2.kbi")
    _assert_same_directions(kb._remex_doc_matrix()[: len(TEXTS)])
    assert kb.search("A late addition about ravens.", embedder=emb, k=1)[0].chunk_id == "new"


def test_new_artifacts_record_remex_default_rotation(tmp_path):
    from remex import Quantizer
    default = Quantizer(d=32, bits=4, seed=7).rotation
    emb = DeterministicEmbedder()
    pack(chunks(), tmp_path / "new.kb", embedder=emb, dim=32, seed=7, codec="remex", bits=4)
    w = KBWriter.create(name="new", output_dir=tmp_path, embedder=emb, dim=32, seed=7,
                        codec="remex", bits=4)
    w.add_chunks(chunks()); w.commit()
    assert _manifest(tmp_path / "new.kb")["binarizer"]["rotation"] == default
    assert _manifest(tmp_path / "new.kbi")["binarizer"]["rotation"] == default
    for kb, hit in ((KBv1.open(tmp_path / "new.kb"), lambda h: h[1]["id"]),
                    (KBv2.open(tmp_path / "new.kbi"), lambda h: h.chunk_id)):
        for i, t in enumerate(TEXTS):
            assert hit(kb.search(t, embedder=emb, k=1)[0]) == f"c{i}"


def test_remax_manifest_carries_no_rotation(tmp_path):
    emb = DeterministicEmbedder()
    pack(chunks(), tmp_path / "r.kb", embedder=emb, dim=32, k=2, seed=7)
    assert "rotation" not in _manifest(tmp_path / "r.kb")["binarizer"]
