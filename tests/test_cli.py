"""CLI surface tests.

Drives ``remax_kb.cli.main`` directly via argv, with embedders monkey-
patched to a stub so we don't load torch or hit the network. The
contract we're locking in:

* ``remax-kb pack <dir> -o <kb> --embedder <stub>`` writes a .kb.
* ``remax-kb info <kb>`` prints a JSON summary that includes the
  embedder model_id and chunk_count.
* ``remax-kb query <kb> <query>`` returns JSON ``{"hits": [...]}``
  with ``len(hits) == --k``.
* ``--embedder gemini`` is wired through to ``GeminiEmbedder``
  (verified by checking ``model_id`` in the manifest after pack).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("remax")


FULL_DIM = 64


class StubEmbedder:
    model_id = "stub/cli-embedder"
    model_revision = "0" * 40
    task_adapter = "retrieval"
    pooling = "stub"
    full_dim = FULL_DIM
    normalize_l2 = True
    release_url = None
    release_sha256 = None
    prompts = {"query": "Q: ", "document": "D: "}

    def fingerprint(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_adapter": self.task_adapter,
            "pooling": self.pooling,
            "full_dim": self.full_dim,
        }

    def encode(self, texts, *, prompt):
        out = np.empty((len(texts), self.full_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(
                hashlib.sha256(t.encode("utf-8")).digest()[:8], "little"
            )
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.full_dim).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            out[i] = v
        return out


@pytest.fixture
def stub_embedder(monkeypatch):
    """Make ``--embedder stub`` resolve to StubEmbedder by monkeypatching
    the CLI's embedder factory."""
    from remax_kb import cli

    real_build = cli._build_embedder

    def build(name, args):
        if name == "stub":
            return StubEmbedder()
        return real_build(name, args)

    monkeypatch.setattr(cli, "_build_embedder", build)
    return cli


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("# Cats\nCats purr and meow.\n", encoding="utf-8")
    (root / "b.txt").write_text("Dogs bark and wag tails.\n", encoding="utf-8")
    return root


def test_cli_pack_writes_kb(stub_embedder, corpus_dir: Path, tmp_path: Path, capsys):
    out = tmp_path / "out.kb"
    rc = stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0"]
    )
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0
    captured = capsys.readouterr()
    assert "wrote" in captured.out


def test_cli_info_prints_manifest(stub_embedder, corpus_dir: Path, tmp_path: Path, capsys):
    out = tmp_path / "out.kb"
    stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0"]
    )
    capsys.readouterr()  # discard pack output

    rc = stub_embedder.main(["info", str(out)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["embedder"]["model_id"] == StubEmbedder.model_id
    assert payload["chunk_count"] > 0
    assert payload["binarizer"]["dim"] == 32
    assert payload["binarizer"]["k"] == 4


def test_cli_query_returns_k_hits(stub_embedder, corpus_dir: Path, tmp_path: Path, capsys):
    out = tmp_path / "out.kb"
    stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0"]
    )
    capsys.readouterr()

    rc = stub_embedder.main(
        ["query", str(out), "cats purring", "--k", "1", "--embedder", "stub"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["hits"]) == 1
    assert "id" in payload["hits"][0]
    assert "distance" in payload["hits"][0]


def test_cli_pack_with_gemini_wiring(monkeypatch, corpus_dir: Path, tmp_path: Path, capsys):
    """``--embedder gemini`` should produce a manifest whose model_id
    starts with ``google/`` and whose release_url is None."""
    from remax_kb import cli, KB

    # Stub out the actual Gemini HTTP layer.
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    import httpx

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def post(self, url, *, json):
            n = len(json["requests"])
            dim = json["requests"][0]["outputDimensionality"]
            rng = np.random.default_rng(0)
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {
                    "embeddings": [
                        {"values": rng.standard_normal(dim).tolist()} for _ in range(n)
                    ]
                },
                "text": "",
                "raise_for_status": lambda self: None,
            })()

    monkeypatch.setattr(httpx, "Client", FakeClient)

    out = tmp_path / "gemini.kb"
    rc = cli.main([
        "pack", str(corpus_dir), "-o", str(out),
        "--embedder", "gemini", "--gemini-dim", "64",
        "--dim", "32", "--k", "4", "--seed", "0",
    ])
    assert rc == 0
    kb = KB.open(out)
    assert kb.manifest.embedder.model_id.startswith("google/")
    assert kb.manifest.embedder.release_url is None


def test_cli_unknown_embedder_exits(corpus_dir: Path, tmp_path: Path):
    from remax_kb import cli

    with pytest.raises(SystemExit, match="unknown embedder"):
        cli.main([
            "pack", str(corpus_dir), "-o", str(tmp_path / "x.kb"),
            "--embedder", "nope",
        ])


# --------------------------------------------------------------------------- #
# v2 surface: pack --v2, auto-detecting query/info, migrate
# --------------------------------------------------------------------------- #
def test_cli_pack_v2_and_query_autodetect(stub_embedder, corpus_dir: Path, tmp_path: Path, capsys):
    out = tmp_path / "out.kbi"
    rc = stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0"]
    )
    assert rc == 0
    assert out.exists()
    assert (tmp_path / "out.kbc").is_dir()
    capsys.readouterr()

    # query auto-detects v2 → hybrid output shape (no v1 "distance" key)
    rc = stub_embedder.main(
        ["query", str(out), "cats purring", "--k", "1", "--embedder", "stub"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["spec_version"] == "2"
    assert payload["fusion"] == "rrf"
    assert len(payload["hits"]) == 1
    hit = payload["hits"][0]
    assert "id" in hit and "fused" in hit and "verified" in hit
    assert hit["verified"] is True


def test_cli_info_autodetects_v2(stub_embedder, corpus_dir: Path, tmp_path: Path, capsys):
    out = tmp_path / "out.kbi"
    stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0"]
    )
    capsys.readouterr()

    rc = stub_embedder.main(["info", str(out)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["spec_version"] == "2"
    assert payload["kind"] == "split-index"
    assert payload["chunks"]["live_count"] > 0
    assert payload["embedder"]["model_id"] == StubEmbedder.model_id


def test_cli_migrate_v1_to_v2(stub_embedder, corpus_dir: Path, tmp_path: Path, capsys):
    v1 = tmp_path / "legacy.kb"
    stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(v1), "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0"]
    )
    capsys.readouterr()

    out_dir = tmp_path / "migrated"
    rc = stub_embedder.main(["migrate", str(v1), "--out", str(out_dir), "--name", "legacy"])
    assert rc == 0
    assert (out_dir / "legacy.kbi").is_file()
    assert (out_dir / "legacy.kbc").is_dir()
    assert "wrote" in capsys.readouterr().out


def test_cli_migrate_rejects_v2_input(stub_embedder, corpus_dir: Path, tmp_path: Path, capsys):
    out = tmp_path / "out.kbi"
    stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0"]
    )
    capsys.readouterr()

    rc = stub_embedder.main(["migrate", str(out), "--out", str(tmp_path / "x")])
    assert rc == 1
    assert "already spec v2" in capsys.readouterr().err


# --------------------------------------------------------------------- #
# v2 binarizer / retrieval flags
#
# KBWriter has supported `projection`, `srht_rounds`, `rotations_quant` and
# (now) `min_sim` all along, but the CLI exposed none of them — so `srht` and
# `rademacher`, shipped as v0.4.0 headline features with ~90 lines of normative
# spec and a JS implementation, were unreachable from `remax-kb pack`. These
# tests assert the flags reach the manifest, which is the only place the
# reader (Python or JS) can see them.
# --------------------------------------------------------------------- #


def _v2_manifest(path: Path) -> dict:
    import zipfile

    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("manifest.json"))


@pytest.mark.parametrize("projection", ["haar", "rademacher", "srht"])
def test_cli_pack_v2_projection_flag(stub_embedder, corpus_dir: Path,
                                     tmp_path: Path, capsys, projection):
    out = tmp_path / f"{projection}.kbi"
    rc = stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0", "--projection", projection]
    )
    assert rc == 0
    capsys.readouterr()
    b = _v2_manifest(out)["binarizer"]
    assert b["projection"] == projection
    # rademacher/srht regenerate planes from the seed and ship no sidecar
    if projection == "haar":
        assert b["rotations_quant"] == "float32"
    else:
        assert b["rotations_quant"] == "none"
    if projection == "srht":
        assert b["srht_rounds"] == 3


def test_cli_pack_v2_srht_rounds_flag(stub_embedder, corpus_dir: Path,
                                      tmp_path: Path, capsys):
    out = tmp_path / "srht5.kbi"
    rc = stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0",
         "--projection", "srht", "--srht-rounds", "5"]
    )
    assert rc == 0
    capsys.readouterr()
    assert _v2_manifest(out)["binarizer"]["srht_rounds"] == 5


def test_cli_pack_v2_rotations_quant_flag(stub_embedder, corpus_dir: Path,
                                          tmp_path: Path, capsys):
    import zipfile

    out = tmp_path / "i8.kbi"
    # --projection haar is explicit: rotations_quant only means anything for
    # the projection that HAS a sidecar, and haar stopped being the default in
    # 2026-08. Omitting it here used to pass by accident.
    rc = stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0",
         "--projection", "haar", "--rotations-quant", "int8"]
    )
    assert rc == 0
    capsys.readouterr()
    assert _v2_manifest(out)["binarizer"]["rotations_quant"] == "int8"
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "binarizer/rotations.i8" in names
    assert "binarizer/rotations.scale.f32" in names


def test_cli_pack_v2_default_projection_ships_no_sidecar(
        stub_embedder, corpus_dir: Path, tmp_path: Path, capsys):
    """The DEFAULT pack must be seed-only.

    `--projection` defaults to srht, so a plain `remax-kb pack --v2` produces a
    .kbi with no `binarizer/rotations.*` entry at all — the property that makes
    it readable by js/kb-reader.js, which cannot re-derive Haar planes and has
    to be handed them. This asserts the default itself, not a flag: pass
    nothing and the sidecar must not be there.
    """
    import zipfile

    out = tmp_path / "default.kbi"
    rc = stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0"]
    )
    assert rc == 0
    capsys.readouterr()
    b = _v2_manifest(out)["binarizer"]
    assert b["projection"] == "srht"
    assert b["rotations_quant"] == "none"
    assert b["srht_rounds"] == 3
    with zipfile.ZipFile(out) as zf:
        rot = [n for n in zf.namelist() if n.startswith("binarizer/rotations")]
    assert rot == [], f"default pack shipped a rotation sidecar: {rot}"


def test_cli_pack_v2_projections_are_queryable(stub_embedder, corpus_dir: Path,
                                               tmp_path: Path, capsys):
    """Reaching a projection from the CLI is only worth anything if the
    resulting artifact opens and searches."""
    for projection in ("rademacher", "srht"):
        out = tmp_path / f"q-{projection}.kbi"
        stub_embedder.main(
            ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder",
             "stub", "--dim", "32", "--k", "4", "--seed", "0",
             "--projection", projection]
        )
        capsys.readouterr()
        rc = stub_embedder.main(
            ["query", str(out), "cats purring", "--k", "1", "--embedder", "stub"]
        )
        assert rc == 0, projection
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["hits"]) == 1, projection


def test_cli_pack_v2_min_sim_written_and_read(stub_embedder, corpus_dir: Path,
                                              tmp_path: Path, capsys):
    """read_v2._resolve_min_sim has always read `retrieval.min_sim`; nothing
    wrote it. Assert the round trip, not just the key's presence."""
    from remax_kb.read_v2 import KB as KBv2

    out = tmp_path / "floor.kbi"
    rc = stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0", "--min-sim", "0.75"]
    )
    assert rc == 0
    capsys.readouterr()
    assert _v2_manifest(out)["retrieval"] == {"min_sim": 0.75}
    kb = KBv2.open(str(out))
    assert kb._resolve_min_sim(None) == 0.75          # manifest default applies
    assert kb._resolve_min_sim(0.1) == 0.1            # explicit arg overrides
    assert kb._resolve_min_sim("off") is None         # explicit off overrides


def test_cli_pack_v2_min_sim_omitted_by_default(stub_embedder, corpus_dir: Path,
                                                tmp_path: Path, capsys):
    out = tmp_path / "nofloor.kbi"
    stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0"]
    )
    capsys.readouterr()
    assert "retrieval" not in _v2_manifest(out)


def test_cli_pack_v2_min_sim_auto(stub_embedder, corpus_dir: Path,
                                  tmp_path: Path, capsys):
    from remax_kb.read_v2 import KB as KBv2

    out = tmp_path / "auto.kbi"
    stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0", "--min-sim", "auto"]
    )
    capsys.readouterr()
    assert _v2_manifest(out)["retrieval"] == {"min_sim": "auto"}
    kb = KBv2.open(str(out))
    floor = kb._resolve_min_sim(None)
    assert isinstance(floor, float) and 0.0 < floor < 1.0


def test_writer_rejects_out_of_range_min_sim(tmp_path: Path):
    from remax_kb.pack_v2 import KBWriter

    with pytest.raises(ValueError, match="min_sim"):
        KBWriter.create(name="x", output_dir=tmp_path, embedder=StubEmbedder(),
                        dim=32, k=4, min_sim=1.5)
    with pytest.raises(ValueError, match="min_sim"):
        KBWriter.create(name="x", output_dir=tmp_path, embedder=StubEmbedder(),
                        dim=32, k=4, min_sim="somewhat")


def test_sync_preserves_min_sim_across_recommit(stub_embedder, corpus_dir: Path,
                                                tmp_path: Path, capsys):
    """A sync into an existing .kbi must not silently drop the floor."""
    out = tmp_path / "s.kbi"
    stub_embedder.main(
        ["pack", str(corpus_dir), "-o", str(out), "--v2", "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0", "--min-sim", "0.6"]
    )
    capsys.readouterr()
    (corpus_dir / "c.txt").write_text("Birds sing at dawn.\n", encoding="utf-8")
    rc = stub_embedder.main(
        ["sync", str(corpus_dir), "-o", str(out), "--embedder", "stub"]
    )
    assert rc == 0
    capsys.readouterr()
    assert _v2_manifest(out)["retrieval"] == {"min_sim": 0.6}
