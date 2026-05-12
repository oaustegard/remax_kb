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
