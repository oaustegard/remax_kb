"""CLI ``sync`` subcommand — incremental v2 (re)build from a directory.

``remax-kb sync <dir> -o <prefix>.kbi`` opens an existing index (or
creates one), diffs the directory against it content-addressed, and
commits only the delta. Re-running with no source changes embeds
nothing. Optionally compacts when the tombstone ratio crosses a
threshold.
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

    def __init__(self):
        self.encoded_texts: list[str] = []

    def fingerprint(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "task_adapter": self.task_adapter,
                "pooling": self.pooling, "full_dim": self.full_dim}

    def encode(self, texts, *, prompt):
        self.encoded_texts.extend(texts)
        out = np.empty((len(texts), self.full_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.full_dim).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            out[i] = v
        return out


@pytest.fixture
def stub_embedder(monkeypatch):
    from remax_kb import cli

    holder = StubEmbedder()
    real_build = cli._build_embedder

    def build(name, args):
        if name == "stub":
            return holder
        return real_build(name, args)

    monkeypatch.setattr(cli, "_build_embedder", build)
    cli._test_embedder = holder  # expose for assertions
    return cli


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("# Cats\nCats purr and meow softly.\n", encoding="utf-8")
    (root / "b.txt").write_text("Dogs bark and wag their tails happily.\n", encoding="utf-8")
    return root


def _sync(cli, corpus_dir, out, *extra):
    return cli.main(
        ["sync", str(corpus_dir), "-o", str(out), "--embedder", "stub",
         "--dim", "32", "--k", "4", "--seed", "0", *extra]
    )


def test_sync_creates_then_incrementally_updates(stub_embedder, corpus_dir, tmp_path, capsys):
    cli = stub_embedder
    out = tmp_path / "kb.kbi"

    # First run: green-field create.
    assert _sync(cli, corpus_dir, out) == 0
    first = json.loads(capsys.readouterr().out)
    assert out.exists() and (tmp_path / "kb.kbc").is_dir()
    assert first["added"] >= 2
    assert first["unchanged"] == 0
    assert first["embedded"] == first["added"]
    base_live = first["live_count"]

    # Second run: add a new file → only its chunk(s) embedded.
    (corpus_dir / "c.md").write_text("# Birds\nRavens cache food and remember faces.\n", encoding="utf-8")
    cli._test_embedder.encoded_texts.clear()
    assert _sync(cli, corpus_dir, out) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["added"] >= 1
    assert second["unchanged"] == base_live
    assert second["embedded"] == second["added"]
    assert len(cli._test_embedder.encoded_texts) == second["added"]
    assert second["live_count"] == base_live + second["added"]

    # Third run: no source changes → nothing embedded.
    cli._test_embedder.encoded_texts.clear()
    assert _sync(cli, corpus_dir, out) == 0
    third = json.loads(capsys.readouterr().out)
    assert third["added"] == 0 and third["updated"] == 0 and third["deleted"] == 0
    assert third["embedded"] == 0
    assert cli._test_embedder.encoded_texts == []


def test_sync_reports_full_json_shape(stub_embedder, corpus_dir, tmp_path, capsys):
    cli = stub_embedder
    out = tmp_path / "kb.kbi"
    assert _sync(cli, corpus_dir, out) == 0
    payload = json.loads(capsys.readouterr().out)
    for key in ("added", "updated", "deleted", "unchanged", "embedded",
                "live_count", "total_rows", "compacted"):
        assert key in payload, f"missing {key} in {payload}"
    assert payload["compacted"] is False


def test_sync_compacts_when_threshold_exceeded(stub_embedder, corpus_dir, tmp_path, capsys):
    cli = stub_embedder
    out = tmp_path / "kb.kbi"
    # Seed several files so a single deletion crosses a low threshold.
    for i in range(4):
        (corpus_dir / f"extra{i}.md").write_text(
            f"# Topic {i}\nUnique paragraph {i} discussing retrieval and ravens.\n",
            encoding="utf-8",
        )
    assert _sync(cli, corpus_dir, out) == 0
    capsys.readouterr()

    # Remove one file and sync with a low compaction threshold.
    (corpus_dir / "a.md").unlink()
    assert _sync(cli, corpus_dir, out, "--compact-threshold", "0.05") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deleted"] >= 1
    assert payload["compacted"] is True
    # After compaction there are no tombstones.
    assert payload["total_rows"] == payload["live_count"]
