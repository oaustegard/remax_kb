"""End-to-end retrieval test against ``examples/tiny_corpus``.

Skipped unless the heavyweight build deps (jina-v5-nano-mirror torch
loader + remax) AND the lightweight runtime deps (onnxruntime +
tokenizers) are both importable. CI in this repo intentionally does not
install them — this test is an opt-in smoke check, not a gate.

Run locally with:
    pip install -r requirements-build.txt -r requirements-runtime.txt
    pytest -q tests/test_retrieval.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("remax")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

# Torch loader is the packer-side prereq.
try:
    from jina_v5_nano_mirror.scripts.embed import embed as _jina_torch_embed  # noqa: F401
except ImportError:
    pytest.skip(
        "jina-v5-nano-mirror torch loader not importable; install via "
        "`pip install git+https://github.com/oaustegard/jina-v5-nano-mirror.git`",
        allow_module_level=True,
    )


from remax_kb import KB, pack  # noqa: E402
from remax_kb.embedders import JinaONNXEmbedder, JinaTorchEmbedder  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "tiny_corpus"


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    out = tmp_path_factory.mktemp("kb") / "tiny.kb"
    pack(
        CORPUS,
        out,
        embedder=JinaTorchEmbedder(task_adapter="retrieval"),
        dim=256,
        k=8,
        seed=0,
        source_description="tiny_corpus smoke fixture",
    )
    return out


@pytest.mark.parametrize(
    "query,expected_substring",
    [
        ("What is a faction?", "faction"),
        ("Why is the separation of powers necessary?",
         "ambition"),
        ("What was said at the dedication of the battlefield?",
         "dedicate"),
    ],
)
def test_known_query_lands_in_top_3(packed: Path, query: str, expected_substring: str):
    kb = KB.open(packed)
    emb = JinaONNXEmbedder()
    hits = kb.search(query, embedder=emb, k=3)
    assert hits, "search returned no results"
    blob = " ".join(c["text"].lower() for _, c in hits)
    assert expected_substring.lower() in blob, (
        f"query {query!r}: expected substring {expected_substring!r} in top-3, "
        f"got {[c['id'] for _, c in hits]}"
    )
