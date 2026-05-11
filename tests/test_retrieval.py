"""End-to-end retrieval tests against ``examples/tiny_corpus``.

Two layers, both opt-in:

1. ``test_torch_query_lands_in_top_3`` — packs the tiny corpus with the
   jina torch embedder, queries with the same embedder, asserts the
   expected chunk substring shows up in the top-3. Validates the format
   roundtrip + retrieval ranking.

2. ``test_onnx_matches_torch_top1`` — packs with torch, then queries
   the *same* .kb with both the torch embedder and the ONNX embedder,
   asserts they agree on the top-1. This is the actual portability
   claim of the ``.kb`` format: same chunks, same ranking, regardless
   of which embedder produced the query vector. Gated by
   ``REMAX_KB_FULL=1`` because it pulls the 847 MB ONNX export.

Both layers skip cleanly if their prerequisites are missing. CI in this
repo intentionally does not install heavy deps; these tests are local /
manual smoke checks.
"""
from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

import pytest

pytest.importorskip("remax")

# ---- Locate the jina-v5-nano-mirror checkout ----
_mirror = _os.environ.get("JINA_V5_NANO_MIRROR_PATH")
if not _mirror:
    _candidate = _Path(__file__).resolve().parents[2] / "jina-v5-nano-mirror"
    if _candidate.exists():
        _mirror = str(_candidate)
        _os.environ["JINA_V5_NANO_MIRROR_PATH"] = _mirror

if not _mirror:
    pytest.skip(
        "jina-v5-nano-mirror checkout not found; set "
        "$JINA_V5_NANO_MIRROR_PATH or clone next to this repo.",
        allow_module_level=True,
    )

_scripts = _Path(_mirror) / "scripts"
if str(_scripts) not in _sys.path:
    _sys.path.insert(0, str(_scripts))

try:
    import embed as _jina_torch_embed_mod  # noqa: F401
except ImportError as _exc:
    pytest.skip(
        f"could not import jina mirror torch loader from {_scripts}: {_exc}",
        allow_module_level=True,
    )

pytest.importorskip("transformers")
pytest.importorskip("peft")


from remax_kb import KB, pack  # noqa: E402
from remax_kb.embedders import JinaTorchEmbedder  # noqa: E402

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


@pytest.fixture(scope="module")
def torch_embedder():
    return JinaTorchEmbedder(task_adapter="retrieval")


@pytest.mark.parametrize(
    "query,expected_source_file",
    [
        # "What is a faction?" → Federalist 10's faction-definition file.
        ("What is a faction?", "federalist_10_factions.txt"),
        # Madison on checks-and-balances → Federalist 51.
        (
            "Why must government be controlled by checks and balances?",
            "federalist_51_checks.txt",
        ),
        # Lincoln dedicating the cemetery → Gettysburg.
        ("What was said at the dedication of the battlefield?", "gettysburg.txt"),
    ],
)
def test_torch_query_lands_in_top_3(
    packed: Path, torch_embedder, query: str, expected_source_file: str
):
    """Top-3 should contain at least one chunk from the topically-correct
    source file. 1-bit LSH at this corpus size reliably clusters by
    document topic; within-document chunk ranking is noisy and not asserted."""
    kb = KB.open(packed)
    hits = kb.search(query, embedder=torch_embedder, k=3)
    assert hits, "search returned no results"
    sources = {c["meta"]["source_path"] for _, c in hits}
    assert any(expected_source_file in s for s in sources), (
        f"query {query!r}: expected {expected_source_file!r} in top-3 sources, "
        f"got {sorted(sources)} (chunk ids: {[c['id'] for _, c in hits]})"
    )


# ----------------------------------------------------------------- #
# Portability: torch and ONNX query embedders must agree on top-1.
# Gated because it triggers an 847 MB download of model.onnx.
# ----------------------------------------------------------------- #


@pytest.mark.skipif(
    _os.environ.get("REMAX_KB_FULL") != "1",
    reason="set REMAX_KB_FULL=1 to opt in (downloads ~847 MB model.onnx)",
)
def test_onnx_matches_torch_top1(packed: Path, torch_embedder):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from remax_kb.embedders import JinaONNXEmbedder

    kb = KB.open(packed)
    onnx_embedder = JinaONNXEmbedder()
    queries = [
        "What is a faction?",
        "Why must government be controlled by checks and balances?",
        "What was said at the dedication of the battlefield?",
    ]
    mismatches = []
    for q in queries:
        torch_top = kb.search(q, embedder=torch_embedder, k=1)[0][1]["id"]
        onnx_top = kb.search(q, embedder=onnx_embedder, k=1)[0][1]["id"]
        if torch_top != onnx_top:
            mismatches.append((q, torch_top, onnx_top))
    assert not mismatches, (
        "ONNX query path disagreed with torch path on top-1: "
        + "; ".join(f"{q!r} → torch:{t} vs onnx:{o}" for q, t, o in mismatches)
    )
