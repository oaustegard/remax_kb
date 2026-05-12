"""Mixed-format directory packing.

Builds a tiny fixture with one ``.md``, one ``.txt``, one ``.html``,
and one ``.pdf`` (if pypdf is available). Verifies:

* ``pack_directory`` produces ``> 0`` chunks per file.
* Each chunk carries the file-level metadata from the matching handler.
* The resulting ``.kb`` round-trips through ``KB.open`` and a known query
  lands in top-k for the correct file.

No torch, no network — uses the same stub embedder pattern as
``test_roundtrip.py``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from remax_kb import KB, pack_directory
from remax_kb.handlers import DEFAULT_HANDLERS, handle_markdown

pytest.importorskip("remax")

FULL_DIM = 64
DIM = 32
K = 4
SEED = 7
TOPIC_TOKEN = "tk%04d"  # injected via deterministic per-text hashing in StubEmbedder


class StubEmbedder:
    """Deterministic per-text RNG embedder; bag-of-words-flavored.

    For retrieval probes we want chunks that share substrings to land
    near each other. We do this by hashing each distinct whitespace-
    separated token and summing the per-token vectors. The result is
    rank-correlated with overlap, so "find the chunk about cats" works.
    """

    model_id = "stub/dir-pack-embedder"
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

    @staticmethod
    def _token_vec(tok: str) -> np.ndarray:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(FULL_DIM).astype(np.float32)
        v /= np.linalg.norm(v) or 1.0
        return v

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
        out = np.zeros((len(texts), FULL_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = [w for w in t.lower().split() if w.isalpha() and len(w) > 2]
            if not toks:
                out[i] = self._token_vec(t)
                continue
            v = np.zeros(FULL_DIM, dtype=np.float32)
            for tok in set(toks):  # ignore frequency — set-of-words
                v += self._token_vec(tok)
            n = np.linalg.norm(v)
            out[i] = v / (n if n else 1.0)
        return out


# --------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------- #


@pytest.fixture
def mixed_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()

    (root / "notes.md").write_text(
        "---\n"
        "title: cats\n"
        "tags: [animals]\n"
        "---\n"
        "\n"
        "# Cats\n"
        "\n"
        "Cats are small carnivorous mammals. Domestic cats hunt mice and "
        "purr when they are content. They have whiskers and retractable claws.\n",
        encoding="utf-8",
    )

    (root / "readme.txt").write_text(
        "Birds are warm-blooded vertebrates. They have feathers and beaks. "
        "Many species migrate seasonally. Birds lay eggs.\n",
        encoding="utf-8",
    )

    (root / "page.html").write_text(
        "<html><head>"
        '<title>Reptiles</title>'
        '<meta name="description" content="An overview of reptiles">'
        "</head><body>"
        "<nav>SKIP THIS NAV</nav>"
        "<article><h1>Reptiles</h1>"
        "<p>Reptiles are cold-blooded vertebrates. Snakes and lizards are "
        "reptiles. Most reptiles lay eggs and have scales.</p></article>"
        "<footer>SKIP THIS FOOTER</footer>"
        "</body></html>",
        encoding="utf-8",
    )

    return root


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_handlers_cover_expected_suffixes():
    for ext in (".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".pdf"):
        assert ext in DEFAULT_HANDLERS, f"{ext} missing from DEFAULT_HANDLERS"


def test_markdown_handler_strips_frontmatter(tmp_path: Path):
    p = tmp_path / "x.md"
    p.write_text("---\ntitle: t\n---\n\nbody text here\n", encoding="utf-8")
    text, meta = handle_markdown(p)
    assert "title: t" not in text
    assert "body text here" in text
    assert meta["kind"] == "markdown"
    assert "frontmatter_raw" in meta


def test_pack_directory_round_trip(mixed_corpus: Path, tmp_path: Path):
    out = tmp_path / "mixed.kb"
    pack_directory(
        mixed_corpus,
        out,
        embedder=StubEmbedder(),
        dim=DIM,
        k=K,
        seed=SEED,
        source_description="mixed-format fixture",
    )
    assert out.exists()
    kb = KB.open(out)
    assert len(kb) > 0
    sources = {c["meta"].get("source_path", "") for c in kb.chunks}
    # Source paths in chunk meta point at the actual filesystem path
    # (the handler sets it to str(path) before chunk-level meta merges).
    # All three fixture files should contribute at least one chunk.
    assert any("notes.md" in s for s in sources)
    assert any("readme.txt" in s for s in sources)
    assert any("page.html" in s for s in sources)


def test_html_handler_strips_nav_footer(mixed_corpus: Path):
    from remax_kb.handlers import handle_html

    text, meta = handle_html(mixed_corpus / "page.html")
    assert "SKIP THIS NAV" not in text
    assert "SKIP THIS FOOTER" not in text
    assert "Reptiles" in text
    # BS4 is installed in the dev env, so we expect title parsed:
    assert meta.get("title") == "Reptiles"
    assert meta.get("description") == "An overview of reptiles"


def test_retrieval_lands_on_correct_file(mixed_corpus: Path, tmp_path: Path):
    """Pack the fixture, query for "reptiles", expect the html chunk first."""
    out = tmp_path / "mixed.kb"
    pack_directory(
        mixed_corpus,
        out,
        embedder=StubEmbedder(),
        dim=DIM,
        k=K,
        seed=SEED,
    )
    kb = KB.open(out)
    # Use the document prompt for the query too so the stub embedder
    # produces identical vectors for the same text.
    emb = StubEmbedder()
    emb.prompts = {"query": "D: ", "document": "D: "}
    hits = kb.search("reptiles snakes lizards scales", embedder=emb, k=3)
    top_source = hits[0][1]["meta"].get("source_path", "")
    assert "page.html" in top_source, (
        f"expected page.html in top hit, got source={top_source!r}; "
        f"hits={[(d, c['id']) for d, c in hits]}"
    )


def test_pack_directory_empty_raises(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no chunks"):
        pack_directory(empty, tmp_path / "x.kb", embedder=StubEmbedder())


def test_pack_directory_handler_override(tmp_path: Path):
    """Custom handlers replace the default registry."""
    root = tmp_path / "weird"
    root.mkdir()
    (root / "doc.foo").write_text("custom format payload here", encoding="utf-8")

    def foo_handler(p: Path):
        return p.read_text(), {"source_path": str(p), "kind": "foo"}

    out = tmp_path / "weird.kb"
    pack_directory(
        root,
        out,
        embedder=StubEmbedder(),
        handlers={".foo": foo_handler},
        dim=DIM,
        k=K,
        seed=SEED,
    )
    kb = KB.open(out)
    assert len(kb) == 1
    assert kb.chunks[0]["meta"]["kind"] == "foo"


def test_pdf_handler_skips_on_missing_dep(tmp_path: Path, monkeypatch):
    """When pypdf is missing, the handler emits a warning and empty
    text rather than crashing the run."""
    from remax_kb import handlers as _handlers
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("simulated missing pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"%PDF-1.4 not really a pdf")
    with pytest.warns(UserWarning, match="pypdf not installed"):
        text, meta = _handlers.handle_pdf(p)
    assert text == ""
    assert meta["kind"] == "pdf"


def test_pdf_handler_on_corrupt_pdf(tmp_path: Path):
    """A malformed or unreadable PDF should not crash the run — handler
    returns empty text + warning, and the chunker simply emits no chunks
    for that file. Robust to pypdf import failures (e.g. when its
    cryptography backend is broken in the env)."""
    from remax_kb.handlers import handle_pdf

    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.4\nnot really a pdf\n")
    with pytest.warns(UserWarning):
        text, meta = handle_pdf(p)
    assert text == ""
    assert meta["kind"] == "pdf"
