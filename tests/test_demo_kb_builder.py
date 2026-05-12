"""TDD tests for the ``examples/build_claude_docs_kb.py`` build script.

These tests pin the contract of the demo-corpus builder before the
script exists:

* It exposes a ``fetch_sitemap_urls`` helper that parses a sitemap.xml
  string into a list of URLs.
* It exposes a ``html_to_chunks`` helper that takes (url, html) and
  produces ``Chunk`` objects with ``meta["url"]`` set.
* It exposes a ``build_kb(urls, out_path, embedder, ...)`` entry point
  that packs a .kb when given an already-fetched corpus.
* The orchestration ``main()`` function is parameterized enough to swap
  the network fetch out under test.

No live network calls in CI: tests inject a fake fetcher.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("remax")
pytest.importorskip("bs4")


FULL_DIM = 64


class StubEmbedder:
    """Bag-of-words stub: hashes each whitespace token to a per-token
    vector and sums. Chunks that share words land near each other —
    crude but enough to validate the demo-builder's retrieval path."""

    model_id = "stub/demo-builder"
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
    def _tok_vec(tok: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(FULL_DIM).astype(np.float32)
        v /= np.linalg.norm(v) or 1.0
        return v

    def encode(self, texts, *, prompt):
        out = np.zeros((len(texts), self.full_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = [w for w in t.lower().split() if w.isalpha() and len(w) > 2]
            if not toks:
                out[i] = self._tok_vec(t)
                continue
            v = np.zeros(FULL_DIM, dtype=np.float32)
            for tok in set(toks):
                v += self._tok_vec(tok)
            n = np.linalg.norm(v)
            out[i] = v / (n if n else 1.0)
        return out


SAMPLE_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://docs.example.com/intro</loc></url>
  <url><loc>https://docs.example.com/api/clients</loc></url>
  <url><loc>https://docs.example.com/api/sdk</loc></url>
</urlset>
"""


def test_fetch_sitemap_urls_parses_loc_tags():
    """The helper should pull every <loc> from a sitemap.xml string."""
    from examples.build_claude_docs_kb import fetch_sitemap_urls

    urls = fetch_sitemap_urls(SAMPLE_SITEMAP)
    assert urls == [
        "https://docs.example.com/intro",
        "https://docs.example.com/api/clients",
        "https://docs.example.com/api/sdk",
    ]


def test_fetch_sitemap_urls_handles_empty():
    from examples.build_claude_docs_kb import fetch_sitemap_urls

    empty_sitemap = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
    )
    assert fetch_sitemap_urls(empty_sitemap) == []


def test_html_to_chunks_attaches_url_to_meta():
    """Each chunk produced from an HTML page should carry the source URL
    in its meta so users can cite back to the doc."""
    from examples.build_claude_docs_kb import html_to_chunks

    html = (
        "<html><head><title>Prompt caching</title></head><body>"
        "<article><h1>Prompt caching</h1>"
        "<p>Prompt caching is a feature of the Anthropic API that "
        "reduces latency and cost when you reuse prompt prefixes.</p>"
        "<p>To enable caching, set cache_control on a content block.</p>"
        "</article></body></html>"
    )
    url = "https://docs.example.com/prompt-caching"
    chunks = html_to_chunks(url, html)
    assert len(chunks) > 0
    for c in chunks:
        assert c.meta["url"] == url
        assert c.meta.get("title") == "Prompt caching"
        assert c.text  # not empty


def test_html_to_chunks_skips_empty_body():
    """A page with no extractable text should yield zero chunks rather
    than a degenerate empty-text chunk."""
    from examples.build_claude_docs_kb import html_to_chunks

    chunks = html_to_chunks("https://x.invalid/empty", "<html><body></body></html>")
    assert chunks == []


def test_build_kb_packs_provided_corpus(tmp_path: Path):
    """When given a dict of {url: html}, ``build_kb`` should pack a .kb
    with one or more chunks per non-empty page."""
    from examples.build_claude_docs_kb import build_kb
    from remax_kb import KB

    pages = {
        "https://docs.example.com/a": (
            "<html><head><title>A</title></head><body><article>"
            "<p>Document A talks about prompt caching and rate limits.</p>"
            "</article></body></html>"
        ),
        "https://docs.example.com/b": (
            "<html><head><title>B</title></head><body><article>"
            "<p>Document B covers tool use and the MCP server protocol.</p>"
            "</article></body></html>"
        ),
    }
    out = tmp_path / "demo.kb"
    build_kb(
        pages,
        out,
        embedder=StubEmbedder(),
        dim=32,
        k=4,
        seed=0,
        source_description="test fixture",
    )
    kb = KB.open(out)
    assert len(kb) >= 2
    urls = {c["meta"].get("url") for c in kb.chunks}
    assert urls == set(pages.keys())


def test_build_kb_query_lands_on_correct_url(tmp_path: Path):
    """End-to-end: pack a tiny .kb with stub embedder, query a topic,
    expect the matching URL in the top-1 hit."""
    from examples.build_claude_docs_kb import build_kb
    from remax_kb import KB

    pages = {
        "https://docs.example.com/caching": (
            "<html><head><title>Caching</title></head><body><article>"
            "<p>Prompt caching reduces latency and cost when prefixes repeat.</p>"
            "</article></body></html>"
        ),
        "https://docs.example.com/mcp": (
            "<html><head><title>MCP</title></head><body><article>"
            "<p>The Model Context Protocol defines a tool-use server interface.</p>"
            "</article></body></html>"
        ),
    }
    out = tmp_path / "demo.kb"
    build_kb(pages, out, embedder=StubEmbedder(), dim=32, k=4, seed=0)
    kb = KB.open(out)

    emb = StubEmbedder()
    emb.prompts = {"query": "D: ", "document": "D: "}
    hits = kb.search(
        "prompt caching reduces latency and cost when prefixes repeat",
        embedder=emb,
        k=1,
    )
    assert hits[0][1]["meta"]["url"] == "https://docs.example.com/caching"
