"""Build a `.kb` from a sitemap.xml-indexed documentation site.

The non-Muninn demo for the DIY-KB pitch (issue #77). Point this at
any sitemap whose target pages render content server-side and you'll
get a queryable `.kb` out the other end.

**Note on docs.claude.com (early 2026):** the Anthropic developer docs
are fully client-side rendered — fetching the raw HTML yields a JS
shell with "Loading..." placeholders, no extractable content. Until
either (a) Anthropic ships an SSR variant or (b) you wire a headless
browser into ``_crawl``, fall back to a server-rendered docs site.
The FastAPI docs at ``https://fastapi.tiangolo.com/sitemap.xml`` (153
pages, MkDocs-rendered) and the Click docs index work cleanly.

Usage::

    # Option A: Gemini (recommended; small artifact, low memory)
    export GEMINI_API_KEY=...
    python examples/build_claude_docs_kb.py \\
        --sitemap https://fastapi.tiangolo.com/sitemap.xml \\
        --out fastapi-docs.kb \\
        --embedder gemini --gemini-dim 768 \\
        --max-pages 150

    # Option B: Jina ONNX (no API key; downloads ~850 MB model on first use)
    python examples/build_claude_docs_kb.py \\
        --sitemap https://fastapi.tiangolo.com/sitemap.xml \\
        --out fastapi-docs.kb \\
        --embedder jina-onnx \\
        --max-pages 150

Tests in ``tests/test_demo_kb_builder.py`` exercise the pure-function
helpers without hitting the network.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

# Allow running from a checkout without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from remax_kb import pack  # noqa: E402
from remax_kb.pack import Chunk, default_chunker  # noqa: E402


SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def fetch_sitemap_urls(xml_text: str) -> list[str]:
    """Parse a sitemap.xml string and return the list of ``<loc>`` URLs
    in document order."""
    root = ET.fromstring(xml_text)
    locs = root.findall(".//sm:url/sm:loc", SITEMAP_NS)
    if not locs:  # namespace-less sitemaps appear in the wild
        locs = root.findall(".//url/loc")
    return [loc.text.strip() for loc in locs if loc.text]


def html_to_chunks(url: str, html: str) -> list[Chunk]:
    """Convert one HTML page into ``Chunk`` objects.

    Extracts main content via BeautifulSoup (article > main > body),
    strips nav/footer/header/script/style, then runs the default
    chunker. Each chunk's meta carries the source URL and the page
    title for citation-in-prompt.
    """
    from bs4 import BeautifulSoup  # type: ignore[import-not-found]

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    for tag_name in ("script", "style", "nav", "footer", "header", "aside"):
        for t in soup.find_all(tag_name):
            t.decompose()

    body = soup.find("article") or soup.find("main") or soup.find("body") or soup
    text = body.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    raw = default_chunker(text, source_path=url, target_chars=500)
    return [
        Chunk(
            id=f"{url}#chunk-{i:04d}",
            text=c.text,
            meta={"url": url, "title": title, "kind": "html"},
        )
        for i, c in enumerate(raw)
    ]


def build_kb(
    pages: dict[str, str],
    out_path: Path | str,
    *,
    embedder,
    dim: int = 256,
    k: int = 8,
    seed: int = 0,
    source_description: str = "",
    batch_size: int = 16,
) -> Path:
    """Pack a `.kb` from an already-fetched ``{url: html}`` mapping.

    Useful for unit tests, replay-from-cache scenarios, and any case
    where the network fetch lives outside the build step.
    """
    all_chunks: list[Chunk] = []
    for url, html in pages.items():
        all_chunks.extend(html_to_chunks(url, html))
    if not all_chunks:
        raise ValueError("build_kb: produced 0 chunks from supplied pages")
    return pack(
        all_chunks,
        out_path,
        embedder=embedder,
        dim=dim,
        k=k,
        seed=seed,
        source_description=source_description or "claude-docs demo .kb",
        batch_size=batch_size,
    )


# --------------------------------------------------------------------- #
# CLI (network-driven; CI doesn't run this path)
# --------------------------------------------------------------------- #


def _http_get(url: str, *, timeout: float = 30.0) -> str:
    import httpx

    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "remax_kb-demo-builder/0.1"},
    ) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.text


def _crawl(urls: list[str], *, max_pages: int, sleep: float) -> dict[str, str]:
    """Fetch each URL sequentially with a polite delay. Pages that fail
    to load are skipped with a warning. Returns ``{url: html}``."""
    pages: dict[str, str] = {}
    for i, url in enumerate(urls[:max_pages]):
        try:
            html = _http_get(url)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i+1}/{len(urls)}] skip {url}: {exc}", file=sys.stderr)
            continue
        pages[url] = html
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(urls)}] fetched", file=sys.stderr)
        time.sleep(sleep)
    return pages


def _build_embedder(name: str, args: argparse.Namespace):
    if name == "gemini":
        from remax_kb.embedders import GeminiEmbedder

        return GeminiEmbedder(
            api_key=args.gemini_api_key,
            model=args.gemini_model,
            output_dim=args.gemini_dim,
        )
    if name == "jina-onnx":
        from remax_kb.embedders import JinaONNXEmbedder

        return JinaONNXEmbedder()
    raise SystemExit(f"unknown embedder {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sitemap", required=True, help="URL or local path to sitemap.xml")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-pages", type=int, default=600)
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between fetches")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", default="gemini")
    ap.add_argument("--gemini-api-key", default=None)
    ap.add_argument("--gemini-model", default="gemini-embedding-001")
    ap.add_argument("--gemini-dim", type=int, default=768)
    ap.add_argument("--source", default="Anthropic developer docs (docs.claude.com)")
    args = ap.parse_args(argv)

    sm = args.sitemap
    if sm.startswith("http://") or sm.startswith("https://"):
        xml = _http_get(sm)
    else:
        xml = Path(sm).read_text(encoding="utf-8")
    urls = fetch_sitemap_urls(xml)
    if not urls:
        print("no URLs in sitemap", file=sys.stderr)
        return 1
    print(f"sitemap: {len(urls)} URLs (will fetch up to {args.max_pages})", file=sys.stderr)

    pages = _crawl(urls, max_pages=args.max_pages, sleep=args.sleep)
    print(f"fetched {len(pages)} pages", file=sys.stderr)

    embedder = _build_embedder(args.embedder, args)
    out = build_kb(
        pages,
        args.out,
        embedder=embedder,
        dim=args.dim,
        k=args.k,
        seed=args.seed,
        source_description=args.source,
    )
    size = out.stat().st_size
    print(f"wrote {out} ({size / 1024:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
