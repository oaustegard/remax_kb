"""File-type handlers for :func:`remax_kb.pack.pack_directory`.

A ``FileHandler`` is any callable conforming to::

    def handler(path: Path) -> tuple[str, dict[str, Any]]

It returns ``(full_text, file_level_meta)``. The chunker then chunks
the text and decorates each chunk's ``meta`` with the file-level meta.

Default handlers ship for ``.md / .markdown``, ``.txt``, ``.html / .htm``,
``.pdf``, and ``.rst``. Users supply ``handlers={".docx": my_handler, ...}``
to override or extend.

PDF handling uses ``pypdf`` if installed; it returns empty text and a
warning rather than crashing when a PDF is encrypted/scanned. HTML
handling uses ``beautifulsoup4`` if installed; it falls back to a
regex strip if BS4 is unavailable.
"""
from __future__ import annotations

import html
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Protocol

FileHandler = Callable[[Path], "tuple[str, dict[str, Any]]"]


# --------------------------------------------------------------------- #
# Markdown / plain text
# --------------------------------------------------------------------- #

_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


def handle_markdown(path: Path) -> tuple[str, dict[str, Any]]:
    """Read a markdown file; strip YAML frontmatter if present."""
    text = path.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, Any] = {"source_path": str(path), "kind": "markdown"}
    m = _FRONTMATTER_RE.match(text)
    if m:
        meta["frontmatter_raw"] = m.group(0)
        text = text[m.end():]
    return text, meta


def handle_text(path: Path) -> tuple[str, dict[str, Any]]:
    """Read a text file verbatim."""
    return (
        path.read_text(encoding="utf-8", errors="replace"),
        {"source_path": str(path), "kind": "text"},
    )


def handle_rst(path: Path) -> tuple[str, dict[str, Any]]:
    """Read an rST file as plain text. We deliberately don't pull in
    docutils — directive markers slip through, but retrieval over the
    prose is fine."""
    return (
        path.read_text(encoding="utf-8", errors="replace"),
        {"source_path": str(path), "kind": "rst"},
    )


# --------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------- #


def _html_via_bs4(raw: str) -> tuple[str, dict[str, Any]]:
    from bs4 import BeautifulSoup  # type: ignore[import-not-found]

    soup = BeautifulSoup(raw, "html.parser")
    meta: dict[str, Any] = {}

    # get_text(), not .string: Tag.string is None when the title has any child
    # element (<title>A <b>B</b></title>), which would silently drop the title.
    if soup.title is not None:
        title = soup.title.get_text(strip=True)
        if title:
            meta["title"] = title
    for name in ("description", "og:description"):
        tag = soup.find("meta", attrs={"name": name}) or soup.find(
            "meta", attrs={"property": name}
        )
        if tag and tag.get("content"):
            meta["description"] = tag["content"].strip()
            break
    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        meta["url"] = canonical["href"]
    for name in ("article:published_time", "date", "pubdate"):
        tag = soup.find("meta", attrs={"property": name}) or soup.find(
            "meta", attrs={"name": name}
        )
        if tag and tag.get("content"):
            meta["date"] = tag["content"]
            break

    for tag_name in ("script", "style", "nav", "footer", "header", "aside"):
        for t in soup.find_all(tag_name):
            t.decompose()

    body = soup.find("article") or soup.find("main") or soup.find("body") or soup
    text = body.get_text("\n", strip=True)
    return text, meta


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SCRIPT_RE = re.compile(
    r"<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


_HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_HTML_META_RE = re.compile(r"<(meta|link)\b([^>]*)>", re.IGNORECASE)
_HTML_ATTR_RE = re.compile(
    r"""([\w:.-]+)\s*=\s*("([^"]*)"|'([^']*)'|([^\s"'>]+))""", re.IGNORECASE
)


def _html_attrs(blob: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _HTML_ATTR_RE.finditer(blob):
        val = m.group(3) if m.group(3) is not None else (
            m.group(4) if m.group(4) is not None else m.group(5) or ""
        )
        out[m.group(1).lower()] = html.unescape(val)
    return out


def _html_fallback(raw: str) -> tuple[str, dict[str, Any]]:
    """Regex strip, used when beautifulsoup4 is not installed.

    Extracts the SAME head metadata keys as the BS4 path. It used to return an
    empty dict, which meant title/description/url/date were silently dropped
    for every packed HTML file whenever the optional ``handlers`` extra was not
    installed — with no warning and no difference in the returned shape, so the
    only symptom was worse retrieval. ``handle_html``'s contract is not
    conditional on an optional dependency.
    """
    meta: dict[str, Any] = {}

    m = _HTML_TITLE_RE.search(raw)
    if m:
        title = html.unescape(_HTML_TAG_RE.sub(" ", m.group(1))).strip()
        title = re.sub(r"\s+", " ", title)
        if title:
            meta["title"] = title

    metas = [(tag.lower(), _html_attrs(attrs))
             for tag, attrs in _HTML_META_RE.findall(raw)]

    for name in ("description", "og:description"):
        for tag, a in metas:
            if tag != "meta" or not a.get("content"):
                continue
            if a.get("name", "").lower() == name or a.get("property", "").lower() == name:
                meta["description"] = a["content"].strip()
                break
        if "description" in meta:
            break

    for tag, a in metas:
        if tag == "link" and "canonical" in a.get("rel", "").lower() and a.get("href"):
            meta["url"] = a["href"]
            break

    for name in ("article:published_time", "date", "pubdate"):
        for tag, a in metas:
            if tag != "meta" or not a.get("content"):
                continue
            if a.get("property", "").lower() == name or a.get("name", "").lower() == name:
                meta["date"] = a["content"]
                break
        if "date" in meta:
            break

    cleaned = _HTML_SCRIPT_RE.sub(" ", raw)
    # <head> carries no body prose; without removing it the title and every
    # meta/link attribute value would land in the extracted text.
    cleaned = re.sub(r"<head\b.*?</head>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    text = _HTML_TAG_RE.sub(" ", cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text, meta


def handle_html(path: Path) -> tuple[str, dict[str, Any]]:
    """Extract main content from an HTML file. Uses BeautifulSoup if
    installed; otherwise a regex-based fallback. Strips nav/footer/
    header/script/style. Pulls title/description/url/date from head.

    Both paths populate the same metadata keys — the regex fallback is worse at
    malformed markup, not at metadata. See ``tests/test_directory_pack.py``,
    which runs the metadata assertions against both.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        text, meta = _html_via_bs4(raw)
    except ImportError:
        text, meta = _html_fallback(raw)
    meta.setdefault("source_path", str(path))
    meta.setdefault("kind", "html")
    return text, meta


# --------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------- #


def handle_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    """Extract text from a PDF via pypdf. On failure (encrypted /
    scanned / missing pypdf) emit an empty string and a warning rather
    than crashing the run — the packer should keep going."""
    meta: dict[str, Any] = {"source_path": str(path), "kind": "pdf"}
    try:
        import pypdf  # type: ignore[import-not-found]
    except ImportError:
        warnings.warn(
            f"pypdf not installed; skipping {path}. "
            f"`pip install pypdf` to enable PDF extraction.",
            stacklevel=2,
        )
        return "", meta
    except BaseException as exc:  # noqa: BLE001 — e.g. cryptography rust-panic
        warnings.warn(
            f"pypdf import failed ({type(exc).__name__}: {exc}); skipping {path}.",
            stacklevel=2,
        )
        return "", meta

    try:
        reader = pypdf.PdfReader(str(path))
        if reader.is_encrypted:
            warnings.warn(f"{path} is encrypted; emitting empty text.", stacklevel=2)
            return "", meta
        meta["page_count"] = len(reader.pages)
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(pages).strip()
        return text, meta
    except Exception as exc:  # noqa: BLE001 — pypdf has many failure modes
        warnings.warn(f"pypdf failed on {path}: {exc}; emitting empty text.", stacklevel=2)
        return "", meta


# --------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------- #


DEFAULT_HANDLERS: dict[str, FileHandler] = {
    ".md": handle_markdown,
    ".markdown": handle_markdown,
    ".txt": handle_text,
    ".rst": handle_rst,
    ".html": handle_html,
    ".htm": handle_html,
    ".pdf": handle_pdf,
}


__all__ = [
    "FileHandler",
    "DEFAULT_HANDLERS",
    "handle_markdown",
    "handle_text",
    "handle_html",
    "handle_pdf",
    "handle_rst",
]
