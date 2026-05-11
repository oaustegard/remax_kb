"""Skill entry point: search a .kb from the command line.

Caches the opened KB and the loaded embedder in module-level state so
repeated invocations within one Python process (e.g. multiple tool
calls in the same session) reuse them. The cache is keyed on the .kb
path's resolved string, so switching .kb files is supported.

Usage:
    python skill/search.py --kb /mnt/user-data/uploads/foo.kb \
                           --query "..." --k 5

Output: JSON list of {distance, id, text, meta}, sorted ascending.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from a checkout without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from remax_kb import KB  # noqa: E402
from remax_kb.embedders import JinaONNXEmbedder  # noqa: E402


_kb_cache: dict[str, KB] = {}
_embedder: JinaONNXEmbedder | None = None


def _get_embedder() -> JinaONNXEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = JinaONNXEmbedder()
    return _embedder


def _get_kb(path: str) -> KB:
    key = str(Path(path).resolve())
    if key not in _kb_cache:
        _kb_cache[key] = KB.open(key)
    return _kb_cache[key]


def search(kb_path: str, query: str, k: int = 5) -> list[dict]:
    kb = _get_kb(kb_path)
    emb = _get_embedder()
    hits = kb.search(query, embedder=emb, k=k)
    return [
        {
            "distance": int(dist),
            "id": chunk["id"],
            "text": chunk["text"],
            "meta": chunk.get("meta", {}),
        }
        for dist, chunk in hits
    ]


def _resolve_default_kb() -> str | None:
    """Implement the resolution chain from SKILL.md."""
    candidates: list[Path] = []
    for root in ("/mnt/user-data/uploads", "/mnt/project"):
        p = Path(root)
        if p.exists():
            candidates.extend(p.glob("*.kb"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kb", help="path to .kb (default: auto-resolve)")
    ap.add_argument("--query", required=True, help="user query string")
    ap.add_argument("--k", type=int, default=5, help="number of results")
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="indent the JSON output for human reading",
    )
    args = ap.parse_args()

    kb_path = args.kb or _resolve_default_kb()
    if not kb_path:
        print(
            "No --kb given and no .kb files found under /mnt/user-data/uploads "
            "or /mnt/project. Pass --kb explicitly.",
            file=sys.stderr,
        )
        return 2

    hits = search(kb_path, args.query, k=args.k)
    indent = 2 if args.pretty else None
    print(json.dumps({"kb": kb_path, "query": args.query, "hits": hits}, indent=indent, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
