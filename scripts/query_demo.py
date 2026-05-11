"""Query a .kb from the command line.

Usage:
    python scripts/query_demo.py path/to/file.kb "query string" [--k 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from remax_kb import KB  # noqa: E402
from remax_kb.embedders import JinaONNXEmbedder  # noqa: E402


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kb", help=".kb path")
    ap.add_argument("query", help="query string")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    kb = KB.open(args.kb)
    emb = JinaONNXEmbedder()
    hits = kb.search(args.query, embedder=emb, k=args.k)
    for dist, chunk in hits:
        snippet = chunk["text"].replace("\n", " ")[:200]
        print(f"[{chunk['id']}, hamming={dist}] {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
