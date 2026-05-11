"""Pack a directory of text files into a .kb.

Usage:
    python scripts/pack_demo.py <corpus_dir> <out.kb> [--dim 256] [--k 8] [--seed 0]

Uses :class:`remax_kb.embedders.JinaTorchEmbedder` (heavy — needs torch).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from remax_kb import pack  # noqa: E402
from remax_kb.embedders import JinaTorchEmbedder  # noqa: E402


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", help="directory of .txt / .md files (or a single file)")
    ap.add_argument("out_kb", help="destination .kb path")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="", help="free-text source description")
    args = ap.parse_args()

    embedder = JinaTorchEmbedder(task_adapter="retrieval")
    out = pack(
        args.corpus,
        args.out_kb,
        embedder=embedder,
        dim=args.dim,
        k=args.k,
        seed=args.seed,
        source_description=args.source,
    )
    size = out.stat().st_size
    print(f"wrote {out}  ({size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
