"""``remax-kb`` command-line interface.

Subcommands:

* ``pack`` — pack a directory of mixed-format documents into a ``.kb``.
* ``query`` — open a ``.kb`` and run a query.
* ``info`` — print the manifest of a ``.kb`` (no embedder needed).

Examples::

    remax-kb pack ./docs/ -o knowledge.kb --dim 256 --k 8 --embedder jina-onnx
    remax-kb pack ./docs/ -o knowledge.kb --embedder gemini --gemini-dim 768
    remax-kb query knowledge.kb "How does X work?" --k 5
    remax-kb info knowledge.kb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_embedder(name: str, args: argparse.Namespace):
    """Construct an embedder by short name. Imports are lazy because
    ``jina-onnx`` and ``jina-torch`` pull heavy deps that the user
    won't have unless they asked for that path."""
    name = name.lower()
    if name == "jina-onnx":
        from .embedders import JinaONNXEmbedder
        return JinaONNXEmbedder()
    if name == "jina-torch":
        from .embedders import JinaTorchEmbedder
        return JinaTorchEmbedder(task_adapter=args.task_adapter or "retrieval")
    if name == "gemini":
        from .embedders import GeminiEmbedder
        return GeminiEmbedder(
            api_key=args.gemini_api_key,
            model=args.gemini_model,
            output_dim=args.gemini_dim,
        )
    raise SystemExit(
        f"unknown embedder {name!r}; choose from: jina-onnx, jina-torch, gemini"
    )


def _cmd_pack(args: argparse.Namespace) -> int:
    from . import pack_directory

    embedder = _build_embedder(args.embedder, args)
    out = pack_directory(
        args.corpus,
        args.out,
        embedder=embedder,
        pattern=args.pattern,
        dim=args.dim,
        k=args.k,
        seed=args.seed,
        source_description=args.source,
        batch_size=args.batch_size,
    )
    size = out.stat().st_size
    print(f"wrote {out} ({size / 1024:.1f} KB)")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from . import KB

    embedder = _build_embedder(args.embedder, args)
    kb = KB.open(args.kb)
    hits = kb.search(args.query, embedder=embedder, k=args.k)
    payload = {
        "kb": str(Path(args.kb).resolve()),
        "query": args.query,
        "hits": [
            {
                "distance": int(dist),
                "id": chunk["id"],
                "text": chunk["text"],
                "meta": chunk.get("meta", {}),
            }
            for dist, chunk in hits
        ],
    }
    print(json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from . import KB

    kb = KB.open(args.kb)
    m = kb.manifest
    payload = {
        "path": str(Path(args.kb).resolve()),
        "spec_version": m.spec_version,
        "chunk_count": m.corpus.chunk_count,
        "built_at": m.corpus.built_at,
        "source": m.corpus.source,
        "embedder": {
            "model_id": m.embedder.model_id,
            "task_adapter": m.embedder.task_adapter,
            "pooling": m.embedder.pooling,
            "full_dim": m.embedder.full_dim,
            "release_url": m.embedder.release_url,
        },
        "binarizer": {
            "kind": m.binarizer.kind,
            "dim": m.binarizer.dim,
            "k": m.binarizer.k,
            "seed": m.binarizer.seed,
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def _embedder_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--embedder",
        default="jina-onnx",
        help="embedder backend: jina-onnx (default), jina-torch, gemini",
    )
    p.add_argument(
        "--task-adapter",
        default=None,
        help="(jina-torch only) task adapter name; default: retrieval",
    )
    p.add_argument(
        "--gemini-api-key",
        default=None,
        help="(gemini only) API key; falls back to $GEMINI_API_KEY",
    )
    p.add_argument(
        "--gemini-model",
        default="gemini-embedding-001",
        help="(gemini only) embedding model id",
    )
    p.add_argument(
        "--gemini-dim",
        type=int,
        default=768,
        help="(gemini only) output dimensionality",
    )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="remax-kb", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pack_p = sub.add_parser("pack", help="pack a directory into a .kb")
    pack_p.add_argument("corpus", help="directory of documents (or a single file)")
    pack_p.add_argument("-o", "--out", required=True, help="destination .kb path")
    pack_p.add_argument("--pattern", default="**/*", help='glob (default: "**/*")')
    pack_p.add_argument("--dim", type=int, default=256, help="binarizer dim (default 256)")
    pack_p.add_argument("--k", type=int, default=8, help="stacked-SimHash stack count (default 8)")
    pack_p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    pack_p.add_argument("--source", default="", help="free-text source description")
    pack_p.add_argument("--batch-size", type=int, default=16, help="embed batch size")
    _embedder_args(pack_p)
    pack_p.set_defaults(func=_cmd_pack)

    q_p = sub.add_parser("query", help="query a .kb")
    q_p.add_argument("kb", help="path to .kb")
    q_p.add_argument("query", help="user query string")
    q_p.add_argument("--k", type=int, default=5, help="number of results")
    q_p.add_argument("--pretty", action="store_true", help="indent JSON output")
    _embedder_args(q_p)
    q_p.set_defaults(func=_cmd_query)

    i_p = sub.add_parser("info", help="print manifest summary")
    i_p.add_argument("kb", help="path to .kb")
    i_p.set_defaults(func=_cmd_info)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
