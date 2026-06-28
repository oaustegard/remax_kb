"""``remax-kb`` command-line interface.

Subcommands:

* ``pack`` — pack a directory of mixed-format documents into a ``.kb``
  (v1, single zip) or, with ``--v2``, a split ``.kbi`` + ``.kbc/`` pair.
* ``query`` — open a ``.kb``/``.kbi`` and run a query. The format is
  auto-detected; v2 adds hybrid (dense + BM25) retrieval.
* ``info`` — print the manifest of a ``.kb``/``.kbi`` (no embedder needed).
* ``migrate`` — one-shot upgrade a v1 ``.kb`` to a v2 ``.kbi`` + ``.kbc/``
  (no embedder, no re-embedding).

Examples::

    remax-kb pack ./docs/ -o knowledge.kb --dim 256 --k 8 --embedder jina-onnx
    remax-kb pack ./docs/ -o knowledge.kbi --v2 --embedder gemini --gemini-dim 768
    remax-kb query knowledge.kb "How does X work?" --k 5
    remax-kb query knowledge.kbi "How does X work?" --k 5 --alpha 0.5
    remax-kb info knowledge.kbi
    remax-kb migrate knowledge.kb --out ./out/ --name knowledge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .formats import detect_format


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


# --------------------------------------------------------------------------- #
# pack
# --------------------------------------------------------------------------- #
def _cmd_pack(args: argparse.Namespace) -> int:
    if args.v2:
        return _cmd_pack_v2(args)

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
        codec=args.codec,
        bits=args.bits,
        source_description=args.source,
        batch_size=args.batch_size,
    )
    size = out.stat().st_size
    print(f"wrote {out} ({size / 1024:.1f} KB)")
    return 0


def _cmd_pack_v2(args: argparse.Namespace) -> int:
    from .pack import walk_directory
    from .pack_v2 import KBWriter

    out_path = Path(args.out)
    name = out_path.stem
    output_dir = out_path.parent if out_path.parent != Path("") else Path(".")

    embedder = _build_embedder(args.embedder, args)
    chunks = walk_directory(args.corpus, pattern=args.pattern)
    if not chunks:
        print("no chunks produced from corpus", file=sys.stderr)
        return 1

    writer = KBWriter.create(
        name=name,
        output_dir=output_dir,
        embedder=embedder,
        dim=args.dim,
        k=args.k,
        seed=args.seed,
        codec=args.codec,
        bits=args.bits,
        source=args.source,
    )
    writer.add_chunks(chunks)
    writer.commit()

    kbi = output_dir / f"{name}.kbi"
    kbc = output_dir / f"{name}.kbc"
    size = kbi.stat().st_size
    print(f"wrote {kbi} ({size / 1024:.1f} KB) + {kbc}/ ({len(chunks)} chunks)")
    return 0


# --------------------------------------------------------------------------- #
# sync — incremental v2 (re)build
# --------------------------------------------------------------------------- #
def _cmd_sync(args: argparse.Namespace) -> int:
    from .pack import walk_directory
    from .pack_v2 import KBWriter

    out_path = Path(args.out)
    if out_path.suffix != ".kbi":
        print(f"sync targets a v2 .kbi; got {out_path.name!r}", file=sys.stderr)
        return 2
    name = out_path.stem
    output_dir = out_path.parent if out_path.parent != Path("") else Path(".")

    chunks = walk_directory(args.corpus, pattern=args.pattern)
    if not chunks:
        print("no chunks produced from corpus", file=sys.stderr)
        return 1

    embedder = _build_embedder(args.embedder, args)
    if out_path.exists():
        writer = KBWriter.open(name=name, output_dir=output_dir, embedder=embedder)
    else:
        writer = KBWriter.create(
            name=name,
            output_dir=output_dir,
            embedder=embedder,
            dim=args.dim,
            k=args.k,
            seed=args.seed,
            source=args.source,
        )

    stats = writer.sync(chunks)
    writer.commit()

    compacted = False
    if not args.no_compact and writer.should_compact(args.compact_threshold):
        writer.compact()
        compacted = True

    payload = {
        "kb": str(out_path.resolve()),
        "added": stats.added,
        "updated": stats.updated,
        "deleted": stats.deleted,
        "unchanged": stats.unchanged,
        "embedded": stats.embedded,
        "live_count": writer.live_count,
        "total_rows": writer.total_rows,
        "tombstone_ratio": round(writer.tombstone_ratio, 4),
        "compacted": compacted,
    }
    print(json.dumps(payload, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #
def _cmd_query(args: argparse.Namespace) -> int:
    fmt = detect_format(args.kb)
    if fmt == "2":
        return _cmd_query_v2(args)

    from . import KB

    embedder = _build_embedder(args.embedder, args)
    kb = KB.open(args.kb)
    hits = kb.search(args.query, embedder=embedder, k=args.k)
    payload = {
        "kb": str(Path(args.kb).resolve()),
        "spec_version": "1",
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


def _cmd_query_v2(args: argparse.Namespace) -> int:
    from .read_v2 import KB as KBv2

    embedder = _build_embedder(args.embedder, args)
    kb = KBv2.open(args.kb)
    hits = kb.search_and_fetch(args.query, embedder=embedder, k=args.k, alpha=args.alpha)
    payload = {
        "kb": str(Path(args.kb).resolve()),
        "spec_version": "2",
        "query": args.query,
        "fusion": "weighted" if args.alpha is not None else "rrf",
        "hits": [
            {
                "id": h.chunk_id,
                "dense_distance": h.dense_dist,
                "dense_sim": h.dense_sim,
                "bm25_score": h.bm25_score,
                "fused": h.fused,
                "verified": h.verified,
                "text": h.text,
                "meta": h.meta,
            }
            for h in hits
        ],
    }
    print(json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #
def _cmd_info(args: argparse.Namespace) -> int:
    fmt = detect_format(args.kb)
    if fmt == "2":
        return _cmd_info_v2(args)

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
            "bits": m.binarizer.bits,
            "seed": m.binarizer.seed,
            "bytes_per_row": m.bytes_per_row(),
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_info_v2(args: argparse.Namespace) -> int:
    from .read_v2 import KB as KBv2

    kb = KBv2.open(args.kb)
    m = kb.manifest
    payload = {
        "path": str(Path(args.kb).resolve()),
        "spec_version": m["spec_version"],
        "version": m.get("version"),
        "kind": m.get("kind"),
        "built_at": m.get("built_at"),
        "source": m.get("source", ""),
        "chunks": {
            "live_count": m["chunks"]["live_count"],
            "total_rows": m["chunks"]["total_rows"],
            "shard_count": m["chunks"]["shard_count"],
            "merkle_root": m["chunks"]["merkle_root"],
        },
        "embedder": {
            "model_id": m["embedder"]["model_id"],
            "task_adapter": m["embedder"]["task_adapter"],
            "pooling": m["embedder"]["pooling"],
            "full_dim": m["embedder"]["full_dim"],
            "release_url": m["embedder"].get("release_url"),
        },
        "binarizer": {
            "kind": m["binarizer"]["kind"],
            "dim": m["binarizer"]["dim"],
            "k": m["binarizer"]["k"],
            "bits": m["binarizer"].get("bits", 1),
            "seed": m["binarizer"]["seed"],
        },
        "lexical": m.get("lexical"),
    }
    print(json.dumps(payload, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# migrate
# --------------------------------------------------------------------------- #
def _cmd_migrate(args: argparse.Namespace) -> int:
    from .migrate import migrate_v1_to_v2

    fmt = detect_format(args.kb)
    if fmt != "1":
        print(f"{args.kb} is already spec v{fmt}; nothing to migrate", file=sys.stderr)
        return 1

    kbi, kbc = migrate_v1_to_v2(
        args.kb,
        args.out,
        name=args.name,
        shard_max_bytes=args.shard_max_bytes,
    )
    size = kbi.stat().st_size
    shard_count = len(sorted(kbc.glob("shard-*.bin"))) if kbc.exists() else 0
    print(f"wrote {kbi} ({size / 1024:.1f} KB) + {kbc}/ ({shard_count} shard(s))")
    return 0


# --------------------------------------------------------------------------- #
# argument wiring
# --------------------------------------------------------------------------- #
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

    pack_p = sub.add_parser("pack", help="pack a directory into a .kb / .kbi")
    pack_p.add_argument("corpus", help="directory of documents (or a single file)")
    pack_p.add_argument("-o", "--out", required=True, help="destination .kb (v1) or .kbi (v2) path")
    pack_p.add_argument("--v2", action="store_true", help="emit a split .kbi + .kbc/ (spec v2)")
    pack_p.add_argument("--pattern", default="**/*", help='glob (default: "**/*")')
    pack_p.add_argument("--dim", type=int, default=256, help="binarizer dim (default 256)")
    pack_p.add_argument("--k", type=int, default=8, help="stacked-SimHash stack count (default 8; remax codec only)")
    pack_p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    pack_p.add_argument("--codec", choices=("remax", "remex"), default="remax",
                        help="vector codec: remax 1-bit SimHash (default) or remex multi-bit Lloyd-Max (v1 and v2)")
    pack_p.add_argument("--bits", type=int, default=4,
                        help="bits/coord for --codec remex (1..8, default 4)")
    pack_p.add_argument("--source", default="", help="free-text source description")
    pack_p.add_argument("--batch-size", type=int, default=16, help="(v1 only) embed batch size")
    _embedder_args(pack_p)
    pack_p.set_defaults(func=_cmd_pack)

    sync_p = sub.add_parser(
        "sync", help="incrementally (re)build a v2 .kbi from a directory (embeds only the delta)"
    )
    sync_p.add_argument("corpus", help="directory of documents (or a single file)")
    sync_p.add_argument("-o", "--out", required=True, help="destination .kbi (created if absent)")
    sync_p.add_argument("--pattern", default="**/*", help='glob (default: "**/*")')
    sync_p.add_argument("--dim", type=int, default=256, help="(create only) binarizer dim")
    sync_p.add_argument("--k", type=int, default=8, help="(create only) stacked-SimHash stack count")
    sync_p.add_argument("--seed", type=int, default=0, help="(create only) RNG seed")
    sync_p.add_argument("--source", default="", help="(create only) free-text source description")
    sync_p.add_argument(
        "--compact-threshold", type=float, default=0.2,
        help="compact after sync when tombstone ratio exceeds this (default 0.2)",
    )
    sync_p.add_argument(
        "--no-compact", action="store_true", help="never auto-compact, regardless of ratio",
    )
    _embedder_args(sync_p)
    sync_p.set_defaults(func=_cmd_sync)

    q_p = sub.add_parser("query", help="query a .kb / .kbi (format auto-detected)")
    q_p.add_argument("kb", help="path to .kb (v1) or .kbi (v2)")
    q_p.add_argument("query", help="user query string")
    q_p.add_argument("--k", type=int, default=5, help="number of results")
    q_p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="(v2 only) fusion weight; omit for RRF, set 0..1 for weighted dense/lexical",
    )
    q_p.add_argument("--pretty", action="store_true", help="indent JSON output")
    _embedder_args(q_p)
    q_p.set_defaults(func=_cmd_query)

    i_p = sub.add_parser("info", help="print manifest summary (format auto-detected)")
    i_p.add_argument("kb", help="path to .kb (v1) or .kbi (v2)")
    i_p.set_defaults(func=_cmd_info)

    mig_p = sub.add_parser("migrate", help="upgrade a v1 .kb to a v2 .kbi + .kbc/")
    mig_p.add_argument("kb", help="path to the source v1 .kb")
    mig_p.add_argument("--out", required=True, help="output directory for <name>.kbi + <name>.kbc/")
    mig_p.add_argument("--name", default=None, help="artifact base name (default: source stem)")
    mig_p.add_argument(
        "--shard-max-bytes",
        type=int,
        default=None,
        help="shard rotation cap in bytes (default: 20 MiB)",
    )
    mig_p.set_defaults(func=_cmd_migrate)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
