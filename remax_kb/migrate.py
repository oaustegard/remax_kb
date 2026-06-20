"""One-shot v1 `.kb` → v2 `.kbi` + `.kbc/` migration.

A v1 `.kb` already contains everything v2 needs except the BM25 lexical
index and the chunk-store layout:

* ``vectors.bin`` — the 1-bit codes. v2's dense codes are produced by the
  *same* ``StackedSignBitQuantizer(dim, k, seed)`` over the *same*
  centered-and-truncated embeddings, so the v1 codes are bit-identical to
  what a v2 re-pack would emit. Migration reuses them verbatim — **no
  re-embedding, no embedder, no network**.
* ``chunks.jsonl`` — id/text/meta for every chunk, which become the
  byte-addressable shard store (``.kbc/``).
* ``manifest.json`` — carries the frozen ``mean_vector``, ``(dim, k,
  seed)``, and the embedder fingerprint, all copied into the v2 manifest.

What's *built fresh* during migration: the BM25 index (v1 had none), the
``chunk_map.bin`` / ``chunk_ids.bin`` tables, the per-chunk shard headers,
the Merkle root, and ``binarizer/rotations.f32``.

The result is a compacted, tombstone-free v2 artifact — deterministic by
the v1 rule (same ``(corpus, embedder, dim, k, seed)`` → bit-identical
``vectors.bin``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .pack import Chunk
from .pack_v2 import KBWriter
from .read import KB as _KBv1


class _FrozenEmbedder:
    """Embedder shim that carries a v1 manifest's identity for v2 manifest
    construction. Never embeds — migration reuses the v1 codes."""

    def __init__(self, manifest) -> None:
        e = manifest.embedder
        self._fp = {
            "model_id": e.model_id,
            "task_adapter": e.task_adapter,
            "pooling": e.pooling,
            "full_dim": e.full_dim,
        }
        self.model_revision = e.model_revision
        self.normalize_l2 = e.normalize_l2
        self.full_dim = e.full_dim
        self.release_url = e.release_url
        self.release_sha256 = e.release_sha256
        self.prompts = {"query": manifest.prompts.query, "document": manifest.prompts.document}

    def fingerprint(self) -> dict[str, Any]:
        return dict(self._fp)

    def encode(self, *args, **kwargs):  # pragma: no cover - guard
        raise RuntimeError(
            "migration reuses pre-computed v1 codes and cannot embed"
        )


def migrate_v1_to_v2(
    v1_path: str | Path,
    output_dir: str | Path,
    *,
    name: str | None = None,
    shard_max_bytes: int | None = None,
) -> tuple[Path, Path]:
    """Migrate a v1 ``.kb`` to a v2 ``.kbi`` + ``.kbc/`` pair.

    Args:
        v1_path: Path to the source v1 ``.kb``. Opened and fully validated
            (build-hash check included) before any output is written.
        output_dir: Directory to write ``<name>.kbi`` and ``<name>.kbc/``.
        name: Artifact base name. Defaults to the v1 file's stem.
        shard_max_bytes: Optional shard rotation cap; defaults to the
            writer's 20 MiB.

    Returns:
        ``(kbi_path, kbc_dir)``.

    Raises:
        FileExistsError: if either output path already exists.
        ValueError: if the source is not a valid v1 ``.kb``.
    """
    v1 = _KBv1.open(v1_path)
    m = v1.manifest

    name = name or Path(v1_path).stem
    writer_kwargs: dict[str, Any] = dict(
        name=name,
        output_dir=Path(output_dir),
        embedder=_FrozenEmbedder(m),
        dim=m.binarizer.dim,
        k=m.binarizer.k,
        seed=m.binarizer.seed,
        source=m.corpus.source,
    )
    if shard_max_bytes is not None:
        writer_kwargs["shard_max_bytes"] = shard_max_bytes

    writer = KBWriter.create(**writer_kwargs)

    chunks = [
        Chunk(id=c["id"], text=c["text"], meta=c.get("meta", {}))
        for c in v1.chunks
    ]
    codes = np.ascontiguousarray(v1.codes, dtype=np.uint8)

    writer.ingest_precoded(
        chunks, codes, mean_vector=np.asarray(m.binarizer.mean_vector)
    )

    return writer._kbi_path, writer._kbc_dir
