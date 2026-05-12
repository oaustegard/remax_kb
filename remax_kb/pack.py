"""Packer: corpus → .kb file.

Pipeline:
  1. Walk / chunk the corpus
  2. Embed each chunk (document prompt, full_dim)
  3. Compute corpus mean (full_dim) — bake into manifest
  4. Center + truncate to (dim) + StackedSimHash encode
  5. Pack everything into a zip
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .handlers import DEFAULT_HANDLERS, FileHandler
from .manifest import (
    BINARIZER_KIND,
    Binarizer,
    CorpusInfo,
    Embedder as EmbedderField,
    Manifest,
    Prompts,
    SPEC_VERSION,
)


@dataclass
class Chunk:
    id: str
    text: str
    meta: dict[str, Any]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def default_chunker(text: str, *, source_path: str, target_chars: int = 500) -> list[Chunk]:
    """Naive sentence-aware ~500-char chunker. Good enough for sci-fair scope.

    Splits on paragraph blank lines, then greedily packs sentences (period /
    question / exclamation boundaries) into windows of ``target_chars``.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(
                Chunk(
                    id=f"{source_path}#chunk-{len(chunks):04d}",
                    text=buf.strip(),
                    meta={"source_path": source_path},
                )
            )
        buf = ""

    for para in paragraphs:
        # Cheap sentence split: keep delimiters.
        pieces: list[str] = []
        cur = ""
        for ch in para:
            cur += ch
            if ch in ".!?":
                pieces.append(cur)
                cur = ""
        if cur:
            pieces.append(cur)

        for s in pieces:
            if not buf:
                buf = s.strip()
            elif len(buf) + len(s) + 1 <= target_chars:
                buf = f"{buf} {s.strip()}"
            else:
                flush()
                buf = s.strip()
        # End-of-paragraph: keep accumulating across paragraphs only if room.
        if len(buf) >= target_chars * 0.6:
            flush()
    flush()
    return chunks


def walk_directory(
    root: str | Path,
    *,
    handlers: dict[str, FileHandler] | None = None,
    pattern: str = "**/*",
    chunker: Callable[[str, str], list[Chunk]] | None = None,
) -> list[Chunk]:
    """Walk a directory, dispatch each file to a handler keyed by suffix,
    then chunk the extracted text.

    Args:
        root: Directory to walk. May also be a single file.
        handlers: Suffix → ``FileHandler`` map. Defaults to
            :data:`remax_kb.handlers.DEFAULT_HANDLERS`. Unknown suffixes
            are silently skipped.
        pattern: glob pattern relative to ``root`` (default: ``"**/*"``).
        chunker: Override the default chunker. Signature
            ``(text, source_path) -> list[Chunk]``. The default uses
            :func:`default_chunker` with ``target_chars=500``.

    Returns:
        Flat list of chunks across all matched files, sorted by relative
        path for determinism. Each chunk's ``meta`` is the union of the
        file handler's metadata plus the chunker's own metadata; chunker
        keys win on conflict.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    use_handlers = handlers or DEFAULT_HANDLERS
    use_chunker = chunker or (
        lambda text, source_path: default_chunker(text, source_path=source_path)
    )

    if root_path.is_file():
        files = [root_path]
    else:
        files = sorted(
            p for p in root_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in use_handlers
        )

    out: list[Chunk] = []
    for f in files:
        ext = f.suffix.lower()
        handler = use_handlers[ext]
        try:
            text, file_meta = handler(f)
        except Exception as exc:  # noqa: BLE001 — handler failures shouldn't kill the run
            import warnings as _w
            _w.warn(f"handler for {f} raised {type(exc).__name__}: {exc}; skipping.", stacklevel=2)
            continue
        if not text.strip():
            continue
        rel = (
            str(f.relative_to(root_path))
            if root_path.is_dir()
            else f.name
        )
        for chunk in use_chunker(text, rel):
            merged = {**file_meta, **chunk.meta}
            out.append(Chunk(id=chunk.id, text=chunk.text, meta=merged))
    return out


def walk_corpus(corpus_path: str | Path) -> list[Chunk]:
    """Walk a directory, read every ``*.txt`` and ``*.md``, default-chunk each."""
    root = Path(corpus_path)
    if not root.exists():
        raise FileNotFoundError(root)
    out: list[Chunk] = []
    if root.is_file():
        files = [root]
    else:
        files = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in {".txt", ".md"}
        )
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        rel = str(f.relative_to(root.parent if root.is_file() else root))
        out.extend(default_chunker(text, source_path=rel))
    return out


def _remax_version() -> str:
    try:
        import remax

        return getattr(remax, "__version__", "0.0.0")
    except ImportError:
        return "unknown"


def pack(
    corpus: str | Path | Iterable[Chunk],
    out_kb: str | Path,
    *,
    embedder,
    dim: int = 256,
    k: int = 8,
    seed: int = 0,
    chunker: Callable[[str], list[Chunk]] | None = None,
    source_description: str = "",
    batch_size: int = 16,
) -> Path:
    """Build a .kb from a corpus.

    Args:
        corpus: A directory path, a single file, or an iterable of ``Chunk``s.
        out_kb: Destination ``.kb`` path.
        embedder: Embedder exposing ``encode(texts, prompt=...)`` and
            ``fingerprint()``. Typically :class:`embedders.JinaTorchEmbedder`.
            Must also expose ``release_url`` and ``release_sha256`` attributes
            naming the *runtime* (ONNX) asset readers should fetch.
        dim: Truncation dimension. Multiple of 8, ≤ ``embedder.full_dim``.
        k: Stacked-SimHash stack count.
        seed: Master RNG seed for the stacked rotations.
        chunker: Optional override; if given and corpus is a path, called on
            each file's full text. Ignored if ``corpus`` is already an
            iterable of ``Chunk``.
        source_description: Free-text label baked into ``corpus.source``.
        batch_size: Embedder mini-batch size.

    Returns:
        Path to the written .kb.
    """
    from remax import StackedSignBitQuantizer

    # ------------------------- chunking ------------------------- #
    if isinstance(corpus, (str, Path)):
        if chunker is None:
            chunks = walk_corpus(corpus)
        else:
            chunks = []
            root = Path(corpus)
            files = (
                [root]
                if root.is_file()
                else sorted(
                    p
                    for p in root.rglob("*")
                    if p.suffix.lower() in {".txt", ".md"}
                )
            )
            for f in files:
                rel = str(f.relative_to(root.parent if root.is_file() else root))
                chunks.extend(chunker(f.read_text(encoding="utf-8")))
    else:
        chunks = list(corpus)

    if not chunks:
        raise ValueError("corpus is empty; nothing to pack")

    # ------------------------- embed ------------------------- #
    fp = embedder.fingerprint()
    full_dim = fp["full_dim"]
    if dim % 8 != 0 or dim <= 0:
        raise ValueError(f"dim must be a positive multiple of 8, got {dim}")
    if dim > full_dim:
        raise ValueError(f"dim={dim} exceeds embedder full_dim={full_dim}")

    vectors = np.empty((len(chunks), full_dim), dtype=np.float32)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        out = embedder.encode([c.text for c in batch], prompt="document")
        if out.shape != (len(batch), full_dim):
            raise RuntimeError(
                f"embedder returned shape {out.shape}; expected "
                f"({len(batch)}, {full_dim})"
            )
        vectors[start : start + len(batch)] = out.astype(np.float32, copy=False)

    # ------------------------- center + truncate + binarize ------------------------- #
    mean_full = vectors.mean(axis=0).astype(np.float32)
    centered = vectors - mean_full
    truncated = np.ascontiguousarray(centered[:, :dim])

    q = StackedSignBitQuantizer(d=dim, k=k, seed=seed)
    codes = q.encode(truncated)  # (N, dim*k//8) uint8
    bytes_per_row = (dim * k) // 8
    if codes.shape != (len(chunks), bytes_per_row):
        raise RuntimeError(
            f"unexpected codes shape {codes.shape}; expected "
            f"({len(chunks)}, {bytes_per_row})"
        )

    vectors_bin = np.ascontiguousarray(codes).tobytes()
    chunks_jsonl = (
        "\n".join(
            json.dumps(
                {
                    "id": c.id,
                    "sha256": c.sha256,
                    "text": c.text,
                    "meta": c.meta,
                },
                ensure_ascii=False,
            )
            for c in chunks
        )
        + "\n"
    ).encode("utf-8")
    build_hash = hashlib.sha256(vectors_bin + chunks_jsonl).hexdigest()

    # ------------------------- manifest ------------------------- #
    manifest = Manifest(
        spec_version=SPEC_VERSION,
        embedder=EmbedderField(
            model_id=fp["model_id"],
            model_revision=getattr(embedder, "model_revision", ""),
            release_url=getattr(embedder, "release_url", None) or None,
            release_sha256=getattr(embedder, "release_sha256", None) or None,
            task_adapter=fp.get("task_adapter", "retrieval"),
            pooling=fp["pooling"],
            normalize_l2=getattr(embedder, "normalize_l2", True),
            full_dim=full_dim,
        ),
        prompts=Prompts(
            query=getattr(embedder, "prompts", {}).get("query", "Query: "),
            document=getattr(embedder, "prompts", {}).get("document", "Document: "),
        ),
        binarizer=Binarizer.from_mean(
            remax_version=_remax_version(),
            dim=dim,
            k=k,
            seed=seed,
            mean_vector=mean_full,
        ),
        corpus=CorpusInfo(
            chunk_count=len(chunks),
            build_hash=build_hash,
            built_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            source=source_description,
        ),
    )
    manifest_bytes = manifest.to_json(indent=2).encode("utf-8")

    # ------------------------- write zip ------------------------- #
    out_kb = Path(out_kb)
    out_kb.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_kb, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json", manifest_bytes)
        zf.writestr("vectors.bin", vectors_bin)
        zf.writestr("chunks.jsonl", chunks_jsonl)
    return out_kb


def pack_directory(
    root: str | Path,
    out_kb: str | Path,
    *,
    embedder,
    handlers: dict[str, FileHandler] | None = None,
    pattern: str = "**/*",
    dim: int = 256,
    k: int = 8,
    seed: int = 0,
    source_description: str = "",
    batch_size: int = 16,
    chunker: Callable[[str, str], list[Chunk]] | None = None,
) -> Path:
    """Pack a directory of mixed-format documents into a ``.kb``.

    Convenience wrapper around :func:`walk_directory` + :func:`pack` for
    the common "I have a folder of docs" case. Default handlers cover
    ``.md / .markdown / .txt / .rst / .html / .htm / .pdf``. Override
    by passing your own ``handlers`` dict; the default registry is
    :data:`remax_kb.handlers.DEFAULT_HANDLERS`.

    Args:
        root: Directory of documents (or a single file).
        out_kb: Destination ``.kb`` path.
        embedder: Embedder exposing ``encode(texts, prompt=...)`` and
            ``fingerprint()``.
        handlers: Optional suffix → :data:`~remax_kb.handlers.FileHandler`
            map. Defaults to :data:`~remax_kb.handlers.DEFAULT_HANDLERS`.
        pattern: glob pattern (default ``"**/*"``).
        dim, k, seed: Binarizer parameters (see :func:`pack`).
        source_description: Free-text label baked into ``corpus.source``;
            defaults to the absolute path of ``root``.
        batch_size: Embedder mini-batch size.
        chunker: Optional ``(text, source_path) -> list[Chunk]`` override.

    Returns:
        Path to the written ``.kb``.
    """
    chunks = walk_directory(
        root, handlers=handlers, pattern=pattern, chunker=chunker
    )
    if not chunks:
        raise ValueError(
            f"no chunks produced from {root!r}; check handlers + pattern + corpus contents"
        )
    return pack(
        chunks,
        out_kb,
        embedder=embedder,
        dim=dim,
        k=k,
        seed=seed,
        source_description=source_description or str(Path(root).resolve()),
        batch_size=batch_size,
    )
