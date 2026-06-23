"""v2 packer: corpus → `<name>.kbi` (zip) + `<name>.kbc/` (chunk shards).

Implements the writer half of SPEC_v2.md. Supports green-field pack
(`KBWriter.create() + add_chunks() + commit()`) and append-only
mutation (`KBWriter.open() + add_chunks() + commit()`).

Update / delete / compact are scaffolded but deferred to v2.1.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import shutil
import struct
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import bm25s
import numpy as np

from .pack import Chunk  # reuse v1 Chunk dataclass


SPEC_VERSION = "2"
KIND = "split-index"
BINARIZER_KIND = "remax-centered-simhash"

ROW_BYTES_CHUNK_MAP = 24
FLAG_TOMBSTONE = 0x01

DEFAULT_SHARD_MAX_BYTES = 20 * 1024 * 1024  # 20 MiB — fits Pages cap
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75


@dataclass
class SyncStats:
    """Outcome of :meth:`KBWriter.sync` — the staged delta, before commit.

    ``embedded`` is the count of chunks the following ``commit()`` will run
    through the embedder (adds + updates); ``unchanged`` chunks are skipped
    entirely, which is the whole point of content-addressed sync.
    """
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0

    @property
    def embedded(self) -> int:
        return self.added + self.updated

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.deleted)


@dataclass
class _Row:
    """In-memory representation of one chunk_map row + its chunk identity."""
    chunk: Chunk
    shard_id: int
    byte_offset: int
    byte_length: int
    chunk_id_offset: int
    flags: int = 0

    def pack(self) -> bytes:
        return struct.pack(
            "<HBBQIQ",
            self.shard_id,
            self.flags,
            0,  # reserved
            self.byte_offset,
            self.byte_length,
            self.chunk_id_offset,
        )


class KBWriter:
    """Writer for v2 `.kbi` + `.kbc/` artifacts."""

    def __init__(
        self,
        *,
        name: str,
        output_dir: Path,
        embedder,
        dim: int = 256,
        k: int = 8,
        seed: int = 0,
        shard_max_bytes: int = DEFAULT_SHARD_MAX_BYTES,
        bm25_k1: float = DEFAULT_BM25_K1,
        bm25_b: float = DEFAULT_BM25_B,
        chunks_uri: str | None = None,
        source: str = "",
        rotations_quant: str = "float32",
    ):
        if dim % 8 != 0:
            raise ValueError(f"dim must be a multiple of 8, got {dim}")
        if rotations_quant not in ("float32", "int8"):
            raise ValueError(f"rotations_quant must be 'float32' or 'int8', got {rotations_quant!r}")
        self._name = name
        self._output_dir = Path(output_dir)
        self._embedder = embedder
        self._dim = dim
        self._k = k
        self._seed = seed
        self._rotations_quant = rotations_quant
        self._quantizer_obj = None       # cached StackedSignBitQuantizer
        self._rotations_i8 = None        # (codes_i8, scale_f32) when int8
        self._shard_max = shard_max_bytes
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._chunks_uri = chunks_uri  # absolute URI; if None, relative
        self._source = source

        # Mutation state
        self._rows: list[_Row] = []         # all rows including tombstones
        self._vectors: list[np.ndarray] = [] # parallel to _rows, (rowBytes,) uint8
        self._mean_vector: np.ndarray | None = None  # frozen at first commit
        self._chunk_ids_bytes = bytearray()
        self._version = 0
        self._committed = False

        # Pending operations not yet committed
        self._pending_adds: list[Chunk] = []
        self._pending_deletes: set[str] = set()  # chunk_ids to tombstone

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def create(cls, **kwargs) -> "KBWriter":
        """Fresh writer. Output paths must not already exist."""
        w = cls(**kwargs)
        if w._kbi_path.exists():
            raise FileExistsError(f"{w._kbi_path} already exists; use open() for mutation")
        if w._kbc_dir.exists():
            raise FileExistsError(f"{w._kbc_dir} already exists")
        return w

    @classmethod
    def open(cls, *, name: str, output_dir: Path, embedder, **kwargs) -> "KBWriter":
        """Load an existing `.kbi` for mutation."""
        w = cls(name=name, output_dir=output_dir, embedder=embedder, **kwargs)
        if not w._kbi_path.exists():
            raise FileNotFoundError(w._kbi_path)
        w._load_state()
        return w

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    @property
    def _kbi_path(self) -> Path:
        return self._output_dir / f"{self._name}.kbi"

    @property
    def _kbc_dir(self) -> Path:
        return self._output_dir / f"{self._name}.kbc"

    def _shard_path(self, shard_id: int) -> Path:
        return self._kbc_dir / f"shard-{shard_id:04d}.bin"

    # ------------------------------------------------------------------ #
    # Corpus state / compaction policy
    # ------------------------------------------------------------------ #
    @property
    def total_rows(self) -> int:
        """Rows on disk including tombstones (committed + applied)."""
        return len(self._rows)

    @property
    def live_count(self) -> int:
        """Non-tombstoned rows."""
        return sum(1 for r in self._rows if not (r.flags & FLAG_TOMBSTONE))

    @property
    def tombstone_count(self) -> int:
        return self.total_rows - self.live_count

    @property
    def tombstone_ratio(self) -> float:
        """Fraction of rows that are tombstones; 0.0 for an empty writer."""
        return self.tombstone_count / self.total_rows if self.total_rows else 0.0

    def should_compact(self, max_tombstone_ratio: float = 0.2) -> bool:
        """Whether dead weight warrants a compaction.

        Incremental sync never reclaims tombstoned bytes and keeps reusing
        the frozen mean, so both the artifact size and the centering drift
        away from the live corpus over time. Compaction fixes both at the
        cost of a full re-embed — worth it once the tombstone ratio crosses
        ``max_tombstone_ratio``.
        """
        return self.total_rows > 0 and self.tombstone_ratio > max_tombstone_ratio

    # ------------------------------------------------------------------ #
    # Mutation API
    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Stage chunks for the next commit."""
        for c in chunks:
            if not isinstance(c, Chunk):
                raise TypeError(f"expected Chunk, got {type(c)}")
            self._pending_adds.append(c)

    def delete_chunks(self, chunk_ids: Iterable[str]) -> None:
        """Stage tombstones for the next commit."""
        for cid in chunk_ids:
            self._pending_deletes.add(cid)

    def update_chunks(self, replacements: dict[str, Chunk]) -> None:
        """Tombstone old, append new (with new chunk_id)."""
        for old_id, new_chunk in replacements.items():
            self._pending_deletes.add(old_id)
            self._pending_adds.append(new_chunk)

    def sync(self, chunks: Iterable[Chunk]) -> SyncStats:
        """Reconcile the live corpus to *exactly* ``chunks`` (content-addressed).

        Diffs the desired chunk set against the current live rows by
        ``(id, sha256)`` and stages only the delta:

          * id absent from live → **add** (embedded on commit)
          * id present, sha256 differs → **update** (tombstone + re-embed)
          * id present, sha256 identical → **unchanged** (skipped — no embed)
          * live id absent from desired → **delete** (tombstone, no embed)

        Staging only; call :meth:`commit` to apply. The returned
        :class:`SyncStats` reports the delta and how many chunks the next
        commit will embed. This is the efficient path for a living corpus:
        a typical edit re-embeds a handful of chunks instead of all of them.
        """
        live: dict[str, str] = {}
        for row in self._rows:
            if row.flags & FLAG_TOMBSTONE:
                continue
            live[row.chunk.id] = row.chunk.sha256  # later live row wins

        desired: dict[str, Chunk] = {}
        for c in chunks:
            if not isinstance(c, Chunk):
                raise TypeError(f"expected Chunk, got {type(c)}")
            desired[c.id] = c  # last occurrence wins on duplicate id

        adds: list[Chunk] = []
        updates: dict[str, Chunk] = {}
        unchanged = 0
        for cid, chunk in desired.items():
            if cid not in live:
                adds.append(chunk)
            elif live[cid] != chunk.sha256:
                updates[cid] = chunk
            else:
                unchanged += 1

        deletes = [cid for cid in live if cid not in desired]

        if adds:
            self.add_chunks(adds)
        if updates:
            self.update_chunks(updates)
        if deletes:
            self.delete_chunks(deletes)

        return SyncStats(
            added=len(adds),
            updated=len(updates),
            deleted=len(deletes),
            unchanged=unchanged,
        )

    # ------------------------------------------------------------------ #
    # Commit
    # ------------------------------------------------------------------ #
    def commit(self) -> None:
        """Apply pending mutations atomically.

        Pipeline:
          1. Embed any pending adds (document prompt).
          2. If no mean yet (first commit), compute from pending embeddings.
             Else reuse frozen mean.
          3. Center, truncate, stacked-SimHash encode pending adds.
          4. Append shard bytes and chunk_ids; build new _Row entries.
          5. Apply pending deletes by flipping tombstone flag.
          6. Rebuild BM25 over LIVE rows.
          7. Compute Merkle root.
          8. Write `.kbi` (zip) + append-only shard writes.
        """
        if not self._pending_adds and not self._pending_deletes:
            return  # no-op

        new_embeddings: np.ndarray | None = None
        new_codes: np.ndarray | None = None

        if self._pending_adds:
            new_embeddings = self._embedder.encode(
                [c.text for c in self._pending_adds], prompt="document"
            ).astype(np.float32)
            if new_embeddings.shape[1] != self._embedder.full_dim:
                raise ValueError(
                    f"embedder returned dim {new_embeddings.shape[1]}; "
                    f"expected {self._embedder.full_dim}"
                )

            if self._mean_vector is None:
                self._mean_vector = new_embeddings.mean(axis=0).astype(np.float32)

            centered = new_embeddings - self._mean_vector
            truncated = centered[:, : self._dim].astype(np.float32)
            new_codes = self._get_quantizer().encode(truncated)  # (N, rowBytes) uint8

            # Append shard bytes + chunk_ids + rows
            self._append_pending(new_codes)

        # Apply tombstones
        if self._pending_deletes:
            self._apply_tombstones()

        # Clear pending
        self._pending_adds = []
        self._pending_deletes = set()
        self._version += 1

        # Materialize everything
        self._write_artifacts()
        self._committed = True

    # ------------------------------------------------------------------ #
    # Migration ingest (pre-computed codes, no embedder)
    # ------------------------------------------------------------------ #
    def ingest_precoded(
        self,
        chunks: list[Chunk],
        codes: np.ndarray,
        *,
        mean_vector: np.ndarray,
    ) -> None:
        """Materialize a green-field `.kbi` + `.kbc/` from chunks whose binary
        codes were computed elsewhere — e.g. the v1→v2 migration path, where
        v1's ``vectors.bin`` is reused verbatim instead of re-embedding.

        ``codes`` must be ``(N, row_bytes)`` uint8 consistent with this
        writer's ``(dim, k, seed)``, in the same row order as ``chunks``.
        ``mean_vector`` is the frozen corpus mean carried over from the source
        artifact. Single-shot: writes the artifacts and marks the writer
        committed. Only valid on a fresh writer (no prior rows/commit).
        """
        if self._committed or self._rows:
            raise RuntimeError("ingest_precoded() is only valid on a fresh writer")
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
        if codes.ndim != 2 or codes.shape[0] != len(chunks):
            raise ValueError(
                f"codes shape {codes.shape} does not match {len(chunks)} chunks"
            )
        expected_row_bytes = self._dim * self._k // 8
        if codes.shape[1] != expected_row_bytes:
            raise ValueError(
                f"codes row width {codes.shape[1]} != dim*k/8 = {expected_row_bytes}"
            )
        if not chunks:
            raise ValueError("ingest_precoded() called with no chunks")

        self._mean_vector = np.ascontiguousarray(mean_vector, dtype="<f4")
        self._pending_adds = list(chunks)
        self._append_pending(codes)
        self._pending_adds = []
        self._version += 1
        self._write_artifacts()
        self._committed = True

    # ------------------------------------------------------------------ #
    # Compact (v2.1)
    # ------------------------------------------------------------------ #
    def compact(self) -> None:
        """Rebuild from live rows, dropping tombstones and orphaned bytes.

        Recomputes the corpus mean as a side effect (so v1-style
        determinism is restored after compaction).
        """
        live_chunks = [r.chunk for r in self._rows if not (r.flags & FLAG_TOMBSTONE)]
        if not live_chunks:
            raise ValueError("compact() called with no live chunks")

        # Reset state and re-add as a fresh pack
        self._rows = []
        self._vectors = []
        self._mean_vector = None
        self._chunk_ids_bytes = bytearray()
        # Remove existing shards
        if self._kbc_dir.exists():
            shutil.rmtree(self._kbc_dir)
        self._kbc_dir.mkdir(parents=True, exist_ok=True)

        self._pending_adds = list(live_chunks)
        self.commit()

    # ------------------------------------------------------------------ #
    # Internal: appending pending chunks
    # ------------------------------------------------------------------ #
    def _append_pending(self, new_codes: np.ndarray) -> None:
        for chunk, code in zip(self._pending_adds, new_codes):
            shard_bytes = self._render_chunk_bytes(chunk)
            shard_id, byte_offset = self._reserve_shard_space(len(shard_bytes))
            self._write_shard_bytes(shard_id, byte_offset, shard_bytes)

            chunk_id_offset = len(self._chunk_ids_bytes)
            self._chunk_ids_bytes.extend(chunk.id.encode("utf-8"))
            self._chunk_ids_bytes.append(0)  # NUL terminator

            self._rows.append(_Row(
                chunk=chunk,
                shard_id=shard_id,
                byte_offset=byte_offset,
                byte_length=len(shard_bytes),
                chunk_id_offset=chunk_id_offset,
                flags=0,
            ))
            self._vectors.append(code.astype(np.uint8))

    def _render_chunk_bytes(self, chunk: Chunk) -> bytes:
        header = json.dumps(
            {"sha256": chunk.sha256, "meta": chunk.meta},
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return header + b"\n" + chunk.text.encode("utf-8")

    def _reserve_shard_space(self, n_bytes: int) -> tuple[int, int]:
        """Return (shard_id, byte_offset) for a chunk of n_bytes.

        Rolls to a new shard if writing to the current one would exceed
        shard_max_bytes.
        """
        self._kbc_dir.mkdir(parents=True, exist_ok=True)
        # Find current shard
        existing = sorted(self._kbc_dir.glob("shard-*.bin"))
        if not existing:
            shard_id = 0
            offset = 0
        else:
            shard_id = int(existing[-1].stem.split("-")[1])
            offset = existing[-1].stat().st_size

        if offset + n_bytes > self._shard_max and offset > 0:
            shard_id += 1
            offset = 0
        return shard_id, offset

    def _write_shard_bytes(self, shard_id: int, offset: int, data: bytes) -> None:
        path = self._shard_path(shard_id)
        with open(path, "ab") as f:
            if f.tell() != offset:
                # Idempotency guard — should always match for append-only
                raise RuntimeError(
                    f"shard {shard_id} offset mismatch: expected {offset}, got {f.tell()}"
                )
            f.write(data)

    def _apply_tombstones(self) -> None:
        """Flip tombstone flag on the latest live row matching each pending id."""
        for cid in self._pending_deletes:
            # Walk in reverse to find the most recent live row with this id
            for row in reversed(self._rows):
                if row.flags & FLAG_TOMBSTONE:
                    continue
                if row.chunk.id == cid:
                    row.flags |= FLAG_TOMBSTONE
                    break
            else:
                raise ValueError(f"delete: chunk_id {cid!r} not found among live rows")

    # ------------------------------------------------------------------ #
    # Internal: write artifacts
    # ------------------------------------------------------------------ #
    def _get_quantizer(self):
        """StackedSignBitQuantizer used to encode docs AND queries, cached so
        ``commit()`` and ``_write_artifacts()`` share one rotation set. When
        ``rotations_quant == 'int8'`` its rotations are the int8-dequantized
        form, so the corpus codes and the shipped int8 rotations are
        bit-consistent (a reader that dequantizes lands in the same sign-space).
        """
        if self._quantizer_obj is None:
            from remax import StackedSignBitQuantizer
            q = StackedSignBitQuantizer(d=self._dim, k=self._k, seed=self._seed)
            if self._rotations_quant == "int8":
                from .rotations import quantize_int8, dequantize_int8
                codes_i8, scale_f32 = quantize_int8(q.rotations_.astype(np.float32))
                q.rotations_ = dequantize_int8(codes_i8, scale_f32).astype(q.dtype)
                self._rotations_i8 = (codes_i8, scale_f32)
            self._quantizer_obj = q
        return self._quantizer_obj

    def _write_artifacts(self) -> None:
        vectors_bytes = b"".join(v.tobytes() for v in self._vectors)
        chunk_map_bytes = b"".join(r.pack() for r in self._rows)
        chunk_ids_bytes = bytes(self._chunk_ids_bytes)

        # BM25 over live rows
        live_indices = [i for i, r in enumerate(self._rows) if not (r.flags & FLAG_TOMBSTONE)]
        live_texts = [self._rows[i].chunk.text for i in live_indices]
        bm25_files = self._build_bm25(live_texts) if live_texts else None

        # Merkle root over per-chunk sha256, leaves in row-index order
        leaves = [bytes.fromhex(r.chunk.sha256) for r in self._rows]
        merkle_root = _merkle_root(leaves)

        manifest = self._build_manifest(
            total_rows=len(self._rows),
            live_count=len(live_indices),
            merkle_root=merkle_root,
            has_bm25=bm25_files is not None,
        )

        # Write zip atomically (write to temp, then replace)
        tmp_path = self._kbi_path.with_suffix(".kbi.tmp")
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("vectors.bin", vectors_bytes)
            zf.writestr("chunk_map.bin", chunk_map_bytes)
            zf.writestr("chunk_ids.bin", chunk_ids_bytes)
            # Pre-computed Haar rotations — see SPEC_v2 §binarizer/rotations.
            # f32 by default; int8 + per-column scale when rotations_quant set.
            q = self._get_quantizer()
            if self._rotations_quant == "int8":
                codes_i8, scale_f32 = self._rotations_i8
                zf.writestr("binarizer/rotations.i8",
                            np.ascontiguousarray(codes_i8, dtype=np.int8).tobytes())
                zf.writestr("binarizer/rotations.scale.f32",
                            np.ascontiguousarray(scale_f32, dtype="<f4").tobytes())
            else:
                zf.writestr("binarizer/rotations.f32",
                            q.rotations_.astype("<f4").tobytes())
            if bm25_files is not None:
                for name, data in bm25_files.items():
                    zf.writestr(f"bm25/{name}", data)
        tmp_path.replace(self._kbi_path)

    def _build_bm25(self, live_texts: list[str]) -> dict[str, bytes]:
        retriever = bm25s.BM25(k1=self._bm25_k1, b=self._bm25_b)
        tokens = bm25s.tokenize(live_texts, stopwords=None)
        retriever.index(tokens)
        with tempfile.TemporaryDirectory() as d:
            retriever.save(d, corpus=None)
            out = {}
            for p in sorted(Path(d).iterdir()):
                # Map output files into bm25/ subdir; ensure CSC + json present
                out[p.name] = p.read_bytes()
            return out

    def _build_manifest(
        self,
        *,
        total_rows: int,
        live_count: int,
        merkle_root: str,
        has_bm25: bool,
    ) -> dict[str, Any]:
        fp = self._embedder.fingerprint()
        embedder_block: dict[str, Any] = {
            "model_id": fp["model_id"],
            "model_revision": getattr(self._embedder, "model_revision", ""),
            "task_adapter": fp["task_adapter"],
            "pooling": fp["pooling"],
            "normalize_l2": getattr(self._embedder, "normalize_l2", True),
            "full_dim": fp["full_dim"],
        }
        release_url = getattr(self._embedder, "release_url", None)
        release_sha256 = getattr(self._embedder, "release_sha256", None)
        if release_url:
            embedder_block["release_url"] = release_url
            embedder_block["release_sha256"] = release_sha256
        else:
            embedder_block["release_url"] = None
            embedder_block["release_sha256"] = None

        prompts = getattr(self._embedder, "prompts", {"query": "Query: ", "document": "Document: "})

        chunks_uri = self._chunks_uri
        if chunks_uri is None:
            # Relative URI: "<name>.kbc/"
            chunks_uri = f"{self._name}.kbc/"
        elif not chunks_uri.endswith("/"):
            chunks_uri = chunks_uri + "/"

        # Count existing shards to set shard_count
        shards = sorted(self._kbc_dir.glob("shard-*.bin")) if self._kbc_dir.exists() else []
        manifest: dict[str, Any] = {
            "spec_version": SPEC_VERSION,
            "version": self._version,
            "kind": KIND,
            "embedder": embedder_block,
            "prompts": prompts,
            "binarizer": {
                "kind": BINARIZER_KIND,
                "remax_version": _remax_version(),
                "dim": self._dim,
                "k": self._k,
                "seed": self._seed,
                "rotations_quant": self._rotations_quant,
                "mean_vector_b64": _np_to_b64(self._mean_vector),
            },
            "chunks": {
                "uri": chunks_uri,
                "shard_count": len(shards),
                "shard_max_bytes": self._shard_max,
                "live_count": live_count,
                "total_rows": total_rows,
                "merkle_root": merkle_root,
            },
            "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": self._source,
        }
        if has_bm25:
            manifest["lexical"] = {
                "kind": "bm25s",
                "library_version": bm25s.__version__ if hasattr(bm25s, "__version__") else "0.3.9",
                "k1": self._bm25_k1,
                "b": self._bm25_b,
                "tokenizer": "bm25s.default",
                "stopwords": None,
            }
        return manifest

    # ------------------------------------------------------------------ #
    # Load existing state for mutation
    # ------------------------------------------------------------------ #
    def _load_state(self) -> None:
        """Hydrate _rows, _vectors, _mean_vector, _chunk_ids_bytes, _version
        from an existing .kbi."""
        with zipfile.ZipFile(self._kbi_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            vectors = np.frombuffer(zf.read("vectors.bin"), dtype=np.uint8)
            chunk_map = zf.read("chunk_map.bin")
            chunk_ids = zf.read("chunk_ids.bin")

        if manifest["spec_version"] != SPEC_VERSION:
            raise ValueError(f"unsupported spec_version {manifest['spec_version']!r}")
        if manifest["kind"] != KIND:
            raise ValueError(f"unsupported kind {manifest['kind']!r}")

        row_bytes = manifest["binarizer"]["dim"] * manifest["binarizer"]["k"] // 8
        total = manifest["chunks"]["total_rows"]
        vectors = vectors.reshape(total, row_bytes)
        self._dim = manifest["binarizer"]["dim"]
        self._k = manifest["binarizer"]["k"]
        self._seed = manifest["binarizer"]["seed"]
        # Preserve rotation quantization across mutation re-commits.
        self._rotations_quant = manifest["binarizer"].get("rotations_quant", "float32")
        self._mean_vector = _b64_to_np(manifest["binarizer"]["mean_vector_b64"])
        self._version = manifest["version"]
        self._chunk_ids_bytes = bytearray(chunk_ids)

        for i in range(total):
            o = i * ROW_BYTES_CHUNK_MAP
            shard_id, flags, _resv, byte_offset, byte_length, chunk_id_offset = struct.unpack(
                "<HBBQIQ", chunk_map[o : o + ROW_BYTES_CHUNK_MAP]
            )
            chunk_id = _read_null_terminated(chunk_ids, chunk_id_offset)

            # Pull chunk text from the shard
            text, meta, _sha = _read_chunk_from_shard(
                self._shard_path(shard_id), byte_offset, byte_length
            )
            chunk = Chunk(id=chunk_id, text=text, meta=meta)
            self._rows.append(_Row(
                chunk=chunk, shard_id=shard_id, byte_offset=byte_offset,
                byte_length=byte_length, chunk_id_offset=chunk_id_offset, flags=flags,
            ))
            self._vectors.append(vectors[i].copy())


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _np_to_b64(arr: np.ndarray) -> str:
    import base64
    return base64.b64encode(arr.astype("<f4").tobytes()).decode("ascii")


def _b64_to_np(s: str) -> np.ndarray:
    import base64
    return np.frombuffer(base64.b64decode(s), dtype="<f4").copy()


def _read_null_terminated(buf: bytes, offset: int) -> str:
    end = buf.index(0, offset)
    return buf[offset:end].decode("utf-8")


def _read_chunk_from_shard(shard_path: Path, byte_offset: int, byte_length: int):
    with open(shard_path, "rb") as f:
        f.seek(byte_offset)
        data = f.read(byte_length)
    nl = data.index(b"\n")
    header = json.loads(data[:nl].decode("utf-8"))
    text = data[nl + 1 :].decode("utf-8")
    return text, header.get("meta", {}), header["sha256"]


def _merkle_root(leaves: list[bytes]) -> str:
    """Binary Merkle root. Empty corpus → all-zeros."""
    if not leaves:
        return "0" * 64
    level = list(leaves)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(hashlib.sha256(left + right).digest())
        level = nxt
    return level[0].hex()


def _remax_version() -> str:
    try:
        import remax
        return getattr(remax, "__version__", "0.0.0")
    except Exception:
        return "0.0.0"
