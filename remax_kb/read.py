"""Reader: open a .kb, validate, and run Hamming-space top-k search."""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ._hamming import hamming_scan, top_k
from .manifest import Manifest


class Embedder(Protocol):
    """Duck-typed protocol the reader expects from an embedder.

    A conforming embedder exposes a fingerprint dict and an ``encode``
    method that takes a list of texts plus a prompt name ("query" or
    "document") and returns ``(N, full_dim)`` float32 (un-truncated,
    L2-normalized).
    """

    def fingerprint(self) -> dict[str, Any]: ...

    def encode(
        self, texts: list[str], *, prompt: str
    ) -> np.ndarray: ...


@dataclass
class _Loaded:
    manifest: Manifest
    codes: np.ndarray  # (N, bytes_per_row) uint8
    chunks: list[dict[str, Any]]


class KB:
    """An opened .kb, ready to search."""

    def __init__(self, loaded: _Loaded, *, path: Path):
        self._m = loaded.manifest
        self._codes = loaded.codes
        self._chunks = loaded.chunks
        self.path = path

    # ------------------------------------------------------------------ #
    # Construction / validation
    # ------------------------------------------------------------------ #
    @classmethod
    def open(cls, path: str | Path) -> "KB":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            for required in ("manifest.json", "vectors.bin", "chunks.jsonl"):
                if required not in names:
                    raise ValueError(
                        f"{path}: missing required entry {required!r}"
                    )
            manifest_bytes = zf.read("manifest.json")
            vectors_bytes = zf.read("vectors.bin")
            chunks_bytes = zf.read("chunks.jsonl")

        manifest = Manifest.from_json(manifest_bytes.decode("utf-8"))
        manifest.validate_static()

        bpr = manifest.bytes_per_row()
        if len(vectors_bytes) % bpr != 0:
            raise ValueError(
                f"vectors.bin length {len(vectors_bytes)} not a multiple of "
                f"bytes_per_row={bpr}"
            )
        n_rows = len(vectors_bytes) // bpr

        chunks = [
            json.loads(line)
            for line in chunks_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]

        if not (n_rows == len(chunks) == manifest.corpus.chunk_count):
            raise ValueError(
                f"chunk_count mismatch: manifest={manifest.corpus.chunk_count}, "
                f"vectors.bin rows={n_rows}, chunks.jsonl lines={len(chunks)}"
            )

        computed_hash = hashlib.sha256(vectors_bytes + chunks_bytes).hexdigest()
        if computed_hash != manifest.corpus.build_hash:
            raise ValueError(
                f"build_hash mismatch: manifest={manifest.corpus.build_hash}, "
                f"computed={computed_hash}"
            )

        codes = np.frombuffer(vectors_bytes, dtype=np.uint8).reshape(n_rows, bpr)
        # Ensure writable=False is fine; numpy ops we use don't need writes.
        codes = np.ascontiguousarray(codes)

        return cls(_Loaded(manifest=manifest, codes=codes, chunks=chunks), path=path)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def manifest(self) -> Manifest:
        return self._m

    @property
    def chunks(self) -> list[dict[str, Any]]:
        return self._chunks

    @property
    def codes(self) -> np.ndarray:
        return self._codes

    def __len__(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def encode_query(
        self, query: str, *, embedder: Embedder
    ) -> np.ndarray:
        """Embed + center + truncate + stacked-SimHash a single query.

        Returns the (bytes_per_row,) uint8 packed code.
        """
        self._m.validate_against_embedder(embedder.fingerprint())
        prompt = "query"
        vec = embedder.encode([query], prompt=prompt)  # (1, full_dim)
        if vec.shape != (1, self._m.embedder.full_dim):
            raise ValueError(
                f"embedder returned shape {vec.shape}; expected "
                f"(1, {self._m.embedder.full_dim})"
            )
        centered = vec[0].astype(np.float32) - self._m.binarizer.mean_vector
        truncated = centered[: self._m.binarizer.dim]
        return self._stacked_simhash_encode(truncated[None, :])[0]

    def search(
        self,
        query: str,
        *,
        embedder: Embedder,
        k: int = 5,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Return [(hamming_distance, chunk_dict), ...] sorted ascending."""
        q_code = self.encode_query(query, embedder=embedder)
        dists = hamming_scan(self._codes, q_code)
        idx = top_k(dists, k=k)
        return [(int(dists[i]), self._chunks[int(i)]) for i in idx]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _stacked_simhash_encode(self, X: np.ndarray) -> np.ndarray:
        """Encode ``(N, dim)`` float32 via the same StackedSignBitQuantizer
        the packer used. Determined by ``(dim, k, seed)`` from the manifest."""
        from remax import StackedSignBitQuantizer

        q = StackedSignBitQuantizer(
            d=self._m.binarizer.dim,
            k=self._m.binarizer.k,
            seed=self._m.binarizer.seed,
        )
        return q.encode(X)
