"""v2 reader: open a `.kbi`, query via hybrid retrieval, lazily fetch chunks.

Implements the reader half of SPEC_v2.md. Supports both local-file
and HTTP(S) URI sources for `.kbi` and `.kbc/`.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urljoin, urlparse

import bm25s
import numpy as np

from ._hamming import _popcount_rows


SPEC_VERSION = "2"
KIND = "split-index"
ROW_BYTES_CHUNK_MAP = 24
FLAG_TOMBSTONE = 0x01


class Embedder(Protocol):
    def fingerprint(self) -> dict[str, Any]: ...
    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray: ...


@dataclass
class Hit:
    row: int
    chunk_id: str
    dense_dist: int | None = None
    dense_sim: float | None = None
    bm25_score: float | None = None
    fused: float | None = None
    text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None
    verified: bool = False


class KB:
    """An opened v2 `.kbi`."""

    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        vectors: np.ndarray,
        chunk_map: bytes,
        chunk_ids: bytes,
        bm25_retriever: bm25s.BM25 | None,
        chunks_uri: str,
        row_of_live: dict[int, int] | None,  # bm25 doc idx → absolute row
        deq_rotations: np.ndarray | None = None,  # (k,dim,dim) f32 when int8-packed
    ):
        self._m = manifest
        self._vectors = vectors
        self._chunk_map = chunk_map
        self._chunk_ids = chunk_ids
        self._bm25 = bm25_retriever
        self._chunks_uri = chunks_uri
        self._row_of_live = row_of_live or {}
        self._deq_rotations = deq_rotations

        b = manifest["binarizer"]
        self._dim = b["dim"]
        self._k = b["k"]
        self._seed = b["seed"]
        self._row_bytes = self._dim * self._k // 8
        self._total_bits = self._row_bytes * 8

        import base64
        self._mean = np.frombuffer(
            base64.b64decode(b["mean_vector_b64"]), dtype="<f4"
        ).copy()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def open(cls, source: str | Path) -> "KB":
        """Open a `.kbi` from a local path or http(s) URL."""
        kbi_bytes, base_uri = _fetch_bytes(source)
        return cls._from_bytes(kbi_bytes, base_uri)

    @classmethod
    def _from_bytes(cls, kbi_bytes: bytes, base_uri: str) -> "KB":
        with zipfile.ZipFile(io.BytesIO(kbi_bytes), "r") as zf:
            names = set(zf.namelist())
            for req in ("manifest.json", "vectors.bin", "chunk_map.bin", "chunk_ids.bin"):
                if req not in names:
                    raise ValueError(f"missing required entry: {req!r}")
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            vectors_bytes = zf.read("vectors.bin")
            chunk_map = zf.read("chunk_map.bin")
            chunk_ids = zf.read("chunk_ids.bin")

            bm25_dir_entries = [n for n in names if n.startswith("bm25/")]
            bm25_retriever: bm25s.BM25 | None = None
            if bm25_dir_entries:
                required_bm25 = {
                    "bm25/data.csc.index.npy",
                    "bm25/indices.csc.index.npy",
                    "bm25/indptr.csc.index.npy",
                    "bm25/params.index.json",
                    "bm25/vocab.index.json",
                }
                missing = required_bm25 - set(bm25_dir_entries)
                if missing:
                    raise ValueError(f"partial bm25/ subdir; missing {missing}")
                bm25_retriever = _load_bm25_from_zip(zf)

            # Projection planes for the query encoder, in priority order:
            #   - 'rademacher'  → regenerate ±1 planes from (dim,k,seed); nothing shipped
            #   - int8          → dequantize the shipped sidecar
            #   - else (haar)   → None: _dense_search recomputes Haar from the seed
            # 'deq_rotations', when set, is injected into the quantizer so the
            # query lands in the same sign-space the corpus was packed against.
            deq_rotations = None
            _bq = manifest.get("binarizer", {})
            if _bq.get("projection") == "rademacher":
                from .projection import rademacher_planes
                deq_rotations = rademacher_planes(_bq["dim"], _bq["k"], _bq["seed"])
            elif _bq.get("projection") == "srht":
                from .projection import srht_matrix
                deq_rotations = srht_matrix(_bq["dim"], _bq["k"], _bq["seed"],
                                            _bq.get("srht_rounds", 3))
            elif _bq.get("rotations_quant") == "int8":
                if {"binarizer/rotations.i8", "binarizer/rotations.scale.f32"} - names:
                    raise ValueError("rotations_quant=int8 but rotations.i8/scale.f32 missing")
                _dim, _k = _bq["dim"], _bq["k"]
                _i8 = np.frombuffer(zf.read("binarizer/rotations.i8"),
                                    dtype=np.int8).reshape(_k, _dim, _dim)
                _scale = np.frombuffer(zf.read("binarizer/rotations.scale.f32"),
                                       dtype="<f4").reshape(_k, _dim)
                from .rotations import dequantize_int8
                deq_rotations = dequantize_int8(_i8, _scale)

        # Validate
        if manifest["spec_version"] != SPEC_VERSION:
            raise ValueError(f"unsupported spec_version {manifest['spec_version']!r}")
        if manifest["kind"] != KIND:
            raise ValueError(f"unsupported kind {manifest['kind']!r}")

        bin_ = manifest["binarizer"]
        row_bytes = bin_["dim"] * bin_["k"] // 8
        total = manifest["chunks"]["total_rows"]
        if len(vectors_bytes) != total * row_bytes:
            raise ValueError(
                f"vectors.bin size mismatch: {len(vectors_bytes)} != {total} * {row_bytes}"
            )
        if len(chunk_map) != total * ROW_BYTES_CHUNK_MAP:
            raise ValueError(
                f"chunk_map.bin size mismatch: {len(chunk_map)} != {total} * {ROW_BYTES_CHUNK_MAP}"
            )
        live_count = sum(
            1 for i in range(total) if not (chunk_map[i * ROW_BYTES_CHUNK_MAP + 2] & FLAG_TOMBSTONE)
        )
        if live_count != manifest["chunks"]["live_count"]:
            raise ValueError(
                f"live_count mismatch: counted {live_count}, manifest says "
                f"{manifest['chunks']['live_count']}"
            )

        vectors = np.frombuffer(vectors_bytes, dtype=np.uint8).reshape(total, row_bytes)
        vectors = np.ascontiguousarray(vectors)

        # Build BM25 row mapping: bm25 doc idx (live position) → absolute row
        row_of_live: dict[int, int] = {}
        live_idx = 0
        for abs_row in range(total):
            if chunk_map[abs_row * ROW_BYTES_CHUNK_MAP + 2] & FLAG_TOMBSTONE:
                continue
            row_of_live[live_idx] = abs_row
            live_idx += 1

        # Resolve chunks URI against the base
        chunks_uri = manifest["chunks"]["uri"]
        if not chunks_uri.startswith(("http://", "https://", "file://")):
            chunks_uri = urljoin(base_uri, chunks_uri)

        return cls(
            manifest=manifest,
            vectors=vectors,
            chunk_map=chunk_map,
            chunk_ids=chunk_ids,
            bm25_retriever=bm25_retriever,
            chunks_uri=chunks_uri,
            row_of_live=row_of_live,
            deq_rotations=deq_rotations,
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def manifest(self) -> dict[str, Any]:
        return self._m

    @property
    def live_count(self) -> int:
        return self._m["chunks"]["live_count"]

    def __len__(self) -> int:
        return self.live_count

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        *,
        embedder: Embedder,
        k: int = 5,
        alpha: float | None = None,  # None → RRF; float → weighted
    ) -> list[Hit]:
        """Hybrid search. Returns Hit objects without chunk text/meta filled in."""
        self._validate_embedder(embedder.fingerprint())

        # Dense path
        dense_ranked = self._dense_search(query, embedder)
        # Lexical path (optional)
        lex_ranked = self._bm25_search(query) if self._bm25 is not None else None

        # Fusion
        if lex_ranked is None:
            top = dense_ranked[:k]
        else:
            over_fetch = max(k * 4, 20)
            top = _fuse_ranks(
                dense_ranked, lex_ranked, over_fetch=over_fetch, alpha=alpha
            )[:k]

        # Enrich with chunk_id (still no text)
        for hit in top:
            hit.chunk_id = self._chunk_id_at(hit.row)
        return top

    def fetch(self, hits: list[Hit]) -> list[Hit]:
        """Fetch chunk text + meta for hits via HTTP Range (or local file)."""
        for hit in hits:
            text, meta, sha = self._fetch_chunk(hit.row)
            hit.text = text
            hit.meta = meta
            hit.sha256 = sha
            hit.verified = hashlib.sha256(text.encode("utf-8")).hexdigest() == sha
        return hits

    def search_and_fetch(self, query: str, *, embedder: Embedder, k: int = 5,
                         alpha: float | None = None) -> list[Hit]:
        return self.fetch(self.search(query, embedder=embedder, k=k, alpha=alpha))

    # ------------------------------------------------------------------ #
    # Internal: query path
    # ------------------------------------------------------------------ #
    def _dense_search(self, query: str, embedder: Embedder) -> list[Hit]:
        prompt = self._m.get("prompts", {}).get("query", "Query: ")
        # The embedder protocol takes prompt by *name* (query|document), not the literal string.
        vec = embedder.encode([query], prompt="query").astype(np.float32)[0]
        if vec.shape[0] != self._m["embedder"]["full_dim"]:
            raise ValueError(
                f"embedder returned dim {vec.shape[0]}; expected {self._m['embedder']['full_dim']}"
            )
        centered = vec - self._mean
        truncated = centered[: self._dim]
        from remax import StackedSignBitQuantizer
        q_quant = StackedSignBitQuantizer(d=self._dim, k=self._k, seed=self._seed)
        if self._deq_rotations is not None:
            # int8-packed: encode the query with the same dequantized rotations
            # the corpus was packed against (not the exact f32 recompute).
            q_quant.rotations_ = self._deq_rotations.astype(q_quant.dtype)
        q_code = q_quant.encode(truncated[None, :])[0]  # (row_bytes,) uint8

        # Hamming scan, skipping tombstones
        # XOR rows, sum popcount per row
        xor = np.bitwise_xor(self._vectors, q_code[None, :])
        # popcount per row via hardware POPCNT (see _hamming._popcount_rows)
        dists = _popcount_rows(xor)
        # Mask tombstones to a large value so they don't appear
        tomb_mask = self._tombstone_mask()
        dists[tomb_mask] = self._total_bits + 1  # sentinel: never selected

        order = np.argsort(dists, kind="stable")
        hits: list[Hit] = []
        for row_idx in order:
            d = int(dists[row_idx])
            if d > self._total_bits:
                break  # all remaining are tombstones
            hits.append(Hit(
                row=int(row_idx),
                chunk_id="",  # filled later
                dense_dist=d,
                dense_sim=1.0 - d / self._total_bits,
            ))
        return hits

    def _bm25_search(self, query: str) -> list[Hit]:
        # bm25s.get_scores expects list of raw string tokens
        import re
        q_tokens = re.findall(r"[a-z0-9]+", query.lower())
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # scores: array of shape (num_live_docs,)
        order = np.argsort(-scores, kind="stable")
        hits: list[Hit] = []
        for live_idx in order:
            score = float(scores[live_idx])
            if score <= 0:
                break
            abs_row = self._row_of_live[int(live_idx)]
            hits.append(Hit(row=abs_row, chunk_id="", bm25_score=score))
        return hits

    # ------------------------------------------------------------------ #
    # Internal: chunk fetching
    # ------------------------------------------------------------------ #
    def _fetch_chunk(self, row: int) -> tuple[str, dict[str, Any], str]:
        info = self._chunk_map_row(row)
        shard_url = urljoin(self._chunks_uri, f"shard-{info['shard_id']:04d}.bin")
        if shard_url.startswith("file://") or shard_url.startswith("/") or len(shard_url) > 1 and shard_url[1] == ":":
            # local file path
            path = shard_url.replace("file://", "", 1) if shard_url.startswith("file://") else shard_url
            with open(path, "rb") as f:
                f.seek(info["byte_offset"])
                data = f.read(info["byte_length"])
        else:
            # HTTP Range
            req = urllib.request.Request(
                shard_url,
                headers={"Range": f"bytes={info['byte_offset']}-{info['byte_offset'] + info['byte_length'] - 1}"},
            )
            with urllib.request.urlopen(req) as resp:
                data = resp.read()

        nl = data.index(b"\n")
        header = json.loads(data[:nl].decode("utf-8"))
        text = data[nl + 1 :].decode("utf-8")
        return text, header.get("meta", {}), header["sha256"]

    def _chunk_map_row(self, row: int) -> dict[str, int]:
        o = row * ROW_BYTES_CHUNK_MAP
        shard_id, flags, _resv, byte_offset, byte_length, chunk_id_offset = struct.unpack(
            "<HBBQIQ", self._chunk_map[o : o + ROW_BYTES_CHUNK_MAP]
        )
        return {
            "shard_id": shard_id,
            "flags": flags,
            "byte_offset": byte_offset,
            "byte_length": byte_length,
            "chunk_id_offset": chunk_id_offset,
        }

    def _chunk_id_at(self, row: int) -> str:
        offset = self._chunk_map_row(row)["chunk_id_offset"]
        end = self._chunk_ids.index(0, offset)
        return self._chunk_ids[offset:end].decode("utf-8")

    def _tombstone_mask(self) -> np.ndarray:
        N = self._vectors.shape[0]
        mask = np.zeros(N, dtype=bool)
        for i in range(N):
            if self._chunk_map[i * ROW_BYTES_CHUNK_MAP + 2] & FLAG_TOMBSTONE:
                mask[i] = True
        return mask

    def _validate_embedder(self, fp: dict[str, Any]) -> None:
        m = self._m["embedder"]
        for field in ("model_id", "task_adapter", "pooling", "full_dim"):
            if fp[field] != m[field]:
                raise ValueError(
                    f"embedder fingerprint mismatch on {field!r}: "
                    f"manifest={m[field]!r}, embedder={fp[field]!r}"
                )


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _load_bm25_from_zip(zf: zipfile.ZipFile) -> bm25s.BM25:
    """Load bm25s.BM25 from a zip subdir by extracting to a tempdir."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for name in zf.namelist():
            if not name.startswith("bm25/"):
                continue
            inner = name[len("bm25/") :]
            target = Path(d) / inner
            target.write_bytes(zf.read(name))
        return bm25s.BM25.load(d)


def _fuse_ranks(
    dense: list[Hit],
    lex: list[Hit],
    *,
    over_fetch: int,
    alpha: float | None,
) -> list[Hit]:
    dense_n = dense[:over_fetch]
    lex_n = lex[:over_fetch]
    if alpha is None:
        # RRF
        C = 60
        merged: dict[int, Hit] = {}
        for idx, h in enumerate(dense_n):
            merged[h.row] = Hit(
                row=h.row, chunk_id="", dense_dist=h.dense_dist,
                dense_sim=h.dense_sim, fused=1.0 / (C + idx + 1),
            )
        for idx, h in enumerate(lex_n):
            prev = merged.get(h.row)
            score_add = 1.0 / (C + idx + 1)
            if prev is None:
                merged[h.row] = Hit(
                    row=h.row, chunk_id="", bm25_score=h.bm25_score, fused=score_add,
                )
            else:
                prev.bm25_score = h.bm25_score
                prev.fused = (prev.fused or 0) + score_add
        return sorted(merged.values(), key=lambda h: -(h.fused or 0))
    else:
        # Weighted with min-max norm within over-fetched pool
        d_dists = [h.dense_dist for h in dense_n if h.dense_dist is not None]
        l_scores = [h.bm25_score for h in lex_n if h.bm25_score is not None]
        d_min, d_max = (min(d_dists), max(d_dists)) if d_dists else (0, 1)
        l_min, l_max = (min(l_scores), max(l_scores)) if l_scores else (0, 1)
        def norm_d(d: int) -> float:
            return 1.0 if d_max == d_min else (d_max - d) / (d_max - d_min)
        def norm_l(s: float) -> float:
            return 1.0 if l_max == l_min else (s - l_min) / (l_max - l_min)

        merged: dict[int, Hit] = {}
        for h in dense_n:
            merged[h.row] = Hit(
                row=h.row, chunk_id="", dense_dist=h.dense_dist, dense_sim=h.dense_sim,
                fused=alpha * norm_d(h.dense_dist or d_max),
            )
        for h in lex_n:
            prev = merged.get(h.row)
            add = (1 - alpha) * norm_l(h.bm25_score or l_min)
            if prev is None:
                merged[h.row] = Hit(
                    row=h.row, chunk_id="", bm25_score=h.bm25_score, fused=add,
                )
            else:
                prev.bm25_score = h.bm25_score
                prev.fused = (prev.fused or 0) + add
        return sorted(merged.values(), key=lambda h: -(h.fused or 0))


def _fetch_bytes(source: str | Path) -> tuple[bytes, str]:
    """Return (bytes, base_uri). base_uri is the directory for relative resolves."""
    s = str(source)
    if s.startswith(("http://", "https://")):
        with urllib.request.urlopen(s) as r:
            return r.read(), s.rsplit("/", 1)[0] + "/"
    # Local file
    p = Path(s)
    data = p.read_bytes()
    base = f"file://{p.parent.resolve()}/"
    return data, base
