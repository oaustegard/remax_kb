"""Manifest dataclass + JSON (de)serialization + validation.

The manifest is the contract between packer and reader. See SPEC.md for
the field-by-field specification.
"""
from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SPEC_VERSION = "1"
BINARIZER_KIND = "remax-centered-simhash"


@dataclass
class Embedder:
    model_id: str
    model_revision: str
    task_adapter: str
    pooling: str
    normalize_l2: bool
    full_dim: int
    # Optional for API-backed embedders (no runtime asset to fetch).
    # When None, readers identify the embedder by model_id alone and skip
    # the SHA256 verification step. The host-side embedder implementation
    # is expected to talk to the upstream API directly.
    release_url: str | None = None
    release_sha256: str | None = None


@dataclass
class Prompts:
    query: str
    document: str


@dataclass
class Binarizer:
    kind: str
    remax_version: str
    dim: int
    k: int
    seed: int
    mean_vector_b64: str

    @property
    def mean_vector(self) -> np.ndarray:
        raw = base64.b64decode(self.mean_vector_b64)
        return np.frombuffer(raw, dtype="<f4")

    @classmethod
    def from_mean(
        cls,
        *,
        remax_version: str,
        dim: int,
        k: int,
        seed: int,
        mean_vector: np.ndarray,
    ) -> "Binarizer":
        arr = np.ascontiguousarray(mean_vector, dtype="<f4")
        return cls(
            kind=BINARIZER_KIND,
            remax_version=remax_version,
            dim=int(dim),
            k=int(k),
            seed=int(seed),
            mean_vector_b64=base64.b64encode(arr.tobytes()).decode("ascii"),
        )


@dataclass
class CorpusInfo:
    chunk_count: int
    build_hash: str
    built_at: str
    source: str = ""


@dataclass
class Manifest:
    spec_version: str
    embedder: Embedder
    prompts: Prompts
    binarizer: Binarizer
    corpus: CorpusInfo

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        emb = dict(d["embedder"])
        # Tolerate empty-string legacy form for API-backed embedders too.
        if emb.get("release_url") == "":
            emb["release_url"] = None
        if emb.get("release_sha256") == "":
            emb["release_sha256"] = None
        return cls(
            spec_version=d["spec_version"],
            embedder=Embedder(**emb),
            prompts=Prompts(**d["prompts"]),
            binarizer=Binarizer(**d["binarizer"]),
            corpus=CorpusInfo(**d["corpus"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def bytes_per_row(self) -> int:
        return (self.binarizer.dim * self.binarizer.k) // 8

    def validate_static(self) -> None:
        """Static checks not requiring an embedder or the vectors blob."""
        if self.spec_version != SPEC_VERSION:
            raise ValueError(
                f"unsupported spec_version {self.spec_version!r}; "
                f"this reader speaks {SPEC_VERSION!r}"
            )
        if self.binarizer.kind != BINARIZER_KIND:
            raise ValueError(
                f"unsupported binarizer kind {self.binarizer.kind!r}; "
                f"this reader speaks {BINARIZER_KIND!r}"
            )
        if self.binarizer.dim <= 0 or self.binarizer.dim % 8 != 0:
            raise ValueError(
                f"binarizer.dim must be a positive multiple of 8, "
                f"got {self.binarizer.dim}"
            )
        if self.binarizer.k <= 0:
            raise ValueError(
                f"binarizer.k must be positive, got {self.binarizer.k}"
            )
        if self.embedder.full_dim <= 0:
            raise ValueError(
                f"embedder.full_dim must be positive, got {self.embedder.full_dim}"
            )
        if self.binarizer.dim > self.embedder.full_dim:
            raise ValueError(
                f"binarizer.dim ({self.binarizer.dim}) cannot exceed "
                f"embedder.full_dim ({self.embedder.full_dim})"
            )
        mean = self.binarizer.mean_vector
        if mean.shape != (self.embedder.full_dim,):
            raise ValueError(
                f"mean_vector length {mean.shape[0]} != embedder.full_dim "
                f"{self.embedder.full_dim}"
            )

    def validate_against_embedder(self, fingerprint: dict[str, Any]) -> None:
        """Refuse if the supplied embedder fingerprint disagrees with the manifest."""
        e = self.embedder
        expected = {
            "model_id": e.model_id,
            "task_adapter": e.task_adapter,
            "pooling": e.pooling,
            "full_dim": e.full_dim,
        }
        missing = [k for k in expected if k not in fingerprint]
        if missing:
            raise ValueError(f"embedder fingerprint missing keys: {missing}")
        for key, want in expected.items():
            got = fingerprint[key]
            if got != want:
                raise ValueError(
                    f"embedder fingerprint mismatch on {key!r}: "
                    f"manifest expects {want!r}, embedder reports {got!r}"
                )
