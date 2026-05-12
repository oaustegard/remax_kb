"""Thin embedder wrappers exposing the ``Embedder`` protocol expected by
the reader and packer.

Three implementations:

- :class:`JinaONNXEmbedder` — torch-free runtime path. Wraps
  ``jina_v5_nano_mirror.scripts.embed_onnx``. Downloads the merged-
  retrieval-adapter ONNX export from the configured release URL on
  first use and SHA256-verifies it.

- :class:`JinaTorchEmbedder` — packer-side path. Wraps the torch loader
  in the same upstream. Lets the packer pick any task adapter.

- :class:`GeminiEmbedder` — Google Generative Language API path. No
  local model; talks to ``generativelanguage.googleapis.com``. The
  ``.kb`` carries no ``release_url`` for this embedder; readers
  identify it by ``model_id`` alone.

All three expose ``fingerprint()`` and ``encode(texts, prompt=...)``.

Implementing your own embedder is the recommended way to plug in
other providers (Cohere, OpenAI, Voyage, etc.). The protocol is
documented in :mod:`remax_kb.read`. Minimum surface area::

    class MyEmbedder:
        model_id = "vendor/model-name"
        model_revision = ""
        task_adapter = "retrieval"
        pooling = "native"
        full_dim = 1024
        normalize_l2 = True
        release_url = None        # if API-backed
        release_sha256 = None
        prompts = {"query": "...", "document": "..."}

        def fingerprint(self) -> dict:
            return {
                "model_id": self.model_id,
                "task_adapter": self.task_adapter,
                "pooling": self.pooling,
                "full_dim": self.full_dim,
            }

        def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
            ...   # returns (N, full_dim) float32, L2-normalized
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np


# Canonical upstream identifiers — used for fingerprinting and manifest defaults.
JINA_V5_NANO_MODEL_ID = "jinaai/jina-embeddings-v5-text-nano"
JINA_V5_NANO_REVISION = "8a7f00aac812071b69403df470f1038ec85f8925"
JINA_V5_NANO_RELEASE_TAG = "v5-nano-8a7f00aa"
JINA_V5_NANO_RELEASE_BASE = (
    f"https://github.com/oaustegard/jina-v5-nano-mirror/releases/download/"
    f"{JINA_V5_NANO_RELEASE_TAG}"
)
JINA_V5_NANO_ONNX_URL = f"{JINA_V5_NANO_RELEASE_BASE}/model.onnx"
JINA_V5_NANO_ONNX_SHA256 = (
    "9f45091f1a1bc0affdd89245ca56928c7cc7ffefa79403782e1323eec9513ae6"
)
JINA_V5_NANO_TOKENIZER_URL = (
    # Tokenizer JSON ships in the model dir on the upstream HF repo and in
    # the mirror's `model/` subtree. For a torch-free runtime that's
    # bootstrapped from URLs alone, callers can either pre-stage it under
    # the cache root or set $REMAX_KB_TOKENIZER_PATH.
    None
)
JINA_V5_NANO_FULL_DIM = 768
JINA_V5_NANO_POOLING = "last-token"


def _cache_root() -> Path:
    return Path(
        os.environ.get("REMAX_KB_EMBEDDER_CACHE")
        or Path.home() / ".cache" / "remax_kb"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, expected_sha256: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and _sha256(dst) == expected_sha256:
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = Request(url, headers={"User-Agent": "remax_kb"})
    with urlopen(req) as resp, tmp.open("wb") as f:
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            f.write(buf)
    got = _sha256(tmp)
    if got != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 mismatch on {url}: expected {expected_sha256}, got {got}"
        )
    tmp.replace(dst)


# --------------------------------------------------------------------- #
# ONNX (runtime) — torch-free
# --------------------------------------------------------------------- #


class JinaONNXEmbedder:
    """Torch-free embedder built on ``onnxruntime + tokenizers``.

    The retrieval LoRA adapter is merged into the ONNX export, so this
    embedder supports only ``task_adapter="retrieval"``.

    The model asset (~847 MB) is downloaded once on first use and cached
    under ``~/.cache/remax_kb/jina-v5-nano/model.onnx`` (override with
    ``$REMAX_KB_EMBEDDER_CACHE``). A separate ``tokenizer.json`` must be
    discoverable (see :meth:`_load`).
    """

    model_id = JINA_V5_NANO_MODEL_ID
    model_revision = JINA_V5_NANO_REVISION
    task_adapter = "retrieval"
    pooling = JINA_V5_NANO_POOLING
    full_dim = JINA_V5_NANO_FULL_DIM
    normalize_l2 = True
    release_url = JINA_V5_NANO_ONNX_URL
    release_sha256 = JINA_V5_NANO_ONNX_SHA256
    prompts = {"query": "Query: ", "document": "Document: "}

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        max_length: int = 512,
    ):
        self._session = None
        self._tokenizer = None
        self._max_length = int(max_length)
        self._model_path = Path(model_path) if model_path else None
        self._tokenizer_path = Path(tokenizer_path) if tokenizer_path else None

    def fingerprint(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_adapter": self.task_adapter,
            "pooling": self.pooling,
            "full_dim": self.full_dim,
        }

    def _resolve_model(self) -> Path:
        if self._model_path is not None:
            return self._model_path
        # If the upstream mirror's ONNX cache already has the file, reuse it.
        mirror_cache = (
            Path(
                os.environ.get("JINA_V5_NANO_CACHE")
                or Path.home() / ".cache" / "jina-v5-nano-mirror"
            )
            / f"sha-{JINA_V5_NANO_REVISION[:10]}-onnx"
            / "model.onnx"
        )
        if mirror_cache.exists() and _sha256(mirror_cache) == self.release_sha256:
            return mirror_cache
        dst = _cache_root() / "jina-v5-nano" / "model.onnx"
        _download(self.release_url, dst, self.release_sha256)
        return dst

    def _resolve_tokenizer(self) -> Path:
        if self._tokenizer_path is not None:
            return self._tokenizer_path
        env_path = os.environ.get("REMAX_KB_TOKENIZER_PATH")
        if env_path:
            return Path(env_path)
        # Try the cloned mirror checkout if present — its model/ subdir
        # ships the upstream tokenizer.json verbatim.
        mirror = os.environ.get("JINA_V5_NANO_MIRROR_PATH")
        if mirror:
            cand = Path(mirror) / "model" / "tokenizer.json"
            if cand.exists():
                return cand
        # Default cache location. Absent → user must stage it.
        guess = _cache_root() / "jina-v5-nano" / "tokenizer.json"
        if not guess.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found at {guess}. Either set "
                f"$REMAX_KB_TOKENIZER_PATH, point $JINA_V5_NANO_MIRROR_PATH "
                f"at a checkout, or pass tokenizer_path=... to "
                f"JinaONNXEmbedder(). Source: "
                f"https://huggingface.co/{JINA_V5_NANO_MODEL_ID}"
            )
        return guess

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = self._resolve_model()
        tokenizer_path = self._resolve_tokenizer()
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
        if prompt not in self.prompts:
            raise ValueError(
                f"unknown prompt {prompt!r}; expected one of {list(self.prompts)}"
            )
        if not texts:
            return np.zeros((0, self.full_dim), dtype=np.float32)

        self._load()
        prefix = self.prompts[prompt]
        prefixed = [f"{prefix}{t}" for t in texts]

        self._tokenizer.enable_truncation(max_length=self._max_length)
        encoded = self._tokenizer.encode_batch(prefixed)
        # Pad to the batch's longest sequence.
        max_len = max(len(e.ids) for e in encoded)
        ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        mask = np.zeros((len(encoded), max_len), dtype=np.int64)
        for i, e in enumerate(encoded):
            L = len(e.ids)
            ids[i, :L] = e.ids
            mask[i, :L] = e.attention_mask

        hidden = self._session.run(
            ["last_hidden_state"],
            {"input_ids": ids, "attention_mask": mask},
        )[0]  # (N, S, 768) float32

        # Last-token pool (mask.sum(-1) - 1).
        lengths = mask.sum(axis=1) - 1
        rows = np.arange(hidden.shape[0])
        pooled = hidden[rows, lengths]  # (N, 768)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (pooled / norms).astype(np.float32)


# --------------------------------------------------------------------- #
# Torch (packer) — supports any task adapter
# --------------------------------------------------------------------- #


class JinaTorchEmbedder:
    """Heavy embedder: torch + transformers + peft path. Packer-only.

    Wraps the ``embed()`` entry point from the jina-v5-nano-mirror torch
    loader. The wrapper is dynamically imported so installing the
    runtime-only deps doesn't pull torch into the dependency closure.
    """

    model_id = JINA_V5_NANO_MODEL_ID
    model_revision = JINA_V5_NANO_REVISION
    pooling = JINA_V5_NANO_POOLING
    full_dim = JINA_V5_NANO_FULL_DIM
    normalize_l2 = True
    release_url = JINA_V5_NANO_ONNX_URL
    release_sha256 = JINA_V5_NANO_ONNX_SHA256
    prompts = {"query": "Query: ", "document": "Document: "}

    def __init__(self, *, task_adapter: str = "retrieval"):
        self.task_adapter = task_adapter
        self._embed_fn = None

    def fingerprint(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_adapter": self.task_adapter,
            "pooling": self.pooling,
            "full_dim": self.full_dim,
        }

    def _load(self):
        if self._embed_fn is not None:
            return self._embed_fn
        # Lazy import — only when actually invoked. The mirror is not a
        # pip-installable package; resolve via either:
        #   1. A checkout pointed at by $JINA_V5_NANO_MIRROR_PATH
        #      (a path to the cloned repo or its scripts/ subdir).
        #   2. ``embed`` already discoverable on sys.path (e.g. CCotw
        #      muninn-utilities path injection, or a vendored copy).
        import importlib
        import os as _os
        import sys as _sys

        env_path = _os.environ.get("JINA_V5_NANO_MIRROR_PATH")
        if env_path:
            from pathlib import Path as _Path

            root = _Path(env_path)
            scripts_dir = root if root.name == "scripts" else root / "scripts"
            if not (scripts_dir / "embed.py").exists():
                raise FileNotFoundError(
                    f"$JINA_V5_NANO_MIRROR_PATH={env_path!r} does not point to "
                    f"a jina-v5-nano-mirror checkout (no scripts/embed.py)"
                )
            if str(scripts_dir) not in _sys.path:
                _sys.path.insert(0, str(scripts_dir))

        try:
            mod = importlib.import_module("embed")
            embed = getattr(mod, "embed")
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "JinaTorchEmbedder needs the jina-v5-nano-mirror torch "
                "loader. Either set $JINA_V5_NANO_MIRROR_PATH to the cloned "
                "repo, or vendor scripts/embed.py onto PYTHONPATH. The "
                "mirror is not pip-installable. Source: "
                "https://github.com/oaustegard/jina-v5-nano-mirror"
            ) from exc
        self._embed_fn = embed
        return embed

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
        if prompt not in self.prompts:
            raise ValueError(
                f"unknown prompt {prompt!r}; expected one of {list(self.prompts)}"
            )
        if not texts:
            return np.zeros((0, self.full_dim), dtype=np.float32)
        fn = self._load()
        return fn(
            texts,
            task=self.task_adapter,
            prompt_name=prompt,
            truncate_dim=None,  # always return full_dim; truncation is reader's job
        ).astype(np.float32)


# --------------------------------------------------------------------- #
# Gemini (Google Generative Language API)
# --------------------------------------------------------------------- #


GEMINI_DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_DEFAULT_MODEL = "gemini-embedding-001"


class GeminiEmbedder:
    """Google Gemini embedder via the Generative Language API.

    API-backed: there is no local model asset, so the ``.kb`` manifest
    records ``release_url=None``/``release_sha256=None``. Readers and
    packers must both have ``$GEMINI_API_KEY`` (or pass ``api_key=...``)
    available — the embedder talks to ``generativelanguage.googleapis.com``
    directly for every encode call.

    Prompt mapping:

    * ``prompt="document"`` → ``task_type=RETRIEVAL_DOCUMENT``
    * ``prompt="query"``    → ``task_type=RETRIEVAL_QUERY``

    The pair produces embeddings in a compatible space; we still
    L2-normalize on the client side so downstream centering+truncation
    works identically to the Jina path.
    """

    pooling = "native"
    normalize_l2 = True
    release_url: str | None = None
    release_sha256: str | None = None
    # The "task adapter" abstraction maps cleanly to Gemini's task_type
    # parameter. The manifest stores "retrieval" so reader-side validation
    # can match it; the embedder picks DOCUMENT vs QUERY internally.
    task_adapter = "retrieval"
    prompts = {"query": "", "document": ""}  # task_type is passed via param, not prefix

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = GEMINI_DEFAULT_MODEL,
        output_dim: int = 768,
        endpoint: str = GEMINI_DEFAULT_ENDPOINT,
        max_retries: int = 5,
        request_timeout: float = 60.0,
        batch_limit: int = 100,
    ):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GeminiEmbedder requires an API key. Pass api_key=... "
                "or set $GEMINI_API_KEY."
            )
        self.model = model
        self.model_id = f"google/{model}"
        # Empty for hosted API models — Google doesn't expose a per-model SHA.
        self.model_revision = ""
        self.full_dim = int(output_dim)
        self._endpoint = endpoint.rstrip("/")
        self._max_retries = int(max_retries)
        self._timeout = float(request_timeout)
        self._batch_limit = int(batch_limit)

    def fingerprint(self) -> dict[str, Any]:
        # Includes both the manifest-validation keys (model_id,
        # task_adapter, pooling, full_dim) and informational
        # task_type_* labels that document the prompt mapping.
        return {
            "model_id": self.model_id,
            "task_adapter": self.task_adapter,
            "pooling": self.pooling,
            "full_dim": self.full_dim,
            "task_type_doc": "RETRIEVAL_DOCUMENT",
            "task_type_query": "RETRIEVAL_QUERY",
        }

    @staticmethod
    def _task_type_for(prompt: str) -> str:
        if prompt == "document":
            return "RETRIEVAL_DOCUMENT"
        if prompt == "query":
            return "RETRIEVAL_QUERY"
        raise ValueError(
            f"unknown prompt {prompt!r}; expected 'document' or 'query'"
        )

    def encode(self, texts: list[str], *, prompt: str = "document") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.full_dim), dtype=np.float32)
        task_type = self._task_type_for(prompt)

        import httpx  # local import — httpx is an optional dep

        out = np.empty((len(texts), self.full_dim), dtype=np.float32)
        with httpx.Client(timeout=self._timeout) as client:
            for start in range(0, len(texts), self._batch_limit):
                batch = texts[start : start + self._batch_limit]
                batch_vecs = self._batch_embed(client, batch, task_type)
                out[start : start + len(batch)] = batch_vecs
        return out

    def _batch_embed(
        self, client, texts: list[str], task_type: str
    ) -> np.ndarray:
        """One ``:batchEmbedContents`` request, with exponential backoff
        on 429 / 5xx."""
        import httpx

        url = (
            f"{self._endpoint}/models/{self.model}:batchEmbedContents"
            f"?key={self._api_key}"
        )
        body = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                    "outputDimensionality": self.full_dim,
                }
                for t in texts
            ]
        }

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                r = client.post(url, json=body)
            except httpx.HTTPError as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue
            if r.status_code == 200:
                data = r.json()
                vecs = [emb["values"] for emb in data["embeddings"]]
                arr = np.asarray(vecs, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                return (arr / norms).astype(np.float32)
            if r.status_code in (429, 500, 502, 503, 504):
                last_exc = httpx.HTTPStatusError(
                    f"transient {r.status_code}: {r.text[:200]}",
                    request=r.request,
                    response=r,
                )
                self._sleep_backoff(attempt)
                continue
            r.raise_for_status()
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        import time

        # 1s, 2s, 4s, 8s, 16s — capped via max_retries
        time.sleep(min(2 ** attempt, 30))
