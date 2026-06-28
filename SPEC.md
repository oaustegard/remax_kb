# `.kb` format — specification v1

A `.kb` file is a portable, deterministic, single-file knowledgebase artifact.
It carries a corpus's chunked text plus 1-bit binary embeddings of those
chunks, along with everything a reader needs to reproduce the embedding of a
fresh query and run Hamming-space retrieval against the packed corpus.

The format is intentionally minimal — three entries in a zip — and the
reference reader is pure numpy. The proof of concept is that a `.kb` can be
loaded and queried from a vanilla container (no torch, no transformers,
no peft).

## Container

A `.kb` file is a **ZIP_STORED** zip archive. Compression is disabled
because the heavy payload (`vectors.bin`) is dense random-looking bits and
does not compress; the JSON entries are small. `STORED` lets readers
`memmap` `vectors.bin` directly via `zipfile.ZipFile.open()` semantics
(or by extracting to a temp dir).

Required entries, in this exact relative-path form:

```
manifest.json
vectors.bin
chunks.jsonl
```

A reader MUST refuse a `.kb` missing any of these. Unknown additional
entries are ignored (forward-compatibility hint).

## `manifest.json`

UTF-8 JSON, no BOM. Required top-level fields:

```json
{
  "spec_version": "1",
  "embedder": {
    "model_id": "jinaai/jina-embeddings-v5-text-nano",
    "model_revision": "8a7f00aac812071b69403df470f1038ec85f8925",
    "release_url": "https://github.com/oaustegard/jina-v5-nano-mirror/releases/download/v5-nano-8a7f00aa/model.onnx",
    "release_sha256": "9f45091f1a1bc0affdd89245ca56928c7cc7ffefa79403782e1323eec9513ae6",
    "task_adapter": "retrieval",
    "pooling": "last-token",
    "normalize_l2": true,
    "full_dim": 768
  },
  "prompts": {
    "query": "Query: ",
    "document": "Document: "
  },
  "binarizer": {
    "kind": "remax-centered-simhash",
    "remax_version": "0.0.0",
    "dim": 256,
    "k": 8,
    "seed": 0,
    "mean_vector_b64": "<base64 of float32 mean vector, length=full_dim>"
  },
  "corpus": {
    "chunk_count": 1234,
    "build_hash": "<sha256 of vectors.bin || chunks.jsonl>",
    "built_at": "2026-05-11T20:00:00Z",
    "source": "optional free-text description"
  }
}
```

### Field semantics

`spec_version` — string `"1"`. A reader MUST refuse unknown versions.

`embedder.model_id` — canonical HF or model-source identifier. Drives the
fingerprint check.

`embedder.model_revision` — pinned upstream model SHA / commit.

`embedder.release_url` — fetchable URL of the *runtime* embedder asset
(typically an ONNX export). The reader will download from this URL if it
cannot find a cached copy. **Optional** — `null` (JSON) or absent for
API-backed embedders (e.g. Google Gemini, Cohere, OpenAI) where there
is no local asset to fetch. When absent, readers identify the embedder
by `model_id` alone and skip SHA256 verification; the host-side
embedder implementation talks to the upstream API directly.

`embedder.release_sha256` — SHA256 of the asset at `release_url`. The
reader MUST verify after download and refuse mismatched bytes. Optional
on the same condition as `release_url` — both fields must be present
together or both `null`.

`embedder.task_adapter` — name of the embedder's task adapter that was
used to embed documents at pack time. Queries MUST be embedded under the
same adapter.

`embedder.pooling` — pooling strategy. For jina-v5: `"last-token"`.

`embedder.normalize_l2` — whether embeddings are L2-normalized before
centering. `true` for the reference packer.

`embedder.full_dim` — native embedding dimension (e.g. 768 for jina-v5
nano). The `mean_vector_b64` is at this dimension.

`prompts.query` / `prompts.document` — exact prefix strings prepended at
embed time. The reader MUST reproduce these byte-for-byte.

`binarizer.kind` — string identifier selecting the vector codec. v1
recognizes two values; a reader MUST refuse unknown kinds:
- `"remax-centered-simhash"` — the default 1-bit centered-SimHash codec
  (Hamming scan). `dim * k` bits per chunk.
- `"remex-lloyd-max"` — the optional multi-bit remex codec (rotation +
  Lloyd-Max scalar quantization). `dim * bits` bits per chunk. See
  "remex codec" below.

`binarizer.remax_version` — version of the `remax` package used to
encode. Informational; the binary scheme is fully determined by
`(dim, k, seed)`.

`binarizer.dim` — working dimension `d`. The full-dim embedding is
Matryoshka-truncated to this width after centering. Must be a divisor of
8.

`binarizer.k` — stack count for `remax.StackedSignBitQuantizer`. Total
bits per chunk = `dim * k`. For the remex codec `k` is fixed to `1` and
unused.

`binarizer.bits` — bits per coordinate. `1` for the remax codec (the sign
bit). `1..8` for the remex codec, where total bits per chunk = `dim *
bits`. Defaults to `1` when absent, so manifests written before this field
deserialize unchanged. `dim * bits` MUST be a multiple of 8.

`binarizer.seed` — master RNG seed for the stacked Haar rotations. With
the same `(dim, k, seed)`, encoding is bit-identical across machines.

`binarizer.mean_vector_b64` — base64-encoded float32 little-endian array
of length `full_dim`. This is the corpus mean *before* truncation. The
reader subtracts it from a freshly embedded query, then truncates to
`dim`, before encoding.

`corpus.chunk_count` — number of rows in `vectors.bin` and lines in
`chunks.jsonl`. The reader MUST verify both match.

`corpus.build_hash` — `sha256(vectors.bin_bytes || chunks.jsonl_bytes)`,
where `||` is byte concatenation of the raw zip-entry contents. The
reader MUST verify and refuse on mismatch.

`corpus.built_at` — ISO-8601 UTC timestamp.

`corpus.source` — optional free-text description of the source corpus.

## `vectors.bin`

Raw packed bits.

- Layout: `N` rows × `(dim * k // 8)` bytes, row-major, contiguous.
- No header, no padding, no separator.
- Total size: `N * (dim * k // 8)`.
- Memmap-compatible: `np.frombuffer(...).reshape(N, dim * k // 8)`.

Row `i` contains the stacked-SimHash code for chunk `i`. Within a row,
the `k` per-rotation signatures sit contiguously, in seed-derived order
(see `remax.StackedSignBitQuantizer` for the layout).

For the remex codec, `vectors.bin` is the bit-packed `uint8` index array
(`remex.pack(indices, bits)`): `N * dim` indices, each `bits` wide, packed
LSB-first and contiguous, total `N * (dim * bits // 8)` bytes. The reader
recovers indices with `remex.unpack(blob, bits, N * dim).reshape(N, dim)`,
wraps them in a `remex.CompressedVectors` (norms reconstructed as `1.0` —
the codec requires an L2-normalizing embedder, so per-row norms are not
stored), and scores by inner product against the decoded corpus.

## remex codec

`"remex-lloyd-max"` stores multi-bit Lloyd-Max scalar-quantized codes
instead of 1-bit SimHash. It is the higher-fidelity / mid-byte operating
point: on general (isotropic) embedders such as Jina v5-nano, remex 4-bit
@ d=768 reproduces the fp32 ranking near-losslessly, well above any 1-bit
SimHash config at equal bytes. (On specialized, tightly-clustered encoders
like SPECTER2 the ordering can invert — 1-bit wins — so the codec is a
per-embedder choice, not a universal upgrade.)

Differences from the remax codec, all reader-visible via the manifest:

- **No centering.** remex normalizes and rotates each vector; subtracting a
  corpus mean measurably hurts it. `mean_vector_b64` is therefore all zeros
  and the reader's subtract is a no-op. The query is truncated to `dim`
  directly, with no centering.
- **`bits` carries the width** (`dim * bits` bits/chunk); `k = 1`.
- **Scoring is inner-product**, not Hamming. `search()` returns an integer
  angular distance `round((1 - cosine) * 10000)` (≈0 for a near-identical
  match), ascending, so the result contract (best-first) matches the remax
  path.
- **Determinism** is by `(dim, bits, seed)` — the Haar rotation and
  Lloyd-Max boundaries are derived from these alone; nothing extra ships.
- **Dependency:** the reader imports `remex` only for this kind (lazy), so
  remax-codec readers are unaffected.

## `chunks.jsonl`

UTF-8 JSON lines, exactly `N` lines, one chunk per line, **sorted by row
index** in `vectors.bin` (row `i` ↔ line `i`).

Required per-line fields:

```json
{"id": "doc-001#chunk-003",
 "sha256": "<sha256(text)>",
 "text": "...",
 "meta": {"source_path": "...", "page": 12}}
```

`id` — application-defined, unique per chunk within the file.

`sha256` — SHA256 of the chunk `text` (UTF-8 bytes), lower-case hex.

`text` — the chunk content. Newlines and other control bytes MUST be
JSON-escaped per RFC 8259.

`meta` — application-defined dict. Reader passes through verbatim.

## Validation order (reader)

A conforming reader, on opening a `.kb`, MUST in this order:

1. Open the zip and confirm exactly `manifest.json`, `vectors.bin`,
   `chunks.jsonl` are present.
2. Parse `manifest.json` and refuse if `spec_version != "1"` or
   `binarizer.kind` is not one of `"remax-centered-simhash"` /
   `"remex-lloyd-max"`.
3. Verify `corpus.chunk_count == N == len(chunks.jsonl)` where
   `N = len(vectors.bin) // bytes_per_row`, and `bytes_per_row` is
   `dim * k // 8` (remax) or `dim * bits // 8` (remex).
4. Verify `corpus.build_hash == sha256(vectors.bin || chunks.jsonl)`.

A conforming reader, when invoked with an embedder, MUST additionally
verify that the embedder's `(model_id, task_adapter, pooling, full_dim)`
match `manifest.embedder` exactly. Mismatches MUST raise; partial matches
MUST NOT be tolerated.

## Determinism guarantees

Given the same corpus text, chunker, embedder revision, and
`(dim, k, seed)`, the resulting `vectors.bin` is bit-identical across
machines. The packer SHOULD therefore make `dim`, `k`, and `seed`
explicit in CLI invocations rather than relying on defaults that may
change across versions.

## Out of scope (v1)

- Multi-embedder `.kb` files
- Updateable / appendable `.kb` (mutation requires rebuild)
- Compression on `vectors.bin`
- Streaming queries against `.kb` larger than RAM
- Per-chunk weighting / per-chunk binarizers
