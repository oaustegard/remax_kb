# `.kb` format — specification v2 (DRAFT)

> **Status:** draft for discussion. Tracking issue: TBD. v1 (`SPEC.md`)
> remains the normative spec for `binarizer.kind == "remax-centered-simhash"`
> single-file `.kb` artifacts. v2 introduces a split-index layout, lazy
> chunk fetch, hybrid retrieval, and update semantics. The two formats
> coexist; readers identify by `spec_version`.

## Motivation

v1 packs the index *and* the chunks into one zip. That's fine at the
hundreds-of-chunks scale (`muninn-subset.kb` is 226 KB) and convenient
for "one file you can email someone." It breaks down at three places:

1. **Memory.** The full chunks payload sits resident even though, per
   query, you only ever return top-K text. At 1M chunks × ~2 KB/chunk
   you're carrying ~2 GB just to display a handful of hits.
2. **Mutation.** v1 explicitly punts on appends and updates: any change
   requires a full rebuild and a new artifact URL. For a living corpus
   (a blog, a docs site, an inbox) that's wrong by design.
3. **Retrieval quality.** v1 is dense-only. 1-bit codes are great for
   semantic similarity but lose exact-term fidelity — proper nouns,
   code identifiers, rare terms. Hybrid dense+BM25 is the standard
   quality lever once a corpus has any lexical surface.

v2 splits the artifact, adds BM25 alongside the dense codes, and
defines tombstone-based mutation so a live corpus can be served from
a CDN without rebuild ceremony per edit.

## Two artifacts

```
foo.kbi   index (zip)              hot — agent downloads on startup
foo.kbc/  chunk store (directory)  cold — fetched per-hit via HTTP Range
  shard-0001.bin
  shard-0002.bin
  ...
```

The `.kbi` is small enough to live in memory (memmap-friendly) for any
plausible scale. Chunk bytes live in numbered shards in the `.kbc/`
directory, each a flat concatenation of chunk bytes, byte-addressable.

### Hosting expectations

Both artifacts MUST be servable over HTTP/1.1+ with `Range` request
support. Cloudflare Pages, R2, S3, nginx, GitHub Releases — anything
that handles `Range: bytes=N-M` works.

Static CDN (Pages) is sufficient for read-only use. A writeable
backend (R2, S3) is required only if mutation is performed by an
agent rather than a human-driven build pipeline.

## `.kbi` structure

ZIP_STORED archive. Required entries:

```
manifest.json
vectors.bin
chunk_map.bin
chunk_ids.bin
bm25/data.csc.index.npy        (optional — see §bm25/)
bm25/indices.csc.index.npy     (optional)
bm25/indptr.csc.index.npy      (optional)
bm25/params.index.json         (optional)
bm25/vocab.index.json          (optional)
```

A reader MUST refuse a `.kbi` missing any required non-`bm25/`
entry. The `bm25/` subdirectory is collectively optional: either all
five files MUST be present (lexical search supported) or none MUST
be present (dense-only). Mixed states are an error.

Unknown additional entries are ignored (forward-compatibility hint).

## `manifest.json`

UTF-8 JSON, no BOM.

```json
{
  "spec_version": "2",
  "version": 17,
  "kind": "split-index",
  "embedder": {
    "model_id": "jinaai/jina-embeddings-v5-text-nano",
    "model_revision": "8a7f00aac812071b69403df470f1038ec85f8925",
    "release_url": "https://...model.onnx",
    "release_sha256": "9f45091f...",
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
    "remax_version": "0.1.0",
    "dim": 256,
    "k": 8,
    "seed": 0,
    "projection": "haar",
    "rotations_quant": "float32",
    "mean_vector_b64": "..."
  },
  "lexical": {
    "kind": "bm25s",
    "library_version": "0.2.x",
    "k1": 1.5,
    "b": 0.75,
    "tokenizer": "bm25s.default",
    "stopwords": null
  },
  "chunks": {
    "uri": "https://muninn.austegard.com/knowledge/muninn.kbc/",
    "shard_count": 3,
    "shard_max_bytes": 20971520,
    "live_count": 1234,
    "total_rows": 1281,
    "merkle_root": "<sha256 over per-chunk hashes>"
  },
  "built_at": "2026-05-20T20:00:00Z",
  "source": "muninn.austegard.com full corpus"
}
```

### Field semantics

`spec_version` — `"2"`. Readers MUST refuse unknown versions.

`version` — monotonic integer, incremented on every commit. Used by
clients to detect updates without re-parsing the body.

`kind` — `"split-index"`. Distinguishes from a future possible
`"single-file"` (v1-compatible) variant under the same spec_version.

`embedder.*` — identical semantics to v1.

`prompts.*` — identical semantics to v1.

`binarizer.projection` — `"haar"` (default) or `"rademacher"`. Selects the
hyperplane family and, decisively, whether the rotation is *shipped* or
*regenerated* — see §projection below. With `"rademacher"` the `.kbi` carries no
rotation entry and `rotations_quant` is `"none"`.

`binarizer.*` — identical semantics to v1, plus the optional
`binarizer.rotations_quant` (`"float32"` default, or `"int8"`) which
selects how the rotation sidecar is stored — see
`binarizer/rotations.i8` below.

`lexical.kind` — `"bm25s"` for v2. Other lexical scorers may be
added under new kinds. Optional: omit the `lexical` block entirely
for dense-only `.kbi`. Readers MUST gracefully degrade.

`lexical.library_version` — version string of the lexical indexer.
Informational; reader does not need to match.

`lexical.k1`, `lexical.b` — BM25 hyperparameters. Reader applies
these at query time.

`lexical.tokenizer` — `"bm25s.default"` for now. Future kinds may
encode the tokenizer identifier here.

`lexical.stopwords` — `null` or an array of strings. Defaults to no
stopword removal (matches the bm25 skill convention).

`chunks.uri` — absolute or relative URI of the `.kbc/` directory.
Trailing slash required. Shards are at `${uri}shard-NNNN.bin` where
`NNNN` is zero-padded to 4 digits.

`chunks.shard_count` — number of shards currently in the chunk store.

`chunks.shard_max_bytes` — soft cap. New chunks roll to a new shard
when adding to the current one would exceed this cap. Recommended
value: 20 MiB (fits Cloudflare Pages per-asset limit).

`chunks.live_count` — number of non-tombstoned chunks. This is the
authoritative answer to "how many documents are in this corpus."

`chunks.total_rows` — total rows in `vectors.bin`, including
tombstones. Equals `len(chunk_map.bin) / row_bytes`.

`chunks.merkle_root` — root of a binary Merkle tree over the
per-chunk `sha256(text)` values. Leaves are ordered by row index.
Readers MAY verify individual chunks by recomputing the Merkle path
on lazy fetch.

`built_at` — ISO-8601 UTC of the most recent commit.

`source` — free-text description of the source corpus.

## `vectors.bin`

Identical layout to v1: `total_rows × (dim * k // 8)` bytes,
row-major, no header. Row `i` is the stacked-SimHash code for the
chunk at row `i`. Tombstoned rows still have their bits — the reader
skips them via `chunk_map.bin` flags, not by absence from the
vectors file.

## `chunk_map.bin`

Fixed-width binary table, **24 bytes per row**, little-endian.
`total_rows` rows in row-index order.

```
offset  size  field
------  ----  --------------------------------------------------
 0       2    shard_id              uint16
 2       1    flags                 uint8   (bit 0 = tombstone)
 3       1    reserved              uint8   (MUST be 0)
 4       8    byte_offset           uint64  (within shard)
12       4    byte_length           uint32
16       8    chunk_id_offset       uint64  (into chunk_ids.bin)
```

`shard_id` — index of the shard file in `.kbc/`, zero-indexed.

`flags` — bitfield.
  - bit 0: tombstone (chunk is logically deleted; reader MUST skip)
  - bits 1–7: reserved, MUST be 0 in v2.0

`byte_offset` / `byte_length` — location of the chunk's text bytes
within shard `shard_id`. Bytes at this range are UTF-8 encoded text,
optionally preceded by a JSON metadata header (see §`.kbc` shards).

`chunk_id_offset` — byte offset into `chunk_ids.bin` where the
chunk's identifier starts. Identifier extends to the next NUL byte.

## `chunk_ids.bin`

UTF-8 text, NUL-separated chunk identifiers. The `chunk_id_offset`
field in `chunk_map.bin` points at the first byte of each id. Ids
MUST be unique among **non-tombstoned** rows but MAY be reused across
generations: when a chunk is updated, the old row is tombstoned and
a new row is appended that may share the original chunk_id.

Recommended format for human-readable ids: `<source>#<sub-id>`. For a
blog corpus: `post-2026-05-18-where-ast-helps-bm25#chunk-3`. The
reader treats ids as opaque strings; the convention is solely for
display and back-linking by the application.

## `bm25/`

Output of `bm25s.BM25.save(directory)`. As of `bm25s==0.3.9` this
emits five files:

- `bm25/data.csc.index.npy` — CSC sparse matrix values (float32)
- `bm25/indices.csc.index.npy` — CSC row indices (int32)
- `bm25/indptr.csc.index.npy` — CSC column pointers (int32)
- `bm25/vocab.index.json` — `{token: token_id}` mapping
- `bm25/params.index.json` — `{k1, b, delta, method, idf_method,
  num_docs, ...}` self-contained for round-trip load

The CSC matrix has one column per vocabulary token and one row per
indexed document. **Rows correspond to non-tombstoned chunk_map
rows in row-index order.** The reader is responsible for mapping a
BM25 doc index back to an absolute `chunk_map` row by skipping
tombstones.

A `.kbi` MAY omit the `bm25/` directory entirely. Readers MUST
detect absence and gracefully degrade to dense-only search.



## §projection — `haar` vs `rademacher`

`binarizer.projection` chooses the hyperplane family for the stacked-SimHash
binarizer, and with it the single most consequential portability property: **is
the projection shipped, or regenerated from the seed?**

### Why it matters

The query is encoded into the same sign-space as the corpus only if producer and
consumer use the **identical** projection. Two *different* valid projections are
not approximations of each other — they are statistically independent, so mixing
them (corpus hashed with A, query with B) flips ~50% of code bits and collapses
recall to chance. This is not the small, bounded error of int8 quantization
(~0.24% of bits); it is total.

- **`haar`** (default) — orthogonal matrices from numpy's
  `PCG64 + Ziggurat + LAPACK-QR`. Not reproducible outside numpy (LAPACK QR even
  drifts across platforms), so a Haar `.kbi` **MUST ship** the matrices
  (`rotations.f32`, or int8 `rotations.i8` + scale). Highest recall.

- **`rademacher`** — ±1 hyperplane entries from `splitmix64`, a tiny integer
  PRNG every language reproduces bit-for-bit. The `.kbi` ships **nothing**; both
  sides regenerate the planes from `(dim, k, seed)`. ~2 recall@10 points below
  Haar at matched `k` (bought back with one extra stack), but the rotation
  sidecar vanishes and the cross-language mismatch failure is structurally
  impossible. Best for small / portable / skill-bundled KBs.

### Normative `rademacher` algorithm

Entry at flat C-order index `i` in the `(k, dim, dim)` plane tensor
(`i = (j*dim + row)*dim + col`) is the `i`-th `splitmix64` draw seeded by
`binarizer.seed`, mapped to a sign. All arithmetic is **unsigned 64-bit
modular**:

```
GOLDEN = 0x9E3779B97F4A7C15
M1     = 0xBF58476D1CE4E5B9
M2     = 0x94D049BB133111EB

z = (seed + (i+1) * GOLDEN)  mod 2^64
z = ((z XOR (z >> 30)) * M1) mod 2^64
z = ((z XOR (z >> 27)) * M2) mod 2^64
z =  (z XOR (z >> 31))
entry = -1.0 if (z >> 63) & 1 else +1.0
```

A reader with `projection == "rademacher"` MUST regenerate planes by this exact
recipe and MUST NOT expect any `binarizer/rotations.*` entry. Reference
implementations: `remax_kb.projection.rademacher_planes` (numpy uint64) and
`rademacherPlanes` in `js/kb-reader.js` (BigInt masked to 64 bits); a
Python↔Node round-trip pins them bit-identical.

## `binarizer/rotations.f32`

**Optional; `haar` projection only.** Pre-computed Haar rotation matrices for
the stacked-SimHash binarizer.

Layout: `k × dim × dim` float32 little-endian values, concatenated.
Total size: `k * dim * dim * 4` bytes. Rotation `j` of the k-stack
occupies offset `j * dim * dim * 4` to `(j+1) * dim * dim * 4`,
laid out row-major within each `(dim, dim)` matrix.

This entry exists because bit-identical reproduction of NumPy's
`SeedSequence` + Ziggurat-driven Gaussian sampling + LAPACK
Householder QR in non-NumPy environments (e.g. JavaScript, WASM) is
fragile in ways that produce silent retrieval-quality failures.
Shipping the rotations sidesteps the problem at a fixed cost:

| dim | k  | rotations size |
|----:|---:|---------------:|
| 256 |  8 |        2.0 MiB |
| 384 |  4 |        2.3 MiB |
| 768 |  4 |        9.0 MiB |

This is **constant per binarizer configuration**, independent of
corpus size. At ≥10K-chunk corpora the relative overhead is under
20%; at 100K+ it's negligible.

Readers that have a faithful NumPy-equivalent stack MAY ignore this
entry and re-derive rotations from `(dim, k, seed)`. Readers in
JavaScript and other environments where bit-fidelity is impractical
MUST use this entry when present and MUST refuse to operate against
a `.kbi` lacking it.

## `binarizer/rotations.i8` + `binarizer/rotations.scale.f32`

**Optional, int8-quantized alternative to `rotations.f32`.** Selected at
pack time via `binarizer.rotations_quant == "int8"` (default
`"float32"`). When present, `rotations.f32` is absent and these two
entries replace it.

The rotations feed only a sign test (`x·Q ≥ 0`), so f32 precision is
unnecessary. Quantizing to int8 with a per-output-column scale shrinks
the sidecar **4×** (`k·dim²` bytes + `k·dim·4` bytes of scale) while
flipping a negligible fraction of code bits (~0.24% on a real
768-d/k=2 corpus, with no measurable recall loss).

Layout:

- `rotations.i8` — `k × dim × dim` **int8** values, same C-order
  (row-major) element order as `rotations.f32`.
- `rotations.scale.f32` — `k × dim` float32 little-endian, one scale per
  `(stack j, output column e)`.

Dequantize before use: `Q[j, d, e] = i8[j, d, e] * scale[j, e]`. The
column axis `e` is the projection output (hyperplane index); the scale
is `max_d |Q[j, d, e]| / 127` per column, clamped away from zero.

A reader that encounters `rotations_quant == "int8"` MUST dequantize
these entries and MUST NOT re-derive rotations from `(dim, k, seed)`:
the corpus codes were packed against the **dequantized** rotations, so
re-deriving the exact f32 matrices would land queries in a slightly
different sign-space. Per-config sizes:

| dim | k  | f32 sidecar | int8 sidecar |
|----:|---:|------------:|-------------:|
| 256 |  8 |     2.0 MiB |      0.5 MiB |
| 512 |  4 |     4.0 MiB |      1.0 MiB |
| 768 |  2 |     4.5 MiB |      1.1 MiB |

## `.kbc/shard-NNNN.bin`

Flat byte concatenation, no header, no separators. Each chunk
occupies `byte_length` bytes starting at `byte_offset` (as recorded
in `chunk_map.bin`).

A chunk's bytes are formatted as:

```
<JSON header line>\n<chunk text>
```

The JSON header is a single line (no embedded newlines, no leading
whitespace) ending in `\n`. Required header fields:

```json
{"sha256": "<sha256(text)>", "meta": {"source": "...", ...}}
```

The reader, on fetching a chunk:
1. Reads the requested byte range
2. Splits on the first `\n` — left is header JSON, right is text
3. Verifies `sha256` matches the recomputed hash of text
4. Optionally verifies the Merkle path back to `chunks.merkle_root`

`meta` is application-defined and passed through verbatim.

### Why header-then-text instead of pure text

Two reasons:
1. The application's `meta` (source URL, page number, etc.) is
   needed at display time but not at retrieval time. Putting it next
   to the text instead of in `.kbi` keeps the hot index lean.
2. Per-chunk `sha256` enables Merkle verification without trusting
   the chunk transport, even when shards are served from a CDN that
   might serve stale or corrupted bytes.

## Update semantics

v2 supports four mutation primitives. All are committed atomically
via a manifest write at the end.

### Append

New chunks are added to the end of `vectors.bin`, `chunk_map.bin`,
and the current shard (or a new shard if the soft cap is exceeded).
`chunks.total_rows` and `chunks.live_count` both increment.

The BM25 index is **rebuilt from scratch** on each commit. At
v2.0-scale corpora (≤1M chunks), bm25s rebuild is sub-minute and
acceptable. Incremental BM25 updates are an explicit non-goal.

### Update

A chunk_id is replaced with new text. Implemented as:

1. Locate the live row for the chunk_id; set its flags bit 0
   (tombstone).
2. Append a new row with the new text (per Append semantics).

The new row may use the same chunk_id or a new one. The reader
resolves "the current text for chunk_id X" by scanning
`chunk_ids.bin` for non-tombstoned rows.

### Delete

The chunk_id's live row is tombstoned (flags bit 0 set). The chunk's
bytes are **not** removed from the shard — they become orphaned. The
reader filters tombstones at scan time.

### Compact

Rebuild `.kbi` and `.kbc` from scratch, discarding tombstoned rows
and orphaned shard bytes. Equivalent to a full v2 pack from the live
chunks. Optional but recommended when tombstones exceed ~25% of
total rows.

## Reader contract

```python
kb = KB.open(
    index_uri="https://muninn.austegard.com/knowledge/muninn.kbi",
    cache_dir="~/.cache/remax_kb",   # optional; default ephemeral
)

# Hybrid search — index-only, no chunk fetch
hits = kb.search(
    "centered simhash vs random projection",
    embedder=GeminiEmbedder(...),
    k=5,
    alpha=0.5,           # 0 = pure BM25, 1 = pure dense; default 0.5
    fusion="rrf",        # "rrf" | "weighted"
)
# hits: [{"row": int, "chunk_id": str, "dense_dist": int,
#         "bm25_score": float, "fused_score": float}, ...]

# Lazy chunk fetch
chunks = kb.fetch(hits)
# chunks: [{"chunk_id": str, "text": str, "meta": dict, "sha256": str,
#          "verified": bool}, ...]

# Convenience: search + fetch in one call
results = kb.search_and_fetch(query, embedder=..., k=5)
```

### `KB.open(index_uri, ...)`

Reader MUST:

1. Resolve `index_uri` to bytes via HTTP GET (or local file read for
   `file://` and bare paths).
2. Cache the `.kbi` keyed by ETag if `cache_dir` provided.
3. Parse `manifest.json`; refuse if `spec_version != "2"` or
   `kind != "split-index"`.
4. Validate that all required zip entries are present.
5. Memmap or load `vectors.bin`, `chunk_map.bin`, `chunk_ids.bin`.
6. If `bm25/` present, load via `bm25s.BM25.load(...)`.
7. Resolve `manifest.chunks.uri` to an absolute URL (relative
   resolves against `index_uri`'s directory).

### `kb.search(...)`

Reader MUST:

1. Embed the query under `manifest.prompts.query` + the v1 embedder
   contract.
2. Center, truncate, encode via stacked-SimHash → query code.
3. Hamming-scan against `vectors.bin`, skipping tombstoned rows.
4. If BM25 present and `alpha < 1`, tokenize query, score against
   the bm25 index.
5. Fuse scores via RRF or weighted sum (see fusion contract below).
6. Return top-K with both raw scores and the fused score.

### `kb.fetch(hits)`

Reader MUST:

1. Group hits by `shard_id`.
2. For each shard, issue one or more HTTP Range requests covering
   the union of needed byte ranges. Implementations MAY coalesce
   adjacent ranges or use a multi-range request.
3. For each fetched chunk: split on first `\n`, parse JSON header,
   verify `sha256(text)` matches header.
4. Optionally verify Merkle path against `chunks.merkle_root`.
5. Return text + meta + verification status.

### Fusion contract

**RRF** (default): `fused = Σ 1 / (60 + rank)` summed over each scorer
the chunk appears in. The `60` constant is the conventional RRF
parameter.

**Weighted**: `fused = alpha * dense_score + (1 - alpha) * bm25_norm`
where `dense_score` is `1 - hamming/total_bits` and `bm25_norm` is
min-max normalized across the top-N candidates from each scorer.

Implementations SHOULD over-fetch candidates from each scorer (e.g.,
`top_k * 4` from each) before fusion, then truncate to K.

## Cache and freshness

Clients SHOULD cache the `.kbi` with HTTP `ETag` / `If-None-Match`.
On `304 Not Modified`, reuse the cached payload. On `200 OK` with a
new ETag, reload.

Clients MAY poll `manifest.json` directly (cheap: a few KB) by
fetching it from inside the cached zip after a `HEAD` returns a new
ETag. Pattern: HEAD the `.kbi`, if ETag changed, GET fresh.

`manifest.version` provides an in-payload monotonic check, useful
when ETag isn't honored by the transport.

## Validation order

A conforming reader, on opening a `.kbi`, MUST in this order:

1. Confirm all required entries present.
2. Parse `manifest.json`; refuse on unknown `spec_version` or `kind`.
3. Compute `N = len(vectors.bin) / row_bytes`. Verify
   `N == manifest.chunks.total_rows`.
4. Verify `len(chunk_map.bin) == N * 24`.
5. Scan `chunk_map.bin`; verify all `chunk_id_offset` values point
   to NUL-terminated valid UTF-8 within `chunk_ids.bin`.
6. Count non-tombstoned rows; verify
   `manifest.chunks.live_count == count`.
7. If `bm25/` present, load and verify the postings matrix has
   `live_count` rows.
8. Validate the embedder fingerprint (delegate, same as v1).

Lazy chunk fetches MUST additionally verify `sha256(text)` against
the per-chunk header on each read.

## Determinism vs mutability

v1's determinism guarantee was: same `(corpus, embedder, dim, k,
seed)` → bit-identical `vectors.bin`.

v2 weakens this to: same `(corpus, mutation history, embedder, dim,
k, seed)` → bit-identical artifacts. Different mutation orderings of
the same final chunk set produce identical `live_count` but
different `total_rows` (more or fewer tombstones), different shard
layouts, and different bm25 row→absolute-row mappings.

For bit-identical reproducibility across machines, `compact()` after
the last mutation. The compacted form is deterministic by the v1
rule.

## Out of scope for v2

- Per-chunk weighting / per-chunk binarizers
- Incremental BM25 updates (always full rebuild on commit)
- Compression on `vectors.bin` or shard files
- Streaming queries against a `.kbi` larger than RAM
  (open question — likely a v3 concern when index size > 1 GB)
- Multi-embedder `.kbi`
- Encrypted shards / per-chunk ACLs
- Cross-`.kbi` joins / federated retrieval

## Compatibility

v1 readers MUST refuse v2 artifacts (`spec_version` mismatch is a
hard error). v2 readers MAY transparently fall back to v1 by
inspecting `spec_version` before validating; recommended that they
do so to ease migration.

The CLI (`remax-kb`) SHOULD detect format by inspecting the file:
files containing `chunks.jsonl` are v1, files containing
`chunk_map.bin` are v2.
