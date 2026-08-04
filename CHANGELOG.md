# Changelog

## Unreleased

Nothing yet.

## v0.5.0 — 2026-08-04

Cross-reader parity. The Python and JavaScript readers are now executed against
each other in CI, and three places where they disagreed — or where one of them
disagreed with `remax` — are closed.

### Fixed

- **BM25 query/index tokenizer mismatch, in both readers**
  ([#28](https://github.com/oaustegard/remax_kb/pull/28)). The query was
  tokenized differently from the index it was querying.

- **`_hamming.top_k` had silently re-introduced a bug `remax` already fixed**
  ([#29](https://github.com/oaustegard/remax_kb/pull/29)). remax's
  `stable_top_k` does argpartition *then widens to `dists <= pivot`* before the
  stable sort, so a lower-indexed tie cannot be stranded outside the partition
  — that widening is remax PR #32. `_hamming.top_k` did argpartition and sort
  with no widening, under a comment saying *"same recipe as remax"* (it was
  not) and a docstring promising *"Stable ties (lower index first)"* (it did
  not). Hamming distances are heavily tied, so this was reachable: the
  un-widened form disagreed with `np.argsort(kind="stable")[:k]` on **36/36**
  tie-dense synthetic cases and **9/9** real `hamming_scan` outputs. Scoped
  honestly — every row it returned was still at the minimum distance. Now
  delegates to `remax.packing.stable_top_k`
  ([#29](https://github.com/oaustegard/remax_kb/pull/29)).

- **Sign convention at exactly 0.0**
  ([#30](https://github.com/oaustegard/remax_kb/pull/30)). `kb-reader.js`
  packed on `sum >= 0`; `remax.packing.encode_signs` packs on `X > 0`. At
  exactly `0.0` the two produced opposite bits, independent of float rounding.
  JS now uses strict `> 0`. **Issue #20 stays open** for the other half —
  numpy f32 BLAS and the JS f64 accumulation loop can land on opposite sides of
  zero for a *near*-zero projection, which is not closable without one reader
  changing its arithmetic.

- **HTML head metadata no longer depends on beautifulsoup4**
  ([#28](https://github.com/oaustegard/remax_kb/pull/28)).

### Added

- **`js/kb-reader.js` runs in CI against the Python reader**
  ([#28](https://github.com/oaustegard/remax_kb/pull/28)). Cross-reader checks
  went 36 → **115**, open-validation 29 → **41**
  ([#30](https://github.com/oaustegard/remax_kb/pull/30)).

- **SPEC_v2 validation steps 2, 5, 6 and 7** implemented in both readers, and
  the ones nobody can implement retracted from the spec rather than left as
  unmet MUSTs ([#29](https://github.com/oaustegard/remax_kb/pull/29),
  [#30](https://github.com/oaustegard/remax_kb/pull/30)). The JS reader now
  refuses remex-coded files by name rather than mis-reading them.

### Changed

- **`--projection` defaults to `srht`**
  ([#30](https://github.com/oaustegard/remax_kb/pull/30)), so a newly packed
  `.kbi` ships no rotation sidecar at all. The legacy no-projection fallback is
  pinned to `haar` and tested, so existing files keep reading.

## v0.4.0 and earlier

No tags were ever cut, so this section is the repository's history to date
rather than a release. Entries are grouped by the work they belong to and
derived from pull request titles; each links the PR that carries the detail.

### The v2 format

Split-index layout with hybrid retrieval and mutation
([#5](https://github.com/oaustegard/remax_kb/pull/5)), `rotations.f32` plus the
first JS reader ([#6](https://github.com/oaustegard/remax_kb/pull/6)), CLI
`migrate` with v1/v2 auto-detect and skill support
([#8](https://github.com/oaustegard/remax_kb/pull/8)), and incremental rebuilds
via content-addressed sync with a compaction policy
([#9](https://github.com/oaustegard/remax_kb/pull/9)).

### Projections and the rotation sidecar

An int8-quantized rotation sidecar, 4× smaller and recall-neutral
([#10](https://github.com/oaustegard/remax_kb/pull/10)); then a seed-only
Rademacher projection that drops the sidecar entirely
([#11](https://github.com/oaustegard/remax_kb/pull/11)); then SRHT at
Haar-grade recall, also seed-only
([#13](https://github.com/oaustegard/remax_kb/pull/13)) — which is what v0.5.0
above makes the default.

### The remex codec

Multi-bit codec (rotation + Lloyd-Max) added to v1 `.kb`
([#18](https://github.com/oaustegard/remax_kb/pull/18)), hardened with CI, a
reproducible benchmark and docs
([#21](https://github.com/oaustegard/remax_kb/pull/21)), then wired into the v2
`.kbi`/`.kbc` path ([#22](https://github.com/oaustegard/remax_kb/pull/22)).

### Embedders and retrieval quality

`JinaQ4ONNXEmbedder`, a 170 MB int4 runtime with fp32-parity retrieval
([#14](https://github.com/oaustegard/remax_kb/pull/14)); `LFM25Embedder` for
LiquidAI/LFM2.5-Embedding-350M
([#27](https://github.com/oaustegard/remax_kb/pull/27)); v2 hybrid search tuned
with a semantic floor and fusion knobs
([#26](https://github.com/oaustegard/remax_kb/pull/26)). A q4 head-to-head
found the official Optimum q4 dominates ours, and recorded the decision *not*
to upload ([#24](https://github.com/oaustegard/remax_kb/pull/24)).

### Performance

Hardware popcount for the 1-bit scan — ~10×, beating BLAS float cosine
([#16](https://github.com/oaustegard/remax_kb/pull/16)).

### Packaging and demos

Directory packer, Gemini embedder and skill quickstart
([#2](https://github.com/oaustegard/remax_kb/pull/2)), plus the live demos at
muninn.austegard.com ([#1](https://github.com/oaustegard/remax_kb/pull/1),
[#3](https://github.com/oaustegard/remax_kb/pull/3)).
