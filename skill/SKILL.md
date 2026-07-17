---
name: remax-kb
description: Query a portable `.kb` (v1) or `.kbi` (v2) knowledgebase artifact (remax_kb format). Use when the user references a `.kb`/`.kbi` file (uploaded, or a path under `/mnt/project`) and asks a question about its contents. The skill downloads the embedder on first use, then runs retrieval over the packed 1-bit embeddings — Hamming top-k for v1, hybrid dense+BM25 with RRF fusion for v2.
---

# remax-kb — query a `.kb` / `.kbi` knowledgebase

This skill exposes a knowledgebase artifact's contents to a Claude.ai
session via top-k retrieval over 1-bit binary embeddings. Two formats
are supported and **auto-detected**:

- **v1 `.kb`** — a single zip; dense Hamming-space top-k.
- **v2 `.kbi`** — a split index (hot `.kbi` + cold `.kbc/` chunk
  shards); hybrid retrieval (dense SimHash + BM25, fused with RRF) and
  lazy chunk fetch with per-chunk `sha256` verification.

## When to invoke

Invoke this skill when **either**:

- The user has uploaded a `.kb`/`.kbi` file (it appears at
  `/mnt/user-data/uploads/`), **or**
- A `.kb`/`.kbi` file is reachable under `/mnt/project/` (or any path
  the user names explicitly), **and**
- The user asks a question whose answer is likely to be in the
  packed corpus.

If multiple artifacts are present, prefer the one in
`/mnt/user-data/uploads/` (most recent upload wins on ties).

> **v2 note:** a `.kbi` needs its sibling `.kbc/` directory (the chunk
> store) in the same folder to return chunk text. If only the `.kbi`
> was uploaded, retrieval ranks hits but cannot fetch their text —
> ask the user to also upload the `.kbc/` shards.

## Resolution chain

1. `/mnt/user-data/uploads/*.{kb,kbi}` (most recent by mtime)
2. `/mnt/project/*.{kb,kbi}` (most recent by mtime)
3. An explicit path the user provides.

## How to invoke

Run `skill/search.py` from a Python tool call (the shell tool will do).
It is a thin CLI wrapper around `remax_kb.KB` (v1) / `remax_kb.KBv2`
(v2), routing on the detected format:

```bash
python skill/search.py --kb <path_to_kb_or_kbi> --query "<user question>" --k 10
```

### v2 tuning knobs

- **`--k`** (default 10) is a *cap*, not a target. The semantic floor is
  the real quality gate, so a narrow query returns fewer than `k` and a
  broad one fills to `k`. Raise it freely on a large corpus.
- **`--min-sim`** (default `auto`) is the **semantic floor**. Dense
  (SimHash) retrieval always ranks *something* nearest — even for a
  nonsense query — so its top hit would otherwise earn fusion rank-credit
  regardless of relevance. The floor drops dense candidates at or below a
  similarity threshold *before* fusion, so a query with no genuine dense
  match contributes no spurious dense signal (and, with no lexical hit
  either, returns nothing). `auto` scales the floor to the codec's bit
  budget and corpus size (the expected best-of-N noise level plus a
  margin). Pass `off` to disable, or a float to set it explicitly
  (dense_sim units: cosine for the remex codec, fraction of agreeing bits
  for the Hamming codec).
- **`--alpha 0..1`** switches fusion from RRF (default) to weighted
  dense/lexical; omit for parameter-free RRF.
- **`--over-fetch`** (default `max(k*8, 64)`) is how many candidates each
  modality feeds into fusion. Deeper pools let fusion surface a document
  that ranks mid-list in each modality but agrees across both.
- **`--rrf-c`** (default 60) is the RRF rank constant; lower sharpens
  toward rank-1.

A per-corpus default floor can be baked into the `.kbi` manifest under
`retrieval.min_sim`; an explicit `--min-sim` always overrides it.

On first invocation per session the script downloads the ONNX
embedder asset (`model.onnx`, ~847 MB) from the URL recorded in the
`.kb`'s manifest into `~/.cache/remax_kb/jina-v5-nano/`. It also needs
a `tokenizer.json`; if not already present, the script will tell you
where to stage one. Subsequent queries reuse the loaded embedder.

## Output handling

The script prints JSON: a list of hits under `hits`. The score fields
differ by format:

- **v1:** `{id, distance, text, meta}`, sorted ascending by Hamming
  distance (lower = closer).
- **v2:** `{id, fused, dense_distance, bm25_score, verified, text,
  meta}`, sorted descending by `fused` score (higher = better). A
  `verified: true` flag means the fetched chunk text matched its
  on-disk `sha256`.

After running it, surface the result to the user as:

> I found N chunks in `<kb_filename>`:
>
> **[chunk_id, hamming=X]** chunk text…
>
> **[chunk_id, hamming=Y]** chunk text…

Then answer the user's question using the surfaced chunks as
authoritative context. Cite chunk ids inline. Do **not** answer from
your prior knowledge if the chunks contradict it — the `.kb` is the
source of truth the user uploaded.

## Caveats

- 1-bit cosine LSH is rank-correct but noisy at small `k`; lower
  hamming distance ≠ guaranteed best answer. Surface the top handful
  and let the user adjudicate.
- The semantic floor (`--min-sim auto`) means a genuinely-unanswerable
  query can legitimately return **zero** hits. That is the floor working
  as intended, not a failure — tell the user the corpus has nothing above
  the noise floor rather than surfacing junk. If you suspect the floor is
  too aggressive for a niche query, retry with `--min-sim off` to inspect
  the raw ranking.
- The skill validates that the embedder fingerprint matches the
  manifest. A mismatch is unrecoverable — tell the user the `.kb` was
  built against a different model.
- The first query of a session is slow (~30s+) due to the embedder
  download. Subsequent queries are sub-second.

## Failure modes

- `ModuleNotFoundError: onnxruntime` / `tokenizers` / `numpy` —
  install the runtime deps:
  `pip install onnxruntime tokenizers numpy scipy 'remax @ git+https://github.com/oaustegard/remax.git'`
- `FileNotFoundError: tokenizer.json` — download the tokenizer from
  the upstream HF repo
  (`https://huggingface.co/jinaai/jina-embeddings-v5-text-nano/resolve/main/tokenizer.json`)
  to `~/.cache/remax_kb/jina-v5-nano/tokenizer.json` or set
  `$REMAX_KB_TOKENIZER_PATH`.
- Embedder fingerprint mismatch — the `.kb` was packed against a
  different model. There is no fallback; report to the user.
