---
name: remax-kb
description: Query a portable `.kb` knowledgebase artifact (remax_kb format v1). Use when the user references a `.kb` file (uploaded, or a path under `/mnt/project`) and asks a question about its contents. The skill downloads the embedder on first use, then runs Hamming-space top-k retrieval over the packed 1-bit embeddings.
---

# remax-kb — query a `.kb` knowledgebase

This skill exposes a `.kb` file's contents to a Claude.ai session via
top-k retrieval over 1-bit binary embeddings.

## When to invoke

Invoke this skill when **either**:

- The user has uploaded a `.kb` file (it appears at
  `/mnt/user-data/uploads/*.kb`), **or**
- A `.kb` file is reachable under `/mnt/project/` (or any path the
  user names explicitly), **and**
- The user asks a question whose answer is likely to be in the
  packed corpus.

If multiple `.kb` files are present, prefer the one in
`/mnt/user-data/uploads/` (most recent upload wins on ties).

## Resolution chain

1. `/mnt/user-data/uploads/*.kb` (most recent by mtime)
2. `/mnt/project/*.kb` (most recent by mtime)
3. An explicit path the user provides.

## How to invoke

Run `skill/search.py` from a Python tool call (the shell tool will do).
It is a thin CLI wrapper around `remax_kb.KB`:

```bash
python skill/search.py --kb <path_to_kb> --query "<user question>" --k 5
```

On first invocation per session the script downloads the ONNX
embedder asset (`model.onnx`, ~847 MB) from the URL recorded in the
`.kb`'s manifest into `~/.cache/remax_kb/jina-v5-nano/`. It also needs
a `tokenizer.json`; if not already present, the script will tell you
where to stage one. Subsequent queries reuse the loaded embedder.

## Output handling

The script prints JSON: a list of `{distance, id, text, meta}`
objects, sorted ascending by Hamming distance.

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
  hamming distance ≠ guaranteed best answer. Surface top-3 to top-5
  and let the user adjudicate.
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
