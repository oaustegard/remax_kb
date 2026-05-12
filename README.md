# remax_kb

**A portable, DIY knowledgebase index for small teams.**

`.kb` is a single-file artifact carrying a corpus's chunks plus their
[remax](https://github.com/oaustegard/remax) centered-SimHash binary
embeddings. The reference reader is pure numpy: query a `.kb` from a
vanilla container with `onnxruntime + tokenizers + numpy` — no torch,
no transformers, no peft.

The pitch: a small consulting shop, law firm, FOSS project, or
academic department with a few hundred to a few thousand docs and a
Claude Team subscription doesn't need to stand up Pinecone to get
useful retrieval. Pack your docs once, ship the `.kb` file to your
project, query via a 100-line skill. The artifact is single-file,
deterministic, and queryable from a constrained container.

The format itself is the demonstration that 1-bit codes (the rank-correct
half of [one-bit-beats-two](https://muninn.austegard.com/blog/one-bit-beats-two.html))
make this practical at the size that small teams actually have.

## Live demos

Two `.kb` files published as sidecar artifacts:

- **`muninn-subset.kb`** — 179 chunks from 11 curated personal-blog
  posts. Hosted at
  [muninn.austegard.com/knowledge/muninn-subset.kb](https://muninn.austegard.com/knowledge/muninn-subset.kb).
  Useful for AI/ML-curious readers. Try a query like *"How does
  centered SimHash differ from random projection?"*
- **`fastapi-docs.kb`** — the FastAPI developer docs
  ([fastapi.tiangolo.com](https://fastapi.tiangolo.com), MIT-licensed,
  ~150 pages). A credible, dev-team-flavored corpus for testing the
  format on something that isn't a personal blog. To build it
  locally:

  ```bash
  export GEMINI_API_KEY=...           # or use --embedder jina-onnx
  python examples/build_claude_docs_kb.py \
      --sitemap https://fastapi.tiangolo.com/sitemap.xml \
      --out fastapi-docs.kb \
      --embedder gemini --gemini-dim 768 --max-pages 150
  ```

  Once built it's published as a Release asset on this repo's
  [v0.1.0 release](https://github.com/oaustegard/remax_kb/releases/tag/v0.1.0).
  Try queries like *"how do I handle dependencies in path operations?"*
  or *"async vs sync route handlers"*.

  > Aside: the originally-intended target was the Anthropic developer
  > docs at [docs.claude.com](https://docs.claude.com). As of early
  > 2026 those pages are fully client-side rendered, so the raw HTML
  > contains only "Loading…" placeholders. The build script is
  > sitemap-driven and will work against docs.claude.com unchanged
  > once an SSR variant or a headless-browser fetcher is wired in.

Query either one locally:

```bash
curl -O https://muninn.austegard.com/knowledge/muninn-subset.kb
remax-kb query muninn-subset.kb "How does centered SimHash differ from random projection?" --k 3
```

→ Format spec: [`SPEC.md`](./SPEC.md)

## Why

A `.kb` file:

- Carries everything needed for retrieval — chunks, vectors, the corpus
  mean, the embedder pointer with SHA pin, the binarizer seed.
- Is bit-identical across machines given the same corpus and parameters
  (`dim`, `k`, `seed`).
- Is small — at `dim=256, k=8` each chunk is 256 bytes, so a 10k-chunk
  `.kb` is ~2.5 MB of vectors plus the JSONL text.
- Runs in a constrained container — the reader path needs only
  `onnxruntime`, `tokenizers`, `numpy`.

Companion to [`remax`](https://github.com/oaustegard/remax) (the
research library) and [`jina-v5-nano-mirror`](https://github.com/oaustegard/jina-v5-nano-mirror)
(the embedder).

## Install

### Packer (heavy — needs torch)

```bash
pip install -r requirements-build.txt
pip install -e .
```

### Reader / skill runtime (light — no torch)

```bash
pip install -r requirements-runtime.txt
pip install -e .
```

## Usage

### Pack a directory of mixed-format documents

```bash
remax-kb pack ./my-docs/ -o knowledge.kb --dim 256 --k 8 \
    --embedder jina-onnx          # or: gemini, jina-torch
```

Built-in handlers cover `.md / .markdown / .txt / .rst / .html / .htm
/ .pdf`. Pass your own with the Python API to extend (`.docx` etc.).
Frontmatter is stripped from markdown; HTML pulls main content out of
`<article>` / `<main>` / `<body>`; PDFs use `pypdf` and emit empty
text with a warning rather than crashing on encrypted / scanned files.

### Query a `.kb`

```bash
remax-kb query knowledge.kb "How does X work?" --k 5 --pretty
```

### Inspect a `.kb` manifest

```bash
remax-kb info knowledge.kb
```

### Programmatic API

```python
from remax_kb import KB, pack_directory
from remax_kb.embedders import JinaONNXEmbedder  # or GeminiEmbedder

emb = JinaONNXEmbedder()  # downloads + caches model.onnx on first use
kb_path = pack_directory("./my-docs/", "knowledge.kb", embedder=emb,
                         dim=256, k=8, seed=0)
kb = KB.open(kb_path)
hits = kb.search("How does X work?", embedder=emb, k=3)
for dist, chunk in hits:
    print(f"[{chunk['id']}, hamming={dist}] {chunk['text'][:120]}...")
```

## Embedders

Three implementations ship with the package; "use any embedder" is the
DIY-KB pitch, and the protocol is small enough to plug your own
provider in.

| Embedder | Path | When |
|----------|------|------|
| `JinaONNXEmbedder` | torch-free, ONNX | reader / skill runtime |
| `JinaTorchEmbedder` | heavy (torch + peft) | packer-side |
| `GeminiEmbedder` | API (`generativelanguage.googleapis.com`) | either side; needs `$GEMINI_API_KEY` |

For Cohere / OpenAI / Voyage / your own, implement the
[`Embedder` protocol](./remax_kb/embedders.py) — five attributes
(`model_id`, `task_adapter`, `pooling`, `full_dim`, `prompts`) plus
`fingerprint()` and `encode(texts, prompt=...)`. The `release_url` /
`release_sha256` fields are `None` for API-backed embedders, in which
case the `.kb` reader matches on `model_id` instead of verifying a
downloaded asset.

Example skeleton:

```python
class MyEmbedder:
    model_id = "vendor/my-model"
    model_revision = ""
    task_adapter = "retrieval"
    pooling = "native"
    full_dim = 1024
    normalize_l2 = True
    release_url = None
    release_sha256 = None
    prompts = {"query": "", "document": ""}

    def fingerprint(self) -> dict:
        return {"model_id": self.model_id, "task_adapter": self.task_adapter,
                "pooling": self.pooling, "full_dim": self.full_dim}

    def encode(self, texts, *, prompt) -> np.ndarray:
        # call your provider, return (N, full_dim) float32, L2-normalized
        ...
```

## Installing the skill in a Claude.ai project

`skill/SKILL.md` makes a `.kb` queryable from a Claude.ai project. The
skill itself is two files; installation is a 30-second copy-paste, not
a build step:

1. **Download** [`skill/SKILL.md`](./skill/SKILL.md) and
   [`skill/search.py`](./skill/search.py) from this repo.
2. **Add to your Claude.ai project.** In the project settings, attach
   both files as project knowledge / project files. (Claude.ai's
   "Skills" surface is rolling out unevenly — if you don't see a
   "Skills" tab, just drop the two files in as plain project
   knowledge; the SKILL.md frontmatter makes Claude pick it up.)
3. **Set the embedder credentials** the `.kb` expects:
   - If the `.kb` was packed with the Jina ONNX embedder: nothing to
     do; the model downloads on first use (~850 MB into
     `~/.cache/remax_kb/`).
   - If with Gemini: add `GEMINI_API_KEY` to the project environment.
4. **Make the `.kb` reachable.** Three options:
   - URL: include it in your message ("query against
     `https://example.com/foo.kb`") — the skill will fetch.
   - Upload: drop the file into the chat directly; the skill
     resolves `/mnt/user-data/uploads/*.kb`.
   - Project file: add `.kb` to your project's files; the skill
     resolves `/mnt/project/*.kb`.
5. **Ask a question.** Claude detects the `.kb`, runs the skill, and
   surfaces top-k retrieved chunks before answering.

→ Skill instructions: [`skill/SKILL.md`](./skill/SKILL.md)

## Repo layout

```
remax_kb/
├── README.md
├── SPEC.md                         # the format spec — citable
├── LICENSE                         # MIT
├── pyproject.toml                  # registers `remax-kb` console script
├── requirements-build.txt          # remax + torch + transformers (packer)
├── requirements-runtime.txt        # onnxruntime + tokenizers + numpy (reader)
├── remax_kb/
│   ├── __init__.py
│   ├── manifest.py                 # dataclass + validation
│   ├── pack.py                     # corpus → .kb (pack, pack_directory)
│   ├── read.py                     # .kb → in-memory index + search()
│   ├── handlers.py                 # md/txt/html/pdf/rst extractors
│   ├── embedders.py                # Jina ONNX/torch + Gemini wrappers
│   ├── cli.py                      # `remax-kb pack|query|info` CLI
│   └── _hamming.py                 # numpy popcount scan
├── skill/
│   ├── SKILL.md
│   └── search.py
├── scripts/
│   ├── pack_demo.py
│   └── query_demo.py
├── examples/
│   ├── tiny_corpus/                # ~30 chunks of public-domain text
│   └── build_claude_docs_kb.py     # builds the docs.claude.com demo .kb
└── tests/
    ├── test_roundtrip.py           # synthetic stub pack/read roundtrip
    ├── test_directory_pack.py      # mixed-format handlers + pack_directory
    ├── test_gemini.py              # GeminiEmbedder mocked
    ├── test_gemini_live.py         # gated on GEMINI_API_KEY
    ├── test_cli.py                 # remax-kb CLI surface
    └── test_retrieval.py           # gated on REMAX_KB_FULL + torch
```

## Validation

The `tests/` suite exercises the format end-to-end:

- `test_roundtrip.py` (7 tests) — pack + read mechanics with a
  deterministic synthetic-stub embedder. No torch, no model downloads.
- `test_retrieval.py::test_torch_query_lands_in_top_3` — packs
  `examples/tiny_corpus/` via the jina torch loader, queries with
  the same loader, asserts the topically-correct source file appears
  in top-3.
- `test_retrieval.py::test_onnx_matches_torch_top1` — the actual
  proof-of-concept claim: pack with torch, query the *same* `.kb`
  with both torch and ONNX embedders, assert they agree on top-1.
  Gated by `REMAX_KB_FULL=1` because it pulls the ~847 MB ONNX export.

To run the heavy tests locally, clone `jina-v5-nano-mirror` next to
this repo (or set `$JINA_V5_NANO_MIRROR_PATH`), install the heavy
deps, then:

```bash
pip install -r requirements-build.txt -r requirements-runtime.txt
REMAX_KB_FULL=1 pytest -q
```

## Scope

This is a proof-of-concept artifact, not production retrieval
infrastructure. Out of scope:

- Multi-model / multi-embedder `.kb`
- Streaming reads for `.kb` larger than RAM
- Quantized ONNX (the 847 MB cold-start is annoying but tolerable)
- HF Hub / PyPI publishing
- Backwards-compat versioning beyond `spec_version: "1"`

## Licenses

- `remax_kb` itself: MIT (this repo)
- `remax` (transitive): MIT
- `jinaai/jina-embeddings-v5-text-nano` (transitive, via
  `jina-v5-nano-mirror`): **CC-BY-NC-4.0** — non-commercial only.

If you build a `.kb` whose `embedder.model_id` is the jina-v5 nano,
distribution of that `.kb` inherits the CC-BY-NC-4.0 restriction.
