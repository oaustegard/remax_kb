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

## Format versions

- **v1 `.kb`** ([`SPEC.md`](./SPEC.md)) — the original single-file
  artifact: chunks + dense codes in one zip, dense Hamming top-k.
  Stable and fully supported. The fastest path to ship one file.
- **v2 `.kbi` + `.kbc/`** ([`SPEC_v2.md`](./SPEC_v2.md)) — a split
  index built for static-CDN, mutable, hybrid-retrieval deployments
  (the "search-on-mac" architecture). A hot `.kbi` (manifest +
  vectors + BM25 postings, one GET) and a cold byte-addressable
  `.kbc/` chunk store (one HTTP Range per hit). Adds BM25 + RRF
  fusion, tombstone-based mutation, and per-chunk `sha256` + Merkle
  verification.

**New builds that need hybrid retrieval, mutation, or CDN range-fetch
should target v2.** v1 remains the right choice for a single portable
file you hand someone. Upgrade an existing v1 file with
`remax-kb migrate` (no re-embedding — see Usage below).

## Live demos

Three `.kb` files published as sidecar artifacts:

- **`muninn.kb`** — full muninn.austegard.com corpus: 1238 chunks
  across 73 posts under `blog/`, `perch/`, `scratch/`. Hosted at
  [muninn.austegard.com/knowledge/muninn.kb](https://muninn.austegard.com/knowledge/muninn.kb).
  Same embedder/binarizer params as the subset; use this one for
  realistic retrieval against the whole site.
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

Query any of them locally:

```bash
curl -O https://muninn.austegard.com/knowledge/muninn.kb
remax-kb query muninn.kb "How does centered SimHash differ from random projection?" --k 3
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

#### Codec: 1-bit SimHash (default) vs multi-bit remex

By default a `.kb` stores 1-bit centered-SimHash codes (`dim*k/8` B/row,
Hamming scan). For a **higher-fidelity / mid-byte** alternative, the
optional **remex** codec stores multi-bit Lloyd-Max scalar-quantized
codes (`dim*bits/8` B/row):

```bash
remax-kb pack ./my-docs/ -o knowledge.kb --codec remex --bits 4 --dim 512 \
    --embedder jina-onnx          # needs:  pip install -e ".[remex]"
```

On general (isotropic) embedders like Jina, remex reproduces the fp32
ranking far more faithfully than 1-bit at equal bytes (`bench/RESULTS.md`).
On specialized, tightly-clustered embedders (e.g. SPECTER2) the ordering
can invert and 1-bit wins — so it's a per-embedder choice. remex does not
center, requires an L2-normalizing embedder, and is deterministic from
`(dim, bits, seed)`. Works for both v1 `.kb` and v2 `.kbi` (`--v2 --codec
remex --bits 4`); under v2 it fuses with BM25 like the default codec.

### Query a `.kb`

```bash
remax-kb query knowledge.kb "How does X work?" --k 5 --pretty
```

`query` and `info` auto-detect the format, so the same commands work
against a v2 `.kbi`:

```bash
remax-kb query knowledge.kbi "How does X work?" --k 5 --alpha 0.5  # weighted; omit for RRF
remax-kb info knowledge.kbi
```

### Pack a v2 split index (`.kbi` + `.kbc/`)

```bash
remax-kb pack ./my-docs/ -o knowledge.kbi --v2 --embedder gemini --gemini-dim 768
```

Writes `knowledge.kbi` (hot index) plus a sibling `knowledge.kbc/`
(chunk shards) in the same directory.

### Incrementally (re)build a v2 index — `sync`

```bash
remax-kb sync ./my-docs/ -o knowledge.kbi --embedder gemini --gemini-dim 768
```

For a *living* corpus, `sync` is what you run on every edit instead of a
fresh `pack`. It opens the existing `.kbi` (or creates one on first run),
diffs the directory against it content-addressed by `(chunk id, sha256)`,
and commits only the delta — **re-embedding nothing for unchanged
chunks**. Adds are embedded, edits are tombstoned-and-re-embedded,
removed chunks are tombstoned. Embedding cost scales with the change, not
the corpus.

Output is a JSON summary (`added`/`updated`/`deleted`/`unchanged`/
`embedded`, plus `live_count`/`total_rows`/`tombstone_ratio`). Tombstones
and the frozen corpus mean accumulate over many syncs, so `sync`
auto-compacts (a full re-embed that reclaims dead bytes and refreshes the
centering) once the tombstone ratio crosses `--compact-threshold`
(default `0.2`); pass `--no-compact` to disable.

### Migrate a v1 `.kb` to v2

```bash
remax-kb migrate knowledge.kb --out ./out/ --name knowledge
```

Reuses the v1 dense codes verbatim (the v2 `vectors.bin` is
byte-identical — no re-embedding, no embedder, no network), builds the
BM25 index and the `.kbc/` chunk store fresh, and writes
`out/knowledge.kbi` + `out/knowledge.kbc/`.

### Inspect a `.kb` / `.kbi` manifest

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

v2 split index (writer + hybrid reader):

```python
from remax_kb import KBWriter, KBv2, migrate_v1_to_v2
from remax_kb.pack import walk_directory

# Green-field v2 pack
writer = KBWriter.create(name="knowledge", output_dir="./out", embedder=emb,
                         dim=256, k=8, seed=0)
writer.add_chunks(walk_directory("./my-docs/"))
writer.commit()                      # writes out/knowledge.kbi + out/knowledge.kbc/

# Incremental rebuild on a living corpus — embeds only new/changed chunks
writer = KBWriter.open(name="knowledge", output_dir="./out", embedder=emb)
stats = writer.sync(walk_directory("./my-docs/"))   # diff by (id, sha256)
writer.commit()
print(stats.added, stats.updated, stats.deleted, stats.unchanged, stats.embedded)
if writer.should_compact():          # tombstones / mean drift past threshold
    writer.compact()                 # full re-embed of live rows

# Or upgrade an existing v1 file (no re-embedding)
migrate_v1_to_v2("knowledge.kb", "./out", name="knowledge")

# Hybrid query
kb = KBv2.open("./out/knowledge.kbi")
for h in kb.search_and_fetch("How does X work?", embedder=emb, k=3):
    print(f"[{h.chunk_id}, fused={h.fused:.4f}, verified={h.verified}] {h.text[:120]}...")
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
├── SPEC.md                         # the v1 format spec — citable
├── SPEC_v2.md                      # the v2 split-index spec
├── LICENSE                         # MIT
├── pyproject.toml                  # registers `remax-kb` console script
├── requirements-build.txt          # remax + torch + transformers (packer)
├── requirements-runtime.txt        # onnxruntime + tokenizers + numpy (reader)
├── remax_kb/
│   ├── __init__.py
│   ├── manifest.py                 # v1 dataclass + validation
│   ├── pack.py                     # corpus → .kb (pack, pack_directory)
│   ├── read.py                     # .kb → in-memory index + search()
│   ├── pack_v2.py                  # corpus → .kbi + .kbc/ (KBWriter)
│   ├── read_v2.py                  # .kbi → hybrid retrieval (KBv2)
│   ├── migrate.py                  # v1 .kb → v2 .kbi/.kbc (no re-embed)
│   ├── formats.py                  # v1/v2 zip-layout detection
│   ├── handlers.py                 # md/txt/html/pdf/rst extractors
│   ├── embedders.py                # Jina ONNX/torch + Gemini wrappers
│   ├── cli.py                      # `remax-kb pack|query|info|migrate` CLI
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
