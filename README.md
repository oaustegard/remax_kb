# remax_kb

**A portable 1-bit binary-embedding knowledgebase format.**

`.kb` is a single-file artifact carrying a corpus's chunks plus their
[remax](https://github.com/oaustegard/remax) centered-SimHash binary
embeddings. The reference reader is pure numpy: query a `.kb` from a
vanilla container with `onnxruntime + tokenizers + numpy` — no torch,
no transformers, no peft.

This is an acknowledged-not-practical proof-of-concept. The artifact is
the format and the demonstration that 1-bit codes (the rank-correct
half of [one-bit-beats-two](https://muninn.austegard.com/blog/one-bit-beats-two.html))
produce a portable, deterministic, queryable knowledgebase runnable from
a Claude.ai project container.

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

### Pack a corpus

```bash
python scripts/pack_demo.py examples/tiny_corpus/ examples/tiny.kb \
    --dim 256 --k 8 --seed 0
```

### Query a `.kb`

```bash
python scripts/query_demo.py examples/tiny.kb "What does the author say about liberty?" --k 3
```

### Programmatic API

```python
from remax_kb import KB
from remax_kb.embedders import JinaONNXEmbedder

emb = JinaONNXEmbedder()  # downloads + caches model.onnx on first use
kb = KB.open("examples/tiny.kb")
hits = kb.search("What does the author say about liberty?", embedder=emb, k=3)
for dist, chunk in hits:
    print(f"[{chunk['id']}, hamming={dist}] {chunk['text'][:120]}...")
```

## Claude skill

`skill/SKILL.md` is installable into a claude.ai project. Once installed,
upload a `.kb` to the project (or to the session as a file) and ask
questions about it; the skill resolves the `.kb`, downloads the
embedder asset on first invocation, and runs top-k retrieval.

→ Skill instructions: [`skill/SKILL.md`](./skill/SKILL.md)

## Repo layout

```
remax_kb/
├── README.md
├── SPEC.md                         # the format spec — citable
├── LICENSE                         # MIT
├── pyproject.toml
├── requirements-build.txt          # remax + torch + transformers (packer)
├── requirements-runtime.txt        # onnxruntime + tokenizers + numpy (reader)
├── remax_kb/
│   ├── __init__.py
│   ├── manifest.py                 # dataclass + validation
│   ├── pack.py                     # corpus → .kb
│   ├── read.py                     # .kb → in-memory index + search()
│   ├── _hamming.py                 # numpy popcount scan
│   └── embedders.py                # ONNX + torch embedder wrappers
├── skill/
│   ├── SKILL.md
│   └── search.py
├── scripts/
│   ├── pack_demo.py
│   └── query_demo.py
├── examples/
│   └── tiny_corpus/                # ~30 chunks of public-domain text
└── tests/
    ├── test_roundtrip.py
    └── test_retrieval.py
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
