# q4 head-to-head — official Optimum ONNX vs ours (NFCorpus)

Resolves [remax_kb#23](https://github.com/oaustegard/remax_kb/issues/23). Run on
CCotw (claude.ai egress blocks both weight hosts; this box reaches HF Xet + the
GitHub release asset).

```bash
pip install onnxruntime tokenizers numpy datasets
python bench/bench_q4_official_vs_ours.py \
  --fp32 onnx/model.onnx --ours-q4 model.q4.onnx --official-q4 onnx/model_q4.onnx \
  --tokenizer tokenizer.json --n-docs 1500 --n-queries 100
```

Corpus: BEIR NFCorpus, subsampled to **2058 docs / 100 queries** (all judged-relevant
docs kept, then padded toward the 1500 floor — judged set alone exceeds it). 4 vCPU.

## Verdict table

| model | nDCG@10 | ΔnDCG vs fp32 | MB | encode (s) |
|---|--:|--:|--:|--:|
| fp32 | 0.4408 | 0.0000 | 849.3 | 815.6 |
| ours-q4 | 0.4250 | −0.0158 | 169.7 | 844.4 |
| **official-q4** | **0.4291** | **−0.0118** | **138.0** | 816.5 |

## Fidelity to fp32

| model | per-doc cosine | recall@10 vs fp32-kNN | Spearman ρ |
|---|--:|--:|--:|
| ours-q4 | 0.9743 | 0.8620 | 0.9764 |
| **official-q4** | **0.9763** | **0.8700** | **0.9801** |

## Decision: do NOT upload ours — recommend the official ONNX

The official `onnx/model_q4.onnx` **dominates our build on every axis**: it is
smaller (138.0 vs 169.7 MB, −19%) *and* strictly more faithful to fp32 — higher on
all four quality/fidelity metrics (nDCG@10, per-doc cosine, recall@10-vs-fp32kNN,
Spearman ρ), with the same direction on each. There is no axis where ours wins.

The hypothesis that ours would hold where the official degrades — on the premise
that our int8 embedding-table mop-up was a Jina-specific fix Optimum's generic q4
might skip — is **refuted**: Optimum's q4 edges ours on fidelity, so it handles the
EuroBERT embedding `Gather` at least as well. Per the issue's decision rule
(`official ≥ ours on fidelity AND smaller → ours is dominated`), **do not upload our
q4 to HF; close the upload loop.** `JinaQ4ONNXEmbedder` now **defaults to the official
upstream q4** (split ONNX from `jinaai/jina-embeddings-v5-text-nano-retrieval`, pinned
by commit + per-file sha256); our earlier build is reachable only via the example-only
`JinaOursQ4ONNXEmbedder` subclass, kept for provenance/reproducibility.

### Docstring cross-check (passes)

`JinaQ4ONNXEmbedder` claims "NFCorpus per-doc cosine 0.975 to fp32." This run measures
ours-q4 at **0.9743** (≈0.974) on a larger, independent NFCorpus subsample — consistent
with the prior 600-doc/120-query 0.975, so the claim holds.

### Encode time note

q4 (MatMulNBits int4) is **not** faster than fp32 on CPU here — all three encode in
~13–14 min; per-block int4 dequant offsets the smaller footprint. The q4 win is size
and download, not CPU latency. (The docstring's "~2x faster CPU decode" was not
reproduced on this run; treat it as workload-dependent.)

### Bench fix

`encode_all` is now mini-batched (`--batch`, default 64). The original one-shot
`emb.encode(all_docs)` asks the attention-mask `Expand` for ~26 GB at 2000 docs and
OOMs; mini-batching is numerically identical (last-token pooling indexes true token
lengths, L2-norm is per-row, so batch boundaries change nothing).
