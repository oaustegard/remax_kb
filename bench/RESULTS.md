# Codec benchmark — remex vs remax

Reproducible, download-free evidence that the `remex` codec (multi-bit Lloyd-Max
scalar quantization) is more faithful to the fp32 ranking than the `remax` 1-bit
SimHash codec at equal bytes. Run:

```bash
pip install -e ".[dev,remex]"
python bench/bench_codecs.py
```

## Why fidelity-to-fp32, not recall-vs-qrels

Scoring a codec by recall against human relevance judgments conflates *embedder
quality* with *quantization loss*: on an easy corpus everything ceilings, on a
hard one everything floors at the embedder's own (low) recall. Neither exposes
what the codec costs. So we score each code against **fp32's own ranking** —
recall@k vs fp32-kNN and Spearman ρ of the scores — which cannot saturate from
either end. (This is also remax's own bench metric.)

Each codec is exercised exactly as `remax_kb` deploys it: remax centers on the
corpus mean (its design); remex does not (centering measurably hurts it).

## Result (isotropic synthetic, n=2000, d=256, 200 queries)

| codec | B/row | R@5 | R@10 | ρ |
|---|--:|--:|--:|--:|
| remex 1-bit | 32 | 0.397 | 0.358 | 0.767 |
| remex 2-bit | 64 | 0.587 | 0.588 | 0.914 |
| remax k=2 | 64 | 0.358 | 0.347 | 0.746 |
| remex 4-bit | 128 | 0.828 | 0.836 | 0.980 |
| remax k=4 | 128 | 0.473 | 0.457 | 0.845 |
| remex 8-bit | 256 | 0.918 | 0.905 | 0.991 |
| remax k=8 | 256 | 0.588 | 0.587 | 0.912 |

**Matched byte budgets — remex wins every one:**

| B/row | remex ρ | remax ρ |
|--:|--:|--:|
| 64 | **0.914** | 0.746 |
| 128 | **0.980** | 0.845 |
| 256 | **0.991** | 0.912 |

**Bits beat stacks:** spending a byte budget on graded per-coordinate magnitude
(Lloyd-Max) beats spending it on more 1-bit sign rotations (stacked SimHash).

## Real embeddings + the embedder-specific caveat

On real **Jina v5-nano** vectors the same ordering holds (remex 4-bit @ d768 is
near-lossless, ρ≈0.998). But the bit/recall ordering is **embedder-specific**: on
specialized, tightly-clustered **SPECTER2** embeddings it can invert — 1-bit
*beats* 2-bit (the "one bit beats two" result that motivated remax's 1-bit
default). That reversal does not reproduce on synthetic isotropic data, so it is
not asserted here; the full SPECTER2-vs-Jina reconciliation (same harness, both
embedders) lives in `oaustegard/claude-workspace`
`experiments/jina-remex-vs-remax/`.

**Takeaway:** pick the codec per embedder. General/isotropic encoders (Jina) →
remex multi-bit. Specialized/tightly-clustered encoders (SPECTER2) → 1-bit remax.
The bench above is the per-embedder check; swap the synthetic corpus for your own
cached vectors to run it on your embedder.
