"""Head-to-head: official Optimum q4 ONNX vs remax_kb's q4 ONNX, against fp32.

Answers "does the official model_q4.onnx perform the same or better than ours?"
on BEIR NFCorpus (the corpus the JinaQ4ONNXEmbedder docstring claims parity on).

Both q4 files quantize the SAME fp32 retrieval ONNX, so the fair comparison is
each one's loss relative to fp32 (cannot saturate from either end — the metric
bench/RESULTS.md argues for), plus an actual qrels check since "performs" is
ultimately a retrieval-quality question.

Inputs (all local paths — fetch in an env that can reach HF Xet + GitHub assets):

  --fp32       jinaai .../onnx/model.onnx           (+ model.onnx_data sibling)
  --ours-q4    jina-v5-nano-mirror release model.q4.onnx
  --official-q4 jinaai .../onnx/model_q4.onnx        (+ model_q4.onnx_data sibling)
  --tokenizer  tokenizer.json (upstream repo)
  --corpus     dir with corpus.jsonl, queries.jsonl, qrels.tsv (BeIR NFCorpus
               layout). If omitted, attempts `datasets` load of BeIR/nfcorpus.
  --n-docs / --n-queries  subsample sizes (default 1500 / 100)

Run:
  pip install onnxruntime tokenizers numpy datasets
  python bench/bench_q4_official_vs_ours.py \
      --fp32 model.onnx --ours-q4 model.q4.onnx --official-q4 model_q4.onnx \
      --tokenizer tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from remax_kb.embedders import JinaONNXEmbedder


# --------------------------------------------------------------------------- #
# corpus loading
# --------------------------------------------------------------------------- #

def load_nfcorpus(corpus_dir: str | None):
    """Return (docs: dict[id,text], queries: dict[id,text], qrels: dict[qid,{did:rel}])."""
    if corpus_dir:
        d = Path(corpus_dir)
        docs = {}
        for line in (d / "corpus.jsonl").read_text().splitlines():
            o = json.loads(line)
            docs[o["_id"]] = (o.get("title", "") + " " + o.get("text", "")).strip()
        queries = {}
        for line in (d / "queries.jsonl").read_text().splitlines():
            o = json.loads(line)
            queries[o["_id"]] = o["text"]
        qrels: dict[str, dict[str, int]] = {}
        tsv = (d / "qrels.tsv").read_text().splitlines()
        for line in tsv[1:]:  # skip header
            qid, did, rel = line.split("\t")[:3]
            qrels.setdefault(qid, {})[did] = int(rel)
        return docs, queries, qrels

    # fallback: HF datasets
    from datasets import load_dataset
    corpus = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
    qds = load_dataset("BeIR/nfcorpus", "queries")["queries"]
    qrels_ds = load_dataset("BeIR/nfcorpus-qrels")["test"]
    docs = {r["_id"]: (r.get("title", "") + " " + r["text"]).strip() for r in corpus}
    queries = {r["_id"]: r["text"] for r in qds}
    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    return docs, queries, qrels


def subsample(docs, queries, qrels, n_docs, n_queries, seed=0):
    rng = np.random.default_rng(seed)
    test_qids = [q for q in queries if q in qrels]
    rng.shuffle(test_qids)
    qids = test_qids[:n_queries]
    keep_docs = {d for q in qids for d in qrels[q]}          # all judged-relevant docs
    pool = [d for d in docs if d not in keep_docs]
    rng.shuffle(pool)
    for d in pool[: max(0, n_docs - len(keep_docs))]:
        keep_docs.add(d)
    docs = {d: docs[d] for d in keep_docs}
    queries = {q: queries[q] for q in qids}
    qrels = {q: {d: r for d, r in qrels[q].items() if d in docs} for q in qids}
    return docs, queries, qrels


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def encode_all(model_path, tok, docs, queries, max_length=512, batch=64):
    """Encode docs + queries, mini-batched.

    JinaONNXEmbedder.encode runs its whole input as one onnxruntime batch; on a
    real corpus (1500+ docs) the attention-mask Expand alone asks for ~26 GB and
    OOMs. Mini-batching is numerically identical here — last-token pooling indexes
    each row's true token length and the L2-norm is per-row, so batch boundaries
    and per-batch padding never change a vector.
    """
    emb = JinaONNXEmbedder(model_path=model_path, tokenizer_path=tok, max_length=max_length)

    def run(texts, prompt):
        if not texts:
            return np.zeros((0, emb.full_dim), dtype=np.float32)
        return np.vstack(
            [emb.encode(texts[i : i + batch], prompt=prompt) for i in range(0, len(texts), batch)]
        )

    t0 = time.time()
    dvec = run(list(docs.values()), "document")
    qvec = run(list(queries.values()), "query")
    dt = time.time() - t0
    return dvec, qvec, dt


def ndcg_at_k(ranked_dids, rel, k=10):
    dcg = 0.0
    for i, did in enumerate(ranked_dids[:k]):
        g = rel.get(did, 0)
        if g:
            dcg += (2 ** g - 1) / np.log2(i + 2)
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal) if g)
    return dcg / idcg if idcg else 0.0


def retrieval_scores(dvec, qvec, doc_ids, query_ids, qrels, k=10):
    sims = qvec @ dvec.T                     # (Q, D), both L2-normalized
    order = np.argsort(-sims, axis=1)
    ndcgs = []
    for qi, qid in enumerate(query_ids):
        ranked = [doc_ids[j] for j in order[qi]]
        ndcgs.append(ndcg_at_k(ranked, qrels[qid], k))
    return float(np.mean(ndcgs)), sims, order


def fidelity_to_fp32(qvec_x, dvec_x, qvec_ref, dvec_ref, k=10):
    """How faithful is model X to fp32: per-doc cosine, recall@k vs fp32-kNN, Spearman rho."""
    doc_cos = float(np.mean((dvec_x * dvec_ref).sum(1)))
    sims_x = qvec_x @ dvec_x.T
    sims_ref = qvec_ref @ dvec_ref.T
    topk_x = np.argsort(-sims_x, axis=1)[:, :k]
    topk_ref = np.argsort(-sims_ref, axis=1)[:, :k]
    recalls = [len(set(a) & set(b)) / k for a, b in zip(topk_x, topk_ref)]
    # Spearman of per-query score vectors, averaged
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        ra = ra - ra.mean(); rb = rb - rb.mean()
        d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        return float((ra * rb).sum() / d) if d else 0.0
    rho = float(np.mean([spearman(sims_x[i], sims_ref[i]) for i in range(sims_x.shape[0])]))
    return doc_cos, float(np.mean(recalls)), rho


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--ours-q4", required=True)
    ap.add_argument("--official-q4", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--n-docs", type=int, default=1500)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64, help="forward-pass mini-batch size")
    a = ap.parse_args()

    docs, queries, qrels = load_nfcorpus(a.corpus)
    docs, queries, qrels = subsample(docs, queries, qrels, a.n_docs, a.n_queries)
    doc_ids, query_ids = list(docs), list(queries)
    print(f"corpus: {len(docs)} docs, {len(queries)} queries")

    out = {}
    for name, path in [("fp32", a.fp32), ("ours-q4", a.ours_q4), ("official-q4", a.official_q4)]:
        dvec, qvec, dt = encode_all(path, a.tokenizer, docs, queries, batch=a.batch)
        ndcg, _, _ = retrieval_scores(dvec, qvec, doc_ids, query_ids, qrels)
        mb = (Path(path).stat().st_size + Path(str(path) + "_data").stat().st_size
              if Path(str(path) + "_data").exists() else Path(path).stat().st_size) / 1e6
        out[name] = dict(dvec=dvec, qvec=qvec, ndcg=ndcg, mb=mb, secs=dt)
        print(f"  {name:12s} nDCG@10={ndcg:.4f}  size={mb:6.1f}MB  encode={dt:5.1f}s")

    ref = out["fp32"]
    print("\nfidelity to fp32 (per-doc cosine / recall@10-vs-fp32kNN / Spearman-rho):")
    for name in ("ours-q4", "official-q4"):
        m = out[name]
        cos, rec, rho = fidelity_to_fp32(m["qvec"], m["dvec"], ref["qvec"], ref["dvec"])
        print(f"  {name:12s} cos={cos:.4f}  recall@10={rec:.4f}  rho={rho:.4f}")

    print("\nverdict table:")
    print(f"  {'model':12s} {'nDCG@10':>8s} {'dNDCG_vs_fp32':>14s} {'MB':>7s} {'enc_s':>6s}")
    for name in ("fp32", "ours-q4", "official-q4"):
        m = out[name]
        print(f"  {name:12s} {m['ndcg']:8.4f} {m['ndcg']-ref['ndcg']:14.4f} {m['mb']:7.1f} {m['secs']:6.1f}")


if __name__ == "__main__":
    main()
