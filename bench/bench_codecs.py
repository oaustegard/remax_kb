#!/usr/bin/env python3
"""Codec bench: remex (multi-bit Lloyd-Max) vs remax (1-bit SimHash) fidelity.

Self-contained (numpy + scipy + remax + remex; no downloads, no torch). Scores
each codec against the fp32 ranking itself — recall@k vs fp32-kNN and Spearman
rho — which, unlike recall-vs-qrels, does not saturate from either end. Each
codec is exercised exactly as remax_kb deploys it: remax centers on the corpus
mean (its design), remex does not (centering measurably hurts it).

Synthetic isotropic unit vectors stand in for a general embedder (e.g. Jina).
On real *specialized* embeddings (SPECTER2) the bit/recall ordering can invert
(1-bit > 2-bit) — that embedder-specific reversal is documented in
claude-workspace experiments/jina-remex-vs-remax (not reproducible on synthetic
isotropic data, so it is not asserted here).

    python bench/bench_codecs.py            # default n=2000, d=256, 200 queries
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from remax import StackedSignBitQuantizer
from remex import Quantizer

KS = (1, 5, 10)


def _unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _fp32_truth(D, Q):
    scores = Q @ D.T
    return scores, np.argsort(-scores, axis=1)


def _metrics(code_scores, gt_scores, gt_order, ks):
    m = code_scores.shape[0]
    rec = {k: 0.0 for k in ks}
    rho = 0.0
    for j in range(m):
        order = np.argsort(-code_scores[j])
        for k in ks:
            rec[k] += len(set(gt_order[j, :k].tolist()) & set(order[:k].tolist())) / k
        rho += spearmanr(code_scores[j], gt_scores[j]).statistic
    return {k: rec[k] / m for k in ks}, rho / m


def _remax_scores(D, Q, dim, k, seed):
    mean = D.mean(0).astype(np.float32)
    qz = StackedSignBitQuantizer(d=dim, k=k, seed=seed)
    dc = qz.encode(np.ascontiguousarray((D - mean)[:, :dim]))
    qc = qz.encode(np.ascontiguousarray((Q - mean)[:, :dim]))
    from remax_kb._hamming import hamming_scan
    return np.vstack([-hamming_scan(dc, qc[j]).astype(np.float32) for j in range(len(Q))])


def _remex_scores(D, Q, dim, bits, seed):
    qz = Quantizer(d=dim, bits=bits, seed=seed)
    comp = qz.encode(np.ascontiguousarray(D[:, :dim]))
    xhat = qz.decode(comp)
    return (Q[:, :dim] @ xhat.T).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    D = _unit(rng.standard_normal((args.n, args.d)).astype(np.float32))
    qi = rng.choice(args.n, args.queries, replace=False)
    Q = D[qi]
    gt_scores, gt_order = _fp32_truth(D, Q)

    rows = []  # (label, bytes, rec, rho)
    for bits in (1, 2, 4, 8):
        s = _remex_scores(D, Q, args.d, bits, args.seed)
        rec, rho = _metrics(s, gt_scores, gt_order, KS)
        rows.append((f"remex {bits}-bit", args.d * bits // 8, rec, rho))
    for k in (2, 4, 8):
        s = _remax_scores(D, Q, args.d, k, args.seed)
        rec, rho = _metrics(s, gt_scores, gt_order, KS)
        rows.append((f"remax k={k}", args.d * k // 8, rec, rho))

    rows.sort(key=lambda r: r[1])
    print(f"# Codec fidelity vs fp32 — isotropic synthetic "
          f"(n={args.n}, d={args.d}, {args.queries} queries)\n")
    print(f"{'codec':<14}{'B/row':>6}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'rho':>8}")
    print("-" * 49)
    for lab, by, rec, rho in rows:
        print(f"{lab:<14}{by:>6}{rec[1]:>7.3f}{rec[5]:>7.3f}{rec[10]:>7.3f}{rho:>8.3f}")

    # Matched-byte head-to-head (bits beat stacks).
    print("\n## Matched byte budgets (remex vs remax)")
    by_bytes = {}
    for lab, by, rec, rho in rows:
        by_bytes.setdefault(by, {})[lab.split()[0]] = (rec[10], rho)
    print(f"{'B/row':>6}  {'remex R@10/rho':>18}  {'remax R@10/rho':>18}  winner")
    for by in sorted(by_bytes):
        e = by_bytes[by]
        if "remex" in e and "remax" in e:
            er, ar = e["remex"], e["remax"]
            win = "remex" if er[1] > ar[1] else "remax"
            print(f"{by:>6}  {er[0]:.3f}/{er[1]:.3f}{'':>9}  {ar[0]:.3f}/{ar[1]:.3f}{'':>9}  {win}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
