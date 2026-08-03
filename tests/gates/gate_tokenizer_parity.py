#!/usr/bin/env python3
"""GATE — BM25 query tokenizer parity with the writer's indexer.

The writer indexes with ``bm25s.tokenize(live_texts, stopwords=None)``
(``pack_v2._build_bm25``). Both readers must split a *query* the same way, or
in-vocabulary terms become unreachable and the lexical arm silently scores 0.

ANCHOR: ``bm25s.tokenize`` itself — the writer's own library, an
implementation neither reader wrote. The expected tokenization is DERIVED by
running bm25s over a fixture here, never hardcoded, so this gate stays correct
if upstream changes its default ``token_pattern``.

Second, independent anchor for the end-to-end arm: the vocabulary inside the
committed fixture ``.kbi``, which was produced by the real writer.

    python3 tests/gates/gate_tokenizer_parity.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bm25s  # noqa: E402

from gate import Gate  # noqa: E402
from remax_kb.read_v2 import KB, tokenize_query  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURE = ROOT / "tests" / "fixtures" / "jsparity" / "jsparity.kbi"

# Query shapes a developer-docs KB actually receives. The identifier and
# accented cases are the ones the old `[a-z0-9]+` regex shredded; the
# single-character and punctuation cases pin the `\w\w+` minimum length and the
# empty-result path.
QUERIES = [
    "response_model",
    "get_user",
    "item_id path parameter",
    "status_code=404 HTTPException",
    "café résumé naïve",
    "MiXeD_CaSe Tokens",
    "a I x 9",
    "  !!  ...  ",
    "GET /items/{item_id}",
    "hamming distance between packed codes",
]

# The pre-fix reader tokenizer, kept HERE and nowhere else: it is this gate's
# known-bad, not a code path anything ships.
LEGACY_RE = re.compile(r"[a-z0-9]+")

# Vocabulary entries the writer created that the legacy regex cannot reach.
IDENTIFIERS = ("response_model", "get_user", "item_id", "status_code")


def legacy_tokenize(q: str) -> list[str]:
    return LEGACY_RE.findall(q.lower())


def bm25s_reference(q: str) -> list[str]:
    """The anchor: what the writer's own tokenizer does with this string."""
    toks = bm25s.tokenize(q, stopwords=None, return_ids=False, show_progress=False)
    return list(toks[0]) if toks else []


def node_tokenize(queries: list[str], *, legacy: bool) -> list[list[str]] | None:
    cmd = ["node", str(HERE / "js_tokenize.mjs")] + (["--legacy"] if legacy else [])
    try:
        r = subprocess.run(cmd, input=json.dumps(queries), capture_output=True,
                           text=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        raise RuntimeError(f"node failed: {r.stderr.strip()}")
    return json.loads(r.stdout)


def main() -> int:
    g = Gate("BM25 query tokenizer parity (readers vs bm25s)")

    ref = {q: bm25s_reference(q) for q in QUERIES}
    g.note(f"bm25s {bm25s.__version__}, default token_pattern "
           f"{bm25s.tokenize.__defaults__[1]!r}")

    # ---- arm 1: the Python reader ------------------------------------- #
    for q in QUERIES:
        got = tokenize_query(q)
        g.check(got == ref[q], f"python query tokenization: {q!r}",
                f"reader={got} bm25s={ref[q]}")

    # ---- arm 2: the shipped JS reader --------------------------------- #
    js = node_tokenize(QUERIES, legacy=False)
    if js is None:
        g.coverage("node not on PATH — the JS arm did not run, so this gate "
                   "says nothing about js/kb-reader.js on this machine. A "
                   "Python-only run cannot certify the JS reader.")
    else:
        for q, got in zip(QUERIES, js):
            g.check(got == ref[q], f"js query tokenization: {q!r}",
                    f"reader={got} bm25s={ref[q]}")

    # ---- arm 3: the consequence, end to end ---------------------------- #
    # A tokenization difference only matters because it moves BM25 scores. The
    # anchor here is the vocabulary the real writer put in the fixture .kbi.
    if FIXTURE.is_file():
        kb = KB.open(str(FIXTURE))
        vocab = json.loads(
            zipfile.ZipFile(FIXTURE).read("bm25/vocab.index.json")
        )
        for term in IDENTIFIERS:
            # Ground truth for a one-term query: the score vector you get by
            # handing bm25s the vocabulary entry the writer actually created.
            in_vocab = term in vocab
            want = kb._bm25.get_scores([term]) if in_vocab else None
            got = kb._bm25.get_scores(tokenize_query(term))
            ok = (in_vocab and want.max() > 0
                  and got.shape == want.shape and (got == want).all())
            g.check(ok, f"end-to-end lexical scores for {term!r} match the "
                        f"writer's own vocabulary entry",
                    f"in writer vocab={in_vocab} tokens={tokenize_query(term)} "
                    f"max score={float(got.max()):.4f} "
                    f"want max={float(want.max()):.4f}" if in_vocab
                    else f"{term!r} absent from the writer's vocabulary")
    else:
        g.coverage(f"fixture {FIXTURE} missing — the end-to-end scoring arm "
                   f"did not run")

    # ---- known-bad: the regex both readers used to use ------------------ #
    py_bad = [q for q in QUERIES if legacy_tokenize(q) != ref[q]]
    g.known_bad(
        "legacy `[a-z0-9]+` query regex is rejected (Python side)",
        rejected=bool(py_bad),
        detail=f"{len(py_bad)}/{len(QUERIES)} queries disagree with bm25s, "
               f"e.g. {py_bad[0]!r} -> {legacy_tokenize(py_bad[0])} "
               f"instead of {ref[py_bad[0]]}" if py_bad else "none disagreed",
        covers=("python query tokenization",),
    )
    if js is not None:
        js_bad_toks = node_tokenize(QUERIES, legacy=True)
        js_bad = [q for q, t in zip(QUERIES, js_bad_toks) if t != ref[q]]
        g.known_bad(
            "legacy `[a-z0-9]+` query regex is rejected (JS side, via node)",
            rejected=bool(js_bad),
            detail=f"{len(js_bad)}/{len(QUERIES)} queries disagree with bm25s",
            covers=("js query tokenization",),
        )
    if FIXTURE.is_file():
        kb = KB.open(str(FIXTURE))
        detail = {}
        all_wrong = True
        for t in IDENTIFIERS:
            want = kb._bm25.get_scores([t])
            toks = legacy_tokenize(t)
            got = kb._bm25.get_scores(toks) if toks else want * 0
            all_wrong &= not bool((got == want).all())
            detail[t] = (f"{toks} -> max {float(got.max()):.4f} "
                         f"(correct: {float(want.max()):.4f})")
        g.known_bad(
            "legacy regex mis-scores every identifier query on the fixture",
            rejected=all_wrong,
            detail="; ".join(f"{k}: {v}" for k, v in detail.items()),
            covers=("end-to-end lexical scores",),
        )

    # ---- coverage ------------------------------------------------------ #
    g.coverage(
        "stemming and stopword divergence is NOT covered: the writer hardcodes "
        "stopwords=None and no stemmer, and the reader hardcodes the same. "
        "Neither choice is recorded in the manifest, so a writer that starts "
        "passing a stemmer (or a stopword list) would leave the readers "
        "tokenizing differently and this gate would still be green."
    )
    g.coverage(
        "Unicode-table drift is NOT covered: Python's `\\w` follows the "
        "interpreter's UCD and the JS class follows Node's ICU. A codepoint "
        "whose general category differs between those two builds tokenizes "
        "differently and only shows up here if it appears in QUERIES."
    )
    g.coverage(
        "This gate covers tokenization only. Whether the two readers agree on "
        "decoded query codes and ranking is a separate claim — see "
        "tests/gates/gate_cross_reader.py."
    )
    return g.report()


if __name__ == "__main__":
    raise SystemExit(main())
