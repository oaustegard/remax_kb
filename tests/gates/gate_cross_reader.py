#!/usr/bin/env python3
"""GATE — cross-reader parity: js/kb-reader.js vs remax_kb.read_v2.

The .kbi format's central promise is that any conforming reader decodes an
artifact identically. Until now nothing executed ``js/kb-reader.js`` at all:
every "JS compat" assertion in ``tests/test_js_reader_compat.py`` is a Python
re-implementation of the JS source, and its own docstring concedes it assumes
"the JS implementation follows the Python logic faithfully (it does, by
construction)". A transcription bug is invisible to a transcription.

ANCHOR: the Python reader — a genuinely independent implementation (numpy /
BLAS / zipfile / bm25s) — over a committed fixture built by the real writer.
The JS reader runs for real, in Node, from the same bytes:

  * query codes must be BYTE-IDENTICAL;
  * top-k ordering (chunk_id sequence) must match, with distances, BM25 scores
    and RRF scores agreeing;
  * fetched chunk text and sha256 must match, and verify on both sides.

    python3 tests/gates/gate_cross_reader.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from build_jsparity_fixture import DeterministicEmbedder  # noqa: E402
from gate import Gate  # noqa: E402
from remax_kb.read_v2 import KB  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "jsparity"
KBI = FIXTURE_DIR / "jsparity.kbi"
KBC = FIXTURE_DIR / "jsparity.kbc"
QUERIES = FIXTURE_DIR / "queries.json"

TOP_K = 5
# NOTHING IS PINNED. Both readers run with no tuning arguments, so what is
# compared is their *defaults*. This gate used to pin over_fetch to 20 on both
# sides and min_sim to 'off', because the defaults genuinely disagreed — JS
# max(4k, 20) vs Python max(8k, 64), a hardcoded RRF constant vs an exposed
# rrf_c, and no JS min_sim at all. Pinning made the gate green by arranging for
# the divergence to be out of frame: the configuration a caller hits first was
# the one configuration parity was never asserted for.


def run_js(kbi: Path, kbc: Path, queries: Path = QUERIES, *,
           over_fetch: int | None = None, rrf_c: int | None = None,
           min_sim: str | float | None = None,
           reader: Path | None = None) -> dict:
    """Drive js/kb-reader.js. Every knob defaults to *not supplied*, so the
    reader's own default applies."""
    cmd = ["node", str(HERE / "js_cross_reader.mjs"),
           "--kbi", str(kbi), "--kbc", str(kbc), "--queries", str(queries),
           "--k", str(TOP_K)]
    if over_fetch is not None:
        cmd += ["--over-fetch", str(over_fetch)]
    if rrf_c is not None:
        cmd += ["--rrf-c", str(rrf_c)]
    if min_sim is not None:
        cmd += ["--min-sim", str(min_sim)]
    if reader is not None:
        cmd += ["--reader", str(reader)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"node reader failed: {r.stderr.strip()[:2000]}")
    return json.loads(r.stdout)


def run_probe(spec: dict, reader: Path | None = None) -> list[str]:
    """Encode caller-supplied vectors with js/kb-reader.js's encodeQueryCode().

    `reader` points the harness at a mutated copy of the reader, which is how
    the exact-zero known-bad is driven red.
    """
    d = Path(tempfile.mkdtemp(prefix="jsparity-probe-"))
    spec_path = d / "probe.json"
    spec_path.write_text(json.dumps(spec))
    cmd = ["node", str(HERE / "js_encode_probe.mjs"), "--spec", str(spec_path)]
    if reader is not None:
        cmd += ["--reader", str(reader)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"node probe failed: {r.stderr.strip()[:2000]}")
    return json.loads(r.stdout)["codes"]


def mutated_reader(old: str, new: str) -> Path:
    """A copy of js/kb-reader.js with one substring replaced. Asserts the
    substring was actually there, so a rename cannot turn the known-bad into a
    silent no-op that "passes"."""
    src = (ROOT / "js" / "kb-reader.js").read_text(encoding="utf-8")
    if src.count(old) != 1:
        raise RuntimeError(
            f"known-bad mutation target {old!r} appears {src.count(old)} times "
            f"in js/kb-reader.js; expected exactly 1")
    d = Path(tempfile.mkdtemp(prefix="jsparity-mutant-"))
    dst = d / "kb-reader.js"
    dst.write_text(src.replace(old, new), encoding="utf-8")
    return dst


def exact_zero_sign_parity(g: Gate) -> None:
    """Issue #20, mechanism (a): the sign convention AT exactly 0.0.

    Constructed, not sampled. Rademacher planes have entries in {-1, +1}, so
    for ``x = e_0 - e_1`` the projection onto output column ``c`` is
    ``planes[j, 0, c] - planes[j, 1, c]`` — exactly 0.0 wherever those two
    entries agree, and exactly ±2.0 where they do not. Both values are
    representable, and both are reached by any summation order in any
    precision: this probe isolates the *convention* from the *arithmetic*.

    The anchor is the packer itself — ``remax.StackedSignBitQuantizer.encode``,
    i.e. ``np.packbits(rotated > 0)`` — driven down the same code path
    ``pack_v2`` uses for ``--projection rademacher``. Whatever it does with a
    zero is by definition what the corpus bits mean.
    """
    from remax import StackedSignBitQuantizer

    from remax_kb.projection import rademacher_planes

    DIM, K, SEED = 32, 2, 12345
    planes = rademacher_planes(DIM, K, SEED).astype(np.float32)
    x = np.zeros(DIM, dtype=np.float32)
    x[0], x[1] = 1.0, -1.0
    vectors = [x, -x]

    proj = np.concatenate([x @ planes[j] for j in range(K)])
    n_zero = int((proj == 0.0).sum())
    total_bits = DIM * K
    g.note(f"exact-zero probe: {n_zero}/{total_bits} projected coordinates are "
           f"exactly 0.0 (the rest are exactly ±2.0)")
    g.bracket("the exact-zero probe actually reaches the zero case",
              value=float(n_zero), lo=0.0, hi=float(total_bits),
              why="0 zeros would make the sign-convention check vacuous; all "
                  "zeros would mean the probe lost the non-zero control bits "
                  "that prove the rest of the codeword still encodes")

    q = StackedSignBitQuantizer(d=DIM, k=K, seed=SEED)
    q.rotations_ = planes.astype(q.dtype)
    ref = [q.encode(np.asarray(v)[None, :])[0].tobytes().hex() for v in vectors]

    spec = {
        "dim": DIM, "k": K,
        "mean": [0.0] * DIM,
        "rotations": [float(v) for v in planes.ravel()],
        "vectors": [[float(t) for t in v] for v in vectors],
    }
    js = run_probe(spec)
    g.check(js == ref,
            "exact-zero projections pack the PACKER's way (strict `> 0`)",
            f"python={ref} js={js}")

    # known-bad: the convention this reader shipped with until 2026-08.
    js_ge = run_probe(spec, reader=mutated_reader("if (sum > 0) {",
                                                  "if (sum >= 0) {"))
    g.known_bad("js/kb-reader.js packing on `>= 0` diverges from the packer at "
                "an exact-zero projection",
                rejected=js_ge != ref,
                detail=f"packer={ref} js(>=0)={js_ge}",
                covers=("exact-zero projections pack the PACKER's way",))


# --------------------------------------------------------------------- #
# Second fixture: seed-only srht, built at gate runtime, large enough that
# over_fetch actually bites.
# --------------------------------------------------------------------- #

SRHT_WORDS = [
    "response_model", "get_user", "item_id", "status_code", "HTTPException",
    "dependency", "middleware", "pydantic", "validator", "router",
    "background_task", "websocket", "oauth2", "scopes", "session",
]


def build_srht_fixture(dest: Path) -> tuple[Path, Path, Path]:
    """Build a DEFAULT `.kbi` with the REAL writer, at runtime.

    Two things this fixture exists to prove, neither of which the committed
    haar fixture can:

    1. **The DEFAULT `.kbi` round-trips through the JS reader with NO sidecar
       shipped.** `js/kb-reader.js` MUST have the rotations for `haar` and
       cannot re-derive them, which is why every haar artifact carries up to
       9 MiB of planes for a consumer that may never need them. Since 2026-08
       the writer defaults to `srht`, which regenerates from
       `(dim, k, seed, srht_rounds)` on both sides — so the sidecar-free path
       is the ordinary one, and this arm is what proves it works end to end
       rather than in principle. No `projection=` is passed below, on purpose:
       flip the default back to `haar` and this goes red.
    2. **The readers' DEFAULTS agree.** 200 live rows, so the fusion pools do
       not saturate at `max(4k, 20)` the way seven rows do — `over_fetch` is
       observable here, and the known-bad below shows it.

    Rebuilt rather than committed: it is deterministic, it takes under a
    second, and committing it would add a second artifact to keep in sync with
    the writer for no gain.
    """
    from remax_kb.pack import Chunk
    from remax_kb.pack_v2 import KBWriter

    emb = DeterministicEmbedder()
    # NO projection argument: this fixture is built the way an ordinary caller
    # builds one, so the gate tests the writer's DEFAULT path rather than a
    # configuration it opted into. The assertions below then pin two things at
    # once — that the default is seed-only, and that the default artifact reads
    # identically in both readers.
    w = KBWriter.create(
        name="jsdefaults", output_dir=dest, embedder=emb,
        dim=32, k=4, seed=7, min_sim="auto",
    )
    corpus = []
    for i in range(200):
        words = [SRHT_WORDS[(i * 7 + j * 3) % len(SRHT_WORDS)] for j in range(6)]
        corpus.append(Chunk(id=f"doc#{i:04d}",
                            text=f"chunk {i} covers " + " ".join(words) + ".",
                            meta={"i": i}))
    w.add_chunks(corpus)
    w.commit()

    # Queries: two exact chunk texts (dense-perfect, so they survive the
    # 'auto' floor) and three lexical ones that hit deep in the postings.
    texts = [corpus[3].text, corpus[137].text,
             "response_model dependency router",
             "status_code HTTPException validator",
             "session oauth2 scopes websocket"]
    qvecs = emb.encode(texts, prompt="query")
    qpath = dest / "queries.json"
    qpath.write_text(json.dumps({
        "queries": [{"text": t, "embedding": [float(x) for x in qvecs[i]]}
                    for i, t in enumerate(texts)],
    }, indent=1))
    return dest / "jsdefaults.kbi", dest / "jsdefaults.kbc", qpath


def defaults_parity(g: Gate) -> None:
    """Item 3 + item 5: the readers' defaults, and the sidecar-free path."""
    d = Path(tempfile.mkdtemp(prefix="jsdefaults-"))
    kbi, kbc, queries = build_srht_fixture(d)

    with zipfile.ZipFile(kbi) as zf:
        names = zf.namelist()
        proj = json.loads(zf.read("manifest.json"))["binarizer"]["projection"]
    rot_entries = [n for n in names if n.startswith("binarizer/rotations")]
    g.note(f"default fixture: projection={proj}, {kbi.stat().st_size} B, "
           f"entries={names}")
    g.check(proj == "srht",
            "the writer's DEFAULT projection is seed-only (srht)",
            f"manifest says projection={proj!r}; nothing was passed to "
            f"KBWriter.create")
    g.check(not rot_entries,
            "a DEFAULT-built .kbi ships NO binarizer/rotations.* entry",
            f"rotation entries present: {rot_entries}" if rot_entries
            else "none — the projection regenerates from (dim, k, seed, "
                 "srht_rounds) on both sides")

    # ---- defaults: NOTHING supplied on either side ---------------------- #
    py = run_python(kbi, queries)
    js = run_js(kbi, kbc, queries)

    g.check(js["has_rotation_entry"] is False,
            "js/kb-reader.js opens the sidecar-free DEFAULT .kbi and searches "
            "it",
            f"live_count={js['live_count']} "
            f"hits per query={[len(q['hits']) for q in js['queries']]}")
    g.check(all(len(q["hits"]) > 0 for q in js["queries"])
            and all(len(q["hits"]) > 0 for q in py["queries"]),
            "every query returns hits on both readers (the floor did not empty "
            "the result set)",
            f"python={[len(q['hits']) for q in py['queries']]} "
            f"js={[len(q['hits']) for q in js['queries']]}")

    g.check(py["default_over_fetch"] == js["default_over_fetch"]
            == max(TOP_K * 8, 64),
            "default over_fetch agrees between readers",
            f"python={py['default_over_fetch']} js={js['default_over_fetch']} "
            f"(python's value is OBSERVED at the fusion call, not restated)")
    g.check(py["rrf_c_default"] == js["rrf_c_default"] == 60,
            "default RRF constant agrees between readers",
            f"python={py['rrf_c_default']} js={js['rrf_c_default']}")
    g.check(py["resolved_min_sim"] is not None
            and js["resolved_min_sim"] is not None
            and abs(py["resolved_min_sim"] - js["resolved_min_sim"]) < 1e-12,
            "manifest retrieval.min_sim='auto' resolves to the same floor in "
            "both readers",
            f"python={py['resolved_min_sim']!r} js={js['resolved_min_sim']!r}")

    compare(g, py, js, label="srht/defaults")

    # ---- caller-supplied floor MUST beat the manifest -------------------- #
    # SPEC_v2 §retrieval.min_sim: "a caller-supplied floor MUST take precedence
    # over the manifest value". Asserted in both directions: 'off' must
    # override a manifest that says 'auto', on both readers, identically.
    py_off = run_python(kbi, queries, min_sim="off")
    js_off = run_js(kbi, kbc, queries, min_sim="off")
    g.check(py_off["resolved_min_sim"] is None
            and js_off["resolved_min_sim"] is None,
            "an explicit min_sim='off' overrides the manifest's 'auto' on both "
            "readers",
            f"python={py_off['resolved_min_sim']!r} "
            f"js={js_off['resolved_min_sim']!r}")
    moved = sum(
        [h["chunk_id"] for h in a["hits"]] != [h["chunk_id"] for h in b["hits"]]
        for a, b in zip(py["queries"], py_off["queries"]))
    g.bracket("the floor is load-bearing: turning it off changes results",
              value=float(moved), lo=0.0, hi=float(len(py["queries"])),
              hi_inclusive=True,
              why="if 0, the min_sim parity checks above are comparing two "
                  "readers that both happen to do nothing")
    compare(g, py_off, js_off, label="srht/min_sim=off")

    # ---- a non-default rrf_c must plumb through on both sides ------------ #
    py_c5 = run_python(kbi, queries, rrf_c=5, min_sim="off")
    js_c5 = run_js(kbi, kbc, queries, rrf_c=5, min_sim="off")
    compare(g, py_c5, js_c5, label="srht/rrf_c=5")
    c_moved = sum(
        [h["chunk_id"] for h in a["hits"]] != [h["chunk_id"] for h in b["hits"]]
        for a, b in zip(py_off["queries"], py_c5["queries"]))
    g.bracket("rrf_c is load-bearing: c=5 ranks differently from c=60",
              value=float(c_moved), lo=0.0, hi=float(len(py["queries"])),
              hi_inclusive=True,
              why="if 0, 'both readers agree at rrf_c=5' says nothing about "
                  "whether either reader read the argument")

    # ---- known-bads ------------------------------------------------------ #
    # 1. the over_fetch default this reader shipped with until 2026-08.
    mutant = mutated_reader("  return Math.max(k * 8, 64);",
                            "  return Math.max(k * 4, 20);")
    js_of = run_js(kbi, kbc, queries, min_sim="off", reader=mutant)
    n_of = disagreement_count(py_off, js_of, "order")
    g.known_bad("the old JS over_fetch default max(4k,20) diverges from "
                "Python's max(8k,64)",
                rejected=n_of > 0,
                detail=f"{n_of}/{len(py_off['queries'])} queries return a "
                       f"different top-{TOP_K} order",
                covers=("default over_fetch agrees between readers",
                        "top-5 ordering matches (srht/min_sim=off)",
                        "per-hit scores match (srht/min_sim=off)",
                        "fetched chunk bytes + sha256 match and verify "
                        "(srht/min_sim=off)"))

    # 2. a reader that ignores the manifest's retrieval.min_sim — the exact
    #    behaviour SPEC_v2 permits ("a reader that ignores it entirely is
    #    conforming") and which this reader had. Conforming is not the same as
    #    matching, and the format's promise is that readers match.
    mutant2 = mutated_reader(
        "      minSim = (r && r.min_sim !== undefined) ? r.min_sim : null;",
        "      minSim = null;")
    js_nofloor = run_js(kbi, kbc, queries, reader=mutant2)
    n_ms = disagreement_count(py, js_nofloor, "order")
    g.known_bad("a JS reader that ignores manifest retrieval.min_sim diverges "
                "from Python at defaults",
                rejected=n_ms > 0,
                detail=f"{n_ms}/{len(py['queries'])} queries return a "
                       f"different top-{TOP_K} order",
                covers=("manifest retrieval.min_sim='auto' resolves",
                        "top-5 ordering matches (srht/defaults)",
                        "per-hit scores match (srht/defaults)",
                        "fetched chunk bytes + sha256 match and verify "
                        "(srht/defaults)"))

    # 3. a reader that accepts rrfC but drops it on the floor.
    mutant3 = mutated_reader("    const C = rrfC;", "    const C = 60;")
    js_c = run_js(kbi, kbc, queries, rrf_c=5, min_sim="off", reader=mutant3)
    n_c = disagreement_count(py_c5, js_c, "order")
    g.known_bad("a JS reader that ignores its rrfC argument diverges from "
                "Python at rrf_c=5",
                rejected=n_c > 0,
                detail=f"{n_c}/{len(py_c5['queries'])} queries return a "
                       f"different top-{TOP_K} order",
                covers=("top-5 ordering matches (srht/rrf_c=5)",
                        "per-hit scores match (srht/rrf_c=5)",
                        "fetched chunk bytes + sha256 match and verify "
                        "(srht/rrf_c=5)"))


def run_python(kbi: Path, queries: Path = QUERIES, **search_kw) -> dict:
    kb = KB.open(str(kbi))
    emb = DeterministicEmbedder()
    spec = json.loads(queries.read_text())

    # OBSERVE the defaults search() computes; do not restate them. `max(8k, 64)`
    # and `60` live inside the method body, so the only honest way to report
    # what Python used is to watch the value arrive at the fusion call. A copy
    # of the expression here would agree with the JS side even if both had
    # drifted away from the reader they claim to describe.
    import remax_kb.read_v2 as _rv2
    seen: dict[str, int] = {}
    original = _rv2._fuse_ranks

    def spy(dense, lex, *, over_fetch, alpha, rrf_c=60):
        seen["over_fetch"] = over_fetch
        seen["rrf_c"] = rrf_c
        return original(dense, lex, over_fetch=over_fetch, alpha=alpha,
                        rrf_c=rrf_c)

    out = {
        "live_count": kb.live_count,
        "total_rows": kb.manifest["chunks"]["total_rows"],
        "row_bytes": kb._row_bytes,
        "resolved_min_sim": kb._resolve_min_sim(search_kw.get("min_sim")),
        "queries": [],
    }
    _rv2._fuse_ranks = spy
    try:
        _python_queries(kb, emb, spec, out, search_kw)
    finally:
        _rv2._fuse_ranks = original
    out["default_over_fetch"] = seen.get("over_fetch")
    out["rrf_c_default"] = seen.get("rrf_c")
    return out


def _python_queries(kb, emb, spec, out, search_kw) -> None:
    for q in spec["queries"]:
        vec = np.asarray(q["embedding"], dtype=np.float32)
        hits = kb.search(q["text"], embedder=emb, k=TOP_K, alpha=None,
                         **search_kw)
        kb.fetch(hits)
        out["queries"].append({
            "text": q["text"],
            "query_code_hex": kb.encode_query_code(vec).tobytes().hex(),
            "hits": [{
                "chunk_id": h.chunk_id, "row": h.row,
                "dense_dist": h.dense_dist, "bm25_score": h.bm25_score,
                "fused": h.fused, "text": h.text, "sha256": h.sha256,
                "verified": h.verified,
            } for h in hits],
        })


def tamper(entry: str, mutate) -> Path:
    """Rewrite the fixture .kbi into a temp dir with one zip entry mutated."""
    d = Path(tempfile.mkdtemp(prefix="jsparity-bad-"))
    dst = d / KBI.name
    with zipfile.ZipFile(KBI) as src, \
            zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as out:
        for name in src.namelist():
            data = src.read(name)
            out.writestr(name, mutate(data) if name == entry else data)
    return dst


def compare(g: Gate, py: dict, js: dict, label: str = "haar/fixture") -> None:
    """Assert the two readers agree, tagging every check with `label` so one
    gate can carry several configurations without collapsing them into
    same-named checks that shadow each other in the report."""
    g.check(py["live_count"] == js["live_count"]
            and py["total_rows"] == js["total_rows"]
            and py["row_bytes"] == js["row_bytes"],
            f"readers agree on live_count / total_rows / row_bytes ({label})",
            f"python={py['live_count']}/{py['total_rows']}/{py['row_bytes']} "
            f"js={js['live_count']}/{js['total_rows']}/{js['row_bytes']}")

    for p, j in zip(py["queries"], js["queries"]):
        q = p["text"][:60]
        g.check(p["query_code_hex"] == j["query_code_hex"],
                f"query code is byte-identical ({label}): {q!r}",
                f"python={p['query_code_hex']} js={j['query_code_hex']}")

        p_ids = [h["chunk_id"] for h in p["hits"]]
        j_ids = [h["chunk_id"] for h in j["hits"]]
        g.check(p_ids == j_ids and len(p_ids) > 0,
                f"top-{TOP_K} ordering matches ({label}): {q!r}",
                f"python={p_ids} js={j_ids}")

        if p_ids == j_ids:
            dist_ok = all(a["dense_dist"] == b["dense_dist"]
                          for a, b in zip(p["hits"], j["hits"]))
            fused_ok = all(abs((a["fused"] or 0) - (b["fused"] or 0)) < 1e-12
                           for a, b in zip(p["hits"], j["hits"]))
            bm_ok = all(
                (a["bm25_score"] is None) == (b["bm25_score"] is None)
                and (a["bm25_score"] is None
                     or abs(a["bm25_score"] - b["bm25_score"])
                     <= 1e-6 * max(1.0, abs(a["bm25_score"])))
                for a, b in zip(p["hits"], j["hits"]))
            g.check(dist_ok and fused_ok and bm_ok,
                    f"per-hit scores match ({label}): {q!r}",
                    f"hamming={dist_ok} rrf={fused_ok} bm25={bm_ok}")

            text_ok = all(a["text"] == b["text"] and a["sha256"] == b["sha256"]
                          for a, b in zip(p["hits"], j["hits"]))
            verified = all(a["verified"] and b["verified"]
                           for a, b in zip(p["hits"], j["hits"]))
            g.check(text_ok and verified,
                    f"fetched chunk bytes + sha256 match and verify ({label}): {q!r}",
                    f"identical={text_ok} both_verified={verified}")
        else:
            g.check(False, f"per-hit scores match ({label}): {q!r}",
                    "ordering differed")
            g.check(False,
                    f"fetched chunk bytes + sha256 match and verify ({label}): {q!r}",
                    "ordering differed")


def disagreement_count(py: dict, js: dict, field: str) -> int:
    if field == "query_code_hex":
        return sum(p[field] != j[field]
                   for p, j in zip(py["queries"], js["queries"]))
    if field == "order":
        return sum([h["chunk_id"] for h in p["hits"]]
                   != [h["chunk_id"] for h in j["hits"]]
                   for p, j in zip(py["queries"], js["queries"]))
    if field == "text":
        return sum(any(a["text"] != b["text"]
                       for a, b in zip(p["hits"], j["hits"]))
                   for p, j in zip(py["queries"], js["queries"]))
    raise ValueError(field)


def sign_margin(g: Gate) -> None:
    """How close does any fixture query come to a sign flip?

    The two readers project with different arithmetic (numpy float32 BLAS vs a
    naive float64 JS loop), so byte-identical codes are only a stable claim
    while every projected coordinate sits comfortably away from zero. Measure
    both, rather than assuming.
    """
    kb = KB.open(str(KBI))
    with zipfile.ZipFile(KBI) as zf:
        rot = np.frombuffer(zf.read("binarizer/rotations.f32"),
                            dtype="<f4").reshape(kb._k, kb._dim, kb._dim)
    spec = json.loads(QUERIES.read_text())
    worst_margin, worst_disc = np.inf, 0.0
    for q in spec["queries"]:
        v = np.asarray(q["embedding"], dtype=np.float32)
        x32 = (v - kb._mean)[: kb._dim]
        p32 = np.concatenate([x32 @ rot[j] for j in range(kb._k)])
        p64 = np.concatenate([x32.astype(np.float64) @ rot[j].astype(np.float64)
                              for j in range(kb._k)])
        worst_margin = min(worst_margin, float(np.abs(p64).min()))
        worst_disc = max(worst_disc, float(np.abs(p32 - p64).max()))
    g.note(f"smallest |projection| over the fixture = {worst_margin:.3e}; "
           f"largest float32-vs-float64 discrepancy = {worst_disc:.3e} "
           f"(ratio {worst_margin / worst_disc:.0f}x)")
    g.bracket("fixture sign margin is 100x the arithmetic discrepancy",
              value=worst_margin / max(worst_disc, 1e-30),
              lo=100.0, hi=1e12,
              why="below this the byte-identical claim is a coin flip, not a "
                  "property; above 1e12 would mean the discrepancy measurement "
                  "collapsed and is no longer measuring anything")


def main() -> int:
    g = Gate("cross-reader parity — js/kb-reader.js vs remax_kb.read_v2")

    if shutil.which("node") is None:
        print("node is not on PATH; this gate cannot run.", file=sys.stderr)
        return 2
    for p in (KBI, KBC, QUERIES):
        if not p.exists():
            print(f"missing fixture {p}; run "
                  f"tests/fixtures/build_jsparity_fixture.py", file=sys.stderr)
            return 2

    node_v = subprocess.run(["node", "--version"], capture_output=True,
                            text=True).stdout.strip()
    g.note(f"node {node_v}, fixture {KBI.relative_to(ROOT)} "
           f"({KBI.stat().st_size} B), k={TOP_K}, no knob pinned on either "
           f"side")

    py = run_python(KBI)
    js = run_js(KBI, KBC)

    # The fixture's stored query embeddings must be what the fixture's embedder
    # produces, or the two readers are not being handed the same vector.
    emb = DeterministicEmbedder()
    spec = json.loads(QUERIES.read_text())
    regen = emb.encode([q["text"] for q in spec["queries"]], prompt="query")
    stored = np.asarray([q["embedding"] for q in spec["queries"]], dtype=np.float32)
    g.check(np.array_equal(regen, stored),
            "committed query embeddings reproduce from the fixture embedder",
            f"max abs delta = {float(np.abs(regen - stored).max()):.3e}")

    # Exonerate the instrument before comparing readers: the Python reader
    # RECOMPUTES the haar planes from (dim, k, seed) while the JS reader reads
    # the shipped sidecar. If the installed remax no longer generates the
    # planes this fixture was built with, every comparison below is noise —
    # so fail here, with a message that names the cause.
    from remax import StackedSignBitQuantizer
    kb0 = KB.open(str(KBI))
    with zipfile.ZipFile(KBI) as zf:
        shipped = zf.read("binarizer/rotations.f32")
    regen_rot = StackedSignBitQuantizer(
        d=kb0._dim, k=kb0._k, seed=kb0._seed
    ).rotations_.astype("<f4").tobytes()
    g.check(regen_rot == shipped,
            "installed remax regenerates the planes the fixture ships",
            "identical" if regen_rot == shipped else
            "DIFFERENT — the installed remax generates different haar planes "
            "than this fixture was built with; rebuild it with "
            "tests/fixtures/build_jsparity_fixture.py before reading anything "
            "into the comparisons below")

    sign_margin(g)
    exact_zero_sign_parity(g)
    compare(g, py, js)
    defaults_parity(g)

    # ---- known-bad 1: the JS reader's rotation sidecar is perturbed ------ #
    # Shape of a real bug: a wrong dequant, a transposed plane, an endianness
    # slip — anything that lands the query in a different sign-space.
    def bump_one_plane(data: bytes) -> bytes:
        arr = np.frombuffer(data, dtype="<f4").copy()
        arr[0] = -arr[0]          # flip the sign of a single plane entry
        return arr.tobytes()

    bad_rot = tamper("binarizer/rotations.f32", bump_one_plane)
    js_bad = run_js(bad_rot, KBC)
    n = disagreement_count(py, js_bad, "query_code_hex")
    g.known_bad("one flipped rotation entry (JS side only) breaks query-code "
                "byte identity",
                rejected=n > 0,
                detail=f"{n}/{len(py['queries'])} query codes diverge",
                covers=("query code is byte-identical (haar/fixture)",))

    # ---- known-bad 2: corpus codes bit-reversed within each byte --------- #
    # The exact bug the JS bit-pack comment warns about (packbits big vs little
    # bitorder). Ordering must move.
    REV = bytes(int(f"{b:08b}"[::-1], 2) for b in range(256))
    bad_vec = tamper("vectors.bin", lambda d: bytes(REV[b] for b in d))
    js_bad2 = run_js(bad_vec, KBC)
    n2 = disagreement_count(py, js_bad2, "order")
    g.known_bad("bit-reversed vectors.bin (JS side only) breaks top-k ordering",
                rejected=n2 > 0,
                detail=f"{n2}/{len(py['queries'])} queries return a different "
                       f"top-{TOP_K} order",
                covers=("top-5 ordering matches (haar/fixture)",
                        "per-hit scores match (haar/fixture)"))

    # ---- known-bad 3: the JS reader's chunk shard is perturbed ----------- #
    tmp_kbc = Path(tempfile.mkdtemp(prefix="jsparity-badkbc-")) / KBC.name
    shutil.copytree(KBC, tmp_kbc)
    shard = tmp_kbc / "shard-0000.bin"
    raw = bytearray(shard.read_bytes())
    # Flip a case bit in the final byte of row 0's text — a live row that every
    # query's top-k reaches. Located via the chunk_map, not a guessed offset.
    info = KB.open(str(KBI))._chunk_map_row(0)
    pos = info["byte_offset"] + info["byte_length"] - 1
    raw[pos] ^= 0x20
    shard.write_bytes(bytes(raw))
    js_bad3 = run_js(KBI, tmp_kbc)
    n3 = disagreement_count(py, js_bad3, "text")
    g.known_bad("one flipped byte in the JS reader's shard breaks chunk-bytes "
                "parity",
                rejected=n3 > 0,
                detail=f"{n3}/{len(py['queries'])} queries return differing "
                       f"chunk text",
                covers=("fetched chunk bytes + sha256 match and verify (haar/fixture)",))

    # ---- known-bad 4: a tombstone flag set on the JS side only ---------- #
    # Exercises the live→absolute row mapping both readers build. A reader that
    # refuses to load the inconsistent artifact counts as a rejection.
    def set_tombstone(data: bytes) -> bytes:
        b = bytearray(data)
        b[2] |= 0x01  # flags byte of row 0
        return bytes(b)

    bad_map = tamper("chunk_map.bin", set_tombstone)
    try:
        js_bad4 = run_js(bad_map, KBC)
        rejected4 = js_bad4["live_count"] != py["live_count"]
        detail4 = f"js live_count={js_bad4['live_count']} python={py['live_count']}"
    except RuntimeError as exc:
        rejected4 = True
        msg = " ".join(str(exc).split())
        detail4 = f"JS reader refused the artifact: ...{msg[-110:]}"
    g.known_bad("an extra tombstone (JS side only) breaks the live-row mapping",
                rejected=rejected4, detail=detail4,
                covers=("readers agree on live_count / total_rows / row_bytes (haar/fixture)",))

    # ---- coverage ------------------------------------------------------- #
    g.coverage(
        "Issue #20 has TWO mechanisms and this gate closes exactly one. "
        "(a) SIGN CONVENTION at exactly 0.0 — remax packs on `> 0`, "
        "js/kb-reader.js packed on `>= 0`, so a projection landing on 0.0 "
        "produced OPPOSITE bits by construction. Closed: the reader now packs "
        "on `> 0`, and the exact-zero probe above drives a constructed 0.0 "
        "through both the packer and Node, with a known-bad that restores "
        "`>= 0` and goes red. (b) FLOAT SUMMATION ORDER — numpy's float32 BLAS "
        "matmul and the JS float64 accumulation loop can still land on "
        "opposite sides of zero for a NEAR-zero projection. NOT closed, and "
        "not closable without one of the two readers changing arithmetic. This "
        "fixture is measured to sit far from that regime (see the sign-margin "
        "bracket); a query that does not, still can. The xfail in "
        "tests/test_js_reader_compat.py stays, for mechanism (b) only."
    )
    g.coverage(
        "Two projections are covered across readers: haar with a float32 "
        "sidecar (committed fixture, which pins projection='haar' explicitly "
        "so the sidecar path survives the 2026-08 default flip) and srht with "
        "NO sidecar (the writer's default, built at runtime). The "
        "int8-quantized sidecar and rademacher are still "
        "unexercised end-to-end here — rademacher's plane generator is pinned "
        "bit-identically by tests/test_js_reader_compat.py, but no .kbi in "
        "either projection is read by both readers. remex is out of scope by "
        "construction: js/kb-reader.js refuses it (see gate_open_validation)."
    )
    g.coverage(
        "The default/srht fixture is REBUILT by the writer on every run rather "
        "than "
        "committed. That is what makes it cheap, and it means a writer "
        "regression moves both readers' input together and stays invisible "
        "here: this arm asserts the two READERS agree about an artifact, never "
        "that the artifact is what the writer used to produce. The committed "
        "haar fixture is the arm that would notice a writer change."
    )
    g.coverage(
        "Default-argument parity IS now covered, but only for the knobs that "
        "exist: over_fetch, rrf_c and min_sim. `alpha` (weighted fusion) is "
        "exercised on neither reader here — every run above uses RRF — so a "
        "divergence in the min-max normalization would pass this gate."
    )
    g.coverage(
        "Corpus-side encoding is not cross-checked: the JS reader never packs "
        "vectors, so only the query encoder and the decode/scan path are "
        "compared against Python."
    )
    g.coverage(
        "Four checks are deliberately unreached by any known-bad because they "
        "exonerate the instrument rather than assert parity: embedding "
        "reproducibility, plane regeneration under the installed remax, the "
        "sign-margin bracket, and the exact-zero probe's own non-vacuity "
        "bracket. All four fail loudly if the fixture and the environment have "
        "drifted apart; none claims anything about the JS reader."
    )
    return g.report()


if __name__ == "__main__":
    raise SystemExit(main())
