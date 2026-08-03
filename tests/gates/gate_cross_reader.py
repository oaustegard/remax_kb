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
# Pinned on BOTH sides: the readers' *defaults* differ (JS max(4k,20), Python
# max(8k,64)), which this gate deliberately does not exercise — see coverage.
OVER_FETCH = 20


def run_js(kbi: Path, kbc: Path) -> dict:
    r = subprocess.run(
        ["node", str(HERE / "js_cross_reader.mjs"),
         "--kbi", str(kbi), "--kbc", str(kbc), "--queries", str(QUERIES),
         "--k", str(TOP_K), "--over-fetch", str(OVER_FETCH)],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(f"node reader failed: {r.stderr.strip()[:2000]}")
    return json.loads(r.stdout)


def run_python(kbi: Path) -> dict:
    kb = KB.open(str(kbi))
    emb = DeterministicEmbedder()
    spec = json.loads(QUERIES.read_text())
    out = {
        "live_count": kb.live_count,
        "total_rows": kb.manifest["chunks"]["total_rows"],
        "row_bytes": kb._row_bytes,
        "queries": [],
    }
    for q in spec["queries"]:
        vec = np.asarray(q["embedding"], dtype=np.float32)
        hits = kb.search(q["text"], embedder=emb, k=TOP_K, alpha=None,
                         over_fetch=OVER_FETCH, min_sim="off")
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
    return out


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


def compare(g: Gate, py: dict, js: dict) -> None:
    g.check(py["live_count"] == js["live_count"]
            and py["total_rows"] == js["total_rows"]
            and py["row_bytes"] == js["row_bytes"],
            "readers agree on live_count / total_rows / row_bytes",
            f"python={py['live_count']}/{py['total_rows']}/{py['row_bytes']} "
            f"js={js['live_count']}/{js['total_rows']}/{js['row_bytes']}")

    for p, j in zip(py["queries"], js["queries"]):
        q = p["text"]
        g.check(p["query_code_hex"] == j["query_code_hex"],
                f"query code is byte-identical: {q!r}",
                f"python={p['query_code_hex']} js={j['query_code_hex']}")

        p_ids = [h["chunk_id"] for h in p["hits"]]
        j_ids = [h["chunk_id"] for h in j["hits"]]
        g.check(p_ids == j_ids and len(p_ids) > 0,
                f"top-{TOP_K} ordering matches: {q!r}",
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
                    f"per-hit scores match: {q!r}",
                    f"hamming={dist_ok} rrf={fused_ok} bm25={bm_ok}")

            text_ok = all(a["text"] == b["text"] and a["sha256"] == b["sha256"]
                          for a, b in zip(p["hits"], j["hits"]))
            verified = all(a["verified"] and b["verified"]
                           for a, b in zip(p["hits"], j["hits"]))
            g.check(text_ok and verified,
                    f"fetched chunk bytes + sha256 match and verify: {q!r}",
                    f"identical={text_ok} both_verified={verified}")
        else:
            g.check(False, f"per-hit scores match: {q!r}", "ordering differed")
            g.check(False, f"fetched chunk bytes + sha256 match and verify: {q!r}",
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
           f"({KBI.stat().st_size} B), k={TOP_K} over_fetch={OVER_FETCH}")

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
    compare(g, py, js)

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
                covers=("query code is byte-identical",))

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
                covers=("top-5 ordering matches", "per-hit scores match"))

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
                covers=("fetched chunk bytes",))

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
                covers=("readers agree on live_count",))

    # ---- coverage ------------------------------------------------------- #
    g.coverage(
        "Near-zero projection sign parity (issue #20) is NOT closed by this "
        "gate. numpy's float32 BLAS matmul and the JS float64 accumulation "
        "loop can still disagree on the sign of a projected coordinate that "
        "rounds to either side of zero — and remax packs on `> 0` while "
        "js/kb-reader.js packs on `>= 0`, so an exact 0.0 diverges by "
        "construction. This fixture is measured to sit far from that regime "
        "(see the sign-margin bracket); a query that does not, still can. The "
        "xfail in tests/test_js_reader_compat.py stays."
    )
    g.coverage(
        "Only the haar/float32-sidecar configuration is covered. The int8, "
        "rademacher, srht and remex paths ship different (or no) sidecars and "
        "no fixture exercises them across readers."
    )
    g.coverage(
        "Default-argument parity is NOT covered: over_fetch is pinned to "
        f"{OVER_FETCH} on both sides because the readers disagree by default "
        "(JS max(4k,20) vs Python max(8k,64)), and Python's min_sim floor has "
        "no JS counterpart at all (pinned 'off' here). On a fixture of "
        f"{py['live_count']} live rows the pools saturate, so this gate could "
        "not see that difference even unpinned."
    )
    g.coverage(
        "Corpus-side encoding is not cross-checked: the JS reader never packs "
        "vectors, so only the query encoder and the decode/scan path are "
        "compared against Python."
    )
    g.coverage(
        "Three checks are deliberately unreached by any known-bad because they "
        "exonerate the instrument rather than assert parity: embedding "
        "reproducibility, plane regeneration under the installed remax, and "
        "the sign-margin bracket. All three fail loudly if the fixture and the "
        "environment have drifted apart; none claims anything about the JS "
        "reader."
    )
    return g.report()


if __name__ == "__main__":
    raise SystemExit(main())
