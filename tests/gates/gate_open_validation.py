#!/usr/bin/env python3
"""GATE — SPEC_v2 open-time validation refuses corrupted artifacts.

A `.kbi` is designed to be fetched over HTTP from a third party. Every field
this gate attacks is therefore attacker-influenceable input, and the wrong
conclusion it blocks is: *this reader validates the artifacts it opens*, when in
fact it was accepting three classes of corruption —

  * **step 2 / binarizer.kind** — ``read_v2`` chose its codec with
    ``"remex" if kind == REMEX_KIND else "remax"``, so ANY unknown kind was
    silently decoded as remax centered-simhash. v1 refuses (manifest.py:165);
    v2 regressed. SPEC_v2 §Field semantics: "A reader MUST refuse unknown kinds."
  * **step 5 / chunk_id_offset** — never scanned at open, only dereferenced
    lazily in ``_chunk_id_at``, where an out-of-range offset surfaced as a bare
    ``ValueError`` from ``bytes.index`` mid-query.
  * **step 7 / bm25 row count** — never checked in Python. ``_bm25_search``
    indexes ``scores[live_idx]`` against a ``_row_of_live`` built from a
    different row count, so a mismatched index returns WRONG DOCUMENTS with no
    error at all. ``js/kb-reader.js:550`` has thrown on this all along.

ANCHORS — two, both outside the Python reader:

  1. **SPEC_v2.md's own numbered "Validation order" list.** The gate parses the
     spec text and requires every numbered step to have a corruption case
     registered here. Add a step to the spec with no case and this goes red;
     the spec is not something the reader's author can quietly satisfy.
  2. **js/kb-reader.js**, executed in Node on the identical corrupted bytes.
     It is an independent implementation, and for step 7 it is the one that was
     already right — so agreement is evidence, not transcription.

PARITY IS ASSERTED, NOT RECORDED. This gate used to *note* that the JS reader
accepted four artifacts the Python reader refused (an unknown ``binarizer.kind``
and all three malformed-``chunk_id`` cases) and call that a coverage limit. A
recorded gap is not a gate: it reads the same whether the gap is four cases or
eleven, and it never goes red. Both readers now implement SPEC_v2 steps 1-7 and
the agreement is a check.

KNOWN-BADS, two kinds:

  * corrupted copies of the committed fixture ``jsparity.kbi``, one per
    validation step, each built by mutating the real writer's output rather than
    by hand-rolling a fake artifact;
  * **mutated copies of js/kb-reader.js itself** — the validation deleted, one
    step at a time — driven through the same corruptions to prove the parity
    assertion can fail. A gate that only ever saw two agreeing readers has not
    been shown to notice a disagreeing one.

    PYTHONDONTWRITEBYTECODE=1 python3 tests/gates/gate_open_validation.py
"""
from __future__ import annotations

import io
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from gate import Gate  # noqa: E402

from remax_kb.read_v2 import KB, ROW_BYTES_CHUNK_MAP  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "jsparity"
KBI = FIXTURE_DIR / "jsparity.kbi"
KBC = FIXTURE_DIR / "jsparity.kbc"
SPEC = ROOT / "SPEC_v2.md"


# --------------------------------------------------------------------- #
# Anchor 1: the spec's own numbered validation list
# --------------------------------------------------------------------- #

def spec_validation_steps() -> dict[int, str]:
    """Parse the numbered MUST list under '## Validation order' in SPEC_v2.md.

    Read from the document, never hardcoded here: a hardcoded copy would be a
    golden, and a golden written by whoever edited the reader is not an anchor.
    """
    text = SPEC.read_text(encoding="utf-8")
    m = re.search(r"^## Validation order\s*$(.*?)^## ", text,
                  re.MULTILINE | re.DOTALL)
    if not m:
        raise RuntimeError("SPEC_v2.md has no '## Validation order' section")
    body = m.group(1)
    steps: dict[int, str] = {}
    for num, first, cont in re.findall(
        r"^(\d+)\.\s+(.*?)$((?:\n(?!\s*\d+\.)(?!\s*$).*)*)", body, re.MULTILINE
    ):
        steps[int(num)] = " ".join((first + " " + cont).split())
    return steps


# --------------------------------------------------------------------- #
# Corruption helpers — mutate the REAL writer's output
# --------------------------------------------------------------------- #

def read_entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        return {n: zf.read(n) for n in zf.namelist()}


def write_kbi(entries: dict[str, bytes], dest: Path) -> Path:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return dest


def set_chunk_id_offset(chunk_map: bytes, row: int, offset: int) -> bytes:
    buf = bytearray(chunk_map)
    struct.pack_into("<Q", buf, row * ROW_BYTES_CHUNK_MAP + 16, offset)
    return bytes(buf)


def set_tombstone(chunk_map: bytes, row: int) -> bytes:
    buf = bytearray(chunk_map)
    buf[row * ROW_BYTES_CHUNK_MAP + 2] |= 0x01
    return bytes(buf)


# Each corruption returns a mutated entry dict. Keyed by the SPEC_v2 validation
# step it is meant to trip.
Corruption = tuple[str, int, Callable[[dict[str, bytes]], dict[str, bytes]]]


def _drop_vectors(e: dict[str, bytes]) -> dict[str, bytes]:
    e = dict(e)
    del e["vectors.bin"]
    return e


def _bad_spec_version(e: dict[str, bytes]) -> dict[str, bytes]:
    e = dict(e)
    m = json.loads(e["manifest.json"])
    m["spec_version"] = "3"
    e["manifest.json"] = json.dumps(m).encode()
    return e


def _bad_kind(e: dict[str, bytes]) -> dict[str, bytes]:
    e = dict(e)
    m = json.loads(e["manifest.json"])
    m["kind"] = "single-file"
    e["manifest.json"] = json.dumps(m).encode()
    return e


def _bad_binarizer_kind(e: dict[str, bytes]) -> dict[str, bytes]:
    """The v1->v2 regression: a plausible future codec name, not gibberish."""
    e = dict(e)
    m = json.loads(e["manifest.json"])
    m["binarizer"]["kind"] = "remax-centered-simhash-v2"
    e["manifest.json"] = json.dumps(m).encode()
    return e


def _truncate_vectors(e: dict[str, bytes]) -> dict[str, bytes]:
    e = dict(e)
    m = json.loads(e["manifest.json"])
    b = m["binarizer"]
    row_bytes = b["dim"] * b["k"] // 8
    e["vectors.bin"] = e["vectors.bin"][:-row_bytes]
    return e


def _truncate_chunk_map(e: dict[str, bytes]) -> dict[str, bytes]:
    e = dict(e)
    e["chunk_map.bin"] = e["chunk_map.bin"][:-ROW_BYTES_CHUNK_MAP]
    return e


def _offset_out_of_range(e: dict[str, bytes]) -> dict[str, bytes]:
    e = dict(e)
    e["chunk_map.bin"] = set_chunk_id_offset(
        e["chunk_map.bin"], 3, len(e["chunk_ids.bin"]) + 4096
    )
    return e


def _offset_unterminated(e: dict[str, bytes]) -> dict[str, bytes]:
    """Truncated chunk_ids.bin: the last id loses its NUL terminator."""
    e = dict(e)
    ids = e["chunk_ids.bin"]
    assert ids.endswith(b"\x00")
    e["chunk_ids.bin"] = ids[:-1]
    return e


def _offset_bad_utf8(e: dict[str, bytes]) -> dict[str, bytes]:
    """A lone 0xFF inside an id — a continuation byte with no lead byte."""
    e = dict(e)
    ids = bytearray(e["chunk_ids.bin"])
    ids[1] = 0xFF
    e["chunk_ids.bin"] = bytes(ids)
    return e


def _live_count_wrong(e: dict[str, bytes]) -> dict[str, bytes]:
    """A row tombstoned without the manifest being updated."""
    e = dict(e)
    e["chunk_map.bin"] = set_tombstone(e["chunk_map.bin"], 0)
    return e


def _bm25_rows_mismatch(e: dict[str, bytes]) -> dict[str, bytes]:
    """delete_chunks() that tombstoned a row and updated live_count but did
    NOT rebuild the bm25 index — the plausible writer bug, not a random edit.
    Step 6 passes (manifest agrees with the map); step 7 is the only thing
    standing between this artifact and silently wrong search results."""
    e = dict(e)
    e["chunk_map.bin"] = set_tombstone(e["chunk_map.bin"], 0)
    m = json.loads(e["manifest.json"])
    m["chunks"]["live_count"] -= 1
    e["manifest.json"] = json.dumps(m).encode()
    return e


def _as_remex(e: dict[str, bytes], bits: int) -> dict[str, bytes]:
    """Re-badge the fixture as a well-formed ``remex-lloyd-max`` artifact.

    NOT a corruption — SPEC_v2 §remex codec describes exactly this shape: a
    ``bits`` field with ``k: 1``, no ``projection`` / ``rotations_quant``, no
    rotation sidecar, an all-zero ``mean_vector_b64``. ``vectors.bin`` is
    resized to the remex row width ``dim * bits // 8`` so the artifact is
    internally consistent; its *contents* are meaningless, which does not
    matter because the question is what the JS reader does at OPEN.

    Built this way rather than with the real writer because ``remex`` is an
    optional dependency (absent here, present in CI) — and the reader's
    behaviour at open is decided by the manifest and the row widths alone, both
    of which this reproduces faithfully.
    """
    import base64

    e = dict(e)
    e.pop("binarizer/rotations.f32", None)
    m = json.loads(e["manifest.json"])
    b = dict(m["binarizer"])
    dim = b["dim"]
    b.update(kind="remex-lloyd-max", k=1, bits=bits)
    b.pop("projection", None)
    b.pop("rotations_quant", None)
    b["mean_vector_b64"] = base64.b64encode(
        np.zeros(m["embedder"]["full_dim"], dtype="<f4").tobytes()
    ).decode("ascii")
    m["binarizer"] = b
    e["manifest.json"] = json.dumps(m).encode()
    need = m["chunks"]["total_rows"] * (dim * bits // 8)
    v = e["vectors.bin"]
    e["vectors.bin"] = (v * (need // len(v) + 1))[:need]
    return e


CORRUPTIONS: list[Corruption] = [
    ("required entry vectors.bin removed", 1, _drop_vectors),
    ("unknown spec_version", 2, _bad_spec_version),
    ("unknown manifest kind", 2, _bad_kind),
    ("unknown binarizer.kind", 2, _bad_binarizer_kind),
    ("vectors.bin short by one row", 3, _truncate_vectors),
    ("chunk_map.bin short by one row", 4, _truncate_chunk_map),
    ("chunk_id_offset past end of chunk_ids.bin", 5, _offset_out_of_range),
    ("chunk_id not NUL-terminated", 5, _offset_unterminated),
    ("chunk_id is not valid UTF-8", 5, _offset_bad_utf8),
    ("tombstone set without updating live_count", 6, _live_count_wrong),
    ("bm25 postings rows != live_count", 7, _bm25_rows_mismatch),
]

# Step 8 (embedder fingerprint) is deliberately absent from CORRUPTIONS: the
# reader validates it in search(), not open(). See the check below, which
# asserts the spec text and the code agree about WHERE that happens.
STEP_8_AT_SEARCH = 8


def open_refuses(path: Path) -> tuple[bool, str]:
    try:
        KB.open(str(path))
    except ValueError as exc:
        return True, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 — anything else is the wrong shape
        return False, f"raised {type(exc).__name__} (not ValueError): {exc}"
    return False, "opened without complaint"


def js_open(path: Path, reader: Path | None = None) -> dict | None:
    if shutil.which("node") is None:
        return None
    cmd = ["node", str(HERE / "js_open_validation.mjs"), "--kbi", str(path)]
    if reader is not None:
        cmd += ["--reader", str(reader)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"node harness failed: {r.stderr.strip()[:800]}")
    return json.loads(r.stdout)


def mutated_reader(old: str, new: str) -> Path:
    """A copy of js/kb-reader.js with one substring replaced.

    The parity assertion below claims the JS reader refuses everything the
    Python reader refuses. A known-bad for that claim has to be a JS reader
    that DOESN'T — so it is built by deleting the validation from the real
    source, not by hand-writing a stub. The count assert means a rename turns
    the known-bad into a hard error instead of a silent no-op that "passes".
    """
    src = (ROOT / "js" / "kb-reader.js").read_text(encoding="utf-8")
    if src.count(old) != 1:
        raise RuntimeError(
            f"known-bad mutation target {old!r} appears {src.count(old)} times "
            f"in js/kb-reader.js; expected exactly 1")
    d = Path(tempfile.mkdtemp(prefix="gate-openval-mutant-"))
    dst = d / "kb-reader.js"
    dst.write_text(src.replace(old, new), encoding="utf-8")
    return dst


def main() -> int:
    g = Gate("SPEC_v2 open-time validation refuses corrupted .kbi artifacts")

    if not KBI.is_file():
        g.check(False, "fixture present", f"{KBI} missing")
        g.coverage("fixture missing; nothing ran")
        return g.report()

    pristine = read_entries(KBI)
    steps = spec_validation_steps()
    g.note(f"SPEC_v2 '## Validation order' declares {len(steps)} numbered "
           f"steps: {sorted(steps)}")

    tmp = Path(tempfile.mkdtemp(prefix="gate-openval-"))
    try:
        # ---- positive control ------------------------------------------- #
        # Validation that refuses everything is not validation. The pristine
        # artifact must still open, and still search.
        good = write_kbi(pristine, tmp / "good.kbi")
        try:
            kb = KB.open(str(good))
            opened = True
            live = kb.live_count
        except Exception as exc:  # noqa: BLE001
            opened, live = False, f"{type(exc).__name__}: {exc}"
        g.check(opened and live == json.loads(pristine["manifest.json"])
                ["chunks"]["live_count"],
                "positive control: the pristine fixture still opens",
                f"live_count={live}")

        if opened:
            sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
            from build_jsparity_fixture import DeterministicEmbedder
            qspec = json.loads((FIXTURE_DIR / "queries.json").read_text())
            q = qspec["queries"][0]
            hits = kb.search(q["text"], embedder=DeterministicEmbedder(),
                             k=3, min_sim="off")
            g.check(len(hits) > 0 and all(h.chunk_id for h in hits),
                    "positive control: search still returns resolvable chunk_ids",
                    f"{len(hits)} hits, ids={[h.chunk_id for h in hits]}")

        # ---- anchor 1: every spec step has a corruption case -------------- #
        covered_steps = {step for _, step, _ in CORRUPTIONS} | {STEP_8_AT_SEARCH}
        missing = sorted(set(steps) - covered_steps)
        g.check(not missing,
                "anchor: every numbered step in SPEC_v2 '## Validation order' "
                "has a corruption case in this gate",
                f"uncovered spec steps: {missing}" if missing
                else f"steps {sorted(covered_steps & set(steps))} all covered")
        extra = sorted(covered_steps - set(steps))
        g.check(not extra,
                "anchor: this gate invents no validation step the spec does not "
                "state",
                f"steps here but not in the spec: {extra}" if extra else "none")

        # ---- step 8: spec says 'at open', code does it at search ---------- #
        # This is a real disagreement between the spec and the reader, not a
        # corruption. It is asserted so the gate goes red if someone "fixes"
        # one side without the other.
        import inspect

        from remax_kb import read_v2 as _rv2

        open_src = inspect.getsource(_rv2.KB._from_bytes)
        search_src = inspect.getsource(_rv2.KB.search)
        code_at_search = ("_validate_embedder" in search_src
                          and "_validate_embedder" not in open_src)
        spec_8 = steps.get(8, "")
        spec_says_search = "search()" in spec_8
        g.check(code_at_search and spec_says_search,
                "step 8 (embedder fingerprint): SPEC_v2 and the code agree it "
                "happens at search(), not open()",
                f"code_at_search={code_at_search} spec_says_search="
                f"{spec_says_search}; spec text: {spec_8!r}")

        # ---- the corruptions --------------------------------------------- #
        js_results: dict[str, dict | None] = {}
        bad_paths: dict[str, Path] = {}
        for label, step, mutate in CORRUPTIONS:
            path = write_kbi(mutate(pristine),
                             tmp / f"bad-{step}-{abs(hash(label))}.kbi")
            bad_paths[label] = path
            refused, detail = open_refuses(path)
            g.known_bad(
                f"step {step}: {label} is REFUSED at open",
                rejected=refused,
                detail=detail,
                covers=(f"refusal is enforced for step {step}",),
            )
            # A companion substantive check, so the known-bad has something to
            # cover and the report shows reach rather than a bare list.
            g.check(refused, f"refusal is enforced for step {step}: {label}",
                    detail)
            try:
                js_results[label] = js_open(path)
            except RuntimeError as exc:
                js_results[label] = {"opened": None, "error": str(exc)}

        # ---- anchor 2: the JS reader, on the same bytes -------------------- #
        if shutil.which("node") is None:
            g.coverage("node is not on PATH — the js/kb-reader.js arm did not "
                       "run. A Python-only run of this gate certifies nothing "
                       "about the shipped JS reader's validation.")
        else:
            js_good = js_open(good)
            g.check(bool(js_good and js_good["opened"]),
                    "positive control (JS): kb-reader.js opens the pristine "
                    "fixture", json.dumps(js_good))
            # The step this gate exists for: the JS reader has ALWAYS thrown on
            # a bm25 row-count mismatch. If it stops, the anchor is gone.
            r = js_results.get("bm25 postings rows != live_count")
            g.check(bool(r and r["opened"] is False),
                    "anchor: js/kb-reader.js independently refuses the bm25 "
                    "row-count mismatch (SPEC_v2 step 7)",
                    json.dumps(r))
            both = [lab for lab, _, _ in CORRUPTIONS
                    if (js_results.get(lab) or {}).get("opened") is False]
            js_accepts = [lab for lab, _, _ in CORRUPTIONS
                          if (js_results.get(lab) or {}).get("opened") is True]
            g.note(f"JS reader refuses {len(both)}/{len(CORRUPTIONS)} of the "
                   f"same corruptions")

            # ---- the parity ASSERTION -------------------------------------- #
            # This used to be a g.coverage() note listing four corruptions the
            # JS reader accepted (an unknown binarizer.kind and all three
            # malformed-chunk-id cases). A recorded gap is not a gate: it reads
            # identically whether the gap is four cases or eleven, and it does
            # not go red when a fifth appears. It is now a check, so the two
            # readers are pinned to one definition of "valid artifact".
            g.check(
                not js_accepts and len(both) == len(CORRUPTIONS),
                "PARITY: js/kb-reader.js refuses every corruption the Python "
                "reader refuses (SPEC_v2 validation steps 1-7)",
                f"{len(both)}/{len(CORRUPTIONS)} refused by both"
                + (f"; JS still ACCEPTS: {'; '.join(js_accepts)}"
                   if js_accepts else ""))

            # ---- known-bads for the parity assertion ----------------------- #
            # Each deletes one of the validations added to js/kb-reader.js and
            # confirms the assertion above notices. Without these, "the readers
            # agree" is a claim that has never been shown capable of failing.
            #
            # Step 6 needs its own artifact rather than the CORRUPTIONS entry:
            # on the committed fixture, bm25/ is present, so step 7 refuses the
            # same bytes and MASKS step 6 entirely. That is worth knowing on its
            # own — the JS step-6 check only does any work on a dense-only .kbi,
            # which is exactly the artifact used here.
            dense_only_live = write_kbi(
                {k: v for k, v in _live_count_wrong(pristine).items()
                 if not k.startswith("bm25/")},
                tmp / "bad-6-dense-only.kbi")
            py_refuses_dense, py_detail = open_refuses(dense_only_live)
            g.check(py_refuses_dense,
                    "step 6 in isolation: a dense-only .kbi whose live_count "
                    "disagrees with its tombstones is refused (Python)",
                    py_detail)
            js_dense = js_open(dense_only_live) or {}
            g.check(js_dense.get("opened") is False,
                    "step 6 in isolation: js/kb-reader.js refuses it too "
                    "(nothing else can catch it without bm25/)",
                    json.dumps(js_dense))

            for kb_label, old_src, new_src, cases in (
                ("step 5 (chunk_id offsets)",
                 "    validateChunkIdOffsets(this._chunkMapView, this._chunkIds, total);",
                 "    void 0;",
                 [bad_paths["chunk_id_offset past end of chunk_ids.bin"],
                  bad_paths["chunk_id not NUL-terminated"],
                  bad_paths["chunk_id is not valid UTF-8"]]),
                ("step 2 (binarizer.kind)",
                 "    if (!SUPPORTED_BINARIZER_KINDS.includes(bin.kind)) {",
                 "    if (false) {",
                 [bad_paths["unknown binarizer.kind"]]),
                ("step 6 (live_count)",
                 "    if (declaredLive != null && this._rowOfLive.length !== declaredLive) {",
                 "    if (false) {",
                 [dense_only_live]),
            ):
                mutant = mutated_reader(old_src, new_src)
                leaked = [p.name for p in cases
                          if (js_open(p, reader=mutant) or {})
                          .get("opened") is True]
                g.known_bad(
                    f"a js/kb-reader.js with {kb_label} validation removed "
                    f"breaks reader parity",
                    rejected=len(leaked) == len(cases),
                    detail=f"{len(leaked)}/{len(cases)} corruptions leak "
                           f"through the mutant: {leaked}",
                    covers=("PARITY: js/kb-reader.js refuses every corruption",
                            "step 6 in isolation: js/kb-reader.js refuses it"),
                )

            # ---- SPEC_v2 §remex codec: JS refuses, and says why ------------ #
            # Not a corruption — a well-formed artifact this reader genuinely
            # cannot decode (see REMEX_REFUSAL in js/kb-reader.js for why it is
            # impossible rather than unimplemented).
            #
            # What the reader did BEFORE is the point. `_rowBytes` was
            # `dim*k/8` unconditionally, which is the wrong width for remex's
            # `dim*bits/8` — but the reader never got that far, because a remex
            # .kbi ships no rotation sidecar and `bin.projection` is absent, so
            # `projection || "haar"` sent it down the haar branch and it threw
            # about a missing `binarizer/rotations.f32`. Two bit widths are
            # exercised because the width arithmetic differs (bits=1 coincides
            # at 4 bytes/row, bits=4 does not), and NEITHER changes the message:
            # the caller was told to go find a rotations sidecar for an artifact
            # that is not supposed to have one, and the implied remedy
            # (repack with a shipped rotation) would not have helped.
            def remex_refusal_ok(res: dict) -> bool:
                err = (res or {}).get("error") or ""
                return ((res or {}).get("opened") is False
                        and "remex-lloyd-max" in err
                        and "--projection srht" in err)

            for bits, why in ((1, "remex row width coincides with dim*k/8"),
                              (4, "remex row width differs from dim*k/8")):
                rx = write_kbi(_as_remex(pristine, bits), tmp / f"remex-{bits}.kbi")
                res = js_open(rx) or {}
                g.check(remex_refusal_ok(res),
                        f"remex-lloyd-max (bits={bits}) is refused by name, "
                        f"with a remedy",
                        f"{why}; opened={res.get('opened')} "
                        f"error={str(res.get('error'))[:200]!r}")
                if bits == 1:
                    mutant = mutated_reader(
                        "    if (bin.kind === REMEX_KIND) throw new Error(REMEX_REFUSAL);",
                        "    void 0;")
                    leak = js_open(rx, reader=mutant) or {}
                    g.known_bad(
                        "without the remex refusal, js/kb-reader.js fails on a "
                        "remex .kbi with a message that never names the codec",
                        rejected=not remex_refusal_ok(leak),
                        detail=f"mutant opened={leak.get('opened')} "
                               f"error={str(leak.get('error'))[:200]!r}",
                        covers=("remex-lloyd-max (bits=1) is refused by name",),
                    )

            # ---- the haar sidecar asymmetry, and its remedy --------------- #
            # A haar .kbi that arrives WITHOUT binarizer/rotations.f32 is
            # readable by Python (it re-derives the planes from (dim, k, seed)
            # and ignores the sidecar entirely) and unreadable by JS (it
            # cannot). That asymmetry is why every haar artifact ships up to
            # 9 MiB of rotations for a consumer that may never load them.
            # The requirement stays — SPEC_v2 §binarizer/rotations.f32 makes it
            # a MUST for readers in this position — but the refusal has to name
            # the way out, which is repacking seed-only, not hunting for a file
            # that was never generated.
            no_sidecar = write_kbi(
                {k: v for k, v in pristine.items()
                 if not k.startswith("binarizer/rotations")},
                tmp / "no-sidecar.kbi")
            try:
                kb_ns = KB.open(str(no_sidecar))
                py_ns = f"opened, live_count={kb_ns.live_count}"
                py_opened = True
            except Exception as exc:  # noqa: BLE001
                py_ns, py_opened = f"{type(exc).__name__}: {exc}", False
            g.check(py_opened,
                    "asymmetry: the Python reader OPENS a haar .kbi with no "
                    "rotation sidecar (it re-derives from the seed)", py_ns)

            def sidecar_msg_ok(res: dict) -> bool:
                err = (res or {}).get("error") or ""
                return ((res or {}).get("opened") is False
                        and "rotations.f32" in err
                        and "--projection srht" in err)

            ns_res = js_open(no_sidecar) or {}
            g.check(sidecar_msg_ok(ns_res),
                    "js/kb-reader.js refuses it AND names the remedy "
                    "(repack --projection srht)",
                    f"opened={ns_res.get('opened')} "
                    f"error={str(ns_res.get('error'))[:220]!r}")
            mutant = mutated_reader(
                '          "but binarizer/rotations.f32 is absent. " + SIDECAR_REMEDY',
                '          "but binarizer/rotations.f32 is absent."')
            leak = js_open(no_sidecar, reader=mutant) or {}
            g.known_bad(
                "a refusal that names the missing file but not the remedy "
                "leaves the caller hunting for an artifact that was never "
                "generated",
                rejected=not sidecar_msg_ok(leak),
                detail=f"mutant error={str(leak.get('error'))[:160]!r}",
                covers=("js/kb-reader.js refuses it AND names the remedy",),
            )

        # ---- coverage ---------------------------------------------------- #
        g.coverage(
            "Validation is checked at OPEN only. Per-chunk sha256 verification "
            "on lazy fetch (SPEC_v2: 'Lazy chunk fetches MUST additionally "
            "verify sha256') is a different claim; a shard whose bytes were "
            "swapped after open is not exercised here."
        )
        g.coverage(
            "Each corruption is a SINGLE mutation of a valid artifact. A "
            "coherently-forged .kbi — one where every internal count agrees "
            "with every other and only the CONTENT is hostile — passes every "
            "check in the validation order by construction, and this gate "
            "would call it fine. The validation order is an integrity check, "
            "not an authenticity check; nothing in the format signs the "
            "manifest."
        )
        g.coverage(
            "Step 5 checks that ids are well-formed, NOT that they are unique "
            "or that distinct rows point at distinct ids. Two rows sharing one "
            "chunk_id_offset is conforming per the spec text and undetected "
            "here."
        )
        g.coverage(
            "The bm25 check compares the postings matrix ROW COUNT to "
            "live_count. It does not check that row i of the postings "
            "corresponds to the i-th live row — a permuted bm25 index has the "
            "right shape and returns wrong documents, and neither reader nor "
            "this gate would notice."
        )
        g.coverage(
            "Only the remax centered-simhash codec path is exercised for "
            "DECODING: the committed fixture is 1-bit haar. (The remex path is "
            "now exercised on the JS side, but only to the point of refusal — "
            "see the separate limit below.) The srht/rademacher manifests take "
            "different branches at open and are unvalidated by this gate. "
            "Measured, not assumed: `mutate.py --target "
            "remax_kb/read_v2.py --max 45 -- <this gate>` kills 23/45, and "
            "every survivor is on a codec/projection branch the fixture cannot "
            "reach (the remex row_bytes arithmetic, the rademacher/srht/int8 "
            "dispatch). Closing this needs fixtures in those codecs, not more "
            "corruptions of this one."
        )
        g.coverage(
            "The sidecar asymmetry is checked at the level of the MESSAGE, not "
            "the outcome. This gate asserts that js/kb-reader.js refuses a "
            "sidecar-free haar .kbi and points at `--projection srht`; that "
            "the suggested repack actually round-trips is a different claim, "
            "made by tests/gates/gate_cross_reader.py, which builds an srht "
            "artifact and reads it in both readers with no rotation entry "
            "shipped."
        )
        g.coverage(
            "Parity is asserted for REFUSALS of CORRUPTED artifacts, not for "
            "acceptances. The two readers deliberately disagree on one class "
            "of well-formed input: a remex-lloyd-max .kbi, which the Python "
            "reader opens and js/kb-reader.js refuses by design. That is not a "
            "bug being papered over — JS cannot re-derive remex's Lloyd-Max "
            "centroids or its numpy-generated Haar rotation, and a remex .kbi "
            "ships neither — but it does mean 'the readers agree' is a claim "
            "about the corruption list, not about the format as a whole."
        )
        g.coverage(
            "The remex artifacts here are SYNTHESIZED by re-badging the haar "
            "fixture's manifest (kind, k=1, bits, zero mean, sidecar removed, "
            "vectors.bin resized), because remex is an optional dependency "
            "that is absent in some environments. That is enough to pin what "
            "the JS reader DOES at open — which is decided by the manifest and "
            "the row widths — and nothing at all about whether the Python "
            "reader decodes a genuine remex corpus correctly."
        )
        g.coverage(
            "The UTF-8 refusal is exercised with ONE malformed sequence (a "
            "lone 0xFF). Python's strict codec and WHATWG TextDecoder are "
            "believed to agree on overlong encodings, lone surrogates and "
            "out-of-range code points, but none of those is tested here, so a "
            "disagreement in that corner would pass this gate."
        )
        g.coverage(
            "Even the codec-specific unit tests do not close that hole: the "
            "same mutation run against `pytest test_pack_v2_remex.py "
            "test_srht.py test_projection.py test_int8_rotations.py "
            "test_pack_v2.py` still leaves 15/45 alive, concentrated in the "
            "remex `dim * bits // 8` row-width arithmetic. Nothing in this "
            "repo currently notices if that expression changes."
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return g.report()


if __name__ == "__main__":
    raise SystemExit(main())
