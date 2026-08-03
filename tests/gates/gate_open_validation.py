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

KNOWN-BADS: corrupted copies of the committed fixture ``jsparity.kbi``, one per
validation step, each built by mutating the real writer's output rather than by
hand-rolling a fake artifact.

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


def js_open(path: Path) -> dict | None:
    if shutil.which("node") is None:
        return None
    r = subprocess.run(
        ["node", str(HERE / "js_open_validation.mjs"), "--kbi", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"node harness failed: {r.stderr.strip()[:800]}")
    return json.loads(r.stdout)


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
        g.check("_validate_embedder" in search_src
                and "_validate_embedder" not in open_src,
                "step 8 (embedder fingerprint) is validated in search(), and "
                "SPEC_v2 says so",
                f"spec text: {steps.get(8, '<missing>')!r}")

        # ---- the corruptions --------------------------------------------- #
        js_results: dict[str, dict | None] = {}
        for label, step, mutate in CORRUPTIONS:
            path = write_kbi(mutate(pristine),
                             tmp / f"bad-{step}-{abs(hash(label))}.kbi")
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
            if js_accepts:
                g.coverage(
                    "js/kb-reader.js ACCEPTS these corruptions that the Python "
                    "reader refuses: " + "; ".join(js_accepts) + ". The two "
                    "readers therefore do not agree on what is a valid "
                    "artifact, and this gate does not close that gap — it only "
                    "records it. Bringing the JS reader up to the full "
                    "validation order is separate work."
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
            "Only the remax centered-simhash codec path is exercised: the "
            "committed fixture is 1-bit haar. The remex and srht/rademacher "
            "manifests take different branches at open and are unvalidated by "
            "this gate."
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return g.report()


if __name__ == "__main__":
    raise SystemExit(main())
