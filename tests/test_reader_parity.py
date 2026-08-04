"""Run the reader-parity gates from pytest.

The gates live in ``tests/gates/`` and are runnable standalone (they print a
full report); these wrappers just make ``pytest -q`` — and therefore CI — fail
when either goes red.

Unlike ``tests/test_js_reader_compat.py``, which re-implements the JS reader in
Python, ``gate_cross_reader.py`` executes ``js/kb-reader.js`` itself in Node.
It self-skips where Node is unavailable; the pure-Python tokenizer arm of
``gate_tokenizer_parity.py`` still runs.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GATES = Path(__file__).resolve().parent / "gates"


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(GATES / script)],
                          capture_output=True, text=True, timeout=600)


def test_bm25_tokenizer_parity_gate():
    """Both readers must tokenize a query the way bm25s tokenizes the corpus."""
    r = _run("gate_tokenizer_parity.py")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not on PATH; cannot execute js/kb-reader.js")
def test_cross_reader_parity_gate():
    """js/kb-reader.js and remax_kb.read_v2 must decode the fixture alike."""
    r = _run("gate_cross_reader.py")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not on PATH; cannot execute js/kb-reader.js")
def test_open_validation_gate():
    """Both readers must refuse the same corrupted artifacts.

    Runs from pytest as well as CI because the gate now *asserts* JS/Python
    refusal parity (SPEC_v2 validation steps 1-7) instead of recording the gap;
    a regression in either reader should fail the ordinary test run, not only
    the gates step.
    """
    r = _run("gate_open_validation.py")
    assert r.returncode == 0, r.stdout + r.stderr
