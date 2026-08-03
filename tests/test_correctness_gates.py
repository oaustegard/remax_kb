"""Run the non-parity correctness gates from pytest.

The gates live in ``tests/gates/`` and are runnable standalone (they print a
full report with their known-bads and coverage limits); these wrappers just make
``pytest -q`` — and therefore CI — fail when one goes red.

``tests/test_reader_parity.py`` covers the two cross-reader gates. The gates
here are single-implementation correctness claims:

* ``gate_topk_stability.py`` — top-k selection order at the k-th boundary,
  anchored on ``np.argsort(kind="stable")``.
* ``gate_open_validation.py`` — the SPEC_v2 open-time validation order refuses
  corrupted artifacts instead of silently mis-serving them, anchored on the
  spec's own numbered list and on ``js/kb-reader.js`` run in Node.

``PYTHONDONTWRITEBYTECODE=1`` is set for the subprocess so a stale ``.pyc``
cannot let a mutation survive a mutation run.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

GATES = Path(__file__).resolve().parent / "gates"

GATE_SCRIPTS = ["gate_topk_stability.py", "gate_open_validation.py"]


def _run(script: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(GATES / script)],
                          capture_output=True, text=True, timeout=600, env=env)


@pytest.mark.parametrize("script", GATE_SCRIPTS)
def test_gate_is_green(script: str) -> None:
    r = _run(script)
    assert r.returncode == 0, r.stdout + r.stderr
