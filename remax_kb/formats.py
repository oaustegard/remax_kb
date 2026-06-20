"""Detect whether a ``.kb``/``.kbi`` artifact is spec v1 or v2.

Per SPEC_v2 §Compatibility: inspect the zip contents — files containing
``chunks.jsonl`` are v1, files containing ``chunk_map.bin`` are v2.
"""
from __future__ import annotations

import zipfile
from pathlib import Path


def detect_format(path: str | Path) -> str:
    """Return ``"1"`` or ``"2"`` for a local ``.kb``/``.kbi`` zip.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the file is not a zip or matches neither layout.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if not zipfile.is_zipfile(p):
        raise ValueError(f"{p} is not a zip archive (not a .kb/.kbi)")
    with zipfile.ZipFile(p, "r") as zf:
        names = set(zf.namelist())
    if "chunk_map.bin" in names:
        return "2"
    if "chunks.jsonl" in names:
        return "1"
    raise ValueError(
        f"{p}: neither chunk_map.bin (v2) nor chunks.jsonl (v1) present"
    )
