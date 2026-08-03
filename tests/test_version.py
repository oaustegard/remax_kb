"""``remax_kb.__version__`` must not drift from ``pyproject.toml``.

It did: ``__init__.py`` said ``0.1.0`` while ``[project].version`` said
``0.4.0``. Two independently-maintained numbers with nothing comparing them.
pyproject is now the single source of truth and ``__version__`` is read back
out of the installed distribution metadata; this test is what keeps that true.

The comparison only means something when the distribution is actually
installed, so it skips otherwise (and says so) rather than passing vacuously.
"""
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

import pytest

import remax_kb

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject_version() -> str:
    if sys.version_info < (3, 11):
        pytest.skip("tomllib requires Python 3.11+")
    import tomllib

    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]


def test_version_matches_pyproject() -> None:
    try:
        installed = pkg_version("remax_kb")
    except PackageNotFoundError:
        pytest.skip(
            "remax_kb is not installed in this interpreter, so distribution "
            "metadata does not exist and there is nothing to compare against. "
            "CI installs with `pip install -e .`, where this test does run."
        )
    assert remax_kb.__version__ == installed
    assert installed == _pyproject_version(), (
        "installed distribution metadata disagrees with pyproject.toml — "
        "reinstall, or the editable install is stale"
    )


def test_version_is_not_a_hardcoded_literal() -> None:
    """The specific regression: a second version literal in __init__.py.

    Runs whether or not the package is installed, so the guard is not silently
    inert in an uninstalled checkout — which is exactly where a well-meaning
    "just hardcode it, importlib is slow" edit would land.
    """
    src = (Path(remax_kb.__file__)).read_text(encoding="utf-8")
    import re

    literals = re.findall(r'^__version__\s*=\s*["\']', src, re.MULTILINE)
    assert not literals, (
        "__init__.py assigns __version__ from a string literal; it must come "
        "from importlib.metadata so pyproject.toml stays the only place a "
        "version number is written"
    )
