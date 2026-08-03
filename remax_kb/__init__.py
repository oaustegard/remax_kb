"""remax_kb — portable 1-bit binary embedding knowledgebase format."""

from .manifest import Manifest, Embedder, Binarizer, CorpusInfo, Prompts
from .read import KB
from .read_v2 import KB as KBv2, Hit
from .pack import pack, pack_directory, Chunk
from .pack_v2 import KBWriter, SyncStats
from .formats import detect_format
from .migrate import migrate_v1_to_v2

# Single source of truth: pyproject.toml's [project].version, read back through
# the installed distribution metadata. The hardcoded literal that used to live
# here said 0.1.0 while pyproject said 0.4.0 — two numbers, both wrong to
# somebody, and nothing that could ever notice the drift.
#
# The fallback matters for a source tree that was never installed (a bare
# `sys.path` checkout, e.g. the .kbi packing scripts): PackageNotFoundError is
# not a reason to fail an import, but "0.0.0+unknown" is honest about not
# knowing rather than asserting a stale number.
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("remax_kb")
except PackageNotFoundError:  # not installed — running from a source checkout
    __version__ = "0.0.0+unknown"
del _pkg_version, PackageNotFoundError

__all__ = [
    "__version__",
    "KB",
    "KBv2",
    "Hit",
    "KBWriter",
    "SyncStats",
    "Manifest",
    "Embedder",
    "Binarizer",
    "CorpusInfo",
    "Prompts",
    "Chunk",
    "pack",
    "pack_directory",
    "detect_format",
    "migrate_v1_to_v2",
]
