"""remax_kb — portable 1-bit binary embedding knowledgebase format."""

from .manifest import Manifest, Embedder, Binarizer, CorpusInfo, Prompts
from .read import KB
from .read_v2 import KB as KBv2, Hit
from .pack import pack, pack_directory, Chunk
from .pack_v2 import KBWriter, SyncStats
from .formats import detect_format
from .migrate import migrate_v1_to_v2

__version__ = "0.1.0"

__all__ = [
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
