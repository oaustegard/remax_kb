"""remax_kb — portable 1-bit binary embedding knowledgebase format."""

from .manifest import Manifest, Embedder, Binarizer, CorpusInfo, Prompts
from .read import KB
from .pack import pack, pack_directory

__version__ = "0.1.0"

__all__ = [
    "KB",
    "Manifest",
    "Embedder",
    "Binarizer",
    "CorpusInfo",
    "Prompts",
    "pack",
    "pack_directory",
]
