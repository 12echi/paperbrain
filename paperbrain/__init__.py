"""PaperBrain M4 (Recursive Semantic Chunking & Adaptive Multi-Level Outline)."""

from .sections import (
    chunk_sections,
    split_sections,
    recursive_semantic_chunk,
    RecursiveSemanticChunker,
    ChunkingEngine,
)
from .outline import build_outline, draft_from_outline, OutlineEngine
from .consistency import polish, unify_terms, inject_transitions, check_contradicts

__all__ = [
    "chunk_sections",
    "split_sections",
    "recursive_semantic_chunk",
    "RecursiveSemanticChunker",
    "ChunkingEngine",
    "build_outline",
    "draft_from_outline",
    "OutlineEngine",
    "polish",
    "unify_terms",
    "inject_transitions",
    "check_contradicts",
]
