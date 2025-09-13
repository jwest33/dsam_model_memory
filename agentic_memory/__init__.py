"""
JAM (Journalistic Agent Memory) - Local-first memory system for LLM agents

A sophisticated memory system that provides persistent, searchable memory using
journalistic 5W1H (Who, What, When, Where, Why, How) semantic extraction.
"""

__version__ = "0.0.1"

# Core components
from .router import MemoryRouter
from .config_manager import ConfigManager
from .config import Config
from .retrieval import HybridRetriever
from .block_builder import BlockBuilder
from .tokenization import TokenizerAdapter
from .types import (
    RawEvent,
    MemoryRecord,
    MemoryBlock,
    RetrievalQuery,
    Candidate,
    Who,
    Where,
    MemoryPointer,
    EventType,
)

# Storage components
from .storage.sql_store import MemoryStore
from .storage.faiss_index import FaissIndex

# Extraction components
from .extraction.llm_extractor import extract_5w1h
from .extraction.multi_part_extractor import extract_multi_part_5w1h, extract_batch_5w1h

# Advanced features
from .attention import AdaptiveEmbeddingSpace
from .cluster.concept_cluster import LiquidMemoryClusters

# Server components (optional imports)
try:
    from .server.flask_app import create_app
except ImportError:
    create_app = None

# Tools (optional imports)
try:
    from .tools.tool_handler import ToolHandler
    from .tools.memory_evaluator import MemoryEvaluator
except ImportError:
    ToolHandler = None
    MemoryEvaluator = None

__all__ = [
    # Version
    "__version__",

    # Core
    "MemoryRouter",
    "ConfigManager",
    "Config",
    "HybridRetriever",
    "BlockBuilder",
    "TokenizerAdapter",

    # Types
    "RawEvent",
    "MemoryRecord",
    "MemoryBlock",
    "RetrievalQuery",
    "Candidate",
    "Who",
    "Where",
    "MemoryPointer",
    "EventType",

    # Storage
    "MemoryStore",
    "FaissIndex",

    # Extraction
    "extract_5w1h",
    "extract_multi_part_5w1h",
    "extract_batch_5w1h",

    # Advanced
    "AdaptiveEmbeddingSpace",
    "LiquidMemoryClusters",

    # Optional
    "create_app",
    "ToolHandler",
    "MemoryEvaluator",
]