"""Centralized Enterprise Settings & Constants."""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# File & Question Limits
MAX_FILES = 250               # Supports bulk 200+ Excel workbooks
MAX_FILE_MB = 50              # 50 MB per file limit
MAX_QUESTION_CHARS = 2000

# Chunking & Phase 3 Micro-Batching
CHUNK_SIZE = 1000             # Used for PDF, TXT, MD
CHUNK_OVERLAP = 200
MICRO_BATCH_SIZE = 250        # Phase 3: Stream 100-250 rows per batch into DB

# Retrieval & Storage Paths
DEFAULT_TOP_K = 5
HISTORY_TURNS = 4
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASE_CHROMA_DIR = "./chroma_db"
METADATA_FILE = os.path.join(BASE_CHROMA_DIR, "workspace_state.json")
HYBRID_SQLITE_DB = os.path.join(BASE_CHROMA_DIR, "hybrid_fts_index.db")

# Priority order of verified Groq production chat models
FALLBACK_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "llama3-70b-8192",
    "llama3-8b-8192",
]
