"""Centralized Enterprise Settings & Constants (1,500+ Workbook Scale)."""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# File & Question Limits
MAX_FILES = 2000              # Supports bulk 1,500+ Excel workbooks
MAX_FILE_MB = 200             # 200 MB per file limit (matches Streamlit server limit)
MAX_QUESTION_CHARS = 2000

# Chunking & Phase 3 Micro-Batching
CHUNK_SIZE = 1000             # Used for PDF, DOCX, TXT, MD, and Web URLs
CHUNK_OVERLAP = 200
MICRO_BATCH_SIZE = 500        # Phase 3: Stream up to 500 chunks per WAL/Chroma transaction

# Retrieval & Storage Paths
DEFAULT_TOP_K = 6
HISTORY_TURNS = 4
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASE_CHROMA_DIR = "./chroma_db"
METADATA_FILE = os.path.join(BASE_CHROMA_DIR, "workspace_state.json")
HYBRID_SQLITE_DB = os.path.join(BASE_CHROMA_DIR, "hybrid_fts_index.db")

# Priority order of verified active Groq production chat models
FALLBACK_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20ab",
]
