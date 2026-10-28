"""
Phase 4 & Phase 5 (Step 1): Dual-Layer Database Index.
- Index 1: Categorical Payload B-Tree Indexes on `medicine_name` and `sheet_name`.
- Index 2: Full-Text Keyword Index (SQLite FTS5) for exact medical terms,
           molecular variants (e.g., del(17p)), and clinical trial IDs (e.g., NCT02475681).
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any

from langchain_core.documents import Document

from engine.config import BASE_CHROMA_DIR, HYBRID_SQLITE_DB


def get_hybrid_conn() -> sqlite3.Connection:
    """Return a thread-safe SQLite connection to the Dual-Layer Hybrid Index."""
    os.makedirs(BASE_CHROMA_DIR, exist_ok=True)
    conn = sqlite3.connect(HYBRID_SQLITE_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_hybrid_db() -> None:
    """Create Categorical B-Tree Indexes and FTS5 Full-Text Inverted Index."""
    with get_hybrid_conn() as conn:
        # Index 1: Categorical Payload Table + B-Tree Indexes
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_metadata (
                chunk_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                file_name TEXT,
                source TEXT,
                medicine_name TEXT,
                sheet_name TEXT,
                page INTEGER,
                content TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_med ON chunk_metadata(session_id, medicine_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_sheet ON chunk_metadata(session_id, sheet_name)")

        # Index 2: Full-Text Keyword Index (Fixed SQLite FTS5 tokenize syntax)
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                chunk_id UNINDEXED,
                session_id UNINDEXED,
                medicine_name,
                sheet_name,
                content,
                tokenize="unicode61 tokenchars '-_'"
            )
            """
        )
        conn.commit()


def index_micro_batch_into_hybrid_db(session_id: str, docs: list[Document], ids: list[str]) -> None:
    """Insert a micro-batch of 100-250 chunks into the Dual-Layer Index."""
    init_hybrid_db()
    meta_rows = []
    fts_rows = []

    for cid, doc in zip(ids, docs):
        m = doc.metadata or {}
        med = str(m.get("medicine_name", "")).lower().strip()
        sheet = str(m.get("sheet_name", "")).lower().strip()
        fname = str(m.get("file_name") or m.get("source", "unknown"))
        src = str(m.get("source", fname))
        page = int(m.get("page", 1)) if isinstance(m.get("page"), int) else 1

        meta_rows.append((cid, session_id, fname, src, med, sheet, page, doc.page_content))
        fts_rows.append((cid, session_id, med, sheet, doc.page_content))

    with get_hybrid_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunk_metadata
            (chunk_id, session_id, file_name, source, medicine_name, sheet_name, page, content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            meta_rows,
        )
        conn.executemany("DELETE FROM chunk_fts WHERE chunk_id = ?", [(c,) for c in ids])
        conn.executemany(
            "INSERT INTO chunk_fts (chunk_id, session_id, medicine_name, sheet_name, content) VALUES (?, ?, ?, ?, ?)",
            fts_rows,
        )
        conn.commit()


def wipe_session_from_hybrid_db(session_id: str) -> None:
    """Remove all categorical and full-text index entries for a session."""
    init_hybrid_db()
    with get_hybrid_conn() as conn:
        conn.execute("DELETE FROM chunk_metadata WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chunk_fts WHERE session_id = ?", (session_id,))
        conn.commit()


def detect_categorical_filters(session_id: str, question: str) -> dict[str, str | None]:
    """
    Phase 5 (Step 1 - Hard Filtering via Categorical Indexes):
    Checks if the user's question references a specific indexed `medicine_name` or `sheet_name`
    so 99.9% of unrelated rows can be hidden before vector search.
    """
    init_hybrid_db()
    q_lower = question.lower()
    matched_med = None
    matched_sheet = None

    try:
        with get_hybrid_conn() as conn:
            meds = [
                r["medicine_name"]
                for r in conn.execute(
                    "SELECT DISTINCT medicine_name FROM chunk_metadata WHERE session_id = ? AND medicine_name != ''",
                    (session_id,),
                ).fetchall()
            ]
            sheets = [
                r["sheet_name"]
                for r in conn.execute(
                    "SELECT DISTINCT sheet_name FROM chunk_metadata WHERE session_id = ? AND sheet_name != ''",
                    (session_id,),
                ).fetchall()
            ]

        for med in meds:
            if len(med) >= 4 and med in q_lower:
                matched_med = med
                break
        for sheet in sheets:
            if len(sheet) >= 4 and sheet in q_lower and not sheet.startswith("page_"):
                matched_sheet = sheet
                break
    except Exception:
        pass

    return {"medicine_name": matched_med, "sheet_name": matched_sheet}


def search_fts_keywords(
    session_id: str,
    question: str,
    top_k: int,
    med_filter: str | None = None,
    sheet_filter: str | None = None,
) -> list[dict]:
    """
    Phase 5: Full-Text Keyword search using BM25 ranking over FTS5.
    Matches exact clinical trial IDs (NCT02475681), molecular variants (del(17p)), and dosages.
    """
    init_hybrid_db()
    stop_words = {
        "what", "where", "when", "which", "who", "this", "that", "with", "from",
        "about", "the", "and", "for", "are", "was", "were", "have", "has", "does",
        "summarize", "list", "show", "tell",
    }
    tokens = [t for t in re.findall(r"[A-Za-z0-9_\-]{2,}", question.lower()) if t not in stop_words][:8]
    if not tokens:
        return []

    fts_query = " OR ".join(f'"{t}"' for t in tokens)
    sql = """
        SELECT m.source, m.page, m.content
        FROM chunk_fts f
        JOIN chunk_metadata m ON f.chunk_id = m.chunk_id
        WHERE f.session_id = ? AND chunk_fts MATCH ?
    """
    params: list[Any] = [session_id, fts_query]

    if med_filter:
        sql += " AND m.medicine_name = ?"
        params.append(med_filter)
    if sheet_filter:
        sql += " AND m.sheet_name = ?"
        params.append(sheet_filter)

    sql += " ORDER BY bm25(chunk_fts) LIMIT ?"
    params.append(top_k)

    hits: list[dict] = []
    try:
        with get_hybrid_conn() as conn:
            for r in conn.execute(sql, params).fetchall():
                hits.append(
                    {
                        "source": r["source"],
                        "page": r["page"],
                        "match_pct": 95,
                        "text": r["content"],
                    }
                )
    except Exception:
        return []
    return hits
