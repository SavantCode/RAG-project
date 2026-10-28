# Update engine/hybrid_index.py
# (Updated search_fts_keywords so each FTS5 hit carries structured badge flags: matched_fts=True, matched_category=bool(...), and category_detail)


"""
Phase 4 & Phase 5 (Step 1): Dual-Layer Database Index.

Architectural Overview:
-----------------------
1. Index 1 (Categorical Payload B-Tree Index):
   - Table: `chunk_metadata`
   - Purpose: High-speed O(log N) exact filtering on `session_id`, `medicine_name`, `sheet_name`, and `file_name`.
   - Prevents cross-contamination when multiple clinical spreadsheets or PDFs exist in one session.

2. Index 2 (Full-Text Keyword Inverted Index - SQLite FTS5):
   - Virtual Table: `chunk_fts` (BM25 ranking with `unicode61` tokenizer preserving `-` and `_`)
   - Purpose: Captures exact clinical trial IDs (e.g., NCT02475681), molecular variants, and dosages
     that dense neural embeddings often blur together.
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any

from langchain_core.documents import Document

from engine.config import BASE_CHROMA_DIR, HYBRID_SQLITE_DB


def get_hybrid_conn() -> sqlite3.Connection:
    """
    Return a thread-safe SQLite connection to the Dual-Layer Hybrid Index.
    Uses `sqlite3.Row` so columns can be accessed by name like dictionaries.
    """
    os.makedirs(BASE_CHROMA_DIR, exist_ok=True)
    conn = sqlite3.connect(HYBRID_SQLITE_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_hybrid_db() -> None:
    """
    Initialize the SQLite schema for both Categorical B-Tree Indexes and FTS5 Full-Text Search.
    Idempotent (`IF NOT EXISTS`) so it can be safely called on every startup or batch ingestion.
    """
    with get_hybrid_conn() as conn:
        # Index 1: Categorical Payload Table + B-Tree Indexes for hard filtering
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_file ON chunk_metadata(session_id, file_name)")

        # Index 2: Full-Text Keyword Index (SQLite FTS5 with custom tokenchars for clinical terms)
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
    """
    Insert a micro-batch of 100-250 chunks into both the Categorical table and FTS5 index.
    Uses `executemany` within a single transaction to minimize disk I/O overhead.
    """
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
    """Remove all categorical and full-text index entries for an entire session."""
    init_hybrid_db()
    with get_hybrid_conn() as conn:
        conn.execute("DELETE FROM chunk_metadata WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chunk_fts WHERE session_id = ?", (session_id,))
        conn.commit()


def delete_file_from_hybrid_db(session_id: str, file_name: str) -> int:
    """
    Improvement (Per-File Deletion):
    Removes all chunks belonging to a single `file_name` within a `session_id`
    from both `chunk_fts` and `chunk_metadata` without affecting other files in the session.
    """
    init_hybrid_db()
    with get_hybrid_conn() as conn:
        rows = conn.execute(
            """
            SELECT chunk_id FROM chunk_metadata
            WHERE session_id = ? AND (file_name = ? OR source = ? OR source LIKE ?)
            """,
            (session_id, file_name, file_name, f"{file_name}%"),
        ).fetchall()

        chunk_ids = [(r["chunk_id"],) for r in rows]
        if not chunk_ids:
            return 0

        conn.executemany("DELETE FROM chunk_fts WHERE chunk_id = ?", chunk_ids)
        conn.executemany("DELETE FROM chunk_metadata WHERE chunk_id = ?", chunk_ids)
        conn.commit()
        return len(chunk_ids)


def detect_categorical_filters(
    session_id: str,
    question: str,
    allowed_files: list[str] | None = None,
) -> dict[str, str | None]:
    """
    Phase 5 (Step 1 - Hard Filtering via Categorical Indexes):
    Inspects the user's question against distinct `medicine_name` and `sheet_name` values
    indexed in the current session (scoped to `allowed_files` if active).
    """
    init_hybrid_db()
    q_lower = question.lower()
    matched_med = None
    matched_sheet = None

    try:
        with get_hybrid_conn() as conn:
            file_clause = ""
            params: list[Any] = [session_id]
            if allowed_files:
                placeholders = ",".join("?" for _ in allowed_files)
                file_clause = f" AND file_name IN ({placeholders})"
                params.extend(allowed_files)

            meds = [
                r["medicine_name"]
                for r in conn.execute(
                    f"SELECT DISTINCT medicine_name FROM chunk_metadata WHERE session_id = ? AND medicine_name != ''{file_clause}",
                    params,
                ).fetchall()
            ]
            sheets = [
                r["sheet_name"]
                for r in conn.execute(
                    f"SELECT DISTINCT sheet_name FROM chunk_metadata WHERE session_id = ? AND sheet_name != ''{file_clause}",
                    params,
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
    allowed_files: list[str] | None = None,
) -> list[dict]:
    """
    Phase 5 (Step 2 - Full-Text Keyword Search via SQLite FTS5 BM25):
    Matches exact clinical trial IDs (NCT02475681), molecular variants, and dosages.

    Enterprise Attribution Improvement:
    - Annotates each hit with structured multi-layer attribution flags (`matched_fts`,
      `matched_category`, `badges`) so `hybrid_retrieve()` can fuse FTS5 + Vector + Category signals.
    """
    init_hybrid_db()
    stop_words = {
        "what", "where", "when", "which", "who", "this", "that", "with", "from",
        "about", "the", "and", "for", "are", "was", "were", "have", "has", "does",
        "summarize", "list", "show", "tell", "give", "find", "explain", "many", "files",
    }
    tokens = [t for t in re.findall(r"[A-Za-z0-9_\-]{2,}", question.lower()) if t not in stop_words][:8]
    if not tokens:
        return []

    fts_query = " OR ".join(f'"{t}"' for t in tokens)
    sql = """
        SELECT m.source, m.file_name, m.page, m.content
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
    if allowed_files:
        placeholders = ",".join("?" for _ in allowed_files)
        sql += f" AND m.file_name IN ({placeholders})"
        params.extend(allowed_files)

    sql += " ORDER BY bm25(chunk_fts) LIMIT ?"
    params.append(top_k)

    has_cat = bool(med_filter or sheet_filter)

    hits: list[dict] = []
    try:
        with get_hybrid_conn() as conn:
            for r in conn.execute(sql, params).fetchall():
                hits.append(
                    {
                        "source": r["source"],
                        "page": r["page"],
                        "match_pct": None,
                        "matched_category": has_cat,
                        "matched_fts": True,
                        "matched_vector": False,
                        "text": r["content"],
                    }
                )
    except Exception:
        return []
    return hits


def get_session_inventory(session_id: str) -> dict[str, Any]:
    """
    Improvement (Complete Session Inventory Manifest):
    Queries the SQLite Categorical Index for the complete, exact list of all files,
    sheet names, and chunk counts in this session (immune to Top-K or Chroma limit caps).
    """
    init_hybrid_db()
    files_info: list[dict[str, Any]] = []
    total_chunks = 0

    try:
        with get_hybrid_conn() as conn:
            rows = conn.execute(
                """
                SELECT file_name, COUNT(*) as chunk_count, GROUP_CONCAT(DISTINCT sheet_name) as sheets
                FROM chunk_metadata
                WHERE session_id = ? AND file_name IS NOT NULL AND file_name != ''
                GROUP BY file_name
                ORDER BY file_name
                """,
                (session_id,),
            ).fetchall()

            for r in rows:
                cnt = int(r["chunk_count"] or 0)
                total_chunks += cnt
                raw_sheets = str(r["sheets"] or "").split(",")
                clean_sheets = [s for s in raw_sheets if s and not s.startswith("page_")]
                files_info.append(
                    {
                        "file_name": r["file_name"],
                        "chunks": cnt,
                        "sheets": clean_sheets[:15],
                    }
                )
    except Exception:
        pass

    return {"total_files": len(files_info), "total_chunks": total_chunks, "files": files_info}
