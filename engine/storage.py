"""
Phase 3 & Phase 5: ChromaDB Vector Storage, Streaming Micro-Batch Ingestion,
Session State Persistence, and High-Precision Hybrid Search.
"""

from __future__ import annotations

import gc
import hashlib
import html
import json
import os
import shutil
import time
from typing import Callable

import chromadb
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from engine.config import (
    BASE_CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MAX_FILE_MB,
    METADATA_FILE,
    MICRO_BATCH_SIZE,
)
from engine.hybrid_index import (
    detect_categorical_filters,
    index_micro_batch_into_hybrid_db,
    search_fts_keywords,
    wipe_session_from_hybrid_db,
)
from excelreader import parse_excel_or_csv_bytes
from filereader import parse_pdf_bytes, parse_text_or_md_bytes


# --------------------------------------------------------------------------- #
# Disk Metadata Persistence
# --------------------------------------------------------------------------- #
def load_disk_metadata() -> dict:
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_disk_metadata() -> None:
    os.makedirs(BASE_CHROMA_DIR, exist_ok=True)
    try:
        payload = {
            "current_session_id": st.session_state.get("current_session_id"),
            "sessions": st.session_state.get("sessions", []),
            "session_data": st.session_state.get("session_data", {}),
        }
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# ChromaDB & Embedding Singletons
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local HuggingFace embedding model once per server process."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_chroma_client() -> chromadb.PersistentClient:
    """Shared PersistentClient prevents Windows SQLite file lock collisions."""
    os.makedirs(BASE_CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=BASE_CHROMA_DIR)


def list_collection_names(client: chromadb.PersistentClient) -> list[str]:
    """Compatible across ChromaDB 0.4, 0.5, 0.6, and 1.x."""
    try:
        cols = client.list_collections()
        return [c if isinstance(c, str) else c.name for c in cols]
    except Exception:
        return []


def discover_existing_sessions() -> list[str]:
    """Auto-discover existing sessions in Chroma collections and legacy subfolders."""
    discovered: list[str] = []
    client = get_chroma_client()

    for col_name in list_collection_names(client):
        if col_name.startswith("session_"):
            sid = col_name.replace("session_", "", 1)
            try:
                if client.get_collection(col_name).count() > 0 and sid not in discovered:
                    discovered.append(sid)
            except Exception:
                pass

    if os.path.exists(BASE_CHROMA_DIR):
        for item in os.listdir(BASE_CHROMA_DIR):
            sub_path = os.path.join(BASE_CHROMA_DIR, item)
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "chroma.sqlite3")):
                if item not in discovered:
                    discovered.append(item)

    return discovered


def get_session_vector_store(session_id: str) -> Chroma | None:
    """Load Chroma store for session_id if it contains indexed vectors."""
    client = get_chroma_client()
    collection_name = f"session_{session_id}"

    if collection_name in list_collection_names(client):
        try:
            store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=get_embeddings(),
            )
            if store._collection.count() > 0:
                return store
        except Exception:
            pass

    legacy_dir = os.path.join(BASE_CHROMA_DIR, session_id)
    if os.path.isdir(legacy_dir) and os.path.exists(os.path.join(legacy_dir, "chroma.sqlite3")):
        try:
            legacy_store = Chroma(
                persist_directory=legacy_dir,
                embedding_function=get_embeddings(),
            )
            if legacy_store._collection.count() > 0:
                return legacy_store
        except Exception:
            pass

    return None


def extract_indexed_filenames(store: Chroma | None) -> list[str]:
    """Read unique filenames from ChromaDB chunk metadata."""
    if store is None:
        return []
    try:
        data = store._collection.get(include=["metadatas"], limit=15000)
        metadatas = data.get("metadatas") or []
        return sorted(
            {
                m.get("file_name") or m.get("source", "unknown")
                for m in metadatas
                if isinstance(m, dict) and (m.get("file_name") or m.get("source"))
            }
        )
    except Exception:
        return []


def wipe_session_data(session_id: str) -> None:
    """Completely delete a session's Chroma collection, legacy folder, and Hybrid FTS5 index."""
    client = get_chroma_client()
    collection_name = f"session_{session_id}"

    try:
        if collection_name in list_collection_names(client):
            client.delete_collection(name=collection_name)
    except Exception:
        pass

    legacy_dir = os.path.join(BASE_CHROMA_DIR, session_id)
    if os.path.exists(legacy_dir):
        try:
            legacy_store = Chroma(persist_directory=legacy_dir, embedding_function=get_embeddings())
            legacy_store.delete_collection()
            del legacy_store
        except Exception:
            pass
        gc.collect()
        shutil.rmtree(legacy_dir, ignore_errors=True)

    wipe_session_from_hybrid_db(session_id)


# --------------------------------------------------------------------------- #
# Modular File Router & Phase 3 Streaming Micro-Batch Ingestion
# --------------------------------------------------------------------------- #
def route_uploaded_file_to_chunks(uploaded_file) -> list[Document]:
    """
    Dispatch uploaded file by extension to its dedicated parser module:
      - .xlsx, .xls, .csv -> excelreader.py (Polars + Calamine + 1-Row-Per-Chunk)
      - .pdf              -> filereader.py  (parse_pdf_bytes)
      - .txt, .md         -> filereader.py  (parse_text_or_md_bytes)
    """
    name = uploaded_file.name
    raw_data = uploaded_file.getvalue()

    if not raw_data:
        raise ValueError("file is empty.")
    if len(raw_data) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"file is larger than {MAX_FILE_MB} MB.")

    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if suffix in ("xlsx", "xls", "csv"):
        return parse_excel_or_csv_bytes(name, raw_data)
    if suffix == "pdf":
        return parse_pdf_bytes(name, raw_data, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if suffix in ("txt", "md"):
        return parse_text_or_md_bytes(name, raw_data, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    raise ValueError(f"Unsupported file format: .{suffix}")


def ingest_files_with_micro_batches(
    uploaded_files: list,
    session_id: str,
    append_mode: bool,
    log_fn: Callable[[str], None],
    progress_fn: Callable[[float, str], None],
) -> dict:
    """
    Phase 3: Streaming Micro-Batch Architecture.
    Takes groups of MICRO_BATCH_SIZE (250) rows/chunks at a time, embeds them,
    and immediately commits them to ChromaDB + Dual-Layer Hybrid DB to protect RAM.
    """
    started = time.time()
    embeddings = get_embeddings()
    client = get_chroma_client()
    collection_name = f"session_{session_id}"

    if not append_mode:
        wipe_session_data(session_id)

    store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    good_files: list[str] = []
    failed: list[str] = []
    total_files = len(uploaded_files)
    committed_chunks = 0

    batch_docs: list[Document] = []
    batch_ids: list[str] = []
    seen_ids: set[str] = set()

    def _flush_micro_batch() -> None:
        nonlocal committed_chunks
        if not batch_docs:
            return
        store.add_documents(documents=list(batch_docs), ids=list(batch_ids))
        index_micro_batch_into_hybrid_db(session_id, list(batch_docs), list(batch_ids))
        committed_chunks += len(batch_docs)
        batch_docs.clear()
        batch_ids.clear()

    for f_idx, uf in enumerate(uploaded_files, start=1):
        try:
            file_chunks = route_uploaded_file_to_chunks(uf)
            good_files.append(uf.name)
            log_fn(f"✅ **{html.escape(uf.name)}** ({len(file_chunks):,} row/page chunks)")

            for c_idx, chunk in enumerate(file_chunks):
                cid = hashlib.sha256(
                    f"{chunk.metadata.get('source','')}_{chunk.metadata.get('page',0)}_{c_idx}_{chunk.page_content[:140]}".encode("utf-8")
                ).hexdigest()[:24]

                if cid not in seen_ids:
                    seen_ids.add(cid)
                    batch_docs.append(chunk)
                    batch_ids.append(cid)

                if len(batch_docs) >= MICRO_BATCH_SIZE:
                    _flush_micro_batch()
                    progress_fn(
                        min(1.0, f_idx / total_files),
                        f"Streamed {committed_chunks:,} chunks to DB ({f_idx}/{total_files} files)…",
                    )
        except Exception as exc:
            failed.append(f"{uf.name}: {exc}")
            log_fn(f"⚠️ **{html.escape(uf.name)}** skipped — {exc}")

    # Flush final micro-batch
    _flush_micro_batch()
    elapsed = time.time() - started

    return {
        "store": store,
        "good_files": good_files,
        "chunks": store._collection.count(),
        "seconds": elapsed,
        "failed": failed,
    }


# --------------------------------------------------------------------------- #
# Phase 5: High-Precision Hybrid Search (Retrieval)
# --------------------------------------------------------------------------- #
def hybrid_retrieve(
    store: Chroma,
    session_id: str,
    question: str,
    top_k: int,
    use_mmr: bool = True,
) -> list[dict]:
    """
    Phase 5:
      - Step 1 (Hard Filtering via Categorical Indexes): Narrow search pool by medicine_name & sheet_name.
      - Step 2 (Full-Text Keyword Lookup): Match exact clinical trial IDs (NCT...), variants (del(17p)).
      - Step 3 (Semantic Vector Scan): Perform neural vector search on the filtered pool.
    """
    detected = detect_categorical_filters(session_id, question)
    med_filter = detected.get("medicine_name")
    sheet_filter = detected.get("sheet_name")

    chroma_where: dict | None = None
    if med_filter and sheet_filter:
        chroma_where = {"$and": [{"medicine_name": med_filter}, {"sheet_name": sheet_filter}]}
    elif med_filter:
        chroma_where = {"medicine_name": med_filter}
    elif sheet_filter:
        chroma_where = {"sheet_name": sheet_filter}

    # FTS5 Keyword Search
    fts_hits = search_fts_keywords(
        session_id=session_id,
        question=question,
        top_k=max(2, top_k // 2),
        med_filter=med_filter,
        sheet_filter=sheet_filter,
    )

    # Scoped Semantic Vector Search
    vector_results = []
    try:
        if use_mmr:
            docs = store.max_marginal_relevance_search(
                question, k=top_k, fetch_k=max(10, top_k * 3), filter=chroma_where
            )
            vector_results = [(d, None) for d in docs]
        else:
            vector_results = store.similarity_search_with_score(question, k=top_k, filter=chroma_where)
    except Exception:
        vector_results = []

    # Fallback: If hard filter yielded 0 vector hits, run unfiltered semantic search
    if not vector_results and chroma_where is not None:
        try:
            if use_mmr:
                docs = store.max_marginal_relevance_search(question, k=top_k, fetch_k=max(10, top_k * 3))
                vector_results = [(d, None) for d in docs]
            else:
                vector_results = store.similarity_search_with_score(question, k=top_k)
        except Exception:
            fallback_docs = store.similarity_search(question, k=top_k)
            vector_results = [(d, None) for d in fallback_docs]

    # Merge & deduplicate FTS5 + Vector results
    sources: list[dict] = []
    seen_texts: set[str] = set()

    for hit in fts_hits:
        sig = hit["text"][:200]
        if sig not in seen_texts:
            seen_texts.add(sig)
            sources.append(hit)

    for doc, distance in vector_results:
        sig = doc.page_content[:200]
        if sig in seen_texts:
            continue
        seen_texts.add(sig)
        match_pct = (
            max(0, min(100, int((1.0 / (1.0 + max(0.0, float(distance)))) * 100)))
            if isinstance(distance, (int, float))
            else None
        )
        sources.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "match_pct": match_pct,
                "text": doc.page_content,
            }
        )

    return sources[:top_k]
