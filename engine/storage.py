# 3. Update engine/storage.py
# (Upgraded hybrid_retrieve() with Multi-Signal Fusion so if a chunk matches FTS5 + Vector + Category, all badges and the exact Vector similarity % are combined onto the same source card!)

"""
Phase 3 & Phase 5: ChromaDB Vector Storage, Streaming Micro-Batch Ingestion,
Session State Persistence, Per-File Management, and High-Precision Hybrid Search.
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
    delete_file_from_hybrid_db,
    detect_categorical_filters,
    get_session_inventory,
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
    """Load persisted session names, chat history, and stats from disk JSON."""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_disk_metadata() -> None:
    """Persist current session list, active session ID, and session buckets to disk JSON."""
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


def extract_indexed_filenames(store: Chroma | None, session_id: str | None = None) -> list[str]:
    """
    Read unique filenames from both SQLite Hybrid DB (unlimited rows, O(1) index scan)
    and ChromaDB metadata so 10+ large Excel files are never truncated.
    """
    names: set[str] = set()

    if session_id:
        inv = get_session_inventory(session_id)
        for item in inv.get("files", []):
            if item.get("file_name"):
                names.add(str(item["file_name"]))

    if store is not None:
        try:
            data = store._collection.get(include=["metadatas"], limit=50000)
            metadatas = data.get("metadatas") or []
            for m in metadatas:
                if isinstance(m, dict) and (m.get("file_name") or m.get("source")):
                    names.add(str(m.get("file_name") or m.get("source", "unknown")))
        except Exception:
            pass

    return sorted(names)


def delete_single_file_from_session(session_id: str, file_name: str) -> int:
    """
    Improvement (Granular Document Management):
    Deletes all vector embeddings and FTS5/Categorical rows belonging to `file_name`
    inside `session_id` without wiping the rest of the session's knowledge base.
    """
    store = get_session_vector_store(session_id)
    removed_vectors = 0

    if store is not None:
        try:
            data = store._collection.get(include=["metadatas"], limit=50000)
            ids = data.get("ids") or []
            metadatas = data.get("metadatas") or []

            ids_to_delete = [
                doc_id
                for doc_id, meta in zip(ids, metadatas)
                if isinstance(meta, dict)
                and (
                    meta.get("file_name") == file_name
                    or meta.get("source") == file_name
                    or str(meta.get("source", "")).startswith(f"{file_name}")
                )
            ]

            if ids_to_delete:
                store._collection.delete(ids=ids_to_delete)
                removed_vectors = len(ids_to_delete)
        except Exception:
            pass

    removed_fts = delete_file_from_hybrid_db(session_id, file_name)
    return max(removed_vectors, removed_fts)


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
    Ensures every chunk carries `file_name` metadata for per-file filtering and deletion.
    """
    name = uploaded_file.name
    raw_data = uploaded_file.getvalue()

    if not raw_data:
        raise ValueError("file is empty.")
    if len(raw_data) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"file is larger than {MAX_FILE_MB} MB.")

    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if suffix in ("xlsx", "xls", "csv"):
        chunks = parse_excel_or_csv_bytes(name, raw_data)
    elif suffix == "pdf":
        chunks = parse_pdf_bytes(name, raw_data, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    elif suffix in ("txt", "md"):
        chunks = parse_text_or_md_bytes(name, raw_data, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    else:
        raise ValueError(f"Unsupported file format: .{suffix}")

    for c in chunks:
        if c.metadata is None:
            c.metadata = {}
        c.metadata.setdefault("file_name", name)

    return chunks


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

    _flush_micro_batch()
    elapsed = time.time() - started

    return {
        "store": store,
        "good_files": good_files,
        "chunks": store._collection.count(),
        "seconds": elapsed,
        "failed": failed,
    }


def _build_source_badges(item: dict) -> list[dict[str, str]]:
    """
    Construct the list of structured, color-coded badge descriptors for a retrieved chunk:
      - 🏷️ Exact Category Match (badge-category)
      - 🔤 FTS5 Keyword Match   (badge-fts)
      - 🧠 Semantic Vector (X%) (badge-vector)
    """
    badges: list[dict[str, str]] = []
    if item.get("matched_category"):
        badges.append({"label": "🏷️ Exact Category Match", "css": "badge-category"})
    if item.get("matched_fts"):
        badges.append({"label": "🔤 FTS5 Keyword Match", "css": "badge-fts"})
    if item.get("matched_vector"):
        pct = item.get("match_pct")
        vec_label = f"🧠 Semantic Vector ({pct}%)" if isinstance(pct, int) else "🧠 Semantic Vector (MMR)"
        badges.append({"label": vec_label, "css": "badge-vector"})
    return badges


# --------------------------------------------------------------------------- #
# Phase 5: High-Precision Hybrid Search with Multi-Signal Attribution Fusion
# --------------------------------------------------------------------------- #
def hybrid_retrieve(
    store: Chroma,
    session_id: str,
    question: str,
    top_k: int,
    use_mmr: bool = True,
    allowed_files: list[str] | None = None,
) -> list[dict]:
    """
    Phase 5 Enterprise Multi-Stage Hybrid Search:
      - Step 1 (Categorical B-Tree Hard Filter): Narrows pool by `medicine_name` & `sheet_name`.
      - Step 2 (SQLite FTS5 Keyword Search): BM25 ranking for exact clinical terms & IDs.
      - Step 3 (Chroma Semantic Vector Search): MMR or Similarity search with score lookup.
      - Step 4 (Multi-Signal Attribution Fusion): If a chunk is retrieved by BOTH FTS5 and Vector
        search, fuses their attribution flags onto a single source entry and computes color-coded
        pill badges (`🏷️ Exact Category Match`, `🔤 FTS5 Keyword Match`, `🧠 Semantic Vector (84%)`).
    """
    detected = detect_categorical_filters(session_id, question, allowed_files=allowed_files)
    med_filter = detected.get("medicine_name")
    sheet_filter = detected.get("sheet_name")

    conditions: list[dict] = []
    if med_filter:
        conditions.append({"medicine_name": med_filter})
    if sheet_filter:
        conditions.append({"sheet_name": sheet_filter})
    if allowed_files:
        if len(allowed_files) == 1:
            conditions.append({"file_name": allowed_files[0]})
        else:
            conditions.append({"file_name": {"$in": allowed_files}})

    chroma_where: dict | None = None
    if len(conditions) == 1:
        chroma_where = conditions[0]
    elif len(conditions) > 1:
        chroma_where = {"$and": conditions}

    # Stage 2: FTS5 Keyword Search (BM25)
    fts_hits = search_fts_keywords(
        session_id=session_id,
        question=question,
        top_k=max(2, top_k // 2),
        med_filter=med_filter,
        sheet_filter=sheet_filter,
        allowed_files=allowed_files,
    )

    # Stage 3: Scoped Semantic Vector Search
    # Also fetch similarity scores in a fast lookup map so MMR results still get an exact % score!
    score_lookup: dict[str, int] = {}
    try:
        scored_pairs = store.similarity_search_with_score(
            question, k=max(10, top_k * 2), filter=chroma_where
        )
        for s_doc, dist in scored_pairs:
            if isinstance(dist, (int, float)):
                pct = max(0, min(100, int((1.0 / (1.0 + max(0.0, float(dist)))) * 100)))
                score_lookup[s_doc.page_content[:200]] = pct
    except Exception:
        scored_pairs = []

    vector_results = []
    used_cat_filter = bool(med_filter or sheet_filter)
    try:
        if use_mmr:
            docs = store.max_marginal_relevance_search(
                question, k=top_k, fetch_k=max(10, top_k * 3), filter=chroma_where
            )
            vector_results = [(d, None) for d in docs]
        else:
            vector_results = scored_pairs[:top_k]
    except Exception:
        vector_results = []

    # Fallback: If hard categorical filter yielded 0 vector hits, relax categorical filter
    if not vector_results and chroma_where is not None:
        used_cat_filter = False
        file_only_where: dict | None = None
        if allowed_files:
            file_only_where = (
                {"file_name": allowed_files[0]}
                if len(allowed_files) == 1
                else {"file_name": {"$in": allowed_files}}
            )
        try:
            scored_fallback = store.similarity_search_with_score(
                question, k=max(10, top_k * 2), filter=file_only_where
            )
            for s_doc, dist in scored_fallback:
                if isinstance(dist, (int, float)):
                    pct = max(0, min(100, int((1.0 / (1.0 + max(0.0, float(dist)))) * 100)))
                    score_lookup[s_doc.page_content[:200]] = pct

            if use_mmr:
                docs = store.max_marginal_relevance_search(
                    question, k=top_k, fetch_k=max(10, top_k * 3), filter=file_only_where
                )
                vector_results = [(d, None) for d in docs]
            else:
                vector_results = scored_fallback[:top_k]
        except Exception:
            fallback_docs = store.similarity_search(question, k=top_k)
            vector_results = [(d, None) for d in fallback_docs]

    # Stage 4: Multi-Signal Attribution Fusion (Merge FTS5 + Vector + Categorical signals)
    merged_by_sig: dict[str, dict] = {}
    ordered_sigs: list[str] = []

    for hit in fts_hits:
        sig = hit["text"][:200]
        if sig not in merged_by_sig:
            # Attach similarity % if Chroma also scored this chunk
            hit["match_pct"] = score_lookup.get(sig)
            merged_by_sig[sig] = hit
            ordered_sigs.append(sig)

    for doc, distance in vector_results:
        sig = doc.page_content[:200]
        match_pct = (
            max(0, min(100, int((1.0 / (1.0 + max(0.0, float(distance)))) * 100)))
            if isinstance(distance, (int, float))
            else score_lookup.get(sig)
        )

        if sig in merged_by_sig:
            # Chunk was already found by FTS5! Fuse the Vector signal onto the existing entry
            existing = merged_by_sig[sig]
            existing["matched_vector"] = True
            existing["matched_category"] = existing.get("matched_category") or used_cat_filter
            if match_pct is not None:
                existing["match_pct"] = match_pct
        else:
            merged_by_sig[sig] = {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "match_pct": match_pct,
                "matched_category": used_cat_filter,
                "matched_fts": False,
                "matched_vector": True,
                "text": doc.page_content,
            }
            ordered_sigs.append(sig)

    # Finalize structured badges list for each source card
    final_sources: list[dict] = []
    for sig in ordered_sigs[:top_k]:
        item = merged_by_sig[sig]
        item["badges"] = _build_source_badges(item)
        final_sources.append(item)

    return final_sources
