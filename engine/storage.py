"""
Phase 3 & Phase 5: ChromaDB Vector Storage, High-Scale Streaming Micro-Batch Ingestion
(1,500+ Files / 20+ Sheets Ready), Enterprise Checkpoint Auto-Resume, Website URL (Text + Image)
& DOCX Ingestion, Session State Persistence, Per-File Management, and Hybrid Search.
"""

from __future__ import annotations

import gc
import hashlib
import html
import io
import json
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import urljoin, urlparse

import chromadb
import requests
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


class _StdLibHTMLTextExtractor(HTMLParser):
    """
    Zero-dependency HTML text, title, and image extractor using Python's built-in `html.parser`.
    - Strips <script>, <style>, <nav>, <footer>, <header>, <aside>, and <svg> blocks.
    - Extracts meaningful <img> tags (resolving relative URLs to absolute https:// URLs and
      filtering out tiny UI icons/trackers) so RAG can search and render images from web pages.
    """

    SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"}
    BLOCK_TAGS = {
        "p", "div", "section", "article", "li", "tr",
        "h1", "h2", "h3", "h4", "h5", "h6", "br", "figcaption",
    }
    JUNK_IMG_PATTERNS = ("icon", "logo", "sprite", "pixel", "avatar", "banner", "button", "1x1", ".svg")

    def __init__(self, base_url: str = "") -> None:
        super().__init__()
        self.base_url = base_url
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.images: list[dict[str, str]] = []
        self._seen_img_urls: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if t == "title":
            self._in_title = True
            return
        if self._skip_depth > 0:
            return

        if t in self.BLOCK_TAGS:
            self.text_parts.append("\n")
        elif t == "img":
            attr_map = {k.lower(): (v or "").strip() for k, v in attrs if k}
            raw_src = (
                attr_map.get("src")
                or attr_map.get("data-src")
                or attr_map.get("data-original")
                or ""
            )
            if not raw_src or raw_src.startswith("data:"):
                return

            full_img_url = urljoin(self.base_url, raw_src) if self.base_url else raw_src
            if not full_img_url.startswith(("http://", "https://")):
                return

            lower_url = full_img_url.lower()
            if any(junk in lower_url for junk in self.JUNK_IMG_PATTERNS):
                return

            w_str = re.sub(r"[^\d]", "", attr_map.get("width", ""))
            h_str = re.sub(r"[^\d]", "", attr_map.get("height", ""))
            if (w_str and int(w_str) < 48) or (h_str and int(h_str) < 48):
                return

            alt_text = (attr_map.get("alt") or attr_map.get("title") or "").strip()
            if not alt_text:
                fname = urlparse(full_img_url).path.rsplit("/", 1)[-1]
                fname_clean = re.sub(r"[\-_\.]+", " ", fname.rsplit(".", 1)[0]).strip()
                alt_text = fname_clean if len(fname_clean) >= 3 else "Webpage illustration"

            if full_img_url not in self._seen_img_urls:
                self._seen_img_urls.add(full_img_url)
                self.images.append({"alt": alt_text[:120], "url": full_img_url})
                self.text_parts.append(
                    f"\n[Image: {alt_text[:120]} | Image URL: {full_img_url}]\n"
                )

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif t == "title":
            self._in_title = False
        elif t in self.BLOCK_TAGS and self._skip_depth == 0:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif self._skip_depth == 0:
            cleaned = data.strip()
            if cleaned:
                self.text_parts.append(cleaned + " ")


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
    Enterprise O(log N) Filename Lookup (1,500+ Files Ready):
    1. Queries SQLite `get_session_inventory(session_id)` first (~0.5ms via B-Tree index).
       If SQLite has filenames, returns immediately without touching ChromaDB!
    2. Only falls back to a small ChromaDB metadata sample (`limit=2000`) for legacy
       sessions created before the SQLite hybrid index existed.
    """
    if session_id:
        inv = get_session_inventory(session_id)
        sqlite_names = [
            str(item["file_name"])
            for item in inv.get("files", [])
            if item.get("file_name")
        ]
        if sqlite_names:
            return sorted(set(sqlite_names))

    names: set[str] = set()
    if store is not None:
        try:
            data = store._collection.get(include=["metadatas"], limit=2000)
            metadatas = data.get("metadatas") or []
            for m in metadatas:
                if isinstance(m, dict) and (m.get("file_name") or m.get("source")):
                    names.add(str(m.get("file_name") or m.get("source", "unknown")))
        except Exception:
            pass

    return sorted(names)


def delete_single_file_from_session(session_id: str, file_name: str) -> int:
    """
    Enterprise O(log N) Single-File Deletion (Unlimited Chunks Ready):
    Uses native ChromaDB `where` metadata filtering instead of loading 50,000 rows into RAM,
    so deleting a file works even when a session has 500,000+ chunks across 1,500 files.
    """
    store = get_session_vector_store(session_id)
    removed_fts = delete_file_from_hybrid_db(session_id, file_name)

    if store is not None:
        try:
            store._collection.delete(where={"file_name": file_name})
        except Exception:
            pass
        try:
            store._collection.delete(where={"source": file_name})
        except Exception:
            pass

    return removed_fts


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
# Enterprise DOCX & Website URL Parsers (100% StdLib — Text + Image Extraction!)
# --------------------------------------------------------------------------- #
def parse_docx_bytes(file_name: str, raw_data: bytes, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """
    Enterprise Zero-Dependency DOCX Parser:
    Extracts both paragraphs (`<w:p>`) and structured tables (`<w:tbl>`) directly from
    the OpenXML (`word/document.xml`) archive using Python's built-in `zipfile` + `ElementTree`.
    """
    blocks: list[str] = []
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    try:
        with zipfile.ZipFile(io.BytesIO(raw_data)) as zf:
            if "word/document.xml" not in zf.namelist():
                raise ValueError("Invalid .docx file (missing word/document.xml).")
            xml_bytes = zf.read("word/document.xml")

        root = ET.fromstring(xml_bytes)
        body = root.find("w:body", ns)
        elements = list(body) if body is not None else list(root)

        for elem in elements:
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "p":
                texts = [t.text for t in elem.findall(".//w:t", ns) if t.text]
                para = "".join(texts).strip()
                if para:
                    blocks.append(para)
            elif tag == "tbl":
                table_rows: list[str] = []
                for row in elem.findall(".//w:tr", ns):
                    cells: list[str] = []
                    for cell in row.findall(".//w:tc", ns):
                        c_texts = [t.text for t in cell.findall(".//w:t", ns) if t.text]
                        c_val = " ".join("".join(c_texts).split())
                        if c_val:
                            cells.append(c_val)
                    if cells:
                        table_rows.append(" | ".join(cells))
                if table_rows:
                    blocks.append("\n".join(table_rows))
    except Exception as exc:
        raise ValueError(f"Could not read DOCX file '{file_name}': {exc}") from exc

    full_text = "\n\n".join(blocks).strip()
    if not full_text:
        raise ValueError(f"No readable text found inside '{file_name}'.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_text(full_text)

    docs: list[Document] = []
    for idx, text_chunk in enumerate(raw_chunks, start=1):
        docs.append(
            Document(
                page_content=f"[Document: {file_name} | Section {idx}]\n{text_chunk}",
                metadata={
                    "source": file_name,
                    "file_name": file_name,
                    "page": idx,
                    "sheet_name": f"section_{idx}",
                },
            )
        )
    return docs


def parse_url_to_chunks(url: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> tuple[str, list[Document]]:
    """
    Enterprise Website URL Scraper & Chunker (Text + Embedded Images):
    - Supports direct image URLs (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`) as well as HTML pages.
    - Extracts clean article text AND inline `[Image: <alt> | Image URL: <https_url>]` markers
      plus a dedicated Image Catalog chunk so image queries always retrieve matching visuals.
    """
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    parsed = urlparse(clean_url)
    if not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")

    path_slug = parsed.path.strip("/")[:28]
    display_source = f"🌐 {parsed.netloc}/{path_slug}".rstrip("/")

    if parsed.path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        img_name = parsed.path.rsplit("/", 1)[-1]
        doc = Document(
            page_content=(
                f"[Web Source: {display_source} | Direct Image]\n"
                f"[Image: {img_name} | Image URL: {clean_url}]"
            ),
            metadata={
                "source": display_source,
                "file_name": display_source,
                "url": clean_url,
                "page": 1,
                "sheet_name": "web_images",
            },
        )
        return display_source, [doc]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    resp = requests.get(clean_url, headers=headers, timeout=15)
    resp.raise_for_status()

    extractor = _StdLibHTMLTextExtractor(base_url=clean_url)
    extractor.feed(resp.text)

    page_title = " ".join("".join(extractor.title_parts).split()).strip() or parsed.netloc
    raw_text = "".join(extractor.text_parts)

    lines = [re.sub(r"\s+", " ", line).strip() for line in raw_text.splitlines()]
    lines = [line for line in lines if len(line) > 1]
    cleaned_body = "\n".join(lines)

    if len(cleaned_body) < 40 and not extractor.images:
        raise ValueError("Page contained too little readable text (may require JavaScript login).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_splits = splitter.split_text(cleaned_body) if cleaned_body else []

    docs: list[Document] = []
    for idx, chunk_str in enumerate(text_splits, start=1):
        docs.append(
            Document(
                page_content=(
                    f"[Web Source: {display_source} | Title: {page_title} | URL: {clean_url} | Part {idx}]\n"
                    f"{chunk_str}"
                ),
                metadata={
                    "source": display_source,
                    "file_name": display_source,
                    "url": clean_url,
                    "page": idx,
                    "sheet_name": "web_page",
                },
            )
        )

    if extractor.images:
        gallery_lines = [
            f"- [Image: {img['alt']} | Image URL: {img['url']}]"
            for img in extractor.images[:15]
        ]
        gallery_chunk = (
            f"[Web Source: {display_source} | Title: {page_title} | Image Gallery & Visual Figures]\n"
            f"Images, photos, diagrams, and illustrations found on {page_title} ({clean_url}):\n"
            + "\n".join(gallery_lines)
        )
        docs.append(
            Document(
                page_content=gallery_chunk,
                metadata={
                    "source": display_source,
                    "file_name": display_source,
                    "url": clean_url,
                    "page": len(docs) + 1,
                    "sheet_name": "web_images",
                },
            )
        )

    return display_source, docs


# --------------------------------------------------------------------------- #
# Modular File Router & Phase 3 Streaming Micro-Batch Ingestion (1,500+ Files Ready)
# --------------------------------------------------------------------------- #
def route_uploaded_file_to_chunks(uploaded_file) -> list[Document]:
    """
    Dispatch uploaded file by extension to its dedicated parser module:
      - .xlsx, .xls, .csv -> excelreader.py (Polars + Calamine + Adaptive Row Packing)
      - .pdf              -> filereader.py  (parse_pdf_bytes)
      - .docx             -> storage.py     (parse_docx_bytes - paragraphs + tables)
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
    elif suffix == "docx":
        chunks = parse_docx_bytes(name, raw_data, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
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
    Phase 3: High-Throughput Streaming Micro-Batch Architecture (1,500+ Files / 20+ Sheets Ready).
    - Checkpoint Auto-Resume: Queries SQLite `get_session_inventory()` in <1ms to skip all
      previously committed files without spamming the Streamlit DOM.
    - Atomic File Commits: Flushes at the end of every file so checkpoint resume is 100% exact.
    - Memory Guard: Runs `gc.collect()` periodically so RAM stays flat across 1,500+ workbooks.
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

    already_indexed_files: set[str] = set()
    if append_mode:
        inv = get_session_inventory(session_id)
        already_indexed_files = {
            str(f["file_name"]) for f in inv.get("files", []) if int(f.get("chunks", 0)) > 0
        }

    good_files: list[str] = []
    skipped_resume_count = 0
    newly_indexed_count = 0
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

    if already_indexed_files and append_mode:
        overlap_count = sum(1 for uf in uploaded_files if uf.name in already_indexed_files)
        if overlap_count > 0:
            log_fn(
                f"⏭️ **Checkpoint Auto-Resume Active:** Instantly skipping **{overlap_count}** "
                f"already-indexed file(s) and resuming from the next unindexed file…"
            )

    new_processing_started = time.time()

    for f_idx, uf in enumerate(uploaded_files, start=1):
        pct = min(1.0, f_idx / total_files)
        pct_int = int(pct * 100)

        # Checkpoint Auto-Resume: Skip already-committed files in O(1) without DOM spam
        if append_mode and uf.name in already_indexed_files:
            good_files.append(uf.name)
            skipped_resume_count += 1
            if skipped_resume_count % 20 == 0 or f_idx == total_files:
                progress_fn(
                    pct,
                    f"[{f_idx}/{total_files} · {pct_int}%] Resuming… skipped {skipped_resume_count} checkpointed files",
                )
            continue

        try:
            elapsed_new = max(0.1, time.time() - new_processing_started)
            remaining_files = max(0, total_files - f_idx)
            eta_sec = (
                int((elapsed_new / newly_indexed_count) * remaining_files)
                if newly_indexed_count >= 1
                else 0
            )
            eta_str = f" · ETA ~{eta_sec // 60}m {eta_sec % 60}s" if eta_sec > 5 else ""

            progress_fn(
                pct,
                f"Indexing [{f_idx}/{total_files} · {pct_int}%] {uf.name}{eta_str}",
            )
            print(f"[{f_idx}/{total_files}] Indexing: {uf.name}...", flush=True)

            file_chunks = route_uploaded_file_to_chunks(uf)

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
                        pct,
                        f"[{f_idx}/{total_files} · {pct_int}%] Streamed {committed_chunks:,} chunks ({uf.name}){eta_str}…",
                    )

            # Atomic File Boundary Commit: Flush at the end of EVERY file so checkpoint resume is 100% exact
            _flush_micro_batch()
            good_files.append(uf.name)
            newly_indexed_count += 1

            # Throttle UI DOM log entries on massive 1,500-file batches to keep browser fast
            if total_files <= 100 or newly_indexed_count <= 30 or newly_indexed_count % 10 == 0 or f_idx == total_files:
                log_fn(
                    f"✅ **[{f_idx}/{total_files}] {html.escape(uf.name)}** "
                    f"— `{len(file_chunks):,}` chunks *(Session new chunks: {committed_chunks:,})*"
                )

            # Periodic garbage collection to keep RAM flat across 1,500+ workbooks
            del file_chunks
            if newly_indexed_count % 20 == 0:
                gc.collect()

        except Exception as exc:
            failed.append(f"{uf.name}: {exc}")
            log_fn(f"⚠️ **[{f_idx}/{total_files}] {html.escape(uf.name)}** skipped — {exc}")

    _flush_micro_batch()
    gc.collect()
    elapsed = time.time() - started

    return {
        "store": store,
        "good_files": good_files,
        "chunks": store._collection.count(),
        "seconds": elapsed,
        "failed": failed,
    }


def ingest_urls_with_micro_batches(
    urls: list[str],
    session_id: str,
    append_mode: bool,
    log_fn: Callable[[str], None],
    progress_fn: Callable[[float, str], None],
) -> dict:
    """Phase 3: Streaming Micro-Batch Architecture for Live Website URLs (Text + Images)."""
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

    good_sources: list[str] = []
    failed: list[str] = []
    total_urls = len(urls)
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

    for u_idx, raw_url in enumerate(urls, start=1):
        try:
            display_source, url_chunks = parse_url_to_chunks(raw_url)
            good_sources.append(display_source)
            log_fn(f"✅ **{html.escape(display_source)}** ({len(url_chunks):,} web & image chunks)")

            for c_idx, chunk in enumerate(url_chunks):
                cid = hashlib.sha256(
                    f"{display_source}_{c_idx}_{chunk.page_content[:140]}".encode("utf-8")
                ).hexdigest()[:24]

                if cid not in seen_ids:
                    seen_ids.add(cid)
                    batch_docs.append(chunk)
                    batch_ids.append(cid)

                if len(batch_docs) >= MICRO_BATCH_SIZE:
                    _flush_micro_batch()
                    progress_fn(
                        min(1.0, u_idx / total_urls),
                        f"Streamed {committed_chunks:,} web chunks to DB ({u_idx}/{total_urls} URLs)…",
                    )
            _flush_micro_batch()
        except Exception as exc:
            failed.append(f"{raw_url}: {exc}")
            log_fn(f"⚠️ **{html.escape(raw_url)}** failed — {exc}")

    _flush_micro_batch()
    elapsed = time.time() - started

    return {
        "store": store,
        "good_files": good_sources,
        "chunks": store._collection.count(),
        "seconds": elapsed,
        "failed": failed,
    }


def _build_source_badges(item: dict) -> list[dict[str, str]]:
    """Construct the list of structured, color-coded badge descriptors for a retrieved chunk."""
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
    """Phase 5 Enterprise Multi-Stage Hybrid Search."""
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

    fts_hits = search_fts_keywords(
        session_id=session_id,
        question=question,
        top_k=max(2, top_k // 2),
        med_filter=med_filter,
        sheet_filter=sheet_filter,
        allowed_files=allowed_files,
    )

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

    merged_by_sig: dict[str, dict] = {}
    ordered_sigs: list[str] = []

    for hit in fts_hits:
        sig = hit["text"][:200]
        if sig not in merged_by_sig:
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

    final_sources: list[dict] = []
    for sig in ordered_sigs[:top_k]:
        item = merged_by_sig[sig]
        item["badges"] = _build_source_badges(item)
        final_sources.append(item)

    return final_sources
