"""Enterprise-Grade Multi-Session RAG Assistant using Groq, HuggingFace & ChromaDB."""

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
import uuid
from datetime import datetime

import chromadb
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# --------------------------------------------------------------------------- #
# Enterprise Settings & Constants
# --------------------------------------------------------------------------- #
MAX_FILES = 10
MAX_FILE_MB = 15
MAX_QUESTION_CHARS = 2000
DEFAULT_TOP_K = 4
HISTORY_TURNS = 4          # Previous Q&A pairs sent to the model for follow-ups
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASE_CHROMA_DIR = "./chroma_db"
METADATA_FILE = os.path.join(BASE_CHROMA_DIR, "workspace_state.json")

# Priority order of verified Groq production chat models
FALLBACK_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "llama3-70b-8192",
    "llama3-8b-8192",
]

SYSTEM_PROMPT = (
    "You are an enterprise document Q&A assistant. Answer the user's question using ONLY "
    "the provided context passages. Follow these strict rules:\n"
    "1. If the answer is not in the context, state clearly that you could not find it in the uploaded documents.\n"
    "2. Never guess or fabricate information.\n"
    "3. Cite the source file name (and page number when available) inline, e.g. [manual.pdf, page 2].\n"
    "4. Keep answers clear, well-structured, and concise."
)


# --------------------------------------------------------------------------- #
# Styling (Fixes Bug #11: Uses all defined classes cleanly)
# --------------------------------------------------------------------------- #
def apply_dashboard_theme() -> None:
    """Apply sleek dark slate & indigo dashboard styling."""
    st.markdown(
        """
        <style>
        :root {
            --accent: #6366f1;
            --accent-bright: #818cf8;
            --surface: #131620;
            --surface-2: #191d2b;
            --border: #21263d;
            --muted: #94a3b8;
        }
        .stApp { color: #f1f5f9; background: #090a0f; }
        [data-testid="stSidebar"] { background: #0d1017; border-right: 1px solid var(--border); }
        .block-container { max-width: 1180px; padding-top: 1.8rem; }

        /* Hero Header */
        .rag-hero {
            position: relative;
            overflow: hidden;
            padding: 1.75rem 2.25rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(19, 22, 32, 0.96), rgba(11, 14, 20, 0.99));
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
            margin-bottom: 1.5rem;
        }
        .rag-hero::before {
            content: "";
            position: absolute;
            width: 300px;
            height: 300px;
            right: -60px;
            top: -120px;
            border-radius: 50%;
            background: rgba(99, 102, 241, 0.13);
            filter: blur(40px);
            pointer-events: none;
        }
        .rag-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.55rem;
        }
        .rag-eyebrow {
            color: var(--accent-bright);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
        }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .dot-ready { background: #10b981; box-shadow: 0 0 8px #10b981; }
        .dot-empty { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }

        .rag-title {
            color: #ffffff;
            font-size: clamp(1.65rem, 2.8vw, 2.3rem);
            font-weight: 850;
            letter-spacing: -0.03em;
            line-height: 1.15;
            margin: 0 0 0.45rem 0;
        }
        .rag-title span { color: var(--accent-bright); font-family: monospace; }
        .rag-subtitle {
            color: var(--muted);
            font-size: 0.95rem;
            max-width: 780px;
            line-height: 1.5;
            margin: 0;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5, #6366f1);
            color: white;
            font-weight: 700;
            border: none;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #6366f1, #818cf8);
        }

        /* Live "working" indicator */
        .working-pill {
            display: inline-flex; align-items: center; gap: .6rem;
            padding: .45rem .9rem; border-radius: 999px;
            border: 1px solid var(--border); background: rgba(99, 102, 241, .08);
            color: var(--accent-bright); font-size: .9rem; font-weight: 600;
        }
        .working-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: var(--accent-bright);
            animation: rag-pulse 1.1s ease-in-out infinite;
        }
        .working-dots span { animation: rag-blink 1.4s infinite both; }
        .working-dots span:nth-child(2) { animation-delay: .2s; }
        .working-dots span:nth-child(3) { animation-delay: .4s; }
        @keyframes rag-pulse {
            0%   { transform: scale(.7); opacity: .5; box-shadow: 0 0 0 0 rgba(129,140,248,.6); }
            70%  { transform: scale(1);  opacity: 1;  box-shadow: 0 0 0 10px rgba(129,140,248,0); }
            100% { transform: scale(.7); opacity: .5; box-shadow: 0 0 0 0 rgba(129,140,248,0); }
        }
        @keyframes rag-blink { 0%, 80%, 100% { opacity: .15; } 40% { opacity: 1; } }

        .stat-card {
            border: 1px solid var(--border); border-radius: 12px;
            padding: .75rem .9rem; background: var(--surface); margin: .55rem 0;
            font-size: 0.88rem;
        }
        .stat-card b { color: var(--accent-bright); }
        .file-pill {
            display: inline-block;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.15rem 0.5rem;
            margin: 0.15rem 0.2rem 0.15rem 0;
            font-size: 0.78rem;
            color: #cbd5e1;
        }
        .source-card {
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.7rem 0.95rem;
            background: var(--surface);
            margin-bottom: 0.55rem;
        }
        .source-snippet { color: var(--muted); font-size: .84rem; white-space: pre-wrap; margin-top: 0.3rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def working_pill(text: str) -> str:
    """HTML for an animated 'something is happening' badge."""
    safe_text = html.escape(text)
    return (
        '<div class="working-pill"><div class="working-dot"></div>'
        f'<div>{safe_text}<span class="working-dots"><span>.</span><span>.</span><span>.</span></span></div>'
        "</div>"
    )


# --------------------------------------------------------------------------- #
# Dynamic Groq Model Discovery (Fixes 404 Model Not Found Error)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_active_groq_models() -> list[str]:
    """Query Groq's live API for active chat models available to the user's API key."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    env_model = os.getenv("GROQ_MODEL", "").strip()
    models: list[str] = []

    if env_model:
        models.append(env_model)

    if api_key:
        try:
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                skip_terms = ("whisper", "guard", "tts", "vision", "playai", "allam", "embed")
                live_ids = [
                    m["id"]
                    for m in data
                    if isinstance(m, dict)
                    and "id" in m
                    and not any(k in m["id"].lower() for k in skip_terms)
                ]
                for pref in FALLBACK_GROQ_MODELS:
                    if pref in live_ids and pref not in models:
                        models.append(pref)
                for mid in sorted(live_ids):
                    if mid not in models:
                        models.append(mid)
        except Exception:
            pass

    for fb in FALLBACK_GROQ_MODELS:
        if fb not in models:
            models.append(fb)

    return models


# --------------------------------------------------------------------------- #
# Disk Persistence & ChromaDB Management (Fixes Bugs #1, #2, #6)
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


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once per server process."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_chroma_client() -> chromadb.PersistentClient:
    """Shared PersistentClient prevents Windows SQLite file lock collisions."""
    os.makedirs(BASE_CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=BASE_CHROMA_DIR)


def list_collection_names(client: chromadb.PersistentClient) -> list[str]:
    """Compatible with ChromaDB 0.4, 0.5, 0.6, and 1.x."""
    try:
        cols = client.list_collections()
        return [c if isinstance(c, str) else c.name for c in cols]
    except Exception:
        return []


def discover_existing_sessions() -> list[str]:
    """Fixes Bug #2: Discover existing sessions in Chroma collections and legacy folders."""
    discovered: list[str] = []
    client = get_chroma_client()

    # 1. Check unified collections named 'session_<id>'
    for col_name in list_collection_names(client):
        if col_name.startswith("session_"):
            sid = col_name.replace("session_", "", 1)
            try:
                if client.get_collection(col_name).count() > 0 and sid not in discovered:
                    discovered.append(sid)
            except Exception:
                pass

    # 2. Check legacy per-session subfolders (./chroma_db/<session_id>)
    if os.path.exists(BASE_CHROMA_DIR):
        for item in os.listdir(BASE_CHROMA_DIR):
            sub_path = os.path.join(BASE_CHROMA_DIR, item)
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "chroma.sqlite3")):
                if item not in discovered:
                    discovered.append(item)

    return discovered


def get_session_vector_store(session_id: str) -> Chroma | None:
    """Load vector store for session_id if it has indexed vectors."""
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
    """Read unique filenames directly from ChromaDB chunk metadata."""
    if store is None:
        return []
    try:
        data = store._collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        return sorted({m.get("source", "unknown") for m in metadatas if isinstance(m, dict) and m.get("source")})
    except Exception:
        return []


def ensure_session_bucket(session_id: str) -> dict:
    """Fixes Bug #6: Maintain isolated messages, stats, and file lists per session."""
    store = get_session_vector_store(session_id)
    restored_files = extract_indexed_filenames(store)

    if session_id not in st.session_state.session_data:
        st.session_state.session_data[session_id] = {
            "messages": [],
            "indexed_files": restored_files,
            "index_stats": None,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    else:
        if restored_files:
            st.session_state.session_data[session_id]["indexed_files"] = restored_files
    return st.session_state.session_data[session_id]


def init_app_state() -> None:
    os.makedirs(BASE_CHROMA_DIR, exist_ok=True)
    disk_meta = load_disk_metadata()

    if "session_data" not in st.session_state:
        st.session_state.session_data = disk_meta.get("session_data", {})

    if "sessions" not in st.session_state:
        discovered = discover_existing_sessions()
        saved_sessions = disk_meta.get("sessions", [])
        merged: list[str] = []
        for s in discovered + saved_sessions:
            if s not in merged:
                merged.append(s)

        if merged:
            st.session_state.sessions = merged
            saved_curr = disk_meta.get("current_session_id")
            st.session_state.current_session_id = saved_curr if saved_curr in merged else merged[0]
        else:
            initial_id = str(uuid.uuid4())[:8]
            st.session_state.sessions = [initial_id]
            st.session_state.current_session_id = initial_id

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    active_models = fetch_active_groq_models()
    if "selected_model" not in st.session_state or st.session_state.selected_model not in active_models:
        st.session_state.selected_model = active_models[0]

    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K

    if "search_type" not in st.session_state:
        st.session_state.search_type = "MMR (Diverse)"

    for sid in st.session_state.sessions:
        ensure_session_bucket(sid)


def delete_session(session_id: str) -> None:
    """Fixes Bug #1 & #5: Properly wipe Chroma vectors, release locks, and reset file uploader."""
    client = get_chroma_client()
    collection_name = f"session_{session_id}"

    # 1. Delete collection inside ChromaDB
    try:
        if collection_name in list_collection_names(client):
            client.delete_collection(name=collection_name)
    except Exception:
        pass

    # 2. If legacy subfolder exists, delete its internal collection first, then rmtree
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

    # 3. Remove session state & metadata
    st.session_state.session_data.pop(session_id, None)
    if session_id in st.session_state.sessions:
        st.session_state.sessions.remove(session_id)

    # 4. Cycle to next available session or create a fresh one
    if not st.session_state.sessions:
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions = [new_id]
        st.session_state.current_session_id = new_id
    else:
        st.session_state.current_session_id = st.session_state.sessions[0]

    ensure_session_bucket(st.session_state.current_session_id)
    save_disk_metadata()
    st.session_state.uploader_key += 1
    st.toast(f"Session `{session_id}` and its vectors were permanently deleted!", icon="🗑️")
    st.rerun()


# --------------------------------------------------------------------------- #
# Document Loading & Indexing (Fixes Bug #4: Deterministic Deduplication)
# --------------------------------------------------------------------------- #
def uploaded_file_to_documents(uploaded_file) -> list[Document]:
    """Extract text from a Streamlit UploadedFile (one Document per PDF page)."""
    name = uploaded_file.name
    raw_data = uploaded_file.getvalue()

    if not raw_data:
        raise ValueError("file is empty.")
    if len(raw_data) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"file is larger than {MAX_FILE_MB} MB.")

    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    docs: list[Document] = []

    if suffix == "pdf":
        try:
            reader = PdfReader(io.BytesIO(raw_data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("PDF is password-protected.")
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                if page_text.strip():
                    docs.append(
                        Document(
                            page_content=page_text.strip(),
                            metadata={"source": name, "page": page_number},
                        )
                    )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"could not read PDF ({exc}).") from exc

        if not docs:
            raise ValueError("no readable text found (it may be a scanned/image-only PDF).")
    else:
        text = raw_data.decode("utf-8", errors="replace").strip()
        if not text:
            raise ValueError("no readable text found.")
        docs.append(Document(page_content=text, metadata={"source": name, "page": 1}))

    return docs


def process_documents(uploaded_files, session_id: str, append_mode: bool = True) -> None:
    """Read, chunk, deduplicate, and index files into Chroma for the current session."""
    started = time.time()
    failed: list[str] = []
    documents: list[Document] = []
    good_files: list[str] = []

    with st.status("⚙️ Processing documents…", expanded=True) as status:
        st.write(f"📄 Reading {len(uploaded_files)} file(s)…")
        for f in uploaded_files:
            try:
                file_docs = uploaded_file_to_documents(f)
                documents.extend(file_docs)
                good_files.append(f.name)
                st.write(f"✅ **{html.escape(f.name)}** ({len(file_docs)} page/section)")
            except Exception as exc:
                failed.append(f"{f.name}: {exc}")
                st.write(f"⚠️ **{html.escape(f.name)}** skipped — {exc}")

        if not documents:
            status.update(label="No readable documents found", state="error", expanded=True)
            st.error("None of the uploaded files could be read.")
            return

        st.write("✂️ Splitting text into chunks…")
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(documents)
        if not chunks:
            status.update(label="Nothing to index", state="error", expanded=True)
            st.error("The documents did not produce any text chunks.")
            return

        # Fixes Bug #4: Deterministic SHA-256 IDs prevent duplicate chunks on re-upload
        chunk_ids = [
            hashlib.sha256(
                f"{c.metadata.get('source','')}_{c.metadata.get('page',0)}_{idx}_{c.page_content[:120]}".encode("utf-8")
            ).hexdigest()[:24]
            for idx, c in enumerate(chunks)
        ]

        st.write("🧠 Loading embedding model…")
        try:
            embeddings = get_embeddings()
        except Exception as exc:
            status.update(label="Could not load embedding model", state="error", expanded=True)
            st.error(f"Embedding model failed to load: {exc}")
            return

        client = get_chroma_client()
        collection_name = f"session_{session_id}"

        if not append_mode:
            try:
                if collection_name in list_collection_names(client):
                    client.delete_collection(name=collection_name)
            except Exception:
                pass

        st.write(f"🔢 Saving & indexing {len(chunks)} chunks into session Chroma DB…")
        try:
            store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                ids=chunk_ids,
                client=client,
                collection_name=collection_name,
            )
        except Exception as exc:
            status.update(label="Chroma embedding failed", state="error", expanded=True)
            st.error(f"Chroma indexing failed: {exc}")
            return

        elapsed = time.time() - started
        status.update(label=f"Ready — indexed in {elapsed:.1f}s", state="complete", expanded=False)

    bucket = ensure_session_bucket(session_id)
    bucket["indexed_files"] = extract_indexed_filenames(store) or good_files
    bucket["index_stats"] = {
        "files": len(bucket["indexed_files"]),
        "chunks": store._collection.count(),
        "seconds": elapsed,
        "failed": failed,
    }
    save_disk_metadata()
    st.session_state.uploader_key += 1
    st.toast(f"Indexed {len(good_files)} file(s) into Session #{session_id}!", icon="✅")
    st.rerun()


# --------------------------------------------------------------------------- #
# Retrieval, Prompting & Error Handling (Fixes Bugs #3, #8, #9, #10)
# --------------------------------------------------------------------------- #
def friendly_error(exc: Exception) -> str:
    """Fixes Bug #10: Accurate status-code regex matching + 404 handler."""
    raw = str(exc)
    text = raw.lower()
    if re.search(r"\b429\b", text) or "rate limit" in text or "rate_limit" in text:
        return "Groq's rate limit was hit. Wait a few seconds and try again."
    if re.search(r"\b401\b", text) or "invalid api key" in text or "authentication" in text:
        return "Groq rejected the API key. Check `GROQ_API_KEY` in your `.env` file."
    if re.search(r"\b404\b", text) or "model_not_found" in text or "decommissioned" in text:
        return "The selected Groq model is unavailable for your API key. Switch models in Sidebar Settings."
    if "timeout" in text or "timed out" in text:
        return "The request timed out. Please try again."
    if "connection" in text or "network" in text:
        return "Couldn't reach Groq. Check your internet connection and try again."
    if "context" in text and ("length" in text or "too long" in text):
        return "The question plus context was too long for the model. Try a shorter question."
    return f"Something went wrong: {raw}"


def build_messages(prior_messages: list, question: str, context: str) -> list:
    """Fixes Bug #3: Only iterates over prior_messages so current question is never duplicated."""
    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
    history = prior_messages[-(HISTORY_TURNS * 2) :]
    for m in history:
        if m.get("error"):
            continue
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"))
    return messages


def retrieve(store: Chroma, question: str) -> list[dict]:
    """Fixes Bug #8: Converts L2 distance to 0-100% similarity match and supports MMR."""
    top_k = st.session_state.top_k
    use_mmr = "MMR" in st.session_state.get("search_type", "MMR")

    try:
        if use_mmr:
            docs = store.max_marginal_relevance_search(question, k=top_k, fetch_k=max(10, top_k * 3))
            results = [(d, None) for d in docs]
        else:
            results = store.similarity_search_with_score(question, k=top_k)
    except Exception:
        results = [(d, None) for d in store.similarity_search(question, k=top_k)]

    sources = []
    for doc, distance in results:
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
    return sources


def render_sources(sources: list[dict]) -> None:
    """Fixes Bug #9: Escapes HTML in filenames and snippets to prevent DOM/XSS injection."""
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for idx, s in enumerate(sources, start=1):
            safe_source = html.escape(str(s.get("source", "unknown")))
            page_str = f", page {int(s['page'])}" if isinstance(s.get("page"), int) else ""
            match_str = (
                f' · <span style="color:#818cf8">{s["match_pct"]}% similarity</span>'
                if isinstance(s.get("match_pct"), int)
                else ""
            )
            raw_text = str(s.get("text", ""))
            snippet = html.escape(raw_text[:420] + ("…" if len(raw_text) > 420 else ""))
            st.markdown(
                f"""
                <div class="source-card">
                    <div><b>{idx}. 📄 {safe_source}{page_str}</b>{match_str}</div>
                    <div class="source-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def stream_with_auto_fallback(prompt_msgs: list, placeholder) -> tuple[str, str]:
    """Stream response from Groq; automatically fallback to another active model on 404."""
    candidates = [st.session_state.selected_model] + [
        m for m in fetch_active_groq_models() if m != st.session_state.selected_model
    ]
    last_exc: Exception | None = None

    for model_id in candidates:
        try:
            llm = ChatGroq(model=model_id, temperature=0.1, max_retries=2)
            answer = ""
            for chunk in llm.stream(prompt_msgs):
                piece = chunk.content if isinstance(chunk.content, str) else ""
                if piece:
                    answer += piece
                    placeholder.markdown(answer + " ▌")
            if answer.strip():
                if model_id != st.session_state.selected_model:
                    st.session_state.selected_model = model_id
                    st.toast(f"Auto-switched to available model `{model_id}`", icon="🔄")
                return answer, model_id
        except Exception as exc:
            last_exc = exc
            err_lower = str(exc).lower()
            if "404" in err_lower or "model_not_found" in err_lower or "decommissioned" in err_lower:
                continue
            break

    if last_exc:
        raise last_exc
    raise RuntimeError("No Groq models responded.")


def answer_question(question: str, session_id: str, store: Chroma) -> None:
    """Check for deletion triggers, or retrieve and stream the answer."""
    clean_q = question.lower().strip()
    if clean_q in ["delete session", "delete this session", "clear session", "/delete"]:
        delete_session(session_id)
        return

    bucket = ensure_session_bucket(session_id)
    prior_messages = list(bucket["messages"])  # Snapshot BEFORE appending current user question
    bucket["messages"].append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    started = time.time()
    sources: list[dict] = []
    answer = ""
    used_model = st.session_state.selected_model
    errored = False

    with st.chat_message("assistant"):
        with st.status("🔎 Searching your documents…", expanded=False) as status:
            try:
                # Contextualize short follow-up questions with the previous user question
                search_query = question
                if len(question.split()) <= 6 and prior_messages:
                    prev_user_qs = [m["content"] for m in prior_messages if m["role"] == "user"]
                    if prev_user_qs:
                        search_query = f"{prev_user_qs[-1]} {question}"
                sources = retrieve(store, search_query)
                status.update(label=f"Found {len(sources)} relevant passages", state="complete")
            except Exception as exc:
                status.update(label="Search failed", state="error")
                answer = friendly_error(exc)
                errored = True

        placeholder = st.empty()

        if not errored:
            if not sources:
                answer = "I couldn't find anything relevant in the uploaded documents."
                placeholder.markdown(answer)
            else:
                placeholder.markdown(working_pill("Thinking"), unsafe_allow_html=True)
                context = "\n\n".join(
                    f"Source ({s['source']}{', page ' + str(s['page']) if s.get('page') else ''}):\n{s['text']}"
                    for s in sources
                )
                try:
                    prompt_msgs = build_messages(prior_messages, question, context)
                    answer, used_model = stream_with_auto_fallback(prompt_msgs, placeholder)
                    placeholder.markdown(answer)
                except Exception as exc:
                    errored = True
                    partial = f"{answer}\n\n" if answer.strip() else ""
                    answer = f"{partial}⚠️ {friendly_error(exc)}"
                    placeholder.markdown(answer)
        else:
            placeholder.markdown(answer)

        elapsed = time.time() - started
        if not errored:
            st.caption(f"Answered in {elapsed:.1f}s · Model: `{used_model}`")
            render_sources(sources)

    bucket["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [] if errored else sources,
            "error": errored,
            "elapsed": elapsed,
        }
    )
    save_disk_metadata()


# --------------------------------------------------------------------------- #
# Sidebar & Main UI (Fixes Bugs #5, #6, #7)
# --------------------------------------------------------------------------- #
def format_chat_export(session_id: str, messages: list) -> str:
    lines = [f"# RAG Chat Transcript — Session `{session_id}`\n"]
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"### {role}\n{m['content']}\n")
    return "\n".join(lines)


def render_sidebar(session_id: str, store: Chroma | None) -> None:
    bucket = ensure_session_bucket(session_id)

    with st.sidebar:
        st.markdown("### ◈ Sessions Manager")

        selected_session = st.selectbox(
            "Active Session ID",
            options=st.session_state.sessions,
            index=st.session_state.sessions.index(session_id),
            format_func=lambda x: f"📂 #{x} ({len(ensure_session_bucket(x)['indexed_files'])} docs)",
        )

        if selected_session != session_id:
            st.session_state.current_session_id = selected_session
            save_disk_metadata()
            st.session_state.uploader_key += 1
            st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("➕ New", use_container_width=True, help="Create a new isolated session"):
                new_id = str(uuid.uuid4())[:8]
                st.session_state.sessions.append(new_id)
                st.session_state.current_session_id = new_id
                ensure_session_bucket(new_id)
                save_disk_metadata()
                st.session_state.uploader_key += 1
                st.rerun()
        with col_b:
            if st.button("🗑️ Delete", type="secondary", use_container_width=True, help="Delete current session data"):
                delete_session(session_id)

        if store is not None:
            chunk_count = store._collection.count()
            indexed_files = bucket.get("indexed_files") or extract_indexed_filenames(store)
            bucket["indexed_files"] = indexed_files
            pills = "".join(f'<span class="file-pill">📄 {html.escape(f)}</span>' for f in indexed_files)
            stats = bucket.get("index_stats")
            timing_html = (
                f'<br><span style="color:#94a3b8">indexed in {stats["seconds"]:.1f}s</span>'
                if stats and stats.get("seconds")
                else ""
            )
            st.markdown(
                f'<div class="stat-card">✅ <b>{len(indexed_files)}</b> file(s) · '
                f'<b>{chunk_count}</b> chunks{timing_html}'
                f'<div style="margin-top:0.4rem">{pills}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### ◈ Knowledge Base")

        api_key_ok = bool(os.getenv("GROQ_API_KEY"))
        if not api_key_ok:
            st.error("`GROQ_API_KEY` not found in `.env` file.")

        # Fixes Bug #5: Dynamic key clears old files when switching or deleting sessions
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key=f"uploader_{session_id}_{st.session_state.uploader_key}",
            help=f"Up to {MAX_FILES} files, {MAX_FILE_MB} MB each.",
        )

        too_many = bool(uploaded_files) and len(uploaded_files) > MAX_FILES
        if too_many:
            st.warning(f"Please upload at most {MAX_FILES} files.")

        append_mode = st.checkbox("Keep existing session documents", value=True) if store is not None else False

        if st.button(
            "Process documents",
            type="primary",
            disabled=not uploaded_files or too_many or not api_key_ok,
            use_container_width=True,
        ):
            process_documents(uploaded_files, session_id, append_mode=append_mode)

        st.markdown("---")
        with st.expander("⚙️ Model & Search Settings", expanded=False):
            active_models = fetch_active_groq_models()
            curr_idx = (
                active_models.index(st.session_state.selected_model)
                if st.session_state.selected_model in active_models
                else 0
            )
            st.session_state.selected_model = st.selectbox(
                "Groq Model",
                options=active_models,
                index=curr_idx,
            )
            st.session_state.search_type = st.radio(
                "Retrieval Mode",
                options=["MMR (Diverse)", "Similarity (Distance)"],
                index=0,
                horizontal=True,
            )
            st.session_state.top_k = st.slider(
                "Passages to Retrieve (Top-K)",
                min_value=2,
                max_value=10,
                value=st.session_state.top_k,
            )

        if bucket["messages"]:
            col_c, col_d = st.columns(2)
            with col_c:
                if st.button("🧹 Clear Chat", use_container_width=True):
                    bucket["messages"] = []
                    save_disk_metadata()
                    st.rerun()
            with col_d:
                st.download_button(
                    "📥 Export MD",
                    data=format_chat_export(session_id, bucket["messages"]),
                    file_name=f"session_{session_id}_chat.md",
                    mime="text/markdown",
                    use_container_width=True,
                )


def main() -> None:
    st.set_page_config(page_title="Multi-Session Document Q&A", page_icon="📚", layout="wide")
    apply_dashboard_theme()
    init_app_state()

    # Fixes Bug #7: Render sidebar BEFORE hero header so state changes reflect immediately
    session_id = st.session_state.current_session_id
    store = get_session_vector_store(session_id)
    render_sidebar(session_id, store)

    # Re-fetch after sidebar in case session changed
    session_id = st.session_state.current_session_id
    bucket = ensure_session_bucket(session_id)
    store = get_session_vector_store(session_id)
    ready = store is not None and bool(os.getenv("GROQ_API_KEY"))

    dot_class = "dot-ready" if ready else "dot-empty"
    status_text = f"Ready · {store._collection.count()} Indexed Chunks" if ready else "Awaiting Documents"

    # Fixes Bug #11: Uses .rag-topbar, .rag-eyebrow, .rag-title, .rag-subtitle cleanly
    st.markdown(
        f"""
        <section class="rag-hero">
            <div class="rag-topbar">
                <div class="rag-eyebrow">
                    <span class="status-dot {dot_class}"></span>
                    Groq · Multi-Session Chroma RAG · {status_text}
                </div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Model: <code>{html.escape(st.session_state.selected_model)}</code></div>
            </div>
            <h1 class="rag-title">Active Session: <span>#{html.escape(session_id)}</span></h1>
            <p class="rag-subtitle">
                Data is strictly isolated and persisted per session. Type <code>delete session</code> in chat or click the sidebar delete button to wipe it.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    queued_prompt = None
    if not ready:
        st.info("👈 Upload and process documents in the sidebar for this session to begin.")
    elif not bucket["messages"]:
        st.markdown("**Try asking:**")
        q_cols = st.columns(4)
        starters = [
            "Summarize the main points",
            "List key facts, figures, and dates",
            "What are the main conclusions?",
            "Create a quick FAQ from this document",
        ]
        for idx, prompt_text in enumerate(starters):
            if q_cols[idx].button(prompt_text, use_container_width=True):
                queued_prompt = prompt_text

    # Replay conversation history for this session
    for message in bucket["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and not message.get("error"):
                if message.get("elapsed") is not None:
                    st.caption(f"Answered in {message['elapsed']:.1f}s")
                render_sources(message.get("sources", []))

    question = st.chat_input(
        "Ask a question or type 'delete session'..." if ready else "Process documents first…",
        disabled=not ready,
    )

    active_question = queued_prompt or question
    if active_question:
        active_question = active_question.strip()
        if not active_question:
            st.warning("Please type a question.")
            return
        if len(active_question) > MAX_QUESTION_CHARS:
            st.warning(f"Please keep your question under {MAX_QUESTION_CHARS} characters.")
            return

        answer_question(active_question, session_id, store)
        if queued_prompt:
            st.rerun()


if __name__ == "__main__":
    main()
