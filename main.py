"""Streamlit interface for asking questions about uploaded documents using Groq & HuggingFace."""

from __future__ import annotations

import io
import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
MAX_FILES = 10
MAX_FILE_MB = 15
MAX_QUESTION_CHARS = 2000
TOP_K = 4
HISTORY_TURNS = 4          # previous Q&A pairs sent to the model for follow-ups
EMBED_BATCH_SIZE = 64
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer the user's question using ONLY the "
    "provided context passages. If the answer is not in the context, say you could "
    "not find it in the uploaded documents instead of guessing. Be accurate and "
    "concise, and mention the source file name when it helps."
)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def apply_dashboard_theme() -> None:
    """Apply a modern sleek dark slate & indigo dashboard styling with an improved hero header."""
    st.markdown(
        """
        <style>
        :root {
            --accent: #6366f1;
            --accent-bright: #818cf8;
            --surface: #131620;
            --border: #21263d;
            --muted: #94a3b8;
        }
        .stApp { color: #f1f5f9; background: #090a0f; }
        [data-testid="stSidebar"] { background: #0d1017; border-right: 1px solid var(--border); }
        .block-container { max-width: 1180px; padding-top: 2rem; }
        
        /* Enhanced Modern Hero Header */
        .rag-hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.5rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(19, 22, 32, 0.95), rgba(11, 14, 20, 0.98));
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
            margin-bottom: 2rem;
        }
        .rag-hero::before {
            content: "";
            position: absolute;
            width: 300px;
            height: 300px;
            right: -60px;
            top: -120px;
            border-radius: 50%;
            background: rgba(99, 102, 241, 0.12);
            filter: blur(40px);
            pointer-events: none;
        }
        .rag-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: .75rem;
        }
        .rag-eyebrow {
            color: var(--accent-bright);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .rag-title {
            color: #ffffff;
            font-size: clamp(2rem, 3.5vw, 2.8rem);
            font-weight: 850;
            letter-spacing: -.03em;
            line-height: 1.15;
            margin: 0 0 0.6rem 0;
        }
        .rag-title span { color: var(--accent-bright); }
        .rag-subtitle {
            color: var(--muted);
            font-size: 1rem;
            max-width: 720px;
            line-height: 1.55;
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
            padding: .6rem .8rem; background: var(--surface); margin-bottom: .5rem;
        }
        .stat-card b { color: var(--accent-bright); }
        .source-snippet { color: var(--muted); font-size: .85rem; white-space: pre-wrap; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def working_pill(text: str) -> str:
    """HTML for an animated 'something is happening' badge."""
    return (
        '<div class="working-pill"><div class="working-dot"></div>'
        f'<div>{text}<span class="working-dots"><span>.</span><span>.</span><span>.</span></span></div>'
        "</div>"
    )


# --------------------------------------------------------------------------- #
# Document loading
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
            if reader.is_encrypted:
                # Many PDFs are "encrypted" with an empty password.
                if not reader.decrypt(""):
                    raise ValueError("PDF is password-protected.")
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""  # skip a broken page, keep the rest
                if page_text.strip():
                    docs.append(
                        Document(
                            page_content=page_text,
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
        text = raw_data.decode("utf-8", errors="replace")
        if not text.strip():
            raise ValueError("no readable text found.")
        docs.append(Document(page_content=text, metadata={"source": name}))

    return docs


def files_signature(uploaded_files) -> tuple:
    """Cheap fingerprint of the current upload set (names + sizes)."""
    return tuple(sorted((f.name, f.size) for f in uploaded_files))


# --------------------------------------------------------------------------- #
# RAG pipeline
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once per server process (first load downloads it)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_llm() -> ChatGroq:
    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return ChatGroq(model=model_name, temperature=0.1, max_retries=2)


def process_documents(uploaded_files) -> None:
    """Read, chunk, embed and index files while showing live progress."""
    started = time.time()
    failed: list[str] = []
    documents: list[Document] = []
    good_files = 0

    with st.status("Processing documents…", expanded=True) as status:
        # Step 1: read files
        st.write(f"📄 Reading {len(uploaded_files)} file(s)…")
        for f in uploaded_files:
            try:
                file_docs = uploaded_file_to_documents(f)
                documents.extend(file_docs)
                good_files += 1
                st.write(f"✅ {f.name}")
            except Exception as exc:
                failed.append(f"{f.name}: {exc}")
                st.write(f"⚠️ {f.name} skipped — {exc}")

        if not documents:
            status.update(label="No readable documents found", state="error", expanded=True)
            st.error("None of the uploaded files could be read.")
            return

        # Step 2: split
        st.write("✂️ Splitting text into chunks…")
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(documents)
        if not chunks:
            status.update(label="Nothing to index", state="error", expanded=True)
            st.error("The documents did not produce any text chunks.")
            return

        # Step 3: load embedding model
        st.write("🧠 Loading embedding model (first run downloads it, this can take a minute)…")
        try:
            embeddings = get_embeddings()
        except Exception as exc:
            status.update(label="Could not load embedding model", state="error", expanded=True)
            st.error(f"Embedding model failed to load: {exc}")
            return

        # Step 4: embed in batches with a real progress bar
        st.write(f"🔢 Embedding {len(chunks)} chunks…")
        progress = st.progress(0.0, text="Embedding 0%")
        vector_store = InMemoryVectorStore(embeddings)
        try:
            for i in range(0, len(chunks), EMBED_BATCH_SIZE):
                vector_store.add_documents(chunks[i : i + EMBED_BATCH_SIZE])
                done = min(i + EMBED_BATCH_SIZE, len(chunks))
                progress.progress(done / len(chunks), text=f"Embedding {done}/{len(chunks)} chunks")
        except Exception as exc:
            status.update(label="Embedding failed", state="error", expanded=True)
            st.error(f"Embedding failed: {exc}")
            return

        # Step 5: LLM
        st.write("🤖 Connecting to Groq…")
        try:
            llm = build_llm()
        except Exception as exc:
            status.update(label="Could not initialise Groq", state="error", expanded=True)
            st.error(f"Groq setup failed: {exc}")
            return

        elapsed = time.time() - started
        status.update(label=f"Ready — indexed in {elapsed:.1f}s", state="complete", expanded=False)

    # Commit to session only after everything succeeded
    st.session_state.vector_store = vector_store
    st.session_state.llm = llm
    st.session_state.indexed_signature = files_signature(uploaded_files)
    st.session_state.index_stats = {
        "files": good_files,
        "chunks": len(chunks),
        "seconds": elapsed,
        "failed": failed,
    }
    st.session_state.messages = []

    if failed:
        st.warning("Some files were skipped:\n\n" + "\n".join(f"- {item}" for item in failed))


def friendly_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "429" in text or "rate limit" in text or "rate_limit" in text:
        return "Groq's rate limit was hit. Wait a few seconds and try again."
    if "401" in text or "invalid api key" in text or "authentication" in text:
        return "Groq rejected the API key. Check `GROQ_API_KEY` in your `.env` file."
    if "timeout" in text or "timed out" in text:
        return "The request timed out. Please try again."
    if "connection" in text or "network" in text:
        return "Couldn't reach Groq. Check your internet connection and try again."
    if "context" in text and ("length" in text or "too long" in text):
        return "The question plus context was too long for the model. Try a shorter question."
    return f"Something went wrong: {exc}"


def build_messages(question: str, context: str) -> list:
    """System prompt + recent chat history + the new question with retrieved context."""
    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
    history = st.session_state.messages[-(HISTORY_TURNS * 2):]
    for m in history:
        if m.get("error"):
            continue
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"))
    return messages


def retrieve(question: str) -> list[dict]:
    store: InMemoryVectorStore = st.session_state.vector_store
    try:
        results = store.similarity_search_with_score(question, k=TOP_K)
    except Exception:
        results = [(d, None) for d in store.similarity_search(question, k=TOP_K)]

    sources = []
    for doc, score in results:
        sources.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "score": score,
                "text": doc.page_content,
            }
        )
    return sources


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for idx, s in enumerate(sources, start=1):
            page = f", page {s['page']}" if s.get("page") else ""
            score = f" · relevance {s['score']:.2f}" if isinstance(s.get("score"), (int, float)) else ""
            st.markdown(f"**{idx}. {s['source']}{page}**{score}")
            snippet = s["text"][:400] + ("…" if len(s["text"]) > 400 else "")
            st.markdown(f'<div class="source-snippet">{snippet}</div>', unsafe_allow_html=True)


def answer_question(question: str) -> None:
    """Retrieve, stream the answer, and show live status the whole way."""
    started = time.time()
    sources: list[dict] = []
    answer = ""
    errored = False

    with st.chat_message("assistant"):
        # Phase 1: retrieval
        with st.status("🔎 Searching your documents…", expanded=False) as status:
            try:
                sources = retrieve(question)
                status.update(label=f"Found {len(sources)} relevant passages", state="complete")
            except Exception as exc:
                status.update(label="Search failed", state="error")
                answer = friendly_error(exc)
                errored = True

        placeholder = st.empty()

        # Phase 2: generation (streamed)
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
                    for chunk in st.session_state.llm.stream(build_messages(question, context)):
                        piece = chunk.content if isinstance(chunk.content, str) else ""
                        if not piece:
                            continue
                        answer += piece
                        placeholder.markdown(answer + " ▌")
                    if not answer.strip():
                        answer = "The model returned an empty response. Please try rephrasing your question."
                        errored = True
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
            st.caption(f"Answered in {elapsed:.1f}s")
            render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [] if errored else sources,
            "error": errored,
            "elapsed": elapsed,
        }
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ◈ Knowledge base")

        api_key_ok = bool(os.getenv("GROQ_API_KEY"))
        if not api_key_ok:
            st.error("`GROQ_API_KEY` not found. Add it to your `.env` file and restart the app.")

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help=f"Up to {MAX_FILES} files, {MAX_FILE_MB} MB each.",
        )

        too_many = bool(uploaded_files) and len(uploaded_files) > MAX_FILES
        if too_many:
            st.warning(f"Please upload at most {MAX_FILES} files (you selected {len(uploaded_files)}).")

        # Warn if the upload set changed since the last indexing
        if uploaded_files and "vector_store" in st.session_state:
            if files_signature(uploaded_files) != st.session_state.get("indexed_signature"):
                st.info("Files changed — click **Process documents** to update the knowledge base.")

        process_clicked = st.button(
            "Process documents",
            type="primary",
            disabled=not uploaded_files or too_many or not api_key_ok,
            use_container_width=True,
        )

        if process_clicked:
            process_documents(uploaded_files)

        stats = st.session_state.get("index_stats")
        if stats and "vector_store" in st.session_state:
            st.markdown(
                f'<div class="stat-card">✅ <b>{stats["files"]}</b> file(s) · '
                f'<b>{stats["chunks"]}</b> chunks<br>'
                f'<span style="color:#9aaba5">indexed in {stats["seconds"]:.1f}s</span></div>',
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear chat", use_container_width=True, disabled=not st.session_state.messages):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Reset all", use_container_width=True, disabled="vector_store" not in st.session_state):
                for key in ("vector_store", "llm", "indexed_signature", "index_stats", "messages"):
                    st.session_state.pop(key, None)
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")
    apply_dashboard_theme()

    st.markdown(
        """
        <section class="rag-hero">
            <div style="color: #4ade80; font-size: 0.8rem; font-weight: bold; text-transform: uppercase;">Groq · Free Local RAG</div>
            <h1 class="rag-title">Chat with your <span>knowledge</span></h1>
            <p style="color: #9aaba5; margin: 0;">Upload your documents and get grounded answers instantly using local embeddings and Groq.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_sidebar()

    ready = "vector_store" in st.session_state and "llm" in st.session_state

    if not ready:
        st.info("👈 Upload and process at least one document in the sidebar to begin.")
    elif not st.session_state.messages:
        st.markdown("**Try asking:** *“Summarize the main points”* · *“What does the document say about …?”*")

    # Replay history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and not message.get("error"):
                if message.get("elapsed") is not None:
                    st.caption(f"Answered in {message['elapsed']:.1f}s")
                render_sources(message.get("sources", []))

    question = st.chat_input(
        "Ask a question about your documents" if ready else "Process documents first…",
        disabled=not ready,
    )

    if question:
        question = question.strip()
        if not question:
            st.warning("Please type a question.")
            return
        if len(question) > MAX_QUESTION_CHARS:
            st.warning(f"Please keep your question under {MAX_QUESTION_CHARS} characters.")
            return

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        answer_question(question)


if __name__ == "__main__":
    main()