"""
Main Application Entry Point.
Connects the separated Frontend UI (`frontend/`) with the Enterprise RAG Engine (`engine/`).
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from engine.config import BASE_CHROMA_DIR, DEFAULT_TOP_K, MAX_QUESTION_CHARS
from engine.hybrid_index import init_hybrid_db
from engine.llm_service import fetch_active_groq_models
from engine.storage import (
    discover_existing_sessions,
    get_session_vector_store,
    load_disk_metadata,
)
from frontend.components import (
    ensure_session_bucket,
    handle_user_question,
    render_hero_header,
    render_sidebar,
    render_sources,
)
from frontend.styles import apply_dashboard_theme

load_dotenv()


def init_app_state() -> None:
    """Initialize multi-session state and auto-discover existing sessions on disk."""
    os.makedirs(BASE_CHROMA_DIR, exist_ok=True)
    init_hybrid_db()
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
        st.session_state.search_type = "Hybrid + MMR (Diverse)"

    for sid in st.session_state.sessions:
        ensure_session_bucket(sid)


def main() -> None:
    st.set_page_config(page_title="Multi-Session Enterprise RAG", page_icon="📚", layout="wide")
    apply_dashboard_theme()
    init_app_state()

    # 1. Render Sidebar First
    session_id = st.session_state.current_session_id
    store = get_session_vector_store(session_id)
    render_sidebar(session_id, store)

    # 2. Re-fetch state in case session changed in sidebar
    session_id = st.session_state.current_session_id
    bucket = ensure_session_bucket(session_id)
    store = get_session_vector_store(session_id)
    ready = store is not None and bool(os.getenv("GROQ_API_KEY"))

    # 3. Render Hero Banner
    render_hero_header(session_id, ready, store)

    # 4. Starter Prompt Buttons or Empty Info
    queued_prompt = None
    if not ready:
        st.info("👈 Upload Excel (.xlsx, .xls, .csv), PDF, TXT, or MD documents in the sidebar to begin.")
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

    # 5. Replay Chat History
    for message in bucket["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and not message.get("error"):
                if message.get("elapsed") is not None:
                    st.caption(f"Answered in {message['elapsed']:.1f}s")
                render_sources(message.get("sources", []))

    # 6. Chat Input
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

        handle_user_question(active_question, session_id, store)
        if queued_prompt:
            st.rerun()


if __name__ == "__main__":
    main()
