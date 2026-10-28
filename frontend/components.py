"""
Separated Frontend UI Components Module.
Renders Sidebar, Hero Banner, Source Citations, Suggested Follow-Ups, and Interactive Chat Interface.

Key Architectural Features:
---------------------------
1. Zero-Disk-I/O Session Label Resolution (`_peek_bucket`):
   Computes smart 3-tier session labels (Custom Name -> Auto-Numbered Filename -> #UUID)
   purely from in-memory `st.session_state` to prevent ChromaDB lock contention.
2. Per-File Scope Filtering & Granular Deletion:
   Allows users to restrict RAG retrieval to specific documents inside a session or remove
   individual files without wiping the entire session.
3. Enterprise Color-Coded Hybrid Retrieval Badges:
   Displays a Hybrid Retrieval Breakdown bar and color-coded pills (Categorical, FTS5, Vector).
4. Smart Conversational Query Rewriter:
   Replaces naive word-count concatenation with an LLM-powered coreference & entity resolver
   for multi-turn follow-up questions.
5. Grounded Auto-Generated Follow-Up Suggestions (`render_followup_pills`):
   Generates 3 clickable follow-up question buttons below the latest answer grounded in the
   retrieved source passages.
"""

from __future__ import annotations

import html
import os
import re
import time
import uuid
from datetime import datetime

import streamlit as st
from langchain_chroma import Chroma

from engine.config import MAX_FILE_MB, MAX_FILES
from engine.hybrid_index import get_session_inventory
from engine.llm_service import (
    build_messages,
    fetch_active_groq_models,
    friendly_error,
    generate_followup_questions,
    rewrite_conversational_query,
    stream_with_auto_fallback,
)
from engine.storage import (
    delete_single_file_from_session,
    extract_indexed_filenames,
    get_session_vector_store,
    hybrid_retrieve,
    ingest_files_with_micro_batches,
    save_disk_metadata,
    wipe_session_data,
)
from frontend.styles import working_pill


def ensure_session_bucket(session_id: str) -> dict:
    """
    Ensure an isolated state dictionary exists for `session_id`.
    Automatically backfills missing keys (`name`, `active_files`) for backward compatibility
    with older sessions loaded from disk.
    """
    store = get_session_vector_store(session_id)
    restored_files = extract_indexed_filenames(store, session_id)

    if session_id not in st.session_state.session_data:
        st.session_state.session_data[session_id] = {
            "name": "",
            "messages": [],
            "indexed_files": restored_files,
            "active_files": [],
            "index_stats": None,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    else:
        bucket = st.session_state.session_data[session_id]
        if "name" not in bucket:
            bucket["name"] = ""
        if "active_files" not in bucket:
            bucket["active_files"] = []
        if restored_files:
            bucket["indexed_files"] = restored_files
            bucket["active_files"] = [f for f in bucket["active_files"] if f in restored_files]
    return st.session_state.session_data[session_id]


def _peek_bucket(session_id: str) -> dict:
    """Fast in-memory lookup without triggering ChromaDB disk I/O."""
    session_data = st.session_state.get("session_data", {})
    if session_id in session_data and isinstance(session_data[session_id], dict):
        return session_data[session_id]
    return {"name": "", "messages": [], "indexed_files": [], "active_files": []}


def get_session_base_label(session_id: str) -> str:
    """
    Fast in-memory label calculation:
    1. Custom name if set by user
    2. First indexed filename (with (2), (3) if another unnamed session has the same first file)
    3. Fallback to Session #{session_id}
    """
    bucket = _peek_bucket(session_id)
    custom_name = str(bucket.get("name") or "").strip()
    if custom_name:
        return custom_name

    files = bucket.get("indexed_files") or []
    if files:
        first_file = str(files[0])
        short_file = first_file[:20] + "…" if len(first_file) > 20 else first_file

        same_file_sids = []
        for sid in st.session_state.get("sessions", []):
            b = _peek_bucket(sid)
            b_name = str(b.get("name") or "").strip()
            b_files = b.get("indexed_files") or []
            if not b_name and b_files and str(b_files[0]) == first_file:
                same_file_sids.append(sid)

        occurrence = same_file_sids.index(session_id) + 1 if session_id in same_file_sids else 1
        suffix = f" ({occurrence})" if occurrence > 1 else ""
        return f"{short_file}{suffix}"

    return f"Session #{session_id}"


def get_session_display_name(session_id: str) -> str:
    """Return display label with #session_id."""
    bucket = _peek_bucket(session_id)
    custom_name = str(bucket.get("name") or "").strip()
    files = bucket.get("indexed_files") or []
    if not custom_name and not files:
        return f"#{session_id}"
    return f"{get_session_base_label(session_id)} (#{session_id})"


def _is_not_found_answer(answer: str) -> bool:
    """
    Detect if the LLM explicitly stated that the answer was not found in the documents.
    Used to suppress misleading low-similarity source citations on out-of-scope queries.
    """
    lower = answer.lower()
    not_found_patterns = (
        r"could not find",
        r"couldn't find",
        r"not mentioned in the (uploaded |provided )?documents",
        r"no information about",
        r"does not contain (any )?information",
        r"not present in the (uploaded |provided )?context",
    )
    return any(re.search(pat, lower) for pat in not_found_patterns)


def delete_session_action(session_id: str) -> None:
    """Wipe session data from Chroma + Hybrid DB and transition to a clean session."""
    wipe_session_data(session_id)

    st.session_state.session_data.pop(session_id, None)
    if session_id in st.session_state.sessions:
        st.session_state.sessions.remove(session_id)

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


def process_documents_action(uploaded_files: list, session_id: str, append_mode: bool = True) -> None:
    """Run Phase 1-4 ingestion with live Streamlit status & progress bar."""
    progress_bar = st.progress(0.0, text="Starting streaming micro-batch ingestion…")

    with st.status("⚙️ Processing & Streaming Micro-Batches…", expanded=True) as status:
        try:
            result = ingest_files_with_micro_batches(
                uploaded_files=uploaded_files,
                session_id=session_id,
                append_mode=append_mode,
                log_fn=st.write,
                progress_fn=lambda pct, msg: progress_bar.progress(pct, text=msg),
            )
        except Exception as exc:
            progress_bar.empty()
            status.update(label="Ingestion failed", state="error", expanded=True)
            st.error(f"Failed to index documents: {exc}")
            return

        progress_bar.empty()
        if not result["good_files"]:
            status.update(label="No readable documents found", state="error", expanded=True)
            st.error("None of the uploaded files could be read.")
            return

        status.update(
            label=f"Ready — {result['chunks']:,} chunks indexed in {result['seconds']:.1f}s",
            state="complete",
            expanded=False,
        )

    bucket = ensure_session_bucket(session_id)
    bucket["indexed_files"] = extract_indexed_filenames(result["store"], session_id) or result["good_files"]
    bucket["index_stats"] = {
        "files": len(bucket["indexed_files"]),
        "chunks": result["chunks"],
        "seconds": result["seconds"],
        "failed": result["failed"],
    }
    save_disk_metadata()
    st.session_state.uploader_key += 1
    st.toast(f"Indexed {len(result['good_files'])} file(s) into Session #{session_id}!", icon="✅")
    st.rerun()


def format_chat_export(session_id: str, messages: list) -> str:
    display_name = get_session_display_name(session_id)
    lines = [f"# RAG Chat Transcript — Session `{display_name}`\n"]
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"### {role}\n{m['content']}\n")
    return "\n".join(lines)


def render_sidebar(session_id: str, store: Chroma | None) -> None:
    """Render the complete Sessions Manager, Knowledge Base uploader, File Filter, and Settings sidebar."""
    bucket = ensure_session_bucket(session_id)

    with st.sidebar:
        st.markdown("### ◈ Sessions Manager")

        selected_session = st.selectbox(
            "Active Session ID",
            options=st.session_state.sessions,
            index=st.session_state.sessions.index(session_id),
            format_func=lambda x: f"📂 {get_session_display_name(x)} ({len(_peek_bucket(x).get('indexed_files', []))} docs)",
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
                delete_session_action(session_id)

        with st.expander("✏️ Rename Session", expanded=False):
            current_custom_name = str(bucket.get("name") or "").strip()
            new_name = st.text_input(
                "Session Name",
                value=current_custom_name,
                max_chars=40,
                key=f"rename_input_{session_id}",
                placeholder=get_session_base_label(session_id),
            )
            if st.button("Save Name", use_container_width=True, key=f"save_name_btn_{session_id}"):
                cleaned_name = new_name.strip()
                if not cleaned_name:
                    if current_custom_name:
                        bucket["name"] = ""
                        save_disk_metadata()
                        st.toast("Session reset to default name!", icon="🔄")
                        st.rerun()
                    else:
                        st.warning("Please enter a session name.")
                elif cleaned_name.lower() == current_custom_name.lower():
                    st.info("That is already the current name.")
                else:
                    existing_names = {
                        get_session_base_label(sid).strip().lower()
                        for sid in st.session_state.sessions
                        if sid != session_id
                    }
                    if cleaned_name.lower() in existing_names:
                        st.error(f"A session named '{cleaned_name}' already exists!")
                    else:
                        bucket["name"] = cleaned_name
                        save_disk_metadata()
                        st.toast(f"Session renamed to '{cleaned_name}'!", icon="✏️")
                        st.rerun()

        if store is not None:
            chunk_count = store._collection.count()
            indexed_files = bucket.get("indexed_files") or extract_indexed_filenames(store, session_id)
            bucket["indexed_files"] = indexed_files
            pills = "".join(f'<span class="file-pill">📄 {html.escape(f)}</span>' for f in indexed_files[:12])
            if len(indexed_files) > 12:
                pills += f'<span class="file-pill">+{len(indexed_files) - 12} more</span>'
            stats = bucket.get("index_stats")
            timing_html = (
                f'<br><span style="color:#94a3b8">indexed in {stats["seconds"]:.1f}s</span>'
                if stats and stats.get("seconds")
                else ""
            )
            st.markdown(
                f'<div class="stat-card">✅ <b>{len(indexed_files)}</b> file(s) · '
                f'<b>{chunk_count:,}</b> chunks{timing_html}'
                f'<div style="margin-top:0.4rem">{pills}</div></div>',
                unsafe_allow_html=True,
            )

            if indexed_files:
                with st.expander("🗂️ Filter / Manage Files", expanded=False):
                    selected_scope = st.multiselect(
                        "🎯 Scope Search to File(s)",
                        options=indexed_files,
                        default=[f for f in bucket.get("active_files", []) if f in indexed_files],
                        key=f"scope_files_{session_id}",
                        placeholder="All indexed files (default)",
                        help="Select specific file(s) to restrict search, or leave empty to search all files.",
                    )
                    if selected_scope != bucket.get("active_files", []):
                        bucket["active_files"] = selected_scope
                        save_disk_metadata()

                    st.markdown("---")
                    file_to_remove = st.selectbox(
                        "Remove a Single File",
                        options=indexed_files,
                        key=f"del_file_select_{session_id}",
                    )
                    if st.button(
                        "🗑️ Remove Selected File",
                        use_container_width=True,
                        key=f"del_file_btn_{session_id}",
                    ):
                        removed_count = delete_single_file_from_session(session_id, file_to_remove)
                        bucket["indexed_files"] = [f for f in indexed_files if f != file_to_remove]
                        bucket["active_files"] = [
                            f for f in bucket.get("active_files", []) if f != file_to_remove
                        ]
                        save_disk_metadata()
                        st.toast(f"Removed `{file_to_remove}` ({removed_count} chunks)!", icon="🗑️")
                        st.rerun()

        st.markdown("---")
        st.markdown("### ◈ Knowledge Base")

        api_key_ok = bool(os.getenv("GROQ_API_KEY"))
        if not api_key_ok:
            st.error("`GROQ_API_KEY` not found in `.env` file.")

        uploaded_files = st.file_uploader(
            "Upload files (.xlsx, .xls, .csv, .pdf, .txt, .md)",
            type=["xlsx", "xls", "csv", "pdf", "txt", "md"],
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
            process_documents_action(uploaded_files, session_id, append_mode=append_mode)

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
                options=["Hybrid + MMR (Diverse)", "Hybrid + Similarity (Distance)"],
                index=0,
                horizontal=True,
            )
            st.session_state.top_k = st.slider(
                "Passages to Retrieve (Top-K)",
                min_value=2,
                max_value=12,
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


def render_hero_header(session_id: str, ready: bool, store: Chroma | None) -> None:
    """Render the top dashboard hero header."""
    dot_class = "dot-ready" if ready else "dot-empty"
    status_text = f"Ready · {store._collection.count():,} Indexed Chunks" if ready and store else "Awaiting Documents"
    display_title = get_session_display_name(session_id)

    bucket = _peek_bucket(session_id)
    active_files = bucket.get("active_files") or []
    scope_note = f" · 🎯 Scoped to {len(active_files)} file(s)" if active_files else ""

    st.markdown(
        f"""
        <section class="rag-hero">
            <div class="rag-topbar">
                <div class="rag-eyebrow">
                    <span class="status-dot {dot_class}"></span>
                    Groq · Polars Calamine + Hybrid Chroma RAG · {status_text}{html.escape(scope_note)}
                </div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Model: <code>{html.escape(st.session_state.selected_model)}</code></div>
            </div>
            <h1 class="rag-title">Active Session: <span>{html.escape(display_title)}</span></h1>
            <p class="rag-subtitle">
                Data is strictly isolated and persisted per session. Type <code>delete session</code> in chat or click the sidebar delete button to wipe it.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict]) -> None:
    """
    Render expandable source citation cards with:
      1. A Hybrid Search Telemetry Bar summarizing how many sources matched each engine layer.
      2. Color-coded pill badges on every source card:
         - 🏷️ Exact Category Match (Emerald Green)
         - 🔤 FTS5 Keyword Match   (Amber Gold)
         - 🧠 Semantic Vector (X%) (Indigo Purple)
    """
    if not sources:
        return

    with st.expander(f"📚 Sources & Hybrid Attribution ({len(sources)})"):
        cat_count = sum(
            1
            for s in sources
            if s.get("matched_category") or "Categorical" in str(s.get("match_type", ""))
        )
        fts_count = sum(
            1
            for s in sources
            if s.get("matched_fts") or "FTS5" in str(s.get("match_type", ""))
        )
        vec_count = sum(
            1
            for s in sources
            if s.get("matched_vector")
            or "Vector" in str(s.get("match_type", ""))
            or (not s.get("badges") and not s.get("match_type"))
        )

        st.markdown(
            f"""
            <div class="hybrid-summary-bar">
                <b>⚡ Hybrid Retrieval Breakdown:</b>
                <span class="match-badge badge-category">🏷️ Categorical Filter: {cat_count}</span>
                <span class="match-badge badge-fts">🔤 FTS5 Keyword: {fts_count}</span>
                <span class="match-badge badge-vector">🧠 Semantic Vector: {vec_count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for idx, s in enumerate(sources, start=1):
            safe_source = html.escape(str(s.get("source", "unknown")))
            page_str = f", row/page {int(s['page'])}" if isinstance(s.get("page"), int) else ""

            badges = s.get("badges")
            if not badges:
                badges = []
                m_type = str(s.get("match_type") or "")
                pct = s.get("match_pct")
                if "Categorical" in m_type:
                    badges.append({"label": "🏷️ Exact Category Match", "css": "badge-category"})
                if "FTS5" in m_type:
                    badges.append({"label": "🔤 FTS5 Keyword Match", "css": "badge-fts"})
                if "Vector" in m_type or not badges:
                    vec_lbl = f"🧠 Semantic Vector ({pct}%)" if isinstance(pct, int) else "🧠 Semantic Vector"
                    badges.append({"label": vec_lbl, "css": "badge-vector"})

            badges_html = "".join(
                f'<span class="match-badge {html.escape(b.get("css", "badge-vector"))}">'
                f'{html.escape(b.get("label", ""))}</span>'
                for b in badges
            )

            raw_text = str(s.get("text", ""))
            snippet = html.escape(raw_text[:450] + ("…" if len(raw_text) > 450 else ""))
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-header">
                        <div><b>{idx}. 📄 {safe_source}{page_str}</b></div>
                        <div class="source-badges">{badges_html}</div>
                    </div>
                    <div class="source-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_followup_pills(followups: list[str], msg_idx: int, session_id: str) -> str | None:
    """
    Render 3 clickable follow-up question buttons underneath the latest assistant response.
    Returns the clicked question text (if clicked by the user) so `main.py` can execute it.
    """
    if not followups:
        return None

    st.markdown(
        '<div style="font-size:0.8rem;color:#94a3b8;margin:0.35rem 0 0.25rem 0;">'
        "💡 <b>Suggested follow-ups:</b></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(followups))
    clicked_question: str | None = None
    for idx, q_text in enumerate(followups):
        if cols[idx].button(
            f"↪ {q_text}",
            key=f"followup_{session_id}_{msg_idx}_{idx}",
            use_container_width=True,
        ):
            clicked_question = q_text
    return clicked_question


def handle_user_question(question: str, session_id: str, store: Chroma) -> None:
    """Execute Smart Query Rewriting, Phase 5 Hybrid Search, stream LLM answer, and generate follow-ups."""
    clean_q = question.lower().strip()
    if clean_q in ["delete session", "delete this session", "clear session", "/delete"]:
        delete_session_action(session_id)
        return

    bucket = ensure_session_bucket(session_id)
    prior_messages = list(bucket["messages"])
    bucket["messages"].append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    started = time.time()
    sources: list[dict] = []
    answer = ""
    used_model = st.session_state.selected_model
    errored = False
    active_files = bucket.get("active_files") or None
    rewritten_query: str | None = None
    followups: list[str] = []

    with st.chat_message("assistant"):
        with st.status("🔎 Searching your documents (Categorical + FTS5 + Vector)…", expanded=False) as status:
            try:
                search_query = rewrite_conversational_query(question, prior_messages)
                if search_query.strip().lower() != question.strip().lower():
                    rewritten_query = search_query
                    st.write(f"🔄 **Contextualized Search Query:** `{search_query}`")

                use_mmr = "MMR" in st.session_state.get("search_type", "MMR")
                sources = hybrid_retrieve(
                    store=store,
                    session_id=session_id,
                    question=search_query,
                    top_k=st.session_state.top_k,
                    use_mmr=use_mmr,
                    allowed_files=active_files,
                )
                scope_msg = f" (scoped to {len(active_files)} file(s))" if active_files else ""
                status.update(
                    label=f"Found {len(sources)} relevant passages/rows{scope_msg}",
                    state="complete",
                )
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

                inv = get_session_inventory(session_id)
                all_files = bucket.get("indexed_files") or [f["file_name"] for f in inv.get("files", [])]
                if inv.get("files"):
                    file_lines = [
                        f"- {f['file_name']} ({f['chunks']} chunks"
                        + (f", sheets: {', '.join(f['sheets'])}" if f.get("sheets") else "")
                        + ")"
                        for f in inv["files"]
                    ]
                else:
                    file_lines = [f"- {fname}" for fname in all_files]

                manifest_header = (
                    f"Uploaded Files in Current Session ({len(file_lines)} total):\n"
                    + "\n".join(file_lines)
                    + "\n\nRetrieved Context Passages:\n"
                )

                passages_text = "\n\n".join(
                    f"Source ({s['source']}{', row/page ' + str(s['page']) if s.get('page') else ''}):\n{s['text']}"
                    for s in sources
                )
                context = manifest_header + passages_text

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

        if _is_not_found_answer(answer):
            sources = []
        elif not errored and sources:
            all_files_list = bucket.get("indexed_files") or []
            followups = generate_followup_questions(question, answer, sources, all_files_list)

        elapsed = time.time() - started
        if not errored:
            rewrite_note = f" · 🔄 Searched: `{rewritten_query}`" if rewritten_query else ""
            st.caption(f"Answered in {elapsed:.1f}s · Model: `{used_model}`{rewrite_note}")
            render_sources(sources)

    bucket["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [] if errored else sources,
            "followups": followups,
            "rewritten_query": rewritten_query,
            "error": errored,
            "elapsed": elapsed,
        }
    )
    save_disk_metadata()
    if followups:
        st.rerun()
