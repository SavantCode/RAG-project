"""
Groq LLM Service Module.
Handles live model discovery, conversational query rewriting, grounded follow-up
question generation, prompt message assembly, friendly error formatting, and
auto-fallback streaming if a model returns 404 or is decommissioned.
"""

from __future__ import annotations

import os
import re
import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from engine.config import FALLBACK_GROQ_MODELS, HISTORY_TURNS
from prompts import (
    FOLLOWUP_SUGGESTIONS_PROMPT,
    QUERY_REWRITER_PROMPT,
    SYSTEM_PROMPT,
)


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


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _needs_contextual_rewrite(question: str, prior_messages: list) -> bool:
    """
    Stage 1 Gatekeeper (<0.1ms):
    Determines whether the user's question relies on prior conversation context.
    """
    if not prior_messages:
        return False

    q_clean = question.lower().strip()
    words = re.findall(r"[a-z0-9_]+", q_clean)
    if not words:
        return False

    meta_phrases = ("how many files", "what files", "which files", "list all files", "uploaded files")
    if any(p in q_clean for p in meta_phrases):
        return False

    referential_pronouns = {
        "it", "its", "they", "them", "their", "theirs",
        "this", "that", "these", "those", "he", "she", "his", "her",
        "same", "other", "another", "both", "former", "latter",
    }
    if any(w in referential_pronouns for w in words):
        return True

    followup_starters = (
        "what about", "how about", "and ", "also ", "why ", "how much",
        "how many", "what are the", "what is the", "can you", "tell me more",
        "elaborate", "compare", "in which", "on which", "where ", "when ",
    )
    if any(q_clean.startswith(prefix) for prefix in followup_starters):
        return True

    if len(words) <= 7:
        return True

    return False


def rewrite_conversational_query(question: str, prior_messages: list) -> str:
    """
    Stage 2 Smart Conversational Query Rewriter:
    Resolves pronouns and implicit entities while avoiding topic-switch pollution.
    """
    if not _needs_contextual_rewrite(question, prior_messages):
        return question

    valid_history = [m for m in prior_messages if not m.get("error") and m.get("content")][-6:]
    if not valid_history:
        return question

    history_lines: list[str] = []
    for m in valid_history:
        role_label = "User" if m["role"] == "user" else "Assistant"
        snippet = m["content"][:300].replace("\n", " ")
        history_lines.append(f"{role_label}: {snippet}")

    history_block = "\n".join(history_lines)
    rewrite_messages = [
        SystemMessage(content=QUERY_REWRITER_PROMPT),
        HumanMessage(
            content=(
                f"Recent Conversation History:\n{history_block}\n\n"
                f"Latest User Question: {question}\n\n"
                "Standalone Search Query:"
            )
        ),
    ]

    active_models = fetch_active_groq_models()
    fast_candidates = [
        m for m in ("llama-3.1-8b-instant", "llama-3.3-70b-versatile", st.session_state.get("selected_model", ""))
        if m
    ]
    for m in active_models:
        if m not in fast_candidates:
            fast_candidates.append(m)

    for model_id in fast_candidates[:3]:
        try:
            llm = ChatGroq(model=model_id, temperature=0.0, max_tokens=250, max_retries=1)
            resp = llm.invoke(rewrite_messages)
            raw_text = _strip_thinking_tags(resp.content if isinstance(resp.content, str) else "")
            rewritten = re.sub(r"^(standalone search query:|search query:|query:)\s*", "", raw_text, flags=re.I)
            rewritten = rewritten.strip("\"'` \n").splitlines()[0].strip() if rewritten.strip() else ""
            if rewritten and len(rewritten) >= 2:
                return rewritten
        except Exception:
            continue

    return question


def _build_deterministic_followups(sources: list[dict], all_files: list[str] | None = None) -> list[str]:
    """
    Guaranteed zero-latency fallback follow-up generator derived from the actual source filenames.
    """
    source_names: list[str] = []
    for s in sources:
        src = str(s.get("source") or "").split("|")[0].strip()
        if src and src not in source_names:
            source_names.append(src)

    primary = source_names[0] if source_names else (all_files[0] if all_files else "this document")
    short_primary = primary.rsplit(".", 1)[0][:28]

    suggestions = [
        f"What are the key details in {short_primary}?",
        f"Summarize the main findings from {short_primary}?",
    ]
    if all_files and len(all_files) > 1:
        other_files = [f for f in all_files if f != primary]
        if other_files:
            short_other = other_files[0].rsplit(".", 1)[0][:25]
            suggestions.append(f"How does this compare with {short_other}?")
        else:
            suggestions.append("What specific facts or figures are mentioned?")
    else:
        suggestions.append("What specific facts or figures are mentioned?")

    return suggestions[:3]


def generate_followup_questions(
    question: str,
    answer: str,
    sources: list[dict],
    all_files: list[str] | None = None,
) -> list[str]:
    """
    Enterprise Grounded Follow-Up Generator:
    Generates 3 concise, clickable follow-up questions strictly grounded in the
    retrieved `sources` and `all_files` inventory.
    """
    if not sources:
        return []

    snippets = [
        f"[{s.get('source', 'doc')}]: {str(s.get('text', ''))[:260]}"
        for s in sources[:4]
    ]
    files_line = f"Uploaded Files: {', '.join(all_files[:10])}\n" if all_files else ""
    user_payload = (
        f"{files_line}"
        f"Source Passages:\n" + "\n".join(snippets) + "\n\n"
        f"User Question: {question}\n"
        f"Assistant Answer: {answer[:350]}\n\n"
        "Output 3 short follow-up questions (one per line):"
    )

    messages = [
        SystemMessage(content=FOLLOWUP_SUGGESTIONS_PROMPT),
        HumanMessage(content=user_payload),
    ]

    active_models = fetch_active_groq_models()
    fast_candidates = [
        m
        for m in (
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            st.session_state.get("selected_model", ""),
        )
        if m
    ]
    for m in active_models:
        if m not in fast_candidates:
            fast_candidates.append(m)

    for model_id in fast_candidates[:3]:
        try:
            llm = ChatGroq(model=model_id, temperature=0.2, max_tokens=350, max_retries=1)
            resp = llm.invoke(messages)
            raw = _strip_thinking_tags(resp.content if isinstance(resp.content, str) else "")
            lines: list[str] = []
            for line in raw.splitlines():
                cleaned = re.sub(r"^[\d\-\*\•\.\)\s]+", "", line).strip().strip("\"'`")
                if not cleaned or len(cleaned) < 8:
                    continue
                if cleaned.lower().startswith(("here are", "sure", "follow-up", "questions:")):
                    continue
                if not cleaned.endswith("?"):
                    cleaned += "?"
                lines.append(cleaned[:85])
                if len(lines) == 3:
                    break
            if len(lines) >= 2:
                return lines[:3]
        except Exception:
            continue

    return _build_deterministic_followups(sources, all_files)


def build_messages(prior_messages: list, question: str, context: str) -> list:
    """Assemble SystemPrompt + recent conversation turns + current context & question."""
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


def friendly_error(exc: Exception) -> str:
    """Translate raw API exceptions into clear, user-friendly messages."""
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
