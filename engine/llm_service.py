"""
Groq LLM Service Module.
Handles live model discovery, prompt message assembly, friendly error formatting,
and auto-fallback streaming if a model returns 404 or is decommissioned.
"""

from __future__ import annotations

import os
import re
import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from engine.config import FALLBACK_GROQ_MODELS, HISTORY_TURNS
from prompts import SYSTEM_PROMPT


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
