from __future__ import annotations

import html
import streamlit as st


def apply_dashboard_theme() -> None:
    """
    Apply sleek dark slate & indigo dashboard styling.
    Includes enterprise color-coded pill badges for 3-Layer Hybrid Search attribution:
      - .badge-category (Emerald): B-Tree Categorical Hard Filter Match
      - .badge-fts      (Amber)  : SQLite FTS5 BM25 Exact Keyword Match
      - .badge-vector   (Indigo) : ChromaDB Dense Semantic Vector Match
    """
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
            border-radius: 12px;
            padding: 0.8rem 1rem;
            background: var(--surface);
            margin-bottom: 0.6rem;
        }
        .source-header {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.45rem;
            margin-bottom: 0.35rem;
        }
        .source-badges {
            display: inline-flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            align-items: center;
        }
        /* Color-Coded 3-Layer Hybrid Search Pill Badges */
        .match-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.16rem 0.58rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            white-space: nowrap;
        }
        .badge-category {
            background: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(16, 185, 129, 0.45);
            color: #34d399;
        }
        .badge-fts {
            background: rgba(245, 158, 11, 0.14);
            border: 1px solid rgba(245, 158, 11, 0.45);
            color: #fbbf24;
        }
        .badge-vector {
            background: rgba(99, 102, 241, 0.16);
            border: 1px solid rgba(129, 140, 248, 0.45);
            color: #a5b4fc;
        }
        .hybrid-summary-bar {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.8rem;
            margin-bottom: 0.65rem;
            border-radius: 10px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            font-size: 0.8rem;
            color: #cbd5e1;
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
