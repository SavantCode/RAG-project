Here is a complete, production-grade README.md file for your project. Save this directly as README.md in your root RAG_LangChain/ folder.

# ⚡ Enterprise Multi-Session Hybrid RAG Pipeline

An enterprise-grade, multi-session **Retrieval-Augmented Generation (RAG)** system built for high-precision clinical spreadsheets (`.xlsx`, `.xls`, `.csv`) and unstructured documents (`.pdf`, `.txt`, `.md`). 

Powered by **Polars + Rust Calamine**, **ChromaDB**, **SQLite FTS5 + B-Tree Hybrid Indexing**, **HuggingFace Embeddings**, and **Groq LPUs**, with a modularized **Streamlit** dark-slate dashboard.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | `Streamlit` + Custom CSS3 | Multi-session workspace management, live streaming responses, source citation cards |
| **Excel Reading Engine** | `Polars` + `fastexcel` (`python-calamine`) | Rust-powered reading of multi-tab `.xlsx` workbooks in milliseconds with minimal RAM |
| **PDF / Text Engine** | `pypdf` + `LangChain Text Splitters` | Page-level PDF extraction, encryption handling, and recursive semantic chunking |
| **Vector Embeddings** | `HuggingFace` (`all-MiniLM-L6-v2`) | Local, normalized 384-dimensional dense embeddings (batched at `batch_size=64`) |
| **Dense Vector Store** | `ChromaDB` (`PersistentClient`) | Isolated per-session semantic vector collections (`session_<id>`) |
| **Dual-Layer Hybrid DB** | `SQLite3` (`B-Tree` + `FTS5` BM25) | Sub-millisecond categorical pre-filtering + exact keyword matching (e.g., `NCT02475681`, `del(17p)`) |
| **LLM Inference** | `Groq Cloud API` (`ChatGroq`) | Live model discovery (`Llama-3.3-70B`, `Llama-3.1-8B`) with automatic `404` model fallback |

---

## 📂 Project Folder Structure

```text
RAG_LangChain/
│
├── .env                        # Environment variables (GROQ_API_KEY, GROQ_MODEL)
├── main.py                     # Application entry point & state orchestrator
├── prompts.py                  # Centralized system prompts & workflow instructions
├── filereader.py               # PDF, TXT & Markdown parser module
├── excelreader.py              # Phase 1 & 2: Polars + Rust Calamine + 1-Row-Per-Chunk Slicer
│
├── engine/                     # Phase 3, 4 & 5: Core Backend & Retrieval Engine
│   ├── __init__.py
│   ├── config.py               # Centralized constants, batch sizes, and file limits
│   ├── hybrid_index.py         # Phase 4: Dual-Layer Categorical B-Tree + SQLite FTS5 Index
│   ├── storage.py              # Phase 3 & 5: Streaming Micro-Batches, ChromaDB & Hybrid Search
│   └── llm_service.py          # Groq live model discovery, error translation & fallback streaming
│
├── frontend/                   # Separated Presentation Layer
│   ├── __init__.py
│   ├── styles.py               # Dark Slate & Indigo CSS theme + animated status badges
│   └── components.py           # Sidebar, Hero Banner, Source Cards & Chat UI handlers
│
└── chroma_db/                  # Auto-generated persistent storage on disk
    ├── chroma.sqlite3          # ChromaDB vector collections
    ├── hybrid_fts_index.db     # Dual-Layer Categorical B-Tree + FTS5 Full-Text Index
    └── workspace_state.json    # Persisted session list, file registries & chat transcripts
Copy
🏗️ System Architecture & 5-Phase Processing Flow
flowchart TD
    subgraph UI ["1. Frontend Layer (frontend/)"]
        A[User Uploads Files / Asks Question] --> B[frontend/components.py]
    end

    subgraph PARSERS ["2. Modular Ingestion & Slicing (Phase 1 & 2)"]
        B -->|".xlsx, .xls, .csv"| C[excelreader.py<br/>Polars + Rust Calamine]
        B -->|".pdf, .txt, .md"| D[filereader.py<br/>PyPDF + Recursive Splitter]
        C --> E[Schema-Agnostic 1-Row-Per-Chunk Slicer<br/>Prepends: Workbook + Sheet + Medicine Name]
        D --> F[Page-Level Semantic Chunks]
    end

    subgraph STREAM ["3. Streaming Micro-Batches (Phase 3)"]
        E & F --> G[engine/storage.py<br/>SHA-256 Deduplication + 250-Row Micro-Batches]
    end

    subgraph INDEX ["4. Dual-Layer Storage & Indexing (Phase 4)"]
        G -->|Dense Embeddings| H[(ChromaDB<br/>session_id Collection)]
        G -->|Categorical + Raw Text| I[(SQLite Hybrid DB<br/>B-Tree + FTS5 BM25 Index)]
    end

    subgraph SEARCH ["5. High-Precision Hybrid Retrieval (Phase 5)"]
        B -->|User Question| J[detect_categorical_filters<br/>Matches medicine_name & sheet_name]
        J -->|Hard Metadata Filter| H
        J -->|Exact Token Match| I
        H & I --> K[Hybrid Reciprocal Fusion<br/>FTS5 Exact Hits + MMR Vector Hits]
        K --> L[engine/llm_service.py<br/>Groq Streaming + Auto-Fallback]
    end
Copy
🔄 Detailed Step-by-Step Execution Flow
Phase 1: High-Speed Reading Engine (excelreader.py & filereader.py)
When files are uploaded in the sidebar (frontend/components.py), they are passed to route_uploaded_file_to_chunks() in engine/storage.py.
For Excel/CSV files (.xlsx, .xls, .csv):
Routed to excelreader.py (parse_excel_or_csv_bytes).
Uses Rust Calamine (fastexcel) + Polars to load all tabs in the workbook simultaneously into memory in milliseconds without freezing the system.
For Unstructured Documents (.pdf, .txt, .md):
Routed to filereader.py (parse_pdf_bytes / parse_text_or_md_bytes).
Phase 2: Schema-Agnostic Chunking & Context Prepending (excelreader.py)
Spreadsheets have varying column headers across tabs (e.g., Efficacy Evidence vs. Therapeutic Duplications).

Auto-Header Promotion: Scans the first 10 rows to detect and promote the real table header if Rows 1–3 are title banners (Unnamed: 0).
Merged Cell Forward-Filling: Automatically forward-fills Column A (forward_fill()) so visually merged parent categories are preserved on every sub-row.
1-Row = 1-Chunk Isolation: Every non-empty row becomes its own independent Document block. Separate clinical rules or drug interactions are never blended together.
Global Context Prepending: Every single row chunk is prefixed with its global origin before listing cell values:
[Workbook: Abatacept_Rules.xlsx | Sheet Tab: Duplications | Index Medicine: Abatacept | Row 14]
Drug Class: Biologic DMARD | Interaction: TNF Blockers | Severity: Contraindicated
Copy
Phase 3: Streaming Micro-Batch Embeddings (engine/storage.py)
Inside ingest_files_with_micro_batches(), each chunk is assigned a deterministic SHA-256 ID based on source + page/row + content. Re-uploading the same file updates existing records instead of creating duplicates.
Chunks are accumulated in a memory-safe buffer of MICRO_BATCH_SIZE = 250.
As soon as the buffer reaches 250 rows, it is immediately embedded via HuggingFaceEmbeddings (all-MiniLM-L6-v2) and flushed to disk, keeping RAM usage flat even across hundreds of workbooks.
Phase 4: Dual-Layer Database Indexing (engine/hybrid_index.py)
Every micro-batch is written to two synchronized engines inside ./chroma_db/:

ChromaDB Collection (session_<id>): Stores the dense neural vectors and metadata payloads.
SQLite Hybrid Database (hybrid_fts_index.db):
Index 1 — Categorical Payload B-Tree Indexes: Hard lookup indexes on (session_id, medicine_name) and (session_id, sheet_name).
Index 2 — FTS5 Full-Text Keyword Index: Inverted word index configured with tokenchars "-_()" so clinical trial IDs (NCT02475681), gene mutations (del(17p)), and dosages match with 100% precision.
Phase 5: High-Precision Hybrid Search & Groq Streaming (engine/storage.py & engine/llm_service.py)
When a user asks a question in the chat UI:

Step 1 — Hard Categorical Filtering: detect_categorical_filters() checks if the question mentions any medicine_name or sheet_name in the current session's catalog. If matched, it hides 99.9% of unrelated rows.
Step 2 — FTS5 BM25 Keyword Search: search_fts_keywords() retrieves exact token matches from the FTS5 index within the filtered scope.
Step 3 — Scoped Semantic Vector Scan: ChromaDB runs Maximal Marginal Relevance (MMR) or similarity search strictly on the filtered metadata pool (with automatic fallback to global session search if the filter yields 0 hits).
Step 4 — Live LLM Synthesis with Auto-Fallback: stream_with_auto_fallback() in engine/llm_service.py queries Groq's live /v1/models endpoint and streams the response token-by-token. If a model returns 404 or is decommissioned, it transparently switches to the next active model.
🚀 How to Run the Project
1. Activate Virtual Environment
.\clean_env\Scripts\Activate.ps1
Copy
2. Install Dependencies
python -m pip install streamlit python-dotenv langchain-chroma langchain-core langchain-groq langchain-huggingface langchain-text-splitters pypdf sentence-transformers "numpy<2.0.0" polars fastexcel openpyxl requests
Copy
3. Configure .env
Create a .env file in the root directory:

GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
Copy
4. Start the Application
python -m streamlit run main.py
Copy
