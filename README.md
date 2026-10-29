

---
<div align="center">

# 🏗️ Enterprise Multi-Modal & Multi-Sheet Hybrid RAG Assistant

**Upload multi-sheet Excel workbooks, CSVs, PDFs, Word documents, Markdown/text files, or live website URLs — and ask grounded questions through a Streamlit chat interface.**

Powered by Polars · Calamine · ChromaDB · SQLite FTS5 · Groq LLMs

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-CD792C?style=for-the-badge&logo=polars&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6446?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📑 Table of Contents

- [🧠 What Is This?](#-what-is-this)
- [🎬 Demo](#-demo)
- [✨ Features](#-features)
- [⚙️ How It Works](#️-how-it-works)
- [🔬 Deep-Dive Ingestion & Retrieval Flows](#-deep-dive-ingestion--retrieval-flows)
- [📂 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🧩 Key Architectural Concepts](#-key-architectural-concepts)
- [🛠️ Tech Stack](#️-tech-stack)
- [🎛️ Configuration](#️-configuration)
- [📐 Design Decisions](#-design-decisions)
- [⚠️ Limitations](#️-limitations)
- [💡 Ideas for Further Development](#-ideas-for-further-development)
- [🔒 Security & Privacy](#-security--privacy)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🧠 What Is This?

This project is an **enterprise-ready hybrid retrieval-augmented generation (RAG) application** engineered to handle complex, multi-tab tabular data alongside unstructured documents and web pages.

Dense vector embeddings alone often struggle with multi-sheet spreadsheets, exact alphanumeric IDs, and numerical aggregations. Instead of relying solely on them, this application builds a **dual-layer persistent index** that combines:

- 🗃️ **SQLite** — Categorical B-Trees + FTS5 BM25 keyword search
- 🧲 **ChromaDB** — semantic vector search

The repository supports **three unified ingestion and retrieval workflows** inside a single workspace:

| # | Workflow | Module(s) | What it does |
| :-: | :--- | :--- | :--- |
| 1 | 📊 **Multi-sheet tabular analysis** | `excelreader.py` | Parses `.xlsx`, `.xls`, and `.csv` workbooks across 20+ worksheet tabs per file, promotes buried table headers, pre-computes per-sheet statistical profiles, and packs contiguous rows into context-rich chunks |
| 2 | 📄 **Unstructured document ingestion** | `filereader.py`, `engine/storage.py` | Extracts page-tracked text from `.pdf` files, paragraphs and structured tables from `.docx` archives, and recursive chunks from `.txt` / `.md` files |
| 3 | 🌐 **Live web & visual scraping** | `engine/storage.py` | Scrapes public webpage URLs and direct image links, extracting clean article prose alongside inline image markers and a dedicated image catalog chunk |

> The architecture is modular, so developers can trace every stage — from streaming micro-batch ingestion to multi-signal attribution fusion.

---

## 🎬 Demo
<p align="center">
  <img width="800" alt="Screenshot (2304)" src="https://github.com/user-attachments/assets/f9b4eede-b5df-4c1b-bbc0-c5bb2097619f" />
  <br><br>
  <img width="800" alt="Screenshot (2309)" src="https://github.com/user-attachments/assets/97c45cdd-463b-4951-8610-50b858b3e619" />
  <br><br>
  <img width="800" alt="Screenshot (2305)" src="https://github.com/user-attachments/assets/42b821f1-89a9-4a30-af7a-2e1b8dafa237" />
  <br><br>
  <img width="800" alt="Screenshot (2306)" src="https://github.com/user-attachments/assets/d83c23c6-ef94-4ea7-a393-44e418981882" />
  <br><br>
  <img width="800" alt="Screenshot (2307)" src="https://github.com/user-attachments/assets/c1f1ede2-50ef-4981-9602-ad6c7f887c1f" />
  <br><br>
  <img width="800" alt="Screenshot (2308)" src="https://github.com/user-attachments/assets/f12d8d46-222d-4469-a516-0e057556612b" />
</p>






The Streamlit workspace provides:

- 🗂️ Isolated multi-session management
- 🎯 Granular per-file scope filtering
- 🏷️ Hybrid retrieval telemetry badges
- 💬 Clickable follow-up question pills
- 🌙 Document-grounded answers in a custom dark interface

---

## ✨ Features

### 📥 Ingestion

- Upload and index up to **1,500+** `.xlsx`, `.xls`, `.csv`, `.pdf`, `.docx`, `.txt`, and `.md` files per session.
- Scrape and index live website URLs, including embedded `<img>` figures and direct image URLs.
- Parse every worksheet tab in Excel files with a **3-stage failover engine** (`fastexcel` → `polars` → `pandas`).
- Automatically recover single-row sheets, headless key-value tables, and Row 1–3 title banners without data loss.
- Generate a dedicated **Sheet Overview, Column Schema & Statistical Profile** chunk (`page=1`) for every Excel tab, with pre-computed numeric aggregations (count, sum, avg, min, max) and unique categorical values.
- Pack contiguous spreadsheet rows into adaptive, context-prepended blocks with forward-filled merged category cells.
- Create local dense embeddings with HuggingFace (`sentence-transformers`), cached in server memory.

### 🔎 Retrieval

- Store and query chunks across both **ChromaDB** (persistent vector store) and **SQLite WAL** (B-Tree metadata + FTS5 BM25 keyword index).
- Resolve exact, fuzzy, token-overlap, short-code (`PK`, `AE`, `Sheet1`), and ordinal (`"first sheet"`, `"2nd tab"`) sheet references.
- Prevent cross-file filter collisions by verifying in SQLite that detected `file_name`, `sheet_name`, and `medicine_name` filters coexist before applying `$and`.
- Rewrite multi-turn conversational follow-ups to resolve pronouns (`"it"`, `"they"`) and implicit sheet references (`"in the sheet"`, `"this tab"`).

### 💬 Chat Experience

- Stream answers via Groq LLMs with **sub-100ms automatic model failover** on HTTP `429`, `413`, `404`, and `503` errors.
- Display expandable source citation cards with color-coded **Categorical**, **FTS5 Keyword**, and **Semantic Vector** match badges.
- Auto-generate **3 grounded follow-up question buttons** below every assistant response.
- Persist sessions, chat transcripts, and index statistics to disk across server restarts.

---

## ⚙️ How It Works

| Step | Stage | Description |
| :-: | :--- | :--- |
| 1️⃣ | **Ingest & Checkpoint** | The user uploads files or pastes URLs. In append mode, SQLite checks already-indexed filenames in under 1ms and automatically resumes from unindexed files. |
| 2️⃣ | **Route & Parse** | Files are dispatched by extension to `excelreader.py` (Excel/CSV), `filereader.py` (PDF/TXT/MD), or `engine/storage.py` (DOCX/URL). |
| 3️⃣ | **Profile & Chunk** | Excel sheets emit a `page=1` schema and statistical profile chunk followed by adaptive row packs; documents and web pages are split with `RecursiveCharacterTextSplitter`. |
| 4️⃣ | **Stream Micro-Batches** | Chunks are hashed with SHA-256 IDs and committed atomically at file boundaries into ChromaDB and SQLite (`chunk_metadata` + `chunk_fts`). |
| 5️⃣ | **Contextualize Query** | Follow-up questions with pronouns or implicit sheet references are rewritten into standalone search queries using recent chat history. |
| 6️⃣ | **Detect Filters & Guard Collisions** | `detect_categorical_filters` inspects the query for target workbooks, worksheet tabs, and entities, verifying coexistence in SQLite. |
| 7️⃣ | **Hybrid Retrieve & Fuse** | `hybrid_retrieve` fuses direct SQLite Sheet Overview/Row hits, FTS5 BM25 keyword hits, and ChromaDB MMR/Similarity vector hits into deduplicated, badge-annotated passages. |
| 8️⃣ | **Synthesize & Suggest** | The Groq streaming service generates a grounded response with automatic model failover and produces 3 clickable follow-up suggestions. |

### 🗺️ End-to-End Pipeline

```text
   Uploaded Excel / CSV / PDF / DOCX / TXT / MD / Website URLs
                              |
                              v
             Checkpoint auto-resume & file router
                              |
         +--------------------+--------------------+
         |                                         |
         v                                         v
 Excel 3-stage parser                     PDF / DOCX / URL parsers
 (Sheet overview + stats + row packs)     (Recursive text + table + image chunks)
         |                                         |
         +--------------------+--------------------+
                              |
                              v
             Streaming micro-batch commit (SHA-256)
                              |
         +--------------------+--------------------+
         |                                         |
         v                                         v
 ChromaDB persistent vectors              SQLite hybrid index (WAL mode)
 (Cosine distance + MMR)                  (B-Tree metadata + FTS5 BM25)
         |                                         |
         +--------------------+--------------------+
                              |
                              v
   Conversational query rewrite & collision-free filter detection
                              |
                              v
   Triple-engine hybrid fusion (Direct Sheet + FTS5 + Vector MMR)
                              |
                              v
      Groq streaming synthesis (auto-failover) + follow-up pills
```

---

## 🔬 Deep-Dive Ingestion & Retrieval Flows

### 📊 Flow A — Excel (`.xlsx`, `.xls`) & CSV Processing

**Module:** `excelreader.py`

#### 1. Per-Sheet 3-Stage Failover Reader — `parse_excel_or_csv_bytes`

| Stage | Engine | Behavior |
| :-: | :--- | :--- |
| **1** | `fastexcel` (Rust Calamine) | Loads each sheet tab individually. If `header_row=0` consumes the only row on a 1-row sheet and returns an empty DataFrame, `_load_single_sheet_via_fastexcel` automatically reloads that sheet with `header_row=None` so Row 1 is preserved. |
| **2** | `polars.read_excel` | Recovers any worksheet tab that failed during Stage 1. |
| **3** | `pandas.read_excel(sheet_name=None, header=None)` | Final fallback for legacy `.xls` files or non-standard XML table structures. |

#### 2. Smart Header Promotion & Banner Preservation — `_promote_real_header_row`

- Scans the top rows when `Unnamed:` columns dominate (caused by Row 1–3 title banners), promotes the true column header row, and saves banner notes above the header into the Sheet Overview chunk.
- Strips blank columns/rows in Rust/Polars and forward-fills Column A when merged parent category cells span multiple child rows.

#### 3. Sheet Overview, Column Schema & Statistical Profile Chunk (`page=1`)

Emits a high-signal summary chunk for every sheet tab containing:

- The workbook filename
- A 1-based tab index (`Sheet Tab #1`)
- Normalized sheet tokens
- Total row and column counts
- Top unique categorical values
- Pre-computed numeric statistics (count, sum, avg, min, max)

#### 4. Adaptive Row-Pack Slicing — `_slice_polars_df_into_rows`

- Packs contiguous rows (up to **6 rows** or **950 characters**) with self-contained headers.
- Header format: `[Workbook | Sheet Tab # | Index Medicine/Topic | Rows X-Y]`

---

### 📄 Flow B — PDF, DOCX, TXT & Markdown Processing

**Modules:** `filereader.py` & `engine/storage.py`

| Parser | Function | Details |
| :--- | :--- | :--- |
| **PDF** | `parse_pdf_bytes` | Extracts text page by page with `pypdf`, tagging each chunk with its exact 1-based page number and `sheet_name = page_<N>`. |
| **DOCX** | `parse_docx_bytes` | Zero-dependency parsing. Reads `word/document.xml` via Python's built-in `zipfile` and `xml.etree.ElementTree`, extracting paragraphs (`<w:p>`) and structured tables (`<w:tbl>`, cells joined with a pipe character). |
| **TXT / MD** | `parse_text_or_md_bytes` | Decodes UTF-8/Latin-1 bytes and splits with `RecursiveCharacterTextSplitter`. |

---

### 🌐 Flow C — Live Website URL & Image Scraping

**Function:** `parse_url_to_chunks`

- **Direct Image URLs** — Links ending in `.png`, `.jpg`, `.jpeg`, `.webp`, or `.gif` are indexed directly as visual chunks (`sheet_name = web_images`).
- **HTML & Inline Image Extraction** (`_StdLibHTMLTextExtractor`):
  - Strips `<script>`, `<style>`, `<nav>`, `<footer>`, and `<svg>` blocks
  - Extracts clean prose
  - Resolves relative `<img>` URLs to absolute `https://` links
  - Filters out tiny UI icons and trackers
  - Appends a dedicated `[Image Gallery & Visual Figures]` catalog chunk

---

## 📂 Project Structure

```text
.
├── main.py                   # Primary Streamlit entry point, session bootstrap, and chat loop
├── prompts.py                # System prompts for grounded QA, query rewriting, and follow-up generation
├── excelreader.py            # Phase 1 & 2: Multi-engine Excel/CSV parser and statistical sheet profiler
├── filereader.py             # PDF page extractor and TXT/Markdown recursive text splitters
├── engine/
│   ├── config.py             # Global constants, chunk sizes, batch limits, and fallback model lists
│   ├── hybrid_index.py       # Phase 4 & 5: SQLite B-Tree categorical index, FTS5 BM25, and sheet resolver
│   ├── storage.py            # Phase 3 & 5: ChromaDB client, micro-batch ingester, DOCX/URL parsers, hybrid search
│   └── llm_service.py        # Conversational query rewriter, Groq model discovery, streaming, and failover
├── frontend/
│   ├── components.py         # Sidebar session manager, file scope filters, citation badges, and chat handler
│   └── styles.py             # Custom dark theme CSS, telemetry badges, and status indicators
├── requirements.txt          # Python package dependencies
└── README.md
```

### 🧱 Module Responsibilities

| File | Responsibility |
| :--- | :--- |
| `main.py` | Initializes Streamlit state, discovers persisted sessions, and renders the main workspace |
| `prompts.py` | Stores system instructions for grounded answering, coreference query rewriting, and follow-up suggestions |
| `excelreader.py` | Reads `.xlsx`, `.xls`, and `.csv` tabs, promotes buried headers, computes column statistics, and packs rows |
| `filereader.py` | Extracts page-level text from PDFs and splits plain text or Markdown files |
| `engine/config.py` | Centralizes storage paths, embedding model name, chunk parameters, and Groq fallback models |
| `engine/hybrid_index.py` | Manages SQLite `chunk_metadata` and `chunk_fts`, fuzzy/ordinal sheet matching, and direct sheet retrieval |
| `engine/storage.py` | Handles streaming micro-batch ingestion, DOCX/URL parsing, per-file deletion, and hybrid retrieval fusion |
| `engine/llm_service.py` | Rewrites conversational queries, streams Groq completions with auto-failover, and generates follow-up questions |
| `frontend/components.py` | Renders the sidebar session manager, source scope selector, hybrid attribution cards, and chat interface |
| `frontend/styles.py` | Defines custom CSS rules for the hero header, source cards, and retrieval badges |

---

## 🚀 Getting Started

### 📋 Prerequisites

| Requirement | Purpose |
| :--- | :--- |
| **Python 3.10+** | Application runtime |
| **pip** or **uv** | Dependency installation and virtual environment management |
| **Groq API key** | High-speed LLM chat completion, query rewriting, and follow-up generation |

Check your installed Python version:

```bash
python --version
```

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

> Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your GitHub repository details.

### 2️⃣ Create a virtual environment and install dependencies

**Using standard `venv` and `pip`:**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**Or using `uv`:**

```bash
uv venv
uv pip install -r requirements.txt
```

### 3️⃣ Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

> ⚠️ The repository's `.gitignore` excludes `.env`. **Never commit a real API key.**

### 4️⃣ Launch the Streamlit app

```bash
streamlit run main.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>.

### 5️⃣ Use the app

1. Create or select an isolated session under **Sessions Manager** in the sidebar.
2. Upload one or more `.xlsx`, `.xls`, `.csv`, `.pdf`, `.docx`, `.txt`, or `.md` files — or paste public webpage URLs into **Add Website URLs**.
3. Select **Process documents** (or **Index Website URL(s)**) to stream micro-batches into ChromaDB and SQLite.
4. Optionally open **Filter / Manage Files** to scope retrieval to specific files or remove an individual source.
5. Enter a question in the chat box, for example:
   - *"Tell me about the Horse Species sheet"*
   - *"What is the average weight in the 2nd sheet?"*
6. Expand **Sources & Hybrid Attribution** below the answer to inspect the exact rows/pages and retrieval engine badges, or click one of the **Suggested follow-ups**.

---

## 🧩 Key Architectural Concepts

### 🧬 Dual-layer hybrid indexing (B-Tree + FTS5 + Vector)

Dense vector search excels at conceptual similarity but struggles when a user asks for a specific worksheet tab inside a 20-sheet workbook, or searches for an exact identifier like `NCT02475681`.

By pairing ChromaDB vectors with a WAL-tuned SQLite database (`chunk_metadata` B-Tree indexes + `chunk_fts` BM25 inverted index), the system filters to exact files/sheets in under 1ms and ranks exact keyword matches alongside semantic embeddings.

### 📈 Per-sheet statistical & schema profiling

LLMs only receive `top_k` retrieved chunks in their context window, so they cannot manually sum or average 5,000 spreadsheet rows at query time.

During ingestion, `excelreader.py` pre-computes `count`, `sum`, `avg`, `min`, and `max` for numeric columns and extracts distinct values for categorical columns, embedding them into a dedicated `page=1` Sheet Overview & Column Schema chunk for every tab.

### 🛡️ Collision-free categorical filtering

When multiple files share overlapping terms (for example, `Horse.pdf` with `medicine_name="horse"` and `Comparison.xlsx` with `sheet_name="horse species"`), combining naive substring filters with `$and` produces zero hits.

`detect_categorical_filters()` verifies in SQLite whether `sheet_filter` and `med_filter` actually coexist in the same rows, automatically dropping conflicting secondary filters so sheet-targeted queries never starve.

### 🎯 Guaranteed direct sheet retrieval

When a user asks a broad question like *"Tell me about the XYZ sheet"*, there are no cell-specific keywords to match in FTS5.

Whenever a `sheet_filter` is resolved (via exact, normalized, token-overlap, short-code, or ordinal matching), `fetch_direct_sheet_chunks()` pulls that sheet's `page=1` Overview chunk plus top matching data rows directly from SQLite.

### 🔁 Conversational coreference rewriting

In multi-turn chat, follow-up questions like *"What are the column names in that sheet?"* or *"Compare it with the first tab"* omit the explicit entity or sheet name from Turn 1.

A lightweight gatekeeper (`_needs_contextual_rewrite`) detects pronouns and implicit sheet references in under 0.1ms and invokes a fast LLM rewriter before retrieval runs.

### ⚡ Multi-model rate-limit failover

High-volume RAG contexts can trigger per-minute token (TPM) rate limits on cloud LLM APIs.

`stream_with_auto_fallback()` configures each candidate Groq model with `max_retries=0` and automatically fails over across a prioritized pool of active models in under 0.1s whenever a `429`, `413`, `404`, or `503` error occurs.

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Application runtime |
| **UI** | Streamlit | Multi-session workspace, streaming chat, and telemetry badges |
| **Tabular engine** | Polars + FastExcel (Calamine) + Pandas | Multi-engine Excel/CSV parsing, header promotion, and vectorized row slicing |
| **Vector database** | ChromaDB (`PersistentClient`) | Persistent dense vector storage, cosine similarity, and MMR search |
| **Hybrid keyword & metadata DB** | SQLite3 (WAL + FTS5 `unicode61`) | O(log N) categorical B-Tree filtering, BM25 keyword search, and inventory manifest |
| **Embeddings** | HuggingFace (`sentence-transformers`) | Local normalized dense vector embeddings |
| **LLM provider** | Groq (`ChatGroq`) | Conversational query rewriting, grounded streaming synthesis, and follow-up generation |
| **Document & web parsers** | `pypdf`, `zipfile` + `ElementTree`, `HTMLParser` | Zero-dependency DOCX paragraph/table extraction, PDF parsing, and HTML/image scraping |
| **Configuration** | `python-dotenv` | Loads `GROQ_API_KEY` and model preferences from `.env` |

---

## 🎛️ Configuration

Core pipeline parameters are centralized in `engine/config.py` and `excelreader.py`:

| Setting | Default value | Location |
| :--- | :--- | :--- |
| Default chat model | `llama-3.3-70b-versatile` (with auto-failover) | `engine/config.py`, `engine/llm_service.py` |
| Local embedding model | `EMBEDDING_MODEL` (HuggingFace) | `engine/config.py`, `engine/storage.py` |
| Document chunk size | `CHUNK_SIZE` characters | `engine/config.py` |
| Document chunk overlap | `CHUNK_OVERLAP` characters | `engine/config.py` |
| Excel row pack limit | Up to 6 rows / 950 chars per chunk | `excelreader.py` |
| Excel sheet safety caps | 20,000 rows / 80 cols per sheet | `excelreader.py` |
| Streaming micro-batch size | `MICRO_BATCH_SIZE` chunks | `engine/config.py` |
| Retrieved passages (`top_k`) | Configurable 2–12 in sidebar | `frontend/components.py` |
| Conversation history window | `HISTORY_TURNS` turns | `engine/config.py` |

---

## 📐 Design Decisions

| Choice | Why it is useful here |
| :--- | :--- |
| **3-stage Excel reader** (`fastexcel` → `polars` → `pandas`) | Guarantees that a single malformed worksheet tab never causes the rest of a multi-sheet workbook to be skipped |
| **Dedicated `page=1` Sheet Overview chunk per tab** | Ensures broad sheet questions (*"Summarize Sheet X"*) and aggregation queries (*"What is the average of Column Y?"*) always retrieve complete schema and statistical context |
| **Atomic file-boundary micro-batch commits** | Enables exact sub-1ms checkpoint auto-resume if a 1,500-file ingestion job is interrupted |
| **Zero-disk-I/O sidebar label resolution** (`_peek_bucket`) | Computes session names and document counts purely from `st.session_state` in RAM to avoid Windows SQLite lock contention |
| **O(log N) native `where` metadata deletion** | Allows users to remove a single file from a 500,000-chunk session without loading vectors into memory |
| **Zero-token SQLite inventory fallback** | Answers file-count and sheet-name inventory questions directly from SQLite if all LLM models hit API rate limits |

---

## ⚠️ Limitations

- **Scanned PDFs** and image-only documents require external OCR; `pypdf` extracts embedded digital text only.
- **Extremely large Excel sheets** are capped at 20,000 rows and 80 columns per tab (`MAX_ROWS_PER_SHEET`, `MAX_COLS_PER_SHEET`) to maintain predictable memory usage.
- **Numeric column statistics** sample up to the first 2,000 non-null numeric rows per column for ingestion speed.
- **Live URL scraping** uses `requests` and Python's standard `HTMLParser`; single-page applications (SPAs) that require headless browser JavaScript rendering or login authentication are not supported.
- **Generated answers** should always be verified against the expandable source citation cards for clinical, financial, or legal workflows.

---

## 💡 Ideas for Further Development

- [ ] Add optical character recognition (OCR) support (e.g., RapidOCR or Tesseract) for scanned PDFs and embedded chart images.
- [ ] Add natural-language-to-SQL / Polars DataFrame execution for arbitrary multi-column group-by and join queries across sheets.
- [ ] Support cross-encoder reranking (e.g., `ms-marco-MiniLM` or `bge-reranker`) on top of the fused hybrid candidates.
- [ ] Add automated RAG evaluation benchmarks (Ragas / TruLens) for context precision, recall, and faithfulness.
- [ ] Add role-based authentication and cloud object storage (S3 / GCS) for multi-tenant deployments.

---

## 🔒 Security & Privacy

- 🚫 Never commit `.env`, API keys, or sensitive local databases (`chroma_db/`) to version control.
- 🏠 Embeddings and hybrid indexing run **100% locally** on your machine; only the retrieved context passages and recent chat turns are sent to Groq's API during question answering.
- 📜 Review your LLM provider's data-retention and compliance policies before querying confidential or regulated data.

---

## 🤝 Contributing

Contributions, bug reports, and retrieval enhancements are welcome.

1. **Fork** the repository.
2. **Create a feature branch**: `git checkout -b feat/your-feature-name`
3. **Make and test** your changes across multi-sheet Excel, PDF, and URL inputs.
4. **Open a pull request** describing what changed and why.

---

## 📄 License

Released under the **MIT License**.
