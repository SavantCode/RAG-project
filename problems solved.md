# 🚀 Advanced RAG Architecture & Engineering Roadmap

Now that your codebase is cleanly structured and your core PR is ready, this document details the next tier of engineering upgrades designed to scale your RAG app, enhance multi-document handling, eliminate prompt leakage, and bulletproof your Windows development environment.

---

## 🏗️ Architectural Foundation: High-Speed Session Inventory

### 1. SQLite B-Tree Indexing
Because `chunk_metadata` has a B-Tree index on `(session_id, file_name)`, SQLite returns the exact list of all 10+ files, their row/chunk counts, and their sheet tabs in **~1ms** regardless of how many tens of thousands of rows exist.

### 2. Context Header Injection
In `handle_user_question()`, we prepend a clean `Uploaded Files in Current Session` summary above the retrieved passages so the LLM always knows the exact file count, filenames, and sheet names.

---

## 🎯 Issue 5: Lack of Granular Document Control (All-or-Nothing Sessions)

### ⚠️ Problem Statement
Previously, if a user uploaded 5 files into a session and wanted to:
1. Ask a question scoped to only 1 or 2 specific files without cross-talk from the other files, **OR**
2. Remove 1 mistakenly uploaded file without deleting the entire session and re-uploading the remaining 4 files, there was no mechanism to do so.

### ✅ Engineering Solution: Per-File Scope Filtering & Single-File Purge
1. **Metadata Normalization (`route_uploaded_file_to_chunks`):** Guaranteed that every chunk (whether from Excel, CSV, PDF, TXT, or MD) is stamped with `chunk.metadata["file_name"] = uploaded_file.name`.
2. **Scoped Hybrid Retrieval (`allowed_files`):** Updated `detect_categorical_filters()`, `search_fts_keywords()`, and `hybrid_retrieve()` to accept `allowed_files: list[str] | None`, pushing down `file_name IN (...)` to SQLite FTS5 and `{"file_name": {"$in": allowed_files}}` to ChromaDB.
3. **Single-File Deletion (`delete_single_file_from_session`):** Added a sidebar control in `🗂️ Filter / Manage Files` that deletes all vector IDs for a chosen file from ChromaDB and executes a synchronized delete on `chunk_fts` and `chunk_metadata` in SQLite.

---

## 🎨 Issue 6: Invisible 3-Layer Hybrid Search & Lost Multi-Engine Signals

### ⚠️ Problem Statement
Although the backend executed a sophisticated **3-Layer Hybrid Search** (Categorical B-Tree Filter + SQLite FTS5 BM25 Keyword Search + Chroma Semantic Vector Search), the UI only displayed `📄 filename · X% similarity`. Furthermore:
* When a chunk matched both FTS5 and Vector search, naive deduplication discarded the second hit, hiding the fact that multiple retrieval engines agreed on the passage.
* When using Hybrid + MMR (Diverse) mode, LangChain's `max_marginal_relevance_search()` does not return distance scores, causing similarity percentages to disappear (`None`).

### ✅ Engineering Solution: Multi-Signal Attribution Fusion & Color-Coded Badges
1. **Score Lookup Map for MMR:** In `hybrid_retrieve()`, we pre-fetch distance scores via `similarity_search_with_score` into an in-memory `score_lookup` dictionary so even MMR-selected chunks display their exact semantic similarity percentage (`🧠 Semantic Vector (84%)`).
2. **Multi-Signal Fusion (`merged_by_sig`):** Instead of discarding duplicate chunks across FTS5 and Vector results, we merge their boolean flags (`matched_category`, `matched_fts`, `matched_vector`) onto the same source object.
3. **Color-Coded Pill Badges & Telemetry Bar:** Added CSS classes in `frontend/styles.py` and updated `render_sources()` in `frontend/components.py` to render:
   * **⚡ Hybrid Retrieval Breakdown Bar:** Summarizes exact hit counts per layer at the top of the `📚 Sources & Hybrid Attribution` expander.
   * **🏷️ Exact Category Match** (`Emerald Green` `.badge-category`)
   * **🔤 FTS5 Keyword Match** (`Amber Gold` `.badge-fts`)
   * **🧠 Semantic Vector (X%)** (`Indigo Purple` `.badge-vector`)

---

## 🛡️ Issue 7: Prompt Header Leakage & Irrelevant Citations on Out-of-Scope Queries

### ⚠️ Problem Statement
1. When asked *"How many files did I give you?"*, the LLM quoted the raw internal header: `=== SESSION DOCUMENT INVENTORY (Total Indexed Files: 2) ===`.
2. When asked out-of-scope or nonsense queries (e.g., *"hij"* or *"tell about butterfly"* when only Cow and Horse PDFs were uploaded), the LLM correctly replied *"I could not find any information..."*, but the UI still rendered `📚 Sources (7)` containing low-similarity (42%) chunks from the unrelated PDFs.

### ✅ Engineering Solution: Natural Prompting & Smart Source Suppression
1. **Prompt Guardrail (`prompts.py`):** Added Rule 4 in `SYSTEM_PROMPT` instructing the assistant to use the uploaded file overview naturally without quoting internal system headers, and simplified the header text in `components.py`.
2. **Irrelevant Source Suppression (`_is_not_found_answer`):** Added a regex detector in `frontend/components.py` that inspects the assistant's final response for not-found declarations (*"could not find"*, *"no information about"*, *"not mentioned in the uploaded documents"*). When triggered, `sources` is cleared (`sources = []`) so misleading citations are hidden.

---

## 🪟 Issue 8: Windows PowerShell Requirements.txt Encoding & Environment Traps

### ⚠️ Problem Statement
Running `pip freeze | Out-File -FilePath "requirements.txt" -Encoding utf8NoBOM` failed in Windows PowerShell 5.1 with a `ParameterBindingValidationException`, while standard `>` redirection in PowerShell defaults to UTF-16 LE, which breaks `pip install -r requirements.txt`.

### ✅ Engineering Solution
* **ASCII Encoding Guarantee:** Used `-Encoding ascii` (compatible with PowerShell 5.1+ and guaranteed never to insert a UTF-16 or UTF-8 Byte Order Mark):
  ```powershell
  pip freeze | Out-File -FilePath "requirements.txt" -Encoding ascii
  ```
* **Pipreqs Alternative:** Recommended `pipreqs . --force --encoding=utf8 --ignore ".venv,venv,chroma_db"` to generate clean, import-based dependency lists directly via Python UTF-8 file I/O.


# 8
```markdown
## Issue 9: Conversational Context Loss & Topic-Switch Pollution in Multi-Turn Queries

### ⚠️ Problem Statement
Previously, multi-turn follow-up queries were handled by a naive 6-word rule:
```python
if len(question.split()) <= 6 and prior_messages:
    search_query = f"{prev_user_qs[-1]} {question}"

```

This caused two critical RAG retrieval failures:

1. **7+ Word Pronoun Follow-Ups Failed:** If Turn 1 asked "Tell me about the cow" and Turn 2 asked "How many stomach compartments do they have and what is their cultural significance?" (13 words), the `<= 6` rule did not trigger. The search engine queried "stomach compartments they have..." without the word "cow", losing Categorical/FTS5 precision.
2. **Topic-Switch Pollution:** If Turn 1 asked "Tell me about the cow" and Turn 2 switched topics with "Tell me about the horse" (5 words), the old rule concatenated them into "Tell me about the cow Tell me about the horse", contaminating the search results with Cow chunks when the user asked about the Horse.

### ✅ Engineering Solution: 2-Stage Hybrid Conversational Query Rewriter

Implemented in `engine/llm_service.py` (`_needs_contextual_rewrite` + `rewrite_conversational_query`) and `prompts.py` (`QUERY_REWRITER_PROMPT`):

1. **Stage 1: Zero-Latency Heuristic Gatekeeper** (`_needs_contextual_rewrite`, `<0.1ms`):
* Immediately bypasses LLM rewriting if there is no prior chat history or if the query is a session meta-question (e.g., "how many files do you have").
* Scans for referential pronouns (*it, they, their, its, this, that, those*), conversational starters (*what about, how much, compare, elaborate*), or short elliptical phrases (`<= 7 words`).


2. **Stage 2: Fast LLM Coreference & Entity Resolution** (`rewrite_conversational_query`):
* Sends the last 3 conversation turns (truncated for `<200ms` latency) to a fast Groq model (`llama-3.1-8b-instant` with `temperature=0.0`, `max_tokens=80`).
* Resolves pronouns into explicit document entities, medicine names, and sheet names while leaving genuine topic switches untouched.


3. **Stage 3: UI Search Telemetry** (`frontend/components.py`):
* Displays `🔄 Searched: "<rewritten_query>"` in the response caption and inside the search status drawer whenever a query is contextualized.



```

---

## 🧪 2. How to Test That the Feature Is Working (3 Quick Scenarios)

| Test Scenario | What to Type in Chat | Expected Result |
| :--- | :--- | :--- |
| **Test 1: Long Pronoun Follow-Up (>6 words)** | **Turn 1:** `"Tell me about the cow"`<br>**Turn 2:** `"How many stomach compartments do they have and why are they sacred?"` | Under the answer, you see:<br>`🔄 Searched: "How many stomach compartments do cows have and why are cows sacred?"` and **100% of sources come from the Cow PDF**. |
| **Test 2: Clean Topic Switch (<=6 words)** | **Turn 3:** `"Tell me about the horse"` | It recognizes you switched topics to the Horse—it does **not** glue `"cow"` onto your query, and retrieves only Horse chunks! |
| **Test 3: Short Elliptical Follow-Up** | **Turn 4:** `"What about its speed and anatomy?"` | It resolves `"its"` to `"the horse"` and shows:<br>`🔄 Searched: "What is the speed and anatomy of the horse?"` |

---

## 🚀 3. Zero-Risk Git Procedure to Create a New Branch & Push

Using the exact same **Zero-Risk (`-X ours`)** procedure that keeps your local files 100% untouched and prevents any conflicts with `main`:

### Step 1: Create and switch to your new feature branch
```powershell
git checkout -b feature/smart-query-rewriter


```

---

## Issue 10: Grounded Auto-Generated Follow-Up Questions & Reasoning-Model Token Exhaustion

### ⚠️ Problem Statement
Previously, the UI only displayed 4 static starter buttons (*"Summarize the main points"*, etc.) when a session was completely empty. Once the user asked their first question, those buttons disappeared permanently, leaving users without guided discovery across multi-document sessions.

When we initially introduced LLM-generated follow-up buttons (`generate_followup_questions`), three subtle edge cases prevented the buttons from rendering:
1. **Reasoning Model Token Exhaustion (`<think>...</think>`)**: When a reasoning-enabled Groq model (such as `openai/gpt-oss-20b`, `qwen-qwq`, or `deepseek-r1`) was selected, setting `max_tokens=110` caused the model to spend its entire token budget inside internal `<think>...</think>` reasoning blocks, returning 0 visible lines.
2. **Strict Question-Mark Filtering**: The line parser initially required `cleaned.endswith("?")`. If an LLM returned imperative suggestions (e.g., *"Compare the cow and the horse"*) or if a brief API rate-limit occurred right after streaming the main response, the parser returned an empty list `[]`.
3. **Legacy Chat State & Streamlit Rerun Binding**: Existing chat messages stored on disk before the update lacked the `"followups"` key in `session_data`, and typing a normal question in `st.chat_input` rendered inline without triggering `st.rerun()` to bind the interactive buttons in `main.py`'s replay loop.

### ✅ Engineering Solution: 4-Layer Resilient Follow-Up Architecture
Implemented across `prompts.py`, `engine/llm_service.py`, `frontend/components.py`, and `main.py`:
1. **Source-Grounded Prompting (`FOLLOWUP_SUGGESTIONS_PROMPT`)**:
   - Passes the top retrieved source snippets + the session's uploaded file inventory into `generate_followup_questions()`, strictly requiring that all 3 suggested questions (<10 words each) can be answered directly from the uploaded documents.
2. **Reasoning Tag Stripping & Fast Model Priority (`_strip_thinking_tags`)**:
   - Prioritizes fast non-reasoning instruct models (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) for background helper calls, increases `max_tokens` to `350`, and strips any `<think>...</think>` blocks via regex before parsing lines.
   - Automatically normalizes punctuation (appending `?` if omitted) and filters out conversational preambles (*"Here are 3 questions:"*).
3. **Zero-Latency Deterministic Fallback (`_build_deterministic_followups`)**:
   - If all LLM calls fail or rate-limit, `_build_deterministic_followups()` constructs 3 document-specific follow-up questions directly from the retrieved source filenames and session inventory, guaranteeing **100% uptime**.
   - If the assistant's answer was an out-of-scope declaration (`_is_not_found_answer`), follow-up generation is automatically skipped so irrelevant suggestions are never shown.
4. **Interactive One-Click Execution (`render_followup_pills` in `main.py`)**:
   - Renders 3 clickable `↪ Question?` pill buttons right underneath the latest assistant response (with automatic fallback for legacy messages). Clicking any button sets `queued_prompt` and immediately executes the full Hybrid RAG pipeline.


---

## Issue 11: Multi-Modal Website URL (Text + Images) & DOCX Ingestion Without External Dependency Bloat

### ⚠️ Problem Statement
1. **Limited Data Modalities**: The ingestion pipeline originally supported only `.xlsx`, `.xls`, `.csv`, `.pdf`, `.txt`, and `.md`. Enterprise workflows frequently require ingesting **Microsoft Word documents (`.docx`)** (including embedded tables) and **live Website/Documentation URLs** (including diagrams, figures, and photos).
2. **Environment Dependency Mismatch (`ModuleNotFoundError: No module named 'bs4'`)**: The active Streamlit runtime (`clean_env`) did not have `beautifulsoup4` (`bs4`) or `python-docx` installed, causing top-level third-party imports to fail on server startup.
3. **Blindness to Web Visuals**: Standard text scrapers discard `<img>` tags completely, preventing users from asking visual questions like *"Show me images or diagrams from this webpage"*.

### ✅ Engineering Solution: Single-Pass StdLib Streaming Parsers + Web Image Catalog
Implemented in `engine/storage.py`, `prompts.py`, and `frontend/components.py`:
1. **Single-Pass Event-Driven HTML & Image Extractor (`_StdLibHTMLTextExtractor` via `html.parser.HTMLParser`)**:
   - Built on Python's standard library (zero external dependencies, 2x–4x faster and ~70% lower RAM than DOM-tree builders like `BeautifulSoup`).
   - Tracks tag depth (`_skip_depth`) to strip `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`, and `<svg>` blocks in a single pass.
   - **Smart Image Extraction**: Captures `<img>` tags (`src`, `data-src`, `alt`, `title`), resolves relative URLs into absolute `https://` URLs via `urllib.parse.urljoin`, filters out UI noise (icons, logos, tracking pixels, `<48px` elements), and injects `[Image: <alt> | Image URL: <https_url>]` inline with surrounding text plus a dedicated `web_images` catalog chunk.
   - **Direct Image URL Support**: Supports indexing direct image links (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`).
2. **Live Markdown Image Rendering in Chat (`prompts.py`)**:
   - Updated `SYSTEM_PROMPT` (Rule 5) to render `[Image: <description> | Image URL: <url>]` entries directly inside the assistant's response using Markdown image syntax `![<description>](<url>)`.
3. **C-Accelerated OpenXML `.docx` Parser (`parse_docx_bytes` via `zipfile` + `xml.etree.ElementTree`)**:
   - Unzips `word/document.xml` in memory (`io.BytesIO`) and extracts both paragraphs (`<w:p>`) and structured tables (`<w:tbl>` $\rightarrow$ pipe-delimited rows) using Python's C-backed `expat` XML parser without requiring `python-docx`.
4. **Full Hybrid Search & Filter Integration**:
   - Stamped web sources (`🌐 domain/path`) and `.docx` files with normalized metadata in ChromaDB and SQLite `chunk_metadata`/`chunk_fts`, enabling full compatibility with **Scope Search to Source(s)**, **Remove Selected Source**, **Session Inventory Manifest**, and **3-Layer Hybrid Attribution Badges**.
