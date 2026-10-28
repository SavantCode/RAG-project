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