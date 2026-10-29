"""
Enterprise Multi-Modal File Reader for PDF (Text + Tables + Images + Scanned OCR), TXT, and Markdown (.md).
Used by the Streamlit Hybrid RAG Pipeline (`engine/storage.py`).

What was updated:
1. Structured Table Extraction: Detects tables on each PDF page via PyMuPDF (`page.find_tables()`)
   and formats them as pipe-delimited Markdown rows (`Col1 | Col2 | Col3`).
2. Scanned PDF OCR (Zero-Install Groq Vision + Local RapidOCR fallback): When a page has <30 chars
   of digital text, renders the page to an image and extracts text/tables via OCR.
3. Embedded PDF Image Extraction: Extracts figures/diagrams from PDF pages, saves them under
   `./chroma_db/pdf_images/`, and creates inline `[Image: ...]` markers + an Image Gallery chunk.
4. Removed unused legacy CLI functions (`read_pdf_text`, `read_multiple_pdfs`).
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
from pathlib import Path
from typing import List

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from engine.config import BASE_CHROMA_DIR

# Optional high-speed PyMuPDF (fitz) import for Tables, Images & Scanned Page Rendering
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Optional local offline OCR fallback (pip install rapidocr-onnxruntime)
try:
    from rapidocr_onnxruntime import RapidOCR
    _LOCAL_OCR = RapidOCR()
except ImportError:
    _LOCAL_OCR = None

PDF_IMAGES_DIR = os.path.join(BASE_CHROMA_DIR, "pdf_images")
MAX_OCR_PAGES_PER_PDF = 25  # Safety cap so 500-page scanned books don't exhaust API quota


# --------------------------------------------------------------------------- #
# Helper 1: Scanned PDF Page OCR (Groq Vision API + Local RapidOCR Fallback)
# --------------------------------------------------------------------------- #
def _ocr_scanned_page_png(png_bytes: bytes, file_name: str, page_number: int) -> str:
    """
    Extract text and tables from a scanned PDF page image.
    1. Tries Groq Vision (`meta-llama/llama-4-scout-17b-16e-instruct`) using your existing GROQ_API_KEY
       (supports English, Hindi, and structured tables with zero heavy local software required).
    2. Falls back to local `rapidocr_onnxruntime` if installed and API is unreachable.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key:
        b64_img = base64.b64encode(png_bytes).decode("utf-8")
        vision_models = [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
        ]
        for model_id in vision_models:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "temperature": 0.0,
                        "max_tokens": 1500,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Transcribe all readable text and tables from Page {page_number} of '{file_name}'. "
                                            "Format any tables using pipe-separated columns (Col1 | Col2 | Col3). "
                                            "Preserve original language (including English, Hindi, numbers, and clinical terms). "
                                            "Output ONLY the transcribed text without commentary."
                                        ),
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                                    },
                                ],
                            }
                        ],
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    ocr_text = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    if ocr_text:
                        return f"[Scanned Page OCR Transcription]\n{ocr_text}"
            except Exception:
                continue

    # Fallback: Local RapidOCR (if installed via `pip install rapidocr-onnxruntime`)
    if _LOCAL_OCR is not None:
        try:
            result, _ = _LOCAL_OCR(png_bytes)
            if result:
                lines = [str(item[1]).strip() for item in result if len(item) >= 2 and item[1]]
                if lines:
                    return "[Scanned Page Local OCR]\n" + "\n".join(lines)
        except Exception:
            pass

    return ""


# --------------------------------------------------------------------------- #
# Helper 2: Structured PDF Table Extractor (PyMuPDF `find_tables`)
# --------------------------------------------------------------------------- #
def _extract_tables_from_fitz_page(page: "fitz.Page", page_number: int) -> str:
    """
    Detect tables on a PDF page and convert them into structured pipe-delimited rows
    so RAG preserves row/column relationships (similar to `excelreader.py`).
    """
    table_blocks: List[str] = []
    try:
        finder = page.find_tables()
        tables = finder.tables if finder and hasattr(finder, "tables") else []
        for t_idx, tbl in enumerate(tables, start=1):
            raw_rows = tbl.extract()
            if not raw_rows or len(raw_rows) < 2:
                continue

            formatted_rows: List[str] = []
            headers: List[str] = []

            for r_idx, row in enumerate(raw_rows):
                if not row:
                    continue
                cleaned_cells = [
                    " ".join(str(cell or "").split()).strip()
                    for cell in row
                ]
                # Skip 100% empty rows
                if not any(cleaned_cells):
                    continue

                if not headers:
                    headers = [c if c else f"Col_{i+1}" for i, c in enumerate(cleaned_cells)]
                    formatted_rows.append(" | ".join(headers))
                else:
                    # Pair each cell with its column header for higher semantic search accuracy
                    paired = [
                        f"{headers[i] if i < len(headers) else f'Col_{i+1}'}: {val}"
                        for i, val in enumerate(cleaned_cells)
                        if val
                    ]
                    if paired:
                        formatted_rows.append(f"[Table {t_idx} Row {r_idx}] " + " | ".join(paired))

            if len(formatted_rows) >= 2:
                table_blocks.append(
                    f"[Structured Table {t_idx} on Page {page_number}]\n"
                    + "\n".join(formatted_rows)
                )
    except Exception:
        pass

    return "\n\n".join(table_blocks)


# --------------------------------------------------------------------------- #
# Helper 3: Embedded PDF Image & Diagram Extractor
# --------------------------------------------------------------------------- #
def _extract_images_from_fitz_page(
    doc: "fitz.Document",
    page: "fitz.Page",
    file_name: str,
    page_number: int,
    max_images_per_page: int = 4,
) -> List[dict]:
    """
    Extract meaningful embedded figures/diagrams from a PDF page.
    Filters out tiny icons/lines (<80x80 px) and saves valid images to `./chroma_db/pdf_images/`.
    """
    extracted: List[dict] = []
    try:
        image_list = page.get_images(full=True)
        if not image_list:
            return []

        os.makedirs(PDF_IMAGES_DIR, exist_ok=True)
        safe_stem = Path(file_name).stem[:24].replace(" ", "_")

        for img_idx, img_info in enumerate(image_list[:max_images_per_page], start=1):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            if not base_image:
                continue

            width = int(base_image.get("width", 0))
            height = int(base_image.get("height", 0))
            img_bytes = base_image.get("image", b"")
            ext = base_image.get("ext", "png")

            # Filter out tiny decorative lines, bullets, or 1x1 tracking pixels
            if width < 80 or height < 80 or len(img_bytes) < 2048:
                continue

            img_hash = hashlib.md5(img_bytes).hexdigest()[:10]
            img_filename = f"{safe_stem}_p{page_number}_fig{img_idx}_{img_hash}.{ext}"
            disk_path = os.path.join(PDF_IMAGES_DIR, img_filename)

            if not os.path.exists(disk_path):
                with open(disk_path, "wb") as f:
                    f.write(img_bytes)

            desc = f"Figure {img_idx} from {file_name} (Page {page_number}, {width}x{height}px)"
            extracted.append(
                {
                    "alt": desc,
                    "url": disk_path.replace("\\", "/"),
                    "page": page_number,
                }
            )
    except Exception:
        pass

    return extracted


# --------------------------------------------------------------------------- #
# Main Enterprise PDF Parser (Text + Tables + OCR + Embedded Images)
# --------------------------------------------------------------------------- #
def parse_pdf_bytes(
    file_name: str,
    raw_data: bytes,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Enterprise Multi-Modal PDF Parser:
    1. Uses PyMuPDF (`fitz`) when available to extract digital text, structured tables,
       embedded figures/diagrams, and scanned page OCR via Groq Vision / RapidOCR.
    2. Automatically falls back to `pypdf` if PyMuPDF is not installed.
    """
    entity_name = Path(file_name).stem.lower().strip()
    page_docs: List[Document] = []
    all_extracted_images: List[dict] = []
    ocr_pages_used = 0

    # ----------------------------------------------------------------------- #
    # PATH A: High-Precision PyMuPDF Engine (Text + Tables + Images + OCR)
    # ----------------------------------------------------------------------- #
    if HAS_PYMUPDF:
        try:
            pdf_doc = fitz.open(stream=raw_data, filetype="pdf")
            if pdf_doc.is_encrypted and not pdf_doc.authenticate(""):
                raise ValueError("PDF is password-protected.")

            for page_idx in range(len(pdf_doc)):
                page_number = page_idx + 1
                page = pdf_doc[page_idx]

                # 1. Extract standard digital text (UTF-8 / Hindi / Multilingual ready)
                try:
                    raw_text = (page.get_text("text") or "").strip()
                except Exception:
                    raw_text = ""

                # 2. Extract structured tables on this page
                table_text = _extract_tables_from_fitz_page(page, page_number)

                # 3. Extract embedded figures/diagrams on this page
                page_images = (
                    _extract_images_from_fitz_page(pdf_doc, page, file_name, page_number)
                    if len(all_extracted_images) < 30
                    else []
                )
                all_extracted_images.extend(page_images)
                image_markers = "\n".join(
                    f"[Image: {img['alt']} | Image URL: {img['url']}]"
                    for img in page_images
                )

                # 4. Scanned Page Detection & OCR Fallback:
                # If digital text + table text is < 30 chars, treat page as a scanned image!
                ocr_text = ""
                if len(raw_text) + len(table_text) < 30 and ocr_pages_used < MAX_OCR_PAGES_PER_PDF:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        png_bytes = pix.tobytes("png")
                        ocr_text = _ocr_scanned_page_png(png_bytes, file_name, page_number)
                        if ocr_text:
                            ocr_pages_used += 1
                    except Exception:
                        ocr_text = ""

                # Combine all signals for this page
                combined_parts = [
                    part for part in (raw_text, table_text, ocr_text, image_markers) if part
                ]
                full_page_body = "\n\n".join(combined_parts).strip()

                if full_page_body:
                    page_docs.append(
                        Document(
                            page_content=f"[Document: {file_name} | Page {page_number}]\n{full_page_body}",
                            metadata={
                                "source": file_name,
                                "file_name": file_name,
                                "sheet_name": f"page_{page_number}",
                                "medicine_name": entity_name,
                                "page": page_number,
                                "file_type": "pdf",
                            },
                        )
                    )

            pdf_doc.close()
        except ValueError:
            raise
        except Exception as exc:
            # If PyMuPDF encounters an unusual stream error, let it fall through to pypdf below
            if not page_docs:
                pass
            else:
                raise ValueError(f"could not read PDF ({exc}).") from exc

    # ----------------------------------------------------------------------- #
    # PATH B: Fallback `pypdf` Engine (Used if PyMuPDF is not installed)
    # ----------------------------------------------------------------------- #
    if not page_docs:
        try:
            reader = PdfReader(io.BytesIO(raw_data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("PDF is password-protected.")

            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = (page.extract_text() or "").strip()
                except Exception:
                    page_text = ""

                if page_text:
                    page_docs.append(
                        Document(
                            page_content=f"[Document: {file_name} | Page {page_number}]\n{page_text}",
                            metadata={
                                "source": file_name,
                                "file_name": file_name,
                                "sheet_name": f"page_{page_number}",
                                "medicine_name": entity_name,
                                "page": page_number,
                                "file_type": "pdf",
                            },
                        )
                    )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"could not read PDF ({exc}).") from exc

    if not page_docs and not all_extracted_images:
        hint = (
            "Ensure `GROQ_API_KEY` is set or install `rapidocr-onnxruntime` for scanned PDF OCR."
            if HAS_PYMUPDF
            else "Run `pip install pymupdf` to enable Scanned PDF OCR, Table Extraction, and Image Extraction."
        )
        raise ValueError(f"no readable text found in '{file_name}'. {hint}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_chunks = splitter.split_documents(page_docs) if page_docs else []

    # Append a dedicated PDF Image Catalog chunk if figures/diagrams were extracted
    if all_extracted_images:
        gallery_lines = [
            f"- [Image: {img['alt']} | Image URL: {img['url']}]"
            for img in all_extracted_images[:20]
        ]
        gallery_chunk = (
            f"[Document: {file_name} | PDF Figures & Visual Diagrams Catalog]\n"
            f"Figures, charts, diagrams, and illustrations extracted from {file_name}:\n"
            + "\n".join(gallery_lines)
        )
        split_chunks.append(
            Document(
                page_content=gallery_chunk,
                metadata={
                    "source": file_name,
                    "file_name": file_name,
                    "sheet_name": "pdf_images",
                    "medicine_name": entity_name,
                    "page": 1,
                    "file_type": "pdf",
                },
            )
        )

    return split_chunks


# --------------------------------------------------------------------------- #
# Plain Text (.txt) & Markdown (.md) Reader
# --------------------------------------------------------------------------- #
def parse_text_or_md_bytes(
    file_name: str,
    raw_data: bytes,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Decode UTF-8 .txt or .md files and split into chunks with standardized metadata."""
    text = raw_data.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("no readable text found.")

    entity_name = Path(file_name).stem.lower().strip()
    base_doc = Document(
        page_content=f"[Document: {file_name}]\n{text}",
        metadata={
            "source": file_name,
            "file_name": file_name,
            "sheet_name": "document",
            "medicine_name": entity_name,
            "page": 1,
            "file_type": "text",
        },
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents([base_doc])
