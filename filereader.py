"""
Modular File Reader for PDF, TXT, and Markdown (.md) files.
Supports both local file paths (for CLI scripts) and uploaded byte streams (for Streamlit).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# --------------------------------------------------------------------------- #
# 1. Local Path Readers (Backwards-compatible with your CLI scripts)
# --------------------------------------------------------------------------- #
def read_pdf_text(pdf_path: str | Path) -> str:
    """Read and return the text content of a local PDF file."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts)


def read_multiple_pdfs(pdf_paths: List[str | Path]) -> List[Document]:
    """Read text from multiple local PDF files and return a list of LangChain Documents."""
    docs: List[Document] = []
    for path in pdf_paths:
        path_obj = Path(path)
        text = read_pdf_text(path_obj)
        docs.append(Document(page_content=text, metadata={"source": str(path_obj)}))
    return docs


# --------------------------------------------------------------------------- #
# 2. Enterprise Byte-Stream Readers (Used by the Streamlit RAG Pipeline)
# --------------------------------------------------------------------------- #
def parse_pdf_bytes(
    file_name: str,
    raw_data: bytes,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Extract text per PDF page, handle encryption edge cases, prepend document context,
    and split into semantic chunks.
    """
    try:
        reader = PdfReader(io.BytesIO(raw_data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF is password-protected.")

        page_docs: List[Document] = []
        entity_name = Path(file_name).stem.lower().strip()

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

    if not page_docs:
        raise ValueError("no readable text found (it may be a scanned/image-only PDF).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(page_docs)


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
