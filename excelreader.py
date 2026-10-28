"""
Phase 1 & Phase 2: Enterprise Excel & CSV Processing Module.
- Phase 1 (Reading Engine): Uses Polars + Rust Calamine (fastexcel) to read all tabs
  in multi-sheet Excel workbooks at Rust speed with minimal RAM overhead.
- Phase 2 (Schema-Agnostic Slicing): Dynamically adapts to any sheet's column headers,
  slices strictly 1 row = 1 independent Document block, and prepends global context:
  [Workbook Filename | Sheet Tab Name | Index Medicine Name | Row #].
- Edge Cases Handled:
  * Multi-row title banners (auto-promotes real header row)
  * Merged parent category cells in Column A (forward-filled automatically)
  * Empty sheets, NaN/null cells, duplicate column names, and legacy .xls fallback
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, List

import fastexcel
import pandas as pd
import polars as pl
from langchain_core.documents import Document


def _infer_index_medicine_name(file_name: str, df: pl.DataFrame) -> str:
    """
    Determine the global Index Medicine / Entity Name for Context Prepending.
    1. Checks if the sheet has an explicit drug/medicine/molecule column with a single value.
    2. Otherwise extracts a clean medicine/entity name from the workbook filename.
    """
    med_keywords = (
        "medicine", "drug", "molecule", "compound",
        "generic_name", "brand_name", "product", "agent", "therapy",
    )
    for col in df.columns:
        if any(k in str(col).lower() for k in med_keywords):
            non_null = [
                str(v).strip()
                for v in df.get_column(col).drop_nulls().to_list()
                if v is not None and str(v).strip() not in ("", "None", "nan", "null")
            ]
            unique_vals = list(dict.fromkeys(non_null))
            if len(unique_vals) == 1:
                return unique_vals[0]

    # Fallback: Clean workbook filename (e.g., "Abatacept_Duplications_v2.xlsx" -> "Abatacept Duplications v2")
    stem = Path(file_name).stem
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    return cleaned or file_name


def _promote_real_header_row(df: pl.DataFrame) -> pl.DataFrame:
    """
    Edge-Case Handler: Many clinical/enterprise spreadsheets have title banners in Rows 1-3,
    causing the reader to assign 'Unnamed: 0' or '__UNNAMED__1' as column names.
    Scans the first 10 rows to locate and promote the true table header row.
    """
    if df.is_empty():
        return df

    unnamed_count = sum(
        1 for c in df.columns
        if str(c).lower().startswith(("unnamed", "__unnamed", "column_"))
    )
    if unnamed_count > len(df.columns) / 2 and df.height > 1:
        preview_rows = df.head(10).to_dicts()
        for row_idx, row_dict in enumerate(preview_rows):
            non_empty = [
                str(v).strip()
                for v in row_dict.values()
                if v is not None and str(v).strip() not in ("", "None", "null", "NaN")
            ]
            # Promote the first row that has at least half of its columns populated with header labels
            if len(non_empty) >= max(2, int(len(df.columns) * 0.5)):
                new_headers: List[str] = []
                seen: dict[str, int] = {}
                for col_i, val in enumerate(row_dict.values()):
                    raw_h = str(val).strip() if val is not None and str(val).strip() != "" else f"Col_{col_i + 1}"
                    if raw_h in seen:
                        seen[raw_h] += 1
                        raw_h = f"{raw_h}_{seen[raw_h]}"
                    else:
                        seen[raw_h] = 1
                    new_headers.append(raw_h)

                df = df.slice(row_idx + 1)
                df.columns = new_headers
                break

    return df


def _slice_polars_df_into_rows(
    df: pl.DataFrame,
    file_name: str,
    sheet_name: str,
    sheet_idx: int,
) -> List[Document]:
    """
    Phase 2: Schema-Agnostic Loop & 1-Row-Per-Chunk Slicing.
    Reads whatever column headers exist on the current sheet and turns every row
    into an independent Document block with prepended global context.
    """
    if df.is_empty():
        return []

    # Step 1: Promote real headers if top rows were title banners
    df = _promote_real_header_row(df)

    # Step 2: Cast all columns to UTF-8 strings and replace nulls with empty strings
    df = df.select([pl.col(c).cast(pl.Utf8).fill_null("") for c in df.columns])

    # Step 3: Forward-fill Column A if it acts as a merged parent category across rows
    if len(df.columns) >= 2:
        first_col = df.columns[0]
        df = df.with_columns(
            pl.when(pl.col(first_col).str.strip_chars() == "")
            .then(None)
            .otherwise(pl.col(first_col))
            .forward_fill()
            .fill_null("")
            .alias(first_col)
        )

    global_medicine = _infer_index_medicine_name(file_name, df)
    headers = [str(c).strip() for c in df.columns]
    row_documents: List[Document] = []

    # Step 4: Schema-Agnostic Row-by-Row Slicing
    for row_number, row_tuple in enumerate(df.iter_rows(), start=1):
        cell_entries: List[str] = []
        row_specific_medicine: str | None = None

        for col_header, cell_value in zip(headers, row_tuple):
            val_str = str(cell_value).strip() if cell_value is not None else ""
            if not val_str or val_str.lower() in ("nan", "none", "null"):
                continue
            cell_entries.append(f"{col_header}: {val_str}")

            # Capture row-level medicine name if this sheet lists multiple drugs
            if row_specific_medicine is None and any(
                k in col_header.lower() for k in ("medicine", "drug", "molecule", "compound", "generic", "product")
            ):
                row_specific_medicine = val_str

        # Skip blank separator rows
        if not cell_entries:
            continue

        effective_medicine = row_specific_medicine or global_medicine
        source_display = f"{file_name} | Sheet: {sheet_name}"

        # Phase 2 Context Prepending:
        # Glues Workbook Filename, Sheet Tab Name, and Index Medicine Name to the front of the row
        context_rich_row = (
            f"[Workbook: {file_name} | Sheet Tab: {sheet_name} | "
            f"Index Medicine: {effective_medicine} | Row {row_number}]\n"
            + " | ".join(cell_entries)
        )

        row_documents.append(
            Document(
                page_content=context_rich_row,
                metadata={
                    "source": source_display,
                    "file_name": file_name,
                    "sheet_name": sheet_name.lower().strip(),
                    "medicine_name": effective_medicine.lower().strip(),
                    "page": row_number,    # Row number stored in 'page' for unified citation
                    "file_type": "excel",  # Pre-sliced row; bypasses character text splitter
                },
            )
        )

    return row_documents


def parse_excel_or_csv_bytes(file_name: str, raw_data: bytes) -> List[Document]:
    """
    Phase 1: High-Speed Reading Engine using Rust Calamine (fastexcel) + Polars.
    Reads all tabs in an Excel workbook (.xlsx, .xls) or .csv into row-level Documents.
    """
    suffix = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    all_row_docs: List[Document] = []

    # Handle CSV files via Polars
    if suffix == "csv":
        try:
            df = pl.read_csv(io.BytesIO(raw_data), infer_schema_length=0, ignore_errors=True)
        except Exception:
            pdf = pd.read_csv(io.BytesIO(raw_data), dtype=str, low_memory=False).fillna("")
            df = pl.from_pandas(pdf)
        all_row_docs.extend(_slice_polars_df_into_rows(df, file_name, "CSV_Sheet", 1))
        if not all_row_docs:
            raise ValueError("CSV file contains no readable rows.")
        return all_row_docs

    # Primary Path: Polars + Rust Calamine Engine (sheet_name=None equivalent across all tabs)
    try:
        # 1. Try Polars native calamine multi-sheet reader (`sheet_Id=0` / `sheet_name=None`)
        sheets_map = pl.read_excel(io.BytesIO(raw_data), sheet_id=0, engine="calamine")
        if isinstance(sheets_map, dict):
            for sheet_idx, (sheet_name, df) in enumerate(sheets_map.items(), start=1):
                all_row_docs.extend(_slice_polars_df_into_rows(df, file_name, str(sheet_name), sheet_idx))
        elif isinstance(sheets_map, pl.DataFrame):
            all_row_docs.extend(_slice_polars_df_into_rows(sheets_map, file_name, "Sheet1", 1))
    except Exception:
        # 2. Direct fastexcel (Calamine) per-sheet loader for irregular schemas
        try:
            reader = fastexcel.read_excel(raw_data)
            for sheet_idx, sheet_name in enumerate(reader.sheet_names, start=1):
                try:
                    df = reader.load_sheet(sheet_name, header_row=0).to_polars()
                    all_row_docs.extend(_slice_polars_df_into_rows(df, file_name, str(sheet_name), sheet_idx))
                except Exception:
                    continue
        except Exception:
            # 3. Final fallback to Pandas + openpyxl for legacy/unusual .xls formats
            try:
                sheets_dict: dict[str, Any] = pd.read_excel(io.BytesIO(raw_data), sheet_name=None, dtype=str)
                for sheet_idx, (sheet_name, pdf) in enumerate(sheets_dict.items(), start=1):
                    pdf = pdf.dropna(how="all").fillna("")
                    if not pdf.empty:
                        df = pl.from_pandas(pdf.astype(str))
                        all_row_docs.extend(_slice_polars_df_into_rows(df, file_name, str(sheet_name), sheet_idx))
            except Exception as exc:
                raise ValueError(f"could not read Excel file ({exc}).") from exc

    if not all_row_docs:
        raise ValueError("Excel workbook contains no readable rows across any of its sheets.")

    return all_row_docs
