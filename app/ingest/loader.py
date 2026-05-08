# app/ingest/loader.py
from __future__ import annotations
import io, os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import pandas as pd

@dataclass
class LoadMeta:
    n_rows: int
    n_cols: int
    source_name: str           # np. nazwa pliku
    engine: str                # np. 'csv_pandas', 'excel_openpyxl', 'parquet_pyarrow'
    encoding: Optional[str] = None
    notes: str = ""
    # ile wierszy jest w PREVIEW (podglądzie); None = brak informacji
    preview_rows: Optional[int] = None
    # nowe pola – opcjonalne (ważne!)
    file_size_mb: Optional[float] = None
    sample_used: bool = False

# ---------- helpers ----------
def excel_sheet_names(file_bytes: bytes) -> List[str]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    return list(xls.sheet_names)

def _try_read_csv(file_bytes: bytes, sep=None, encoding=None, nrows=None) -> pd.DataFrame:
    return pd.read_csv(
        io.BytesIO(file_bytes),
        sep=sep,
        encoding=encoding,
        engine="python",  # umożliwia auto-sep
        nrows=nrows,
    )

# ---------- CSV ----------
def load_csv_smart(
    file_bytes: bytes,
    file_name: str,
    preview_limit: Optional[int] = None,
    sep_choice: Optional[str] = None,   # ',', ';', '\t', None (auto)
) -> Tuple[pd.DataFrame, LoadMeta]:
    size_mb = round(len(file_bytes) / (1024 ** 2), 2)
    if sep_choice in (",",";","\t"):
        try_encs = ["utf-8","cp1250","latin1"]
        last_err = None
        for enc in try_encs:
            try:
                df = _try_read_csv(file_bytes, sep=sep_choice, encoding=enc, nrows=preview_limit)
                meta = LoadMeta(
                    source_name=file_name, file_size_mb=size_mb,
                    n_rows=df.shape[0], n_cols=df.shape[1],
                    sample_used=bool(preview_limit), engine=f"read_csv(sep='{sep_choice}')",
                    encoding=enc
                )
                return df, meta
            except Exception as e:
                last_err = e
        raise RuntimeError(f"CSV nie wczytany (sep='{sep_choice}'). Ostatni błąd: {last_err}")
    # auto-sep try order
    try_order = [
        dict(sep=None, encoding="utf-8"),
        dict(sep=None, encoding="cp1250"),
        dict(sep=None, encoding="latin1"),
    ]
    errors: List[str] = []
    for attempt in try_order:
        try:
            df = _try_read_csv(file_bytes, nrows=preview_limit, **attempt)
            meta = LoadMeta(
                source_name=file_name, file_size_mb=size_mb,
                n_rows=df.shape[0], n_cols=df.shape[1],
                sample_used=bool(preview_limit), engine="read_csv(auto-sep)",
                encoding=attempt["encoding"]
            )
            return df, meta
        except Exception as e:
            errors.append(f"{attempt} -> {type(e).__name__}: {e}")
    raise RuntimeError("Nie udało się wczytać CSV żadnym podejściem.\n" + "\n".join(errors))

# ---------- EXCEL ----------
def load_excel_smart(
    file_bytes: bytes,
    file_name: str,
    sheet_name: Optional[str] = None,
    preview_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, LoadMeta]:
    size_mb = round(len(file_bytes) / (1024 ** 2), 2)
    xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    use_sheet = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
    df = xls.parse(use_sheet, nrows=preview_limit)
    meta = LoadMeta(
        source_name=f"{file_name}::{use_sheet}",
        file_size_mb=size_mb,
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        sample_used=bool(preview_limit),
        engine="read_excel(openpyxl)",
        sheet_name=use_sheet,
    )
    return df, meta

# ---------- PDF (tables) ----------
def load_pdf_tables(
    file_bytes: bytes,
    file_name: str,
    pages: str = "1-end",
    flavor: str = "lattice",   # 'lattice' | 'stream'
    preview_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, LoadMeta]:
    size_mb = round(len(file_bytes) / (1024 ** 2), 2)
    all_tables: List[pd.DataFrame] = []
    used = "pdfplumber"
    notes = []

    # 1) Camelot
    try:
        import camelot  # noqa
        tmp = "_tmp_in_memory.pdf"
        with open(tmp, "wb") as f:
            f.write(file_bytes)
        tables = camelot.read_pdf(tmp, pages=pages, flavor=flavor)
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass
        for t in tables:
            all_tables.append(t.df)
        used = f"camelot({flavor})"
        notes.append(f"camelot wykrył {len(all_tables)} tabel(e)")
    except Exception as e:
        notes.append(f"camelot nieudany: {e}")

    # 2) pdfplumber fallback
    if not all_tables:
        try:
            import pdfplumber as _pp  # noqa
            import pypdfium2  # noqa
            with _pp.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    for tbl in page.extract_tables() or []:
                        all_tables.append(pd.DataFrame(tbl))
            used = "pdfplumber"
            notes.append(f"pdfplumber wykrył {len(all_tables)} tabel(e)")
        except Exception as e:
            raise RuntimeError(f"Nie udało się wyekstrahować tabel z PDF. Ostatni błąd: {e}")

    if not all_tables:
        raise RuntimeError("Brak wykrytych tabel w PDF.")

    df = pd.concat(all_tables, axis=0, ignore_index=True)
    if preview_limit:
        df = df.head(preview_limit)

    meta = LoadMeta(
        source_name=file_name,
        file_size_mb=size_mb,
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        sample_used=bool(preview_limit),
        engine=used,
        notes="; ".join(notes) if notes else None,
    )
    return df, meta

# ---------- PARQUET ----------

def load_parquet_smart(
    file_bytes: bytes,
    file_name: str,
    preview_limit: int | None = None,
) -> tuple[pd.DataFrame, LoadMeta]:
    """Read Parquet preview without materializing the whole file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    buffer = io.BytesIO(file_bytes)
    parquet_file = pq.ParquetFile(buffer)
    n_full_rows = int(parquet_file.metadata.num_rows)
    n_cols = len(parquet_file.schema_arrow.names)

    if preview_limit is not None and n_full_rows > preview_limit:
        batches = []
        rows_left = int(preview_limit)
        batch_size = max(1, min(rows_left, 10_000))
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            if rows_left <= 0:
                break
            if batch.num_rows > rows_left:
                batch = batch.slice(0, rows_left)
            batches.append(batch)
            rows_left -= batch.num_rows
        table = pa.Table.from_batches(batches, schema=parquet_file.schema_arrow)
        sample_used = True
    else:
        table = parquet_file.read()
        sample_used = False

    df_preview = table.to_pandas()
    n_prev_rows = int(df_preview.shape[0])

    base_name = os.path.basename(file_name)
    size_mb = round(len(file_bytes) / (1024 ** 2), 2)

    meta = LoadMeta(
        n_rows=n_full_rows,
        n_cols=n_cols,
        source_name=base_name,
        engine="parquet_pyarrow",
        encoding=None,
        notes="Plik Parquet (PyArrow)",
        preview_rows=n_prev_rows,
        file_size_mb=size_mb,
        sample_used=sample_used,
    )

    return df_preview, meta

# ---------- Router ----------
def load_any(
    file_bytes: bytes,
    file_name: str,
    preview_limit: Optional[int] = None,
    csv_sep: Optional[str] = None,
    xlsx_sheet: Optional[str] = None,
    pdf_pages: str = "1-end",
    pdf_flavor: str = "lattice",
) -> Tuple[pd.DataFrame, LoadMeta]:
    name = file_name.lower()

    # CSV
    if name.endswith(".csv"):
        return load_csv_smart(
            file_bytes,
            file_name,
            preview_limit,
            sep_choice=csv_sep,
        )

    # XLS / XLSX
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return load_excel_smart(
            file_bytes,
            file_name,
            sheet_name=xlsx_sheet,
            preview_limit=preview_limit,
        )

    # PDF
    elif name.endswith(".pdf"):
        return load_pdf_tables(
            file_bytes,
            file_name,
            pages=pdf_pages,
            flavor=pdf_flavor,
            preview_limit=preview_limit,
        )

    # PARQUET  <<< NOWY BLOK >>>
    elif name.endswith(".parquet"):
        return load_parquet_smart(
            file_bytes,
            file_name,
            preview_limit=preview_limit,
        )


    # nic nie pasuje
    else:
        raise ValueError("Nieobsługiwane rozszerzenie: CSV, XLSX, PDF, PARQUET.")


