# app/pages/01_Analiza_Danych.py
from __future__ import annotations

import os
import io
import json
import hashlib
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Auto EDA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Parquet-only storage helpers (Stage 1/2/3)
# ─────────────────────────────────────────────────────────────
def _df_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to Parquet (snappy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    except Exception:
        # fallback: let pandas pick engine if available
        df.to_parquet(path, index=False, compression="snappy")

def _optimize_dtypes_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight dtype downcast to reduce size and speed IO."""
    try:
        import numpy as np
        out = df.copy()
        # numeric downcast
        for c in out.select_dtypes(include=["int", "int64", "int32"]).columns:
            out[c] = pd.to_numeric(out[c], downcast="integer")
        for c in out.select_dtypes(include=["float", "float64", "float32"]).columns:
            out[c] = pd.to_numeric(out[c], downcast="float")
        # bool-ish objects
        for c in out.select_dtypes(include=["object"]).columns:
            s = out[c]
            # convert low-cardinality object to category
            try:
                nun = s.nunique(dropna=True)
                if nun > 0 and nun <= 200 and nun / max(len(s), 1) <= 0.2:
                    out[c] = s.astype("category")
            except Exception:
                pass
        return out
    except Exception:
        return df

from pandas.api.types import is_datetime64_any_dtype

from core.i18n import t
from core.config import load_config, resolve_artifacts_dir
from core.pii import mask_dataframe
from ingest import load_any, excel_sheet_names
from core.top_nav import (
    hide_default_multipage_nav,
    render_flow_nav,
    render_sidebar_links,
)


from streamlit.components.v1 import html as st_html

# -------------------------------------------------
# GLOBALNY STYL — spójny z Etapem 2 + uploader
# -------------------------------------------------

st.markdown(
    """
<style>
/* =========================================================
   1) FILE UPLOADER – ukryj wbudowany pasek z nazwą pliku + X
   ========================================================= */

/* Cała linia z nazwą pliku i X-em pod dropzone */
div[data-testid="stFileUploaderFile"] {
    display: none !important;
}


/* Na wszelki wypadek – nazwa pliku i przycisk kasowania */
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderDeleteBtn"],
[data-testid="stFileUploaderPagination"] {
    display: none !important;
}

/* =========================================================
   2) KPI – taki sam wygląd jak w Etapie 2
   ========================================================= */

.kpi-card {
    background-color: #f5f5f7;
    border-radius: 0.75rem;
    padding: 0.75rem 1rem;
    border: 1px solid #e5e7eb;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 90px;
}

.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0;
    text-transform: none;
    color: #6b7280;
}

.kpi-value {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 0.1rem;
    color: #111827;
}

.kpi-sub {
    font-size: 0.75rem;
    margin-top: 0.1rem;
    color: #6b7280;
}

.kpi-value--ok {
    color: #16a34a;
}

.kpi-value--warn {
    color: #ea580c;
}
</style>
""",
    unsafe_allow_html=True,
)

# =================================================
# itables / DataTables – helper
# =================================================


def _render_itables_html(
    df: pd.DataFrame, body_px: int, page_len: int
) -> tuple[str, int]:
    """
    body_px  -> wysokość przewijanej części tabeli (scrollY) sterowana suwakiem
    page_len -> entries per page (np. 10/25/50/100)
    Zwraca: (html, iframe_height)
    """
    import json as _json

    HEADER = 56
    FOOTER = 54
    BORDERS = 10
    iframe_h = int(body_px + HEADER + FOOTER + BORDERS)

    records = df.to_dict(orient="records")
    columns = [{"title": c, "data": c} for c in df.columns]

    html = f"""
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>

<style>
  html, body, #wrap {{ margin:0 !important; padding:0 !important; }}
  .dataTables_scroll {{ margin-bottom:0 !important; }}
  .dataTables_wrapper .dataTables_info {{ margin-top:6px !important; }}
  .dataTables_wrapper .dataTables_paginate {{ margin-top:6px !important; }}
  .dataTables_wrapper, table.dataTable thead th, table.dataTable tbody td {{
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
      font-size: 14px !important;
      line-height: 1.35;
  }}
</style>

<div id="wrap" style="padding:0;margin:0;">
  <table id="t" class="display compact stripe" style="width:100%"></table>
</div>

<script>
  const data = {_json.dumps(records)};
  const columns = {_json.dumps(columns)};
  const scrollY = {int(body_px)};
  const pageLen = {int(page_len)};

  $(document).ready(function(){{
    $('#t').DataTable({{
      data: data,
      columns: columns,
      scrollY: scrollY + 'px',
      scrollX: true,
      deferRender: true,
      autoWidth: false,
      paging: true,
      pageLength: pageLen,
      lengthMenu: [10,25,50,100],
      info: true
    }});
  }});
</script>
"""
    return html, iframe_h


def _to_excel_bytes(df: pd.DataFrame) -> bytes | None:
    """
    Zwraca bytes gotowego pliku XLSX.
    Najpierw próbujemy użyć xlsxwriter (jeśli jest zainstalowany),
    a jeśli go nie ma – fallback do openpyxl.
    """
    for engine in ("xlsxwriter", "openpyxl"):
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine=engine) as writer:
                df.to_excel(writer, sheet_name="data", index=False)
            buf.seek(0)
            return buf.getvalue()
        except (ImportError, ModuleNotFoundError):
            continue
    return None


def _preview_download_buttons(df_view: pd.DataFrame):
    """
    Przyciski pobierania podglądu (CSV / XLSX) – na df po maskowaniu PII.
    """
    col_csv, col_xlsx, _ = st.columns([1, 1, 6])

    with col_csv:
        st.download_button(
            "⬇ CSV (podgląd)",
            data=df_view.to_csv(index=False).encode("utf-8"),
            file_name="preview_masked.csv",
            mime="text/csv",
            key="dl_csv_preview",
            help="Eksport podglądu (po maskowaniu PII) do CSV.",
        )

    with col_xlsx:
        st.button(
            "⬇ XLSX (podgląd)",
            disabled=True,
            help=(
                "Eksport XLSX jest wyłączony na Streamlit Community Cloud, "
                "żeby ładowanie danych nie zależało od silników Excela. "
                "Użyj CSV albo ZIP."
            ),
        )


def _zip_bytes(df_out: pd.DataFrame, meta_obj, base_name: str) -> bytes:
    """
    ZIP z artefaktami podglądu (CSV + meta.json).
    """
    buf = io.BytesIO()
    safe_base = base_name.replace(" ", "_")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{safe_base}__preview_masked.csv", df_out.to_csv(index=False))
        z.writestr(
            f"{safe_base}__meta.json",
            json.dumps(meta_obj.__dict__, ensure_ascii=False, indent=2),
        )
    buf.seek(0)
    return buf.getvalue()


# =================================================
# KPI – kafelki jak w Etapie 2
# =================================================

def _show_kpi_block(df_preview: pd.DataFrame, mask_pii: bool, meta_obj):
    """
    Lekkie KPI po wczytaniu:
    - Wiersze × Kolumny
    - Typy (num / kat / data)
    - Status maskowania PII
    """
    num_cols = df_preview.select_dtypes(include="number").columns
    cat_cols = df_preview.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
    date_cols = [
        col for col in df_preview.columns
        if is_datetime64_any_dtype(df_preview[col])
    ]

    rows_txt = f"{meta_obj.n_rows:,}"
    cols_txt = f"{meta_obj.n_cols:,}"
    types_txt = (
        f"{len(num_cols)} num · {len(cat_cols)} kat · {len(date_cols)} data"
    )
    pii_txt = "Włączone" if mask_pii else "Wyłączone"
    pii_class = "kpi-value kpi-value--ok" if mask_pii else "kpi-value"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Wiersze × Kolumny</div>
              <div class="kpi-value">{rows_txt} × {cols_txt}</div>
              <div class="kpi-sub">Rozmiar próbki podglądu</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Kolumny numeryczne</div>
              <div class="kpi-value">{len(num_cols)}</div>
              <div class="kpi-sub">
                Możliwe cechy do modeli regresji / klasyfikacji
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Typy kolumn</div>
              <div class="kpi-value">{types_txt}</div>
              <div class="kpi-sub">Rozkład typów w aktualnym podglądzie</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Maskowanie PII</div>
              <div class="{pii_class}">{pii_txt}</div>
              <div class="kpi-sub">
                Ochrona danych wrażliwych w podglądzie
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =================================================
# DEMO – PyCaret + 3 publiczne TimeSeries (DO)
# =================================================

# 3 publiczne szeregi czasowe – wczytywane z URL (np. DigitalOcean Spaces).
# Użytkownik ustawia zmienne środowiskowe:
# TS_RETAIL_URL, TS_TEMPERATURE_URL, TS_ENERGY_URL
PUBLIC_TS_DEMOS: dict[str, dict] = {
    "🌍 Publiczny TS (DO) · Sprzedaż detaliczna": {
        # uniwersalny typ: remote_file (CSV lub Parquet)
        "kind": "remote_file",
        "url_env": "TS_RETAIL_URL",
        # lokalny fallback jest w formacie Parquet:
        "local_path": "data/ts_online_retail_transactions.parquet",
        "file_format": "parquet",  # ⬅️ nowy klucz: jasno mówimy, że to Parquet
        "task": "Szereg czasowy",
        "description": "Miesięczna sprzedaż detaliczna (dane publiczne).",
        "label_short": "TS: sprzedaż detaliczna",
    },
    "🌍 Publiczny TS (DO) · Temperatury": {
        "kind": "remote_file",
        "url_env": "TS_TEMPERATURE_URL",
        "local_path": "data/ts_temperature_global_berkeley_monthly.csv",
        "file_format": "csv",
        "task": "Szereg czasowy",
        "description": "Długoterminowe pomiary temperatury.",
        "label_short": "TS: temperatury",
    },
    "🌍 Publiczny TS (DO) · Zużycie energii": {
        "kind": "remote_file",
        "url_env": "TS_ENERGY_URL",
        "local_path": "data/ts_energy_household_daily.csv",
        "file_format": "csv",
        "task": "Szereg czasowy",
        "description": "Profil zużycia energii elektrycznej.",
        "label_short": "TS: energia",
    },
}


PYCARET_DATASET_BASE_URL = "https://raw.githubusercontent.com/pycaret/datasets/main"
PYCARET_DEMO_INDEX_CACHE_VERSION = "full-pycaret-fallback-2026-05-01-v2"

PYCARET_FALLBACK_DEMOS: tuple[dict[str, str], ...] = (
    # Friendly examples with short Polish descriptions.
    {
        "dataset": "jewellery",
        "task": "Klasteryzacja",
        "description": "Segmentacja klientow sklepu jubilerskiego.",
    },
    {
        "dataset": "seeds",
        "task": "Klasteryzacja",
        "description": "Cechy nasion do grupowania.",
    },
    {
        "dataset": "iris",
        "task": "Klasyfikacja",
        "description": "Klasyczny zbior wieloklasowy.",
    },
    {
        "dataset": "juice",
        "task": "Klasyfikacja",
        "description": "Zakup produktu przez klienta.",
    },
    {
        "dataset": "titanic",
        "task": "Klasyfikacja",
        "description": "Przezycie pasazera Titanica.",
    },
    {
        "dataset": "diamond",
        "task": "Regresja",
        "description": "Cena diamentu.",
    },
    {
        "dataset": "insurance",
        "task": "Regresja",
        "description": "Koszty ubezpieczenia.",
    },
    # Full PyCaret common dataset index fallback. This keeps Streamlit Cloud useful
    # even when pycaret.datasets.get_data("index") is unavailable.
    {
        "dataset": "asia_gdp",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "elections",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "facebook",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "ipl",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "jewellery",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "mice",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "migration",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "perfume",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "pokemon",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "population",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "public_health",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "seeds",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "wholesale",
        "task": "Klasteryzacja",
        "description": "",
    },
    {
        "dataset": "amazon",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "bank",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "blood",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "cancer",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "credit",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "CTG",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "diabetes",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "electrical_grid",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "employee",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "glass",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "heart",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "heart_disease",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "hepatitis",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "income",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "iris",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "juice",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "kiva",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "nba",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "poker",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "questions",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "satellite",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "telescope",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "titanic",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "us_presidential_election_results",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "wikipedia",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "wine",
        "task": "Klasyfikacja",
        "description": "",
    },
    {
        "dataset": "automobile",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "bike",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "boston",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "concrete",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "diamond",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "energy",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "forest",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "gold",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "house",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "insurance",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "parkinsons",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "spx",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "traffic",
        "task": "Regresja",
        "description": "",
    },
    {
        "dataset": "airline",
        "task": "Szereg czasowy",
        "description": "Linie lotnicze (airline)",
    },
    {
        "dataset": "uschange",
        "task": "Szereg czasowy",
        "description": "Zmiany gospodarcze USA (uschange)",
    },
)


def _get_env_or_secret(name: str) -> str:
    """Read Community Cloud secrets and local env with one code path."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        value = st.secrets.get(name, "")
    except Exception:
        return ""
    return str(value).strip() if value is not None else ""


def _resolve_existing_path(path: str | Path | None) -> Path | None:
    if not path:
        return None

    raw = Path(path)
    candidates = [raw]
    if not raw.is_absolute():
        here = Path(__file__).resolve()
        candidates.extend([Path.cwd() / raw, here.parent / raw])
        candidates.extend(parent / raw for parent in here.parents[:3])

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _pycaret_dataset_url(dataset: str, folder: str | None = None) -> str:
    folder = (folder or "common").strip("/")
    return f"{PYCARET_DATASET_BASE_URL}/data/{folder}/{dataset}.csv"


def _read_url_bytes(url: str, label: str) -> bytes:
    req = Request(url, headers={"User-Agent": "AutoEDA-Streamlit/1.0"})
    try:
        with urlopen(req, timeout=60) as response:
            return response.read()
    except HTTPError as e:
        raise RuntimeError(
            f"{label}: serwer zwrocil HTTP {e.code}. Sprawdz, czy URL prowadzi "
            "bezposrednio do publicznego pliku i czy link nie wygasl."
        ) from e
    except URLError as e:
        raise RuntimeError(
            f"{label}: nie udalo sie polaczyc z adresem URL ({e.reason})."
        ) from e
    except TimeoutError as e:
        raise RuntimeError(f"{label}: przekroczono limit czasu pobierania pliku.") from e


def _read_tabular_source(path_or_url: str | Path, file_format: str, label: str) -> pd.DataFrame:
    path_txt = str(path_or_url)
    fmt = (file_format or "").lower()
    if not fmt:
        fmt = "parquet" if path_txt.lower().endswith(".parquet") else "csv"

    if path_txt.lower().startswith(("http://", "https://")):
        raw = _read_url_bytes(path_txt, label)
        buffer = io.BytesIO(raw)
        if fmt == "parquet":
            return pd.read_parquet(buffer)
        return pd.read_csv(buffer)

    if fmt == "parquet":
        return pd.read_parquet(path_or_url)
    return pd.read_csv(path_or_url)


def _builtin_public_ts_dataset(spec: dict) -> pd.DataFrame | None:
    """Small built-in fallback for public TS demos when cloud URLs are unavailable."""
    url_env = spec.get("url_env")

    if url_env == "TS_RETAIL_URL":
        months = pd.date_range("2021-01-01", periods=48, freq="MS")
        regions = ["North", "South", "West"]
        products = ["A", "B"]
        rows = []
        for i, date in enumerate(months):
            for region_idx, region in enumerate(regions):
                for product_idx, product in enumerate(products):
                    units = 80 + i * 3 + region_idx * 11 + product_idx * 17
                    revenue = round(units * (18.5 + product_idx * 4.25), 2)
                    rows.append(
                        {
                            "date": date,
                            "region": region,
                            "product": product,
                            "units": units,
                            "revenue": revenue,
                        }
                    )
        return pd.DataFrame(rows)

    if url_env == "TS_TEMPERATURE_URL":
        months = pd.date_range("2015-01-01", periods=96, freq="MS")
        rows = []
        for i, date in enumerate(months):
            seasonal = [0.2, 1.1, 4.0, 8.2, 12.4, 16.1, 18.5, 17.9, 14.2, 9.6, 4.8, 1.3]
            baseline = seasonal[(date.month - 1) % 12]
            trend = i * 0.018
            rows.append(
                {
                    "date": date,
                    "city": "Demo City",
                    "avg_temp_c": round(baseline + trend, 2),
                    "anomaly_c": round(trend - 0.35, 2),
                }
            )
        return pd.DataFrame(rows)

    if url_env == "TS_ENERGY_URL":
        days = pd.date_range("2024-01-01", periods=365, freq="D")
        rows = []
        for i, date in enumerate(days):
            weekday_factor = 0.86 if date.weekday() >= 5 else 1.0
            seasonal = 1.0 + (0.18 if date.month in (1, 2, 12) else 0.0)
            kwh = round((21.5 + (i % 30) * 0.18) * weekday_factor * seasonal, 2)
            rows.append(
                {
                    "date": date,
                    "household_id": "H001",
                    "energy_kwh": kwh,
                    "weekday": date.day_name(),
                }
            )
        return pd.DataFrame(rows)

    return None


def _fallback_pycaret_demo_index() -> dict[str, dict]:
    demos: dict[str, dict] = {}
    for item in PYCARET_FALLBACK_DEMOS:
        task = item["task"]
        ds_name = item["dataset"]
        desc = item.get("description", "")
        label = f"{task} - {ds_name}"
        if desc:
            label += f" - {desc}"
        demos[label] = {
            "kind": "pycaret",
            "dataset": ds_name,
            "folder": item.get("folder", "common"),
            "task": task,
            "description": desc,
            "label_short": f"{task}: {ds_name}",
        }

    return demos



@st.cache_data(show_spinner=False)
def _build_pycaret_demo_index(cache_version: str) -> dict[str, dict]:
    """
    Buduje słownik demo-zbiorów z PyCaret dla zadań:
    klasyfikacja, regresja, klastrowanie, time series.
    Klucz: label dla selectboxa (bez słowa 'PyCaret').
    """
    _ = cache_version
    try:
        from pycaret.datasets import get_data
    except Exception:
        return _fallback_pycaret_demo_index()

    try:
        idx = get_data("index", verbose=False)
    except Exception:
        return _fallback_pycaret_demo_index()

    df_idx = pd.DataFrame(idx)
    lower_map = {c.lower(): c for c in df_idx.columns}

    dataset_col = next(
        (orig for low, orig in lower_map.items() if "dataset" in low),
        df_idx.columns[0],
    )
    task_col = next(
        (orig for low, orig in lower_map.items() if "task" in low),
        None,
    )
    desc_col = next(
        (orig for low, orig in lower_map.items()
         if "description" in low or "desc" in low or "brief" in low),
        None,
    )

    demos: dict[str, dict] = {}

    for _, row in df_idx.iterrows():
        raw_task = str(row[task_col]).lower() if task_col else ""
        if not any(
            kw in raw_task
            for kw in ["classification", "regression", "cluster", "time series", "forecast"]
        ):
            continue

        ds_name = str(row[dataset_col])
        if not ds_name or ds_name.lower() == "nan":
            continue

        if "class" in raw_task:
            nice_task = "Klasyfikacja"
        elif "regress" in raw_task:
            nice_task = "Regresja"
        elif "cluster" in raw_task:
            nice_task = "Klasteryzacja"
        else:
            nice_task = "Szereg czasowy"

        desc = str(row[desc_col]) if desc_col else ""
        if desc.lower() == "nan":
            desc = ""

        label = f"{nice_task} · {ds_name}"
        if desc:
            label += f" — {desc}"

        demos[label] = {
            "kind": "pycaret",
            "dataset": ds_name,
            "task": nice_task,
            "description": desc,
            "label_short": f"{nice_task}: {ds_name}",
        }

    # Ręczne dociągnięcie klasycznych TS z PyCaret TimeSeries (jeśli brak w index)
    for extra_ds, extra_label in {
        "airline": "Szereg czasowy · Linie lotnicze (airline)",
        "uschange": "Szereg czasowy · Zmiany gospodarcze USA (uschange)",
    }.items():
        if extra_label not in demos:
            demos[extra_label] = {
                "kind": "pycaret",
                "dataset": extra_ds,
                "task": "Szereg czasowy",
                "description": "",
                "label_short": f"Szereg czasowy: {extra_ds}",
            }

    for label, spec in _fallback_pycaret_demo_index().items():
        demos.setdefault(label, spec)

    return demos


def _load_demo_dataset(spec: dict, preview_limit: int | None = None):
    """
    Ładuje zestaw demo (PyCaret / remote CSV/Parquet).

    Zwraca:
    df_preview, df_full, meta_preview, meta_full, approx_size_mb
    """
    kind = spec.get("kind")
    task = spec.get("task") or ""

    # 1) Wczytanie pełnego zbioru
    if kind == "pycaret":
        dataset_name = str(spec.get("dataset", "")).strip()
        folder = spec.get("folder")
        try:
            from pycaret.datasets import get_data
            kwargs = {"verbose": False}
            if folder:
                kwargs["folder"] = folder
            df_full = get_data(dataset_name, **kwargs)
        except Exception as pycaret_error:
            try:
                fallback_url = _pycaret_dataset_url(dataset_name, folder)
                df_full = _read_tabular_source(
                    fallback_url,
                    "csv",
                    f"PyCaret dataset {dataset_name}",
                )
            except Exception as fallback_error:
                st.error(
                    "Nie mogę załadować zestawu demo PyCaret. "
                    "Na Streamlit Community Cloud najczęściej oznacza to brak pakietu "
                    "`pycaret` w `requirements.txt` albo brak dostępu do repozytorium "
                    "z danymi PyCaret.\n\n"
                    f"PyCaret: {pycaret_error}\n\nFallback CSV: {fallback_error}"
                )
                st.stop()

    elif kind in ("remote_csv", "remote_file"):
        url_env    = spec.get("url_env", "")
        url        = _get_env_or_secret(url_env)
        local_path = spec.get("local_path")
        file_format = (spec.get("file_format") or "").lower()

        def _infer_format(path: str) -> str:
            """Ustala format tabeli na podstawie spec/file extension."""
            if file_format in ("csv", "parquet"):
                return file_format
            if path.lower().endswith(".parquet"):
                return "parquet"
            return "csv"

        def _read_table(path: str) -> pd.DataFrame:
            return _read_tabular_source(path, _infer_format(path), url_env or str(path))

        local_existing_path = _resolve_existing_path(local_path)
        if url:
            # wariant 1: wczytujemy tabelę z URL (CSV lub Parquet)
            try:
                df_full = _read_table(url)
            except Exception as e:
                if local_existing_path:
                    st.warning(
                        f"Nie udało się pobrać `{url_env}` z URL, więc używam "
                        f"lokalnego fallbacku: `{local_existing_path}`.\n\nSzczegóły: {e}"
                    )
                    df_full = _read_table(str(local_existing_path))
                else:
                    builtin_df = _builtin_public_ts_dataset(spec)
                    if builtin_df is not None:
                        st.warning(
                            f"Nie udało się pobrać `{url_env}` z URL, więc używam "
                            "wbudowanego zestawu demo dla szeregu czasowego.\n\n"
                            f"Szczegóły: {e}"
                        )
                        df_full = builtin_df
                    else:
                        st.error(
                            f"Nie udało się pobrać danych demo z `{url_env}`.\n\n"
                            f"Szczegóły: {e}\n\n"
                            "Na Streamlit Community Cloud sprawdź, czy sekret zawiera "
                            "bezpośredni publiczny link do pliku, a nie stronę HTML, "
                            "link prywatny albo wygasły signed URL."
                        )
                        st.stop()
        elif local_existing_path:
            # wariant 2: fallback – lokalny plik w repo
            df_full = _read_table(str(local_existing_path))
        else:
            builtin_df = _builtin_public_ts_dataset(spec)
            if builtin_df is not None:
                st.warning(
                    f"Brak dostępnego pliku dla `{url_env}`, więc używam "
                    "wbudowanego zestawu demo dla szeregu czasowego."
                )
                df_full = builtin_df
            else:
                fmt_txt = spec.get("file_format", "CSV lub Parquet")
                msg = (
                    f"Zestaw demo wymaga zdefiniowania zmiennej środowiskowej "
                    f"`{url_env}` z adresem URL do pliku ({fmt_txt}, np. z DigitalOcean "
                    "Spaces) **lub** umieszczenia pliku lokalnie pod ścieżką "
                    f"`{local_path}`.\n\n"
                    "Ustaw URL w `.env` **albo** dodaj lokalny plik i spróbuj ponownie."
                )
                st.error(msg)
                st.stop()

    else:
        st.error(f"Nieznany typ zestawu demo: {kind}")
        st.stop()

    # 2) Bezpieczne rzutowanie na DataFrame (naprawia możliwy błąd 'not enough values to unpack')
    if isinstance(df_full, pd.Series):
        df_full = df_full.to_frame()
    elif not isinstance(df_full, pd.DataFrame):
        df_full = pd.DataFrame(df_full)

    n_full_rows, n_cols = df_full.shape

    # 3) Podgląd (preview) – lekkie przycięcie do preview_limit
    if preview_limit is not None and n_full_rows > preview_limit:
        df_preview = df_full.head(preview_limit).copy()
        n_prev_rows = preview_limit
    else:
        df_preview = df_full.copy()
        n_prev_rows = n_full_rows

    # 4) Notatki + opis silnika
    notes = spec.get("description") or ""
    if kind == "pycaret":
        notes = (notes + " (źródło: pycaret.datasets.get_data)").strip()
    elif kind in ("remote_csv", "remote_file"):
        fmt = (spec.get("file_format") or "").lower()
        if fmt == "parquet":
            suffix = " (plik z URL, Parquet)"
        elif fmt == "csv":
            suffix = " (plik z URL, CSV)"
        else:
            suffix = " (plik z URL)"
        notes = (notes + suffix).strip()

    source_name = spec.get("label_short") or spec.get("dataset") or spec.get("url_env")

    # 5) Engine – rozróżniamy demo_pycaret / demo_remote_csv / demo_remote_parquet
    if kind == "pycaret":
        engine_name = "demo_pycaret"
    else:
        fmt = (spec.get("file_format") or "").lower()
        engine_name = "demo_remote_parquet" if fmt == "parquet" else "demo_remote_csv"

    # 6) Metadane – TU dodajemy task, żeby Etap 2 widział „Szereg czasowy”
    meta_preview = SimpleNamespace(
        n_rows=n_prev_rows,
        n_cols=n_cols,
        source_name=str(source_name),
        engine=engine_name,
        encoding=None,
        notes=notes,
        task=task,        # ⬅️ KLUCZOWE: zadanie trafia do meta.json
    )
    meta_full = SimpleNamespace(
        n_rows=n_full_rows,
        n_cols=n_cols,
        source_name=str(source_name),
        engine=engine_name,
        encoding=None,
        notes=notes,
        task=task,        # ⬅️ to samo dla pełnego meta
    )

    # 7) Szacowany rozmiar w MB – tylko info, nie wpływa na CPU
    approx_size_mb = round(
        df_full.memory_usage(deep=True).sum() / (1024 ** 2),
        2,
    )

    return df_preview, df_full, meta_preview, meta_full, approx_size_mb

# --- NAWIGACJA ---
hide_default_multipage_nav()
render_flow_nav(current_id="01_Analiza_Danych")  # aktywny kafel Etapu 1

# =================================================
# GŁÓWNA LOGIKA STRONY
# =================================================

st.title("Analiza Danych — wczytywanie (Etap 1)")
st.markdown(
    "Etap **1 z 4** – wczytujesz dane i robisz szybki sanity check, "
    "zanim przejdziesz do pełnej analizy i trenowania modelu."
)
st.subheader("", divider="gray")
cfg, problems = load_config()
MAX_MB, WARN_ROWS, SAMPLE_ROWS = cfg.max_file_mb, cfg.warn_rows, cfg.sample_rows

# 🔧 Lokalna korekta limitu – pozwól na pliki do 200 MB
# (nie zmieniamy configu globalnego, tylko to co widzi ta strona)
MAX_MB = max(MAX_MB, 200)

def _upload_cache_dir() -> Path:
    path = resolve_artifacts_dir(cfg) / "upload_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_upload_name(name: str) -> str:
    import re

    safe = os.path.basename(name or "uploaded_file").strip().replace(" ", "_")
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", safe)
    return safe[:120] or "uploaded_file"


def _persist_upload_to_disk(name: str, data: bytes) -> dict:
    root = _upload_cache_dir()
    digest = hashlib.sha256(data).hexdigest()[:16]
    safe_name = _safe_upload_name(name)
    path = root / f"{digest}__{safe_name}"
    if not path.exists():
        path.write_bytes(data)

    state = {
        "name": name,
        "path": str(path),
        "size_mb": round(len(data) / (1024 ** 2), 2),
        "sha256": digest,
    }
    (root / "latest_upload.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return state


def _restore_upload_from_disk() -> dict | None:
    latest = _upload_cache_dir() / "latest_upload.json"
    if not latest.exists():
        return None
    try:
        state = json.loads(latest.read_text(encoding="utf-8"))
        path = Path(str(state.get("path", "")))
        if path.exists() and path.is_file():
            return state
    except Exception:
        return None
    return None


def _clear_persisted_upload() -> None:
    root = _upload_cache_dir()
    latest = root / "latest_upload.json"
    if latest.exists():
        try:
            state = json.loads(latest.read_text(encoding="utf-8"))
            path = Path(str(state.get("path", "")))
            if path.exists() and path.is_file() and path.parent == root:
                path.unlink(missing_ok=True)
        except Exception:
            pass
        latest.unlink(missing_ok=True)

# ---------------- Sidebar: wspólne ustawienia ----------------
with st.sidebar:
    st.subheader("Import — ustawienia")
    mask_pii = st.checkbox("Maskuj PII (zalecane)", value=True)
    preview_rows = st.number_input(
        "Podgląd — maks. wierszy",
        1000,
        200_000,
        1000,
        step=1000,
    )
    st.caption(
        f"Limity: max plik {MAX_MB} MB; ostrzeżenie > {WARN_ROWS:,} wierszy; "
        f"próbkowanie > {SAMPLE_ROWS:,}."
    )

# ---------------- 1. ŹRÓDŁO DANYCH ----------------
st.header("1. Źródło danych")
st.write(
    "Wybierz, czy pracujesz na **własnym pliku**, czy na jednym z przygotowanych "
    "**zestawów demo** (klasyfikacja, regresja, klastrowanie, szeregi czasowe)."
)

source_mode = st.radio(
    "Skąd chcesz wziąć dane?",
    ["Mój plik", "Dane demo"],
    index=0,
    horizontal=True,
)

is_demo = source_mode == "Dane demo"

csv_sep = None
xlsx_sheet = None
pdf_pages = "1-end"
pdf_flavor = "lattice"

file_bytes: bytes | None = None
base_name = ""
meta = None
df_preview: pd.DataFrame | None = None
df_full_demo: pd.DataFrame | None = None
meta_full_demo = None
size_mb = None

if not is_demo:
    # ----- tryb: MÓJ PLIK -----
    st.caption("Wgraj plik CSV / XLSX / PDF / Parquet")

    # prosty mechanizm „resetu” uploadera przez zmianę key
    if "upload_rev" not in st.session_state:
        st.session_state.upload_rev = 0

    upload_state_key = "uploaded_file_state"

    def _clear_upload():
        """Czyści wgrany plik przez zmianę key uploadera."""
        st.session_state.upload_rev += 1
        st.session_state.pop(upload_state_key, None)
        _clear_persisted_upload()

    rev = st.session_state.upload_rev
    upload_key = f"file_uploader_main_{rev}"

    # --- jeden wiersz: po lewej dropzone, po prawej panel kasowania ---
    col_up, col_clear = st.columns([2, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Wgraj plik CSV / XLSX / PDF / Parquet",
            type=["csv", "xlsx", "xls", "pdf", "parquet"],
            key=upload_key,
            label_visibility="collapsed",
        )

    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        st.session_state[upload_state_key] = _persist_upload_to_disk(
            uploaded.name,
            uploaded_bytes,
        )

    uploaded_state = st.session_state.get(upload_state_key) or _restore_upload_from_disk()
    if uploaded_state is not None:
        st.session_state[upload_state_key] = uploaded_state
    has_active_upload = uploaded is not None or uploaded_state is not None
    active_file_name = (
        uploaded.name
        if uploaded is not None
        else str(uploaded_state.get("name", "")) if uploaded_state else ""
    )
    active_file_path = Path(str(uploaded_state.get("path", ""))) if uploaded_state else None
    active_file_bytes = active_file_path.read_bytes() if active_file_path and active_file_path.exists() else b""
    active_size_mb = round(len(active_file_bytes) / (1024 ** 2), 2) if active_file_bytes else 0.0

    with col_clear:
        box = st.container()
        if not has_active_upload:
            # pusty panel – zarezerwowane miejsce
            box.markdown(
                """
<div style="
    background-color:#fef9c3;
    border:1px solid #fde68a;
    border-radius:0.75rem;
    padding:0.55rem 0.9rem;
    font-size:0.8rem;
    color:#92400e;
">
  <strong>Nie ten plik? Skasuj ładowanie</strong><br>
  Plik nie został jeszcze załadowany.
</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            cached_label = "" if uploaded is not None else "<br><em>Uzywam zapamietanego pliku po rerunie.</em>"
            box.markdown(
                f"""
<div style="
    background-color:#fef3c7;
    border:1px solid #facc15;
    border-radius:0.75rem;
    padding:0.55rem 0.9rem;
    font-size:0.8rem;
    color:#92400e;
    margin-bottom:0.35rem;
">
  <strong>Nie ten plik? Skasuj ładowanie</strong><br>
  {active_file_name} - {active_size_mb:.2f} MB{cached_label}
</div>
                """,
                unsafe_allow_html=True,
            )
            box.button(
                "✖ Skasuj ładowanie",
                type="secondary",
                width='stretch',
                on_click=_clear_upload,
                key=f"clear_upload_{rev}",
            )

    # jeśli pliku brak – komunikat + stop (jak wcześniej)
    if not has_active_upload:
        st.info(
            "Wgraj plik, aby rozpocząć. Obsługujemy: CSV, XLSX, PDF (tabele) "
            "oraz Parquet. Jeśli chcesz tylko przetestować działanie, "
            "przełącz się na zakładkę **Dane demo**."
        )
        st.stop()

    # --- mamy plik: przygotowanie do wczytania podglądu ---
    if not active_file_bytes:
        st.warning("Nie moge odtworzyc pliku z uploadu. Wgraj plik ponownie.")
        st.stop()

    file_bytes = active_file_bytes
    name_lower = active_file_name.lower()
    size_mb = active_size_mb
    base_name = os.path.splitext(os.path.basename(active_file_name))[0]

    if size_mb > MAX_MB:
        st.error(f"Plik ma {size_mb} MB, a limit to {MAX_MB} MB.")
        st.stop()

    # ustawienia CSV/XLSX/PDF jak wcześniej – w sidebarze
    with st.sidebar:
        if name_lower.endswith(".csv"):
            st.markdown("**CSV — ustawienia**")
            sep_label = st.selectbox("Separator", ["Auto", ",", ";", "\\t"])
            csv_sep = {",": ",", ";": ";", "\\t": "\t"}.get(sep_label, None)

        elif name_lower.endswith((".xlsx", ".xls")):
            st.markdown("**XLSX — ustawienia**")
            try:
                sheets = excel_sheet_names(file_bytes)
            except Exception as e:
                st.error(f"Nie mogę odczytać listy arkuszy: {e}")
                sheets = []
            if sheets:
                xlsx_sheet = st.selectbox("Arkusz", sheets, index=0)

        elif name_lower.endswith(".pdf"):
            st.markdown("**PDF — ustawienia**")
            pdf_pages = st.text_input("Zakres stron (np. 1, 1-3, 1-end)", "1-end")
            pdf_flavor = st.radio(
                "Tryb detekcji tabel",
                ["lattice", "stream"],
                index=0,
                horizontal=True,
            )

        elif name_lower.endswith(".parquet"):
            st.markdown("**Parquet — ustawienia**")
            st.caption("Format kolumnowy – nie wymaga dodatkowych ustawień.")


    with st.spinner("Wczytywanie podglądu…"):
        try:
            df_preview, meta = load_any(
                file_bytes,
                active_file_name,
                preview_limit=preview_rows,
                csv_sep=csv_sep,
                xlsx_sheet=xlsx_sheet,
                pdf_pages=pdf_pages,
                pdf_flavor=pdf_flavor,
            )
        except Exception as e:
            st.exception(e)
            st.stop()
else:
    # ----- tryb: DANE DEMO -----
    st.markdown(
        "Dane demo pozwalają **bez ryzyka** zobaczyć cały pipeline: od wczytania, "
        "przez EDA, aż po trenowanie modelu – bez przygotowywania własnego pliku."
    )

    pycaret_demos = _build_pycaret_demo_index(PYCARET_DEMO_INDEX_CACHE_VERSION)
    public_demos = PUBLIC_TS_DEMOS

    all_demo_specs: dict[str, dict] = {}
    all_demo_specs.update(pycaret_demos)
    all_demo_specs.update(public_demos)

    def _sort_key(k: str):
        spec = all_demo_specs[k]
        return (spec.get("task", ""), k.lower())

    pycaret_options = sorted(pycaret_demos.keys(), key=_sort_key)
    public_options = sorted(public_demos.keys(), key=_sort_key)

    # Najpierw zestawy PyCaret, potem dane publiczne z DO
    options = pycaret_options + public_options

    demo_label = st.selectbox(
        "Wybierz zestaw demo",
        options,
        index=0,
    )
    demo_spec = all_demo_specs[demo_label]

    # --- STAN DEMO W SESSION_STATE (żeby nie znikało po kliknięciu innych przycisków) ---
    demo_state_key = "demo_ingest_state"
    demo_state = st.session_state.get(demo_state_key, {})

    already_loaded = (
        demo_state.get("label") == demo_label
        and demo_state.get("df_preview") is not None
        and demo_state.get("df_full_demo") is not None
        and demo_state.get("meta") is not None
        and demo_state.get("meta_full_demo") is not None
    )

    load_demo = st.button(
        "Załaduj dane demo",
        type="primary",
        help="Jeśli zmienisz zestaw demo, kliknij ponownie, aby przeładować dane.",
    )

    # 1) Pierwsze użycie – wymagamy kliknięcia przycisku
    if not already_loaded and not load_demo:
        st.info("Wybierz zestaw i kliknij **Załaduj dane demo**, aby przejść dalej.")
        st.stop()

    # 2) Gdy użytkownik kliknie przycisk LUB zmieni wybrany zestaw – przeładuj dane
    if load_demo or not already_loaded:
        with st.spinner("Wczytywanie danych demo…"):
            (
                df_preview,
                df_full_demo,
                meta,
                meta_full_demo,
                size_mb,
            ) = _load_demo_dataset(demo_spec, preview_limit=preview_rows)

        demo_state = {
            "label": demo_label,
            "df_preview": df_preview,
            "df_full_demo": df_full_demo,
            "meta": meta,
            "meta_full_demo": meta_full_demo,
            "size_mb": size_mb,
        }
        st.session_state[demo_state_key] = demo_state
    else:
        # 3) Kolejne reruny – korzystamy z danych zapisanych w stanie
        df_preview = demo_state["df_preview"]
        df_full_demo = demo_state["df_full_demo"]
        meta = demo_state["meta"]
        meta_full_demo = demo_state["meta_full_demo"]
        size_mb = demo_state["size_mb"]

    base_name = str(meta.source_name)


# ---- wspólna dalsza część (po wczytaniu df_preview + meta) ----
if meta.n_rows > WARN_ROWS:
    st.warning(f"Duży zbiór (>{WARN_ROWS:,} wierszy). Uważaj na ciężkie operacje.")
if meta.n_rows >= preview_rows:
    st.info(
        "To jest **podgląd** (próbka danych). Pełny zbiór zostanie wczytany "
        "dopiero przy kroku 👉 **Przelicz na całości i zapisz artefakty**."
    )

# ---------------- 2. STATUS + KPI ----------------
st.header("2. Status wczytania i krótkie KPI")

# Spinner obejmuje maskowanie PII na podglądzie – przy dużych podglądach
# użytkownik widzi, że aplikacja „pracuje”, a nie zawiesiła się.
with st.spinner("Przygotowywanie podglądu (maskowanie PII)…"):
    df_masked_preview, pii_report_preview = (
        mask_dataframe(df_preview) if mask_pii else (df_preview, {})
    )

st.success(
    f"Wczytano: **{meta.source_name}**"
    + (f" · {size_mb} MB" if size_mb is not None else "")
    + f" · podgląd: {meta.n_rows}×{meta.n_cols} · silnik: {meta.engine}"
    + (f" · kodowanie: {meta.encoding}" if getattr(meta, "encoding", None) else "")
    + (f" · notatki: {meta.notes}" if getattr(meta, "notes", None) else "")
)

_show_kpi_block(df_masked_preview, mask_pii=mask_pii, meta_obj=meta)

# ---------------- 3. PODGLĄD DANYCH ----------------
st.header("3. Podgląd danych (próbka)")
st.caption(
    "Poniżej widzisz maksymalnie tyle wierszy, ile określono w ustawieniach "
    "podglądu. Pełny zbiór zostanie wczytany dopiero przy kroku "
    "👉 **Przelicz na całości i zapisz artefakty**."
)

mode = st.radio(
    "Tryb podglądu tabeli",
    ["Przewijalna (szybka)", "Paginacja + wyszukiwarka (itables)"],
    index=0,
    horizontal=True,
    help=(
        "Przewijalna (st.dataframe) ma sticky header i spójny font. "
        "Tryb itables daje paginację i wyszukiwarkę."
    ),
)

if mode.startswith("Przewijalna"):
    _preview_download_buttons(df_masked_preview)

    height = st.slider(
        "Wysokość widoku (px)",
        min_value=300,
        max_value=1200,
        value=650,
        step=50,
        key="scroll_height",
    )

    st.dataframe(
        df_masked_preview,
        width='stretch',
        height=height,
        hide_index=True,
    )

else:
    # === ITABLES VIEW: paginacja + wyszukiwarka ===

    _preview_download_buttons(df_masked_preview)

    scrolly = st.slider(
        "Wysokość tabeli (px, itables)",
        300,
        1200,
        600,
        20,
        key="itable_scrolly",
    )
    page_len = st.selectbox(
        "Wierszy na stronę",
        (10, 25, 50),
        index=1,
        key="itable_page_len",
    )

    html, iframe_h = _render_itables_html(
        df_masked_preview,
        body_px=int(scrolly),
        page_len=int(page_len),
    )

    st_html(
        html,
        height=int(iframe_h),
        scrolling=False,
    )

    st.markdown(
        """
        <style>
        div[data-testid="stIFrame"] {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stIFrame"]) {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        div[data-testid="stVerticalBlock"] > div:last-child {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        div.element-container {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.download_button(
    "📦 Pobierz artefakty podglądu (ZIP)",
    data=_zip_bytes(df_masked_preview, meta, base_name),
    file_name="ingest_preview.zip",
    mime="application/zip",
    help=(
        "ZIP zawiera: preview_masked.csv (po maskowaniu PII) i meta.json "
        "z informacjami o źródle."
    ),
)

st.write("---")

# ---------------- 4. PRZELICZ NA CAŁOŚCI ----------------
st.header("4. Przelicz na całości i zapisz artefakty")

st.caption(
    "To polecenie wczytuje **pełne dane (cały plik / cały zestaw demo)**, maskuje PII, "
    "zapisuje artefakty lokalnie i oznacza je jako gotowe do trenowania modelu. "
    "Ten krok jest obowiązkowy – bez niego zakładka „Trenowanie Modelu” "
    "nie będzie miała gotowych danych."
)

tooltip_text = (
    "Uruchamia pełne przetwarzanie danych:\n"
    "• wczytuje CAŁY plik / zestaw demo (nie tylko podgląd),\n"
    "• maskuje PII,\n"
    "• zapisuje dane i metadane lokalnie,\n"
    "• ustawia te dane jako 'gotowe do uczenia modelu'.\n\n"
    "Po tym kroku możesz bezpiecznie przejść do zakładki "
    "'Trenowanie Modelu' bez ponownego wgrywania pliku."
)

def _request_full_run():
    st.session_state["full_run_requested"] = True


clicked = st.button(
    "🔄 Przelicz teraz (pełny zbiór)",
    key="full_run_button",
    type="primary",
    help=tooltip_text,
    on_click=_request_full_run,
)

if st.session_state.get("full_run_requested", False):
    # Pasek statusu obejmuje CAŁY pipeline:
    # 1/3 – wczytywanie pełnego zbioru
    # 2/3 – maskowanie PII
    # 3/3 – zapis artefaktów (CSV + meta.json)
    with st.status(
        "1/3 – Wczytywanie pełnego zbioru…",
        expanded=False,
    ) as status:
        # ------------------ Krok 1/3: wczytywanie pełnego zbioru ------------------
        status.write("1/3 – Wczytywanie pełnego zbioru do pamięci…")

        if is_demo:
            df_full = df_full_demo.copy()
            meta_full = meta_full_demo
            status.write("✔ 1/3 – Wczytywanie pełnego zbioru zakończone (demo).")
        else:
            try:
                df_full, meta_full = load_any(
                    file_bytes,
                    active_file_name,
                    preview_limit=None,
                    csv_sep=csv_sep,
                    xlsx_sheet=xlsx_sheet,
                    pdf_pages=pdf_pages,
                    pdf_flavor=pdf_flavor,
                )
                status.write("✔ 1/3 – Wczytywanie pełnego zbioru zakończone (plik użytkownika).")
            except Exception as e:
                status.update(
                    label="❌ Błąd podczas wczytywania pełnego zbioru",
                    state="error",
                )
                st.exception(e)
                st.session_state["full_run_requested"] = False
                st.stop()

        status.update(
            label="2/3 – Maskowanie PII…",
            state="running",
        )

        # ------------------ Krok 2/3: maskowanie PII na pełnym zbiorze ------------------
        if mask_pii:
            status.write("2/3 – Maskowanie PII w pełnym zbiorze…")
        else:
            status.write("2/3 – Pomijanie maskowania PII (opcja wyłączona)…")

        try:
            df_full_masked, pii_report_full = (
                mask_dataframe(df_full) if mask_pii else (df_full, {})
            )
        except Exception as e:
            status.update(
                label="Blad podczas maskowania PII",
                state="error",
            )
            st.exception(e)
            st.session_state["full_run_requested"] = False
            st.stop()

        status.write("✔ 2/3 – Etap maskowania PII zakończony.")

        status.update(
            label="3/3 – Zapis artefaktów (CSV + meta.json)…",
            state="running",
        )

        # ------------------ Krok 3/3: zapis artefaktów ------------------
        out_dir = resolve_artifacts_dir(cfg) / "ingest"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Bezpieczna nazwa dla systemu plików (bez dwukropków itp.)
        def _slugify_for_fs(name: str) -> str:
            import re

            if not name:
                return "dataset"
            # najpierw spacje -> podkreślenia
            name = name.strip().replace(" ", "_")
            # zostaw tylko bezpieczne znaki (litery, cyfry, ., _, -)
            name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
            # przytnij, żeby nie robić super długich ścieżek
            name = name[:80]
            return name or "dataset"

        safe_base = _slugify_for_fs(base_name)
        run_dir = out_dir / f"{safe_base}__{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = run_dir / f"{safe_base}__full_masked.parquet"
        meta_path = run_dir / f"{safe_base}__meta.json"

        status.write("3/3 – Zapis pełnego zbioru do PARQUET (snappy)…")
        df_full_masked = _optimize_dtypes_for_storage(df_full_masked)
        _df_to_parquet(df_full_masked, parquet_path)

        meta_dump = meta_full.__dict__ | {
            "pii_masked": bool(mask_pii),
            "pii_changes": pii_report_full,
            "source_kind": "demo" if is_demo else "uploaded",
        }
        meta_path.write_text(
            json.dumps(meta_dump, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        st.session_state["latest_artifacts"] = {
            "parquet_path": str(parquet_path),
            "meta_path": str(meta_path),
            "run_dir": str(run_dir),
            "n_rows": int(df_full_masked.shape[0]),
            "n_cols": int(df_full_masked.shape[1]),
            "pii_masked": bool(mask_pii),
            "source_name": meta_full.source_name,
            "timestamp": ts,
            "source_kind": "demo" if is_demo else "uploaded",
        }

        status.write("✔ 3/3 – Artefakty zapisane. Dane są gotowe do trenowania modelu.")
        status.update(
            label="Pipeline pełnego przygotowania danych zakończony ✅",
            state="complete",
        )

    # Po wyjściu z kontekstu st.status wyświetlamy końcowe podsumowanie
    st.success(
        "✅ Dane przygotowane do trenowania modelu.\n\n"
        f"- Pełny zbiór (po maskowaniu PII): `{parquet_path}`  \n"
        f"- Meta-informacje: `{meta_path}`  \n"
        f"- Rozmiar danych: {df_full_masked.shape[0]} wierszy × "
        f"{df_full_masked.shape[1]} kolumn\n\n"
        "Możesz teraz przejść do zakładki **Trenowanie Modelu** — "
        "aplikacja automatycznie użyje tych danych (`latest_artifacts`)."
    )

# --- Powtórzony potok na dole strony ---
if st.session_state.get("full_run_requested", False):
    st.session_state["full_run_requested"] = False

render_flow_nav(current_id="01_Analiza_Danych", key_prefix="flow_bottom")
st.markdown("---")

