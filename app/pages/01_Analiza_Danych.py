# app/pages/01_Analiza_Danych.py
from __future__ import annotations
import os, io, json, zipfile
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd

from core.i18n import t
from core.config import load_config, resolve_artifacts_dir
from core.pii import mask_dataframe
from ingest import load_any, excel_sheet_names

from streamlit.components.v1 import html as st_html
from itables import to_html_datatable

try:
    from itables.streamlit import interactive_table
    HAS_ITABLES = True
except Exception:
    HAS_ITABLES = False

try:
    from itables import options as itbl_options
    # Ustawienia DataTables — domyślnie 25 wierszy i pionowy scroll
    itbl_options.classes = "display stripe compact"
    itbl_options.pageLength = 25
    itbl_options.lengthMenu = [10, 25, 50]
    # itbl_options.scrollY = "60vh"       # przewijanie wewnątrz tabeli
    # itbl_options.scrollCollapse = True
    # itbl_options.scrollX = True
except Exception:
    pass


# === itables via iframe (DataTables) — render helper (minimal patch) ===
def _render_itables_html(df, body_px: int, page_len: int) -> tuple[str, int]:
    """
    body_px  -> wysokość przewijanej części tabeli (scrollY) sterowana suwakiem
    page_len -> entries per page (np. 10/25/50/100)
    Zwraca: (html, iframe_height)
    """
    import json as _json

    # Stałe „nad i pod” ciałem tabeli (nagłówek, info+pagination, obramowania)
    HEADER = 56
    FOOTER = 54
    BORDERS = 10
    iframe_h = int(body_px + HEADER + FOOTER + BORDERS)

    # Dane i kolumny
    records = df.to_dict(orient="records")
    columns = [{"title": c, "data": c} for c in df.columns]

    html = f"""
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>

<style>
  html, body, #wrap {{ margin:0 !important; padding:0 !important; }}
  .dataTables_scroll {{ margin-bottom:0 !important; }}  /* zero luzu pod scrollem */
  .dataTables_wrapper .dataTables_info {{ margin-top:6px !important; }}
  .dataTables_wrapper .dataTables_paginate {{ margin-top:6px !important; }}
  /* spójna czcionka jak w reszcie aplikacji */
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
  const scrollY = {int(body_px)};  // sterowane suwakiem
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

def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Zwraca bytes gotowego pliku XLSX.
    Najpierw próbujemy użyć xlsxwriter (jeśli jest zainstalowany),
    a jeśli go nie ma – fallback do openpyxl.
    """
    buf = io.BytesIO()

    # spróbuj xlsxwriter
    try:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="data", index=False)
    except ModuleNotFoundError:
        # brak xlsxwriter -> użyj openpyxl
        buf = io.BytesIO()  # nowy bufor, żeby nie zwrócić śmieci po nieudanej próbie
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="data", index=False)

    # bardzo ważne: przewiń bufor na początek przed zwrotem
    buf.seek(0)
    return buf.getvalue()


def _preview_download_buttons(df_view: pd.DataFrame):
    """
    Rysuje przyciski pobierania podglądu danych (CSV / XLSX),
    żeby użytkownik mógł od razu zabrać dane 'jak je widzi', tzn.
    po ewentualnym maskowaniu PII.
    """
    col_csv, col_xlsx, _ = st.columns([1, 1, 6])

    with col_csv:
        st.download_button(
            "⬇ CSV (podgląd)",
            data=df_view.to_csv(index=False).encode("utf-8"),
            file_name="preview_masked.csv",
            mime="text/csv",
            key="dl_csv_preview",
            help="Eksport podglądu (po maskowaniu PII) do CSV."
        )

    with col_xlsx:
        st.download_button(
            "⬇ XLSX (podgląd)",
            data=_to_excel_bytes(df_view),
            file_name="preview_masked.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_xlsx_preview",
            help="Eksport podglądu (po maskowaniu PII) do Excela."
        )

st.title("Analiza Danych — Wczytywanie (Etap 1)")

cfg, problems = load_config()
MAX_MB, WARN_ROWS, SAMPLE_ROWS = cfg.max_file_mb, cfg.warn_rows, cfg.sample_rows

# ---------------- Sidebar: wspólne ustawienia ----------------
with st.sidebar:
    st.subheader("Import — ustawienia")
    mask_pii = st.checkbox("Maskuj PII (zalecane)", value=True)
    preview_rows = st.number_input("Podgląd — maks. wierszy", 1000, 200_000, 5000, step=1000)
    st.caption(f"Limity: max plik {MAX_MB} MB; ostrzeżenie > {WARN_ROWS:,} wierszy; próbkowanie > {SAMPLE_ROWS:,}.")

uploaded = st.file_uploader("Wgraj plik CSV / XLSX / PDF", type=["csv","xlsx","xls","pdf"])

if not uploaded:
    st.info("Wgraj plik, aby rozpocząć. Obsługujemy: CSV, XLSX, PDF (tabele).")
    st.stop()

# plik w pamięci (możemy czytać wielokrotnie)
file_bytes = uploaded.getvalue()
name_lower = uploaded.name.lower()
size_mb = round(len(file_bytes) / (1024**2), 2)
if size_mb > MAX_MB:
    st.error(f"Plik ma {size_mb} MB, a limit to {MAX_MB} MB.")
    st.stop()

# ---------------- Dynamiczne opcje per typ ----------------
csv_sep = None
xlsx_sheet = None
pdf_pages = "1-end"
pdf_flavor = "lattice"

with st.sidebar:
    if name_lower.endswith(".csv"):
        st.markdown("**CSV — ustawienia**")
        sep_label = st.selectbox("Separator", ["Auto", ",", ";", "\\t"])
        csv_sep = {",": ",", ";": ";", "\\t": "\t"}.get(sep_label, None)

    elif name_lower.endswith((".xlsx",".xls")):
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
        pdf_flavor = st.radio("Tryb detekcji tabel", ["lattice","stream"], index=0, horizontal=True)

# ---------------- Wczytanie podglądu ----------------
with st.spinner("Wczytywanie podglądu…"):
    try:
        df_preview, meta = load_any(
            file_bytes, uploaded.name,
            preview_limit=preview_rows,
            csv_sep=csv_sep,
            xlsx_sheet=xlsx_sheet,
            pdf_pages=pdf_pages,
            pdf_flavor=pdf_flavor,
        )
    except Exception as e:
        st.exception(e)
        st.stop()

if meta.n_rows > WARN_ROWS:
    st.warning(f"Duży zbiór (>{WARN_ROWS:,} wierszy). Uważaj na ciężkie operacje.")
if meta.n_rows >= preview_rows:
    st.info("To jest **podgląd**. Użyj przycisku niżej, aby przeliczyć **na pełnym zbiorze**.")

# ---------------- Maskowanie PII (podgląd) ----------------
df_masked_preview, pii_report_preview = mask_dataframe(df_preview) if mask_pii else (df_preview, {})

st.success(
    f"Wczytano: **{meta.source_name}** · {size_mb} MB · "
    f"podgląd: {meta.n_rows}×{meta.n_cols} · silnik: {meta.engine}"
    + (f" · kodowanie: {meta.encoding}" if meta.encoding else "")
    + (f" · notatki: {meta.notes}" if meta.notes else "")
)

with st.expander("Podsumowanie i schema", expanded=True):
    st.write("**Wymiary (podgląd):**", df_preview.shape)
    st.write("**Typy kolumn (podgląd):**")
    st.dataframe(pd.DataFrame({"dtype": df_preview.dtypes.astype(str)}))
    nulls = df_preview.isna().sum()
    st.write("**Braki danych (podgląd):**")
    st.dataframe(pd.DataFrame({"nulls": nulls, "null_%": (nulls/len(df_preview))*100}).round(2))
    if pii_report_preview:
        st.write("**Maskowanie PII – liczba zmian (podgląd):**")
        st.json(pii_report_preview)

st.subheader("Podgląd danych")

# Wybór trybu renderu tabeli
mode = st.radio(
    "Tryb podglądu tabeli",
    ["Przewijalna (szybka)", "Paginacja + wyszukiwarka (itables)"],
    index=0,
    horizontal=True,
    help="Przewijalna (st.dataframe) ma sticky header i spójny font. "
         "Tryb itables daje paginację i wyszukiwarkę."
)

if mode.startswith("Przewijalna"):
    # przyciski pobierania (CSV / XLSX) - zawsze patrzymy na df_masked_preview
    _preview_download_buttons(df_masked_preview)

    # wysokość widoku do wygodnego scrollowania
    height = st.slider(
        "Wysokość widoku (px)",
        min_value=300,
        max_value=1200,
        value=650,
        step=50,
        key="scroll_height"
    )

    st.dataframe(
        df_masked_preview,
        use_container_width=True,
        height=height,
        hide_index=True,
    )


else:
    # === ITABLES VIEW: paginacja + wyszukiwarka ===
    from itables import to_html_datatable

    # przyciski pobierania (CSV / XLSX) - nad tabelą
    _preview_download_buttons(df_masked_preview)

    # 1) Suwak wysokości przewijania w środku tabeli (px) + wybór liczby wierszy
    scrolly = st.slider(
        "Wysokość tabeli (px, itables)", 300, 1200, 600, 20,
        key="itable_scrolly"
    )
    page_len = st.selectbox(
        "Wierszy na stronę", (10, 25, 50), index=1, key="itable_page_len"
    )
    
    # 3) wygeneruj pełny HTML tabeli (z JS i CSS) + policz wysokość iframe
    #    WAŻNE: używamy df_masked_preview, żeby respektować maskowanie PII
    html, iframe_h = _render_itables_html(
        df_masked_preview,
        body_px=int(scrolly),
        page_len=int(page_len),
    )

    # 4) wstaw gotowy HTML (DataTables) do iframa o policzonej wysokości
    st_html(
        html,
        height=int(iframe_h),
        scrolling=False,
    )

    # 5) CSS hack – zbij zbędny „oddech” pod komponentem iframe,
    #    który Streamlit potrafi dodać jako margin/padding.
    st.markdown(
        """
        <style>
        /* iframe wygenerowane przez components.html / st_html */
        div[data-testid="stIFrame"] {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }

        /* wrapper tuż POD iframe w pionowym layoucie Streamlita */
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stIFrame"]) {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }

        /* ostatnie dziecko w kolumnie również bez wielkiego dołu */
        div[data-testid="stVerticalBlock"] > div:last-child {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }

        /* każdy element-container — też przytnij dolny margines */
        div.element-container {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Przyciski: ZIP (podgląd) + pełne przeliczenie ----------------
def _zip_bytes(df_out: pd.DataFrame, meta_obj) -> bytes:
    buf = io.BytesIO()
    base = os.path.splitext(os.path.basename(uploaded.name))[0]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{base}__preview_masked.csv", df_out.to_csv(index=False))
        z.writestr(f"{base}__meta.json", json.dumps(meta_obj.__dict__, ensure_ascii=False, indent=2))
    return buf.getvalue()

st.download_button(
    "📦 Pobierz artefakty podglądu (ZIP)",
    data=_zip_bytes(df_masked_preview, meta),
    file_name="ingest_preview.zip",
    mime="application/zip",
    help="ZIP zawiera: preview_masked.csv (po maskowaniu PII) i meta.json z informacjami o źródle."
)

st.write("---")
# --- KROK: pełne przetworzenie danych i zapis artefaktów ---
st.subheader("Przelicz na całości i zapisz artefakty")

# stałe wyjaśnienie (widoczne ZAWSZE, zanim ktoś kliknie)
st.caption(
    "To polecenie wczytuje pełne dane (cały plik), maskuje PII, "
    "zapisuje artefakty lokalnie i oznacza je jako gotowe "
    "do trenowania modelu. Ten krok jest obowiązkowy – "
    "bez niego zakładka „Trenowanie Modelu” nie będzie miała gotowych danych."
)

tooltip_text = (
    "Uruchamia pełne przetwarzanie danych:\n"
    "• wczytuje CAŁY plik (nie tylko podgląd),\n"
    "• maskuje PII,\n"
    "• zapisuje dane i metadane lokalnie,\n"
    "• ustawia te dane jako 'gotowe do uczenia modelu'.\n\n"
    "Po tym kroku możesz bezpiecznie przejść do zakładki "
    "'Trenowanie Modelu' bez ponownego wgrywania pliku."
)

clicked = st.button(
    "🔄 Przelicz teraz (pełny zbiór)",
    key="full_run_button",
    type="primary",
    help=tooltip_text,
    use_container_width=False,
)

if clicked:
    # 1. wczytaj CAŁY zbiór bez limitu wierszy
    with st.spinner("Wczytywanie pełnego zbioru…"):
        try:
            df_full, meta_full = load_any(
                file_bytes,
                uploaded.name,
                preview_limit=None,      # <-- pełny zbiór
                csv_sep=csv_sep,
                xlsx_sheet=xlsx_sheet,
                pdf_pages=pdf_pages,
                pdf_flavor=pdf_flavor,
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    # 2. maskowanie PII na całości
    with st.spinner("Maskowanie PII…"):
        df_full_masked, pii_report_full = (
            mask_dataframe(df_full) if mask_pii else (df_full, {})
        )

    # 3. zapis plików wyjściowych do katalogu artefaktów
    out_dir = resolve_artifacts_dir(cfg) / "ingest"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(uploaded.name))[0]
    run_dir = out_dir / f"{base}__{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / f"{base}__full_masked.csv"
    meta_path = run_dir / f"{base}__meta.json"

    # zapis danych z PII zamaskowanym (lub oryginalnych, jeśli mask_pii=False)
    df_full_masked.to_csv(csv_path, index=False, encoding="utf-8")

    # metadane + info o maskowaniu
    meta_dump = meta_full.__dict__ | {
        "pii_masked": bool(mask_pii),
        "pii_changes": pii_report_full,
    }
    meta_path.write_text(
        json.dumps(meta_dump, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4. przekaż opis najnowszego przetworzenia dalej (do zakładki "Trenowanie Modelu")
    st.session_state["latest_artifacts"] = {
        "csv_path": str(csv_path),
        "meta_path": str(meta_path),
        "run_dir": str(run_dir),
        "n_rows": int(df_full_masked.shape[0]),
        "n_cols": int(df_full_masked.shape[1]),
        "pii_masked": bool(mask_pii),
        "source_name": meta_full.source_name,
        "timestamp": ts,
    }

    # 5. komunikat końcowy dla użytkownika — tylko zielony sukces,
    #    bez dodatkowego niebieskiego info bloku
    st.success(
        "✅ Dane przygotowane do trenowania modelu.\n\n"
        f"• Pełny zbiór (po maskowaniu PII): `{csv_path}`\n"
        f"• Meta-informacje: `{meta_path}`\n"
        f"• Rozmiar danych: {df_full_masked.shape[0]} wierszy × "
        f"{df_full_masked.shape[1]} kolumn\n\n"
        "Możesz teraz przejść do zakładki **Trenowanie Modelu** — "
        "aplikacja automatycznie użyje tych danych (`latest_artifacts`)."
    )

