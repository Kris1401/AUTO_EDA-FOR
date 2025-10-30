# app/ui/theme.py
import streamlit as st

PALETTE = {
    "PRIMARY": "#2e6cff",
    "SECONDARY": "#a347ff",
    "ACCENT1": "#00d4ff",
    "ACCENT2": "#ff8a00",
    "BACKGROUND": "#0b1423",
    "SURFACE": "#111a2b",
    "TEXT": "#e9f0ff",
}

CSS_TMPL = """
<style>
:root {
  --primary: {PRIMARY};
  --secondary: {SECONDARY};
  --accent1: {ACCENT1};
  --accent2: {ACCENT2};
  --bg: {BACKGROUND};
  --surface: {SURFACE};
  --text: {TEXT};
  /* globalny krój – ujednolica tabele Streamlit i DataTables */
  --font-main: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
               Arial, "Noto Sans", "Liberation Sans", sans-serif;
}

.stApp, .stApp p, .stApp .stMarkdown, .stApp table {
  font-family: var(--font-main);
  color: var(--text);
}
.stApp {
  background: radial-gradient(80% 120% at 50% 0%, var(--bg) 0%, #050a14 100%);
}

/* więcej miejsca na dole strony, żeby nie „ucinało” paginacji */
.main .block-container { padding-bottom: 6rem; }

/* ITables / DataTables – pełna szerokość + ujednolicona czcionka */
.itables, .dataTables_wrapper { width: 100% !important; }
.dataTables_wrapper,
.dataTables_wrapper .dataTables_info,
.dataTables_wrapper .dataTables_paginate,
.dataTables_wrapper .dataTables_length,
.dataTables_wrapper .dataTables_filter,
table.dataTable, table.dataTable th, table.dataTable td {
  font-family: var(--font-main) !important;
  font-size: 0.95rem;
}
/* dodatkowy „oddech” pod paginacją, by dało się wygodnie kliknąć */
.dataTables_wrapper .dataTables_paginate {
  margin-top: .75rem !important;
  padding-bottom: 1.6rem !important;
}

/* drobny akcent do tytułów (bez zmian funkcjonalnych) */
.app-gradient-title {
  background: linear-gradient(90deg, var(--primary), var(--secondary));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
</style>
"""

def apply_theme():
    """
    Wstrzykujemy CSS do Streamlit, podstawiając wartości kolorów ręcznie.
    To omija problem z KeyError przy .format_map() gdy w placeholderze
    pojawiają się spacje / nowe linie.
    Zakładamy, że PALETTE to słownik w stylu:
        {
            "--primary": "#5a48f5",
            "--bg-page": "#0f0f1a",
            ...
        }
    a CSS_TMPL zawiera te tokeny literalnie, np. "color: var(--primary);".
    """

    css_filled = CSS_TMPL

    # podmień każde wystąpienie nazwy zmiennej na wartość
    # przykład: "--primary" -> "#5a48f5"
    for var_name, var_value in PALETTE.items():
        # podstawiamy zarówno w postaci "var(--primary)" jak i sam "--primary"
        css_filled = css_filled.replace(f"var({var_name})", var_value)
        css_filled = css_filled.replace(var_name, var_value)

    st.markdown(css_filled, unsafe_allow_html=True)

