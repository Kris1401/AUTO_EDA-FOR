import streamlit as st
from core.i18n import t
from ui.components import page_header
page_header(t("page.ml.title"))
st.info("Etap 4 doda: Auto-ML (clf/regr/cluster/TS), tuning, stabilność i FAST mode.")


art = st.session_state.get("latest_artifacts")
if art:
    st.success(f"Znaleziono przygotowany zbiór: {art['csv_path']} ({art['n_rows']}×{art['n_cols']})")
    # dalej: wczytaj CSV_path -> df, itd.
else:
    st.warning("Brak przygotowanych danych. Wróć do zakładki 'Analiza Danych' i wykonaj krok pełnego przeliczenia.")
