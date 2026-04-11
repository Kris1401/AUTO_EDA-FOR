import streamlit as st
from core.i18n import t
from ui.components import page_header
from core.top_nav import (
    hide_default_multipage_nav,
    render_flow_nav,
    render_sidebar_links,
)


hide_default_multipage_nav()
render_flow_nav(current_id="04_Trenowanie_Modelu", key_prefix="flow_top")

page_header(t("page.ml.title"))
st.info("Etap 4 doda: Auto-ML (clf/regr/cluster/TS), tuning, stabilność i FAST mode.")

art = st.session_state.get("latest_artifacts")
if art:
    st.success(
        f"Znaleziono przygotowany zbiór: {art['csv_path']} "
        f"({art['n_rows']}×{art['n_cols']})"
    )
    # dalej: wczytaj csv_path -> df, itd.
else:
    st.warning(
        "Brak przygotowanych danych. Wróć do zakładki 'Analiza Danych' "
        "i wykonaj krok pełnego przeliczenia."
    )

# --- Powtórzony potok na dole strony ---
render_flow_nav(current_id="04_Trenowanie_Modelu", key_prefix="flow_bottom")
st.markdown("---")
