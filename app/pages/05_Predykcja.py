import streamlit as st
from core.i18n import t
from ui.components import page_header
from core.top_nav import (
    hide_default_multipage_nav,
    render_flow_nav,
    render_sidebar_links,
)


hide_default_multipage_nav()
render_flow_nav(current_id="05_Predykcja", key_prefix="flow_top")

page_header(t("page.predict.title"))
st.info("Etap 5 doda: walidację schematu, batch/punktową predykcję i eksport wyników.")

# --- Powtórzony potok na dole strony ---
render_flow_nav(current_id="05_Predykcja", key_prefix="flow_bottom")
st.markdown("---")
