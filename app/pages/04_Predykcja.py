import streamlit as st
from core.i18n import t
from ui.components import page_header
page_header(t("page.predict.title"))
st.info("Etap 6 doda: walidację schematu, batch/punktową predykcję i eksport wyników.")
