import streamlit as st
from core.i18n import t
from ui.components import page_header
page_header(t("page.chat.title"))
st.info("Etap 3 doda: parser NL→(wykres/liczby/tabela), interpretację i guardraile.")
