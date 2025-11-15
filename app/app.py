import streamlit as st
from core.i18n import t, get_locale, set_locale
from ui.theme import apply_theme
from core.config import load_config
import os

st.set_page_config(page_title="AUTO EDA FOR", page_icon="🔮", layout="wide")
apply_theme()

lang = st.sidebar.selectbox("Language / Język", options=["PL", "EN"], index=0 if get_locale()=="PL" else 1)
set_locale(lang)

st.title(t("app.title"))
st.caption(t("app.subtitle"))

cfg, problems = load_config()
with st.expander(t("home.config_status"), expanded=True):
    ok = len(problems) == 0
    if ok:
        st.success(t("home.config_ok"))
    else:
        st.warning(t("home.config_warn"))
        # elegancka lista braków – bez help()/docstringów
        for p in problems:
            st.write(f"• {p}")

st.markdown(t("home.next_steps"))
st.info(t("home.nav_info"))
st.write("---")
st.write(t("home.footer_note"))
