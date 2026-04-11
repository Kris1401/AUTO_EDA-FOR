import streamlit as st
from core.top_nav import hide_default_multipage_nav


hide_default_multipage_nav()

if hasattr(st, "switch_page"):
    try:
        st.switch_page("app.py")
        st.stop()
    except Exception:
        pass

st.info("Ta strona została wycofana z nawigacji. Użyj potoku etapów na stronie głównej.")
