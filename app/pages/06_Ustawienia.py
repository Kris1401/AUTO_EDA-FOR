import streamlit as st
from core.i18n import t
from core.config import snapshot_masked_env
from ui.components import page_header
page_header(t("page.settings.title"))
st.write(t("settings.desc"))
with st.expander(t("settings.env_vars"), expanded=True):
    st.code("OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,\nJWT_SIGNING_KEY, ADMIN_TOKEN")
with st.expander(t("settings.values"), expanded=True):
    st.json(snapshot_masked_env())
