import streamlit as st
def page_header(title: str, subtitle: str = ""):
    st.markdown(f"<h1 class='app-gradient-title'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)
