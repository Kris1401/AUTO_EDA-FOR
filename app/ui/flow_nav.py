from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

KEYCAP_DIGITS = {
    "1": "1\ufe0f\u20e3",
    "2": "2\ufe0f\u20e3",
    "3": "3\ufe0f\u20e3",
    "4": "4\ufe0f\u20e3",
    "5": "5\ufe0f\u20e3",
}

PIPELINE = [
    {
        "id": "01_Analiza_Danych",
        "page_id": "01_Analiza_Danych",
        "target": "pages/01_Analiza_Danych.py",
        "label": "Analiza Danych",
        "step": "1",
        "optional": False,
    },
    {
        "id": "02_Automat_EDA",
        "page_id": "02_Automat_EDA",
        "target": "pages/02_Automat_EDA.py",
        "label": "Automat EDA",
        "step": "2",
        "optional": False,
    },
    {
        "id": "03_Data_Chat",
        "page_id": "03_Data_Chat",
        "target": "pages/03_Data_Chat.py",
        "label": "Data Chat",
        "step": "3",
        "optional": True,
    },
    {
        "id": "04_Trenowanie_Modelu",
        "page_id": "04_Trenowanie_Modelu",
        "target": "pages/04_Trenowanie_Modelu.py",
        "label": "Trenowanie modelu",
        "step": "4",
        "optional": False,
    },
    {
        "id": "05_Predykcja",
        "page_id": "05_Predykcja",
        "target": "pages/05_Predykcja.py",
        "label": "Predykcja",
        "step": "5",
        "optional": False,
    },
]

EXTRA_PAGES: Dict[str, Dict[str, str]] = {
    "how": {
        "page_id": "00_Jak_to_dziala",
        "label": "Jak to dzia\u0142a?",
    },
    "settings": {
        "page_id": "06_Ustawienia",
        "label": "Ustawienia",
    },
}


def _find_step_index(step_id: Optional[str]) -> Optional[int]:
    if step_id is None:
        return None
    for i, step in enumerate(PIPELINE):
        if step.get("id") == step_id:
            return i
    return None


def _step_by_id(step_id: str) -> Optional[Dict[str, object]]:
    for step in PIPELINE:
        if step.get("id") == step_id:
            return step
    return None


def _step_badge(step_no: str) -> str:
    return KEYCAP_DIGITS.get(step_no, step_no)


def go(page_id: str) -> None:
    """Przelacza strone na podstawie identyfikatora etapu."""
    if hasattr(st, "switch_page"):
        target = f"pages/{page_id}.py"
        step = _step_by_id(page_id)
        if step and step.get("target"):
            target = str(step["target"])
        try:
            st.switch_page(target)  # type: ignore[attr-defined]
            return
        except Exception:
            pass

    st.warning(
        "Nie mog\u0119 automatycznie prze\u0142\u0105czy\u0107 strony. "
        "U\u017cyj potoku etap\u00f3w u g\u00f3ry, aby przej\u015b\u0107 dalej."
    )


def render_flow_nav(
    current_id: Optional[str] = None,
    key_prefix: str = "flow_main",
) -> None:
    """Rysuje potok etapow u gory lub na dole strony."""

    st.markdown(
        """
        <style>
        div[data-testid="column"] .stButton > button {
            width: 100% !important;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    layout = [6, 0.9, 6, 0.9, 6, 0.9, 6, 0.9, 6]
    cols = st.columns(layout)
    current_idx = _find_step_index(current_id)

    for i, step in enumerate(PIPELINE):
        col_btn = cols[i * 2]

        status_icon = ""
        if current_idx is not None:
            if i < current_idx:
                status_icon = "\u2705 "
            elif i == current_idx:
                status_icon = "\U0001F534 "

        label = f"{status_icon}{_step_badge(str(step['step']))} {step['label']}"

        if current_idx is not None and i <= current_idx:
            btn_type = "secondary"
        else:
            btn_type = "primary"

        help_text = None
        if step["id"] == "03_Data_Chat":
            help_text = "Opcjonalny krok mi\u0119dzy EDA a trenowaniem modelu."

        clicked = col_btn.button(
            label,
            key=f"{key_prefix}_{step['id']}",
            type=btn_type,
            help=help_text,
            width="stretch",
        )

        if clicked:
            go(str(step["page_id"]))

        if i < len(PIPELINE) - 1:
            arrow_col = cols[i * 2 + 1]
            arrow_col.markdown(
                """
                <div style="
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    min-height:2.6rem;
                    transform:translateY(-0.06rem);
                    font-size:1.55rem;
                    font-weight:700;
                    color:#7b8798;
                    line-height:1;
                ">&#10148;</div>
                """,
                unsafe_allow_html=True,
            )


def render_top_flow_nav(current_id: Optional[str] = None) -> None:
    """Historyczna nazwa kompatybilna z wczesniejszym API."""
    render_flow_nav(current_id=current_id, key_prefix="flow_top")


def render_sidebar_links(active_id: Optional[str] = None) -> None:
    """Historyczny helper nawigacji w sidebarze. Wylaczony produkcyjnie."""
    return


render_sidebar_nav = render_sidebar_links

__all__ = [
    "PIPELINE",
    "go",
    "render_flow_nav",
    "render_top_flow_nav",
    "render_sidebar_links",
    "render_sidebar_nav",
]
