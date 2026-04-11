# app/core/top_nav.py
"""
Cienka warstwa nad ui.flow_nav â€“ aliasy + ukrycie wbudowanej nawigacji.
"""

from typing import Optional

import streamlit as st

from ui.flow_nav import (  # type: ignore[import]
    PIPELINE,
    go as _go,
    render_flow_nav as _render_flow_nav,
    render_top_flow_nav as _render_top_flow_nav,
    render_sidebar_links as _render_sidebar_links,
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ALIASY DO FUNKCJI Z ui.flow_nav
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_flow_nav(
    current_id: Optional[str] = None,
    key_prefix: str = "flow_main",
) -> None:
    """GĹ‚Ăłwna funkcja paska etapĂłw â€“ deleguje do ui.flow_nav.render_flow_nav."""
    _render_flow_nav(current_id=current_id, key_prefix=key_prefix)


def render_top_flow_nav(current_id: Optional[str] = None) -> None:
    """Historyczna nazwa â€“ pasek u gĂłry, z prefiksem 'flow_top'."""
    _render_top_flow_nav(current_id=current_id)


def render_sidebar_links(active_id: Optional[str] = None) -> None:
    """Historyczny helper. Sidebarowe linki nawigacyjne są wyłączone produkcyjnie."""
    return


def go(page_id: str) -> None:
    """Alias na funkcjÄ™ przeĹ‚Ä…czania stron."""
    _go(page_id)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# UKRYCIE WBUdOWANEJ NAWIGACJI STREAMLIT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def hide_default_multipage_nav() -> None:
    """Ukrywa lewy â€žsidebar navâ€ť Streamlita (domyĹ›lne menu stron)."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Dla bardzo starej nazwy:
render_sidebar_nav = render_sidebar_links

__all__ = [
    "PIPELINE",
    "go",
    "render_flow_nav",
    "render_top_flow_nav",
    "render_sidebar_links",
    "render_sidebar_nav",
    "hide_default_multipage_nav",
]


