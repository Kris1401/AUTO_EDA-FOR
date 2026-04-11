"""Small UI compatibility helpers.

Streamlit APIs are evolving (e.g. `use_container_width` -> `width`).
Some Streamlit versions support the new arguments for some elements but not for others.

This module provides tiny wrappers so the app keeps working across versions.
"""

from __future__ import annotations

from typing import Any
import inspect


def _supports_kw(fn: Any, name: str) -> bool:
    """Return True if callable `fn` accepts keyword `name`."""
    try:
        sig = inspect.signature(fn)
        return name in sig.parameters
    except Exception:
        # be conservative
        return False


def altair_chart_stretch(target: Any, chart: Any, **kwargs: Any) -> Any:
    """Render Altair chart to full container width across Streamlit versions.

    Newer versions: st.altair_chart(chart, width='stretch')
    Older versions: st.altair_chart(chart, width="stretch")

    `target` can be `st` or a container like `st.container()`.
    """
    import streamlit as st

    # allow calling with target omitted: altair_chart_stretch(chart)
    if chart is None and target is not None:
        chart = target
        target = st

    fn = getattr(target, "altair_chart", None) or st.altair_chart

    # Avoid passing conflicting params from callers
    kwargs.pop("width", None)
    kwargs.pop("use_container_width", None)

    if _supports_kw(fn, "width"):
        return fn(chart, width="stretch", **kwargs)

    # fallback for older Streamlit
    return fn(chart, width="stretch", **kwargs)


def dataframe_stretch(target: Any, data: Any, **kwargs: Any) -> Any:
    """Render dataframe to full container width across Streamlit versions.

    Newer versions: st.dataframe(data, width='stretch')
    Older versions: st.dataframe(data, width="stretch")
    """
    import streamlit as st

    if data is None and target is not None:
        data = target
        target = st

    fn = getattr(target, "dataframe", None) or st.dataframe

    kwargs.pop("width", None)
    kwargs.pop("use_container_width", None)

    if _supports_kw(fn, "width"):
        return fn(data, width="stretch", **kwargs)

    return fn(data, width="stretch", **kwargs)

