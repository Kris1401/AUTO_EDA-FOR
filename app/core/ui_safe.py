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
    Older versions: st.altair_chart(chart, use_container_width=True)

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
        try:
            return fn(chart, width="stretch", **kwargs)
        except TypeError as exc:
            if "width" not in str(exc).lower():
                raise

    # fallback for older Streamlit
    try:
        return fn(chart, use_container_width=True, **kwargs)
    except TypeError as exc:
        if "use_container_width" not in str(exc).lower():
            raise
    return fn(chart, **kwargs)


def dataframe_stretch(target: Any, data: Any, **kwargs: Any) -> Any:
    """Render dataframe to full container width across Streamlit versions.

    Newer versions: st.dataframe(data, width='stretch')
    Older versions: st.dataframe(data, use_container_width=True)
    """
    import streamlit as st

    if data is None and target is not None:
        data = target
        target = st

    fn = getattr(target, "dataframe", None) or st.dataframe

    kwargs.pop("width", None)
    kwargs.pop("use_container_width", None)

    if _supports_kw(fn, "width"):
        try:
            return fn(data, width="stretch", **kwargs)
        except TypeError as exc:
            if "width" not in str(exc).lower():
                raise

    try:
        return fn(data, use_container_width=True, **kwargs)
    except TypeError as exc:
        if "use_container_width" not in str(exc).lower():
            raise
    return fn(data, **kwargs)


def plotly_chart_stretch(target: Any, fig: Any, **kwargs: Any) -> Any:
    """Render Plotly chart to full container width across Streamlit versions."""
    import streamlit as st

    if fig is None and target is not None:
        fig = target
        target = st

    fn = getattr(target, "plotly_chart", None) or st.plotly_chart

    config = kwargs.pop("config", None)
    kwargs.pop("width", None)
    kwargs.pop("use_container_width", None)

    call_kwargs: dict[str, Any] = {}
    if config is not None:
        call_kwargs["config"] = config
    call_kwargs.update(kwargs)

    if _supports_kw(fn, "width"):
        try:
            return fn(fig, width="stretch", **call_kwargs)
        except TypeError as exc:
            if "width" not in str(exc).lower():
                raise

    try:
        return fn(fig, use_container_width=True, **call_kwargs)
    except TypeError as exc:
        if "use_container_width" not in str(exc).lower():
            raise
    return fn(fig, **call_kwargs)

