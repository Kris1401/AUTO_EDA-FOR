# -*- coding: utf-8 -*-
"""Lightweight in-app profiler for Stage3 (Data Chat).

Cel:
- Zdiagnozować "zawieszenia" UI przez pomiar czasu per gałąź i per blok.
- Brak zewnętrznych zależności; bezpieczne do trzymania w repo.

Użycie w kodzie gałęzi:

    from core.perf_debug import span
    with span("Stage3/render/distribution", rows=len(df)):
        ...

UI jest renderowane w sidebarze przez 03_Data_Chat.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Iterator, List

import streamlit as st


STATE_KEY = "dc_profiler_stage3_v1"


@dataclass
class _Evt:
    name: str
    dur_s: float
    meta: Dict[str, Any]


def _get_state() -> Dict[str, Any]:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = {
            "enabled": False,
            "events": [],  # List[_Evt]
            "total_s": 0.0,
        }
    return st.session_state[STATE_KEY]


def set_enabled(enabled: bool) -> None:
    _get_state()["enabled"] = bool(enabled)


def reset() -> None:
    s = _get_state()
    s["events"] = []
    s["total_s"] = 0.0


@contextmanager
def span(name: str, **meta: Any) -> Iterator[None]:
    """Zapisz pomiar czasu do session_state, jeśli profiler jest włączony."""
    s = _get_state()
    if not s.get("enabled", False):
        yield
        return

    t0 = perf_counter()
    try:
        yield
    finally:
        dur = perf_counter() - t0
        s["total_s"] = float(s.get("total_s", 0.0)) + float(dur)
        s["events"].append(_Evt(name=str(name), dur_s=float(dur), meta=dict(meta)))


def render_sidebar(title: str = "Stage3") -> None:
    """Sidebar UI. Wywołuj raz na run (bezpieczne)."""
    s = _get_state()

    # Hidden debug flag: profiler can still be enabled programmatically via session_state.
    enabled = bool(st.session_state.get("dc_profiler_stage3_enabled_v1", s.get("enabled", False)))
    set_enabled(enabled)

    if not enabled:
        return

    with st.sidebar.expander("🐢 Profiler — wyniki", expanded=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Wyczyść", key="dc_profiler_stage3_reset_v1"):
                reset()
                st.rerun()
        with c2:
            st.caption("Tip: jeśli UI wisi, patrz na najdłuższy wpis.")

        events: List[_Evt] = list(s.get("events", []))
        if not events:
            st.info("Brak danych. Włącz profiler i wykonaj akcję (np. 'Przeanalizuj pytanie').")
            return

        # sortuj malejąco (pokaż ostatnie ~80, żeby nie zapchać sidebara)
        events_sorted = sorted(events, key=lambda e: e.dur_s, reverse=True)[:80]
        st.write(
            f"Łącznie zmierzono: **{float(s.get('total_s', 0.0)):.2f}s** | zdarzenia: **{len(events)}**"
        )

        for ev in events_sorted:
            meta_txt = ""
            if ev.meta:
                parts = []
                for k, v in list(ev.meta.items())[:6]:
                    parts.append(f"{k}={v}")
                meta_txt = " | " + ", ".join(parts)
            st.write(f"• **{ev.name}** — {ev.dur_s:.2f}s{meta_txt}")
