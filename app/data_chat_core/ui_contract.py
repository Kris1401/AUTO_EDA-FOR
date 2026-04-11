"""UI Contract v1.0 — single source of truth for block rendering (Data Chat).

This module centralizes rendering of:
- Executive takeaway (LLM-generated)
- Guidance trio: Sens / Interpretacja / Najlepsza praktyka
- Spacing + visual hierarchy

Why here?
- Shared across branches (Distribution / Composition Static / ...).
- Keeps a single, consistent Gestalt + McKinsey/Bain style.

Usage (inside a branch):
    from data_chat_core.ui_contract import render_exec_takeaway, render_guidance

    render_exec_takeaway(text)
    render_guidance(sens=..., interp=..., best=...)
"""

from __future__ import annotations

from typing import Optional
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Visual tokens (keep minimal, stable)
# ─────────────────────────────────────────────────────────────────────────────

EXEC_ICON = "🚀"
SENS_ICON = "✅"
INTERP_ICON = "🧠"
BEST_ICON = "💎"

_EXEC_BORDER = "#ff4b4b"  # Streamlit red accent used in your UI
_TEXT = "#111"
_MUTED = "#6b7280"

# ─────────────────────────────────────────────────────────────────────────────
# Guidance v1.0 — global CSS (theme-independent) + 1x injector
# ─────────────────────────────────────────────────────────────────────────────

_GUIDANCE_CSS_V1 = """
<style>
/* Guidance v1.0: stable typography across Streamlit themes */
.dc-guidance-v1{
  font-size:14px !important;
  line-height:22px !important;
  color:#111 !important;
}

.dc-guidance-v1 .dc-gline{
  display:flex;
  align-items:flex-start;
  gap:8px;
  margin:0 0 6px 0;
  padding:0;
  line-height:22px !important;
}

.dc-guidance-v1 .dc-gline:last-child{
  margin-bottom:0;
}

.dc-guidance-v1 .dc-gicon{
  width:20px;
  flex:0 0 20px;
  text-align:center;
  line-height:22px !important;
  transform: translateY(0.5px);
}

.dc-guidance-v1 .dc-gtext{
  flex:1 1 auto;
  margin:0;
  padding:0;
  line-height:22px !important;
  white-space:normal !important;
  overflow-wrap:anywhere !important;
  word-break:break-word !important;
}

.dc-guidance-v1 .dc-glabel{font-weight:700 !important}
</style>
"""

def _ensure_guidance_css_v1() -> None:
    """
    Guidance CSS MUST be injected on every Streamlit rerun.
    Streamlit reruns rebuild the DOM, so a "inject once per session_state"
    guard causes styles (incl. bold labels) to disappear after interactions.
    """
    try:
        st.markdown(_GUIDANCE_CSS_V1, unsafe_allow_html=True)
    except Exception:
        pass

def _safe(text: Optional[str]) -> str:
    return (text or "").strip()

def render_exec_takeaway(text: Optional[str]) -> None:
    """Render a single Executive takeaway in a compact callout.
    Expectation: 2 sentences (fact+numbers, then decision).
    """
    txt = _safe(text)
    if not txt:
        # keep consistent placeholder (but subtle)
        st.markdown(f"{EXEC_ICON} **Executive takeaway:** —")
        return

    st.markdown(
        f"""<div style="
            border-left:4px solid {_EXEC_BORDER};
            padding:10px 12px;
            margin:8px 0 10px 0;
            background:#fff;
            border-radius:8px;
        ">
            <div style="font-size:13px;color:{_TEXT};"><b>Executive takeaway</b></div>
            <div style="font-size:14px;color:{_TEXT};margin-top:4px;
                        white-space:normal !important;
                        overflow:visible !important;
                        text-overflow:clip !important;
                        word-break:break-word !important;
                        overflow-wrap:anywhere !important;">
                {txt}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_guidance(
    *args,
    sens: Optional[str] = None,
    interp: Optional[str] = None,
    best: Optional[str] = None,
    title: str = "Guidance",
    expanded: bool = False,
) -> None:
    """Render the 3-line guidance block in a consistent UI contract.

    Backward compatible:
      - render_guidance(sens, interp, best)
      - render_guidance(sens=..., interp=..., best=...)
    """
    # Backward compatibility with positional args
    if args:
        # Accept up to 3 positional args: (sens, interp, best)
        if len(args) >= 1 and sens is None:
            sens = args[0]
        if len(args) >= 2 and interp is None:
            interp = args[1]
        if len(args) >= 3 and best is None:
            best = args[2]

    sens = (sens or "").strip()
    interp = (interp or "").strip()
    best = (best or "").strip()

    if not (sens or interp or best):
        return

    _ensure_guidance_css_v1()

    # Render deterministic HTML (no markdown variability across themes)
    rows_html = []
    if sens:
        rows_html.append(
            f'<div class="dc-gline">'
            f'  <span class="dc-gicon">✅</span>'
            f'  <span class="dc-gtext"><span class="dc-glabel">Sens:</span> {_safe(sens)}</span>'
            f'</div>'
        )
    if interp:
        rows_html.append(
            f'<div class="dc-gline">'
            f'  <span class="dc-gicon">🔎</span>'
            f'  <span class="dc-gtext"><span class="dc-glabel">Interpretacja:</span> {_safe(interp)}</span>'
            f'</div>'
        )
    if best:
        rows_html.append(
            f'<div class="dc-gline">'
            f'  <span class="dc-gicon">💡</span>'
            f'  <span class="dc-gtext"><span class="dc-glabel">Najlepsza praktyka:</span> {_safe(best)}</span>'
            f'</div>'
        )

    html = '<div class="dc-guidance-v1">' + "".join(rows_html) + "</div>"

    with st.expander(title, expanded=expanded):
        st.markdown(html, unsafe_allow_html=True)

