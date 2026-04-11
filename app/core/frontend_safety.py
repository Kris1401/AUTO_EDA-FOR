# -*- coding: utf-8 -*-
"""Enterprise-grade Stage3 hardening: SAFE FRONTEND MODE.

Solves:
- UI freezes when charts serialize too much data (e.g., 1M rows sent to frontend via Altair/Plotly).
- Some chart types (ECDF/KDE/violin/box with points) can explode payload size.

Policy:
- Automatic fallback when rows > threshold (default 100k): safe mode is *forced*.
- Sampling only affects chart payloads; KPIs/statistics should still be computed on full data upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

import json

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class SafeFrontendConfig:
    enabled: bool
    effective: bool
    reason: str

    hard_row_threshold: int = 100_000

    max_plot_rows: int = 5_000
    max_outlier_points: int = 2_000
    max_ecdf_points: int = 2_000

    warn_payload_kb: int = 2_500   # ~2.5MB
    hard_payload_kb: int = 10_000  # ~10MB


_SS_ENABLED_KEY = "dc_safe_frontend_enabled_v1"
_SS_MAX_PLOT_ROWS_KEY = "dc_safe_frontend_max_plot_rows_v1"
_SS_MAX_ECDF_POINTS_KEY = "dc_safe_frontend_max_ecdf_points_v1"
_SS_HARD_THRESHOLD_KEY = "dc_safe_frontend_hard_row_threshold_v1"


def init_safe_frontend_sidebar(df_rows: int) -> Dict[str, Any]:
    """Render SAFE FRONTEND MODE controls in sidebar; returns config dict.

    Streamlit constraint: session_state defaults must be set BEFORE widget instantiation.
    """
    df_rows = int(df_rows)

    if _SS_HARD_THRESHOLD_KEY not in st.session_state:
        st.session_state[_SS_HARD_THRESHOLD_KEY] = 100_000
    hard_thr = int(st.session_state[_SS_HARD_THRESHOLD_KEY])

    if _SS_ENABLED_KEY not in st.session_state:
        st.session_state[_SS_ENABLED_KEY] = bool(df_rows > hard_thr)
    if _SS_MAX_PLOT_ROWS_KEY not in st.session_state:
        st.session_state[_SS_MAX_PLOT_ROWS_KEY] = 5_000
    if _SS_MAX_ECDF_POINTS_KEY not in st.session_state:
        st.session_state[_SS_MAX_ECDF_POINTS_KEY] = 2_000

    st.markdown("---")
    st.markdown("### 🛡️ SAFE FRONTEND MODE")
    st.caption(
        "Ochrona UI dla dużych danych. "
        "Gdy wiersze > próg, tryb jest wymuszany automatycznie (fallback), aby uniknąć zawieszania."
    )

    st.checkbox(
        "Włącz SAFE FRONTEND MODE (auto dla > progu)",
        key=_SS_ENABLED_KEY,
        help="Wymusza sampling danych wysyłanych do wykresów. Statystyki/KPI licz na pełnych danych.",
    )

    st.slider(
        "Maks. liczba wierszy na wykres (sampling)",
        min_value=500,
        max_value=30_000,
        step=500,
        key=_SS_MAX_PLOT_ROWS_KEY,
    )

    st.slider(
        "Maks. liczba punktów ECDF (kwantyle)",
        min_value=200,
        max_value=10_000,
        step=200,
        key=_SS_MAX_ECDF_POINTS_KEY,
    )

    enabled = bool(st.session_state[_SS_ENABLED_KEY])
    effective = bool(enabled or (df_rows > hard_thr))
    reason = "enabled_by_user" if enabled else ("forced_rows_gt_threshold" if df_rows > hard_thr else "disabled")

    cfg = SafeFrontendConfig(
        enabled=enabled,
        effective=effective,
        reason=reason,
        hard_row_threshold=hard_thr,
        max_plot_rows=int(st.session_state[_SS_MAX_PLOT_ROWS_KEY]),
        max_ecdf_points=int(st.session_state[_SS_MAX_ECDF_POINTS_KEY]),
    )
    return asdict(cfg)


def safe_mode_effective(filters_or_ctx: Dict[str, Any], df_rows: int) -> Tuple[bool, Dict[str, Any]]:
    """Resolve SAFE FRONTEND MODE from filters/context."""
    df_rows = int(df_rows)
    cfg = (filters_or_ctx or {}).get("safe_frontend_cfg") or {}

    enabled = bool((filters_or_ctx or {}).get("safe_frontend", False)) or bool(cfg.get("enabled", False))
    hard_thr = int(cfg.get("hard_row_threshold", 100_000))

    effective = bool(enabled or (df_rows > hard_thr))
    cfg_out = dict(cfg)
    cfg_out.setdefault("hard_row_threshold", hard_thr)
    cfg_out.setdefault("max_plot_rows", 5_000)
    cfg_out.setdefault("max_outlier_points", 2_000)
    cfg_out.setdefault("max_ecdf_points", 2_000)
    cfg_out.setdefault("warn_payload_kb", 2_500)
    cfg_out.setdefault("hard_payload_kb", 10_000)
    cfg_out["enabled"] = bool(cfg.get("enabled", enabled))
    cfg_out["effective"] = effective
    cfg_out.setdefault("reason", "forced_rows_gt_threshold" if df_rows > hard_thr else ("enabled" if enabled else "disabled"))
    return effective, cfg_out


def sample_df_for_frontend(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    """Sample rows for chart payloads (best-effort)."""
    try:
        n = int(max_rows)
        if n <= 0 or df is None:
            return df
        if len(df) <= n:
            return df
        return df.sample(n=n, random_state=random_state)
    except Exception:
        return df


def quantile_ecdf(series: pd.Series, max_points: int = 2000) -> pd.DataFrame:
    """ECDF aggregated by quantiles to keep payload small."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame({"x": [], "ecdf": []})
    n = int(max_points)
    n = max(50, min(n, 10_000))
    qs = np.linspace(0.0, 1.0, num=n, endpoint=True)
    x = np.quantile(s.to_numpy(), qs)
    return pd.DataFrame({"x": x, "ecdf": qs})


def approx_payload_kb(obj: Any) -> float:
    """Approximate JSON payload size in KB (best-effort)."""
    try:
        if obj is None:
            return 0.0
        if hasattr(obj, "to_json"):
            js = obj.to_json()
        else:
            js = json.dumps(obj, default=str)
        return float(len(js.encode("utf-8")) / 1024.0)
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Global sampling decorator (optional, per-branch/per-block)
# ──────────────────────────────────────────────────────────────────────────────

def safe_frontend_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return (df_for_frontend, cfg) based on current Streamlit session_state.

    This is useful when you *do* have access to st.session_state.
    """
    try:
        enabled = bool(st.session_state.get("dc_safe_frontend_enabled_v1", False))
        hard_thr = int(st.session_state.get("dc_safe_frontend_hard_row_threshold_v1", 100_000))
        effective = bool(enabled or (len(df) > hard_thr))
        cfg = {
            "enabled": enabled,
            "effective": effective,
            "reason": "enabled_by_user" if enabled else ("forced_rows_gt_threshold" if len(df) > hard_thr else "disabled"),
            "hard_row_threshold": hard_thr,
            "max_plot_rows": int(st.session_state.get("dc_safe_frontend_max_plot_rows_v1", 5000)),
        }
        if not effective:
            return df, cfg
        return sample_df_for_frontend(df, max_rows=int(cfg["max_plot_rows"])), cfg
    except Exception:
        return df, {"enabled": False, "effective": False, "reason": "resolve_failed"}


def safe_frontend_sampler(branch: str, block: str):
    """Decorator to enforce sampling for charts (per-branch / per-block).

    The wrapped function must accept a DataFrame as first positional argument.
    """
    def _decorator(fn):
        def _wrapped(df: pd.DataFrame, *args, **kwargs):
            df2, cfg = safe_frontend_df(df)
            # Provide cfg as kwarg for downstream logging (optional)
            kwargs.setdefault("safe_frontend_cfg", cfg)
            return fn(df2, *args, **kwargs)
        _wrapped.__name__ = getattr(fn, "__name__", "wrapped")
        _wrapped.__doc__ = getattr(fn, "__doc__", None)
        return _wrapped
    return _decorator
