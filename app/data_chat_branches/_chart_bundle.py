from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd


# =============================================================================
# Chart Bundle — stable helpers for Data Chat
# =============================================================================
# IMPORTANT:
# - This module MUST be import-safe (no NameError / syntax errors).
# - It provides a basic bundle of charts from a chart_spec.
# - Branches (e.g. Distribution) may override charts later — that's OK.


# -----------------------------------------------------------------------------
# Column inference
# -----------------------------------------------------------------------------

def _infer_numeric_cols(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    cols: List[str] = []
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        except Exception:
            continue
    return cols


def _infer_datetime_cols(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    cols: List[str] = []
    for c in df.columns:
        try:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                cols.append(c)
                continue
            # Light inference (do not coerce entire col to datetime if huge)
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                sample = s.dropna().astype(str).head(50)
                if len(sample) == 0:
                    continue
                ok = 0
                for v in sample:
                    try:
                        pd.to_datetime(v)
                        ok += 1
                    except Exception:
                        pass
                if ok / max(1, len(sample)) >= 0.8:
                    cols.append(c)
        except Exception:
            continue
    return cols


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------

def _apply_filters_to_df(df: pd.DataFrame, filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Apply simple filters (IN / range) from dict.

    Supported:
    - {col: [v1, v2]} for categorical IN
    - {col: (min, max)} for numeric range
    - {col: {'min':..., 'max':...}} for numeric range
    """
    if df is None or df.empty or not filters:
        return df

    out = df
    for col, rule in filters.items():
        if col not in out.columns:
            continue
        try:
            if isinstance(rule, (list, tuple)) and len(rule) == 2 and not isinstance(rule[0], str) and not isinstance(rule[1], str):
                # numeric range (min, max)
                lo, hi = rule
                out = out[(out[col] >= lo) & (out[col] <= hi)]
            elif isinstance(rule, dict) and ("min" in rule or "max" in rule):
                lo = rule.get("min", -np.inf)
                hi = rule.get("max", np.inf)
                out = out[(out[col] >= lo) & (out[col] <= hi)]
            else:
                # categorical list
                vals = rule if isinstance(rule, list) else [rule]
                out = out[out[col].isin(vals)]
        except Exception:
            continue
    return out


# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------

def _scale_for_categories(categories: Optional[List[Any]] = None) -> alt.Scale:
    """Return stable nominal scale. We avoid hard-coded colors; use Vega scheme."""
    # categories arg is kept for future mapping; currently use scheme.
    return alt.Scale(scheme="category20")


# -----------------------------------------------------------------------------
# Primitive charts
# -----------------------------------------------------------------------------

def _chart_hist(df: pd.DataFrame, x: str, bins: int = 30, height: int = 320) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(x=alt.X(x, bin=alt.Bin(maxbins=bins), title=x), y=alt.Y("count()", title="Liczba"))
        .properties(height=height)
    )


def _chart_hist_log(df: pd.DataFrame, x: str, bins: int = 30, height: int = 320) -> alt.Chart:
    # Add small epsilon to avoid log(0)
    tmp = df.copy()
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp = tmp.dropna(subset=[x])
    tmp["_logx"] = np.log10(tmp[x].clip(lower=1e-12))
    return (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x=alt.X("_logx:Q", bin=alt.Bin(maxbins=bins), title=f"log10({x})"),
            y=alt.Y("count()", title="Liczba"),
            tooltip=[alt.Tooltip("count()", title="count")],
        )
        .properties(height=height)
    )


def _chart_box(df: pd.DataFrame, x: str, height: int = 160) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_boxplot()
        .encode(x=alt.X(f"{x}:Q", title=x))
        .properties(height=height)
    )


def _chart_ecdf(df: pd.DataFrame, x: str, height: int = 220) -> alt.Chart:
    tmp = df[[x]].copy()
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp = tmp.dropna()
    tmp = tmp.sort_values(x)
    tmp["ecdf"] = np.arange(1, len(tmp) + 1) / max(1, len(tmp))
    return (
        alt.Chart(tmp)
        .mark_line()
        .encode(x=alt.X(f"{x}:Q", title=x), y=alt.Y("ecdf:Q", title="ECDF"))
        .properties(height=height)
    )


def _chart_line(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, height: int = 360) -> alt.Chart:
    enc: Dict[str, Any] = {
        "x": alt.X(x, title=x),
        "y": alt.Y(y, title=y),
    }
    if color and color in df.columns:
        enc["color"] = alt.Color(color, scale=_scale_for_categories(df[color].unique().tolist()))
    return alt.Chart(df).mark_line().encode(**enc).properties(height=height)


def _chart_bar(df: pd.DataFrame, x: str, y: str, height: int = 360) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(x=alt.X(x, sort="-y", title=x), y=alt.Y(y, title=y))
        .properties(height=height)
    )


def _chart_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, height: int = 360) -> alt.Chart:
    enc: Dict[str, Any] = {
        "x": alt.X(x, title=x),
        "y": alt.Y(y, title=y),
    }
    if color and color in df.columns:
        enc["color"] = alt.Color(color, scale=_scale_for_categories(df[color].unique().tolist()))
    return alt.Chart(df).mark_circle(size=40, opacity=0.75).encode(**enc).properties(height=height)


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------

def _build_distribution_charts(df: pd.DataFrame, spec: Dict[str, Any]) -> Tuple[alt.Chart, List[alt.Chart], Dict[str, Any]]:
    x = spec.get("x") or spec.get("col")
    if not x:
        num = _infer_numeric_cols(df)
        x = num[0] if num else (df.columns[0] if len(df.columns) else None)
    if not x or x not in df.columns:
        # empty safe chart
        empty = alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_text(text="Brak danych").encode(x="x:Q", y="y:Q")
        return empty, [], {"reason": "no_col"}

    bins = int(spec.get("bins") or 30)
    primary = _chart_hist(df, x=x, bins=bins, height=360)
    alt_charts = [_chart_box(df, x=x, height=160), _chart_ecdf(df, x=x, height=220)]
    meta: Dict[str, Any] = {"x": x, "bins": bins}
    return primary, alt_charts, meta


def _build_composition_charts(df: pd.DataFrame, spec: Dict[str, Any]) -> Tuple[alt.Chart, List[alt.Chart], Dict[str, Any]]:
    group = spec.get("group") or spec.get("group_col")
    value = spec.get("value") or spec.get("value_col")

    if not group or group not in df.columns:
        group = df.columns[0] if len(df.columns) else None
    if not value or value not in df.columns:
        nums = _infer_numeric_cols(df)
        value = nums[0] if nums else None

    if not group or not value or group not in df.columns or value not in df.columns:
        empty = alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_text(text="Brak danych").encode(x="x:Q", y="y:Q")
        return empty, [], {"reason": "no_cols"}

    agg = df[[group, value]].copy()
    agg[value] = pd.to_numeric(agg[value], errors="coerce")
    agg = agg.dropna(subset=[group, value])
    agg = agg.groupby(group, as_index=False)[value].sum().sort_values(value, ascending=False)

    primary = _chart_bar(agg, x=group, y=value, height=420)
    meta: Dict[str, Any] = {"group": group, "value": value}
    return primary, [], meta


def _build_ts_composition_chart(df: pd.DataFrame, spec: Dict[str, Any]) -> Tuple[alt.Chart, List[alt.Chart], Dict[str, Any]]:
    date_col = spec.get("date") or spec.get("date_col")
    group = spec.get("group") or spec.get("group_col")
    value = spec.get("value") or spec.get("value_col")

    if not date_col or date_col not in df.columns:
        dcols = _infer_datetime_cols(df)
        date_col = dcols[0] if dcols else None
    if not group or group not in df.columns:
        group = None
    if not value or value not in df.columns:
        nums = _infer_numeric_cols(df)
        value = nums[0] if nums else None

    if not date_col or not value or date_col not in df.columns or value not in df.columns:
        empty = alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_text(text="Brak danych").encode(x="x:Q", y="y:Q")
        return empty, [], {"reason": "no_cols"}

    tmp = df[[date_col] + ([group] if group else []) + [value]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[value] = pd.to_numeric(tmp[value], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, value])

    # month granularity (matches earlier requirements)
    tmp["_month"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
    gb_cols = ["_month"] + ([group] if group else [])
    agg = tmp.groupby(gb_cols, as_index=False)[value].sum()

    primary = _chart_line(agg, x="_month:T", y=f"{value}:Q", color=group, height=420)
    meta: Dict[str, Any] = {"date_col": date_col, "group": group, "value": value}
    return primary, [], meta


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _build_chart_from_spec(df: pd.DataFrame, spec: Dict[str, Any]) -> Tuple[alt.Chart, List[alt.Chart], Dict[str, Any]]:
    chart_type = (spec.get("chart") or spec.get("type") or "").lower().strip()

    # High-level "intent" fallback
    intent = (spec.get("intent") or "").lower().strip()

    if chart_type in {"hist", "histogram"}:
        x = spec.get("x") or spec.get("col")
        bins = int(spec.get("bins") or 30)
        if x and x in df.columns:
            return _chart_hist(df, x=x, bins=bins, height=360), [], {"x": x, "bins": bins}

    if chart_type in {"hist_log", "histogram_log"}:
        x = spec.get("x") or spec.get("col")
        bins = int(spec.get("bins") or 30)
        if x and x in df.columns:
            return _chart_hist_log(df, x=x, bins=bins, height=360), [], {"x": x, "bins": bins}

    if chart_type in {"box", "boxplot"}:
        x = spec.get("x") or spec.get("col")
        if x and x in df.columns:
            return _chart_box(df, x=x), [], {"x": x}

    if chart_type in {"ecdf"}:
        x = spec.get("x") or spec.get("col")
        if x and x in df.columns:
            return _chart_ecdf(df, x=x), [], {"x": x}

    if chart_type in {"line"}:
        x = spec.get("x")
        y = spec.get("y")
        color = spec.get("color")
        if x and y and x in df.columns and y in df.columns:
            return _chart_line(df, x=x, y=y, color=color), [], {"x": x, "y": y, "color": color}

    if chart_type in {"bar"}:
        x = spec.get("x")
        y = spec.get("y")
        if x and y and x in df.columns and y in df.columns:
            return _chart_bar(df, x=x, y=y), [], {"x": x, "y": y}

    if chart_type in {"scatter"}:
        x = spec.get("x")
        y = spec.get("y")
        color = spec.get("color")
        if x and y and x in df.columns and y in df.columns:
            return _chart_scatter(df, x=x, y=y, color=color), [], {"x": x, "y": y, "color": color}

    # Intent fallbacks
    if "distribution" in intent:
        return _build_distribution_charts(df, spec)

    if "composition_over_time" in intent or "time" in intent:
        return _build_ts_composition_chart(df, spec)

    if "composition" in intent:
        return _build_composition_charts(df, spec)

    # Last resort: distribution-ish
    return _build_distribution_charts(df, spec)

def _build_chart_bundle_from_spec(df: pd.DataFrame, chart_spec: Dict[str, Any]):
    """Builds a UI-ready chart bundle from a compact chart_spec.

    Contract (v1):
      - always returns: (bundle, meta)
      - bundle = {"primary": alt.Chart|None, "secondary": List[alt.Chart]}
      - meta = dict with diagnostics (never raises for bad specs)

    NOTE: Older iterations returned 3 values or sometimes a raw dict which broke callers.
    This function is the single stable contract used by branches.
    """
    meta: Dict[str, Any] = {}

    # Defensive defaults
    if not isinstance(chart_spec, dict) or not chart_spec:
        empty = alt.Chart(pd.DataFrame()).mark_point()
        return {"primary": empty, "secondary": []}, {"reason": "empty_chart_spec"}

    try:
        primary, secondaries, meta = _build_chart_from_spec(df, chart_spec)
        bundle = {"primary": primary, "secondary": (secondaries or [])}
        return bundle, (meta or {})
    except Exception as e:
        empty = alt.Chart(pd.DataFrame()).mark_point()
        return {"primary": empty, "secondary": []}, {"error": f"{type(e).__name__}: {e}"}
