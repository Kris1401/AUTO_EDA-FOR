from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import math
import numpy as np
import streamlit as st
from core.ui_safe import altair_chart_stretch
from core.safe_frontend import get_cfg_from_ctx, safe_sample_df
from data_chat_core.ui_contract import render_exec_takeaway, render_guidance
import altair as alt

# -----------------------------------------------------------------------------
# Local Chart Bundle helpers (NO _chart_bundle imports)
# -----------------------------------------------------------------------------
# NOTE: This is an intentionally small, import-safe subset of the old
# app/data_chat_branches/_chart_bundle.py helpers. Distribution overrides
# most visuals anyway, but we keep the contract stable for callers.

def _infer_numeric_cols(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    out: List[str] = []
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                out.append(c)
        except Exception:
            continue
    return out


def _build_chart_bundle_from_spec(df: pd.DataFrame, chart_spec: Dict[str, Any]):
    """Build a minimal chart bundle from chart_spec.

    Contract:
      - returns (bundle, meta) and never raises for bad specs
      - bundle keys used by branches may include: primary, alt/secondary, scenario, ts, clusters
    """
    if not isinstance(chart_spec, dict) or not chart_spec:
        empty = alt.Chart(pd.DataFrame()).mark_point()
        return {"primary": empty, "secondary": [], "alt": []}, {"reason": "empty_chart_spec"}

    try:
        intent = str(chart_spec.get("intent") or "").lower()
        primary_cfg = chart_spec.get("primary_chart") or {}
        if intent == "distribution":
            x = primary_cfg.get("x") or chart_spec.get("x") or chart_spec.get("col")
            if not x:
                nums = _infer_numeric_cols(df)
                x = nums[0] if nums else (df.columns[0] if len(df.columns) else None)
            if not x or x not in df.columns:
                empty = alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_text(text="Brak danych").encode(x="x:Q", y="y:Q")
                return {"primary": empty, "secondary": [], "alt": []}, {"reason": "no_col"}
            bins = int(primary_cfg.get("bins") or chart_spec.get("bins") or 30)
            primary = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(f"{x}:Q", bin=alt.Bin(maxbins=bins), title=x),
                    y=alt.Y("count():Q", title="Liczba"),
                )
                .properties(height=CHART_BLOCK_HEIGHT)
            )
            return {"primary": primary, "secondary": [], "alt": []}, {"kind": "distribution", "x": x, "bins": bins}

        # Fallback generic chart
        empty = alt.Chart(pd.DataFrame()).mark_point()
        return {"primary": empty, "secondary": [], "alt": []}, {"kind": intent or "generic"}
    except Exception as e:
        empty = alt.Chart(pd.DataFrame()).mark_point()
        return {"primary": empty, "secondary": [], "alt": []}, {"error": f"{type(e).__name__}: {e}"}


from data_chat_core.exec_takeaway import get_exec_takeaway

def _fmt_num(x: float) -> str:
    """Format number for KPI labels (PL style thousands separator)."""
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "—"
        # ints as no decimals, floats up to 2 decimals (trim)
        if abs(float(x) - int(float(x))) < 1e-9:
            return f"{int(float(x)):,}".replace(",", " ")
        s = f"{float(x):,.2f}".replace(",", " ")
        # strip trailing zeros
        s = s.rstrip("0").rstrip(".")
        return s
    except Exception:
        return "—"


CHART_BLOCK_HEIGHT = 360  # zachowane jak w oryginale


def _dist_checkpoints_enabled() -> bool:
    return bool(
        st.session_state.get("dist_internal_checkpoints_enabled", True)
        or st.session_state.get("dist_debug_checkpoints", False)
    )


def _compact_debug_value(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= 280 else f"{value[:277]}..."
    if isinstance(value, list):
        head = value[:5]
        if all(isinstance(x, (str, int, float, bool)) or x is None for x in head):
            return {"type": "list", "len": len(value), "head": head}
        return {"type": "list", "len": len(value)}
    if isinstance(value, dict):
        items = list(value.items())
        compact: Dict[str, Any] = {"type": "dict", "len": len(items)}
        for key, item_value in items[:6]:
            compact[str(key)] = _compact_debug_value(item_value)
        return compact
    return str(value)


def record_debug_checkpoint(where: str, **payload: Any) -> None:
    if not _dist_checkpoints_enabled():
        return
    entry: Dict[str, Any] = {"where": where}
    for key, value in payload.items():
        entry[str(key)] = _compact_debug_value(value)
    log = st.session_state.setdefault("dist_debug_log", [])
    log.append(entry)
    if len(log) > 250:
        del log[:-250]


def get_debug_checkpoints() -> list[dict[str, Any]]:
    log = st.session_state.get("dist_debug_log")
    return list(log) if isinstance(log, list) else []


def clear_debug_checkpoints() -> None:
    st.session_state["dist_debug_log"] = []

def _clean_label(s: str) -> str:
    """Bezpieczna etykieta do caption/legendy."""
    s = "" if s is None else str(s)
    s = s.replace("_", " ").strip()
    s = " ".join(s.split())
    return s


# -----------------------------------------------------------------------------
# Histogram helpers (Distribution overview)
# -----------------------------------------------------------------------------

# Controlled padding for histogram chart (single source of truth).
HIST_VIEW_PADDING = {"left": 24, "right": 12, "top": 2, "bottom": 18}
INSIGHT_CHART_PADDING = {"left": 16, "right": 8, "top": 4, "bottom": 14}


def _is_categorical_color_col(df: pd.DataFrame, col: str | None) -> bool:
    """Return True if col should be treated as categorical for color segmentation."""
    if not col or not isinstance(col, str) or df is None or df.empty or col not in df.columns:
        return False
    s = df[col]
    try:
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_categorical_dtype(s):
            return True
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            return True
        # numeric: allow only low cardinality to avoid 1000+ legend entries
        if pd.api.types.is_numeric_dtype(s):
            nunique = int(s.nunique(dropna=True))
            return nunique <= 20
        return False
    except Exception:
        return False


def _topk_domain_and_df(
    df: pd.DataFrame,
    color_col: str,
    k: int,
    other_label: str,
) -> tuple[pd.DataFrame, str, list[str] | None]:
    """Bucket color_col into Top-K + Other and return (df2, col2, domain_list).

    Top-K is based on COUNT of records after filters (df is already filtered/sampled for chart).
    Domain order is exactly: Top1..TopK (+ Other if created).
    If unique categories <= K -> no Other, but domain is still frequency-ordered.
    """
    if df is None or df.empty or (not isinstance(color_col, str)) or (color_col not in df.columns):
        return df, color_col, None
    try:
        k = int(k or 0)
    except Exception:
        k = 0
    if k <= 0:
        # still ensure stable string dtype for legend/tooltips
        out = df.copy()
        try:
            out[color_col] = out[color_col].astype(str)
        except Exception:
            pass
        return out, color_col, None

    s = df[color_col].astype(str).fillna(str(other_label))
    vc = s.value_counts(dropna=False)
    top = vc.head(k).index.astype(str).tolist()

    new_col = f"{color_col}__topk"
    out = df.copy()
    out[new_col] = s

    if len(vc) > k:
        top_set = set(top)
        out[new_col] = s.where(s.isin(top_set), str(other_label))
        # Ensure "Other" (bucket) is last in domain even if it exists as a real category.
        _other = str(other_label)
        _top_no_other = [c for c in top if c != _other]
        domain = _top_no_other + ([_other] if (_other in top_set or (len(vc) > len(_top_no_other))) else [])
    else:
        # No bucketing, but keep frequency order; if "Other" exists as a real category,
        # push it to the end for UI consistency.
        _other = str(other_label)
        _top_no_other = [c for c in top if c != _other]
        domain = _top_no_other + ([_other] if (_other in top) else [])

    return out, new_col, domain


def _safe_color_scale(domain_list: list[str] | None) -> alt.Scale | None:
    """Return a valid Altair Scale or None (never a Scale with domain=None)."""
    if not domain_list:
        return None
    dom = [str(x) for x in domain_list if x is not None]
    if len(dom) == 0:
        return None
    return alt.Scale(domain=dom, scheme="tableau20")

def _compute_iqr_fences(series: pd.Series):
    """
    Zwraca:
    s (Series, numeric, bez NaN),
    q1, q3, iqr,
    lower_fence, upper_fence (progi IQR: 1.5 * IQR)
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        nan = float("nan")
        return s, nan, nan, nan, nan, nan

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = float(q3 - q1)

    # klasyczne progi outlierów
    lower_fence = float(q1 - 1.5 * iqr)
    upper_fence = float(q3 + 1.5 * iqr)

    return s, q1, q3, iqr, lower_fence, upper_fence


def _distribution_view_bounds(series: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    min_v = float(s.min())
    max_v = float(s.max())
    if not np.isfinite(min_v) or not np.isfinite(max_v) or max_v <= min_v:
        return min_v, max_v
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = max(float(q3 - q1), 0.0)
    p01 = float(s.quantile(0.01))
    p99 = float(s.quantile(0.99))
    try:
        skew = float(s.skew())
    except Exception:
        skew = None

    view_lo = min_v
    view_hi = max_v
    extreme_right = max_v > max(p99, q3 + 8.0 * iqr) if iqr > 0 else max_v > p99
    extreme_left = min_v < min(p01, q1 - 8.0 * iqr) if iqr > 0 else min_v < p01
    if skew is not None:
        if skew > 1.0:
            extreme_right = True
        if skew < -1.0:
            extreme_left = True
    if extreme_right and not extreme_left:
        candidate_hi = max(p99, q3 + 3.0 * iqr) if iqr > 0 else p99
        if np.isfinite(candidate_hi) and candidate_hi < max_v:
            view_hi = float(candidate_hi)
    elif extreme_left and not extreme_right:
        candidate_lo = min(p01, q1 - 3.0 * iqr) if iqr > 0 else p01
        if np.isfinite(candidate_lo) and candidate_lo > min_v:
            view_lo = float(candidate_lo)
    elif extreme_left and extreme_right:
        candidate_lo = min(p01, q1 - 3.0 * iqr) if iqr > 0 else p01
        candidate_hi = max(p99, q3 + 3.0 * iqr) if iqr > 0 else p99
        if np.isfinite(candidate_lo) and candidate_lo > min_v:
            view_lo = float(candidate_lo)
        if np.isfinite(candidate_hi) and candidate_hi < max_v:
            view_hi = float(candidate_hi)
    if not np.isfinite(view_lo) or not np.isfinite(view_hi) or view_hi <= view_lo:
        return min_v, max_v
    return view_lo, view_hi


def _distribution_view_note_text(series: pd.Series) -> Optional[str]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    data_min = float(s.min())
    data_max = float(s.max())
    view_lo, view_hi = _distribution_view_bounds(s)
    clipped_left = view_lo > data_min
    clipped_right = view_hi < data_max
    if clipped_left and clipped_right:
        return "Widok skupia środkowy zakres; oba skrajne ogony są poza osią."
    if clipped_right:
        return "Widok skupia typowy zakres; skrajny prawy ogon jest poza osią."
    if clipped_left:
        return "Widok skupia typowy zakres; skrajny lewy ogon jest poza osią."
    return None


def _distribution_full_view_domain(
    series: pd.Series,
    lower_fence: float | None = None,
    upper_fence: float | None = None,
) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    min_v = float(s.min())
    max_v = float(s.max())
    if not np.isfinite(min_v) or not np.isfinite(max_v) or max_v <= min_v:
        return min_v, max_v
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    p05 = float(s.quantile(0.05))
    p95 = float(s.quantile(0.95))
    iqr = max(float(q3 - q1), 0.0)
    robust_span = max(float(p95 - p05), iqr * 1.5, 1e-9)
    full_span = max_v - min_v
    left_pad = max(robust_span * 0.03, iqr * 0.12, 0.5 if float(np.nanmax(np.abs(s.values))) <= 100 else 1.0)
    right_pad = max(robust_span * 0.03, full_span * 0.01, iqr * 0.12, 0.5 if float(np.nanmax(np.abs(s.values))) <= 100 else 1.0)
    try:
        has_left_outliers = bool((s < float(lower_fence)).any()) if lower_fence is not None else False
    except Exception:
        has_left_outliers = False
    try:
        has_right_outliers = bool((s > float(upper_fence)).any()) if upper_fence is not None else False
    except Exception:
        has_right_outliers = False
    if min_v >= 0 and not has_left_outliers:
        lo = max(0.0, min_v - min(left_pad, max(min_v * 0.10, 1.0)))
    else:
        lo = min_v - left_pad
    if lower_fence is not None and has_left_outliers:
        lo = min(lo, float(lower_fence))
    hi = max_v + right_pad
    if upper_fence is not None and has_right_outliers:
        hi = max(hi, float(upper_fence))
    return float(lo), float(hi)


def _build_hist_edges(values: np.ndarray, view_lo: float, view_hi: float, maxbins: int) -> np.ndarray:
    if values.size <= 1 or not np.isfinite(view_lo) or not np.isfinite(view_hi) or view_hi <= view_lo:
        return np.array([view_lo - 0.5, view_hi + 0.5], dtype=float)
    try:
        is_int_like = np.allclose(values, np.round(values))
    except Exception:
        is_int_like = False
    unique_n = int(pd.Series(values).nunique(dropna=True))
    if is_int_like and unique_n <= maxbins and (view_hi - view_lo) <= maxbins:
        start = math.floor(view_lo) - 0.5
        stop = math.ceil(view_hi) + 0.5
        edges = np.arange(start, stop + 1.0, 1.0, dtype=float)
        if edges.size >= 2:
            return edges
    n_bins = max(12, min(int(maxbins), max(20, unique_n if unique_n > 1 else 12)))
    return np.linspace(view_lo, view_hi, n_bins + 1, dtype=float)


def _preaggregate_histogram(
    df: pd.DataFrame,
    col: str,
    *,
    view_lo: float,
    view_hi: float,
    maxbins: int,
    color_field: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = pd.to_numeric(df[col], errors="coerce")
    mask = num.notna() & (num >= view_lo) & (num <= view_hi)
    if not bool(mask.any()):
        return pd.DataFrame(), pd.DataFrame()
    values = num.loc[mask].to_numpy(dtype=float)
    edges = _build_hist_edges(values, view_lo, view_hi, maxbins)
    if edges.size < 2:
        return pd.DataFrame(), pd.DataFrame()
    counts, _ = np.histogram(values, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    density = counts / max(float(values.size), 1.0) / np.where(widths > 0, widths, 1.0)
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    density_df = pd.DataFrame({col: centers, "density": np.convolve(density, kernel, mode="same")})
    bin_idx = np.searchsorted(edges, values, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    hist_base = pd.DataFrame({"bin_idx": bin_idx})
    if color_field and color_field in df.columns:
        hist_base[color_field] = df.loc[mask, color_field].astype(str).fillna("Brak").to_numpy()
        hist_df = hist_base.groupby(["bin_idx", color_field], dropna=False).size().reset_index(name="count")
    else:
        hist_df = hist_base.groupby("bin_idx", dropna=False).size().reset_index(name="count")
    hist_df["x"] = hist_df["bin_idx"].map(lambda i: float(edges[int(i)]))
    hist_df["x2"] = hist_df["bin_idx"].map(lambda i: float(edges[int(i) + 1]))
    hist_df["bin_start"] = hist_df["x"]
    hist_df["bin_end"] = hist_df["x2"]
    return hist_df, density_df

def _build_distribution_primary_hist_kde(
    df: pd.DataFrame,
    col: str,
    base_maxbins: int = 60,
    color_by: Optional[str] = None,
    color_col: str | None = None,
    height_scale: float = 1.35,
    topk_k: int = 0,
    other_label: str = "Other",
    color_domain: Optional[list[str]] = None,
) -> Optional[alt.Chart]:
    """
    Wykres rekomendowany dla gałęzi 'Distribution':

    - histogram liczności (granatowe słupki),
    - granatowa krzywa gęstości KDE,
    - czarna, grubsza linia z medianą,
    - dwie czerwone, przerywane linie progów IQR:
        * dolny próg – poniżej tego progu obserwacje traktujemy
          jako wartości odstające,
        * górny próg – powyżej tego progu obserwacje traktujemy
          jako wartości odstające.

    Uwaga: linie progów pokazują **granice definicji outlierów**.
    Jeśli w danych nie ma obserwacji poza tymi progami, nie sugerujemy
    sztucznie ich istnienia (brak cieni).
    """
    # ───── 1. Dane numeryczne ─────
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None

    # Wspólny helper: IQR + progi outlierów
    s, q1, q3, iqr, lower_fence, upper_fence = _compute_iqr_fences(series)

    if color_by and color_by in df.columns:
        work_df = pd.DataFrame({col: s, color_by: pd.Series(df.loc[s.index, color_by]).astype(str).values})
    else:
        work_df = pd.DataFrame({col: s})
        if isinstance(color_col, str) and (color_col in df.columns) and (color_col != col):
            try:
                work_df[color_col] = df.loc[work_df.index, color_col]
            except Exception:
                try:
                    work_df[color_col] = df[color_col]
                except Exception:
                    pass

    # ── SAFE: kolorowanie (ogranicz liczbę kategorii: Top-K + Other)
    if isinstance(color_col, str) and (color_col in work_df.columns) and (color_col != col):
        try:
            work_df[color_col] = work_df[color_col].astype(str).fillna("Brak")
        except Exception:
            work_df[color_col] = work_df[color_col]
        _color_field = color_col
    else:
        _color_field = None

    # Domain for legend/order (Top-K + Other) is computed upstream and passed in.
    _color_domain = list(color_domain) if color_domain else None

    data_min = float(s.min())
    data_max = float(s.max())
    median_val = float(s.median())

    maxbins = min(base_maxbins, max(10, s.nunique() // 2 or 10))
    navy = globals().get("_DISTRIB_NAVY", "#0057B7")

    # ───── 2. Histogram liczności ─────
    _color_scale = _safe_color_scale(_color_domain)

    hist = (
        alt.Chart(work_df)
        .mark_bar(opacity=1.0)
        .encode(
            x=alt.X(
                f"{col}:Q",
                bin=alt.Bin(maxbins=maxbins),
                title=col,
                axis=alt.Axis(
                    grid=False,
                    labelFontSize=11,
                    titleFontSize=12,
                ),
            ),
            y=alt.Y(
                "count():Q",
                title="Liczba rekordów",
                axis=alt.Axis(
                    gridOpacity=0.5,
                    labelFontSize=11,
                    titleFontSize=12,
                    tickMinStep=20,
                ),
            ),
            
color=(
    (
        alt.Color(
            f"{_color_field}:N",
            title=str(color_col),
            legend=alt.Legend(orient="right"),
            scale=_color_scale,
        )
        if _color_scale is not None
        else alt.Color(
            f"{_color_field}:N",
            title=str(color_col),
            legend=alt.Legend(orient="right"),
        )
    )
    if _color_field
    else alt.value(navy)
),
            tooltip=(
                [
                    alt.Tooltip(f"{_color_field}:N", title=_clean_label(color_col)),
                    alt.Tooltip("count():Q", title="Liczba rekordów"),
                ]
                if (isinstance(_color_field, str) and (_color_field in work_df.columns) and (_color_field != col))
                else [alt.Tooltip("count():Q", title="Liczba rekordów")]
            ),
        )
    )

    # ───── 3. Krzywa gęstości KDE ─────
    kde = (
        alt.Chart(work_df)
        .transform_density(col, as_=[col, "density"])
        .mark_line(color=navy)
        .encode(
            x=alt.X(f"{col}:Q", title=col),
            y=alt.Y("density:Q", title="Gęstość"),
        )
    )


    # ───── 4. Mediana (czarna, grubsza linia) ─────
    median_df = pd.DataFrame(
        {"value": [median_val], "label": ["Mediana"]}
    )

    median_rule = (
        alt.Chart(median_df)
        .mark_rule(color="#222222", strokeWidth=2, strokeDash=[6,6])
        .encode(x="value:Q")
    )

    median_label = (
        alt.Chart(median_df)
        .mark_text(
            dy=-10,                 # lekko pod linią
            baseline="bottom",    # tekst „przyklejony” do góry
            fontSize=11,
            color="#222222",
            align="center",
        )
        .encode(
            x="value:Q",
            y=alt.value(0),       # na górnej krawędzi wykresu
            text="label:N",
        )
    )

    # ───── 5. Linie progów outlierów + opisy ─────
    fence_df = pd.DataFrame(
        {
            "value": [lower_fence, upper_fence],
            "label": [
                "dolny próg IQR",
                "górny próg IQR",
            ],
        }
    )

    fence_rules = (
        alt.Chart(fence_df)
        .mark_rule(color="#e76f51", strokeDash=[2, 2])
        .encode(x="value:Q")
    )

    fence_labels = (
        alt.Chart(fence_df)
        .mark_text(
            dy=-10,                  # lekko pod linią
            baseline="bottom",     # tekst „przyklejony” do góry
            fontSize=11,
            color="#444444",
            align="center",
        )
        .encode(
            x="value:Q",
            y=alt.value(0),        # na górnej krawędzi wykresu
            text="label:N",
        )
    )

    # ───── 6. Warstwy razem ─────
    # UWAGA: w Altair nie można wkładać wykresu z `padding` jako elementu LayerChart.
    # Dlatego wszystkie warstwy składamy w JEDNYM alt.layer(...), a padding ustawiamy dopiero na końcu.
    chart = (
        alt.layer(
            hist,
            kde,
            median_rule,
            fence_rules,
            median_label,
            fence_labels,
        )
        .resolve_scale(y="independent")
    )

    # Layout: kontrolowany padding (jedno miejsce), bez ucinania osi i legendy.
    return (
        chart
        .properties(height=int(450 * height_scale), padding=HIST_VIEW_PADDING)
        .configure_axis(gridOpacity=0.5, titlePadding=10, labelPadding=6)
        .configure_legend(orient="right", titlePadding=8, padding=6, labelLimit=220)
        .configure_view(strokeOpacity=0)
    )

def _build_distribution_primary_hist_kde_v2(
    df: pd.DataFrame,
    col: str,
    base_maxbins: int = 60,
    color_by: Optional[str] = None,
    color_col: str | None = None,
    height_scale: float = 1.35,
    topk_k: int = 0,
    other_label: str = "Other",
    color_domain: Optional[list[str]] = None,
) -> Optional[alt.Chart]:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None

    s, q1, q3, iqr, lower_fence, upper_fence = _compute_iqr_fences(series)

    _color_source = color_by if (isinstance(color_by, str) and color_by in df.columns and color_by != col) else None
    _color_title = None
    if isinstance(color_col, str) and color_col and color_col != col:
        _color_title = color_col
    elif isinstance(_color_source, str):
        _color_title = _color_source

    if _color_source:
        work_df = pd.DataFrame(
            {col: s, _color_source: pd.Series(df.loc[s.index, _color_source]).astype(str).values}
        )
    else:
        work_df = pd.DataFrame({col: s})
        if isinstance(color_col, str) and (color_col in df.columns) and (color_col != col):
            try:
                work_df[color_col] = df.loc[work_df.index, color_col]
            except Exception:
                try:
                    work_df[color_col] = df[color_col]
                except Exception:
                    pass

    if _color_source and (_color_source in work_df.columns):
        try:
            work_df[_color_source] = work_df[_color_source].astype(str).fillna("Brak")
        except Exception:
            work_df[_color_source] = work_df[_color_source]
        _color_field = _color_source
    elif isinstance(color_col, str) and (color_col in work_df.columns) and (color_col != col):
        try:
            work_df[color_col] = work_df[color_col].astype(str).fillna("Brak")
        except Exception:
            work_df[color_col] = work_df[color_col]
        _color_field = color_col
    else:
        _color_field = None

    _color_domain = list(color_domain) if color_domain else None
    _color_scale = _safe_color_scale(_color_domain)

    data_min = float(s.min())
    data_max = float(s.max())
    median_val = float(s.median())
    maxbins = min(base_maxbins, max(10, s.nunique() // 2 or 10))
    navy = globals().get("_DISTRIB_NAVY", "#0057B7")

    view_lo, view_hi = _distribution_view_bounds(s)
    hist_df, density_df = _preaggregate_histogram(
        work_df,
        col,
        view_lo=view_lo,
        view_hi=view_hi,
        maxbins=maxbins,
        color_field=_color_field,
    )
    if hist_df.empty or density_df.empty:
        return None

    x_scale = alt.Scale(domain=[view_lo, view_hi], nice=False, zero=False)
    max_count = float(hist_df["count"].max()) if not hist_df.empty else 0.0
    max_density = float(density_df["density"].max()) if not density_df.empty else 0.0
    density_format = ".4f" if max_density < 0.01 else (".3f" if max_density < 0.1 else ".2f")

    color_enc = (
        (
            alt.Color(
                f"{_color_field}:N",
                title=_clean_label(_color_title or _color_field),
                legend=alt.Legend(orient="right"),
                scale=_color_scale,
            )
            if _color_scale is not None
            else alt.Color(
                f"{_color_field}:N",
                title=_clean_label(_color_title or _color_field),
                legend=alt.Legend(orient="right"),
            )
        )
        if _color_field
        else alt.value(navy)
    )

    hist_tooltip = [
        alt.Tooltip("bin_start:Q", title=f"{col} od", format=",.2f"),
        alt.Tooltip("bin_end:Q", title=f"{col} do", format=",.2f"),
        alt.Tooltip("count:Q", title="Liczba rekordów", format=",.0f"),
    ]
    if _color_field:
        hist_tooltip.insert(0, alt.Tooltip(f"{_color_field}:N", title=_clean_label(_color_title or _color_field)))

    hist = (
        alt.Chart(hist_df)
        .mark_bar(opacity=1.0)
        .encode(
            x=alt.X(
                "x:Q",
                title=col,
                scale=x_scale,
                axis=alt.Axis(grid=False, labelFontSize=11, titleFontSize=12),
            ),
            x2="x2:Q",
            y=alt.Y(
                "count:Q",
                title="Liczba rekordów",
                axis=alt.Axis(
                    gridOpacity=0.5,
                    labelFontSize=11,
                    titleFontSize=12,
                    tickMinStep=20,
                    orient="left",
                    format=",.0f",
                ),
            ),
            color=color_enc,
            tooltip=hist_tooltip,
        )
    )

    kde = (
        alt.Chart(density_df)
        .mark_line(color=navy)
        .encode(
            x=alt.X(f"{col}:Q", title=col, scale=x_scale),
            y=alt.Y(
                "density:Q",
                title="Gęstość",
                axis=alt.Axis(
                    orient="right",
                    format=density_format,
                    labelFontSize=11,
                    titleFontSize=12,
                    grid=False,
                    tickCount=5,
                    labelColor=navy,
                    titleColor=navy,
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{col}:Q", title=col, format=",.2f"),
                alt.Tooltip("density:Q", title="Gęstość", format=",.4f"),
            ],
        )
    )

    median_df = pd.DataFrame({"value": [median_val], "label": ["Mediana"]})
    median_rule = (
        alt.Chart(median_df)
        .mark_rule(color="#222222", strokeWidth=2, strokeDash=[6, 6])
        .encode(x=alt.X("value:Q", scale=x_scale))
    )
    median_label = (
        alt.Chart(median_df)
        .mark_text(dy=-10, baseline="bottom", fontSize=11, color="#222222", align="center")
        .encode(x=alt.X("value:Q", scale=x_scale), y=alt.value(0), text="label:N")
    )

    fence_df = pd.DataFrame(
        {
            "value": [lower_fence, upper_fence],
            "label": ["dolny próg IQR", "górny próg IQR"],
        }
    )
    fence_rules = (
        alt.Chart(fence_df)
        .mark_rule(color="#e76f51", strokeDash=[2, 2])
        .encode(x=alt.X("value:Q", scale=x_scale))
    )
    fence_labels = (
        alt.Chart(fence_df)
        .mark_text(dy=-10, baseline="bottom", fontSize=11, color="#444444", align="center")
        .encode(x=alt.X("value:Q", scale=x_scale), y=alt.value(0), text="label:N")
    )

    chart = alt.layer(
        hist,
        kde,
        median_rule,
        fence_rules,
        median_label,
        fence_labels,
    ).resolve_scale(y="independent")

    return (
        chart
        .properties(width="container", height=int(450 * height_scale), padding=HIST_VIEW_PADDING)
        .configure_axis(gridOpacity=0.5, titlePadding=10, labelPadding=6)
        .configure_legend(orient="right", titlePadding=8, padding=6, labelLimit=220)
        .configure_view(strokeOpacity=0)
    )

def _augment_distribution_alternatives(
    df: pd.DataFrame,
    col: str,
    alt_charts: list[Any],
) -> list[tuple[str, alt.Chart]]:
    """
    Komplet alternatywnych wykresów dla gałęzi 'Distribution':
    - Boxplot – rozkład i outliery (czerwone kropki)
    - Gęstość rozkładu (KDE)
    - Dystrybuanta (ECDF)
    - Histogram – węższe przedziały
    - Histogram – skala logarytmiczna
    - Violin – rozkład i gęstość (z medianą i fence’ami + opisami)
    - Rug – wartości odstające (przygotowany pod drugą zmienną)

    Na końcu doklejamy ewentualne alternatywy z LLM (alt_charts),
    pilnując braku duplikatów po etykietach.
    """
    if col not in df.columns:
        return []

    work_df = df[[col]].dropna().copy()
    if work_df.empty:
        return []

    s = pd.to_numeric(work_df[col], errors="coerce").dropna()
    if s.empty:
        return []

    # ── Statystyki IQR ─────────────────────────────────────────────
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr < 0:
        iqr = 0.0

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    min_v = float(s.min())
    max_v = float(s.max())
    median_v = float(s.median())
    insight_lo, insight_hi = _distribution_full_view_domain(s, lower_fence, upper_fence)
    insight_x_scale = alt.Scale(domain=[insight_lo, insight_hi], nice=False, zero=False)

    base_maxbins = 40

    base_alts: list[tuple[str, alt.Chart]] = []

    # ── 1) Boxplot – rozkład i outliery (wariant B, uporządkowany) ────────────────
    BOX_HEIGHT = 200          # wysokość wykresu (możesz podbić np. do 180)
    OUTLIER_SIZE = 25         # rozmiar kropek outlierów

    # dane pomocnicze do mediany i fence’ów
    median_df = pd.DataFrame({col: [median_v]})
    fence_df = pd.DataFrame(
        {
            col: [lower_fence, upper_fence],
            "label": ["dolny próg IQR", "górny próg IQR"],
        }
    )

    # outliery
    outliers_df = work_df[
        (work_df[col] < lower_fence) | (work_df[col] > upper_fence)
    ]

    # Główny boxplot (bez mediany / fence’ów – to dorysujemy sami)
    base_box = (
        alt.Chart(work_df)
        .mark_boxplot(
            size=30,
            ticks=True,
            extent=1.5,       # wąsy do 1.5 * IQR (standard boxplotu)
            outliers=False,   # NIE rysuj wbudowanych outlierów
        )
        .encode(
            x=alt.X(
                f"{col}:Q",
                title=col,
                scale=insight_x_scale,
                # scale=alt.Scale(domain=[lower_fence - (iqr * 0.2), max_v + (iqr * 0.2)]),     # opcjonalnie – rozszerzenie zakresu osi żeby wyśroidować boxplota
                axis=alt.Axis(
                    grid=True,
                    gridOpacity=0.35,   # siatka dla osi X
                ),
            )
        )
        .properties(height=BOX_HEIGHT)
    )


    layers = [base_box]

    # ── mediana: gruba czarna linia przerywana + etykieta nad linią ───────────────
    median_rule = (
        alt.Chart(median_df)
        .mark_rule(color="black", strokeWidth=2, strokeDash=[6, 6])
        .encode(x=alt.X(f"{col}:Q", scale=insight_x_scale))
    )

    median_label = (
        alt.Chart(median_df)
        .mark_text(
            text="Mediana",
            dy=-20,                 # kilka pikseli poniżej górnej krawędzi
            align="center",
            baseline="top",       # tekst „wisi” od góry w dół
            color="black",
            fontSize=11,
        )
        .encode(
            x=alt.X(f"{col}:Q", scale=insight_x_scale),
            y=alt.value(0),       # ← SZCZYT WYKRESU
        )
    )


    layers.extend([median_rule, median_label])

    # ── fence’y IQR: pionowe różowe linie przez cały wykres + czarne etykiety ────
    fence_rules = (
        alt.Chart(fence_df)
        .mark_rule(color="#e76f51", strokeDash=[2, 2], strokeWidth=1.3)
        .encode(x=alt.X(f"{col}:Q", scale=insight_x_scale))
    )

    fence_labels = (
        alt.Chart(fence_df)
        .mark_text(
            dy=-20,
            align="center",
            baseline="top",
            color="black",
            fontSize=11,
        )
        .encode(
            x=alt.X(f"{col}:Q", scale=insight_x_scale),
            y=alt.value(0),   # ← górna krawędź wykresu
            text="label:N",
        )
    )

    layers.extend([fence_rules, fence_labels])

    # ── outliery: mniejsze kropki z czerwonym obrysem ────────────────────────────
    if not outliers_df.empty:
        outliers_chart = (
            alt.Chart(outliers_df)
            .mark_point(
                size=OUTLIER_SIZE,
                filled=False,        # pusty środek
                color="#e63946",     # czerwony obrys
                strokeWidth=1.2,
                opacity=0.9,
            )
            .encode(x=alt.X(f"{col}:Q", scale=insight_x_scale))
        )
        layers.append(outliers_chart)


    # finalna warstwa boxplota:
    #  - pełna oś X z siatką,
    #  - pionowe linie mediany + fence’ów,
    #  - czarne etykiety nad liniami,
    #  - czerwone obrysy outlierów
    box_layer = (
        alt.layer(*layers)
        .properties(height=BOX_HEIGHT, width="container", padding=INSIGHT_CHART_PADDING)
        .configure_axis(gridOpacity=0.4)   # ← tu wklejamy konfigurację siatki
    )

    base_alts.append(("Boxplot – rozkład i outliery", box_layer))


    # ── 2) Gęstość rozkładu (KDE) – sam wykres gęstości ────────────
    try:
        kde_only = (
            alt.Chart(work_df)
            .transform_density(
                col,
                as_=[col, "density"],
                extent=[min_v, max_v],
            )
            .mark_line(color="#0057B7", strokeWidth=2)
            .encode(
                x=alt.X(
                    f"{col}:Q",
                    title=col,
                    scale=insight_x_scale,
                    axis=alt.Axis(gridOpacity=0.35),
                ),
                y=alt.Y(
                    "density:Q",
                    title="Gęstość",
                    axis=alt.Axis(gridOpacity=0.35),
                ),
            )
            .properties(height=300, width="container", padding=INSIGHT_CHART_PADDING)
        )
        base_alts.append(("Gęstość rozkładu (KDE)", kde_only))
    except Exception as e:  # noqa: BLE001
        print("[Distribution] Błąd przy budowie KDE:", e)

    # ── 3) Dystrybuanta (ECDF) ─────────────────────────────────────
    try:
        ecdf_df = work_df.sort_values(col).copy()
        n_total = len(ecdf_df)
        ecdf_df["_rank"] = range(1, n_total + 1)
        ecdf_df["ecdf"] = ecdf_df["_rank"] / float(n_total)

        # Podstawowa linia ECDF – ten sam granat, co KDE itp.
        base_ecdf = (
            alt.Chart(ecdf_df)
            .mark_line(color="#0057B7")
            .encode(
                x=alt.X(
                    f"{col}:Q",
                    title=col,
                    scale=insight_x_scale,
                    axis=alt.Axis(gridOpacity=0.35),
                ),
                y=alt.Y(
                    "ecdf:Q",
                    title="Dystrybuanta (ECDF)",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(gridOpacity=0.35),
                ),
            )
            .properties(height=300)
        )

        layered = base_ecdf

        # Jeśli mamy sensowny IQR – dodaj pionowe linie progów + opisy
        if iqr > 0 and lower_fence is not None and upper_fence is not None:
            fences_df = pd.DataFrame(
                {
                    col: [lower_fence, upper_fence],
                    "label": ["dolny próg IQR", "górny próg IQR"],
                }
            )

            fence_rules = (
                alt.Chart(fences_df)
                .mark_rule(strokeDash=[2, 2], stroke="#e76f51")
                .encode(x=alt.X(f"{col}:Q", scale=insight_x_scale))
            )

            fence_labels = (
                alt.Chart(fences_df)
                .mark_text(
                    dy=-10,           # lekko nad linią
                    baseline="bottom",     # tekst „przyklejony” do góry
                    fontSize=10,
                    color="#444444",
                    align="center",
                )
                .encode(
                    x=alt.X(f"{col}:Q", scale=insight_x_scale),
                    y=alt.value(0),        # na górnej krawędzi wykresu
                    text="label:N",
                )
            )

            layered = alt.layer(base_ecdf, fence_rules, fence_labels)

        ecdf_chart = layered.properties(height=300, width="container", padding=INSIGHT_CHART_PADDING)
        base_alts.append(("Dystrybuanta (ECDF)", ecdf_chart))

    except Exception as e:  # noqa: BLE001
        print("[Distribution] Błąd przy budowie ECDF:", e)


    # ── 4) Histogram – węższe przedziały ──────────────────────────
    hist_narrow = (
        alt.Chart(work_df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{col}:Q",
                bin=alt.Bin(maxbins=base_maxbins * 2),
                title=col,
                scale=insight_x_scale,
                axis=alt.Axis(gridOpacity=0.35),
            ),
            y=alt.Y(
                "count():Q",
                title="Liczba rekordów",
                axis=alt.Axis(gridOpacity=0.35),
            ),
        )
        .properties(height=300, width="container", padding=INSIGHT_CHART_PADDING)
    )
    base_alts.append(("Histogram – węższe przedziały", hist_narrow))

    # ── 5) Histogram – skala logarytmiczna ────────────────────────
    hist_log = (
        alt.Chart(work_df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{col}:Q",
                bin=alt.Bin(maxbins=base_maxbins),
                title=col,
                scale=insight_x_scale,
                axis=alt.Axis(gridOpacity=0.35),
            ),
            y=alt.Y(
                "count():Q",
                title="Liczba rekordów (skala log)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(gridOpacity=0.35),
            ),
        )
        .properties(height=300, width="container", padding=INSIGHT_CHART_PADDING)
    )
    base_alts.append(("Histogram – skala logarytmiczna", hist_log))

    # 6) Violin – symetryczny (IQR + mediana + fence’y + outliery) — McKinsey-grade
    _NAVY = "#1f77b4"     # gęstość
    _FENCE = "#e76f51"    # linie progów IQR
    _OUT = "#e63946"      # outliery

    violin_data = pd.DataFrame({col: s})

    # Max gęstości do skali i pozycjonowania etykiet (bez SciPy)
    try:
        _hist, _edges = np.histogram(
            s.values.astype(float),
            bins=160,
            range=(min_v, max_v),
            density=True,
        )
        _max_density = float(np.nanmax(_hist)) if _hist.size else 1.0
    except Exception:
        _max_density = 1.0
    if not np.isfinite(_max_density) or _max_density <= 0:
        _max_density = 1.0

    # Etykiety nad „górną połówką”
    _label_y = _max_density * 0.96

    # --- IQR band: delikatna „wstęga” przez cały violin (od - do +) ---
    iqr_band_df = pd.DataFrame(
        {"x1": [q1], "x2": [q3], "y1": [-_max_density], "y2": [_max_density]}
    )
    iqr_band = (
        alt.Chart(iqr_band_df)
        .mark_rect(color=_NAVY, opacity=0.10)
        .encode(
            x=alt.X("x1:Q", scale=insight_x_scale),
            x2="x2:Q",
            y=alt.Y("y1:Q"),
            y2="y2:Q",
        )
    )

    # --- Violin density (KDE) – SYMETRYCZNY: y = -density, y2 = +density ---
    violin_density = (
        alt.Chart(violin_data)
        .transform_density(
            col,
            as_=[col, "density"],
            extent=[min_v, max_v],
            steps=220,
        )
        .transform_calculate(
            y1="-(datum.density)",
            y2="datum.density",
        )
        .mark_area(
            color=_NAVY,
            opacity=0.35,
        )
        .encode(
            x=alt.X(
                f"{col}:Q",
                title=col,
                scale=insight_x_scale,
                axis=alt.Axis(
                    grid=True,
                    gridOpacity=0.35,
                    tickCount=8,
                    labelFlush=True,
                ),
            ),
            y=alt.Y(
                "y1:Q",
                title=None,  # bez „Gęstość”
                scale=alt.Scale(domain=[-_max_density * 1.02, _max_density * 1.02]),
                axis=alt.Axis(
                    labels=False,
                    ticks=False,
                    domain=False,
                    grid=True,
                    gridOpacity=0.35,
                ),
            ),
            y2="y2:Q",
        )
        .properties(height=260)
    )
        # Kontur górnej połówki (y2)
    violin_outline_top = (
        alt.Chart(violin_data)
        .transform_density(
            col,
            as_=[col, "density"],
            extent=[min_v, max_v],
            steps=220,
        )
        .mark_line(color=_NAVY, opacity=0.9, strokeWidth=1.2)
        .encode(
            x=alt.X(f"{col}:Q", scale=insight_x_scale),
            y=alt.Y("density:Q"),
        )
    )

    # Kontur dolnej połówki (y1 = -density)
    violin_outline_bottom = (
        alt.Chart(violin_data)
        .transform_density(
            col,
            as_=[col, "density"],
            extent=[min_v, max_v],
            steps=220,
        )
        .transform_calculate(y="-(datum.density)")
        .mark_line(color=_NAVY, opacity=0.9, strokeWidth=1.2)
        .encode(
            x=alt.X(f"{col}:Q", scale=insight_x_scale),
            y=alt.Y("y:Q"),
        )
    )

    # --- Mediana: linia + etykieta nad linią ---
    median_rule = (
        alt.Chart(pd.DataFrame({col: [median_v]}))
        .mark_rule(color="#000000", strokeWidth=2, strokeDash=[6,6])
        .encode(x=alt.X(f"{col}:Q", scale=insight_x_scale))
    )

    median_label = (
        alt.Chart(pd.DataFrame({col: [median_v], "label": ["Mediana"]}))
        .mark_text(
            dy=-6,
            baseline="bottom",
            color="#000000",
            fontSize=11,
            align="center",
        )
        .encode(
            x=alt.X(f"{col}:Q", scale=insight_x_scale),
            y=alt.value(_label_y),
            text="label:N",
        )
    )

    # --- Fence’y IQR: linie przerywane + czarne opisy ---
    fence_df_violin = pd.DataFrame(
        {
            col: [lower_fence, upper_fence],
            "label": ["dolny próg IQR", "górny próg IQR"],
        }
    )

    fence_rules = (
        alt.Chart(fence_df_violin)
        .mark_rule(color=_FENCE, strokeDash=[2, 2], strokeWidth=1.3)
        .encode(x=alt.X(f"{col}:Q", scale=insight_x_scale))
    )

    fence_labels = (
        alt.Chart(fence_df_violin)
        .mark_text(
            dy=-6,
            baseline="bottom",
            color="#000000",
            fontSize=11,
            align="center",
        )
        .encode(
            x=alt.X(f"{col}:Q", scale=insight_x_scale),
            y=alt.value(_label_y),
            text="label:N",
        )
    )

    # --- Outliery: punkty na osi 0 (środek violina) — czytelnie i neutralnie ---
    try:
        out_s = s[(s < lower_fence) | (s > upper_fence)]
        if len(out_s) > 2500:
            out_s = out_s.sample(2500, random_state=7)
        out_df = pd.DataFrame({col: out_s.values.astype(float), "y": np.zeros(len(out_s), dtype=float)})
    except Exception:
        out_df = pd.DataFrame({col: [], "y": []})

    outliers_layer = (
        alt.Chart(out_df)
        .mark_point(filled=True, size=28, color=_OUT, opacity=0.80)
        .encode(
            x=alt.X(f"{col}:Q", scale=insight_x_scale),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-_max_density * 1.02, _max_density * 1.02])),
            tooltip=[alt.Tooltip(f"{col}:Q", title=col)],
        )
    )

    violin_chart = (
        alt.layer(
            iqr_band,
            violin_density,
            violin_outline_top,
            violin_outline_bottom,
            outliers_layer,
            fence_rules,
            median_rule,
            fence_labels,
            median_label,
        )
        .configure_axis(gridOpacity=0.35)
        .configure_view(strokeWidth=0)
        .properties(height=260, width="container", padding=INSIGHT_CHART_PADDING)
    )

    base_alts.append(
        ("Violin – rozkład i gęstość (IQR + mediana + fence’y + outliery)", violin_chart)
    )


    # 7) Rug – wartości odstające (przygotowany pod drugą zmienną)

    def _build_rug_chart(
        df: pd.DataFrame,
        col: str,
        lower_fence: float | None = None,
        upper_fence: float | None = None,
        title: str | None = None,
    ) -> Optional[alt.Chart]:
        """
        Rug plot (pozycje obserwacji na osi X) dla rozkładu.
        Poprawki pod UI:
        - oś X (ticki + etykiety + tytuł) MA BYĆ wyraźnie widoczna
        - nie ucinać tytułu osi
        - outliery na czerwono (spójnie z violin)
        """
        if df is None or col is None or col not in df.columns:
            return None

        work_df = df[[col]].dropna().copy()
        if work_df.empty:
            return None

        # (opcjonalnie) downsample dla wydajności – zostawiamy Twoje zachowanie
        if len(work_df) > 8000:
            work_df = work_df.sample(8000, random_state=42).sort_values(col)

        # flaga outlierów (jeśli mamy fence’y)
        if lower_fence is not None and upper_fence is not None:
            work_df["_is_outlier"] = (work_df[col] < float(lower_fence)) | (work_df[col] > float(upper_fence))
        else:
            work_df["_is_outlier"] = False

        # oś X: dopinamy paddingi żeby nie ucinało label/title
        # jeśli dostajesz gdzieś x_axis z zewnątrz – tu robimy wersję “bezpieczną”
        x_axis = alt.Axis(
            title=str(col),
            labelOverlap=False,
            tickCount=8,
            labelPadding=8,
            titlePadding=14,
            tickSize=4,
            domain=True,
            grid=True,
        )

        ch = (
            alt.Chart(work_df)
            .mark_tick(size=35, thickness=1.6, opacity=0.55)
            .encode(
                x=alt.X(f"{col}:Q", axis=x_axis, scale=insight_x_scale),
                y=alt.value(0),
                color=alt.Color(
                    "_is_outlier:N",
                    scale=alt.Scale(domain=[False, True], range=[_NAVY, _OUT]),
                    legend=None,
                ),
                tooltip=[alt.Tooltip(f"{col}:Q", title=str(col))],
            )
            # Klucz: dajemy realny bottom padding, żeby oś X + tytuł NIE były „podcięte”
            .properties(
                height=85,
                width="container",
                padding={"left": 12, "right": 6, "top": 0, "bottom": 0},
            )
            # dodatkowo: czytelność osi X jak w referencji
            .configure_axisX(
                labelFontSize=12,
                titleFontSize=13,
                labelPadding=8,
                titlePadding=14,
                tickSize=4,
            )
            .configure_view(strokeWidth=0)
        )

        if title:
            ch = ch.properties(title=str(title))

        return ch


    rug_chart = _build_rug_chart(df, col, lower_fence, upper_fence, None)
    base_alts.append(("Rug – położenie punktów", rug_chart))


    # ── 8) Doklejamy alternatywy z LLM (jeśli są) ──────────────────
    used_labels = {label for label, _ in base_alts}

    for item in alt_charts or []:
        if item is None:
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, chart = item[0], item[1]
        else:
            label = (
                getattr(item, "label", None)
                or getattr(item, "name", None)
                or "Alternatywna wizualizacja"
            )
            chart = getattr(item, "chart", item)

        if chart is None:
            continue

        label_str = str(label)
        if label_str in used_labels:
            # unikamy duplikatów po etykiecie
            continue

        base_alts.append((label_str, chart))
        used_labels.add(label_str)

    return base_alts


def render(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Render gałęzi Distribution wewnątrz dostarczonych tabów.
    ctx MUST: schema_ctx, chart_spec, question, tabs(dict: overview/insights/segments/whatif)
    Zwraca meta potrzebne routerowi (do interpretacji/TTS).
    """
    schema_ctx = ctx.get("schema_ctx") or {}
    chart_spec = ctx.get("chart_spec") or {}
    question = str(ctx.get("question") or "")
    tabs = ctx.get("tabs") or {}
    record_debug_checkpoint(
        "dist.render.start",
        question=question.strip(),
        chart_intent=chart_spec.get("intent") if isinstance(chart_spec, dict) else None,
        filter_keys=list(((ctx.get("filters") or {}) if isinstance(ctx, dict) else {}).keys()),
    )

    _res = _build_chart_bundle_from_spec(df, chart_spec)

    if isinstance(_res, tuple) and len(_res) >= 2:

        chart_bundle, chart_meta = _res[0], _res[1]

    else:

        chart_bundle, chart_meta = _res, {}

    primary_chart = chart_bundle.get("primary")
    alt_charts = chart_bundle.get("alt") or []
    scenario = chart_bundle.get("scenario") or {}
    ts_bundle = chart_bundle.get("ts") or {}
    clusters_bundle = chart_bundle.get("clusters") or {}

    # --- Dodatkowe ulepszenia Distribution (jak w oryginale) ---
    dist_col: Optional[str] = None

    def _looks_like_id(name: str | None) -> bool:
        if not name:
            return False
        low = str(name).lower()
        if low in {"id", "idx", "index"}:
            return True
        if low.endswith("_id"):
            return True
        return ("id" in low) and (len(low) <= 6)

    if isinstance(chart_spec, dict) and (chart_spec.get("intent") == "distribution"):
        primary_cfg = chart_spec.get("primary_chart") or {}
        cand = primary_cfg.get("x")

        # --- Sidebar/ctx filters override (MVP) ---
        _filters = (ctx or {}).get("filters") or st.session_state.get("dc_filters") or {}

        # ✅ 0) dist_col z filtra ma absolutny priorytet (jeśli jest numeryczny)
        _override_col = _filters.get("dist_col")
        if _override_col and _override_col in df.columns and pd.api.types.is_numeric_dtype(df[_override_col]):
            cand = _override_col

        # Apply optional filtering (range + category) — na df_work
        df_work = df

        # ✅ zakres dla miary (klucz z routera: dist_range)
        _rng = _filters.get("dist_range")
        if _rng and isinstance(_rng, (list, tuple)) and len(_rng) == 2 and cand in df_work.columns:
            lo, hi = _rng
            try:
                df_work = df_work[
                    (pd.to_numeric(df_work[cand], errors="coerce") >= float(lo))
                    & (pd.to_numeric(df_work[cand], errors="coerce") <= float(hi))
                ]
            except Exception:
                pass

        # ✅ filtr kategorii (klucze z routera: filt_col + filt_values)
        _fcol = _filters.get("filt_col")
        _fvals = _filters.get("filt_values")
        if _fcol and isinstance(_fvals, (list, tuple)) and len(_fvals) > 0 and _fcol in df_work.columns:
            try:
                df_work = df_work[df_work[_fcol].isin(_fvals)]
            except Exception:
                pass

        # ── SAFE FRONTEND MODE: prevent browser overload (Altair JSON) on large data
        # KPIs/stats remain computed on df_work (full filtered dataset).
        _safe_cfg = get_cfg_from_ctx(ctx, fallback_rows=len(df_work))
        df_chart = safe_sample_df(df_work, _safe_cfg)

        
        # ✅ kolorowanie (klucz z routera: color_col)
        _color_by = _filters.get("color_col")
        if _color_by in (None, "(brak)"):
            _color_by = None

        # ✅ Color segmentation only when it makes sense (categorical / low-card numeric)
        if isinstance(_color_by, str) and _color_by and (_color_by in df_chart.columns):
            if not _is_categorical_color_col(df_chart, _color_by) or (_color_by == cand):
                _color_by = None
        else:
            _color_by = None

        # ✅ Top-K policy (SAFE): deterministic Top-K by COUNT (only for categorical color_by)
        _topk_k: int = 0
        if isinstance(_color_by, str) and (_color_by in df_chart.columns) and (_color_by != cand):
            try:
                _vc = df_chart[_color_by].astype(str).fillna("Brak").value_counts(dropna=False)
                _topk_k = min(12, int(len(_vc)))
            except Exception:
                _topk_k = 0

        # ✅ Utrwalamy w filters: będzie użyte do stabilizacji rerenderu (cache/render key)
        # i do wyjaśnień/diagnostyki (debug).
        try:
            _filters_local = dict(_filters)
            _filters_local["topk_k"] = int(_topk_k)
            _filters_local["safe"] = {
                "enabled": bool(_safe_cfg.get("enabled", False)),
                "sample_size": int(_safe_cfg.get("sample_size", 0) or 0),
            }
            ctx["filters"] = _filters_local
            st.session_state["dc_filters"] = _filters_local
            _filters = _filters_local
        except Exception:
            pass

        # 1) Priorytet: kolumna wspomniana w pytaniu (jeśli nie ma override)
        q_lower = question.lower()
        cols = [c for c in df.columns if isinstance(c, str)]
        mentioned = next((c for c in cols if c.lower() in q_lower and not _looks_like_id(c)), None)

        dist_col = cand
        if mentioned and not _override_col:
            dist_col = mentioned

        # fallbacki bezpieczeństwa
        if (not isinstance(dist_col, str)) or (dist_col not in df.columns) or _looks_like_id(dist_col):
            cand_name = (dist_col or "")
            cand_token = cand_name.strip().lower() if isinstance(cand_name, str) else ""
            if cand_token:
                for c in df.columns:
                    if isinstance(c, str) and (cand_token in c.lower()) and (not _looks_like_id(c)):
                        dist_col = c
                        break

        if (not isinstance(dist_col, str)) or (dist_col not in df.columns) or _looks_like_id(dist_col):
            numeric_cols = schema_ctx.get("numeric_cols") or []
            numeric_cols = [c for c in numeric_cols if isinstance(c, str) and c in df.columns]
            if not numeric_cols:
                numeric_cols = [c for c in _infer_numeric_cols(df) if isinstance(c, str) and c in df.columns]
            clean_numeric = [c for c in numeric_cols if not _looks_like_id(c)]
            dist_col = clean_numeric[0] if clean_numeric else (numeric_cols[0] if numeric_cols else None)
        # ✅ KLUCZOWA NAPRAWA METRYK: Aktualizuj metryki PO ustaleniu dist_col (bez migania etykiet)
        try:
            _ui_ctx = ctx.get("ui") or {}
            _overview_ui = (_ui_ctx.get("overview") or {})
            _mval = _overview_ui.get("metric_value")
            _mqty = _overview_ui.get("metric_qty")
            _mmed = _overview_ui.get("metric_median") or _mqty
            _mtxn = _overview_ui.get("metric_txn")

            # KPI #3: liczba transakcji (liczba rekordów)
            if _mtxn is not None:
                _mtxn.metric("Liczba transakcji", f"{len(df_work):,}".replace(",", " "))

            # KPI #1: suma wybranej „Miary do rozkładu” (dist_col)
            if _mval is not None:
                if dist_col and dist_col in df_work.columns and pd.api.types.is_numeric_dtype(df_work[dist_col]):
                    _total_metric = float(pd.to_numeric(df_work[dist_col], errors="coerce").sum())
                    _mval.metric(f"Suma {dist_col}", f"{_total_metric:,.0f}".replace(",", " "))
                else:
                    _mval.metric("Suma", "—")

            # KPI #2: mediana wybranej „Miary do rozkładu” (dist_col)
            if _mmed is not None:
                if dist_col and dist_col in df_work.columns and pd.api.types.is_numeric_dtype(df_work[dist_col]):
                    _med_metric = float(pd.to_numeric(df_work[dist_col], errors="coerce").median())
                    _mmed.metric(f"Mediana {dist_col}", _fmt_num(_med_metric))
                else:
                    _mmed.metric("Mediana", "—")
        except Exception:
            pass
        record_debug_checkpoint(
            "dist.render.resolved",
            dist_col=dist_col,
            color_by=_color_by,
            filt_col=_filters.get("filt_col"),
            filt_values=list(_filters.get("filt_values") or []),
            dist_range=list(_filters.get("dist_range") or []),
            topk_k=int(_filters.get("topk_k") or 0),
            filtered_rows=int(len(df_work)),
            chart_rows=int(len(df_chart)),
            safe_enabled=bool((_filters.get("safe") or {}).get("enabled", False)),
            safe_sample_size=int((_filters.get("safe") or {}).get("sample_size", 0) or 0),
        )
        _primary_view_note = None
        if dist_col and dist_col in df.columns:
            try:
                # ✅ buduj primary na df_work (uwzględnia range + category)
                # bins/top-k (stable across rerenders)
                _maxbins = int(_filters.get("base_maxbins") or _filters.get("maxbins") or 40)
                _topk_k = int(_filters.get("topk_k") or 0)
                _other_label = str(_filters.get("other_label") or "Other")
                # Apply stable Top-K + Other bucketing for the selected color dimension (if enabled)
                # on the full filtered dataset so the primary histogram is exact, not sample-based.
                _df_for_chart = df_work
                _color_col = _color_by
                _color_domain = None
                if isinstance(_color_by, str) and _color_by and int(_topk_k or 0) > 0 and (_color_by in _df_for_chart.columns):
                    _df_for_chart, _color_col, _color_domain = _topk_domain_and_df(
                        _df_for_chart,
                        _color_by,
                        int(_topk_k),
                        str(_other_label),
                    )
                new_primary = _build_distribution_primary_hist_kde_v2(
                    _df_for_chart,
                    dist_col,
                    color_by=_color_col,
                    color_col=_color_by,
                    color_domain=_color_domain,
                    base_maxbins=_maxbins,
                    topk_k=0,
                    other_label=_other_label,
                )
                if new_primary is not None:
                    primary_chart = new_primary
                    try:
                        _primary_view_note = _distribution_view_note_text(_df_for_chart[dist_col])
                    except Exception:
                        _primary_view_note = None
                else:
                    # ✅ NOWE: Komunikat dla użytkownika gdy wykres nie może być zbudowany
                    _n_valid = int(pd.to_numeric(df_chart[dist_col], errors="coerce").notna().sum())
                    _n_total = len(df_chart)
                    st.warning(
                        f"⚠️ Nie można zbudować wykresu dla miary **{dist_col}**.\n\n"
                        f"Dane po filtrach: {_n_total:,} wierszy, z czego {_n_valid:,} ma wartość numeryczną.\n\n"
                        f"**Możliwe przyczyny:**\n"
                        f"- Zakres wartości wyklucza wszystkie obserwacje\n"
                        f"- Kolumna nie zawiera danych numerycznych\n"
                        f"- Filtry kategorii są zbyt restrykcyjne\n\n"
                        f"**Rozwiązanie:** Sprawdź filtry w panelu bocznym i rozszerz zakres."
                    )
            except Exception as e:
                st.error(f"❌ Błąd budowania wykresu: {type(e).__name__}: {e}")

            try:
                # ✅ alternatywy też na df_work (żeby zgadzały się z filtrem zakresu)
                alt_charts = _augment_distribution_alternatives(
                    df=df_chart,
                    col=dist_col,
                    alt_charts=alt_charts,
                )
            except Exception:
                pass

    chart_context: Dict[str, Any] = {
        "primary_kind": chart_meta.get("kind", "generic"),
        "has_primary": primary_chart is not None,
        "has_alt": bool(alt_charts),
        "has_scenario": bool(scenario),
        "has_ts": bool(ts_bundle),
        "has_clusters": bool(clusters_bundle),
    }

    # Render w tabach (zachowujemy 4-tab UI routera)
    tab_overview = tabs.get("overview") or st
    tab_insights = tabs.get("insights") or st
    # tab_segments = tabs.get("segments") or st
    # tab_whatif = tabs.get("whatif") or st

    # ✅ KLUCZOWA NAPRAWA: użyj chart_slot z kontekstu jeśli jest dostępny
    _ui_ctx = ctx.get("ui") or {}
    _overview_ui = _ui_ctx.get("overview") or {}
    _chart_slot = _overview_ui.get("chart_slot")
    
    # Jeśli mamy dedykowany slot - używamy go, w przeciwnym razie używamy tab_overview
    # ✅ Stabilizator rerenderu: czyścimy slot gdy zmieni się spec (miara/range/kolorowanie/Top-K/SAFE/filtry)
    _render_key = (
        str(dist_col),
        tuple((_filters.get("dist_range") or [])[:2]),
        str(_filters.get("color_col")),
        int(_filters.get("topk_k") or 0),
        bool((_filters.get("safe") or {}).get("enabled", False)),
        int((_filters.get("safe") or {}).get("sample_size", 0) or 0),
        str(_filters.get("filt_col")),
        tuple(_filters.get("filt_values") or []),
    )
    try:
        _last_key = st.session_state.get("dist__overview_chart_key")
        if _last_key != _render_key and _chart_slot is not None and hasattr(_chart_slot, "empty"):
            _chart_slot.empty()
        st.session_state["dist__overview_chart_key"] = _render_key
    except Exception:
        pass

    if _chart_slot is not None and hasattr(_chart_slot, "container"):
        _render_target = _chart_slot.container()
    else:
        _render_target = tab_overview

    with _render_target:
        if isinstance(primary_chart, alt.TopLevelMixin):
            altair_chart_stretch(st, primary_chart.properties(height=int(CHART_BLOCK_HEIGHT * 1.35)), width='stretch')

            if _primary_view_note:
                st.caption(f"ℹ️ {_primary_view_note}")

            # ✅ legenda/wyjaśnienie kolorów (jeśli kolorowanie aktywne)
            _filters = (ctx or {}).get("filters") or st.session_state.get("dc_filters") or {}
            _color_by = _filters.get("color_col")
            if _color_by in (None, "(brak)"):
                _color_by = None
            if isinstance(_color_by, str) and (_color_by in df.columns):
                st.caption(
                    f"ℹ️ Kolor słupków oznacza: **{_clean_label(_color_by)}** "
                    f"(wartość segmentu zobacz w tooltipie)."
                )


        else:
            st.info("Brak rekomendowanego wykresu dla tego pytania.")

    # ─────────────────────────────────────────────────────────────
    # Kluczowe insighty (kolejność + układ 1:1 jak referencja)
    # Tytuł (pytanie) → opis → wykres → Executive takeaway → guidance
    # ─────────────────────────────────────────────────────────────
    with tab_insights:
        # 1) Zbuduj mapę: label -> chart (bo alt_charts to lista krotek/dictów)
        _chart_by_label: Dict[str, alt.TopLevelMixin] = {}

        for item in (alt_charts or []):
            lbl = None
            ch = None

            if item is None:
                continue

            if isinstance(item, tuple) and len(item) == 2:
                lbl, ch = item
            elif isinstance(item, dict):
                lbl, ch = item.get("label"), item.get("chart")
            else:
                # czasem przychodzi sam chart
                ch = item

            if not isinstance(lbl, str) or not lbl.strip():
                continue
            if not isinstance(ch, alt.TopLevelMixin):
                continue

            _chart_by_label[lbl.strip()] = ch

        # 2) Specyfikacja bloków w kolejności jak referencja (Distribution_zakładka2_MA_BYĆ)
        _INSIGHT_BLOCKS = [
            {
                "label": "Boxplot – rozkład i outliery",
                "icon": "🧭",
                "title": "Gdzie jest typowy zakres danych?",
                "desc": "Ten widok pokazuje typowy zakres (IQR) oraz ogon/outliery — baza do oceny skali i ryzyka.",
                "llm_focus": "Zakotwicz w liczbach: mediana, IQR oraz % outlierów (fence). Druga linia: decyzja (progi / segmentacja / transformacja).",
                "exec": "Najważniejsza informacja to położenie mediany i szerokość IQR — to one opisują „typowy” poziom, a punkty poza fence’ami to potencjalne outliery do weryfikacji.",
                "guidance": [
                    ("✅ Sense", "Boxplot szybko pokazuje medianę, IQR i outliery w jednym ujęciu."),
                    ("💡 Interpretacja", "Duża liczba outlierów lub asymetria sugerują skośność i możliwą potrzebę segmentacji."),
                    ("🏅 Najlepsza praktyka", "Zawsze porównaj boxplot z histogramem i ECDF, żeby zrozumieć ogon rozkładu."),
                ],
            },
            {
                "label": "Violin – rozkład i gęstość (IQR + mediana + fence’y + outliery)",
                "icon": "🎻",
                "title": "Jak wygląda pełny rozkład i gdzie leżą obserwacje?",
                "desc": "Violin pokazuje kształt całego rozkładu, a Rug ujawnia dokładne położenie pojedynczych obserwacji.",
                "llm_focus": "Zakotwicz: mediana + p95/p99 lub tail_ratio; wskaż czy rozkład wygląda na mieszany. Druga linia: segmentuj lub normalizuj (np. log).",
                "exec": "Jeśli widzisz długi ogon lub skupiska, rozważ log-transformację / winsoryzację lub segmentację — violin + rug najszybciej to ujawniają.",
                "guidance": [
                    ("✅ Sense", "Violin pokazuje gęstość + medianę/IQR; Rug pokazuje rozkład pojedynczych obserwacji."),
                    ("💡 Interpretacja", "Długi ogon = rzadkie, wysokie wartości; skupiska mogą oznaczać różne pod-populacje."),
                    ("🏅 Najlepsza praktyka", "Zestaw violin/rug z ECDF i histogramem (różne biny), żeby potwierdzić wnioski."),
                ],
            },
            {
                "label": "Rug – położenie punktów",
                "icon": "📍",
                "title": "Gdzie leżą obserwacje?",
                "desc": "Rug ujawnia dokładne położenie pojedynczych obserwacji — idealne do oceny ogona i luk.",
                "llm_focus": "Zakotwicz: outliers_pct + p99 i opisz czy ogon to pojedyncze punkty czy ciągły rozkład. Druga linia: weryfikacja jakości / reguły obróbki outlierów.",
                "exec": "Rug pokazuje „ziarno” danych: czy ogon to pojedyncze punkty, czy ciągły rozkład — to zmienia decyzję o obróbce outlierów.",
                "guidance": [
                    ("✅ Sense", "Rug pokazuje rozkład pojedynczych obserwacji (idealne na ogon i luki)."),
                    ("💡 Interpretacja", "Jeśli outliery tworzą ciągły ogon — to raczej naturalna skośność niż błąd danych."),
                    ("🏅 Najlepsza praktyka", "Jeśli widać długi ogon lub skupiska, rozważ log-transformację, winsoryzację lub segmentację."),
                ],
            },
            {
                "label": "Dystrybuanta (ECDF)",
                "icon": "📈",
                "title": "Jakie jest ryzyko przekroczenia progów?",
                "desc": "Dystrybuanta ECDF pozwala ocenić, jaki odsetek danych mieści się poniżej/powyżej wybranego progu.",
                "llm_focus": "Zakotwicz: % obserwacji powyżej upper_fence i wartości p95/p99. Druga linia: rekomenduj progi (np. p95) i monitoring przekroczeń.",
                "exec": "To najlepszy wykres do pytań o percentyle i limity: od razu widzisz, jaki % obserwacji przekracza próg (np. fence).",
                "guidance": [
                    ("✅ Sense", "ECDF odpowiada wprost: jaki odsetek danych jest poniżej/powyżej progu."),
                    ("💡 Interpretacja", "Strome fragmenty = koncentracja wartości; płaskie = rzadki ogon."),
                    ("🏅 Najlepsza praktyka", "Używaj do decyzji progowych (limity, SLA, segmenty) - lepsze niż histogram przy skośności."),
                ],
            },
            {
                "label": "Histogram – skala logarytmiczna",
                "icon": "🧮",
                "title": "Jak często występują konkretne przedziały wartości?",
                "desc": "Histogram w skali log pomaga zobaczyć rzadkie, ekstremalne wartości bez „spłaszczania” ogona.",
                "llm_focus": "Zakotwicz: tail_ratio_p95_median oraz p95/p99. Druga linia: decyzja (log-transformacja / winsoryzacja / osobne zasady dla ogona).",
                "exec": "Jeśli ogon „znika” na zwykłym histogramie, log-hist pokazuje realną częstość rzadkich wartości i ułatwia ustawienie progów.",
                "guidance": [
                    ("✅ Sense", "Log-hist uwidacznia ogon i rzadkie wartości, których nie widać w zwykłej skali."),
                    ("💡 Interpretacja", "Jeśli po log-skali rozkład wygląda „normalniej”, log-transformacja może być dobrym krokiem."),
                    ("🏅 Najlepsza praktyka", "Używaj log-hist do diagnostyki (gdy ogon jest silny, rozważ log-transformację lub winsoryzację), a do komunikacji biznesowej zwykle wróć do skali liniowej."),
                ],
            },
            {
                "label": "Gęstość rozkładu (KDE)",
                "icon": "📉",
                "title": "Gęstość rozkładu (KDE)",
                "desc": "KDE pokazuje kształt rozkładu (piki i ogon) bez wrażliwości na dobór binów.",
                "llm_focus": "Zakotwicz: mediana + IQR i wskaż czy widać potencjalną multi-modalność (jeśli brak, napisz to). Druga linia: rekomendacja segmentacji lub ujednolicenia miary.",
                "exec": "KDE ułatwia wykrycie multi-modalności i dominującego zakresu — traktuj jako „kształt”, a nie liczniki.",
                "guidance": [
                    ("✅ Sense", "KDE pokazuje kształt rozkładu (gęstość) niezależnie od binów."),
                    ("💡 Interpretacja", "Wiele pików może oznaczać różne segmenty; długi ogon = skośność."),
                    ("🏅 Najlepsza praktyka", "Porównuj KDE z histogramem o różnych binach i z ECDF — razem dają pełen obraz."),
                ],
            },
            {
                "label": "Histogram – węższe przedziały",
                "icon": "📶",
                "title": "Histogram — węższe przedziały",
                "desc": "Więcej binów zwiększa rozdzielczość — lepiej widać lokalne piki i anomalie.",
                "llm_focus": "Zakotwicz: mediana + IQR oraz p95; wskaż ryzyko artefaktów (zaokrąglenia) lub segmentów. Druga linia: co sprawdzić (grupowanie po kategorii/źródle).",
                "exec": "Ten histogram służy do „mikro-struktury”: jeśli widać dodatkowe piki, to sygnał segmentów albo artefaktów danych.",
                "guidance": [
                    ("✅ Sense", "Węższe biny pokazują drobne struktury w rozkładzie, które giną przy szerokich binach."),
                    ("💡 Interpretacja", "Jeśli pojawiają się liczne „zęby”/piki, możliwe że mieszasz kilka segmentów lub jest to efekt zaokrągleń/artefaktów."),
                    ("🏅 Najlepsza praktyka", "Gdy histogram jest „poszarpany”, porównaj z KDE/ECDF i rozważ segmentację."),
                ],
            },
        ]

        # 3) Render w kolejności referencyjnej

        # ─────────────────────────────────────────────────────────────
        # Executive Takeaway (LLM) — cache + payload + fallback (NO-REGRESSION)
        # - nie dotyka UI (tylko podmienia exec_txt)
        # - działa nawet bez LLM (fallback)
        # - render-once: cache per block+stats signature
        # ─────────────────────────────────────────────────────────────
        import json, hashlib, re

        _exec_cache = st.session_state.setdefault("exec_takeaway_cache_v1", {})

        def _sha1(obj) -> str:
            s = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
            return hashlib.sha1(s.encode("utf-8")).hexdigest()

        def _fallback_exec(_stats: dict) -> str:
            s = _stats or {}
            out_pct = float(s.get("outliers_pct") or 0.0)
            skew = float(s.get("skewness") or 0.0)
            tail = float(s.get("tail_ratio_p95_median") or 0.0)

            # proste, stabilne reguły (możesz tuningować bez dotykania UI)
            if out_pct >= 3.0 and skew >= 1.0:
                return f"Widać prawy ogon i outliery (~{out_pct:.1f}%) — rozważ log-transformację/winsoryzację lub segmentację przed wnioskami."
            if out_pct >= 3.0:
                return f"Wartości odstające są istotne (~{out_pct:.1f}%) — ustaw progi (np. p95/p99) i zweryfikuj outliery vs segmenty."
            if skew >= 1.0 or tail >= 1.8:
                return "Rozkład jest wyraźnie prawoskośny (długi ogon) — średnia może mylić; raportuj medianę/IQR i pracuj na percentylach."
            return "Brak mocnych anomalii — typowy zakres wygląda stabilnie; jako domyślne KPI trzymaj medianę i IQR."

        _BANNED = re.compile(r"\b(ten wykres|histogram pokazuje|violin pokazuje|ecdf|rug pokazuje)\b", re.IGNORECASE)

        def _valid_llm_resp(d: dict) -> bool:
            if not isinstance(d, dict):
                return False
            txt = d.get("executive_takeaway")
            if not isinstance(txt, str):
                return False
            txt = txt.strip()
            if not txt:
                return False
            if len(txt) > 240:
                return False
            if _BANNED.search(txt):
                return False
            return True

        def _build_payload(blk: dict) -> dict:
            # Bezpieczne minimum: wszystko optional, brak kluczy = brak ryzyka
            _ui_question = (ctx.get("question") or ctx.get("user_question") or "").strip()
            _metric = (blk.get("metric") or blk.get("col") or ctx.get("metric") or ctx.get("value_col") or "").strip()

            return {
                "task": "exec_takeaway_v1",
                "lang": "pl",
                "style": {
                    "tone": "McKinsey/Bain",
                    "audience": "business",
                    "max_chars": 240,
                    "must_not_repeat_guidance": True,
                    "must_be_data_dependent": True,
                },
                "context": {
                    "branch": "distribution",
                    "block_id": (blk.get("id") or blk.get("label") or blk.get("title") or "").strip(),
                    "question": _ui_question,
                    "metric": {"name": _metric or None, "unit": None},
                    "chart": {"kind": (blk.get("kind") or blk.get("label") or "").strip(), "title": (blk.get("title") or "").strip()},
                },
                # jeśli masz stats per blok — wrzuć do blk["stats"] i samo zadziała
                "stats": (blk.get("stats") or {}),
                # guidance może być listą (tag, text) — tu mapujemy do dict
                "guidance": {
                    "sense": "",
                    "interpretation": "",
                    "best_practice": "",
                },
            }

        def _get_exec_takeaway(blk: dict) -> str:
            # 1) Klucz cache = branch + block_id + metric + stats signature + question
            payload = _build_payload(blk)

            # guidance list -> dict (jeśli jest)
            g_list = blk.get("guidance") or []
            if isinstance(g_list, list):
                for tag, txt in g_list:
                    t = (tag or "").lower()
                    if ("sens" in t) or ("sense" in t):
                        payload["guidance"]["sense"] = (txt or "").strip()
                    elif ("interpret" in t) or ("insight" in t):
                        payload["guidance"]["interpretation"] = (txt or "").strip()
                    elif "praktyk" in t:
                        payload["guidance"]["best_practice"] = (txt or "").strip()

            cache_key = _sha1({
                "v": 1,
                "branch": payload["context"]["branch"],
                "block_id": payload["context"]["block_id"],
                "metric": payload["context"]["metric"]["name"],
                "question": payload["context"]["question"],
                "stats_sig": payload.get("stats") or {},
            })

            if cache_key in _exec_cache:
                return (_exec_cache.get(cache_key) or "").strip()

            # 2) LLM opcjonalnie (TYLKO jeśli ctx ma callable: exec_takeaway_llm(payload)->dict)
            llm_fn = ctx.get("exec_takeaway_llm")
            if callable(llm_fn):
                try:
                    resp = llm_fn(payload)  # MUST return dict
                    if _valid_llm_resp(resp):
                        txt = (resp.get("executive_takeaway") or "").strip()
                        _exec_cache[cache_key] = txt
                        return txt
                except Exception:
                    pass  # NO-REGRESSION: lecimy fallback

            # 3) fallback deterministyczny
            txt = _fallback_exec(payload.get("stats") or {})
            _exec_cache[cache_key] = txt
            return txt

        # --- ExecTakeaway: stats + fallback keys (no-UI-change, no-regression) ---
        # (stats są wspólne dla Distribution — LLM później “różnicuje” po block.title + stats)
        try:
            _work_df = locals().get("work_df") or locals().get("_work_df") or df

            # "col" bywa różnie nazwany w gałęziach – bierzemy bezpieczny fallback
            _col = (
                locals().get("col")
                or locals().get("value_col")
                or locals().get("metric_col")
                or (ctx.get("value_col") if isinstance(ctx, dict) else None)
            )

            if (not isinstance(_col, str)) or (_col not in _work_df.columns):
                _num_cols = list(_work_df.select_dtypes(include="number").columns)
                _col = _num_cols[0] if _num_cols else None

            if not _col:
                raise ValueError("No numeric column for DIST_STATS")

            _s = _work_df[_col].dropna()
            _n = int(_s.shape[0])

            _q1 = float(_s.quantile(0.25)) if _n else None
            _med = float(_s.quantile(0.50)) if _n else None
            _q3 = float(_s.quantile(0.75)) if _n else None
            _p90 = float(_s.quantile(0.90)) if _n else None
            _p95 = float(_s.quantile(0.95)) if _n else None
            _p99 = float(_s.quantile(0.99)) if _n else None
            _mean = float(_s.mean()) if _n else None
            _std = float(_s.std(ddof=1)) if _n else None
            try:
                _skew = float(_s.skew()) if _n else None
            except Exception:
                _skew = None
            _iqr = float(_q3 - _q1) if (_q1 is not None and _q3 is not None) else None
            _lf = float(_q1 - 1.5 * _iqr) if (_iqr is not None and _q1 is not None) else None
            _uf = float(_q3 + 1.5 * _iqr) if (_iqr is not None and _q3 is not None) else None
            _out_n = int(((_s < _lf) | (_s > _uf)).sum()) if (_n and _lf is not None and _uf is not None) else 0

            _out_share = (float(_out_n) / float(_n) if _n else 0.0)
            _tail_ratio = (float(_p95) / float(_med) if (_p95 is not None and _med not in (None, 0.0)) else None)
            _DIST_STATS = {
                "col": str(_col),
                "n": _n,
                "q1": _q1,
                "median": _med,
                "q3": _q3,
                "p90": _p90,
                "p95": _p95,
                "p99": _p99,
                "mean": _mean,
                "std": _std,
                "skewness": _skew,
                "tail_ratio_p95_median": _tail_ratio,
                "iqr": _iqr,
                "lower_fence": _lf,
                "upper_fence": _uf,
                "outliers_n": _out_n,
                "outliers_share": _out_share,
                "outliers_pct": (100.0 * _out_share),
                "min": float(_s.min()) if _n else None,
                "max": float(_s.max()) if _n else None,
            }
        except Exception:
            _col_fallback = (
                locals().get("col")
                or locals().get("value_col")
                or locals().get("metric_col")
                or (ctx.get("value_col") if isinstance(ctx, dict) else None)
                or ""
            )
            _DIST_STATS = {"col": str(_col_fallback)}

        record_debug_checkpoint(
            "dist.render.stats_ready",
            dist_col=_DIST_STATS.get("col"),
            n=_DIST_STATS.get("n"),
            median=_DIST_STATS.get("median"),
            q1=_DIST_STATS.get("q1"),
            q3=_DIST_STATS.get("q3"),
            p95=_DIST_STATS.get("p95"),
            outliers_pct=_DIST_STATS.get("outliers_pct"),
            skewness=_DIST_STATS.get("skewness"),
        )

        def _build_log_hist_exec_stats(base_stats: dict) -> dict:
            s = dict(base_stats or {})
            try:
                _min_v = float(s.get("min")) if s.get("min") is not None else None
                _max_v = float(s.get("max")) if s.get("max") is not None else None
                _p95_v = float(s.get("p95")) if s.get("p95") is not None else None
                _p99_v = float(s.get("p99")) if s.get("p99") is not None else None
                _positive_min = _min_v if (_min_v is not None and _min_v > 0) else None
                _log_span = None
                if (_positive_min is not None) and (_max_v is not None) and (_max_v > 0):
                    _log_span = round(float(math.log10(_max_v) - math.log10(_positive_min)), 2)
                _p99_p95 = None
                if (_p95_v is not None) and (_p95_v > 0) and (_p99_v is not None):
                    _p99_p95 = round(float(_p99_v / _p95_v), 2)
                s.update(
                    {
                        "positive_min": _positive_min,
                        "log10_span": _log_span,
                        "tail_ratio_p99_p95": _p99_p95,
                    }
                )
            except Exception:
                pass
            return s

        def _build_kde_exec_stats(base_stats: dict) -> dict:
            s = dict(base_stats or {})
            try:
                _q1_v = float(s.get("q1")) if s.get("q1") is not None else None
                _q3_v = float(s.get("q3")) if s.get("q3") is not None else None
                _med_v = float(s.get("median")) if s.get("median") is not None else None
                _p95_v = float(s.get("p95")) if s.get("p95") is not None else None
                _q3_p95_gap = None
                if (_q3_v is not None) and (_p95_v is not None):
                    _q3_p95_gap = round(float(_p95_v - _q3_v), 2)
                s.update(
                    {
                        "dominant_range_lo": _q1_v,
                        "dominant_range_hi": _q3_v,
                        "central_median": _med_v,
                        "q3_p95_gap": _q3_p95_gap,
                    }
                )
            except Exception:
                pass
            return s

        def _repair_exec_takeaway(block_label: str, stats: dict) -> str:
            s = stats or {}
            _median = s.get("median")
            _q1 = s.get("q1")
            _q3 = s.get("q3")
            _p95 = s.get("p95")
            _max_v = s.get("max")
            _tail = s.get("tail_ratio_p95_median")
            _label = str(block_label or "").strip()
            if _label == "Histogram – skala logarytmiczna":
                _tail_txt = f"{float(_tail):.1f}x" if _tail not in (None, 0) else "wysoki"
                return (
                    f"Mediana {_fmt_num(_median)} i p95 {_fmt_num(_p95)} dają tail ratio ~{_tail_txt}, "
                    f"a maksimum {_fmt_num(_max_v)} potwierdza długi ogon. "
                    "Rekomendacja: do diagnostyki ogona używaj skali log i progów p95/p99, a KPI raportuj medianą/IQR."
                )
            if _label == "Gęstość rozkładu (KDE)":
                return (
                    f"Typowy zakres {_fmt_num(_q1)}–{_fmt_num(_q3)} skupia dane wokół mediany {_fmt_num(_median)}, "
                    f"a p95 {_fmt_num(_p95)} wskazuje na wydłużony ogon. "
                    "Rekomendacja: traktuj KDE jako test kształtu i potwierdzaj segmentację dopiero po rozbiciu na grupy."
                )
            return ""

        _DIST_LOG_HIST_STATS = _build_log_hist_exec_stats(_DIST_STATS)
        _DIST_KDE_STATS = _build_kde_exec_stats(_DIST_STATS)
        record_debug_checkpoint(
            "dist.exec.block_specific_stats",
            log_hist_keys=list(_DIST_LOG_HIST_STATS.keys()),
            kde_keys=list(_DIST_KDE_STATS.keys()),
        )

        for _b in _INSIGHT_BLOCKS:
            _label = str(_b.get("label") or "").strip()
            if _label == "Histogram – skala logarytmiczna":
                _b["stats"] = dict(_DIST_LOG_HIST_STATS)
            elif _label == "Gęstość rozkładu (KDE)":
                _b["stats"] = dict(_DIST_KDE_STATS)
            else:
                _b.setdefault("stats", _DIST_STATS)
            _b.setdefault("intent", "distribution.key_insight")
            # naprawa fallback keys: nowe -> stare
            if "exec_fallback" not in _b and ("exec" in _b):
                _b["exec_fallback"] = _b.get("exec") or ""
        
        # --- /ExecTakeaway ---
        from data_chat_core.exec_takeaway_llm import get_exec_takeaways_llm

        _dc_llm_text = st.session_state.get("dc_llm_text")
        if callable(_dc_llm_text) and not callable(ctx.get("llm_text")):
            ctx["llm_text"] = _dc_llm_text
        _dc_llm_status = st.session_state.get("dc_llm_status_v1") or {}
        if isinstance(_dc_llm_status, dict) and not ctx.get("openai_model"):
            _model = str(_dc_llm_status.get("model") or "").strip()
            if _model:
                ctx["openai_model"] = _model

        record_debug_checkpoint(
            "dist.exec.start",
            llm_text_ready=bool(callable(ctx.get("llm_text"))),
            openai_model=ctx.get("openai_model"),
            blocks_requested=len(_INSIGHT_BLOCKS),
        )

        _exec_by_label: Dict[str, str] = {}
        if not callable(ctx.get("llm_text")):
            record_debug_checkpoint(
                "dist.exec.skipped",
                reason="llm_text_missing_in_ctx",
                llm_status=st.session_state.get("dc_llm_status_v1"),
            )
        else:
            _exec_by_label = get_exec_takeaways_llm(
                ctx=ctx,
                intent="distribution_key_insight",
                blocks=_INSIGHT_BLOCKS,
            ) or {}
            if not _exec_by_label:
                record_debug_checkpoint(
                    "dist.exec.empty_result",
                    blocks_requested=len(_INSIGHT_BLOCKS),
                    llm_status=st.session_state.get("dc_llm_status_v1"),
                    helper_debug=ctx.get("_exec_takeaway_llm_last_error"),
                )

        for b in _INSIGHT_BLOCKS:
            lbl = b.get("label")
            llm_exec = (_exec_by_label or {}).get(lbl, "").strip()
            if llm_exec:
                b["exec"] = llm_exec
                b["_exec_source"] = "llm"
            else:
                repaired_exec = _repair_exec_takeaway(str(lbl or ""), b.get("stats") or {})
                if repaired_exec:
                    b["exec"] = repaired_exec
                    b["_exec_source"] = "llm_repaired"
                else:
                    b["_exec_source"] = "fallback"

        _llm_count = 0
        _det_count = 0
        for b in _INSIGHT_BLOCKS:
            _label = str(b.get("label") or "")
            _block_id = str(b.get("id") or _label).strip()
            _source_kind = str(b.get("_exec_source") or "").strip()
            if _source_kind == "llm":
                _src = "llm_batch_direct"
            elif _source_kind == "llm_repaired":
                _src = "llm_batch_direct_repaired"
            else:
                _src = "deterministic_fallback"
            _txt = str((b.get("exec") or b.get("exec_fallback") or "")).strip()
            if _src in ("llm_batch_direct", "llm_batch_direct_repaired"):
                _llm_count += 1
            else:
                _det_count += 1
            record_debug_checkpoint(
                "dist.exec.block",
                block_id=_block_id,
                src=_src,
                final_len=len(_txt),
            )

        _blocks_count = int(len(_INSIGHT_BLOCKS))
        _summary = {
            "blocks_count": _blocks_count,
            "llm_count": int(_llm_count),
            "det_count": int(_det_count),
            "llm_share_pct": round(float(_llm_count) / float(_blocks_count), 3) if _blocks_count else 0.0,
        }
        st.session_state["dist_exec_summary_v1"] = _summary
        record_debug_checkpoint("dist.exec.summary", **_summary)

        for blk in _INSIGHT_BLOCKS:

            ch = _chart_by_label.get(blk["label"])
            if not isinstance(ch, alt.TopLevelMixin):
                continue

            _icon = (blk.get("icon") or "").strip()
            _title = blk.get("title") or ""
            _h = f"{_icon} {_title}".strip()

            st.markdown(f"### {_h}")
            if blk.get("desc"):
                st.caption(blk["desc"])

            altair_chart_stretch(st, ch, width='stretch')

            # ✅ Executive takeaway ma być POMIĘDZY wykresem a guidance (ciasny spacing)
            exec_txt, _exec_meta = get_exec_takeaway(
                intent=(blk.get("intent") or "distribution.key_insight"),
                block=blk,
                stats=(blk.get("stats") or {}),
                question=((ctx.get("question") if isinstance(ctx, dict) else "") or ""),
                session_state=st.session_state,
                llm_fn=None,  # bundle LLM already injected into block["exec"]
                  # <- LLM ON (JSON-validated) + safe logging (Langfuse jeśli dostępne)
            )

            exec_txt = (exec_txt or "").strip()
            if exec_txt:
                render_exec_takeaway(exec_txt)

            # Guidance (Sense / Interpretacja / Najlepsza praktyka) — unified UI contract
            _sens = _interp = _best = ""
            for tag, gtxt in (blk.get("guidance") or []):
                t_raw = (tag or "").strip().lower()
                # Handle labels like "✅ Sens:", "🔎 Interpretacja:" etc.
                t = re.sub(r"[^a-ząćęłńóśźż ]+", " ", t_raw).strip()
                g = (gtxt or "").strip()
                if not g:
                    continue
                if "sens" in t:
                    _sens = g
                elif "interpret" in t:
                    _interp = g
                elif ("praktyk" in t) or t.startswith("naj") or ("najleps" in t):
                    _best = g
            render_guidance(sens=_sens, interp=_interp, best=_best)

            st.divider()



    # Global debug panel (exec takeaway)
    try:
        if bool(st.session_state.get("dist_debug_enabled", False) or st.session_state.get("dc_debug", False)):
            from data_chat_core.exec_takeaway_debug import render_exec_takeaway_debug_panel
            render_exec_takeaway_debug_panel(expanded=False)
    except Exception:
        pass

    return {
        "chart_meta": chart_meta,
        "chart_context": chart_context,
    }
