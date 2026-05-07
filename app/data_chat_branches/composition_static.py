from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import json
import hashlib
import math
import re
import time
import numpy as np
import pandas as pd
try:
    import plotly.express as px  # type: ignore
except Exception:
    px = None
try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None
import streamlit as st
from core.ui_safe import altair_chart_stretch, plotly_chart_stretch
from data_chat_core.ui_contract import render_exec_takeaway, render_guidance
from data_chat_core.exec_takeaway import get_exec_takeaway
from data_chat_core.exec_takeaway_llm import get_exec_takeaways_llm

import altair as alt

from ._chart_bundle import _infer_numeric_cols

CHART_BLOCK_HEIGHT = 360

# =========================
# Helpers: schema + typing
# =========================

def _sanitize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure unique, string column names (prevents Pandas 'Grouper not 1-dimensional' from dup cols)."""
    if df is None or df.empty:
        return df
    cols = [str(c) for c in df.columns]
    if len(cols) != len(set(cols)):
        # make unique with suffixes
        seen: Dict[str, int] = {}
        new_cols = []
        for c in cols:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}__dup{seen[c]}")
        df = df.copy()
        df.columns = new_cols
    else:
        # ensure str
        if any(not isinstance(c, str) for c in df.columns):
            df = df.copy()
            df.columns = cols
    return df


def _is_numeric_continuous_legacy(s: pd.Series, max_low_card: int = 30) -> bool:
    """Numeric but *continuous-like* (high cardinality)."""
    if s is None or (not pd.api.types.is_numeric_dtype(s)) or pd.api.types.is_bool_dtype(s):
        return False
    nuniq = int(s.nunique(dropna=True))
    return nuniq > max_low_card


def _infer_value_col(df: pd.DataFrame) -> str | None:
    """Heurystyka: preferuj realna wartosc sprzedaży, nie cene jednostkowa."""
    candidates = get_value_candidate_columns(df)
    if candidates:
        return candidates[0]
    num_cols = _infer_numeric_cols(df)
    return num_cols[0] if num_cols else None


def _infer_best_group_col(df: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
    """Fallback heurystyka: wybierz sensowną kolumnę kategoryczną (nie time-like, nie ID)."""
    exclude = exclude or set()
    preferred = ["sector", "segment", "category", "kategoria", "mszoning", "type", "brand", "subclass", "class"]
    obj_cols = [c for c in df.columns if c not in exclude and str(df[c].dtype) in ("object", "category", "bool")]
    # add low-card ints as categories
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_integer_dtype(df[c]) and 2 <= df[c].nunique(dropna=True) <= 30:
            obj_cols.append(c)

    def _time_like(col: str) -> bool:
        cl = col.lower()
        return any(tok in cl for tok in ("date", "data", "time", "rok", "mies", "month", "year"))

    def _id_like(col: str) -> bool:
        cl = col.lower()
        return any(tok in cl for tok in ("id", "uuid", "guid", "hash", "index"))

    # preferred first
    for c in obj_cols:
        if _time_like(c) or _id_like(c):
            continue
        if any(p in c.lower() for p in preferred):
            n = int(df[c].nunique(dropna=True))
            if 2 <= n <= 60:
                return c
    # then any reasonable
    for c in obj_cols:
        if _time_like(c) or _id_like(c):
            continue
        n = int(df[c].nunique(dropna=True))
        if 2 <= n <= 60:
            return c
    return None


def _bin_numeric(series: pd.Series, max_bins: int = 30, method: str = "quantile") -> Tuple[pd.Series, Dict[str, Any]]:
    """Create bins for numeric series; returns categorical labels and meta.

    method:
      - 'quantile' (default): equal-frequency bins (can have uneven numeric widths)
      - 'fixed'   : equal-width bins with an automatic 'nice' step (best for prices)
      - 'cut'     : equal-width bins with pandas default bin edges
    """
    s = pd.to_numeric(series, errors="coerce")
    s_non = s.dropna()
    meta: Dict[str, Any] = {"n": int(len(s_non)), "method": method, "bins": None, "step": None, "clip": None}

    if s_non.empty:
        return pd.Series(["(brak)"] * len(series), index=series.index), meta

    nuniq = int(s_non.nunique())
    if nuniq <= max_bins:
        # treat as discrete buckets
        labels = s.round(6).astype("Int64", errors="ignore").astype(str)
        meta["bins"] = nuniq
        return labels, meta

    # target bins ~ 15–25 (then capped by max_bins)
    target_bins = int(min(max_bins, max(12, min(24, int(round(math.sqrt(len(s_non))))))))

    def _nice_step(raw_step: float) -> float:
        """Snap raw step to 1/2/5 * 10^k."""
        if not math.isfinite(raw_step) or raw_step <= 0:
            return 1.0
        k = 10 ** math.floor(math.log10(raw_step))
        m = raw_step / k
        if m <= 1:
            nice = 1
        elif m <= 2:
            nice = 2
        elif m <= 5:
            nice = 5
        else:
            nice = 10
        return float(nice * k)

    try:
        if method == "fixed":
            # robust clip for extreme outliers (keeps bins usable across datasets)
            lo = float(s_non.quantile(0.01))
            hi = float(s_non.quantile(0.99))
            mn = float(s_non.min())
            mx = float(s_non.max())
            use_lo, use_hi = mn, mx
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                # if the 1–99% span is much tighter than full range, clip
                if (hi - lo) < 0.85 * (mx - mn):
                    use_lo, use_hi = lo, hi
                    meta["clip"] = {"q01": lo, "q99": hi}

            rng = max(1e-12, use_hi - use_lo)
            step = _nice_step(rng / target_bins)

            # if too many bins, inflate step
            def _n_bins(_step: float) -> int:
                return int(math.ceil((use_hi - use_lo) / _step))

            while _n_bins(step) > max_bins:
                step = _nice_step(step * 1.25)

            start = math.floor(use_lo / step) * step
            end = math.ceil(use_hi / step) * step
            # ensure at least 2 edges
            if end <= start:
                end = start + step

            edges = np.arange(start, end + step * 0.999999, step)
            binned = pd.cut(s, bins=edges, include_lowest=True)
            labels = binned.astype(str)
            meta["bins"] = int(labels.nunique(dropna=True))
            meta["step"] = step
            return labels, meta

        elif method == "quantile":
            binned = pd.qcut(s, q=target_bins, duplicates="drop")
        else:
            binned = pd.cut(s, bins=target_bins)

        labels = binned.astype(str)
        meta["bins"] = int(labels.nunique(dropna=True))
        return labels, meta

    except Exception:
        # robust fallback
        binned = pd.cut(s, bins=max_bins)
        labels = binned.astype(str)
        meta["bins"] = int(labels.nunique(dropna=True))
        return labels, meta


def _coerce_numeric_series(s: pd.Series | None) -> pd.Series:
    """Best-effort numeric coercion for true numerics and numeric-like strings."""
    if s is None:
        return pd.Series(dtype="float64")
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    txt = s.astype(str).str.strip()
    txt = txt.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    both = txt.str.contains(",", regex=False) & txt.str.contains(".", regex=False)
    txt = txt.where(~both, txt.str.replace(",", "", regex=False))
    txt = txt.where(both, txt.str.replace(",", ".", regex=False))
    return pd.to_numeric(txt, errors="coerce")


def _looks_numeric_like(s: pd.Series | None, min_share: float = 0.90) -> bool:
    if s is None or pd.api.types.is_bool_dtype(s):
        return False
    if pd.api.types.is_numeric_dtype(s):
        return True
    non_null = s.dropna()
    if non_null.empty:
        return False
    parsed = _coerce_numeric_series(non_null)
    return float(parsed.notna().mean() or 0.0) >= float(min_share)


def _is_numeric_continuous(s: pd.Series, max_low_card: int = 30) -> bool:
    """Numeric or numeric-like and high-cardinality enough to behave like a measure."""
    if s is None or pd.api.types.is_bool_dtype(s) or not _looks_numeric_like(s):
        return False
    nuniq = int(_coerce_numeric_series(s).dropna().nunique())
    return nuniq > max_low_card


def _is_time_like(col: str) -> bool:
    cl = str(col).lower()
    return any(tok in cl for tok in (
        "date", "data", "time", "rok", "mies", "month", "year",
        "quarter", "week", "day", "hour", "godz", "dzien",
    ))


def _is_id_like(col: str) -> bool:
    cl = str(col).lower()
    return any(tok in cl for tok in (
        " id", "id_", "_id", "uuid", "guid", "hash", "index",
        "invoice", "customer", "client", "account", "stockcode",
        "sku", "ean", "barcode", "kod", "code",
    ))


def _is_generated_helper_col(col: str) -> bool:
    cl = str(col).lower()
    return (
        cl.startswith("__")
        or cl.startswith("is_outlier_")
        or cl.startswith("outlier_")
        or cl.endswith("_outlier")
        or "helper" in cl
    )


def _score_group_col(name: str) -> int:
    norm = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    score = 0
    for kw, weight in [
        ("category", 140), ("kategoria", 140), ("segment", 130),
        ("subcategory", 115), ("subcat", 110), ("podkategoria", 110),
        ("sector", 100), ("brand", 90), ("marka", 90),
        ("channel", 85), ("country", 85), ("kraj", 85), ("region", 80),
        ("market", 75), ("type", 70), ("class", 65), ("subclass", 65),
        ("department", 60), ("family", 55),
    ]:
        if kw in norm:
            score += weight
    if _is_generated_helper_col(name):
        score -= 300
    if _is_time_like(name):
        score -= 220
    if _is_id_like(name):
        score -= 180
    return score


def _cs_schema_cache_key(df: pd.DataFrame, *, tag: str, exclude: set[str] | None = None) -> str:
    source_info = st.session_state.get("datachat_source_info") or {}
    payload = {
        "tag": tag,
        "rows": int(len(df) if isinstance(df, pd.DataFrame) else 0),
        "cols": [str(c) for c in (df.columns.tolist() if isinstance(df, pd.DataFrame) else [])],
        "dtypes": [str(x) for x in (df.dtypes.tolist() if isinstance(df, pd.DataFrame) else [])],
        "exclude": sorted(str(x) for x in (exclude or set())),
        "source_path": str(source_info.get("path") or ""),
        "source_mtime": source_info.get("mtime"),
    }
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _cs_schema_series_sample(s: pd.Series, sample_n: int = 50_000) -> pd.Series:
    try:
        if s is None:
            return pd.Series(dtype="object")
        return s.dropna().head(sample_n)
    except Exception:
        return pd.Series(dtype="object")


def get_groupable_columns(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
    max_unique: int = 120,
) -> list[str]:
    """Return sidebar-safe grouping columns for Composition Static."""
    if df is None or df.empty:
        return []

    exclude = exclude or set()
    cache = st.session_state.get("cs_schema_candidates_cache_v1")
    if not isinstance(cache, dict):
        cache = {}
    cache_key = _cs_schema_cache_key(df, tag=f"groupable:{int(max_unique)}", exclude=exclude)
    cached = cache.get(cache_key)
    if isinstance(cached, list):
        return [str(x) for x in cached]

    cols: list[str] = []
    nuniq_map: Dict[str, int] = {}
    for col in df.columns:
        if col in exclude:
            continue
        if _is_generated_helper_col(col) or _is_time_like(col) or _is_id_like(col):
            continue

        s = df[col]
        sample = _cs_schema_series_sample(s)
        if sample.empty:
            continue
        nuniq = int(sample.nunique(dropna=True))
        nuniq_map[str(col)] = nuniq
        if nuniq < 2 or nuniq > max_unique:
            continue

        if pd.api.types.is_bool_dtype(s):
            cols.append(col)
            continue

        if _looks_numeric_like(sample):
            if _is_numeric_continuous(sample):
                continue
            if nuniq <= 30:
                cols.append(col)
            continue

        if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or str(s.dtype) == "string":
            cols.append(col)

    result = sorted(
        list(dict.fromkeys(cols)),
        key=lambda c: (-_score_group_col(c), int(nuniq_map.get(str(c), 999999)), str(c).lower()),
    )
    cache[cache_key] = list(result)
    if len(cache) > 16:
        for _old_key in list(cache.keys())[:-16]:
            cache.pop(_old_key, None)
    st.session_state["cs_schema_candidates_cache_v1"] = cache
    return result


def get_price_candidate_columns(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> list[str]:
    if df is None or df.empty:
        return []

    exclude = exclude or set()
    cache = st.session_state.get("cs_schema_candidates_cache_v1")
    if not isinstance(cache, dict):
        cache = {}
    cache_key = _cs_schema_cache_key(df, tag="price_candidates", exclude=exclude)
    cached = cache.get(cache_key)
    if isinstance(cached, list):
        return [str(x) for x in cached]

    strong_scored: list[tuple[int, int, str]] = []
    weak_scored: list[tuple[int, int, str]] = []
    for col in df.columns:
        if col in exclude or _is_generated_helper_col(col) or _is_time_like(col) or _is_id_like(col):
            continue

        cl = str(col).lower()
        if any(tok in cl for tok in (
            "quantity", "qty", "count", "units", "unit_count", "pieces",
            "items", "invoice", "customer", "client", "stock", "day",
            "week", "month", "year", "hour", "rank", "share", "pct",
            "percent", "index",
        )):
            continue

        s = df[col]
        sample = _cs_schema_series_sample(s)
        if sample.empty or pd.api.types.is_bool_dtype(s) or not _looks_numeric_like(sample):
            continue
        nuniq = int(_coerce_numeric_series(sample).dropna().nunique())
        if nuniq < 5:
            continue

        base_score = 25 if _is_numeric_continuous(sample) else 0
        strong_score = base_score
        weak_score = base_score

        for hint, weight in [
            ("unit price", 180), ("unitprice", 180), ("unit_price", 180),
            ("sale price", 175), ("saleprice", 175), ("salesprice", 175),
            ("sellprice", 170), ("net price", 170), ("net_price", 170),
            ("gross price", 170), ("gross_price", 170), ("price", 165),
            ("cost price", 150), ("cost_price", 150),
        ]:
            if hint in cl:
                strong_score += weight

        for hint, weight in [
            ("amount", 90), ("value", 85), ("revenue", 80),
            ("cost", 75), ("margin", 60),
        ]:
            if hint in cl:
                weak_score += weight

        if strong_score > base_score:
            strong_scored.append((strong_score, nuniq, col))
        elif weak_score > base_score:
            weak_scored.append((weak_score, nuniq, col))

    if strong_scored:
        strong_scored.sort(key=lambda row: (-row[0], -row[1], str(row[2]).lower()))
        result = [col for _, _, col in strong_scored]
        cache[cache_key] = list(result)
        if len(cache) > 16:
            for _old_key in list(cache.keys())[:-16]:
                cache.pop(_old_key, None)
        st.session_state["cs_schema_candidates_cache_v1"] = cache
        return result

    weak_scored.sort(key=lambda row: (-row[0], -row[1], str(row[2]).lower()))
    result = [col for _, _, col in weak_scored]
    cache[cache_key] = list(result)
    if len(cache) > 16:
        for _old_key in list(cache.keys())[:-16]:
            cache.pop(_old_key, None)
    st.session_state["cs_schema_candidates_cache_v1"] = cache
    return result


def get_value_candidate_columns(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> list[str]:
    if df is None or df.empty:
        return []

    exclude = exclude or set()
    cache = st.session_state.get("cs_schema_candidates_cache_v1")
    if not isinstance(cache, dict):
        cache = {}
    cache_key = _cs_schema_cache_key(df, tag="value_candidates", exclude=exclude)
    cached = cache.get(cache_key)
    if isinstance(cached, list):
        return [str(x) for x in cached]

    scored: list[tuple[int, int, str]] = []
    for col in df.columns:
        if col in exclude or _is_generated_helper_col(col) or _is_time_like(col) or _is_id_like(col):
            continue

        s = df[col]
        sample = _cs_schema_series_sample(s)
        if sample.empty or pd.api.types.is_bool_dtype(s) or not _looks_numeric_like(sample):
            continue

        nuniq = int(_coerce_numeric_series(sample).dropna().nunique())
        if nuniq < 2:
            continue

        cl = str(col).lower()
        score = 20 if _is_numeric_continuous(sample) else 0

        for hint, weight in [
            ("line_value", 260), ("linevalue", 260),
            ("total_value", 240), ("totalvalue", 240),
            ("sales_value", 235), ("sale_value", 235),
            ("revenue", 230), ("gmv", 220),
            ("amount", 210), ("value", 180),
            ("sales", 160), ("sale", 155),
            ("net_value", 150), ("gross_value", 150),
        ]:
            if hint in cl:
                score += weight

        for hint, penalty in [
            ("unit price", 220), ("unitprice", 220), ("unit_price", 220),
            ("saleprice", 210), ("salesprice", 210),
            ("price", 180),
            ("quantity", 160), ("qty", 160), ("count", 160), ("units", 160),
        ]:
            if hint in cl:
                score -= penalty

        if score > 0:
            scored.append((score, nuniq, col))

    scored.sort(key=lambda row: (-row[0], -row[1], str(row[2]).lower()))
    result = [col for _, _, col in scored]
    cache[cache_key] = list(result)
    if len(cache) > 16:
        for _old_key in list(cache.keys())[:-16]:
            cache.pop(_old_key, None)
    st.session_state["cs_schema_candidates_cache_v1"] = cache
    return result


def _infer_best_group_col(df: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
    groupable = get_groupable_columns(df, exclude=exclude)
    return groupable[0] if groupable else None


def resolve_grouping_selection(
    df: pd.DataFrame,
    group_col_sel: str | None = None,
    group_col2_sel: str | None = None,
    price_col_sel: str | None = None,
) -> Dict[str, Any]:
    """Resolve the effective category and price columns used by charts and interpretation."""
    groupable_cols = get_groupable_columns(df)
    price_candidates = get_price_candidate_columns(df)

    selected_group = group_col_sel if isinstance(group_col_sel, str) and group_col_sel in df.columns else None
    selected_group2 = group_col2_sel if isinstance(group_col2_sel, str) and group_col2_sel in df.columns else None
    selected_price = (
        price_col_sel
        if isinstance(price_col_sel, str) and price_col_sel not in ("", "(auto)") and price_col_sel in df.columns
        else None
    )

    price_mode = False
    price_source = "explicit" if selected_price in price_candidates else "auto"
    price_col = selected_price if selected_price in price_candidates else None
    if price_col is None and price_candidates:
        price_col = price_candidates[0]

    if selected_group and selected_group in price_candidates and selected_group not in groupable_cols:
        price_mode = True
        price_col = price_col or selected_group
        if price_col == selected_group:
            price_source = "group_auto"
        selected_group = None

    if selected_group not in groupable_cols:
        selected_group = None

    group_col = selected_group or _infer_best_group_col(df, exclude={price_col} if price_col else None)
    groupable_group2 = get_groupable_columns(df, exclude={group_col} if group_col else None)
    group_col2 = selected_group2 if selected_group2 in groupable_group2 else None
    if group_col2 is None and selected_group2 and selected_group2 != group_col:
        try:
            sample = _cs_schema_series_sample(df[selected_group2])
            nuniq = int(sample.nunique(dropna=True))
            is_bad_numeric = pd.api.types.is_numeric_dtype(df[selected_group2]) and _is_numeric_continuous(sample)
            if (
                2 <= nuniq <= 2_000
                and not is_bad_numeric
                and not _is_generated_helper_col(selected_group2)
                and not _is_time_like(selected_group2)
                and not _is_id_like(selected_group2)
            ):
                group_col2 = selected_group2
        except Exception:
            group_col2 = None

    return {
        "group_col": group_col,
        "group_col2": group_col2,
        "price_col": price_col,
        "price_mode": price_mode,
        "price_source": price_source if price_col else "none",
        "groupable_cols": groupable_cols,
        "groupable_group2_cols": groupable_group2,
        "price_candidates": price_candidates,
    }


def is_cs_debug_enabled() -> bool:
    return bool(st.session_state.get("cs_debug_enabled", False))


def is_cs_debug_fallbacks_enabled() -> bool:
    return bool(st.session_state.get("cs_debug_show_fallbacks", False))


def _cs_checkpoints_enabled() -> bool:
    return bool(
        st.session_state.get("cs_internal_checkpoints_enabled", True)
        or st.session_state.get("cs_debug_checkpoints", False)
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
    if not _cs_checkpoints_enabled():
        return
    entry: Dict[str, Any] = {"where": where}
    for key, value in payload.items():
        entry[str(key)] = _compact_debug_value(value)
    log = st.session_state.setdefault("cs_debug_log", [])
    log.append(entry)
    if len(log) > 250:
        del log[:-250]


def get_debug_checkpoints() -> list[dict[str, Any]]:
    log = st.session_state.get("cs_debug_log")
    return list(log) if isinstance(log, list) else []


def clear_debug_checkpoints() -> None:
    st.session_state["cs_debug_log"] = []


def build_stats_payload(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group_col2: str | None = None,
    top_n: int = 10,
    price_col: str | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "group_col": group_col,
        "value_col": value_col,
        "group_col2": group_col2,
        "top_n": int(top_n or 10),
        "total_rows": int(len(df) if isinstance(df, pd.DataFrame) else 0),
        "price_col": price_col,
    }

    if not isinstance(df, pd.DataFrame) or df.empty:
        payload["error"] = "Brak danych do analizy struktury."
        return payload

    if not group_col or group_col not in df.columns or not value_col or value_col not in df.columns:
        payload["error"] = "Brak kolumny grupowej lub wartościowej."
        return payload

    try:
        tmp = pd.DataFrame({
            "__group": df[group_col],
            "__value": _coerce_numeric_series(df[value_col]).fillna(0.0).clip(lower=0.0),
        })
        agg = (
            tmp.dropna(subset=["__group"])
            .groupby("__group", dropna=False, sort=False)["__value"]
            .sum()
            .reset_index()
            .rename(columns={"__group": "group", "__value": "value"})
        )
        agg["value"] = pd.to_numeric(agg["value"], errors="coerce").fillna(0.0).clip(lower=0.0)
        agg = agg[agg["value"] > 0].sort_values("value", ascending=False).reset_index(drop=True)

        total_value = float(agg["value"].sum())
        n_groups = int(len(agg))
        if total_value <= 0 or n_groups == 0:
            payload["error"] = "Brak wartości po agregacji dla wybranego układu kolumn."
            return payload

        agg["share"] = agg["value"] / total_value
        topn_n = int(top_n or 10)
        agg_topn = agg.head(topn_n)
        top_labels = [str(g) for g in agg_topn["group"]]
        top_values = [round(float(v), 2) for v in agg_topn["value"]]
        top_shares = [round(float(s) * 100.0, 2) for s in agg_topn["share"]]
        other_value = max(0.0, float(total_value - float(agg_topn["value"].sum())))
        other_share_pct = round((other_value / total_value) * 100.0, 2) if total_value > 0 else 0.0
        display_n = min(n_groups, max(25, topn_n))
        agg_display = agg.head(display_n)
        display_labels = [str(g) for g in agg_display["group"]]
        display_values = [round(float(v), 2) for v in agg_display["value"]]
        display_shares = [round(float(s) * 100.0, 2) for s in agg_display["share"]]
        display_other_value = max(0.0, float(total_value - float(agg_display["value"].sum())))
        display_other_share_pct = round((display_other_value / total_value) * 100.0, 2) if total_value > 0 else 0.0
        cum = agg["share"].cumsum()
        pareto_80_n = int((cum >= 0.80).idxmax()) + 1 if (cum >= 0.80).any() else n_groups
        hhi = float((agg["share"] ** 2).sum() * 10000.0)

        payload.update({
            "n_groups": n_groups,
            "total_value": round(total_value, 2),
            "top_labels": top_labels,
            "top_values": top_values,
            "top_shares_pct": top_shares,
            "other_value": round(other_value, 2),
            "other_share_pct": other_share_pct,
            "display_labels": display_labels,
            "display_values": display_values,
            "display_shares_pct": display_shares,
            "display_other_value": round(display_other_value, 2),
            "display_other_share_pct": display_other_share_pct,
            "top1_share_pct": round(top_shares[0], 2) if top_shares else 0.0,
            "pareto_80_n": pareto_80_n,
            "hhi": round(hhi, 1),
            "hhi_class": "rozproszona" if hhi < 1500 else ("umiarkowana" if hhi < 2500 else "skoncentrowana"),
        })

        if group_col2 and group_col2 in df.columns and group_col2 != group_col:
            tmp2 = pd.DataFrame({"__group": df[group_col], "__group2": df[group_col2]})
            payload["n_subcategories"] = int(
                tmp2.dropna(subset=["__group", "__group2"])
                .drop_duplicates()
                .shape[0]
            )

    except Exception as e:
        payload["error"] = "Nie udalo sie przygotowac statystyk struktury dla wybranego ukladu kolumn."
        payload["debug_error"] = f"{type(e).__name__}: {e}"
        return payload

    if price_col and price_col in df.columns and value_col in df.columns:
        try:
            price_tmp = pd.DataFrame({
                "__price": _coerce_numeric_series(df[price_col]),
                "__value": _coerce_numeric_series(df[value_col]).fillna(0.0).clip(lower=0.0),
            })
            price_tmp = price_tmp.dropna(subset=["__price"])
            price_tmp = price_tmp[price_tmp["__value"] > 0]
            s = price_tmp["__price"].dropna()
            if not s.empty:
                lo = float(s.quantile(0.01))
                hi = float(s.quantile(0.99))
                mn = float(s.min())
                mx = float(s.max())
                use_lo, use_hi = mn, mx
                clip = None
                if math.isfinite(lo) and math.isfinite(hi) and hi > lo and (hi - lo) < 0.85 * (mx - mn):
                    use_lo, use_hi = lo, hi
                    clip = {"q01": lo, "q99": hi}

                def _nice_step(raw_step: float) -> float:
                    if not math.isfinite(raw_step) or raw_step <= 0:
                        return 1.0
                    k = 10 ** math.floor(math.log10(raw_step))
                    m = raw_step / k
                    if m <= 1:
                        nice = 1
                    elif m <= 2:
                        nice = 2
                    elif m <= 5:
                        nice = 5
                    else:
                        nice = 10
                    return float(nice * k)

                rng = max(1e-12, use_hi - use_lo)
                step = _nice_step(rng / 20.0)

                def _n_bins(step_value: float) -> int:
                    return int(math.ceil((use_hi - use_lo) / step_value))

                while _n_bins(step) > 30:
                    step = _nice_step(step * 1.25)

                start = math.floor(use_lo / step) * step
                end = math.ceil(use_hi / step) * step
                if end <= start:
                    end = start + step

                price_tmp = price_tmp[
                    (price_tmp["__price"] >= start) & (price_tmp["__price"] <= end)
                ].copy()
                if price_tmp.empty:
                    raise ValueError("Price corridor clipping removed all priced rows.")

                edges = np.arange(start, end + step * 0.999999, step)
                price_tmp["price_bin"] = pd.cut(price_tmp["__price"], bins=edges, include_lowest=True)
                price_tmp = price_tmp.dropna(subset=["price_bin"]).copy()
                if price_tmp.empty:
                    raise ValueError("Price corridor binning produced no valid bins.")
                price_tmp["price_bin"] = price_tmp["price_bin"].astype(str)
                corr = (
                    price_tmp.groupby("price_bin", dropna=False, sort=False)["__value"]
                    .sum()
                    .reset_index()
                    .rename(columns={"__value": "value"})
                )
                corr = corr[corr["value"] > 0].copy()

                def _bin_start(label: str) -> float:
                    try:
                        match = re.match(r"[\[\(]([^,]+),", str(label))
                        return float(match.group(1)) if match else float("inf")
                    except Exception:
                        return float("inf")

                corr["_start"] = corr["price_bin"].map(_bin_start)
                corr = corr.sort_values("_start").drop(columns=["_start"]).reset_index(drop=True)

                total = float(corr["value"].sum() or 1.0)
                corr["share"] = corr["value"] / total
                corr["cum_pct"] = (corr["share"].cumsum() * 100.0).clip(0, 100)

                lo_idx = int((corr["cum_pct"] >= 20.0).idxmax()) if (corr["cum_pct"] >= 20.0).any() else int(corr.index.min())
                hi_idx = int((corr["cum_pct"] >= 80.0).idxmax()) if (corr["cum_pct"] >= 80.0).any() else int(corr.index.max())
                corridor_share_pct = float(corr.loc[lo_idx:hi_idx, "share"].sum() * 100.0)

                def _bin_hi(label: str) -> Optional[float]:
                    try:
                        match = re.match(r"[\[\(]([^,]+),\s*([^\]\)]+)[\]\)]", str(label))
                        return float(match.group(2)) if match else None
                    except Exception:
                        return None

                payload["price_corridor"] = {
                    "bin_method": "fixed",
                    "bin_step": step,
                    "bin_count": int(corr["price_bin"].nunique()),
                    "clip": clip,
                    "p80_price": _bin_hi(corr.loc[hi_idx, "price_bin"]),
                    "corridor_share_pct": corridor_share_pct,
                    "bins": corr[["price_bin", "value", "share", "cum_pct"]].to_dict(orient="records"),
                }
        except Exception as e:
            payload["price_corridor_error"] = "Nie udalo sie przygotowac statystyk korytarza cenowego."
            payload["price_corridor_debug_error"] = f"{type(e).__name__}: {e}"

    return payload


def _frame_from_stats_lists(
    labels: list[Any] | None,
    values: list[Any] | None,
    shares_pct: list[Any] | None,
    total_value: float,
    tail_label: str | None = None,
    tail_value: float | None = None,
) -> pd.DataFrame:
    labels = list(labels or [])
    values = list(values or [])
    shares_pct = list(shares_pct or [])
    n = min(len(labels), len(values))
    if n <= 0 or total_value <= 0:
        return pd.DataFrame(columns=["group", "value", "share_full", "cum_share_full", "rank"])

    rows: list[dict[str, Any]] = []
    for i in range(n):
        value = float(values[i] or 0.0)
        if value <= 0:
            continue
        share_full = (
            float(shares_pct[i]) / 100.0
            if i < len(shares_pct) and shares_pct[i] is not None
            else (value / total_value)
        )
        rows.append({
            "group": str(labels[i]),
            "value": value,
            "share_full": share_full,
        })

    if tail_label and tail_value and float(tail_value) > 0:
        value = float(tail_value)
        rows.append({
            "group": str(tail_label),
            "value": value,
            "share_full": value / total_value,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["group", "value", "share_full", "cum_share_full", "rank"])

    frame = frame.sort_values("value", ascending=False).reset_index(drop=True)
    frame["cum_share_full"] = frame["share_full"].cumsum()
    frame["rank"] = range(1, len(frame) + 1)
    return frame


def _build_cs_structure_frames_from_stats(
    stats_payload: Dict[str, Any] | None,
    top_n: int,
    synthetic_group_tail: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    stats_payload = stats_payload if isinstance(stats_payload, dict) else {}
    total_full = float(stats_payload.get("total_value") or 0.0)
    hhi = float(stats_payload.get("hhi") or 0.0)
    top_n = min(max(5, int(top_n or 10)), 50)

    base = _frame_from_stats_lists(
        labels=list(stats_payload.get("display_labels") or []),
        values=list(stats_payload.get("display_values") or []),
        shares_pct=list(stats_payload.get("display_shares_pct") or []),
        total_value=total_full,
        tail_label=synthetic_group_tail,
        tail_value=float(stats_payload.get("display_other_value") or 0.0),
    )
    head = _frame_from_stats_lists(
        labels=list(stats_payload.get("top_labels") or []),
        values=list(stats_payload.get("top_values") or []),
        shares_pct=list(stats_payload.get("top_shares_pct") or []),
        total_value=total_full,
        tail_label=synthetic_group_tail,
        tail_value=float(stats_payload.get("other_value") or 0.0),
    )

    if head.empty and not base.empty:
        head = base.head(top_n).copy()
        tail = base.iloc[top_n:].copy()
        if not tail.empty:
            other_val = float(tail["value"].sum())
            other_row = pd.DataFrame([{
                "group": synthetic_group_tail,
                "value": other_val,
                "share_full": other_val / float(total_full or 1.0),
            }])
            head = pd.concat([head, other_row], ignore_index=True)
        head["cum_share_full"] = head["share_full"].cumsum()
        head["rank"] = range(1, len(head) + 1)

    return base, head, total_full, hhi


def _build_price_corridor_from_stats(
    stats_payload: Dict[str, Any] | None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    stats_payload = stats_payload if isinstance(stats_payload, dict) else {}
    pc = stats_payload.get("price_corridor") or {}
    bins = list(pc.get("bins") or [])
    if not bins:
        return pd.DataFrame(), {}

    corr = pd.DataFrame(bins).copy()
    if corr.empty or "price_bin" not in corr.columns or "value" not in corr.columns:
        return pd.DataFrame(), {}

    corr["value"] = pd.to_numeric(corr["value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "share" in corr.columns:
        corr["share"] = pd.to_numeric(corr["share"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        total = float(corr["value"].sum() or 1.0)
        corr["share"] = corr["value"] / total
    if "cum_pct" in corr.columns:
        corr["cum_pct"] = pd.to_numeric(corr["cum_pct"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    else:
        corr["cum_pct"] = (corr["share"].cumsum() * 100.0).clip(0, 100)
    corr["x_i"] = range(len(corr))

    def _bin_edges(lbl: Any) -> Tuple[float | None, float | None]:
        try:
            m = re.match(r"[\[\(]([^,]+),\s*([^\]\)]+)[\]\)]", str(lbl))
            if not m:
                return None, None
            return float(m.group(1)), float(m.group(2))
        except Exception:
            return None, None

    lo_idx = int((corr["cum_pct"] >= 20.0).idxmax()) if (corr["cum_pct"] >= 20.0).any() else int(corr.index.min())
    hi_idx = int((corr["cum_pct"] >= 80.0).idxmax()) if (corr["cum_pct"] >= 80.0).any() else int(corr.index.max())
    lo_edge = _bin_edges(corr.loc[lo_idx, "price_bin"])[0]
    hi_edge = _bin_edges(corr.loc[hi_idx, "price_bin"])[1]

    meta = {
        "step": pc.get("bin_step"),
        "clip": pc.get("clip"),
        "p80_price": pc.get("p80_price"),
        "corridor_share_pct": pc.get("corridor_share_pct"),
        "corridor_low": lo_edge,
        "corridor_high": hi_edge,
        "lo_idx": lo_idx,
        "hi_idx": hi_idx,
        "bin80_label": str(corr.loc[hi_idx, "price_bin"]) if hi_idx in corr.index else None,
    }
    return corr, meta


def _make_cs_runtime_signature(df: pd.DataFrame, filters: Dict[str, Any] | None = None) -> str:
    payload = {
        "renderer_version": "composition_static_charts_v9",
        "rows": int(len(df) if isinstance(df, pd.DataFrame) else 0),
        "columns": [str(c) for c in (df.columns if isinstance(df, pd.DataFrame) else [])],
        "dtypes": [str(t) for t in (df.dtypes if isinstance(df, pd.DataFrame) else [])],
        "filters": filters or {},
    }
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _cs_cache_get(bucket: str, key: str) -> Any:
    cache = st.session_state.get(bucket)
    if not isinstance(cache, dict):
        return None
    value = cache.get(key)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return deepcopy(value)
    return value


def _cs_cache_put(bucket: str, key: str, value: Any, max_entries: int = 8) -> None:
    cache = st.session_state.get(bucket)
    if not isinstance(cache, dict):
        cache = {}
    if isinstance(value, pd.DataFrame):
        cache[key] = value.copy()
    elif isinstance(value, dict):
        cache[key] = deepcopy(value)
    else:
        cache[key] = value
    while len(cache) > max_entries:
        cache.pop(next(iter(cache)))
    st.session_state[bucket] = cache


def _build_treemap_frame(
    df: pd.DataFrame,
    group_col: str | None,
    group_col2: str | None,
    value_col: str | None,
    top_n: int,
) -> tuple[pd.DataFrame, bool]:
    if (
        not isinstance(df, pd.DataFrame)
        or df.empty
        or not group_col
        or not value_col
        or group_col not in df.columns
        or value_col not in df.columns
    ):
        return pd.DataFrame(), False

    use_two_levels = bool(group_col2) and (group_col2 in df.columns) and (group_col2 != group_col)
    if use_two_levels:
        cols_need = [group_col, group_col2, value_col]
        agg = (
            df[cols_need]
            .dropna(subset=[group_col, group_col2])
            .groupby([group_col, group_col2], dropna=False, sort=False)[value_col]
            .sum()
            .reset_index()
            .rename(columns={group_col: "group", group_col2: "subgroup", value_col: "value"})
        )
    else:
        agg = (
            df[[group_col, value_col]]
            .dropna(subset=[group_col])
            .groupby(group_col, dropna=False, sort=False)[value_col]
            .sum()
            .reset_index()
            .rename(columns={group_col: "group", value_col: "value"})
        )

    agg["value"] = pd.to_numeric(agg["value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    agg = agg[agg["value"] > 0]
    if agg.empty:
        return pd.DataFrame(), use_two_levels

    if top_n and top_n > 0:
        treemap_tail_label = f"Pozostale (poza Top-{int(top_n)})"
        if treemap_tail_label in set(agg["group"].astype(str).tolist()):
            treemap_tail_label = f"{treemap_tail_label} [ogon]"
        top_groups = agg.groupby("group", sort=False)["value"].sum().nlargest(top_n).index.tolist()
        other = agg[~agg["group"].isin(top_groups)]
        agg = agg[agg["group"].isin(top_groups)]
        if not other.empty:
            other_sum = float(other["value"].sum())
            if use_two_levels:
                agg = pd.concat(
                    [agg, pd.DataFrame([{"group": treemap_tail_label, "subgroup": treemap_tail_label, "value": other_sum}])],
                    ignore_index=True,
                )
            else:
                agg = pd.concat([agg, pd.DataFrame([{"group": treemap_tail_label, "value": other_sum}])], ignore_index=True)

    return agg.reset_index(drop=True), use_two_levels


def _binary_treemap_layout(
    rows: list[dict[str, Any]],
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    x1: float = 100.0,
    y1: float = 100.0,
) -> list[dict[str, Any]]:
    """Small dependency-free binary treemap layout in percent coordinates."""
    clean = [dict(r) for r in rows if float(r.get("value") or 0.0) > 0]
    if not clean:
        return []
    if len(clean) == 1:
        r = dict(clean[0])
        r.update({"x0": float(x0), "x1": float(x1), "y0": float(y0), "y1": float(y1)})
        return [r]

    clean.sort(key=lambda r: float(r.get("value") or 0.0), reverse=True)
    total = float(sum(float(r.get("value") or 0.0) for r in clean) or 0.0)
    if total <= 0:
        return []

    running = 0.0
    split_idx = 1
    half = total / 2.0
    for i, r in enumerate(clean, 1):
        running += float(r.get("value") or 0.0)
        split_idx = i
        if running >= half:
            break

    left = clean[:split_idx]
    right = clean[split_idx:]
    if not right:
        left = clean[:-1]
        right = clean[-1:]
    left_sum = float(sum(float(r.get("value") or 0.0) for r in left) or 0.0)
    frac = left_sum / total if total else 0.5

    width = float(x1 - x0)
    height = float(y1 - y0)
    if width >= height:
        xm = x0 + width * frac
        return (
            _binary_treemap_layout(left, x0=x0, y0=y0, x1=xm, y1=y1)
            + _binary_treemap_layout(right, x0=xm, y0=y0, x1=x1, y1=y1)
        )
    ym = y0 + height * frac
    return (
        _binary_treemap_layout(left, x0=x0, y0=ym, x1=x1, y1=y1)
        + _binary_treemap_layout(right, x0=x0, y0=y0, x1=x1, y1=ym)
    )


def _build_treemap_rects(agg: pd.DataFrame, *, use_two_levels: bool) -> pd.DataFrame:
    if not isinstance(agg, pd.DataFrame) or agg.empty:
        return pd.DataFrame()

    src = agg.copy()
    src["value"] = pd.to_numeric(src.get("value"), errors="coerce").fillna(0.0).clip(lower=0.0)
    src = src[src["value"] > 0].copy()
    if src.empty:
        return pd.DataFrame()

    total = float(src["value"].sum() or 0.0)
    group_rows = (
        src.groupby("group", dropna=False, sort=False)["value"]
        .sum()
        .reset_index()
        .rename(columns={"group": "label"})
        .to_dict("records")
    )
    group_rects = _binary_treemap_layout(group_rows)

    out: list[dict[str, Any]] = []
    if use_two_levels and "subgroup" in src.columns:
        for gr in group_rects:
            g = str(gr.get("label"))
            part = src[src["group"].astype(str) == g].copy()
            sub_rows = (
                part.groupby("subgroup", dropna=False, sort=False)["value"]
                .sum()
                .reset_index()
                .rename(columns={"subgroup": "label"})
                .to_dict("records")
            )
            sub_rects = _binary_treemap_layout(
                sub_rows,
                x0=float(gr["x0"]),
                y0=float(gr["y0"]),
                x1=float(gr["x1"]),
                y1=float(gr["y1"]),
            )
            for sr in sub_rects:
                val = float(sr.get("value") or 0.0)
                area = abs(float(sr["x1"]) - float(sr["x0"])) * abs(float(sr["y1"]) - float(sr["y0"]))
                out.append({
                    **sr,
                    "group_label": g,
                    "subgroup_label": str(sr.get("label")),
                    "display_label": str(sr.get("label")) if area >= 85 else "",
                    "share": val / total if total else 0.0,
                })
    else:
        for gr in group_rects:
            val = float(gr.get("value") or 0.0)
            area = abs(float(gr["x1"]) - float(gr["x0"])) * abs(float(gr["y1"]) - float(gr["y0"]))
            out.append({
                **gr,
                "group_label": str(gr.get("label")),
                "subgroup_label": "",
                "display_label": str(gr.get("label")) if area >= 85 else "",
                "share": val / total if total else 0.0,
            })

    rects = pd.DataFrame(out)
    if rects.empty:
        return rects
    rects["label_x"] = rects["x0"].astype(float) + 0.8
    rects["label_y"] = rects["y1"].astype(float) - 3.0
    return rects


def _render_controlled_treemap(agg: pd.DataFrame, *, use_two_levels: bool) -> bool:
    """Render deterministic treemap rectangles with labels in each top-left corner."""
    try:
        import plotly.graph_objects as go_mod  # type: ignore
    except Exception:
        go_mod = go
    if go_mod is None or not isinstance(agg, pd.DataFrame) or agg.empty:
        return False

    src = agg.copy()
    src["value"] = pd.to_numeric(src.get("value"), errors="coerce").fillna(0.0).clip(lower=0.0)
    src = src[src["value"] > 0].copy()
    if src.empty or "group" not in src.columns:
        return False

    # The overview treemap must explain the primary business segments first.
    # Kategoria 2 powers Mix/Marimekko, but here it made the treemap label
    # countries instead of categories, so the overview intentionally stays
    # at the category level.
    group_df = (
        src.groupby("group", dropna=False, sort=False)["value"]
        .sum()
        .reset_index()
        .rename(columns={"group": "label"})
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )
    rows = group_df.to_dict("records")
    rects = pd.DataFrame(_binary_treemap_layout(rows))
    if rects.empty:
        return False

    total = float(rects["value"].sum() or 0.0)
    rects["share"] = np.where(total > 0, rects["value"] / total, 0.0)
    rects["area"] = (rects["x1"].astype(float) - rects["x0"].astype(float)).abs() * (
        rects["y1"].astype(float) - rects["y0"].astype(float)
    ).abs()

    palette = [
        "#FFD166", "#2CB1A1", "#0B70C9", "#76E39B", "#FF9EA5",
        "#FF8700", "#D1D5DB", "#7EC5F4", "#FF2D2D", "#6F42C1",
        "#9CA3AF", "#F59E0B", "#10B981", "#3B82F6", "#EF4444",
    ]
    fig = go_mod.Figure()
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_cd: list[list[Any]] = []

    for i, r in rects.iterrows():
        x0, x1 = float(r["x0"]), float(r["x1"])
        y0, y1 = float(r["y0"]), float(r["y1"])
        label = str(r.get("label") or "")
        value = float(r.get("value") or 0.0)
        share = float(r.get("share") or 0.0)
        fill = palette[int(i) % len(palette)]
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color="#ffffff", width=1.5),
            fillcolor=fill,
        )
        if float(r.get("area") or 0.0) >= 65:
            fig.add_annotation(
                x=x0 + 0.8,
                y=y1 - 2.2,
                text=f"<b>{label}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                font=dict(size=12, color="#111827"),
                bgcolor="rgba(255,255,255,0.72)",
                bordercolor="rgba(255,255,255,0)",
                borderpad=2,
            )
        hover_x.append(x0 + (x1 - x0) / 2.0)
        hover_y.append(y0 + (y1 - y0) / 2.0)
        hover_cd.append([label, value, share])

    fig.add_trace(
        go_mod.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(size=18, opacity=0),
            customdata=hover_cd,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Wartosc: %{customdata[1]:,.0f}<br>"
                "Udzial: %{customdata[2]:.1%}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 100], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 100], visible=False, fixedrange=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    plotly_chart_stretch(st, fig, config={"displayModeBar": False})
    return True


def _render_altair_overview_composition(agg: pd.DataFrame, *, use_two_levels: bool, legend_title: str = "Kategoria") -> None:
    """Render an overview chart when Plotly treemap is not available."""
    if not isinstance(agg, pd.DataFrame) or agg.empty:
        st.info("Brak danych do wizualizacji struktury.")
        return

    chart_df = agg.copy()
    chart_df["value"] = pd.to_numeric(chart_df.get("value"), errors="coerce").fillna(0.0)
    chart_df = chart_df[chart_df["value"] > 0].copy()
    if chart_df.empty:
        st.info("Brak dodatnich wartości do wizualizacji struktury.")
        return

    rects = _build_treemap_rects(chart_df, use_two_levels=use_two_levels)
    if rects.empty:
        st.info("Brak dodatnich wartości do wizualizacji struktury.")
        return

    base = (
        alt.Chart(rects)
        .mark_rect(stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("x0:Q", title=None, scale=alt.Scale(domain=[0, 100]), axis=None),
            x2="x1:Q",
            y=alt.Y("y0:Q", title=None, scale=alt.Scale(domain=[0, 100]), axis=None),
            y2="y1:Q",
            color=alt.Color(
                "group_label:N" if use_two_levels else "group_label:N",
                title=str(legend_title or "Kategoria"),
                legend=None if not use_two_levels else alt.Legend(orient="bottom", columns=4),
            ),
            tooltip=[
                alt.Tooltip("group_label:N", title="Kategoria"),
                alt.Tooltip("subgroup_label:N", title="Podkategoria"),
                alt.Tooltip("value:Q", format=",.0f", title="Wartość"),
                alt.Tooltip("share:Q", format=".1%", title="Udział w całości"),
            ],
        )
    )
    labels = (
        alt.Chart(rects[rects["display_label"].astype(str) != ""])
        .mark_text(
            align="left",
            baseline="top",
            dx=2,
            dy=2,
            color="#111827",
            fontSize=12,
            fontWeight="bold",
        )
        .encode(
            x=alt.X("label_x:Q", scale=alt.Scale(domain=[0, 100]), axis=None),
            y=alt.Y("label_y:Q", scale=alt.Scale(domain=[0, 100]), axis=None),
            text="display_label:N",
        )
    )
    altair_chart_stretch(st, (base + labels).properties(height=520).configure_view(strokeOpacity=0), width="stretch")


def _build_altair_waterfall_chart(wf: pd.DataFrame, *, synthetic_group_tail: str) -> alt.Chart:
    wf2 = wf.copy()
    wf2["label"] = wf2["label"].astype(str)
    wf2["value"] = pd.to_numeric(wf2["value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    wf2 = wf2[wf2["value"] > 0].copy()
    total_value = float(wf2["value"].sum() or 0.0)

    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for _, r in wf2.iterrows():
        value = float(r["value"])
        label = str(r["label"])
        rows.append({
            "label": label,
            "x_start": cumulative,
            "x_end": cumulative + value,
            "value": value,
            "kind": "Other" if label == synthetic_group_tail else "Wkład",
            "text_x": cumulative + value,
        })
        cumulative += value
    rows.append({
        "label": "TOTAL",
        "x_start": 0.0,
        "x_end": total_value,
        "value": total_value,
        "kind": "TOTAL",
        "text_x": total_value,
    })
    plot = pd.DataFrame(rows)
    order = plot["label"].tolist()
    total_axis = float(plot["x_end"].max() or total_value or 1.0)
    label_pad = total_axis * 0.008
    height = min(520, max(320, 70 + 28 * int(plot.shape[0])))

    bars = (
        alt.Chart(plot)
        .mark_bar(cornerRadius=1)
        .encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("x_start:Q", title="Narastający wkład do totalu"),
            x2="x_end:Q",
            color=alt.Color(
                "kind:N",
                title="",
                scale=alt.Scale(domain=["Wkład", "Other", "TOTAL"], range=["#2AAE6A", "#d9d9d9", "#2D6CDF"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Segment"),
                alt.Tooltip("value:Q", format=",.0f", title="Wartość"),
                alt.Tooltip("x_end:Q", format=",.0f", title="Wkład narastająco"),
            ],
        )
    )
    labels = (
        alt.Chart(plot)
        .mark_text(align="left", dx=6, fontSize=11, color="#374151")
        .encode(
            y=alt.Y("label:N", sort=order, axis=None),
            x=alt.X("text_x:Q"),
            text=alt.Text("value:Q", format=",.0f"),
        )
    )
    cat_labels_df = plot.copy()
    cat_labels_df["label_x"] = cat_labels_df["x_start"] + label_pad
    cat_labels = (
        alt.Chart(cat_labels_df)
        .mark_text(align="left", baseline="middle", fontSize=12, fontWeight="bold", color="#111827")
        .encode(
            y=alt.Y("label:N", sort=order, axis=None),
            x=alt.X("label_x:Q"),
            text="label:N",
        )
    )
    return (bars + cat_labels + labels).properties(height=height).configure_view(strokeOpacity=0)


def _build_marimekko_rects(mix_agg: pd.DataFrame, group_col: str, group_col2: str) -> pd.DataFrame:
    if not isinstance(mix_agg, pd.DataFrame) or mix_agg.empty:
        return pd.DataFrame()
    required = {group_col, group_col2, "value", "group_total", "pct", "group_share_pct", "area_share_pct"}
    if not required.issubset(set(mix_agg.columns)):
        return pd.DataFrame()

    mm = mix_agg.copy()
    mm["value"] = pd.to_numeric(mm["value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    mm["pct"] = pd.to_numeric(mm["pct"], errors="coerce").fillna(0.0).clip(lower=0.0)
    mm["group_share_pct"] = pd.to_numeric(mm["group_share_pct"], errors="coerce").fillna(0.0).clip(lower=0.0)
    mm["area_share_pct"] = pd.to_numeric(mm["area_share_pct"], errors="coerce").fillna(0.0).clip(lower=0.0)
    mm = mm[mm["value"] > 0].copy()
    if mm.empty:
        return pd.DataFrame()

    group_totals = (
        mm[[group_col, "group_total", "group_share_pct"]]
        .drop_duplicates()
        .sort_values("group_total", ascending=False)
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    x0 = 0.0
    for _, g_row in group_totals.iterrows():
        group = str(g_row[group_col])
        width = float(g_row["group_share_pct"])
        y0 = 0.0
        part = mm[mm[group_col].astype(str) == group].sort_values(["pct", "value"], ascending=[False, False])
        for _, r in part.iterrows():
            height = float(r["pct"])
            area = float(r["area_share_pct"])
            rows.append({
                "group_label": group,
                "subgroup_label": str(r[group_col2]),
                "x0": x0,
                "x1": x0 + width,
                "y0": y0,
                "y1": y0 + height,
                "value": float(r["value"]),
                "width_pct": width,
                "height_pct": height,
                "area_share_pct": area,
                "display_label": str(r[group_col2]) if width >= 5 and height >= 5 and (width * height) >= 1.0 else "",
                "label_x": x0 + width / 2.0,
                "label_y": y0 + height / 2.0,
            })
            y0 += height
        x0 += width
    return pd.DataFrame(rows)


def _build_mix_exec_frame(
    df: pd.DataFrame,
    group_col: str | None,
    group_col2: str | None,
    value_col: str | None,
    top_n: int = 10,
    top_k: int = 7,
) -> pd.DataFrame:
    """Build the exact Top-N + Top-K mix frame used by Mix/Marimekko charts."""
    if (
        not isinstance(df, pd.DataFrame)
        or df.empty
        or not group_col
        or not group_col2
        or not value_col
        or group_col not in df.columns
        or group_col2 not in df.columns
        or value_col not in df.columns
        or group_col == group_col2
    ):
        return pd.DataFrame(columns=[group_col or "group", group_col2 or "subgroup", "value", "group_total", "pct", "group_share_pct", "area_share_pct"])

    try:
        sub = df[[group_col, group_col2, value_col]].dropna(subset=[group_col, group_col2]).copy()
        sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        sub = sub[sub[value_col] > 0]
        if sub.empty:
            return pd.DataFrame(columns=[group_col, group_col2, "value", "group_total", "pct", "group_share_pct", "area_share_pct"])

        mix_agg = (
            sub.groupby([group_col, group_col2], dropna=False, sort=False)[value_col]
            .sum()
            .reset_index()
            .rename(columns={value_col: "value"})
        )
        if mix_agg.empty:
            return pd.DataFrame(columns=[group_col, group_col2, "value", "group_total", "pct", "group_share_pct", "area_share_pct"])

        full_total_value = float(mix_agg["value"].sum() or 0.0)
        if full_total_value <= 0:
            return pd.DataFrame(columns=[group_col, group_col2, "value", "group_total", "pct", "group_share_pct", "area_share_pct"])

        top_groups = (
            mix_agg.groupby(group_col, sort=False)["value"]
            .sum()
            .nlargest(max(1, int(top_n or 10)))
            .index
            .tolist()
        )
        mix_agg = mix_agg[mix_agg[group_col].isin(top_groups)].copy()
        if mix_agg.empty:
            return pd.DataFrame(columns=[group_col, group_col2, "value", "group_total", "pct", "group_share_pct", "area_share_pct"])

        top_sub = (
            mix_agg.groupby(group_col2, sort=False)["value"]
            .sum()
            .nlargest(max(1, int(top_k or 7)))
            .index
            .tolist()
        )
        existing_components = [str(v).strip() for v in mix_agg[group_col2].dropna().astype(str).tolist()]
        tail_label = f"Pozostale (poza Top-{max(1, int(top_k or 7))})"
        if tail_label in existing_components:
            tail_label = f"{tail_label} [ogon]"
        mix_agg[group_col2] = mix_agg[group_col2].where(mix_agg[group_col2].isin(top_sub), other=tail_label)
        mix_agg = mix_agg.groupby([group_col, group_col2], dropna=False, sort=False)["value"].sum().reset_index()
        if mix_agg.empty:
            return pd.DataFrame(columns=[group_col, group_col2, "value", "group_total", "pct", "group_share_pct", "area_share_pct"])

        group_totals = mix_agg.groupby(group_col, sort=False)["value"].sum().rename("group_total").reset_index()
        if group_totals.empty:
            return pd.DataFrame(columns=[group_col, group_col2, "value", "group_total", "pct", "group_share_pct", "area_share_pct"])

        mix_agg = mix_agg.merge(group_totals, on=group_col, how="left")
        mix_agg["pct"] = mix_agg["value"] / mix_agg["group_total"] * 100.0
        mix_agg["group_share_pct"] = mix_agg["group_total"] / full_total_value * 100.0
        mix_agg["area_share_pct"] = mix_agg["value"] / full_total_value * 100.0
        mix_agg = mix_agg.sort_values(["group_total", "pct", "value"], ascending=[False, False, False]).reset_index(drop=True)
        return mix_agg
    except Exception:
        return pd.DataFrame(columns=[group_col or "group", group_col2 or "subgroup", "value", "group_total", "pct", "group_share_pct", "area_share_pct"])


def _build_resilient_mix_frame(
    df: pd.DataFrame,
    group_col: str | None,
    group_col2: str | None,
    value_col: str | None,
    top_n: int = 10,
    top_k: int = 7,
) -> pd.DataFrame:
    """Forgiving Mix/Marimekko source used when strict frame creation returns empty."""
    empty_cols = [group_col or "group", group_col2 or "subgroup", "value", "group_total", "pct", "group_share_pct", "area_share_pct"]
    if (
        not isinstance(df, pd.DataFrame)
        or df.empty
        or not group_col
        or not group_col2
        or group_col == group_col2
        or group_col not in df.columns
        or group_col2 not in df.columns
    ):
        return pd.DataFrame(columns=empty_cols)

    try:
        sub = pd.DataFrame({
            group_col: df[group_col].astype("string").fillna("(brak)"),
            group_col2: df[group_col2].astype("string").fillna("(brak)"),
        })
        if value_col and value_col in df.columns:
            sub["value"] = _coerce_numeric_series(df[value_col]).fillna(0.0).clip(lower=0.0)
        else:
            sub["value"] = 1.0
        sub = sub[(sub[group_col].astype(str).str.strip() != "") & (sub[group_col2].astype(str).str.strip() != "")]
        sub = sub[sub["value"] > 0]
        if sub.empty:
            return pd.DataFrame(columns=empty_cols)

        mix = sub.groupby([group_col, group_col2], dropna=False, sort=False)["value"].sum().reset_index()
        if mix.empty:
            return pd.DataFrame(columns=empty_cols)

        full_total = float(mix["value"].sum() or 0.0)
        if full_total <= 0:
            return pd.DataFrame(columns=empty_cols)

        top_groups = (
            mix.groupby(group_col, sort=False)["value"]
            .sum()
            .nlargest(min(50, max(5, int(top_n or 10))))
            .index
            .tolist()
        )
        mix = mix[mix[group_col].isin(top_groups)].copy()
        if mix.empty:
            return pd.DataFrame(columns=empty_cols)

        top_subs = (
            mix.groupby(group_col2, sort=False)["value"]
            .sum()
            .nlargest(max(2, int(top_k or 7)))
            .index
            .tolist()
        )
        tail_label = f"Pozostale (poza Top-{max(2, int(top_k or 7))})"
        if tail_label in set(mix[group_col2].astype(str).tolist()):
            tail_label = f"{tail_label} [ogon]"
        mix[group_col2] = mix[group_col2].where(mix[group_col2].isin(top_subs), tail_label)
        mix = mix.groupby([group_col, group_col2], dropna=False, sort=False)["value"].sum().reset_index()

        totals = mix.groupby(group_col, sort=False)["value"].sum().rename("group_total").reset_index()
        mix = mix.merge(totals, on=group_col, how="left")
        mix["pct"] = np.where(mix["group_total"] > 0, mix["value"] / mix["group_total"] * 100.0, 0.0)
        mix["group_share_pct"] = np.where(full_total > 0, mix["group_total"] / full_total * 100.0, 0.0)
        mix["area_share_pct"] = np.where(full_total > 0, mix["value"] / full_total * 100.0, 0.0)
        return mix.sort_values(["group_total", "pct", "value"], ascending=[False, False, False]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=empty_cols)


def _build_mix_exec_stats(
    df: pd.DataFrame,
    group_col: str | None,
    group_col2: str | None,
    value_col: str | None,
    top_n: int = 10,
) -> Dict[str, Any]:
    mix_agg = _build_mix_exec_frame(df, group_col, group_col2, value_col, top_n=top_n, top_k=7)
    if mix_agg.empty:
        mix_agg = _build_resilient_mix_frame(df, group_col, group_col2, value_col, top_n=top_n, top_k=7)
    return _build_mix_exec_stats_from_frame(mix_agg, group_col, group_col2)


def _build_mix_exec_stats_from_frame(
    mix_agg: pd.DataFrame,
    group_col: str | None,
    group_col2: str | None,
) -> Dict[str, Any]:
    if mix_agg.empty or not group_col or not group_col2:
        return {}

    try:
        total_value = float(mix_agg["value"].sum() or 0.0)
        if total_value <= 0:
            return {}

        group_totals = (
            mix_agg[[group_col, "group_total", "group_share_pct"]]
            .drop_duplicates()
            .sort_values("group_total", ascending=False)
            .reset_index(drop=True)
        )
        focus_group = str(group_totals.loc[0, group_col])
        focus_group_share_pct = round(float(group_totals.loc[0, "group_share_pct"]), 2)
        top3_groups_share_pct = round(float(group_totals.head(min(3, len(group_totals)))["group_share_pct"].sum()), 2)

        focus_rows = (
            mix_agg[mix_agg[group_col].astype(str) == focus_group]
            .sort_values(["pct", "value"], ascending=[False, False])
            .reset_index(drop=True)
        )
        focus_component = str(focus_rows.loc[0, group_col2])
        focus_component_pct = round(float(focus_rows.loc[0, "pct"]), 2)
        second_component = str(focus_rows.loc[1, group_col2]) if len(focus_rows) > 1 else None
        second_component_pct = round(float(focus_rows.loc[1, "pct"]), 2) if len(focus_rows) > 1 else None
        focus_gap_pp = round(float(focus_component_pct - (second_component_pct or 0.0)), 2)

        top_cell = mix_agg.sort_values(["area_share_pct", "pct", "value"], ascending=[False, False, False]).iloc[0]
        top_cell_group = str(top_cell[group_col])
        top_cell_component = str(top_cell[group_col2])
        top_cell_share_pct = round(float(top_cell["area_share_pct"]), 2)

        comp_profile = mix_agg[mix_agg[group_col2].astype(str) == focus_component]
        profile_spread_pp = 0.0
        if len(comp_profile) >= 2:
            profile_spread_pp = round(
                float(comp_profile["pct"].max() - comp_profile["pct"].min()),
                2,
            )

        return {
            "mix_groups_count": int(group_totals.shape[0]),
            "mix_components_count": int(mix_agg[group_col2].nunique(dropna=True)),
            "mix_focus_group": focus_group,
            "mix_focus_group_share_pct": focus_group_share_pct,
            "mix_top3_groups_share_pct": top3_groups_share_pct,
            "mix_focus_component": focus_component,
            "mix_focus_component_pct": focus_component_pct,
            "mix_second_component": second_component,
            "mix_second_component_pct": second_component_pct,
            "mix_focus_gap_pp": focus_gap_pp,
            "mix_top_cell_group": top_cell_group,
            "mix_top_cell_component": top_cell_component,
            "mix_top_cell_share_pct": top_cell_share_pct,
            "mix_profile_spread_pp": profile_spread_pp,
        }
    except Exception:
        return {}


def _build_marimekko_exec_stats(
    df: pd.DataFrame,
    group_col: str | None,
    group_col2: str | None,
    value_col: str | None,
    top_n: int = 10,
) -> Dict[str, Any]:
    mix_agg = _build_mix_exec_frame(df, group_col, group_col2, value_col, top_n=top_n, top_k=7)
    if mix_agg.empty:
        mix_agg = _build_resilient_mix_frame(df, group_col, group_col2, value_col, top_n=top_n, top_k=7)
    return _build_marimekko_exec_stats_from_frame(mix_agg, group_col, group_col2)


def _build_marimekko_exec_stats_from_frame(
    mix_agg: pd.DataFrame,
    group_col: str | None,
    group_col2: str | None,
) -> Dict[str, Any]:
    if mix_agg.empty or not group_col or not group_col2:
        return {}

    try:
        total_value = float(mix_agg["value"].sum() or 0.0)
        if total_value <= 0:
            return {}

        mm = mix_agg.copy()
        mm["width_pct"] = mm["group_share_pct"]
        mm["height_pct"] = mm["pct"]
        mm["area_pct"] = mm["area_share_pct"]

        group_totals = (
            mm[[group_col, "group_total", "width_pct"]]
            .drop_duplicates()
            .sort_values("group_total", ascending=False)
            .reset_index(drop=True)
        )
        top_width_group = str(group_totals.loc[0, group_col])
        top_width_pct = round(float(group_totals.loc[0, "width_pct"]), 2)
        second_width_group = str(group_totals.loc[1, group_col]) if len(group_totals) > 1 else None
        second_width_pct = round(float(group_totals.loc[1, "width_pct"]), 2) if len(group_totals) > 1 else None

        focus_rows = (
            mm[mm[group_col].astype(str) == top_width_group]
            .sort_values(["height_pct", "value"], ascending=[False, False])
            .reset_index(drop=True)
        )
        focus_component = str(focus_rows.loc[0, group_col2])
        focus_component_height_pct = round(float(focus_rows.loc[0, "height_pct"]), 2)
        focus_component_area_pct = round(float(focus_rows.loc[0, "area_pct"]), 2)

        top_cell = mm.sort_values(["area_pct", "height_pct", "value"], ascending=[False, False, False]).iloc[0]
        return {
            "marimekko_groups_count": int(group_totals.shape[0]),
            "marimekko_components_count": int(mm[group_col2].nunique(dropna=True)),
            "marimekko_top_width_group": top_width_group,
            "marimekko_top_width_pct": top_width_pct,
            "marimekko_second_width_group": second_width_group,
            "marimekko_second_width_pct": second_width_pct,
            "marimekko_focus_component": focus_component,
            "marimekko_focus_component_height_pct": focus_component_height_pct,
            "marimekko_focus_component_area_pct": focus_component_area_pct,
            "marimekko_top_cell_group": str(top_cell[group_col]),
            "marimekko_top_cell_component": str(top_cell[group_col2]),
            "marimekko_top_cell_area_pct": round(float(top_cell["area_pct"]), 2),
            "marimekko_top_cell_height_pct": round(float(top_cell["height_pct"]), 2),
            "marimekko_top_cell_width_pct": round(float(top_cell["width_pct"]), 2),
        }
    except Exception:
        return {}


def _build_price_corridor_exec_stats_from_payload(
    stats_payload: Dict[str, Any] | None,
) -> Dict[str, Any]:
    corr, meta = _build_price_corridor_from_stats(stats_payload)
    if corr.empty or not isinstance(meta, dict):
        return {}

    try:
        peak_row = (
            corr.sort_values(["share", "value"], ascending=[False, False])
            .reset_index(drop=True)
            .iloc[0]
        )
        return {
            "price_corridor_p80": meta.get("p80_price"),
            "price_corridor_share_pct": meta.get("corridor_share_pct"),
            "price_corridor_low": meta.get("corridor_low"),
            "price_corridor_high": meta.get("corridor_high"),
            "price_corridor_bin80_label": meta.get("bin80_label"),
            "price_corridor_bin_step": meta.get("step"),
            "price_corridor_peak_bin": str(peak_row.get("price_bin") or ""),
            "price_corridor_peak_bin_share_pct": round(float(peak_row.get("share", 0.0)) * 100.0, 2),
        }
    except Exception:
        return {}


def _format_cs_pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}%".replace(".", ",")
    except Exception:
        return "—"


def _build_mix_exec_repair(stats: Dict[str, Any]) -> str | None:
    focus_group = str(stats.get("mix_focus_group") or "").strip()
    focus_component = str(stats.get("mix_focus_component") or "").strip()
    focus_component_pct = stats.get("mix_focus_component_pct")
    focus_group_share_pct = stats.get("mix_focus_group_share_pct")
    focus_gap_pp = stats.get("mix_focus_gap_pp")
    if not focus_group or not focus_component or focus_component_pct is None or focus_group_share_pct is None:
        return None
    sentence1 = (
        f"W grupie '{focus_group}' udział '{focus_component}' w mixie wynosi "
        f"{_format_cs_pct(focus_component_pct)}, a sama grupa odpowiada za "
        f"{_format_cs_pct(focus_group_share_pct)} całkowitej wartości."
    )
    if focus_gap_pp is not None:
        sentence2 = (
            f"Rekomendacja: traktuj '{focus_component}' jako główny driver tej grupy "
            f"i monitoruj lukę {_format_cs_pct(focus_gap_pp).replace('%', ' pp')} do kolejnego składnika."
        )
    else:
        sentence2 = (
            f"Rekomendacja: traktuj '{focus_component}' jako główny driver tej grupy "
            f"i monitoruj zmianę jego udziału w kolejnych przebiegach."
        )
    return f"{sentence1} {sentence2}"


def _build_marimekko_exec_repair(stats: Dict[str, Any]) -> str | None:
    top_cell_group = str(stats.get("marimekko_top_cell_group") or "").strip()
    top_cell_component = str(stats.get("marimekko_top_cell_component") or "").strip()
    top_cell_area_pct = stats.get("marimekko_top_cell_area_pct")
    top_width_group = str(stats.get("marimekko_top_width_group") or "").strip()
    top_width_pct = stats.get("marimekko_top_width_pct")
    top_cell_height_pct = stats.get("marimekko_top_cell_height_pct")
    if not top_cell_group or not top_cell_component or top_cell_area_pct is None or not top_width_group or top_width_pct is None:
        return None
    sentence1 = (
        f"Największa komórka '{top_cell_group} × {top_cell_component}' wnosi "
        f"{_format_cs_pct(top_cell_area_pct)} całkowitej wartości, szerokość grupy "
        f"'{top_width_group}' wynosi {_format_cs_pct(top_width_pct)} totalu, "
        f"a udział '{top_cell_component}' wewnątrz tej grupy to {_format_cs_pct(top_cell_height_pct) if top_cell_height_pct is not None else '—'}."
    )
    if top_cell_height_pct is not None:
        sentence2 = (
            f"Rekomendacja: zarządzaj tą komórką osobno, bo łączy wysoką skalę grupy "
            f"z udziałem {_format_cs_pct(top_cell_height_pct)} tego składnika wewnątrz grupy."
        )
    else:
        sentence2 = (
            f"Rekomendacja: zarządzaj tą komórką osobno, bo łączy wysoką skalę grupy "
            f"z największym wkładem do całkowitej wartości."
    )
    return f"{sentence1} {sentence2}"


def _build_price_corridor_exec_repair(stats: Dict[str, Any]) -> str | None:
    p80_price = stats.get("price_corridor_p80")
    corridor_share_pct = stats.get("price_corridor_share_pct")
    corridor_low = stats.get("price_corridor_low")
    corridor_high = stats.get("price_corridor_high")
    peak_bin = str(stats.get("price_corridor_peak_bin") or "").strip()
    peak_bin_share_pct = stats.get("price_corridor_peak_bin_share_pct")
    bin_step = stats.get("price_corridor_bin_step")

    if corridor_share_pct is None and p80_price is None:
        return None

    if corridor_low is not None and corridor_high is not None and p80_price is not None:
        sentence1 = (
            f"Korytarz cenowy {float(corridor_low):,.0f}-{float(corridor_high):,.0f} generuje "
            f"{_format_cs_pct(corridor_share_pct)} wartosci, a prog P80 wypada przy "
            f"{float(p80_price):,.0f}."
        ).replace(",", " ")
    elif p80_price is not None:
        sentence1 = (
            f"Do ceny {float(p80_price):,.0f} kumuluje sie {_format_cs_pct(corridor_share_pct)} "
            f"wartosci sprzedazy."
        ).replace(",", " ")
    else:
        sentence1 = (
            f"Korytarz cenowy 20-80 odpowiada za {_format_cs_pct(corridor_share_pct)} wartosci sprzedazy."
        )

    if peak_bin and peak_bin_share_pct is not None:
        sentence2 = (
            f"Rekomendacja: pilnuj dostepnosci i pricingu w binie {peak_bin}, bo sam odpowiada za "
            f"{_format_cs_pct(peak_bin_share_pct)} wartosci; krok binow wynosi {bin_step if bin_step is not None else 'n/d'}."
        )
    else:
        sentence2 = (
            "Rekomendacja: ustaw progi/KPI dla korytarza cenowego i monitoruj odchylenia od zakresu P80."
        )
    return f"{sentence1} {sentence2}"


def _repair_cs_batch_takeaway(label: str, text: str, stats: Dict[str, Any]) -> tuple[str, bool]:
    repaired: str | None = None
    if label == "mix":
        repaired = _build_mix_exec_repair(stats)
    elif label == "marimekko":
        repaired = _build_marimekko_exec_repair(stats)
    elif label == "price_corridor":
        repaired = _build_price_corridor_exec_repair(stats)

    if repaired and isinstance(repaired, str) and repaired.strip():
        return repaired.strip(), True
    return str(text or "").strip(), False


def _render_altair_composition_static_insights(
    *,
    df: pd.DataFrame,
    stats_payload: Dict[str, Any],
    group_col: str,
    group_col2: str | None,
    value_col: str,
    price_col: str | None,
    top_n: int,
    cutoff: float,
    mix_exec_frame: pd.DataFrame,
    exec_takeaway_fn: Any,
    guidance_fn: Any,
) -> Dict[str, Any]:
    topn = min(max(5, int(top_n or 10)), 50)
    synthetic_group_tail = f"Pozostale (poza Top-{topn})"
    all_known_groups = {
        str(v)
        for v in (
            list((stats_payload or {}).get("top_labels") or [])
            + list((stats_payload or {}).get("display_labels") or [])
        )
    }
    if synthetic_group_tail in all_known_groups:
        synthetic_group_tail = f"{synthetic_group_tail} [ogon]"

    base, head, total_full, hhi = _build_cs_structure_frames_from_stats(
        stats_payload=stats_payload,
        top_n=topn,
        synthetic_group_tail=synthetic_group_tail,
    )
    if base.empty or head.empty:
        st.info("Brak danych do analizy struktury po grupowaniu.")
        return {"chart_meta": {"kind": "composition_static", "plotly_available": False}, "chart_context": {}}

    def _safe_exec(key: str, anchors: Dict[str, Any]) -> None:
        try:
            exec_takeaway_fn(key, anchors)
        except Exception:
            pass

    def _safe_guidance(sens: str, interp: str, best: str) -> None:
        try:
            guidance_fn(sens, interp, best)
        except Exception:
            pass

    st.caption(f"HHI (koncentracja): **{hhi:,.0f}**".replace(",", " "))

    # 1. Ranking
    st.markdown("### Jak wygląda struktura wartości (absoluty)?")
    st.caption("Ranking wartości pozwala precyzyjnie porównać wielkość segmentów.")
    rank_df = head.copy()
    rank_df["group_label"] = rank_df["group"].astype(str)
    rank_chart = (
        alt.Chart(rank_df)
        .mark_bar()
        .encode(
            y=alt.Y("group_label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X("value:Q", title="Wartość"),
            tooltip=[
                alt.Tooltip("group_label:N", title="Kategoria"),
                alt.Tooltip("value:Q", format=",.0f", title="Wartość"),
                alt.Tooltip("share_full:Q", format=".1%", title="Udział w całości"),
            ],
        )
        .properties(height=min(520, max(320, 70 + 28 * int(rank_df.shape[0]))))
    )
    altair_chart_stretch(st, rank_chart, width="stretch")
    _safe_exec(
        "ranking",
        {
            "metric": value_col,
            "cat1": group_col,
            "top_segment": str(rank_df.iloc[0].get("group")) if not rank_df.empty else None,
            "top_value": float(rank_df.iloc[0].get("value", 0.0)) if not rank_df.empty else None,
            "top_share_pct": float(rank_df.iloc[0].get("share_full", 0.0)) * 100.0 if not rank_df.empty else None,
            "n_segments": int(rank_df.shape[0]),
            "hhi": float(hhi) if hhi is not None else None,
        },
    )
    _safe_guidance(
        "Daje twardą skalę i porównanie wielkości segmentów.",
        "Najdłuższe słupki wskazują segmenty, które budują największą część wyniku.",
        "Utrzymuj sortowanie malejące i ogranicz liczbę kategorii do Top-N oraz reszty.",
    )
    st.divider()

    # 2. Contribution / waterfall equivalent
    st.markdown("### Co realnie buduje total?")
    st.caption("Wkład segmentów do łącznej wartości (Top-N + reszta).")
    wf = (
        head.assign(label=head["group"].astype(str))
        .groupby("label", as_index=False, dropna=False)
        .agg(value=("value", "sum"))
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )
    wf_total = float(wf["value"].sum() or 0.0)
    wf["share"] = wf["value"] / wf_total if wf_total else 0.0
    wf_chart = _build_altair_waterfall_chart(wf, synthetic_group_tail=synthetic_group_tail)
    altair_chart_stretch(st, wf_chart, width="stretch")
    _safe_exec(
        "waterfall",
        {
            "metric": value_col,
            "cat1": group_col,
            "top_item": str(wf.iloc[0]["label"]) if not wf.empty else None,
            "top_item_value": float(wf.iloc[0]["value"]) if not wf.empty else None,
            "top_item_share_pct": float(wf.iloc[0]["share"] * 100.0) if not wf.empty else None,
            "n_items": int(wf.shape[0]),
        },
    )
    _safe_guidance(
        "Identyfikuje główne dźwignie wyniku i porządkuje priorytety działań.",
        "Największe wkłady to segmenty o największym wpływie na total.",
        "Przy silnym ogonie rozdziel strategię dla top segmentów i long taila.",
    )
    st.divider()

    # 3. Pareto
    st.markdown("### Czy wartość sprzedaży jest skoncentrowana?")
    st.caption("Pareto pokazuje, czy większość wartości generuje niewielka liczba segmentów.")
    pareto = base.copy()
    pareto["group_label"] = pareto["group"].astype(str)
    max_bars = max(25, topn)
    pareto_vis = pareto.head(max_bars).copy()
    if len(pareto) > max_bars:
        rest_val = float(pareto.iloc[max_bars:]["value"].sum())
        rest_row = pd.DataFrame([{"group_label": synthetic_group_tail, "value": rest_val}])
        pareto_vis = pd.concat([pareto_vis, rest_row], ignore_index=True)
    pareto_total = float(pareto_vis["value"].sum() or 1.0)
    pareto_vis["rank"] = range(1, len(pareto_vis) + 1)
    pareto_vis["rank_label"] = pareto_vis["group_label"].astype(str)
    pareto_vis["cum_pct"] = (pareto_vis["value"] / pareto_total).cumsum() * 100.0
    p_n = int((pareto_vis["cum_pct"] >= cutoff * 100.0).idxmax() + 1) if (pareto_vis["cum_pct"] >= cutoff * 100.0).any() else int(pareto_vis.shape[0])
    pareto_bar = alt.Chart(pareto_vis).mark_bar(color="#1f77b4").encode(
        x=alt.X("rank_label:N", sort=None, title=None, axis=alt.Axis(labelAngle=-35, labelLimit=120)),
        y=alt.Y("value:Q", title="Wartość"),
        tooltip=[
            alt.Tooltip("group_label:N", title="Kategoria"),
            alt.Tooltip("value:Q", format=",.0f", title="Wartość"),
            alt.Tooltip("cum_pct:Q", format=".1f", title="Skumulowany udział %"),
        ],
    )
    pareto_line = alt.Chart(pareto_vis).mark_line(color="#D64550", point=True).encode(
        x=alt.X("rank_label:N", sort=None),
        y=alt.Y("cum_pct:Q", title="Skumulowany udział %", scale=alt.Scale(domain=[0, 105])),
    )
    pareto_rule = alt.Chart(pd.DataFrame({"threshold": [cutoff * 100.0]})).mark_rule(color="#D64550", strokeDash=[4, 4]).encode(
        y="threshold:Q"
    )
    altair_chart_stretch(st, (pareto_bar + pareto_line + pareto_rule).resolve_scale(y="independent").properties(height=430), width="stretch")
    _safe_exec(
        "pareto",
        {
            "metric": value_col,
            "cat1": group_col,
            "cutoff_pct": float(cutoff) * 100.0,
            "p_n": int(p_n),
            "n_segments": int(pareto_vis.shape[0]),
            "top_share_pct": float(pareto_vis.iloc[p_n - 1]["cum_pct"]) if p_n and (p_n - 1) < len(pareto_vis) else None,
        },
    )
    _safe_guidance(
        f"Szybko ocenia koncentrację i ryzyko zależności od top segmentów.",
        "Szybki wzrost krzywej na pierwszych segmentach oznacza silną koncentrację wartości.",
        "Przy silnej koncentracji buduj osobne strategie dla top segmentów i long taila.",
    )
    st.divider()

    # 4. Price corridor
    st.markdown("### Korytarz cenowy (Price corridor)")
    st.caption("Które przedziały cenowe generują większość wartości?")
    if price_col and price_col in df.columns and value_col in df.columns:
        corr, meta_bin = _build_price_corridor_from_stats(stats_payload)
        if corr.empty:
            st.info("Brak danych do analizy cenowej po filtrach.")
        else:
            idx80 = int(meta_bin.get("hi_idx", 0))
            lo_idx = int(meta_bin.get("lo_idx", 0))
            corridor_share = float(meta_bin.get("corridor_share_pct") or float(corr.loc[lo_idx:idx80, "share"].sum() * 100.0))
            lo_edge = meta_bin.get("corridor_low")
            hi_edge = meta_bin.get("corridor_high")
            p80_price = meta_bin.get("p80_price")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Cena P80 (80% wartości)", f"{p80_price:,.0f}" if p80_price is not None else "—")
            with c2:
                st.metric("Korytarz (20-80)", f"{lo_edge:,.0f} - {hi_edge:,.0f}" if lo_edge is not None and hi_edge is not None else "—")
            with c3:
                st.metric("Udział korytarza", f"{corridor_share:.1f}%")
            corr_plot = corr.copy()
            corr_plot["in_corridor"] = (corr_plot["x_i"] >= lo_idx) & (corr_plot["x_i"] <= idx80)
            price_bar = alt.Chart(corr_plot).mark_bar().encode(
                x=alt.X("price_bin:N", sort=None, title="Przedział ceny", axis=alt.Axis(labelAngle=-35, labelLimit=140)),
                y=alt.Y("value:Q", title="Wartość"),
                color=alt.condition(alt.datum.in_corridor, alt.value("#1f77b4"), alt.value("#d9d9d9")),
                tooltip=[
                    alt.Tooltip("price_bin:N", title="Przedział"),
                    alt.Tooltip("value:Q", format=",.0f", title="Wartość"),
                    alt.Tooltip("cum_pct:Q", format=".1f", title="Kumulacja %"),
                ],
            )
            price_line = alt.Chart(corr_plot).mark_line(color="#D64550", point=True).encode(
                x=alt.X("price_bin:N", sort=None),
                y=alt.Y("cum_pct:Q", title="Kumulacja (%)", scale=alt.Scale(domain=[0, 105])),
            )
            price_rule = alt.Chart(pd.DataFrame({"threshold": [80.0]})).mark_rule(color="#D64550", strokeDash=[4, 4]).encode(y="threshold:Q")
            altair_chart_stretch(st, (price_bar + price_line + price_rule).resolve_scale(y="independent").properties(height=450), width="stretch")
            _safe_exec(
                "price_corridor",
                {
                    "metric": value_col,
                    "price_col": price_col,
                    "p80_price": meta_bin.get("p80_price"),
                    "corridor_low": meta_bin.get("corridor_low"),
                    "corridor_high": meta_bin.get("corridor_high"),
                    "corridor_share_pct": meta_bin.get("corridor_share_pct"),
                    "bin_step": meta_bin.get("step"),
                },
            )
            _safe_guidance(
                "Identyfikuje przedziały cenowe, które generują większość wartości.",
                "Korytarz 20-80 wskazuje cenowy zakres największej koncentracji sprzedaży.",
                "Zapewnij dostępność i ekspozycję w korytarzu; ofertę premium traktuj osobno.",
            )
    else:
        st.info("Wybierz kolumnę ceny w filtrach CS, aby pokazać korytarz cenowy.")
    st.divider()

    # 5. Mix
    st.markdown("### Jak wygląda mix w ramach grup?")
    st.caption("100% stacked pokazuje udział składników w ramach każdego segmentu.")
    mix_agg = mix_exec_frame.copy() if isinstance(mix_exec_frame, pd.DataFrame) else pd.DataFrame()
    if (
        mix_agg.empty
        and group_col2
        and group_col2 in df.columns
        and group_col2 != group_col
        and group_col in df.columns
        and value_col in df.columns
    ):
        mix_agg = _build_mix_exec_frame(
            df=df,
            group_col=group_col,
            group_col2=group_col2,
            value_col=value_col,
            top_n=topn,
            top_k=7,
        )
        if mix_agg.empty:
            mix_agg = _build_resilient_mix_frame(
                df=df,
                group_col=group_col,
                group_col2=group_col2,
                value_col=value_col,
                top_n=topn,
                top_k=7,
            )
    if group_col2 and group_col2 in df.columns and group_col2 != group_col and not mix_agg.empty:
        mix_agg["group_label"] = mix_agg[group_col].astype(str)
        mix_agg["subgroup_label"] = mix_agg[group_col2].astype(str)
        mix_chart = (
            alt.Chart(mix_agg)
            .mark_bar()
            .encode(
                y=alt.Y("group_label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                x=alt.X("pct:Q", stack="zero", title="Udział %", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("subgroup_label:N", title=str(group_col2)),
                tooltip=[
                    alt.Tooltip("group_label:N", title="Grupa"),
                    alt.Tooltip("subgroup_label:N", title=str(group_col2)),
                    alt.Tooltip("value:Q", format=",.0f", title="Wartość"),
                    alt.Tooltip("pct:Q", format=".1f", title="Udział w grupie %"),
                    alt.Tooltip("area_share_pct:Q", format=".1f", title="Wkład do totalu %"),
                ],
            )
            .properties(height=min(520, max(320, 70 + 28 * int(mix_agg[group_col].nunique()))))
        )
        altair_chart_stretch(st, mix_chart, width="stretch")
    else:
        st.info("Włącz Kategorię 2 w sidebarze albo wybierz parę kategorii z dodatnimi wartościami, aby zobaczyć mix.")
    _safe_exec("mix", {"metric": value_col, "cat1": group_col, "cat2": group_col2, "top_n": int(top_n)})
    _safe_guidance(
        "Porównuje struktury wewnętrzne między segmentami.",
        "Różne proporcje składników oznaczają różne profile kategorii.",
        "Ogranicz składniki do Top-7 + Other, aby wykres był czytelny.",
    )
    st.divider()

    # 6. Marimekko equivalent
    st.markdown("### Marimekko (PRO): skala x struktura")
    st.caption("Szerokość segmentu = udział grupy w totalu, wysokość koloru = udział składnika w grupie.")
    if group_col2 and group_col2 in df.columns and group_col2 != group_col and not mix_agg.empty:
        mm_rects = _build_marimekko_rects(mix_agg, group_col, group_col2)
        if mm_rects.empty:
            st.info("Brak dodatnich wartości do zbudowania Marimekko.")
        else:
            mm_base = (
                alt.Chart(mm_rects)
                .mark_rect(stroke="white", strokeWidth=1)
                .encode(
                    x=alt.X("x0:Q", title="Udział grupy w totalu (%)", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(format=".0f")),
                    x2="x1:Q",
                    y=alt.Y("y0:Q", title="Struktura w grupie (%)", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(format=".0f")),
                    y2="y1:Q",
                    color=alt.Color("subgroup_label:N", title=str(group_col2), legend=alt.Legend(orient="bottom", columns=4)),
                    tooltip=[
                        alt.Tooltip("group_label:N", title="Grupa"),
                        alt.Tooltip("subgroup_label:N", title=str(group_col2)),
                        alt.Tooltip("value:Q", format=",.0f", title="Wartość"),
                        alt.Tooltip("width_pct:Q", format=".1f", title="Udział grupy w totalu %"),
                        alt.Tooltip("height_pct:Q", format=".1f", title="Udział składnika w grupie %"),
                        alt.Tooltip("area_share_pct:Q", format=".1f", title="Wkład komórki do totalu %"),
                    ],
                )
            )
            mm_labels = (
                alt.Chart(mm_rects[mm_rects["display_label"].astype(str) != ""])
                .mark_text(color="#111827", fontSize=11, fontWeight="bold")
                .encode(
                    x=alt.X("label_x:Q", scale=alt.Scale(domain=[0, 100]), axis=None),
                    y=alt.Y("label_y:Q", scale=alt.Scale(domain=[0, 100]), axis=None),
                    text="display_label:N",
                )
            )
            altair_chart_stretch(st, (mm_base + mm_labels).properties(height=430).configure_view(strokeOpacity=0), width="stretch")
    else:
        st.info("Wybierz Kategorię 2 albo parę kategorii z dodatnimi wartościami, aby zbudować widok Marimekko.")
    _safe_exec("marimekko", {"metric": value_col, "cat1": group_col, "cat2": group_col2, "top_n": int(top_n)})
    _safe_guidance(
        "Pokazuje jednocześnie skalę segmentu i jego strukturę.",
        "Najciemniejsze komórki wskazują największy wkład kombinacji grupa x składnik.",
        "Ogranicz liczbę segmentów i składników, aby wykres był czytelny.",
    )

    return {
        "chart_meta": {"kind": "composition_static", "plotly_available": False, "altair_full_fallback": True},
        "chart_context": {
            "group_col": group_col,
            "group_col2": group_col2,
            "value_col": value_col,
            "price_col": price_col,
            "stats_payload": stats_payload if isinstance(stats_payload, dict) else {},
        },
    }


def render(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
    tabs = ctx.get("tabs") or {}
    render_overview = bool(ctx.get("render_overview", True))
    render_insights = bool(ctx.get("render_insights", True))
    filters_signature = ctx.get("filters") or {}

    # ── CS sidebar params (source of truth) ──────────────────────────────────
    group_col_sel  = ctx.get("cs_group_col")   # Kategoria 1 (selected)
    group_col2_sel = ctx.get("cs_group_col2")  # Kategoria 2 (selected)
    top_n          = min(max(5, int(ctx.get("cs_top_n") or 10)), 50)
    cutoff         = float(ctx.get("cs_cutoff") or 0.80)
    price_col_sel  = ctx.get("cs_price_col")   # optional (price corridor)
    runtime_signature = _make_cs_runtime_signature(df, filters_signature)

    # ── metric slots z main (st.empty()) ────────────────────────────────────
    ui_ov       = ((ctx.get("ui") or {}).get("overview") or {})
    metric_val  = ui_ov.get("metric_value")
    metric_txn  = ui_ov.get("metric_txn")
    chart_slot  = ui_ov.get("chart_slot")
    # ─────────────────────────────────────────────
    # Executive takeaway (McKinsey Headline) + per-block guidance
    question = str((ctx.get("question") or "")).strip()
    # Ensure the engine sees the question (batch + single-call paths rely on ctx["question"])
    ctx["question"] = question

    # Batch LLM: one request -> many takeaways keyed by block.label
    _dc_llm_text = st.session_state.get("dc_llm_text")
    if callable(_dc_llm_text) and not callable(ctx.get("llm_text")):
        ctx["llm_text"] = _dc_llm_text
    _dc_llm_status = st.session_state.get("dc_llm_status_v1") or {}
    if isinstance(_dc_llm_status, dict) and not ctx.get("openai_model"):
        _model = str(_dc_llm_status.get("model") or "").strip()
        if _model:
            ctx["openai_model"] = _model
    debug = bool(ctx.get("debug", False))
    llm_takeaway_fn = None
    record_debug_checkpoint(
        "cs.render.start",
        question=question,
        group_col_sel=group_col_sel,
        group_col2_sel=group_col2_sel,
        price_col_sel=price_col_sel,
        top_n=top_n,
        cutoff=cutoff,
        render_overview=render_overview,
        render_insights=render_insights,
        llm_text_ready=bool(callable(ctx.get("llm_text"))),
        openai_model=ctx.get("openai_model"),
    )

    # Keep a shared takeaways dict (router contract)
    if not isinstance(ctx.get("takeaways"), dict):
        ctx["takeaways"] = {}

    def _exec_takeaway(key: str, anchors: Dict[str, Any], question: Optional[str] = None, *, min_chars: int = 70) -> Optional[str]:
        """Render McKinsey-grade executive takeaway with numeric anchors + quality gates.
        Uses the same engine as Distribution (get_exec_takeaway + optional LLM).
        """
        question = str((question or ctx.get('question') or '')).strip()
        # 0) Prefer batch LLM takeaways if available (one call per branch).
        cs_map = ctx.get("cs_takeaways")
        if isinstance(cs_map, dict):
            _t = cs_map.get(key)
            if isinstance(_t, str) and _t.strip():
                injected = _t.strip()
                cs_src_map = ctx.get("cs_takeaways_src") or {}
                injected_src = str((cs_src_map.get(key) if isinstance(cs_src_map, dict) else "") or "llm_batch_direct")
                blk = {"key": key, "exec": injected, "_exec_source": "llm"}
                txt2, _dbg2 = get_exec_takeaway(
                    intent=f"composition_static:{key}",
                    block=blk,
                    stats={**(anchors or {})},
                    question=question,
                    session_state=st.session_state,
                    llm_fn=None,
                )
                txt2 = (txt2 or "").strip()
                if txt2:
                    ctx.setdefault("takeaways", {})[key] = txt2
                    record_debug_checkpoint(
                        "cs.exec.block",
                        block_id=key,
                        src=injected_src,
                        candidate_len=len(injected),
                        final_len=len(txt2),
                    )
                    render_exec_takeaway(txt2)
                    return txt2

        if "takeaways" not in ctx or not isinstance(ctx.get("takeaways"), dict):
            ctx["takeaways"] = {}
        intent = f"composition_static:{key}"
        block = {"key": key, "anchors": anchors or {}}
        stats: Dict[str, Any] = {}

        # stats_payload może pochodzić z routera (ctx) – jeśli nie ma, użyj pustego
        _stats_payload = ctx.get("stats_payload")
        if isinstance(_stats_payload, dict):
            stats.update(_stats_payload)

        if anchors:
            stats.update(anchors)
        # Delegate to shared exec-takeaway engine (same as Distribution).
        # (Hard contract) per-block LLM is handled only via batch call at top; no extra calls here.

        txt, _dbg = get_exec_takeaway(
            intent=intent,
            block=block,
            stats=stats,
            question=question,
            session_state=st.session_state,
            llm_fn=llm_takeaway_fn,
        )
        # ET v1.0 contract: NIE Ucinamy tekstu lokalnie.
        # Jeśli tekst jest za długi, to ma się zawijać w UI (wrap), a nie kończyć "…".
        if isinstance(txt, str):
            txt = txt.strip()
            if min_chars and len(txt) < min_chars:
                txt = None
        if txt:
            ctx["takeaways"][key] = txt
            record_debug_checkpoint(
                "cs.exec.block",
                block_id=key,
                src="deterministic_fallback",
                final_len=len(txt),
            )
            st.markdown(
                f"""<div style="border-left:4px solid #ff4b4b; padding:10px 12px; margin:8px 0 10px 0; background:#fff; border-radius:8px;">
                <div style="font-size:13px; color:#111;"><b>Executive takeaway</b></div>
                <div style="font-size:14px; color:#111; margin-top:4px; white-space:normal; overflow-wrap:anywhere; word-break:break-word;">{txt}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            return txt
        st.markdown("🧠 **Executive takeaway:** —")
        return None


    tab_overview = tabs.get("overview") if render_overview else None
    tab_insights = tabs.get("insights") if render_insights else None

    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Brak danych do analizy.")
        return {"chart_meta": {"kind": "composition_static"}, "chart_context": {}}

    df = _sanitize_df_columns(df)

    # ── value_col: heurystyka (bez hardcoded nazwy kolumny) ──────────────────
    value_col = _infer_value_col(df)

    # ── resolve group_col selection ──────────────────────────────────────────
    if False:
        group_col_sel = _infer_best_group_col(df)

    # ── PRICE MODE: numeric continuous selected OR explicit price_col ────────
    price_col: Optional[str] = None
    if price_col_sel and isinstance(price_col_sel, str) and price_col_sel in df.columns:
        price_col = price_col_sel

    group_col = group_col_sel
    price_mode = False
    if group_col and group_col in df.columns and _is_numeric_continuous(df[group_col]):
        price_mode = True
        price_col = price_col or group_col
        # choose a better structural category (so we don't explode mix/marimekko)
        group_col = _infer_best_group_col(df, exclude={price_col}) or group_col_sel  # fallback to original if nothing else

    # If still no group_col, fallback to any reasonable
    if not group_col or group_col not in df.columns:
        group_col = _infer_best_group_col(df) or group_col_sel

    # ── group_col2 gating ───────────────────────────────────────────────────
    group_col2: Optional[str] = None
    if group_col2_sel and group_col2_sel in df.columns and group_col2_sel != group_col:
        # disallow continuous numeric as category 2 (causes cognitive + technical issues)
        if pd.api.types.is_numeric_dtype(df[group_col2_sel]) and _is_numeric_continuous(df[group_col2_sel]):
            group_col2 = None
        else:
            group_col2 = group_col2_sel

    # ── quick UX note (only once per render) ────────────────────────────────
    if False and price_mode:
        st.info(
            f"Wybrana kolumna **{group_col_sel}** wygląda na cenę (zmienna ciągła). "
            f"Używam jej do analizy **Korytarza cenowego**, a strukturę segmentów buduję na **{group_col}**. "
            f"To chroni Mix/Marimekko przed eksplozją kategorii."
        )

    # ── KPI ─────────────────────────────────────────────────────────────────
    resolved = resolve_grouping_selection(df, group_col_sel, group_col2_sel, price_col_sel)
    group_col = resolved.get("group_col")
    group_col2 = resolved.get("group_col2")
    price_col = resolved.get("price_col")
    price_mode = bool(resolved.get("price_mode"))

    stats_payload: Dict[str, Any] = {}
    ctx["stats_payload"] = stats_payload
    record_debug_checkpoint(
        "cs.render.resolved",
        group_col=group_col,
        group_col2=group_col2,
        value_col=value_col,
        price_col=price_col,
        price_mode=price_mode,
        price_source=resolved.get("price_source"),
        stats_deferred=True,
        stats_error=None,
        stats_keys=[],
    )
    st.session_state["datachat_cs_render_state_v3"] = {
        "sidebar_signature": (group_col_sel, group_col2_sel, price_col_sel, int(top_n)),
        "group_col": group_col,
        "group_col2": group_col2,
        "price_col": price_col,
        "value_col": value_col,
        "price_source": resolved.get("price_source"),
        "stats_payload": {},
    }

    _mix_exec_frame: pd.DataFrame = pd.DataFrame()

    def _ensure_cs_takeaways() -> None:
        if isinstance(ctx.get("cs_takeaways"), dict) and ctx.get("cs_takeaways"):
            return
        try:
            _mix_exec_stats = _build_mix_exec_stats_from_frame(_mix_exec_frame, group_col, group_col2)
            _marimekko_exec_stats = _build_marimekko_exec_stats_from_frame(_mix_exec_frame, group_col, group_col2)
            _price_corridor_exec_stats = _build_price_corridor_exec_stats_from_payload(stats_payload)
            record_debug_checkpoint(
                "cs.exec.block_specific_stats",
                mix_keys=list(_mix_exec_stats.keys()),
                marimekko_keys=list(_marimekko_exec_stats.keys()),
                price_corridor_keys=list(_price_corridor_exec_stats.keys()),
            )
            _common_stats = {
                "top_n": top_n,
                "cutoff": cutoff,
                "group_col": group_col,
                "group_col2": group_col2,
                "price_col": price_col,
                "value_col": value_col,
                "n_groups": stats_payload.get("n_groups") if isinstance(stats_payload, dict) else None,
                "top1_share_pct": stats_payload.get("top1_share_pct") if isinstance(stats_payload, dict) else None,
                "hhi": stats_payload.get("hhi") if isinstance(stats_payload, dict) else None,
                "pareto_80_n": stats_payload.get("pareto_80_n") if isinstance(stats_payload, dict) else None,
            }
            _cs_blocks = [
                {"label": "ranking", "title": "Ranking (wartosc)", "desc": "Porownanie skali segmentow (wartosc absolutna).", "stats": {**_common_stats, "block": "ranking"}},
                {"label": "waterfall", "title": "Waterfall (wklad do total)", "desc": "Wklad segmentow do lacznej wartosci (Top-N + reszta).", "stats": {**_common_stats, "block": "waterfall"}},
                {"label": "pareto", "title": "Pareto / koncentracja", "desc": "Koncentracja wartosci na top segmentach (udzial skumulowany).", "stats": {**_common_stats, "block": "pareto"}},
                {
                    "label": "price_corridor",
                    "title": "Korytarz cenowy",
                    "desc": "Przedzialy cen generujace wiekszosc wartosci. Oprzyj takeaway na korytarzu 20-80, progu P80 i najsilniejszym binie cenowym.",
                    "stats": {**_common_stats, "block": "price_corridor", **_price_corridor_exec_stats},
                },
                {
                    "label": "mix",
                    "title": "Mix w ramach grup",
                    "desc": "100% stacked mix: porownaj sklad wewnetrzny najwiekszych grup. Uzyj dominujacego skladnika w focus_group, luki do drugiego skladnika i najwiekszej komorki group x component.",
                    "stats": {**_common_stats, "block": "mix", **_mix_exec_stats},
                },
                {
                    "label": "marimekko",
                    "title": "Marimekko (PRO)",
                    "desc": "Marimekko: width = udzial grupy w totalu, height = udzial skladnika w grupie, area = wklad komorki do totalu. Oprzyj takeaway na najszerszej grupie i najwiekszej komorce.",
                    "stats": {**_common_stats, "block": "marimekko", **_marimekko_exec_stats},
                },
            ]
            _stats_signature = {
                "exec_semantics_version": "cs_exec_semantic_repair_v1",
                "exec_prompt_version": "cs_exec_mix_marimekko_v2",
                "exec_intent": "composition_static_key_insight",
                "group_col": group_col,
                "group_col2": group_col2,
                "price_col": price_col,
                "value_col": value_col,
                "top_n": top_n,
                "cutoff": cutoff,
                "n_groups": stats_payload.get("n_groups") if isinstance(stats_payload, dict) else None,
                "top1_share_pct": stats_payload.get("top1_share_pct") if isinstance(stats_payload, dict) else None,
                "pareto_80_n": stats_payload.get("pareto_80_n") if isinstance(stats_payload, dict) else None,
                "hhi": stats_payload.get("hhi") if isinstance(stats_payload, dict) else None,
                "price_corridor_p80": ((stats_payload.get("price_corridor") or {}).get("p80_price") if isinstance(stats_payload, dict) else None),
                "price_corridor_share_pct": ((stats_payload.get("price_corridor") or {}).get("corridor_share_pct") if isinstance(stats_payload, dict) else None),
                "price_corridor_exec_stats": _price_corridor_exec_stats,
                "mix_exec_stats": _mix_exec_stats,
                "marimekko_exec_stats": _marimekko_exec_stats,
            }
            _takeaways_cache_key = hashlib.md5(
                json.dumps(_stats_signature, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()
            _cached_key = st.session_state.get("cs_takeaways_cache_key")
            _cached_takeaways = st.session_state.get("cs_takeaways_cache")

            if _takeaways_cache_key == _cached_key and isinstance(_cached_takeaways, dict):
                ctx["cs_takeaways"] = {str(k): str(v) for k, v in _cached_takeaways.items() if v}
                _cached_takeaway_src = st.session_state.get("cs_takeaways_src_cache")
                if isinstance(_cached_takeaway_src, dict):
                    ctx["cs_takeaways_src"] = {str(k): str(v) for k, v in _cached_takeaway_src.items() if v}
                else:
                    ctx["cs_takeaways_src"] = {str(k): "llm_batch_direct" for k in ctx["cs_takeaways"].keys()}
                record_debug_checkpoint(
                    "cs.exec.cache_hit",
                    cache_key=_takeaways_cache_key,
                    blocks_count=len(ctx["cs_takeaways"]),
                )
            elif isinstance(stats_payload, dict) and stats_payload.get("error"):
                ctx["cs_takeaways"] = {}
                record_debug_checkpoint(
                    "cs.exec.skipped",
                    reason=stats_payload.get("error"),
                )
            else:
                if not callable(ctx.get("llm_text")):
                    ctx["cs_takeaways"] = {}
                    record_debug_checkpoint(
                        "cs.exec.skipped",
                        reason="llm_text_missing_in_ctx",
                        llm_status=st.session_state.get("dc_llm_status_v1"),
                    )
                else:
                    _t0 = time.perf_counter()
                    _exec_by_label = get_exec_takeaways_llm(ctx=ctx, intent="composition_static_key_insight", blocks=_cs_blocks)
                    _elapsed_ms = round((time.perf_counter() - _t0) * 1000.0, 2)
                    if isinstance(_exec_by_label, dict):
                        _block_stats_by_label = {
                            str(block.get("label") or ""): (block.get("stats") or {})
                            for block in _cs_blocks
                            if isinstance(block, dict)
                        }
                        _takeaways_clean: Dict[str, str] = {}
                        _takeaways_src: Dict[str, str] = {}
                        for _label, _text in _exec_by_label.items():
                            _label_s = str(_label or "").strip()
                            _text_s = str(_text or "").strip()
                            if not _label_s or not _text_s:
                                continue
                            _fixed_text, _used_repair = _repair_cs_batch_takeaway(
                                _label_s,
                                _text_s,
                                _block_stats_by_label.get(_label_s) or {},
                            )
                            if not _fixed_text:
                                continue
                            _takeaways_clean[_label_s] = _fixed_text
                            _takeaways_src[_label_s] = "llm_batch_direct_repaired" if _used_repair else "llm_batch_direct"
                        if "price_corridor" not in _takeaways_clean:
                            _corridor_repair = _build_price_corridor_exec_repair(
                                _block_stats_by_label.get("price_corridor") or {}
                            )
                            if _corridor_repair:
                                _takeaways_clean["price_corridor"] = _corridor_repair
                                _takeaways_src["price_corridor"] = "llm_batch_direct_repaired"
                                record_debug_checkpoint(
                                    "cs.exec.block_repaired_missing",
                                    block_id="price_corridor",
                                    src="llm_batch_direct_repaired",
                                )
                        ctx["cs_takeaways"] = _takeaways_clean
                        ctx["cs_takeaways_src"] = _takeaways_src
                        st.session_state["cs_takeaways_cache_key"] = _takeaways_cache_key
                        st.session_state["cs_takeaways_cache"] = dict(ctx["cs_takeaways"])
                        st.session_state["cs_takeaways_src_cache"] = dict(ctx.get("cs_takeaways_src") or {})
                        record_debug_checkpoint(
                            "cs.exec.cache_store",
                            cache_key=_takeaways_cache_key,
                            blocks_count=len(ctx["cs_takeaways"]),
                            elapsed_ms=_elapsed_ms,
                        )
                        if not ctx["cs_takeaways"]:
                            record_debug_checkpoint(
                                "cs.exec.empty_result",
                                cache_key=_takeaways_cache_key,
                                elapsed_ms=_elapsed_ms,
                                blocks_requested=len(_cs_blocks),
                                llm_status=st.session_state.get("dc_llm_status_v1"),
                                helper_debug=ctx.get("_exec_takeaway_llm_last_error"),
                            )
        except Exception as e:
            ctx["cs_takeaways"] = {}
            record_debug_checkpoint(
                "cs.exec.error",
                error=f"{type(e).__name__}: {e}",
            )
            if debug:
                try:
                    st.session_state.setdefault("dc_errors", []).append({
                        "where": "composition_static.batch_takeaways",
                        "error": type(e).__name__,
                        "msg": str(e),
                    })
                except Exception:
                    pass

    if price_mode and group_col:
        st.info(
            f"Wybrana kolumna **{group_col_sel}** wygląda na cenę (zmienna ciągła). "
            f"Używam jej do analizy **Korytarza cenowego**, a strukturę segmentów buduję na **{group_col}**. "
            f"To chroni Mix/Marimekko przed eksplozją kategorii."
        )

    metric_qty = ui_ov.get("metric_qty")

    if render_overview:
        if metric_val is not None:
            if value_col and value_col in df.columns:
                total_value = float(pd.to_numeric(df[value_col], errors="coerce").fillna(0).clip(lower=0).sum())
                metric_val.metric("Suma wartości", f"{total_value:,.0f}".replace(",", " "))
            else:
                metric_val.metric("Suma wartości", "—")

        if metric_qty is not None:
            _qty_col = None
            for _hint in ["quantity", "qty", "count", "units", "ilosc", "ilość", "sztuk"]:
                for _c in df.columns:
                    if _hint in str(_c).lower() and pd.api.types.is_numeric_dtype(df[_c]):
                        _qty_col = _c
                        break
                if _qty_col:
                    break

            if _qty_col and _qty_col in df.columns:
                _qty_total = float(pd.to_numeric(df[_qty_col], errors="coerce").fillna(0).clip(lower=0).sum())
                metric_qty.metric("Suma ilości", f"{_qty_total:,.0f}".replace(",", " "))
            else:
                metric_qty.metric("Suma ilości", "—")

        if metric_txn is not None:
            metric_txn.metric("Liczba transakcji", f"{len(df):,}".replace(",", " "))

        _chart_target = chart_slot if chart_slot is not None else (tab_overview or st)
        with _chart_target:
            if group_col and value_col and group_col in df.columns and value_col in df.columns:
                try:
                    treemap_cache_key = hashlib.md5(
                        json.dumps(
                            {
                                "runtime": runtime_signature,
                                "group_col": group_col,
                                "group_col2": group_col2,
                                "value_col": value_col,
                                "top_n": int(top_n or 10),
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()
                    cached_treemap = _cs_cache_get("cs_treemap_frame_cache", treemap_cache_key)
                    if isinstance(cached_treemap, dict):
                        agg = cached_treemap.get("frame", pd.DataFrame()).copy()
                        use_two_levels = bool(cached_treemap.get("use_two_levels"))
                    else:
                        agg, use_two_levels = _build_treemap_frame(
                            df=df,
                            group_col=group_col,
                            group_col2=group_col2,
                            value_col=value_col,
                            top_n=top_n,
                        )
                        _cs_cache_put(
                            "cs_treemap_frame_cache",
                            treemap_cache_key,
                            {"frame": agg, "use_two_levels": use_two_levels},
                        )

                    if not agg.empty:
                        if _render_controlled_treemap(agg, use_two_levels=use_two_levels):
                            pass
                        elif px is None:
                            record_debug_checkpoint("cs.overview.plotly_express_missing")
                            _render_altair_overview_composition(agg, use_two_levels=use_two_levels, legend_title=str(group_col or "Kategoria"))
                        else:
                            agg = (
                                agg.groupby("group", dropna=False, sort=False)["value"]
                                .sum()
                                .reset_index()
                                .sort_values("value", ascending=False)
                                .reset_index(drop=True)
                            )
                            path = ["group"]
                            fig = px.treemap(
                                agg,
                                path=path,
                                values="value",
                                color="group",
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                            )
                            fig.update_traces(
                                sort=False,
                                textinfo="label+percent root",
                                textposition="top left",
                                textfont=dict(color="#111827", size=14),
                                insidetextfont=dict(color="#111827", size=14),
                                outsidetextfont=dict(color="#111827", size=14),
                                marker=dict(line=dict(color="#ffffff", width=2)),
                                tiling=dict(packing="squarify"),
                                hovertemplate="<b>%{label}</b><br>Wartość: %{value:,.0f}<br>Udział: %{percentRoot:.1%}<extra></extra>",
                            )
                            fig.update_layout(
                                margin=dict(l=0, r=0, t=0, b=0),
                                height=520,
                                uniformtext=dict(minsize=10, mode="hide"),
                            )
                            plotly_chart_stretch(st, fig, config={"displayModeBar": False})
                except Exception:
                    st.warning("Nie udało się wyrenderować treemapy dla wybranych ustawień.")
            else:
                st.info("Brak kolumny grupowej lub wartościowej — wybierz w sidebarze inną kategorię lub miarę.")

    if not render_insights:
        return {"chart_meta": {"kind": "composition_static"}, "chart_context": {}}

    stats_cache_key = hashlib.md5(
        json.dumps(
            {
                "runtime": runtime_signature,
                "group_col": group_col,
                "group_col2": group_col2,
                "value_col": value_col,
                "top_n": int(top_n or 10),
                "price_col": price_col,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    cached_stats_payload = _cs_cache_get("cs_stats_payload_cache", stats_cache_key)
    if isinstance(cached_stats_payload, dict):
        stats_payload = cached_stats_payload
    else:
        stats_payload = build_stats_payload(
            df=df,
            group_col=group_col,
            value_col=value_col,
            group_col2=group_col2,
            top_n=top_n,
            price_col=price_col,
        ) if group_col and value_col else {}
        _cs_cache_put("cs_stats_payload_cache", stats_cache_key, stats_payload)
    ctx["stats_payload"] = stats_payload
    st.session_state["datachat_cs_render_state_v3"] = {
        "sidebar_signature": (group_col_sel, group_col2_sel, price_col_sel, int(top_n)),
        "group_col": group_col,
        "group_col2": group_col2,
        "price_col": price_col,
        "value_col": value_col,
        "price_source": resolved.get("price_source"),
        "stats_payload": stats_payload if isinstance(stats_payload, dict) else {},
    }
    record_debug_checkpoint(
        "cs.render.stats_ready",
        stats_error=stats_payload.get("error") if isinstance(stats_payload, dict) else None,
        stats_keys=list(stats_payload.keys()) if isinstance(stats_payload, dict) else [],
    )

    if (
        isinstance(df, pd.DataFrame)
        and not df.empty
        and group_col
        and group_col2
        and value_col
        and group_col in df.columns
        and group_col2 in df.columns
        and value_col in df.columns
        and group_col != group_col2
    ):
        mix_cache_key = hashlib.md5(
            json.dumps(
                {
                    "runtime": runtime_signature,
                    "group_col": group_col,
                    "group_col2": group_col2,
                    "value_col": value_col,
                    "top_n": int(top_n or 10),
                    "top_k": 7,
                },
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        cached_mix_exec_frame = _cs_cache_get("cs_mix_exec_frame_cache", mix_cache_key)
        if isinstance(cached_mix_exec_frame, pd.DataFrame):
            _mix_exec_frame = cached_mix_exec_frame
        else:
            _mix_exec_frame = _build_mix_exec_frame(
                df=df,
                group_col=group_col,
                group_col2=group_col2,
                value_col=value_col,
                top_n=top_n,
                top_k=7,
            )
            if _mix_exec_frame.empty:
                _mix_exec_frame = _build_resilient_mix_frame(
                    df=df,
                    group_col=group_col,
                    group_col2=group_col2,
                    value_col=value_col,
                    top_n=top_n,
                    top_k=7,
                )
            _cs_cache_put("cs_mix_exec_frame_cache", mix_cache_key, _mix_exec_frame)

    # Batch ET only when the user enters Kluczowe insighty.
    _ensure_cs_takeaways()

    # ── TAB INSIGHTS: blocks ────────────────────────────────────────────────
    with tab_insights:
        if not (group_col and value_col and group_col in df.columns and value_col in df.columns):
            st.info("Brak kolumny grupowej lub wartościowej — zmień dane lub wybierz kategorię w sidebarze.")
            return {"chart_meta": {"kind": "composition_static"}, "chart_context": {}}

        try:
            import plotly.graph_objects as go  # type: ignore
        except Exception as exc:
            record_debug_checkpoint("cs.plotly_missing", error=f"{type(exc).__name__}: {exc}")
            def _altair_guidance(sens: str, interp: str, best: str) -> None:
                render_guidance(
                    sens=sens,
                    interp=interp,
                    best=best,
                    title="Guidance",
                    expanded=False,
                )

            return _render_altair_composition_static_insights(
                df=df,
                stats_payload=stats_payload if isinstance(stats_payload, dict) else {},
                group_col=group_col,
                group_col2=group_col2,
                value_col=value_col,
                price_col=price_col,
                top_n=top_n,
                cutoff=cutoff,
                mix_exec_frame=_mix_exec_frame,
                exec_takeaway_fn=_exec_takeaway,
                guidance_fn=_altair_guidance,
            )
            if group_col and value_col and group_col in df.columns and value_col in df.columns:
                try:
                    fallback_agg = (
                        df[[group_col, value_col]]
                        .assign(**{value_col: pd.to_numeric(df[value_col], errors="coerce")})
                        .dropna(subset=[group_col, value_col])
                        .groupby(group_col, dropna=False)[value_col]
                        .sum()
                        .sort_values(ascending=False)
                        .head(int(top_n or 10))
                        .reset_index()
                    )
                    fallback_agg["group_label"] = fallback_agg[group_col].astype(str)
                    total = float(fallback_agg[value_col].sum() or 0.0)
                    fallback_agg["share"] = fallback_agg[value_col] / total if total else 0.0

                    st.markdown("### Ranking wartości według kategorii")
                    rank_chart = (
                        alt.Chart(fallback_agg)
                        .mark_bar()
                        .encode(
                            y=alt.Y(
                                "group_label:N",
                                sort="-x",
                                title=None,
                                axis=alt.Axis(labelLimit=260),
                            ),
                            x=alt.X(f"{value_col}:Q", title="Wartość"),
                            tooltip=[
                                alt.Tooltip("group_label:N", title="Kategoria"),
                                alt.Tooltip(f"{value_col}:Q", format=",.0f", title="Wartość"),
                                alt.Tooltip("share:Q", format=".1%", title="Udział"),
                            ],
                        )
                        .properties(height=min(520, max(280, 56 + 28 * int(fallback_agg.shape[0]))))
                    )
                    altair_chart_stretch(st, rank_chart, width="stretch")

                    _exec_takeaway(
                        "ranking",
                        anchors={
                            "metric": value_col,
                            "cat1": group_col,
                            "top_segment": str(fallback_agg.iloc[0].get(group_col)) if not fallback_agg.empty else None,
                            "top_value": float(fallback_agg.iloc[0].get(value_col, 0.0)) if not fallback_agg.empty else None,
                            "top_share_pct": float(fallback_agg.iloc[0].get("share", 0.0)) * 100.0 if not fallback_agg.empty else None,
                            "n_segments": int(fallback_agg.shape[0]),
                        },
                    )

                    if group_col2 and group_col2 in df.columns and group_col2 != group_col:
                        mix_source = (
                            df[[group_col, group_col2, value_col]]
                            .assign(**{value_col: pd.to_numeric(df[value_col], errors="coerce")})
                            .dropna(subset=[group_col, group_col2, value_col])
                        )
                        top_groups = set(fallback_agg[group_col].astype(str).tolist())
                        mix_source = mix_source[mix_source[group_col].astype(str).isin(top_groups)]
                        if not mix_source.empty:
                            mix_agg = (
                                mix_source
                                .groupby([group_col, group_col2], dropna=False)[value_col]
                                .sum()
                                .reset_index()
                            )
                            sub_totals = mix_agg.groupby(group_col2, dropna=False)[value_col].sum().sort_values(ascending=False)
                            top_subs = set(sub_totals.head(8).index.tolist())
                            mix_agg["subgroup"] = mix_agg[group_col2].where(mix_agg[group_col2].isin(top_subs), "Pozostale")
                            mix_agg = (
                                mix_agg
                                .groupby([group_col, "subgroup"], dropna=False)[value_col]
                                .sum()
                                .reset_index()
                            )
                            mix_agg["group_label"] = mix_agg[group_col].astype(str)
                            mix_agg["group_total"] = mix_agg.groupby(group_col, dropna=False)[value_col].transform("sum")
                            mix_agg["pct"] = np.where(
                                mix_agg["group_total"] > 0,
                                mix_agg[value_col] / mix_agg["group_total"],
                                0.0,
                            )

                            st.markdown("### Mix wewnątrz kategorii")
                            mix_chart = (
                                alt.Chart(mix_agg)
                                .mark_bar()
                                .encode(
                                    y=alt.Y(
                                        "group_label:N",
                                        sort=list(fallback_agg["group_label"]),
                                        title=None,
                                        axis=alt.Axis(labelLimit=260),
                                    ),
                                    x=alt.X("pct:Q", stack="zero", title="Udział w kategorii", axis=alt.Axis(format="%")),
                                    color=alt.Color("subgroup:N", title=str(group_col2)),
                                    tooltip=[
                                        alt.Tooltip("group_label:N", title="Kategoria"),
                                        alt.Tooltip("subgroup:N", title=str(group_col2)),
                                        alt.Tooltip(f"{value_col}:Q", format=",.0f", title="Wartość"),
                                        alt.Tooltip("pct:Q", format=".1%", title="Udział w kategorii"),
                                    ],
                                )
                                .properties(height=min(520, max(280, 56 + 28 * int(fallback_agg.shape[0]))))
                            )
                            altair_chart_stretch(st, mix_chart, width="stretch")

                            _exec_takeaway(
                                "mix",
                                anchors={
                                    "metric": value_col,
                                    "cat1": group_col,
                                    "cat2": group_col2,
                                    "top_n": int(top_n) if "top_n" in locals() else None,
                                },
                            )
                except Exception as fallback_exc:
                    record_debug_checkpoint(
                        "cs.altair_fallback_failed",
                        error=f"{type(fallback_exc).__name__}: {fallback_exc}",
                    )
                    try:
                        st.dataframe(fallback_agg, width="stretch")
                    except Exception:
                        pass
            return {
                "chart_meta": {"kind": "composition_static", "plotly_available": False, "altair_fallback": True},
                "chart_context": {
                    "group_col": group_col,
                    "group_col2": group_col2,
                    "value_col": value_col,
                    "price_col": price_col,
                    "stats_payload": stats_payload if isinstance(stats_payload, dict) else {},
                },
            }

        # takeaways dict (from main LLM cache)
        # fallback: allow also generic key name "takeaways" if used elsewhere
        _tw = dict(ctx.get("cs_takeaways") or ctx.get("takeaways") or {})

        def _guidance(sens: str, interp: str, best: str) -> None:
            # ✅ Use global Guidance v1.0 (same renderer + CSS contract as Distribution)
            render_guidance(
                sens=sens,
                interp=interp,
                best=best,
                title="Guidance",
                expanded=False,
            )

        def _takeaway(key: str) -> None:
            txt = str((_tw.get(key) or "")).strip()
            render_exec_takeaway(txt)

        # ── aggregation: FULL + Top-N + Other (shares over FULL) ────────────
        topn = min(max(5, int(top_n or 10)), 50)
        synthetic_group_tail = f"Pozostale (poza Top-{topn})"
        all_known_groups = {
            str(v) for v in (
                list((stats_payload or {}).get("top_labels") or [])
                + list((stats_payload or {}).get("display_labels") or [])
            )
        }
        if synthetic_group_tail in all_known_groups:
            synthetic_group_tail = f"{synthetic_group_tail} [ogon]"
        base, head, total_full, hhi = _build_cs_structure_frames_from_stats(
            stats_payload=stats_payload,
            top_n=topn,
            synthetic_group_tail=synthetic_group_tail,
        )
        if base.empty or head.empty:
            st.info("Brak danych do analizy struktury po grupowaniu.")
            return {"chart_meta": {"kind": "composition_static"}, "chart_context": {}}
        st.caption(f"HHI (koncentracja): **{hhi:,.0f}**".replace(",", " "))
        head["rank_disp"] = range(1, len(head) + 1)

        # =========================
        # Block 1: Ranking (Top-N)
        # =========================
        st.markdown("### Jak wygląda struktura wartości (absoluty)?")
        st.caption("Ranking wartości pozwala precyzyjnie porównać wielkość segmentów.")

        rank_df = head.copy()
        rank_df["group_label"] = rank_df["group"].astype(str)

        chart_rank = (
            alt.Chart(rank_df)
            .mark_bar()
            .encode(
                y=alt.Y("group_label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("value:Q", title="Wartość (Suma)"),
                tooltip=[
                    alt.Tooltip("group_label:N", title="Kategoria"),
                    alt.Tooltip("value:Q", format=",.0f", title="Wartość"),
                    alt.Tooltip("share_full:Q", format=".1%", title="Udział w całości"),
                ],
            )
            .properties(height=420)
        )
        altair_chart_stretch(st, chart_rank, width='stretch')

        _exec_takeaway(
            "ranking",
            anchors={
                "metric": value_col,
                "cat1": group_col,
                "top_segment": str(rank_df.iloc[0].get("group")) if not rank_df.empty else None,
                "top_value": float(rank_df.iloc[0].get("value", 0.0)) if not rank_df.empty else None,
                "top_share_pct": float(rank_df.iloc[0].get("share_full", 0.0)) * 100.0 if not rank_df.empty else None,
                "n_segments": int(rank_df.shape[0]),
                "hhi": float(hhi) if "hhi" in locals() and hhi is not None else None,
            },
        )
        _guidance(
            sens='Daje „twardą” skalę i porównanie wielkości segmentów.',
            interp='Duże różnice długości słupków oznaczają dominację kilku segmentów.',
            best='Utrzymuj sortowanie malejące i ogranicz liczbę kategorii (TOP‑N + Other).',
        )
        st.divider()

        # =========================
        # Block 2: Waterfall (Top-N + Other)
        # =========================
        st.markdown("### Co realnie buduje total?")
        st.caption("Waterfall pokazuje wkład segmentów do łącznej wartości (Top‑N + reszta).")

        wf = head.copy()
        wf["label"] = wf["group"].astype(str)
        # FIX: ensure there is exactly ONE "Other" bar.
        # In some datasets/flows we can end up with two rows labelled "Other"
        # (e.g. original category name + aggregated tail). Collapse by label.
        wf = (
            wf.groupby("label", as_index=False, dropna=False)
            .agg(value=("value", "sum"))
        )
        if not wf.empty:
            # Sort: largest first, keep Other at end
            if synthetic_group_tail in set(wf["label"].tolist()):
                wf_other = wf[wf["label"] == synthetic_group_tail]
                wf_main = wf[wf["label"] != synthetic_group_tail].sort_values("value", ascending=False)
                wf = pd.concat([wf_main, wf_other], ignore_index=True)
            else:
                wf = wf.sort_values("value", ascending=False).reset_index(drop=True)

        fig_wf = go.Figure()

        # Add TOTAL as the last row (blue bar)
        total_val = float(wf["value"].sum() or 0.0)
        wf2 = wf.copy()
        wf2["measure"] = ["relative"] * len(wf2)
        wf2 = pd.concat(
            [
                wf2,
                pd.DataFrame([{
                    "label": "TOTAL",
                    "value": total_val,
                    "measure": "total",
                }])
            ],
            ignore_index=True
        )

        # Waterfall v2: TOP->DOWN (largest first) + TOTAL bar at the bottom
        wf_labels = list(wf["label"])
        wf_values = [float(v) for v in wf["value"]]
        total_value = float(sum(wf_values))

        y = wf_labels + ["TOTAL"]
        x = wf_values + [total_value]
        measure = (["relative"] * len(wf_values)) + ["total"]

        fig_wf.add_trace(go.Waterfall(
            orientation="h",
            measure=measure,
            y=y,
            x=x,
            text=[f"{v:,.0f}".replace(",", " ") for v in x],
            textposition="outside",
            connector={"line": {"width": 1}},
            # ✅ poprawna nazwa: totals (nie "total")
            totals={"marker": {"color": "#2D6CDF"}},
            increasing={"marker": {"color": "#2AAE6A"}},
            decreasing={"marker": {"color": "#D64550"}},
        ))

        fig_wf.update_layout(
            height=430,
            margin=dict(l=320, r=120, t=10, b=45),
            xaxis_title="Wkład do totalu",
            yaxis_title=None,
            yaxis=dict(
                autorange="reversed",
                categoryorder="array",
                categoryarray=y,
                tickmode="array",
                tickvals=y,
                ticktext=y,
                showticklabels=True,
                ticklabelposition="outside",
                automargin=True,
                tickfont=dict(size=12, color="#374151"),
            ),
            xaxis=dict(range=[0, max(total_value * 1.10, 1.0)]),
            showlegend=False,
        )
        plotly_chart_stretch(st, fig_wf, config={"displayModeBar": False})

        _exec_takeaway(
            "waterfall",
            anchors={
                "metric": value_col,
                "cat1": group_col,
                "top_item": str(wf.iloc[0]["label"]) if "wf" in locals() and not wf.empty else None,
                "top_item_value": float(wf.iloc[0]["value"]) if "wf" in locals() and not wf.empty else None,
                "top_item_share_pct": (float(wf.iloc[0]["value"]) / float(wf["value"].sum()) * 100.0) if "wf" in locals() and not wf.empty and float(wf["value"].sum() or 0) > 0 else None,
                "n_items": int(wf.shape[0]) if "wf" in locals() else None,
            },
        )
        _guidance(
            sens="Identyfikuje główne dźwignie wyniku i porządkuje priorytety działań.",
            interp='Największe „kroki” to segmenty o największym wkładzie; „Other” pokazuje ogon wartości.',
            best="Zwykle wystarczy TOP‑5/10 + Other; zbyt wiele kroków pogarsza czytelność.",
        )
        st.divider()

        # =========================
        # Block 3: Pareto (Concentration)
        # =========================
        st.markdown("### Czy wartość sprzedaży jest skoncentrowana?")
        st.caption("Pareto pokazuje, czy większość wartości generuje niewielka liczba segmentów.")

        pareto = base.copy()
        pareto["group_label"] = pareto["group"].astype(str)

        # limit number of bars for UX
        max_bars = max(25, topn)
        pareto_vis = pareto.head(max_bars).copy()
        if len(pareto) > max_bars:
            rest_val = float(pareto.iloc[max_bars:]["value"].sum())
            rest_row = pd.DataFrame([{"group": synthetic_group_tail, "value": rest_val}])
            rest_row["share_full"] = rest_val / total_full
            pareto_vis = pd.concat([pareto_vis, rest_row.rename(columns={"group": "group_label"})], ignore_index=True)
        pareto_vis["cum"] = (pareto_vis["value"] / float(pareto_vis["value"].sum() or 1.0)).cumsum()
        cutoff_ratio = float(cutoff or 0.80)
        cutoff_ratio = cutoff_ratio / 100.0 if cutoff_ratio > 1.0 else cutoff_ratio
        cutoff_ratio = min(max(cutoff_ratio, 0.0), 1.0)
        cutoff_pct = cutoff_ratio * 100.0

        # number of segments to reach cutoff
        p_n = int((pareto_vis["cum"] >= cutoff_ratio).idxmax() + 1) if (pareto_vis["cum"] >= cutoff_ratio).any() else int(pareto_vis.shape[0])

        # Plotly combo
        fig_p = go.Figure()

        # --- Pareto threshold (x where cum crosses cutoff) + bar colors (grey tail)
        x_labels = pareto_vis["group_label"].astype(str).tolist()

        # p_n = number of segments needed to reach cutoff (already computed above)
        x_cut = None
        if p_n and p_n > 0 and (p_n - 1) < len(x_labels):
            x_cut = x_labels[p_n - 1]

        bar_colors = []
        for i in range(len(x_labels)):
            if p_n and i <= (p_n - 1):
                bar_colors.append("#1f77b4")  # main bars -> Plotly default-ish blue
            else:
                bar_colors.append("#d9d9d9")  # tail beyond 80% -> grey

        fig_p.add_trace(go.Bar(
            x=x_labels,
            y=pareto_vis["value"],
            name="Wartość",
            marker=dict(color=bar_colors),
        ))

        fig_p.add_trace(go.Scatter(
            x=x_labels,
            y=pareto_vis["cum"] * 100,
            name="Skumulowany udział %",
            yaxis="y2",
            mode="lines+markers",
        ))

        # 🔵 Marker w punkcie przecięcia 80% (na krzywej Pareto)
        # idx progu: p_n-1 (p_n = liczba segmentów potrzebnych do osiągnięcia cutoff)
        idx_cut = int(max(0, min((p_n or 1) - 1, len(pareto_vis) - 1)))

        # % skumulowany w punkcie progu (oś y2)
        y_cut_pct = float(pareto_vis.loc[idx_cut, "cum"] * 100.0)

        fig_p.add_trace(
            go.Scatter(
                x=[x_labels[idx_cut]],          # spójne z osią X (lista etykiet)
                y=[y_cut_pct],                  # % na osi y2
                mode="markers",
                marker=dict(
                    size=9,
                    color="red",                # Gestalt: ten sam kod semantyczny co linie progu (czerwone kropkowane)
                    line=dict(color="white", width=1),  # lekki “ring” dla czytelności na niebieskiej krzywej
                ),
                hoverinfo="skip",
                showlegend=False,
                yaxis="y2",
            )
        )

        fig_p.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=20, b=80),
            yaxis=dict(title="Wartość"),
            yaxis2=dict(title="Skumulowany udział %", overlaying="y", side="right", range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        )

        # --- 80% horizontal line (red dotted, explicitly on y2)
        fig_p.add_trace(
            go.Scatter(
                x=x_labels,
                y=[cutoff_pct] * len(x_labels),
                name=f"próg {int(round(cutoff_pct))}%",
                yaxis="y2",
                mode="lines",
                line=dict(color="red", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # --- vertical line at crossing + label "próg 80%"
        if x_cut is not None:
            # limit vline height to primary Y axis (bar scale)
            y_max_val = float(pareto_vis["value"].max()) if len(pareto_vis) else 0.0

            fig_p.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=x_cut,
                x1=x_cut,
                y0=0,
                y1=y_max_val,
                line=dict(color="red", width=1, dash="dot"),
            )

            # label above the vline, black (as caption, not geometry)
            fig_p.add_annotation(
                x=x_cut,
                xref="x",
                y=0.95,              # ⬅️ zamiast 1.0 — faktyczne obniżenie
                yref="paper",
                text=f"próg {int(cutoff*100)}%",
                showarrow=False,
                xanchor="center",  # ładniej: centralnie nad linią
                yanchor="bottom",
                font=dict(color="black", size=11),
                bgcolor="rgba(255,255,255,0.0)",
            )

        plotly_chart_stretch(st, fig_p, config={"displayModeBar": False})

        _exec_takeaway(
            "pareto",
            anchors={
                "metric": value_col,
                "cat1": group_col,
                "cutoff_pct": float(cutoff_pct) if "cutoff_pct" in locals() else None,
                "p_n": int(p_n) if "p_n" in locals() else None,
                "n_segments": int(pareto_vis.shape[0]) if "pareto_vis" in locals() else None,
                "top_share_pct": float(pareto_vis.iloc[p_n - 1]["cum"]*100.0) if "pareto_vis" in locals() and "p_n" in locals() and p_n and p_n > 0 and (p_n-1) < len(pareto_vis) else None,
            },
        )
        _guidance(
            sens=f"Szybko ocenia koncentrację (np. {int(cutoff*100)}/{100-int(cutoff*100)}) i ryzyko zależności od top segmentów.",
            interp="Jeśli kumulacja rośnie bardzo szybko na pierwszych segmentach, wartość jest silnie skoncentrowana.",
            best="Przy silnej koncentracji buduj osobne strategie dla top segmentów i long taila.",
        )
        st.divider()

        # =========================
        # Block 4: Price corridor (NEW)
        # =========================
        st.markdown("### Korytarz cenowy (Price corridor)")
        st.caption("Które przedziały cenowe generują większość wartości? (rekomendowane dla kolumn typu cena).")

        if price_col and price_col in df.columns and value_col in df.columns:
            corr, meta_bin = _build_price_corridor_from_stats(stats_payload)
            if corr.empty:
                st.info("Brak danych do analizy cenowej (po filtrach).")
            else:
                idx80 = int(meta_bin.get("hi_idx", 0))
                lo_idx = int(meta_bin.get("lo_idx", 0))
                corridor_share = float(meta_bin.get("corridor_share_pct") or float(corr.loc[lo_idx:idx80, "share"].sum() * 100.0))
                lo_edge = meta_bin.get("corridor_low")
                hi_edge = meta_bin.get("corridor_high")
                p80_price = meta_bin.get("p80_price")
                bin80_label = str(meta_bin.get("bin80_label") or corr.loc[idx80, "price_bin"])

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Cena P80 (80% wartości)", f"{p80_price:,.0f}" if p80_price is not None else "—")
                with c2:
                    if lo_edge is not None and hi_edge is not None:
                        st.metric("Korytarz (20–80)", f"{lo_edge:,.0f} – {hi_edge:,.0f}")
                    else:
                        st.metric("Korytarz (20–80)", "—")
                with c3:
                    st.metric("Udział korytarza", f"{corridor_share:.1f}%")

                bar_colors = ["#1f77b4" if i <= idx80 else "#d9d9d9" for i in range(len(corr))]
                fig = px.bar(
                    corr,
                    x="x_i",
                    y="value",
                    labels={"x_i": "Przedział ceny", "value": "Wartość (w binie)"},
                    title=None,
                )
                fig.update_traces(marker=dict(color=bar_colors), selector=dict(type="bar"))
                fig.add_scatter(
                    x=corr["x_i"],
                    y=corr["cum_pct"],
                    mode="lines+markers",
                    name="Kumulacja (%)",
                    yaxis="y2",
                )
                fig.add_scatter(
                    x=corr["x_i"],
                    y=[80.0] * len(corr),
                    mode="lines",
                    name="80%",
                    yaxis="y2",
                    line=dict(color="red", dash="dot", width=1),
                    hoverinfo="skip",
                )
                fig.add_vrect(
                    x0=lo_idx - 0.5,
                    x1=idx80 + 0.5,
                    fillcolor="rgba(31,119,180,0.06)",
                    line_width=0,
                    layer="below",
                )
                y_max_val = float(corr["value"].max()) if len(corr) else 0.0
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=idx80,
                    x1=idx80,
                    y0=0,
                    y1=y_max_val,
                    line=dict(color="red", width=1, dash="dot"),
                )
                y_cut = float(corr.loc[idx80, "cum_pct"]) if idx80 in corr.index else 80.0
                fig.add_trace(
                    go.Scatter(
                        x=[idx80],
                        y=[y_cut],
                        mode="markers",
                        yaxis="y2",
                        showlegend=False,
                        hoverinfo="skip",
                        marker=dict(size=9, color="red", line=dict(width=1, color="white")),
                    )
                )
                fig.add_annotation(
                    x=idx80,
                    xref="x",
                    y=0.95,
                    yref="paper",
                    yshift=0,
                    text="próg 80%",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(color="black", size=12),
                    bgcolor="rgba(255,255,255,0.0)",
                )
                fig.update_layout(
                    height=460,
                    margin=dict(l=20, r=20, t=20, b=110),
                    yaxis=dict(title="Wartość"),
                    yaxis2=dict(
                        title="Kumulacja (%)",
                        overlaying="y",
                        side="right",
                        range=[0, 105],
                        showgrid=False,
                    ),
                    xaxis_tickangle=-35,
                    legend=dict(
                        orientation="h",
                        x=0.01,
                        y=0.99,
                        xanchor="left",
                        yanchor="top",
                        bgcolor="rgba(255,255,255,0.6)",
                    ),
                )
                fig.update_xaxes(tickmode="array", tickvals=corr["x_i"], ticktext=corr["price_bin"])
                plotly_chart_stretch(st, fig, config={"displayModeBar": False})

                if not str(_tw.get("price_corridor", "")).strip():
                    if (lo_edge is not None) and (hi_edge is not None) and (p80_price is not None):
                        _tw["price_corridor"] = (
                            f"Wartość jest skoncentrowana cenowo: korytarz {lo_edge:,.0f}–{hi_edge:,.0f} generuje "
                            f"{corridor_share:.1f}% wartości, a próg P80 wypada przy {p80_price:,.0f}."
                        ).replace(",", " ")
                    else:
                        _tw["price_corridor"] = (
                            f"Wartość jest skoncentrowana cenowo: korytarz (20–80) generuje {corridor_share:.1f}% wartości "
                            f"(P80: {p80_price if p80_price is not None else '—'})."
                        ).replace(",", " ")

                _exec_takeaway(
                    "price_corridor",
                    anchors={
                        "metric": value_col,
                        "price_col": price_col,
                        "p80_price": meta_bin.get("p80_price"),
                        "corridor_low": meta_bin.get("corridor_low"),
                        "corridor_high": meta_bin.get("corridor_high"),
                        "corridor_share_pct": meta_bin.get("corridor_share_pct"),
                        "bin_step": meta_bin.get("step"),
                    },
                )
                _guidance(
                    sens="Identyfikuje „sweet spot” cenowy — przedziały, które generują większość wartości.",
                    interp=f"80% wartości osiąga się do przedziału **{bin80_label}** (biny o równym kroku; krok ≈ {meta_bin.get('step')}).",
                    best="Zapewnij dostępność i ekspozycję w korytarzu; ofertę premium traktuj osobno.",
                )
                st.divider()

                if meta_bin.get("clip"):
                    q01 = meta_bin["clip"].get("q01")
                    q99 = meta_bin["clip"].get("q99")
                    st.caption(f"Uwaga: binning przycina skrajne outliery (1–99%): {q01:,.2f} – {q99:,.2f}.")
        else:
            st.info("Wybierz kolumnę ceny (numeryczną) w filtrach CS, aby pokazać korytarz cenowy.")

        st.markdown("### Jak wygląda mix w ramach grup?")
        st.caption("100% stacked pokazuje udział składników w ramach każdego segmentu (bez wpływu skali totalu).")

        mix_agg = pd.DataFrame()
        if group_col2 and group_col2 in df.columns and group_col2 != group_col:
            try:
                mix_agg = _mix_exec_frame.copy() if isinstance(_mix_exec_frame, pd.DataFrame) else pd.DataFrame()
                if mix_agg.empty:
                    mix_agg = _build_mix_exec_frame(
                        df=df,
                        group_col=group_col,
                        group_col2=group_col2,
                        value_col=value_col,
                        top_n=top_n,
                        top_k=7,
                    )
                if mix_agg.empty:
                    mix_agg = _build_resilient_mix_frame(
                        df=df,
                        group_col=group_col,
                        group_col2=group_col2,
                        value_col=value_col,
                        top_n=top_n,
                        top_k=7,
                    )
                if mix_agg.empty:
                    raise ValueError("empty_mix_frame")

                group_order = (
                    mix_agg[[group_col, "group_total"]]
                    .drop_duplicates()
                    .sort_values("group_total", ascending=False)[group_col]
                    .astype(str)
                    .tolist()
                )
                fig_mix = go.Figure()
                sub_labels = sorted(mix_agg[group_col2].unique(), key=lambda s: ("ogon" in str(s).lower(), str(s)))
                for sublab in sub_labels:
                    dsub = mix_agg[mix_agg[group_col2] == sublab]
                    fig_mix.add_trace(go.Bar(
                        y=dsub[group_col].astype(str),
                        x=dsub["pct"],
                        customdata=dsub[["value", "group_share_pct", "area_share_pct"]].to_numpy(),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            + f"{group_col2}: {str(sublab)}<br>"
                            + "Udzial skladnika w grupie: %{x:.1f}%<br>"
                            + "Wartosc: %{customdata[0]:,.0f}<br>"
                            + "Udzial grupy w totalu: %{customdata[1]:.1f}%<br>"
                            + "Wklad komorki do totalu: %{customdata[2]:.1f}%<extra></extra>"
                        ),
                        orientation="h",
                        name=str(sublab),
                    ))
                fig_mix.update_layout(
                    barmode="stack",
                    xaxis=dict(title="Udział %", range=[0, 105]),
                    yaxis=dict(
                        title=None,
                        categoryorder="array",
                        categoryarray=group_order[::-1],
                        automargin=True,
                        tickfont=dict(size=12, color="#374151"),
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.28, title=str(group_col2)),
                    margin=dict(l=160, r=20, t=30, b=70),
                    height=max(360, min(560, 120 + 34 * max(1, len(group_order)))),
                )

                # ✅ zapisz dokładnie to, czego używa MIX (żeby Marimekko było 1:1)
                try:
                    cw = list(fig_mix.layout.colorway) if fig_mix.layout.colorway else []
                    if cw:
                        st.session_state["cs_mix_colorway"] = cw
                except Exception:
                    pass

                plotly_chart_stretch(st, fig_mix, config={"displayModeBar": False})
                
            except Exception as e:
                record_debug_checkpoint("cs.mix_plotly_failed", error=f"{type(e).__name__}: {e}")
                st.info("Brak dodatnich wartości dla wybranej pary kategorii — wybierz inną Kategorię 2 albo miarę.")
        else:
            st.info("Włącz **Kategorię 2** w sidebarze (kategoryczną / low‑card), aby zobaczyć mix.")

        _exec_takeaway(
            "mix",
            anchors={
                "metric": value_col,
                "cat1": group_col,
                "cat2": group_col2,
                "top_n": int(top_n) if "top_n" in locals() else None,
            },
        )
        _guidance(
            sens="Porównuje struktury wewnętrzne między segmentami (mix), niezależnie od wielkości.",
            interp="Różne proporcje składników między segmentami oznaczają różne profile/mix.",
            best="Ogranicz składniki do Top‑7 + Other, inaczej wykres staje się nieczytelny.",
        )
        st.divider()

        # =========================
        # Block 6: Marimekko (PRO) — guarded + fallback
        # =========================
        st.markdown("### Marimekko (PRO): skala × struktura")
        st.caption(
            "Variable width łączy 2 wymiary naraz: szerokość segmentu = udział w totalu (skala), "
            "wysokość kolorów = struktura w ramach segmentu."
        )

        # Jawny opis semantyki (MUST) — utrzymuj nad wykresem
        st.markdown(
            """
        <div style="
            font-size: 0.78rem;
            line-height: 1.20;
            color: #6b7280;
            margin-top: 0.15rem;
            margin-bottom: 0.40rem;
        ">
        <ul style="margin-left: 1rem; padding-left: 0;">
        <li><b>Szerokość segmentu</b> = udział segmentu w totalu (skala).</li>
        <li><b>Wysokość koloru</b> = udział składnika w ramach segmentu (struktura).</li>
        <li><b>Pole prostokąta</b> = wkład <i>segment × składnik</i> do totalu.</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True
        )

        if mix_agg is None or mix_agg.empty:
            st.info("Wybierz Kategorię 2, aby zbudować Marimekko.")
        else:
            # if too many groups/subs -> fallback to stacked
            n_groups = int(mix_agg[group_col].nunique(dropna=True))
            n_subs = int(mix_agg[group_col2].nunique(dropna=True))
            if n_groups > 12 or n_subs > 10:
                st.info("Zbyt wiele segmentów/składników dla czytelnego Marimekko — pokazuję uproszczony 100% stacked.")
            else:
                try:
                    # Total (do tooltipów i sanity)
                    total_value = float(mix_agg["value"].sum() or 1.0)

                    # widths = total share per group; use the same frame as Mix/ET
                    mm = mix_agg.copy()
                    mm["width_pct"] = mm["group_share_pct"]
                    mm["height_pct"] = mm["pct"]
                    gtot = (
                        mm[[group_col, "group_total", "width_pct"]]
                        .drop_duplicates()
                        .sort_values("group_total", ascending=False)
                        .reset_index(drop=True)
                    )

                    # build rectangles manually
                    x0 = 0.0
                    fig_mm = go.Figure()
                    # ✅ kolory 1:1 jak w MIX (zapisane wcześniej)
                    cw = st.session_state.get("cs_mix_colorway") or []
                    if not cw:
                        # awaryjnie: Plotly default colorway (gdyby MIX nie zdążył się zbudować)
                        cw = ["#0B5ED7","#8EC9FF","#ff7f0e","#ffbb78","#2ca02c", 
                              "#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5", 
                              "#8c564b","#c49c94","#e377c2","#f7b731","#17becf",]
                        # "#8EC9FF", "#1f77b4", "#FFB000", "#00A3E0", "#6B5B95"

                    subs = sorted(mm[group_col2].unique(), key=lambda s: ("ogon" in str(s).lower(), str(s)))

                    # "Other" zawsze szary (czytelność + Gestalt)
                    color_map = {s: cw[i % len(cw)] for i, s in enumerate(subs)}
                    for s in subs:
                        if "ogon" in str(s).lower():
                            color_map[s] = "#E0E0E0"

                    # ── Label w polu (warunkowy) + auto kolor tekstu + tooltip ──
                    def _hex_to_rgb(hx: str) -> tuple[int, int, int]:
                        hx = (hx or "").lstrip("#")
                        if len(hx) != 6:
                            return (204, 204, 204)
                        return tuple(int(hx[i : i + 2], 16) for i in (0, 2, 4))

                    def _rel_lum(rgb: tuple[int, int, int]) -> float:
                        # względna luminancja (WCAG-ish)
                        def _ch(c: float) -> float:
                            c = c / 255.0
                            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

                        r, g, b = rgb
                        return 0.2126 * _ch(r) + 0.7152 * _ch(g) + 0.0722 * _ch(b)

                    def _text_color(fill: str) -> str:
                        # czarny na jasnych, biały na ciemnych
                        return "#111111" if _rel_lum(_hex_to_rgb(fill)) > 0.45 else "#ffffff"

                    # progi etykiet (żeby nie zabić czytelności)
                    MIN_W_PCT = 4.0   # min szerokość segmentu
                    MIN_H_PCT = 4.0  # min wysokość komórki
                    MIN_AREA = 0.8  # min pole (w% * h%)

                    hover_x, hover_y, hover_cd = [], [], []

                    for g in gtot.sort_values("group_total", ascending=False)[group_col].tolist():
                        w = float(gtot.loc[gtot[group_col] == g, "width_pct"].iloc[0])
                        y0 = 0.0
                        mm_g = mm[mm[group_col] == g].sort_values("height_pct", ascending=False)
                        for _, r in mm_g.iterrows():
                            h = float(r["height_pct"])
                            comp = str(r[group_col2])
                            fill = color_map.get(comp, "#E0E0E0")
                            v = float(r["value"] or 0.0)
                            group_total = float(r["group_total"] or 0.0)
                            width_pct = float(w)
                            height_pct = float(h)
                            area_pct = (v / total_value * 100.0) if total_value else 0.0

                            fig_mm.add_shape(
                                type="rect",
                                x0=x0, x1=x0+w,
                                y0=y0, y1=y0+h,
                                line=dict(width=0.5),
                                fillcolor=fill,
                            )

                            # Tooltip: dodaj "niewidzialny" punkt w środku prostokąta
                            hover_x.append(x0 + w / 2)
                            hover_y.append(y0 + h / 2)
                            hover_cd.append(
                                [
                                    str(g),
                                    comp,
                                    v,
                                    width_pct,
                                    height_pct,
                                    area_pct,
                                ]
                            )

                            # Label w polu: tylko jeśli komórka jest wystarczająco duża
                            if (width_pct >= MIN_W_PCT) and (height_pct >= MIN_H_PCT) and ((width_pct * height_pct) >= MIN_AREA):
                                fig_mm.add_annotation(
                                    x=x0 + w / 2,
                                    y=y0 + h / 2,
                                    text=comp,
                                    showarrow=False,
                                    font=dict(size=11, color="#111827"),
                                    bgcolor="rgba(255,255,255,0.72)",
                                    bordercolor="rgba(255,255,255,0)",
                                    borderpad=2,
                                )
                            y0 += h
                        # group label
                        fig_mm.add_annotation(
                            x=x0 + w/2,
                            y=102,
                            text=str(g),
                            showarrow=False,
                            font=dict(size=10, color="#111827"),
                            bgcolor="rgba(255,255,255,0.80)",
                            bordercolor="rgba(255,255,255,0)",
                            borderpad=1,
                        )
                        x0 += w

                    # Tooltip trace (po shapes)
                    fig_mm.add_trace(
                        go.Scatter(
                            x=hover_x,
                            y=hover_y,
                            mode="markers",
                            marker=dict(size=18, opacity=0),
                            customdata=hover_cd,
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                f"{group_col2}: " + "%{customdata[1]}<br>"
                                "Wartość: %{customdata[2]:,.0f}<br>"
                                "Udział segmentu w totalu (szerokość): %{customdata[3]:.1f}%<br>"
                                "Udział składnika w segmencie (wysokość): %{customdata[4]:.1f}%<br>"
                                "Udział komórki w totalu (pole): %{customdata[5]:.1f}%<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

                    fig_mm.update_layout(
                        height=420,
                        margin=dict(l=20, r=20, t=30, b=40),
                        xaxis=dict(title="Udział w totalu (%)", range=[0, 100], showgrid=False, zeroline=False),
                        yaxis=dict(title="Struktura (%)", range=[0, 100], showgrid=False, zeroline=False),
                        showlegend=False,
                    )
                    plotly_chart_stretch(st, fig_mm, config={"displayModeBar": False})
                except Exception:
                    st.info("Nie udało się wyrenderować Marimekko dla tych danych — użyj wykresu Mix.")

        _exec_takeaway(
            "marimekko",
            anchors={
                "metric": value_col,
                "cat1": group_col,
                "cat2": group_col2,
                "top_n": int(top_n) if "top_n" in locals() else None,
            },
        )
        _guidance(
            sens="Pokazuje jednocześnie skalę segmentu (udział w totalu) i jego strukturę (skład).",
            interp="Szerokie segmenty mają największy wpływ na wynik, a dominujące kolory wskazują kluczowe składniki.",
            best="Ogranicz liczbę segmentów (Top‑N) i składników (Top‑7 + Other), aby wykres był czytelny.",
        )


    # Global debug panel (exec takeaway)
    try:
        from data_chat_core.exec_takeaway_debug import render_exec_takeaway_debug_panel
        render_exec_takeaway_debug_panel(expanded=False)
    except Exception:
        pass

    return {"chart_meta": {"kind": "composition_static"}, "chart_context": {}}

