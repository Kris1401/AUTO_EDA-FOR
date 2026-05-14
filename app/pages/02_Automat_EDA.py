# app/pages/02_Automat_EDA.py
from __future__ import annotations

import os
import io
import json
import math
import base64
import time
import re
from contextlib import contextmanager
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Parquet-only storage helpers (Stage 1/2/3)
# ─────────────────────────────────────────────────────────────
def _df_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    except Exception:
        df.to_parquet(path, index=False, compression="snappy")


def _evenly_spaced_positions(total_rows: int, max_rows: int | None) -> np.ndarray | None:
    """Deterministic row positions spread across the whole file."""
    if max_rows is None:
        return None
    try:
        total = int(total_rows)
        limit = int(max_rows)
    except Exception:
        return None
    if limit <= 0 or total <= limit:
        return None
    return np.unique(np.linspace(0, total - 1, num=limit, dtype=np.int64))


def _df_from_parquet(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load Parquet. If max_rows is set, take an even sample from the whole file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        total_rows = int(getattr(pf.metadata, "num_rows", 0) or 0)
        positions = _evenly_spaced_positions(total_rows, max_rows)
        if positions is None:
            return pf.read().to_pandas()

        frames: list[pd.DataFrame] = []
        batch_start = 0
        for batch in pf.iter_batches(batch_size=32_768):
            batch_rows = int(batch.num_rows)
            batch_end = batch_start + batch_rows
            start_pos = int(np.searchsorted(positions, batch_start, side="left"))
            end_pos = int(np.searchsorted(positions, batch_end, side="left"))
            if end_pos > start_pos:
                local_positions = positions[start_pos:end_pos] - batch_start
                table = pa.Table.from_batches([batch])
                idx = pa.array(local_positions.astype(np.int64), type=pa.int64())
                frames.append(table.take(idx).to_pandas())
            batch_start = batch_end

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    except Exception:
        if max_rows is not None:
            raise
        return pd.read_parquet(path)

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Auto EDA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from core.ui_safe import altair_chart_stretch, dataframe_stretch
import altair as alt

from core.top_nav import (
    hide_default_multipage_nav,
    render_flow_nav,
    render_sidebar_links,
)
from core.config import load_config, resolve_artifacts_dir


# --- wspólna paleta barw ---
PALETTE_MAIN = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

FACET_CHART_WIDTH = 980  # szerokość jednego panelu w układzie facet

# --- PERF logger (terminal) ---
_PERF_LOGGER = logging.getLogger('eda_perf')
if not _PERF_LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def _perf(section: str) -> float:
    """Return start time and log section entry to terminal."""
    t0 = time.perf_counter()
    _PERF_LOGGER.info(f'[PERF] enter: {section}')
    return t0

def _perf_end(section: str, t0: float) -> None:
    dt = time.perf_counter() - t0
    _PERF_LOGGER.info(f'[PERF] exit:  {section} ({dt:.2f}s)')

def _df_preview_for_ui(obj, max_rows: int = 2000) -> pd.DataFrame:
    """Prepare a lightweight, Arrow-safe preview DataFrame for Streamlit UI.

    Accepts:
      - pd.DataFrame
      - pd.io.formats.style.Styler (we take `.data`)
    Returns:
      - pd.DataFrame (possibly truncated to `max_rows`)
    """
    # If a pandas Styler is passed (e.g., from df.style), unwrap to underlying DataFrame
    try:
        from pandas.io.formats.style import Styler
        if isinstance(obj, Styler):
            obj = obj.data
    except Exception:
        pass

    if obj is None:
        return pd.DataFrame()

    # Styler -> underlying DataFrame
    try:
        from pandas.io.formats.style import Styler  # type: ignore
        if isinstance(obj, Styler):
            obj = obj.data
    except Exception:
        # Styler import path may vary; ignore and treat as generic
        pass

    if not isinstance(obj, pd.DataFrame):
        # best-effort conversion
        try:
            obj = pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame()

    df2 = obj.copy()

    # truncate early for UI safety
    try:
        if max_rows is not None and max_rows > 0 and len(df2) > max_rows:
            df2 = df2.head(max_rows)
    except Exception:
        # if len fails for some exotic object, fallback to head
        try:
            df2 = df2.head(max_rows)
        except Exception:
            return pd.DataFrame()

    return _df_coerce_for_arrow(df2)


def _df_coerce_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """Force DataFrame into Arrow-serializable dtypes for Streamlit.

    Key goal: avoid ArrowInvalid like mixed object column ('Invoice') being inferred as int.
    This function is *display-only*; it should NOT be used for modeling computations.
    """
    if df is None or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    df2 = df.copy()

    for c in df2.columns:
        s = df2[c]
        # categories -> string
        try:
            if str(s.dtype) == "category":
                df2[c] = s.astype(str)
                continue
        except Exception:
            pass

        # object / mixed -> string (prevents Arrow trying to cast to int)
        try:
            if s.dtype == "object":
                df2[c] = s.astype(str)
                continue
        except Exception:
            pass

        # pandas nullable integer -> convert to float to keep NaNs Arrow-safe
        try:
            if str(s.dtype).startswith("Int"):
                df2[c] = s.astype("float64")
        except Exception:
            pass

    return df2


def st_df_safe(obj, *, max_rows: int = 2000, **kwargs):
    """Safe wrapper around st.dataframe that:
    - accepts DataFrame or Styler
    - truncates to max_rows
    - coerces to Arrow-safe dtypes to prevent Streamlit serialization crashes
    """
    df2 = _df_preview_for_ui(obj, max_rows=max_rows)
    return dataframe_stretch(st, df2, **kwargs)


def st_table_safe(obj, *, max_rows: int = 2000, **kwargs):
    """Safe wrapper around st.table with the same guarantees as st_df_safe."""
    df2 = _df_preview_for_ui(obj, max_rows=max_rows)
    return st.table(df2, **kwargs)


def _scale_for_categories(cats: List[str]) -> alt.Scale:
    """Zwraca Scale z domeną = kolejność kategorii i rangą z naszej palety (powiela gdy potrzeba)."""
    if not cats:
        return alt.Scale(range=PALETTE_MAIN)
    if len(cats) <= len(PALETTE_MAIN):
        rng = PALETTE_MAIN[: len(cats)]
    else:
        k = int(np.ceil(len(cats) / len(PALETTE_MAIN)))
        rng = (PALETTE_MAIN * k)[: len(cats)]
    return alt.Scale(domain=cats, range=rng)

import requests
import hashlib, re
try:
    import openai as _openai
except Exception:
    _openai = None
from uuid import uuid4
try:
    from langfuse import Langfuse
except Exception:
    Langfuse = None
try:
    from langfuse.decorators import observe  # opcjonalny dekorator
except Exception:
    def observe(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator

from dotenv import load_dotenv

# Wczytaj .env (OPENAI_API_KEY, ELEVENLABS_API_KEY, VOICE_* itp.)
load_dotenv(override=True)


def _get_env_or_secret(name: str, default: str = "") -> str:
    """Read local env and Streamlit Community Cloud secrets with one code path."""
    value = os.getenv(name, "")
    if value:
        return str(value).strip()
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    if value is None:
        return default
    value = str(value).strip()
    return value or default


def _sync_secret_to_env(name: str) -> None:
    """Expose Streamlit secrets to SDKs that read only environment variables."""
    if os.getenv(name):
        return
    value = _get_env_or_secret(name)
    if value:
        os.environ[name] = value


for _secret_name in (
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "ELEVENLABS_API_KEY",
):
    _sync_secret_to_env(_secret_name)


# Altair: nie tniemy >5k wierszy
alt.data_transformers.disable_max_rows()

try:
    from langfuse.openai import OpenAI as LFOpenAI  # wrapper kompatybilny z openai.OpenAI
except Exception:
    LFOpenAI = None

@st.cache_resource(show_spinner=False)
def get_langfuse():
    try:
        if Langfuse is None:
            return None
        # Zwraca obiekt Langfuse lub None, jeśli brak kluczy/połączenia
        public_key = _get_env_or_secret("LANGFUSE_PUBLIC_KEY")
        secret_key = _get_env_or_secret("LANGFUSE_SECRET_KEY")
        if not (public_key and secret_key):
            return None
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=_get_env_or_secret("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            release=_get_env_or_secret("LANGFUSE_RELEASE", "app-eda@dev"),
            sdk_integration="streamlit"
        )
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_lf_openai_client():
    # Zwraca klienta OpenAI z automatycznym tracingiem Langfuse (dla Chat)
    try:
        if LFOpenAI is None:
            return None
        if not (_get_env_or_secret("LANGFUSE_PUBLIC_KEY") and _get_env_or_secret("LANGFUSE_SECRET_KEY")):
            return None
        api_key = _get_env_or_secret("OPENAI_API_KEY")
        if not api_key:
            return None
        return LFOpenAI(api_key=api_key)
    except Exception:
        return None


def get_plain_openai_client():
    try:
        if _openai is None:
            return None
        api_key = _get_env_or_secret("OPENAI_API_KEY")
        if not api_key:
            return None
        return _openai.OpenAI(api_key=api_key)
    except Exception:
        return None


def _openai_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            msg = err.get("message") or err.get("type") or str(err)
        else:
            msg = str(payload)[:500]
    except Exception:
        msg = (response.text or "").strip()[:500]

    if response.status_code == 401:
        return "OpenAI zwrocil 401: sprawdz OPENAI_API_KEY w Streamlit Secrets."
    if response.status_code == 429:
        return f"OpenAI zwrocil 429: limit lub brak srodkow na koncie. {msg}"
    if response.status_code == 404:
        return f"OpenAI zwrocil 404: sprawdz nazwe modelu. {msg}"
    return f"OpenAI HTTP {response.status_code}: {msg}"


def _openai_chat_completion_via_rest(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    **kwargs,
):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **{k: v for k, v in kwargs.items() if v is not None},
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    if response.status_code >= 400:
        raise RuntimeError(_openai_error_message(response))

    data = response.json()
    try:
        content = data["choices"][0]["message"].get("content") or ""
    except Exception as exc:
        raise RuntimeError(f"OpenAI REST zwrocil nieoczekiwany format odpowiedzi: {exc}") from exc

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        raw=data,
    )


def _openai_chat_completion(model: str, messages: list[dict], temperature: float = 0.2, **kwargs):
    api_key = _get_env_or_secret("OPENAI_API_KEY")
    errors: list[str] = []

    lf_openai = get_lf_openai_client()
    if lf_openai is not None:
        try:
            return lf_openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )
        except Exception as exc:
            errors.append(f"Langfuse SDK: {exc}")

    client = get_plain_openai_client()
    if client is not None:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )
        except Exception as exc:
            errors.append(f"OpenAI SDK: {exc}")

    if not api_key:
        errors.append("brak OPENAI_API_KEY w .env i Streamlit Secrets")
        raise RuntimeError("; ".join(errors))

    try:
        return _openai_chat_completion_via_rest(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
    except Exception as exc:
        errors.append(f"OpenAI REST: {exc}")
        raise RuntimeError("; ".join(errors)) from exc


def _eda_internal_checkpoints_enabled() -> bool:
    return True


def _eda_collect_checkpoints() -> bool:
    return _eda_internal_checkpoints_enabled()


def _eda_reset_run_debug_state() -> None:
    st.session_state["eda_debug_log_v1"] = []
    st.session_state["eda_exec_units_v1"] = {}
    st.session_state["eda_exec_summary_v1"] = {}


def _eda_clear_debug_state() -> None:
    for key in [
        "eda_debug_log_v1",
        "eda_exec_units_v1",
        "eda_exec_summary_v1",
        "eda_summary_debug_v1",
        "eda_cluster_debug_v1",
    ]:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if key.startswith("cluster_ai_labels__") and key.endswith("__debug"):
            st.session_state.pop(key, None)


def _eda_debug_sanitize(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return str(type(value).__name__)

    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "rows": int(len(value)),
            "cols": int(len(value.columns)),
            "columns": [str(c) for c in list(value.columns[:8])],
        }

    if isinstance(value, pd.Series):
        return {
            "type": "series",
            "len": int(len(value)),
            "name": str(value.name),
            "head": [_eda_debug_sanitize(v, depth + 1) for v in value.head(5).tolist()],
        }

    if isinstance(value, dict):
        items = list(value.items())
        out = {}
        for k, v in items[:20]:
            out[str(k)] = _eda_debug_sanitize(v, depth + 1)
        if len(items) > 20:
            out["_truncated_keys"] = len(items) - 20
        return out

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        return {
            "type": "list",
            "len": len(seq),
            "head": [_eda_debug_sanitize(v, depth + 1) for v in seq[:8]],
        }

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        try:
            if not np.isfinite(value):
                return None
        except Exception:
            return None
        return float(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if value is None:
        return None

    return str(value)


def _eda_record_checkpoint(where: str, **payload: Any) -> None:
    if not _eda_collect_checkpoints():
        return

    row = {"where": where}
    for key, value in payload.items():
        row[str(key)] = _eda_debug_sanitize(value)

    log = st.session_state.setdefault("eda_debug_log_v1", [])
    log.append(row)
    if len(log) > 400:
        del log[:-400]


def _eda_register_exec_result(block_id: str, src: str, **extra: Any) -> None:
    units = st.session_state.setdefault("eda_exec_units_v1", {})
    item = {"block_id": block_id, "src": src}
    for key, value in extra.items():
        item[str(key)] = value
    units[block_id] = item

    _eda_record_checkpoint("eda.exec.block", **item)

    values = list(units.values())
    blocks_count = len(values)
    llm_count = sum(1 for row in values if str(row.get("src", "")).startswith("llm"))
    det_count = max(0, blocks_count - llm_count)
    llm_share_pct = float(llm_count / blocks_count) if blocks_count else 0.0

    summary = {
        "blocks_count": int(blocks_count),
        "llm_count": int(llm_count),
        "det_count": int(det_count),
        "llm_share_pct": round(llm_share_pct, 3),
    }
    st.session_state["eda_exec_summary_v1"] = summary
    _eda_record_checkpoint("eda.exec.summary", **summary)


def _eda_parse_json_like(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
        text = text.replace("json", "", 1).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed


def _eda_norm_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _eda_tldr_fallback_sections(
    source_name: str,
    readiness_score: int,
    n_rows_raw: int,
    n_cols_raw: int,
    global_missing_pct: float,
    duplicates_count: int,
    duplicates_pct: float,
    auto_drop_candidates: list[str],
    pairs_sorted: list[dict[str, Any]],
) -> dict[str, list[str]]:
    strong_pairs = [p for p in pairs_sorted if abs(float(p.get("corr", 0.0))) >= 0.9]
    example_pair = strong_pairs[0] if strong_pairs else None
    example_pair_txt = (
        f"{example_pair['col1']} -> {example_pair['col2']} (r={float(example_pair['corr']):.2f})"
        if example_pair
        else "brak par o bardzo wysokiej korelacji"
    )
    drops_preview = ", ".join(auto_drop_candidates[:4]) if auto_drop_candidates else "brak"

    what = [
        f"Zbior '{source_name}' zawiera {int(n_rows_raw):,} wierszy i {int(n_cols_raw):,} kolumn.",
        f"Braki globalne wynosza ok. {float(global_missing_pct):.1f}%, a pelne duplikaty {int(duplicates_count):,} ({float(duplicates_pct):.1f}%).",
        f"Heurystyka wskazuje {len(auto_drop_candidates)} kolumn do weryfikacji lub wykluczenia ({drops_preview}).",
    ]
    insights = [
        f"Readiness score to {int(readiness_score)}/100, wiec zbior wymaga kontrolowanego przygotowania przed trenowaniem.",
        f"Silnych par korelacji (|r|>=0.9) wykryto {len(strong_pairs)}; przyklad: {example_pair_txt}.",
        "Najwieksze ryzyka na tym etapie to redundantne cechy, braki danych i potencjalne duplikaty rekordow.",
    ]
    next_steps = [
        "Przejrzyj kandydatow do wykluczenia i potwierdz, czy kolumny techniczne lub identyfikatory nie powinny wejsc do modelu.",
        "Po akceptacji przygotowania danych uruchom krok budowy zbioru treningowego i zweryfikuj raport po czyszczeniu.",
        "Przed trenowaniem sprawdz jeszcze docelowa zmienna oraz ewentualne przecieki informacji.",
    ]
    return {"what": what, "insights": insights, "next_steps": next_steps}


def _eda_tldr_one_sentence(
    readiness_score: int,
    global_missing_pct: float,
    duplicates_count: int,
    duplicates_pct: float,
    auto_drop_candidates: list[str],
    pairs_sorted: list[dict[str, Any]],
) -> str:
    strong_pairs = sum(1 for p in pairs_sorted if abs(float(p.get("corr", 0.0))) >= 0.9)
    return (
        f"Jakosc danych oceniamy na {int(readiness_score)}/100: braki globalne to ~{float(global_missing_pct):.1f}%, "
        f"duplikaty {int(duplicates_count):,} ({float(duplicates_pct):.1f}%), a do weryfikacji pozostaje "
        f"{len(auto_drop_candidates)} kolumn i {strong_pairs} silnych zaleznosci."
    )


def _eda_format_tldr_markdown(one_sentence: str, sections: dict[str, list[str]]) -> str:
    ordered = [
        ("Odpowiedź w jednym zdaniu", [one_sentence]),
        ("Podsumowanie danych", sections.get("what", [])),
        ("Insight z analizy", sections.get("insights", [])),
        ("Co dalej", sections.get("next_steps", [])),
    ]
    parts: list[str] = []
    for title, bullets in ordered:
        clean = [_eda_norm_text(b) for b in bullets if _eda_norm_text(b)]
        if not clean:
            continue
        parts.append(f"### {title}")
        for bullet in clean:
            parts.append(f"- {bullet}")
        parts.append("")
    return "\n".join(parts).strip()


def _eda_generate_tldr_markdown(
    source_name: str,
    readiness_score: int,
    duplicates_count: int,
    duplicates_pct: float,
    global_missing_pct: float,
    auto_drop_candidates: list[str],
    pairs_sorted: list[dict[str, Any]],
    n_rows_raw: int,
    n_cols_raw: int,
    model: str,
    trace: Any = None,
) -> tuple[str, dict[str, Any]]:
    render_key = str(
        (
            "v1",
            source_name,
            int(n_rows_raw),
            int(n_cols_raw),
            float(global_missing_pct),
            int(duplicates_count),
            float(duplicates_pct),
            int(readiness_score),
            tuple(auto_drop_candidates[:8]),
            int(sum(1 for p in pairs_sorted if abs(float(p.get("corr", 0.0))) >= 0.9)),
            model,
        )
    )

    facts = {
        "source_name": source_name,
        "n_rows": int(n_rows_raw),
        "n_cols": int(n_cols_raw),
        "global_missing_pct": round(float(global_missing_pct), 2),
        "duplicates_count": int(duplicates_count),
        "duplicates_pct": round(float(duplicates_pct), 2),
        "readiness_score": int(readiness_score),
        "auto_drop_count": int(len(auto_drop_candidates)),
        "auto_drop_candidates": auto_drop_candidates[:8],
        "strong_corr_pairs_count": int(sum(1 for p in pairs_sorted if abs(float(p.get("corr", 0.0))) >= 0.9)),
    }
    example_pair = next((p for p in pairs_sorted if abs(float(p.get("corr", 0.0))) >= 0.9), None)
    if example_pair:
        facts["strong_corr_example"] = {
            "col1": str(example_pair.get("col1", "")),
            "col2": str(example_pair.get("col2", "")),
            "corr": round(float(example_pair.get("corr", 0.0)), 3),
        }

    fallback_sections = _eda_tldr_fallback_sections(
        source_name=source_name,
        readiness_score=readiness_score,
        n_rows_raw=n_rows_raw,
        n_cols_raw=n_cols_raw,
        global_missing_pct=global_missing_pct,
        duplicates_count=duplicates_count,
        duplicates_pct=duplicates_pct,
        auto_drop_candidates=auto_drop_candidates,
        pairs_sorted=pairs_sorted,
    )
    final_one_sentence = _eda_tldr_one_sentence(
        readiness_score=readiness_score,
        global_missing_pct=global_missing_pct,
        duplicates_count=duplicates_count,
        duplicates_pct=duplicates_pct,
        auto_drop_candidates=auto_drop_candidates,
        pairs_sorted=pairs_sorted,
    )
    fallback_markdown = _eda_format_tldr_markdown(final_one_sentence, fallback_sections)

    prompt = (
        "Jestes senior data strategist. Na podstawie faktow wejscowych przygotuj tylko jeden obiekt JSON.\n"
        "Nie dodawaj zadnego komentarza poza JSON-em.\n"
        "Zasady:\n"
        "- pisz po polsku,\n"
        "- badz konkretny i zarzadczo-praktyczny,\n"
        "- sekcja 'one_sentence' NIE moze uzywac skewness ani innych metryk poza podanymi faktami,\n"
        "- nie halucynuj przyczyn biznesowych,\n"
        "- nie odnos sie do sezonowosci, jesli nie ma jej w faktach.\n"
        "Format JSON:\n"
        "{\n"
        '  "one_sentence": "...",\n'
        '  "what": ["...", "..."],\n'
        '  "insights": ["...", "..."],\n'
        '  "next_steps": ["...", "..."]\n'
        "}\n"
        "Kazda lista powinna miec 2-4 krotkie punkty.\n"
        f"Fakty wejściowe:\n{json.dumps(facts, ensure_ascii=False, indent=2)}"
    )

    _eda_record_checkpoint("eda.summary.start", openai_model=model, render_key=render_key)
    _eda_record_checkpoint("eda.summary.stats_payload", **facts)

    debug = {
        "render_key": render_key,
        "model": model,
        "raw_text": None,
        "gate_ok": False,
        "gate_reasons": [],
        "used_fallback": False,
        "error": None,
        "postprocessed": False,
        "final_one_sentence": final_one_sentence,
    }

    try:
        resp = _openai_chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw_text = (resp.choices[0].message.content or "").strip()
        debug["raw_text"] = raw_text

        parsed = _eda_parse_json_like(raw_text)
        gate_reasons: list[str] = []
        repaired = False
        final_sections: dict[str, list[str]] = {}

        for key in ("what", "insights", "next_steps"):
            raw_items = parsed.get(key, [])
            items = [_eda_norm_text(x) for x in (raw_items if isinstance(raw_items, list) else [])]
            items = [x for x in items if len(x) >= 12]
            if len(items) < 2:
                gate_reasons.append(f"{key}: za malo poprawnych punktow")
                repaired = True
            merged = items[:3]
            for fallback_item in fallback_sections[key]:
                if len(merged) >= 3:
                    break
                if fallback_item not in merged:
                    merged.append(fallback_item)
            final_sections[key] = merged[:3]

        if not isinstance(parsed.get("one_sentence"), str) or len(_eda_norm_text(parsed.get("one_sentence"))) < 20:
            gate_reasons.append("one_sentence: brak sensownego leadu")
            repaired = True

        debug["gate_ok"] = len(gate_reasons) == 0
        debug["gate_reasons"] = gate_reasons
        debug["postprocessed"] = True

        final_markdown = _eda_format_tldr_markdown(final_one_sentence, final_sections)
        final_source = "llm_selected_repaired" if repaired else "llm_selected"
        st.session_state["eda_summary_debug_v1"] = {**debug, "final_source": final_source}
        _eda_record_checkpoint("eda.summary.final_source", src=final_source, used_fallback=False, gate_reasons=gate_reasons, one_sentence=final_one_sentence)
        _eda_register_exec_result("summary_tldr", final_source, model=model)

        if trace:
            trace.update(status="success", output=final_markdown[:2000])
        return final_markdown, st.session_state["eda_summary_debug_v1"]

    except Exception as exc:
        debug["error"] = str(exc)
        debug["used_fallback"] = True
        debug["gate_reasons"] = ["fallback: blad generacji lub parsowania"]
        st.session_state["eda_summary_debug_v1"] = {**debug, "final_source": "fallback_deterministic_selected"}
        _eda_record_checkpoint(
            "eda.summary.final_source",
            src="fallback_deterministic_selected",
            used_fallback=True,
            error=str(exc),
            gate_reasons=debug["gate_reasons"],
            one_sentence=final_one_sentence,
        )
        _eda_register_exec_result("summary_tldr", "fallback_deterministic_selected", model=model)
        if trace:
            trace.update(status="error", metadata={"error": str(exc)})
        return fallback_markdown, st.session_state["eda_summary_debug_v1"]


def _build_cluster_summaries(
    df: pd.DataFrame,
    cluster_col: str,
    max_features: int = 8,
    max_examples_per_cluster: int = 3,
) -> dict[str, dict]:
    if cluster_col not in df.columns:
        return {}

    work = df.copy()
    work = work.dropna(subset=[cluster_col])
    if work.empty:
        return {}

    max_rows = 5000
    if len(work) > max_rows:
        work = work.sample(max_rows, random_state=0)

    work["_cluster_id"] = work[cluster_col].astype(str)

    num_cols = [c for c in _numeric_measure_candidates(work, min_non_null=3) if c not in (cluster_col, "_cluster_id")][:max_features]
    for col in num_cols:
        work[col] = _numeric_series_for_candidate(work[col])

    cat_cols = [
        c for c in _categorical_feature_candidates(work)
        if c not in (cluster_col, "_cluster_id")
    ][:max_features]

    global_means = work[num_cols].mean(numeric_only=True) if num_cols else pd.Series(dtype=float)
    global_stds = work[num_cols].std(numeric_only=True) if num_cols else pd.Series(dtype=float)
    cluster_summaries: dict[str, dict] = {}

    for cid, grp in work.groupby("_cluster_id"):
        size = int(len(grp))
        share = float(size / len(work)) if len(work) > 0 else 0.0

        summary: dict[str, Any] = {
            "size": size,
            "share": round(share, 4),
            "numeric_features": {},
            "categorical_features": {},
            "examples": [],
        }

        for col in num_cols:
            summary["numeric_features"][col] = {
                "mean_cluster": float(grp[col].mean()),
                "mean_global": float(global_means.get(col, np.nan)),
                "std_global": float(global_stds.get(col, np.nan)),
            }

        for col in cat_cols:
            vc = grp[col].astype(str).value_counts().head(3)
            if vc.empty:
                continue
            total = int(vc.sum())
            summary["categorical_features"][col] = [
                {
                    "value": str(v),
                    "count": int(cnt),
                    "share": float(cnt / total) if total > 0 else 0.0,
                }
                for v, cnt in vc.items()
            ]

        ex_cols = [cluster_col] + num_cols + cat_cols
        ex_df = grp[ex_cols].head(max_examples_per_cluster)
        summary["examples"] = ex_df.to_dict(orient="records")
        cluster_summaries[str(cid)] = summary

    return cluster_summaries


def _eda_cluster_numeric_signals(summary: dict[str, Any], top_n: int = 2) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for col, stats in (summary.get("numeric_features") or {}).items():
        mean_cluster = float(stats.get("mean_cluster", np.nan))
        mean_global = float(stats.get("mean_global", np.nan))
        std_global = float(stats.get("std_global", np.nan))
        if not np.isfinite(mean_cluster) or not np.isfinite(mean_global):
            continue
        delta = mean_cluster - mean_global
        if np.isfinite(std_global) and std_global > 1e-9:
            strength = abs(delta / std_global)
        else:
            strength = abs(delta)
        direction = "wyzsza" if delta > 0 else "nizsza"
        if strength >= 0.25:
            signals.append(
                {
                    "col": str(col),
                    "direction": direction,
                    "strength": float(strength),
                    "mean_cluster": mean_cluster,
                    "mean_global": mean_global,
                }
            )
    signals.sort(key=lambda x: x["strength"], reverse=True)
    return signals[:top_n]


def _eda_cluster_code_like_id(cid: str) -> bool:
    cid = str(cid).strip()
    if not cid:
        return False
    if cid.lower() in {"true", "false"}:
        return True
    if cid.isdigit():
        return True
    if len(cid) <= 3 and cid.upper() == cid:
        return True
    return False


def _eda_force_factual_cluster_output(cluster_summaries: dict[str, dict], cluster_col: str) -> bool:
    cids = [str(cid) for cid in cluster_summaries.keys()]
    if not cids:
        return False
    if cluster_col == EDA_TEMP_CLUSTER_COL:
        return True
    if len(cids) > 20:
        return True
    code_like_share = sum(1 for cid in cids if _eda_cluster_code_like_id(cid)) / max(len(cids), 1)
    return code_like_share >= 0.5


def _deterministic_cluster_labels(cluster_summaries: dict[str, dict], object_label: str) -> dict[str, dict]:
    label = (object_label or "obiekty").strip() or "obiekty"
    out: dict[str, dict] = {}
    for cid, summary in cluster_summaries.items():
        cid_str = str(cid)
        size = int(summary.get("size", 0))
        share_pct = 100.0 * float(summary.get("share", 0.0))
        numeric_signals = _eda_cluster_numeric_signals(summary, top_n=2)

        dominant_cat = None
        for col, items in (summary.get("categorical_features") or {}).items():
            if items:
                top = items[0]
                dominant_cat = {
                    "col": str(col),
                    "value": str(top.get("value", "")),
                    "share": 100.0 * float(top.get("share", 0.0)),
                }
                break

        prefix = cid_str if cid_str.lower().startswith("roboczy segment") else f"Segment {cid_str}"
        if numeric_signals:
            main_signal = numeric_signals[0]
            name = f"{prefix} — {main_signal['direction']} {main_signal['col']}"
        elif dominant_cat:
            name = f"{prefix} — dominuje {dominant_cat['col']}"
        else:
            name = prefix

        name = name[:80].strip(" -")
        short_label = prefix if len(prefix) <= 32 else prefix[:32]

        desc_parts = [
            f"Segment obejmuje ok. {size:,} rekordow ({share_pct:.1f}% analizowanej proby {label})."
        ]
        if numeric_signals:
            metrics_txt = []
            for signal in numeric_signals:
                metrics_txt.append(
                    f"{signal['direction']} srednia {signal['col']} "
                    f"({signal['mean_cluster']:.3g} vs {signal['mean_global']:.3g})"
                )
            desc_parts.append("Na tle proby widac " + " oraz ".join(metrics_txt) + ".")
        if dominant_cat:
            desc_parts.append(
                f"Wsrod cech kategorycznych dominuje {dominant_cat['col']}={dominant_cat['value']} "
                f"({dominant_cat['share']:.0f}%)."
            )

        out[cid_str] = {
            "name": name,
            "short_label": short_label,
            "description": " ".join(desc_parts),
        }
    return out


def _eda_cluster_column_candidates(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []

    nunique = df.nunique(dropna=True)
    max_unique = min(100, max(12, int(np.sqrt(max(len(df), 1)) * 2)))
    keyword_hits: list[str] = []
    low_card_hits: list[str] = []

    for col in df.columns:
        series = df[col]
        unique_n = int(nunique.get(col, 0))
        if unique_n < 2:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        logical = _infer_logical_type(series)
        name_l = str(col).lower()
        name_hit = any(tok in name_l for tok in [
            "cluster", "segment", "segm", "klaster", "grupa", "group", "label", "class"
        ])

        if logical == "id_like" and not name_hit:
            continue
        if name_l.startswith("is_") and not name_hit:
            continue

        if name_hit:
            keyword_hits.append(col)
            continue

        if unique_n <= max_unique and logical in {"categorical", "numeric"}:
            low_card_hits.append(col)

    ordered: list[str] = []
    for col in keyword_hits + low_card_hits:
        if col not in ordered:
            ordered.append(col)
    return ordered


EDA_TEMP_CLUSTER_COL = "__eda_temp_cluster_auto"
EDA_TEMP_CLUSTER_STATE_KEY = "eda_temp_cluster_state_v1"


def _eda_temp_cluster_feature_candidates(df: pd.DataFrame, max_features: int = 6) -> list[str]:
    if df is None or df.empty:
        return []

    ranked: list[tuple[float, str]] = []
    for col in df.columns:
        if col == EDA_TEMP_CLUSTER_COL:
            continue

        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if pd.api.types.is_bool_dtype(series):
            continue
        if _infer_logical_type(series) != "numeric":
            continue

        name_l = str(col).lower()
        if name_l.startswith("is_"):
            continue

        non_na = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(non_na.notna().sum()) < min(max(len(df) // 20, 25), len(df)):
            continue

        unique_n = int(non_na.nunique(dropna=True))
        if unique_n < 4:
            continue

        q1 = float(non_na.quantile(0.25))
        q3 = float(non_na.quantile(0.75))
        iqr = q3 - q1
        std = float(non_na.std(skipna=True))
        score = float(max(abs(iqr), abs(std)))
        if not np.isfinite(score) or score <= 0:
            continue

        ranked.append((score, col))

    ranked.sort(key=lambda x: x[0], reverse=True)
    ordered: list[str] = []
    for _, col in ranked:
        if col not in ordered:
            ordered.append(col)
    return ordered[:max_features]


def _eda_prepare_temp_cluster_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    work = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = work.median(numeric_only=True)
    work = work.fillna(med)

    q1 = work.quantile(0.25, numeric_only=True)
    q3 = work.quantile(0.75, numeric_only=True)
    iqr = (q3 - q1).replace(0, np.nan)
    std = work.std(numeric_only=True).replace(0, np.nan)
    scale = iqr.fillna(std).fillna(1.0).replace(0, 1.0)

    scaled = ((work - med) / scale).clip(-6, 6)
    return scaled.to_numpy(dtype=float), med, scale


def _eda_numpy_kmeans(
    x: np.ndarray,
    k: int,
    random_state: int = 0,
    max_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(x.shape[0])
    if n_rows == 0:
        raise ValueError("Brak rekordow do klasteryzacji.")

    k = max(2, min(int(k), n_rows))
    rng = np.random.default_rng(random_state)

    first_idx = int(rng.integers(0, n_rows))
    centers = [x[first_idx]]
    dist_sq = np.sum((x - centers[0]) ** 2, axis=1)

    for _ in range(1, k):
        total = float(dist_sq.sum())
        if not np.isfinite(total) or total <= 0:
            idx = int(rng.integers(0, n_rows))
        else:
            probs = dist_sq / total
            idx = int(rng.choice(n_rows, p=probs))
        centers.append(x[idx])
        dist_sq = np.minimum(dist_sq, np.sum((x - centers[-1]) ** 2, axis=1))

    centers_arr = np.vstack(centers)
    labels = np.zeros(n_rows, dtype=int)

    for _ in range(max_iter):
        d2 = np.sum((x[:, None, :] - centers_arr[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1).astype(int)

        new_centers = []
        for cid in range(k):
            mask = new_labels == cid
            if np.any(mask):
                new_centers.append(x[mask].mean(axis=0))
            else:
                new_centers.append(x[int(rng.integers(0, n_rows))])
        new_centers_arr = np.vstack(new_centers)

        if np.array_equal(new_labels, labels) or np.allclose(new_centers_arr, centers_arr, atol=1e-4):
            labels = new_labels
            centers_arr = new_centers_arr
            break

        labels = new_labels
        centers_arr = new_centers_arr

    return labels, centers_arr


def _eda_build_temp_clusters(
    df: pd.DataFrame,
    k: int = 4,
    max_features: int = 6,
    fit_sample_size: int = 5000,
) -> dict[str, Any]:
    feature_cols = _eda_temp_cluster_feature_candidates(df, max_features=max_features)
    if len(feature_cols) < 2:
        raise ValueError(
            "Do utworzenia roboczych klastrow potrzebujemy przynajmniej 2 sensownych cech liczbowych."
        )

    x_all, _, _ = _eda_prepare_temp_cluster_matrix(df, feature_cols)
    if x_all.shape[0] < 8:
        raise ValueError("Za malo rekordow do zbudowania roboczych klastrow.")

    rng = np.random.default_rng(0)
    if len(x_all) > fit_sample_size:
        fit_idx = np.sort(rng.choice(len(x_all), size=fit_sample_size, replace=False))
        x_fit = x_all[fit_idx]
    else:
        fit_idx = np.arange(len(x_all))
        x_fit = x_all

    labels_fit, centers = _eda_numpy_kmeans(x_fit, k=k, random_state=0)

    if len(x_all) != len(x_fit):
        d2_all = np.sum((x_all[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels_all = np.argmin(d2_all, axis=1).astype(int)
    else:
        labels_all = labels_fit

    if len(np.unique(labels_all)) < 2:
        score = x_all.mean(axis=1)
        ranked = pd.Series(score).rank(method="first")
        q = min(max(2, int(k)), int(ranked.nunique(dropna=True)))
        if q < 2:
            raise ValueError("Nie udalo sie utworzyc co najmniej 2 roznych klastrow roboczych.")
        labels_all = pd.qcut(ranked, q=q, labels=False, duplicates="drop").astype(int).to_numpy()
        centers = np.vstack([x_all[labels_all == cid].mean(axis=0) for cid in sorted(np.unique(labels_all))])

    unique_labels = sorted(np.unique(labels_all).tolist())
    centers_by_label = {
        int(cid): centers[int(cid)] if int(cid) < len(centers) else x_all[labels_all == int(cid)].mean(axis=0)
        for cid in unique_labels
    }
    order = sorted(unique_labels, key=lambda cid: float(np.nanmean(centers_by_label[cid])))
    mapping = {old: idx + 1 for idx, old in enumerate(order)}
    labels_pretty = [f"Roboczy segment {mapping[int(cid)]}" for cid in labels_all]

    counts = pd.Series(labels_pretty).value_counts().sort_index()
    return {
        "col_name": EDA_TEMP_CLUSTER_COL,
        "labels": labels_pretty,
        "features_used": feature_cols,
        "clusters_count": int(len(set(labels_pretty))),
        "cluster_counts": {str(k): int(v) for k, v in counts.items()},
        "fit_rows": int(len(x_fit)),
        "n_rows": int(len(df)),
    }


def _eda_attach_temp_clusters_to_df(df: pd.DataFrame) -> pd.DataFrame:
    state = st.session_state.get(EDA_TEMP_CLUSTER_STATE_KEY) or {}
    if not state:
        return df

    df_key = _df_cache_fingerprint(df)
    labels = state.get("labels")
    col_name = str(state.get("col_name") or EDA_TEMP_CLUSTER_COL)

    if state.get("df_key") != df_key:
        return df
    if not isinstance(labels, list) or len(labels) != len(df):
        return df
    if col_name in df.columns:
        return df

    out = df.copy()
    out[col_name] = pd.Series(labels, index=out.index, dtype="object")
    _eda_record_checkpoint(
        "eda.cluster.temp.cache_hit",
        cluster_col=col_name,
        clusters_count=int(state.get("clusters_count", 0)),
        features_used=state.get("features_used", []),
    )
    return out


def _eda_create_temp_clusters_and_store(
    df: pd.DataFrame,
    k: int = 4,
    max_features: int = 6,
) -> dict[str, Any]:
    df_key = _df_cache_fingerprint(df)
    _eda_record_checkpoint("eda.cluster.temp.start", target_k=int(k), df_key=df_key)

    result = _eda_build_temp_clusters(df, k=k, max_features=max_features)
    state = {
        "df_key": df_key,
        **result,
    }
    st.session_state[EDA_TEMP_CLUSTER_STATE_KEY] = state
    _eda_record_checkpoint(
        "eda.cluster.temp.ready",
        cluster_col=result["col_name"],
        clusters_count=int(result["clusters_count"]),
        features_used=result["features_used"],
        fit_rows=int(result["fit_rows"]),
    )
    return state


# Production cluster narration with gate/checkpoints and deterministic fallback.
def _describe_clusters_with_llm(
    df: pd.DataFrame,
    cluster_col: str,
    max_features: int = 8,
    max_examples_per_cluster: int = 3,
    model: str = "gpt-4o-mini",
    object_label: str | None = None,
) -> dict[str, dict]:
    label = (object_label or "obiekty").strip()
    if len(label) > 40:
        label = "obiekty"

    cluster_summaries = _build_cluster_summaries(
        df,
        cluster_col,
        max_features=max_features,
        max_examples_per_cluster=max_examples_per_cluster,
    )
    fallback_labels = _deterministic_cluster_labels(cluster_summaries, label)
    render_key = str(("v1", cluster_col, label, len(cluster_summaries), model))

    _eda_record_checkpoint(
        "eda.cluster.start",
        cluster_col=cluster_col,
        object_label=label,
        openai_model=model,
        render_key=render_key,
    )
    _eda_record_checkpoint(
        "eda.cluster.stats_payload",
        cluster_col=cluster_col,
        clusters_count=len(cluster_summaries),
        stats_keys=list(cluster_summaries.keys()),
    )

    debug = {
        "cluster_col": cluster_col,
        "render_key": render_key,
        "model": model,
        "raw_text": None,
        "gate_ok": False,
        "gate_reasons": [],
        "used_fallback": False,
        "error": None,
        "labels_count": len(cluster_summaries),
    }

    if not cluster_summaries:
        debug.update(
            {
                "used_fallback": True,
                "gate_reasons": ["cluster_col: brak danych do opisu klastrow"],
                "final_source": "fallback_deterministic_selected",
            }
        )
        st.session_state["eda_cluster_debug_v1"] = debug
        _eda_record_checkpoint(
            "eda.cluster.final_source",
            src="fallback_deterministic_selected",
            used_fallback=True,
            gate_reasons=debug["gate_reasons"],
        )
        _eda_register_exec_result("cluster_labels", "fallback_deterministic_selected", model=model)
        return fallback_labels

    if not _get_env_or_secret("OPENAI_API_KEY"):
        debug.update(
            {
                "used_fallback": True,
                "gate_reasons": ["openai_api_key: brak klucza API"],
                "final_source": "fallback_deterministic_selected",
            }
        )
        st.session_state["eda_cluster_debug_v1"] = debug
        _eda_record_checkpoint(
            "eda.cluster.final_source",
            src="fallback_deterministic_selected",
            used_fallback=True,
            gate_reasons=debug["gate_reasons"],
        )
        _eda_register_exec_result("cluster_labels", "fallback_deterministic_selected", model=model)
        return fallback_labels

    prompt = (
        "Jestes doswiadczonym analitykiem danych.\n"
        "Na podstawie krotkich statystyk klastrow przygotuj zwiezle, praktyczne nazwy i opisy segmentow.\n"
        f"Opisuj segmenty takich {label}.\n\n"
        "Dla kazdego klastra przygotuj:\n"
        '  - "name": pelna, czytelna nazwa segmentu,\n'
        '  - "short_label": bardzo krotka etykieta (2-4 slowa),\n'
        '  - "description": 2-3 zdania opisu po polsku.\n\n'
        "Zasady jakosci:\n"
        "- Opisuj tylko to, co wynika bezposrednio z payloadu.\n"
        "- Nie dopowiadaj motywacji, emocji, stylu zycia, geograficznych stereotypow ani przyczyn biznesowych,\n"
        "  jesli nie sa jawnie obecne w danych.\n"
        "- Nazwy i opisy kotwicz w wielkosci segmentu, dominujacych cechach kategorycznych\n"
        "  i odchyleniach cech liczbowych wzgledem proby.\n"
        "- Jesli cluster_id jest kodem, skrotem, liczba albo wartoscia techniczna, zachowaj go w nazwie\n"
        "  i nie zamieniaj na nowa narracyjna etykiete.\n\n"
        "Zwroc tylko jeden obiekt JSON o strukturze:\n"
        "{\n"
        '  "<cluster_id>": {"name": "...", "short_label": "...", "description": "..."}\n'
        "}\n\n"
        "Uzyj dokladnie tych samych kluczy cluster_id, ktore dostajesz w danych wejsciowych. "
        "Nie zamieniaj ich na nowe numery ani skroty.\n\n"
        "Nie dodawaj zadnych komentarzy poza JSON-em.\n\n"
        "Oto dane klastrow:\n"
    )
    prompt += json.dumps(cluster_summaries, ensure_ascii=False, indent=2)

    try:
        resp = _openai_chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = (resp.choices[0].message.content or "").strip()
        debug["raw_text"] = raw
        parsed = _eda_parse_json_like(raw)

        if isinstance(parsed, dict) and cluster_col == EDA_TEMP_CLUSTER_COL:
            parsed_norm = dict(parsed)
            for cid in fallback_labels.keys():
                cid_str = str(cid)
                cid_lower = cid_str.lower()
                suffix = ""
                if cid_lower.startswith("roboczy segment "):
                    suffix = cid_str.split()[-1].strip()
                candidate_keys = [suffix, f"segment {suffix}", f"roboczy segment {suffix}"] if suffix else []
                if cid_str not in parsed_norm:
                    for alt_key in candidate_keys:
                        alt_val = parsed.get(alt_key)
                        if isinstance(alt_val, dict):
                            parsed_norm[cid_str] = alt_val
                            break
            parsed = parsed_norm

        gate_reasons: list[str] = []
        repaired = False
        force_factual = _eda_force_factual_cluster_output(cluster_summaries, cluster_col)
        factualized = False
        final_labels: dict[str, dict] = {}

        for cid, fallback in fallback_labels.items():
            item = parsed.get(cid) if isinstance(parsed, dict) else None
            if not isinstance(item, dict):
                gate_reasons.append(f"{cid}: brak wpisu JSON")
                final_labels[cid] = fallback
                repaired = True
                continue

            name = _eda_norm_text(item.get("name"))
            short_label = _eda_norm_text(item.get("short_label"))
            description = _eda_norm_text(item.get("description"))
            local_reasons = []

            if len(name) < 3 or len(name) > 80:
                local_reasons.append("name")
            if len(short_label) < 3 or len(short_label.split()) > 4:
                local_reasons.append("short_label")
            if len(description) < 24 or len(description) > 320:
                local_reasons.append("description")

            if local_reasons:
                gate_reasons.append(f"{cid}: niepoprawne pola ({', '.join(local_reasons)})")
                final_labels[cid] = fallback
                repaired = True
                continue

            out_name = name
            out_short_label = short_label
            out_description = fallback["description"]

            if description != out_description:
                factualized = True

            if force_factual:
                out_name = fallback["name"]
                out_short_label = fallback["short_label"]
                factualized = True

            final_labels[cid] = {
                "name": out_name,
                "short_label": out_short_label,
                "description": out_description,
            }

        debug["gate_ok"] = len(gate_reasons) == 0
        debug["gate_reasons"] = gate_reasons
        if factualized:
            final_source = "llm_selected_repaired_factualized" if repaired else "llm_selected_factualized"
        else:
            final_source = "llm_selected_repaired" if repaired else "llm_selected"
        debug["final_source"] = final_source
        debug["factualized"] = factualized
        debug["force_factual"] = force_factual
        st.session_state["eda_cluster_debug_v1"] = debug
        _eda_record_checkpoint(
            "eda.cluster.final_source",
            src=final_source,
            used_fallback=False,
            gate_reasons=gate_reasons,
            labels_count=len(final_labels),
            factualized=factualized,
            force_factual=force_factual,
        )
        _eda_register_exec_result("cluster_labels", final_source, model=model, clusters=len(final_labels))
        return final_labels

    except Exception as exc:
        debug["error"] = str(exc)
        debug["used_fallback"] = True
        debug["gate_reasons"] = ["fallback: blad generacji lub parsowania"]
        debug["final_source"] = "fallback_deterministic_selected"
        st.session_state["eda_cluster_debug_v1"] = debug
        _eda_record_checkpoint(
            "eda.cluster.final_source",
            src="fallback_deterministic_selected",
            used_fallback=True,
            error=str(exc),
            gate_reasons=debug["gate_reasons"],
        )
        _eda_register_exec_result("cluster_labels", "fallback_deterministic_selected", model=model, clusters=len(fallback_labels))
        return fallback_labels


# --- OpenAI TTS: listy wyboru ---
# --- OpenAI: głosy podzielone wg płci (praktyczny podział) ---
OPENAI_VOICES = {
    "female": ["nova", "shimmer", "coral", "ballad", "sage", "marin"],
    "male":   ["alloy", "echo", "fable", "onyx", "verse", "ash", "cedar"],
}
DEFAULT_FEMALE_VOICE = "shimmer"
DEFAULT_MALE_VOICE   = "verse"

# Jeśli kiedyś dodasz inne modele TTS, dopisz je tutaj. Domyślnie używamy
# szybszego modelu, a dokładniejszy model można wymusić przez OPENAI_TTS_MODEL.
OPENAI_TTS_MODELS = [os.getenv("OPENAI_TTS_MODEL", "tts-1").strip() or "tts-1"]
OPENAI_TTS_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TTS_TIMEOUT_SECONDS", "30"))
OPENAI_TTS_SPEED = float(os.getenv("OPENAI_TTS_SPEED", "1.08"))
STAGE2_TTS_MAX_CHARS = int(os.getenv("STAGE2_TTS_MAX_CHARS", "900"))


def _plain_text_for_tts(markdown_text: str, max_chars: int = STAGE2_TTS_MAX_CHARS) -> str:
    """Clean and shorten markdown so TTS stays fast and reads naturally."""
    text = str(markdown_text or "").strip()
    if not text:
        return ""

    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"[*_~>#|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if max_chars <= 0 or len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rstrip()
    sentence_end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
    if sentence_end > int(max_chars * 0.55):
        clipped = clipped[: sentence_end + 1]
    else:
        word_end = clipped.rfind(" ")
        if word_end > int(max_chars * 0.55):
            clipped = clipped[:word_end].rstrip() + "."
    return clipped


# ==== AUDIO PLAYER (jeden kontrolowany odtwarzacz z opcjonalnym autoplay) ====
def _render_tts_audio_player(audio_bytes: bytes, mime: str = "audio/mpeg", autoplay: bool = False):
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    autoplay_attr = " autoplay" if autoplay else ""
    html = f"""
    <audio controls{autoplay_attr} preload="auto" style="width:100%; height:38px;">
      <source src="data:{mime};base64,{b64}" type="{mime}">
    </audio>
    """
    st.markdown(html, unsafe_allow_html=True)


def _browser_tts_chunks(text: str, max_chars: int = 220) -> list[str]:
    """Split text into short browser-speech chunks to avoid truncated narration."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    sentences = re.split(r"(?<=[.!?])\s+", text)

    def _flush(value: str) -> None:
        value = value.strip()
        if value:
            chunks.append(value)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                _flush(current)
                current = ""
            words = sentence.split()
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if len(candidate) <= max_chars:
                    piece = candidate
                else:
                    _flush(piece)
                    piece = word
            _flush(piece)
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            _flush(current)
            current = sentence

    _flush(current)
    return chunks


def _render_browser_tts(markdown_text: str, voice_hint: str = "female", autoplay: bool = True) -> None:
    """Render a non-blocking browser TTS player that reads the full summary."""
    plain_text = _plain_text_for_tts(markdown_text, max_chars=0)
    chunks = _browser_tts_chunks(plain_text)
    if not chunks:
        return

    root_id = f"stage2-browser-tts-{_hash_key(plain_text, voice_hint)[:12]}"
    html = """
    <div id="__ROOT_ID__" class="stage2-browser-tts">
      <style>
        .stage2-browser-tts {
          display: flex;
          align-items: center;
          gap: 10px;
          min-height: 38px;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: #31333f;
        }
        .stage2-browser-tts .status {
          flex: 1;
          font-size: 14px;
          color: #5f6b7a;
        }
        .stage2-browser-tts button {
          border: 1px solid #d8dee9;
          background: #ffffff;
          border-radius: 7px;
          padding: 7px 12px;
          cursor: pointer;
          color: #31333f;
          font-size: 14px;
        }
        .stage2-browser-tts button:hover {
          border-color: #ff4b4b;
          color: #ff4b4b;
        }
      </style>
      <div class="status">Przygotowuję narrację audio...</div>
      <button type="button" class="replay">Odtwórz ponownie</button>
      <button type="button" class="stop">Stop</button>
    </div>
    <script>
    (function () {
      const root = document.getElementById("__ROOT_ID__");
      if (!root) return;

      const chunks = __CHUNKS_JSON__;
      const voiceHint = __VOICE_HINT_JSON__;
      const shouldAutoplay = __AUTOPLAY_JSON__;
      const status = root.querySelector(".status");
      const replay = root.querySelector(".replay");
      const stop = root.querySelector(".stop");
      let idx = 0;
      let active = false;
      let voicesReady = false;

      function setStatus(text) {
        if (status) status.textContent = text;
      }

      function pickVoice() {
        const voices = window.speechSynthesis.getVoices() || [];
        const plVoices = voices.filter((v) => (v.lang || "").toLowerCase().startsWith("pl"));
        const pool = plVoices.length ? plVoices : voices;
        if (!pool.length) return null;

        const hinted = pool.find((v) => {
          const name = (v.name || "").toLowerCase();
          return voiceHint === "male"
            ? /male|mężczy|męski|jan|piotr|marek|adam/.test(name)
            : /female|kobiet|żeński|ewa|zosia|paulina|maria/.test(name);
        });
        return hinted || pool[0];
      }

      function speakNext() {
        if (!active) return;
        if (idx >= chunks.length) {
          active = false;
          setStatus("Narracja zakończona.");
          return;
        }

        const utterance = new SpeechSynthesisUtterance(chunks[idx]);
        const voice = pickVoice();
        if (voice) utterance.voice = voice;
        utterance.lang = (voice && voice.lang) || "pl-PL";
        utterance.rate = 1.02;
        utterance.pitch = voiceHint === "male" ? 0.95 : 1.02;
        utterance.onend = function () {
          idx += 1;
          speakNext();
        };
        utterance.onerror = function () {
          idx += 1;
          speakNext();
        };

        setStatus("Czytam podsumowanie " + (idx + 1) + "/" + chunks.length + "...");
        window.speechSynthesis.speak(utterance);
      }

      function start() {
        if (!("speechSynthesis" in window)) {
          setStatus("Ta przeglądarka nie obsługuje automatycznej narracji.");
          return;
        }
        window.speechSynthesis.cancel();
        idx = 0;
        active = true;
        setTimeout(speakNext, 80);
      }

      function cancel() {
        active = false;
        if ("speechSynthesis" in window) window.speechSynthesis.cancel();
        setStatus("Narracja zatrzymana.");
      }

      replay.addEventListener("click", start);
      stop.addEventListener("click", cancel);
      window.addEventListener("beforeunload", cancel);

      function boot() {
        if (voicesReady) return;
        voicesReady = true;
        setStatus("Narracja gotowa.");
        if (shouldAutoplay) setTimeout(start, 250);
      }

      if ("speechSynthesis" in window) {
        if (window.speechSynthesis.getVoices().length) {
          boot();
        } else {
          window.speechSynthesis.onvoiceschanged = boot;
          setTimeout(boot, 600);
        }
      } else {
        setStatus("Ta przeglądarka nie obsługuje automatycznej narracji.");
      }
    })();
    </script>
    """
    html = (
        html.replace("__ROOT_ID__", root_id)
        .replace("__CHUNKS_JSON__", json.dumps(chunks, ensure_ascii=False))
        .replace("__VOICE_HINT_JSON__", json.dumps(str(voice_hint or "").lower()))
        .replace("__AUTOPLAY_JSON__", "true" if autoplay else "false")
    )
    components.html(html, height=58)

# ==== TL;DR fallback (deterministyczny) ====
def _make_eda_summary_text(
    source_name: str,
    readiness_score: int,
    duplicates_count: int,
    global_missing_pct: float,
    auto_drop_candidates: list[str],
    prep_report: dict | None,
) -> str:
    drops = ", ".join(auto_drop_candidates) if auto_drop_candidates else "brak"
    wins = []
    if prep_report:
        if prep_report.get("duplicates_removed", 0) > 0:
            wins.append(f"usunięto duplikaty: {prep_report['duplicates_removed']}")
        if prep_report.get("outlier_flags"):
            wins.append(f"zaflagowano outliery: {len(prep_report['outlier_flags'])} kolumn")
    wins_txt = ("; " + "; ".join(wins)) if wins else ""
    return (
        f"Jakość danych w zbiorze „{source_name}” oceniamy na {readiness_score}/100. "
        f"Braki globalne wynoszą ok. {global_missing_pct:.1f}%, a liczba pełnych duplikatów to {duplicates_count}. "
        f"Kandydaci do wyłączenia: {drops}. Dane po czyszczeniu są gotowe do trenowania{wins_txt}."
    )

# =====================  UTILITIES  =======================

def _stage2_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


def _iter_ingest_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        cfg, _ = load_config()
        roots.append(resolve_artifacts_dir(cfg) / "ingest")
    except Exception:
        pass

    repo_root = _stage2_repo_root()
    roots.extend(
        [
            repo_root / "artifacts" / "ingest",
            repo_root / "app" / "artifacts" / "ingest",
            Path.cwd() / "artifacts" / "ingest",
        ]
    )

    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _resolve_existing_artifact_path(raw_path: Any, *, expect_dir: bool = False) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None

    raw = Path(text)
    candidates: list[Path] = [raw]
    if not raw.is_absolute():
        repo_root = _stage2_repo_root()
        candidates.extend([repo_root / raw, Path.cwd() / raw])
        for ingest_root in _iter_ingest_roots():
            candidates.append(ingest_root / raw)

    for candidate in candidates:
        try:
            if expect_dir and candidate.is_dir():
                return candidate.resolve()
            if not expect_dir and candidate.is_file():
                return candidate.resolve()
        except Exception:
            continue
    return None


def _valid_latest_artifacts(info: dict | None) -> dict | None:
    if not isinstance(info, dict):
        return None
    parquet_path = info.get("parquet_path") or info.get("csv_path")
    resolved_parquet = _resolve_existing_artifact_path(parquet_path)
    if resolved_parquet is None:
        return None

    valid = dict(info)
    valid["parquet_path"] = str(resolved_parquet)
    valid.pop("csv_path", None)

    meta_path = _resolve_existing_artifact_path(valid.get("meta_path"))
    if meta_path is not None:
        valid["meta_path"] = str(meta_path)
    run_dir = _resolve_existing_artifact_path(valid.get("run_dir"), expect_dir=True)
    if run_dir is not None:
        valid["run_dir"] = str(run_dir)
    else:
        valid["run_dir"] = str(resolved_parquet.parent)
    return valid


def _remember_latest_artifacts(info: dict | None) -> dict | None:
    """Keep the last known-good Stage 1 pointer across widget reruns."""
    valid = _valid_latest_artifacts(info)
    if valid:
        st.session_state["latest_artifacts"] = valid
        st.session_state["_stage2_last_valid_artifacts"] = valid
        return valid
    return None


def _restore_latest_artifacts_from_disk() -> dict | None:
    """Recover Stage 1 artifacts after Streamlit reruns/reconnects."""
    ingest_roots = _iter_ingest_roots()

    for ingest_root in ingest_roots:
        pointer = ingest_root / "latest_artifacts.json"
        if pointer.exists():
            try:
                info = json.loads(pointer.read_text(encoding="utf-8"))
                valid = _remember_latest_artifacts(info)
                if valid:
                    return valid
            except Exception:
                pass

    run_dirs: list[Path] = []
    for ingest_root in ingest_roots:
        try:
            run_dirs.extend([p for p in ingest_root.iterdir() if p.is_dir()])
        except Exception:
            continue
    run_dirs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

    for run_dir in run_dirs:
        try:
            parquet_files = sorted(
                run_dir.glob("*__full_masked.parquet"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not parquet_files:
                parquet_files = sorted(
                    run_dir.glob("*.parquet"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            if not parquet_files:
                continue

            meta_files = sorted(
                run_dir.glob("*__meta.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            parquet_path = parquet_files[0]
            meta_path = meta_files[0] if meta_files else None
            meta_json: dict[str, Any] = {}
            if meta_path and meta_path.exists():
                try:
                    meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta_json = {}

            info = {
                "parquet_path": str(parquet_path),
                "meta_path": str(meta_path) if meta_path else "",
                "run_dir": str(run_dir),
                "n_rows": int(meta_json.get("n_rows") or 0),
                "n_cols": int(meta_json.get("n_cols") or 0),
                "pii_masked": bool(meta_json.get("pii_masked", True)),
                "source_name": meta_json.get("source_name") or parquet_path.name,
                "source_kind": meta_json.get("source_kind") or "uploaded",
                "timestamp": meta_json.get("timestamp") or "",
            }
            valid = _remember_latest_artifacts(info)
            if valid:
                try:
                    for ingest_root in ingest_roots[:1]:
                        ingest_root.mkdir(parents=True, exist_ok=True)
                        (ingest_root / "latest_artifacts.json").write_text(
                            json.dumps(valid, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                except Exception:
                    pass
                return valid
        except Exception:
            continue
    return None


def _load_latest_dataset(max_rows: int | None = None) -> Tuple[pd.DataFrame | None, dict | None, str | None]:
    """Wczytuje zbiór z poprzedniego kroku (PARQUET only)."""
    latest_info = _remember_latest_artifacts(st.session_state.get("latest_artifacts"))
    if latest_info is None:
        latest_info = _remember_latest_artifacts(st.session_state.get("_stage2_last_valid_artifacts"))
    if latest_info is None:
        latest_info = _restore_latest_artifacts_from_disk()
    if latest_info is None:
        cached_df = st.session_state.get("_stage2_loaded_df")
        cached_info = _remember_latest_artifacts(st.session_state.get("_stage2_loaded_info"))
        if isinstance(cached_df, pd.DataFrame) and cached_info is not None:
            return cached_df.copy(), cached_info, None
        return None, None, (
            "Brak gotowych danych w pamięci aplikacji. "
            "Przejdź do zakładki 'Analiza Danych' i kliknij "
            "'Przelicz teraz (pełny zbiór)'."
        )

    parquet_path = latest_info.get("parquet_path") or latest_info.get("csv_path")
    resolved_path = _resolve_existing_artifact_path(parquet_path)
    if resolved_path is None:
        return None, latest_info, (
            f"Nie mogę znaleźć pliku z danymi: {parquet_path!r}. "
            "Najpierw przejdź do 'Analiza Danych' i kliknij "
            "'Przelicz teraz (pełny zbiór)'."
        )

    path = resolved_path
    latest_info = dict(latest_info)
    latest_info["parquet_path"] = str(path)
    latest_info.pop("csv_path", None)
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        read_limit = int(max_rows) if max_rows and max_rows > 0 else None
        # Smart preview for huge files to avoid hangs even when sample mode is off.
        if read_limit is None and size_mb >= 500:
            read_limit = 500_000
        mtime = path.stat().st_mtime
        cache_sig = f"{path}|{mtime}|rows={read_limit or 'all'}"
        if st.session_state.get("_stage2_loaded_sig") == cache_sig:
            cached_df = st.session_state.get("_stage2_loaded_df")
            cached_info = _remember_latest_artifacts(st.session_state.get("_stage2_loaded_info") or latest_info)
            if isinstance(cached_df, pd.DataFrame) and cached_info is not None:
                return cached_df.copy(), cached_info, None
        sample_note = None
        if read_limit:
            df = _df_from_parquet(path, max_rows=read_limit)
            sample_note = "rownomierna probke z calego pliku"
        else:
            df = _df_from_parquet(path, max_rows=None)
        if read_limit:
            st.caption(
                f"Tryb probki EDA: wczytano {len(df):,} wierszy ({sample_note}) "
                f"z pliku ~{size_mb:,.1f} MB."
            )
        st.session_state["_stage2_loaded_df"] = df.copy()
        st.session_state["_stage2_loaded_info"] = latest_info
        st.session_state["_stage2_loaded_sig"] = cache_sig
    except Exception as e:
        return None, latest_info, f"Nie udało się wczytać danych z PARQUET: {e}"

    return df, latest_info, None


def _calc_global_missing_pct(df: pd.DataFrame) -> float:
    """Procent braków w całym dataframe, licząc NaNy + puste stringi."""
    if df.shape[0] == 0 or df.shape[1] == 0:
        return 0.0
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        missing_cells += int(df[obj_cols].eq("").sum().sum())
    return round(100.0 * (missing_cells / max(total_cells, 1)), 2)

# ───────────────────── ROLES & HERO CHARTS (TASK-AWARE EDA) ─────────────────────


# -------------------- Stage 2 column semantics --------------------
_DATE_PART_NAMES = {
    "year", "month", "day", "hour", "minute", "second", "week", "quarter",
    "dayofweek", "weekday", "weekday_name", "weekofyear", "isocalendar_week",
}
_DATE_PART_SUFFIXES = (
    "_year", "_month", "_day", "_hour", "_minute", "_second", "_week",
    "_quarter", "_dow", "_weekday", "_dayofweek",
)
_ID_CODE_TOKENS = (
    "id", "uuid", "guid", "code", "kod", "nr", "no", "number", "invoice",
    "receipt", "stockcode", "customer", "client", "konto", "account", "zip",
    "postal", "phone", "telefon", "email", "mail",
)
_TECH_PREFIXES = ("is_", "flag_", "has_", "__", "tmp_", "temp_")
_VALUE_NAME_PRIORITY = (
    "target", "line_value", "value", "sales", "sprzed", "revenue", "amount",
    "total", "price", "qty", "quantity", "cost", "margin", "profit", "score",
)
_TYPE_DETECTION_SAMPLE_ROWS = int(os.getenv("EDA_TYPE_DETECTION_SAMPLE_ROWS", "1000"))


def _col_name_l(col: Any) -> str:
    return str(col or "").strip().lower()


def _is_date_part_column(col: Any) -> bool:
    name = _col_name_l(col)
    return name in _DATE_PART_NAMES or any(name.endswith(sfx) for sfx in _DATE_PART_SUFFIXES)


def _looks_like_identifier_name(col: Any) -> bool:
    name = _col_name_l(col)
    if not name:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", name)
    parts = [p for p in re.split(r"[^a-z0-9]+", name) if p]

    short_id_parts = {"id", "uuid", "guid", "code", "kod", "nr", "no"}
    exact_compact = {
        "id", "uuid", "guid", "code", "kod", "nr", "no",
        "stockcode", "postcode", "postalcode", "zipcode",
        "email", "mail", "phone", "telefon",
    }
    id_context = {"invoice", "receipt", "order", "customer", "client", "account", "konto"}

    if compact in exact_compact:
        return True
    if any(part in short_id_parts for part in parts):
        return True
    if any(tok in compact for tok in ("uuid", "guid", "email", "phone", "telefon", "stockcode", "postcode", "postalcode")):
        return True
    if any(tok in compact for tok in ("invoice", "receipt")):
        return True
    if "number" in parts and any(part in id_context for part in parts):
        return True
    if compact.endswith("number") and any(tok in compact for tok in id_context):
        return True
    return False


def _is_bool_like_series(series: pd.Series) -> bool:
    try:
        if pd.api.types.is_bool_dtype(series):
            return True
        vals = series.dropna().unique()
        if len(vals) == 0 or len(vals) > 2:
            return False
        as_text = {str(v).strip().lower() for v in vals}
        return as_text.issubset({"0", "1", "true", "false", "yes", "no", "tak", "nie"})
    except Exception:
        return False


def _numeric_series_for_candidate(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    try:
        return _coerce_to_numeric_special(series).replace([np.inf, -np.inf], np.nan)
    except Exception:
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _sample_non_null_for_detection(series: pd.Series, max_rows: int | None = None) -> pd.Series:
    """Small deterministic sample for cheap type detection on text-heavy columns."""
    limit = int(max_rows or _TYPE_DETECTION_SAMPLE_ROWS)
    try:
        s = series.dropna()
        if len(s) <= limit:
            return s
        # Head is intentional here: it avoids full random sampling work on large frames.
        return s.head(limit)
    except Exception:
        return series.dropna().head(limit)


def _is_numeric_measure_candidate(
    df: pd.DataFrame,
    col: Any,
    *,
    min_non_null: int = 2,
    allow_low_cardinality: bool = False,
) -> bool:
    """Return True only for numeric columns that are useful analytical measures.

    Stage 2 charts should not use technical IDs, codes, booleans or derived date parts as
    normal measures. Those columns make correlations, outlier charts and model-prep advice
    look precise while actually being misleading.
    """
    if df is None or col not in df.columns:
        return False
    s = df[col]
    name = _col_name_l(col)
    if not name or name.startswith(_TECH_PREFIXES) or _is_date_part_column(col):
        return False
    if pd.api.types.is_datetime64_any_dtype(s) or _is_bool_like_series(s):
        return False
    if not pd.api.types.is_numeric_dtype(s) and not _try_detect_numeric_from_object(s):
        return False

    sn = _numeric_series_for_candidate(s)
    non_na = sn.dropna()
    if len(non_na) < min_non_null:
        return False
    nunique = int(non_na.nunique(dropna=True))
    if nunique <= 1:
        return False
    if not allow_low_cardinality and nunique <= 2:
        return False

    source_non_null = s.dropna()
    unique_ratio = float(source_non_null.nunique(dropna=True) / max(len(source_non_null), 1))
    integer_like = False
    try:
        sample = non_na.head(5000)
        integer_like = bool(not sample.empty and float((sample - sample.round()).abs().max()) < 1e-9)
    except Exception:
        integer_like = False

    if _looks_like_identifier_name(col):
        return False
    if integer_like and unique_ratio > 0.90 and len(non_na) >= 50:
        return False
    return True


def _numeric_measure_candidates(
    df: pd.DataFrame,
    *,
    min_non_null: int = 2,
    allow_low_cardinality: bool = False,
) -> List[str]:
    if df is None or df.empty:
        return []
    return [
        c for c in df.columns
        if _is_numeric_measure_candidate(
            df, c,
            min_non_null=min_non_null,
            allow_low_cardinality=allow_low_cardinality,
        )
    ]


def _categorical_feature_candidates(
    df: pd.DataFrame,
    *,
    max_cardinality: int = 200,
) -> List[str]:
    if df is None or df.empty:
        return []
    out: List[str] = []
    n = max(len(df), 1)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) or _is_date_part_column(c):
            continue
        if not pd.api.types.is_numeric_dtype(s) and _try_detect_datetime_from_object(s):
            continue
        if _looks_like_identifier_name(c) or _col_name_l(c).startswith(_TECH_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(s) and _is_numeric_measure_candidate(df, c):
            continue
        nunique = int(s.nunique(dropna=True))
        unique_ratio = nunique / n
        if 2 <= nunique <= max_cardinality and unique_ratio <= 0.50:
            out.append(c)
    return out


def _ordered_analysis_columns(df: pd.DataFrame) -> List[str]:
    measures = _numeric_measure_candidates(df, min_non_null=2)
    col_pos = {c: i for i, c in enumerate(df.columns)}
    measures = sorted(measures, key=lambda c: (-_measure_priority_score(c), col_pos.get(c, 10**9)))
    cats = _categorical_feature_candidates(df)
    ordered: List[str] = []
    for c in measures + cats + list(df.columns):
        if c not in ordered:
            ordered.append(c)
    return ordered


def _measure_priority_score(col: Any) -> int:
    name = _col_name_l(col)
    score = 0
    for rank, token in enumerate(_VALUE_NAME_PRIORITY):
        if token in name:
            score = max(score, 100 - rank)
    return score


def _pick_preferred_numeric_measure(cols: List[str]) -> str | None:
    if not cols:
        return None
    scored = []
    for pos, col in enumerate(cols):
        scored.append((_measure_priority_score(col), -pos, col))
    scored.sort(reverse=True)
    return scored[0][2]
def _infer_eda_roles(df: pd.DataFrame, latest_info: dict | None) -> dict:
    """Heurystyczne wykrywanie typu zadania i ról kolumn.
    Korzysta z meta (task) oraz source_name z Etapu 1, a jak trzeba – z samych danych.
    """
    from typing import Any
    import json
    import os

    roles: dict[str, Any] = {
        "task": "regression",   # domyślnie
        "target": None,
        "time_col": None,
        "cluster_col": None,
    }

    latest_info = latest_info or {}

    # ===================== 1. TASK z meta / source_name =====================

    meta_task = None
    meta_path = latest_info.get("meta_path")
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
            raw = str(meta_json.get("task", "") or "")
            if raw:
                meta_task = raw.lower()
        except Exception:
            meta_task = None

    src_raw = str(latest_info.get("source_name", "") or "")
    src_low = src_raw.lower()

    # preferuj task z meta, jeśli jest – inaczej patrz na source_name
    raw_task = meta_task or src_low
    task = None

    # polskie + angielskie warianty
    if any(t in raw_task for t in ["szereg czasowy", "time series", "czasowy"]) or re.search(r"(^|[^a-z])ts([^a-z]|$)", raw_task):
        task = "time_series"
    elif any(t in raw_task for t in ["klasteryzacja", "cluster", "segment"]):
        task = "clustering"
    elif any(t in raw_task for t in ["klasyfikacja", "classification"]):
        task = "classification"
    elif any(t in raw_task for t in ["regresja", "regression"]):
        task = "regression"

    # ===================== 2. Heurystyki z samych danych =====================

    nunique = df.nunique(dropna=True)

    # kandydaci na kolumnę klastrów
    cluster_candidates = [
        c for c in df.columns
        if any(tok in c.lower() for tok in ["cluster", "segment", "segm", "klaster", "grupa", "group"])
    ]

    # maloliczne kolumny (kandydaci na target klasyfikacyjny / kolumne klastrow)
    small_card_cols = []
    small_card_limit = min(50, max(10, df.shape[0] // 10))
    for c in df.columns:
        if _is_date_part_column(c) or _looks_like_identifier_name(c):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if 1 < int(nunique.get(c, 0)) <= small_card_limit:
            small_card_cols.append(c)
    if task is None:
        if cluster_candidates:
            task = "clustering"
        elif small_card_cols:
            task = "classification"
        else:
            task = "regression"

    roles["task"] = task

    # ===================== 3. time_col (dla szeregów czasowych) =====================

    time_col = None
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        time_col = dt_cols[0]
    else:
        for c in df.columns:
            cl = c.lower()
            if any(tok in cl for tok in ["date", "data", "czas", "time", "timestamp", "ds"]):
                time_col = c
                break
    roles["time_col"] = time_col

    # zapamiętaj, czy meta/source sugerowały szereg czasowy
    roles["ts_suspected"] = (roles.get("task") == "time_series")

    # jeśli sugerowano TS, ale nie ma jawnej kolumny czasu,
    # nie wymuszaj tego automatycznie (user może wybrać time_series ręcznie w override)
    if roles["ts_suspected"] and roles["time_col"] is None:
        roles["task"] = "regression"


    # ===================== 4. cluster_col (dla klasteryzacji) =====================

    cluster_col = None
    if cluster_candidates:
        cluster_col = cluster_candidates[0]
    elif task == "clustering" and small_card_cols:
        # jeżeli nie ma oczywistej nazwy, weź pierwszą małoliczną
        cluster_col = small_card_cols[0]
    roles["cluster_col"] = cluster_col

    # ===================== 5. target (dla klasyfikacji / regresji) =====================

    target = None
    if task in ("classification", "regression"):
        # kandydaci po nazwie
        name_cands = [
            c for c in df.columns
            if any(tok in c.lower() for tok in [
                "target", "label", "class", "klasa", "y_", "_y", "wynik", "outcome", "score"
            ])
        ]
        if name_cands:
            target = name_cands[0]
        else:
            if task == "classification" and small_card_cols:
                target = small_card_cols[0]
            elif task == "regression":
                num_cols = _numeric_measure_candidates(df, min_non_null=2, allow_low_cardinality=True)
                target = _pick_preferred_numeric_measure(num_cols)
    roles["target"] = target

    return roles

def _infer_logical_type(series: pd.Series) -> str:
    """
    Heurystyka 'logicznego' typu:
      numeric / datetime / id_like / text_long / categorical / boolean / date_part
    """
    col_name = getattr(series, "name", "")
    if _is_date_part_column(col_name):
        return "date_part"
    if _is_bool_like_series(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_null = series.dropna()
    nunique = int(series.nunique(dropna=True))
    ratio_unique = nunique / max(len(non_null), 1)

    if pd.api.types.is_numeric_dtype(series):
        integer_like = False
        try:
            sample = pd.to_numeric(non_null, errors="coerce").dropna().head(5000)
            integer_like = bool(not sample.empty and float((sample - sample.round()).abs().max()) < 1e-9)
        except Exception:
            integer_like = False
        if _looks_like_identifier_name(col_name) or (integer_like and ratio_unique > 0.90 and len(non_null) >= 50):
            return "id_like"
        return "numeric"

    sample = _sample_non_null_for_detection(series, 200).astype(str)
    if len(sample) > 0:
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        if parsed.notna().mean() > 0.9:
            return "datetime"

    if _looks_like_identifier_name(col_name) or ratio_unique > 0.9:
        return "id_like"

    text_sample = _sample_non_null_for_detection(series, 1000).astype(str)
    avg_len = (text_sample.str.len().mean() if len(text_sample) > 0 else 0)
    if avg_len and avg_len > 50:
        return "text_long"

    return "categorical"

def _try_detect_numeric_from_object(series: pd.Series) -> bool:
    """Czy obiektowa kolumna wygląda na liczbową po naszych regułach specjalnych?"""
    if series.dtype != object:
        return False
    if _is_date_part_column(getattr(series, "name", "")) or _looks_like_identifier_name(getattr(series, "name", "")):
        return False
    s = _sample_non_null_for_detection(series).astype(str)
    if s.empty:
        return False

    # spróbuj parsery specjalne
    parsed = _coerce_to_numeric_special(s)
    if parsed.notna().mean() > 0.9:
        return True

    return False

def _augment_numeric_derivatives_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dla potrzeb podglądu/analizy (Sekcja 2–6) dorzuć kolumny liczbowe pochodne:
    - object: HH:MM:SS/MM:SS -> <col>_sec, min/km -> <col>_sec_per_km, % -> <col>_pct, km/h -> <col>_kmh, waluty/liczby -> <col>_num
    - datetime: sekundy od północy -> <col>_sec
    Nie dotyka oryginałów; zwraca kopię df z dodatkowymi kolumnami.
    """
    out = df.copy()
    for col in list(df.columns):
        s = df[col]

        # --- object: użyj Twoich parserów specjalnych ---
        if s.dtype == object:
            sample = _sample_non_null_for_detection(s).astype(str)
            if sample.empty:
                continue
            sample_threshold = max(5, int(0.6 * len(sample)))
            ss = None

            def _full_strings() -> pd.Series:
                nonlocal ss
                if ss is None:
                    ss = s.astype(str)
                return ss

            sec_sample = sample.map(_parse_duration_to_seconds)
            if pd.notna(sec_sample).sum() >= sample_threshold:
                sec = _full_strings().map(_parse_duration_to_seconds)
                out[f"{col}_sec"] = pd.to_numeric(sec, errors="coerce")
                continue

            pace_sample = sample.map(_parse_pace_min_per_km_to_sec_per_km)
            if pd.notna(pace_sample).sum() >= sample_threshold:
                pace = _full_strings().map(_parse_pace_min_per_km_to_sec_per_km)
                out[f"{col}_sec_per_km"] = pd.to_numeric(pace, errors="coerce")
                continue

            perc_sample = sample.map(_parse_percentage)
            if pd.notna(perc_sample).sum() >= sample_threshold:
                perc = _full_strings().map(_parse_percentage)
                out[f"{col}_pct"] = pd.to_numeric(perc, errors="coerce")
                continue

            kmh_sample = sample.map(_parse_kmh)
            if pd.notna(kmh_sample).sum() >= sample_threshold:
                kmh = _full_strings().map(_parse_kmh)
                out[f"{col}_kmh"] = pd.to_numeric(kmh, errors="coerce")
                continue

            cur_sample = sample.map(_parse_currency_or_plain_number)
            if pd.notna(cur_sample).sum() >= sample_threshold:
                cur = _full_strings().map(_parse_currency_or_plain_number)
                out[f"{col}_num"] = pd.to_numeric(cur, errors="coerce")
                continue

        # --- datetime: zrób sekundę od północy ---
        if pd.api.types.is_datetime64_any_dtype(s):
            try:
                sec = (s.dt.hour * 3600 + s.dt.minute * 60 + s.dt.second).astype("float")
                # jeśli to faktycznie wygląda na czas (ma zmienność), dodaj kolumnę
                if pd.notna(sec).sum() >= max(5, int(0.6 * len(s.dropna()))):
                    out[f"{col}_sec"] = sec
            except Exception:
                pass

    return out

def _try_detect_datetime_from_object(series: pd.Series) -> bool:
    if series.dtype == object:
        sample = _sample_non_null_for_detection(series, 200).astype(str)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        return parsed.notna().mean() > 0.9
    return False

# ---------- Parsowanie specjalnych formatów: H:M:S, min/km, %, waluty, jednostki ----------

_DURATION_RE = re.compile(r"^\s*(\d{1,2}):([0-5]?\d):([0-5]?\d)\s*$")   # HH:MM:SS
_MMSS_RE     = re.compile(r"^\s*([0-5]?\d):([0-5]?\d)\s*$")             # MM:SS (np. 4:35)
_PERCENT_RE  = re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*%\s*$")
CURR_SIGNS   = "€$£złPLN"  # można rozszerzyć
_UNIT_KMH_RE = re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*(?:km/?h|kmh)\s*$", re.I)
_UNIT_MINKM  = re.compile(r"^\s*([0-5]?\d):([0-5]?\d)\s*(?:min/?km|min/km)\s*$", re.I)

def _parse_duration_to_seconds(val: str) -> float | None:
    """ 'HH:MM:SS' -> sekundy, 'MM:SS' -> sekundy """
    if not isinstance(val, str):
        return None
    m = _DURATION_RE.match(val)
    if m:
        h, mi, s = map(int, m.groups())
        return float(h*3600 + mi*60 + s)
    m = _MMSS_RE.match(val)
    if m:
        mi, s = map(int, m.groups())
        return float(mi*60 + s)
    return None

def _parse_percentage(val: str) -> float | None:
    if not isinstance(val, str):
        return None
    m = _PERCENT_RE.match(val)
    if m:
        v = m.group(1).replace(",", ".")
        try:
            return float(v) / 100.0
        except Exception:
            return None
    return None

def _parse_currency_or_plain_number(val: str) -> float | None:
    """ '1 234,56 zł' / '$1,234.56' / '1 234' -> float """
    if not isinstance(val, str):
        return None
    s = val.strip()
    # usuń separatory tysięcy (spacje, niełamliwe, kropki) — ale zostaw kropkę dziesiętną
    s = s.replace("\u202f", " ").replace("\xa0", " ")
    # usuń waluty/słowa
    for token in CURR_SIGNS.split():
        s = s.replace(token, "")
    s = re.sub(r"[€$£]|zł|PLN", "", s, flags=re.I)
    s = s.strip()

    # polski przecinek na kropkę
    s = s.replace(" ", "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    # jeśli wciąż są literki → nie parsujemy
    if re.search(r"[^\d\.\-+eE]", s):
        return None
    try:
        return float(s)
    except Exception:
        return None

def _parse_kmh(val: str) -> float | None:
    if not isinstance(val, str):
        return None
    m = _UNIT_KMH_RE.match(val)
    if not m:
        return None
    v = m.group(1).replace(",", ".")
    try:
        return float(v)
    except Exception:
        return None

def _parse_pace_min_per_km_to_sec_per_km(val: str) -> float | None:
    """ '4:35 min/km' -> sekundy na km (275.0) """
    if not isinstance(val, str):
        return None
    m = _UNIT_MINKM.match(val) or _MMSS_RE.match(val)
    if not m:
        return None
    mi, s = map(int, m.groups())
    return float(mi*60 + s)

def _coerce_to_numeric_special(series: pd.Series) -> pd.Series:
    """Najpierw spróbuj duration, % , min/km, km/h, waluty/liczby. Zwraca numeric z NaN dla nieparsowalnych."""
    s = series.astype(str)

    # 1) HH:MM:SS / MM:SS
    parsed = s.map(_parse_duration_to_seconds)
    if pd.notna(parsed).sum() >= max(5, int(0.6*len(s.dropna()))):
        return pd.to_numeric(parsed, errors="coerce")

    # 2) pace 'min/km' lub MM:SS (jeśli w kolumnie są takie stringi)
    pace = s.map(_parse_pace_min_per_km_to_sec_per_km)
    if pd.notna(pace).sum() >= max(5, int(0.6*len(s.dropna()))):
        return pd.to_numeric(pace, errors="coerce")

    # 3) procenty
    perc = s.map(_parse_percentage)
    if pd.notna(perc).sum() >= max(5, int(0.6*len(s.dropna()))):
        return pd.to_numeric(perc, errors="coerce")

    # 4) km/h
    kmh = s.map(_parse_kmh)
    if pd.notna(kmh).sum() >= max(5, int(0.6*len(s.dropna()))):
        return pd.to_numeric(kmh, errors="coerce")

    # 5) waluty/„gołe” liczby z tysiącami itp.
    cur = s.map(_parse_currency_or_plain_number)
    if pd.notna(cur).sum() >= max(5, int(0.6*len(s.dropna()))):
        return pd.to_numeric(cur, errors="coerce")

    # fallback: oryginalne zachowanie (przecinek→kropka, spacje)
    s_norm = (series.dropna().astype(str)
              .str.replace(",", ".", regex=False)
              .str.replace(" ", "", regex=False))
    return pd.to_numeric(s_norm, errors="coerce")

def _analyze_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Raport jakości kolumn i heurystyki."""
    rows = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        dtype_raw = str(s.dtype)
        logical = _infer_logical_type(s)

        if s.dtype == object:
            null_cnt = int(s.isna().sum() + s.eq("").sum())
        else:
            null_cnt = int(s.isna().sum())
        null_pct = (null_cnt / max(n_rows, 1)) * 100.0 if n_rows else 0.0
        nunique = int(s.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(s):
            col_min = s.min(skipna=True)
            col_mean = s.mean(skipna=True)
            col_max = s.max(skipna=True)
            col_std = s.std(skipna=True)
        else:
            col_min = col_mean = col_max = col_std = None

        drop_candidate = False
        convert_numeric = False
        convert_datetime = False
        long_text_flag = False

        notes = []
        flag_icons = []

        if null_pct > 30:
            notes.append("Dużo braków (>30%).")
        if null_pct > 80 and logical != "numeric":
            notes.append("Prawie cała pusta (>80%).")
            drop_candidate = True
            flag_icons.append("🗑")
        if nunique <= 1:
            notes.append("Kolumna stała / bez zmienności.")
            drop_candidate = True
            if "🗑" not in flag_icons:
                flag_icons.append("🗑")
        if logical == "id_like":
            notes.append("Wygląda jak identyfikator (prawie każda wartość inna).")
            drop_candidate = True
            if "🗑" not in flag_icons:
                flag_icons.append("🗑")
        if _try_detect_numeric_from_object(s):
            notes.append("Wygląda na liczbę, ale jest stringiem → konwersja do float.")
            convert_numeric = True
            flag_icons.append("🔢")
        if _try_detect_datetime_from_object(s):
            notes.append("Wygląda na datę → konwersja do datetime i rozbicie (rok/miesiąc/dzień_tyg).")
            convert_datetime = True
            flag_icons.append("📅")
        if logical == "text_long":
            notes.append("Długi opis tekstowy / wolny tekst → raczej NLP niż klasyczny ML.")
            long_text_flag = True
            flag_icons.append("📝")

        rows.append(
            {
                "kolumna": col,
                "dtype_raw": dtype_raw,
                "logical_type": logical,
                "null_cnt": int(null_cnt),
                "null_pct": round(null_pct, 2),
                "n_unique": nunique,
                "min": col_min,
                "mean": col_mean,
                "max": col_max,
                "std": col_std,
                "drop_candidate": drop_candidate,
                "convert_numeric": convert_numeric,
                "convert_datetime": convert_datetime,
                "text_long_flag": long_text_flag,
                "flags": " ".join(flag_icons),
                "comment": " ".join(notes),
            }
        )

    info_df = pd.DataFrame(rows)
    high_null_cols = info_df.loc[info_df["null_pct"] > 30, "kolumna"].tolist()
    return info_df, high_null_cols



@contextmanager
def perf_step(label: str):
    """Context manager to micro-profile sub-steps."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        try:
            dt = time.perf_counter() - t0
            _PERF_LOGGER.info(f"[PERF] step:  {label} ({dt:.2f}s)")
        except Exception:
            pass


def _df_cache_fingerprint(df: pd.DataFrame, sample_rows: int = 50) -> str:
    """Fast(ish) fingerprint for caching expensive computations on the current DF sample."""
    try:
        cols = tuple(map(str, df.columns))
        dtypes = tuple(map(str, df.dtypes.astype(str).values))
        shape = tuple(df.shape)
        head = df.head(sample_rows)
        h = pd.util.hash_pandas_object(head, index=True).values
        sample_hash = int(h.sum())
        return f"{shape}|{hash(cols)}|{hash(dtypes)}|{sample_hash}"
    except Exception:
        try:
            return f"{df.shape}|{hash(tuple(map(str, df.columns)))}|{hash(tuple(map(str, df.dtypes.astype(str).values)))}"
        except Exception:
            return "df"


def stage2_cached(scope: str, key_parts: tuple, compute_fn):
    """Very small in-session cache keyed by (scope, key_parts)."""
    try:
        cache = st.session_state.setdefault("_stage2_cache_v1", {})
        k = (scope,) + tuple(key_parts)
        if k in cache:
            return cache[k]
        val = compute_fn()
        cache[k] = val
        return val
    except Exception:
        return compute_fn()



# ─────────────────────────────────────────────────────────────
# Stage2 micro-profiler (collect timings -> UI table)

STAGE2_PREP_TIMINGS_KEY = "stage2_prepare_timings_v1"

from contextlib import contextmanager
from time import perf_counter

@contextmanager
def perf_step_collect(step_name: str, bucket_key: str = STAGE2_PREP_TIMINGS_KEY):
    start = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - start
        try:
            st.session_state.setdefault(bucket_key, []).append({
                "step": step_name,
                "seconds": round(dt, 4),
            })
        except Exception:
            # never fail the app because of profiler
            pass
        _PERF_LOGGER.info(f"[INFO] [PERF] step:  {step_name} ({dt:.2f}s)")



@contextmanager
def stage2_prepare_step(step_name: str):
    """Alias for micro-profiler steps during 'Prepare data' pipeline."""
    with perf_step_collect(step_name):
        yield


def stage2_prepare_timing_df():
    """Return timings as a small DataFrame (never raises)."""
    try:
        import pandas as pd
        rows = st.session_state.get(STAGE2_PREP_TIMINGS_KEY) or []
        if not rows:
            return pd.DataFrame(columns=["step", "seconds", "ms"])
        df_t = pd.DataFrame(rows)
        if "seconds" in df_t.columns:
            df_t["ms"] = (df_t["seconds"] * 1000).round(0).astype(int)
        return df_t
    except Exception:
        import pandas as pd
        return pd.DataFrame(columns=["step", "seconds", "ms"])

def render_stage2_prepare_timings():
    rows = st.session_state.get(STAGE2_PREP_TIMINGS_KEY) or []
    if not rows:
        return
    try:
        import pandas as pd
        df_t = pd.DataFrame(rows)
        if not df_t.empty:
            df_t["ms"] = (df_t["seconds"] * 1000).round(0).astype(int)
            total = float(df_t["seconds"].sum())
            st.caption(f"⏱️ Timingi ostatniego przygotowania: {total:.2f}s")
            st_df_safe(df_t.sort_values("seconds", ascending=False), max_rows=50)
    except Exception:
        pass
def _build_correlation_report(
    df: pd.DataFrame,
    info_df: pd.DataFrame,
    threshold: float = 0.9,
) -> Tuple[alt.Chart | None, List[Dict[str, Any]], List[str]]:
    """Heatmapa i lista par korelacji dla kolumn numerycznych."""
    # Guard: tylko sensowne kolumny numeryczne
    numeric_cols = _numeric_measure_candidates(df, min_non_null=3)
    if len(numeric_cols) < 2:
        return None, [], []

    df_key = _df_cache_fingerprint(df)
    corr_source = pd.DataFrame({c: _numeric_series_for_candidate(df[c]) for c in numeric_cols})
    corr_mat = stage2_cached("Stage2/7", (df_key, "corr_mat", tuple(numeric_cols)), lambda: corr_source.corr(method="pearson").fillna(0.0))
    if corr_mat.empty:
        return None, [], []

    with perf_step('Stage2/7.corr_melt'):
        corr_melt = stage2_cached(
            "Stage2/7",
            (df_key, "corr_melt", tuple(numeric_cols)),
            lambda: (
                corr_mat.reset_index()
                .melt("index", var_name="col2", value_name="corr")
                .rename(columns={"index": "col1"})
            ),
        )

    heatmap = (
        alt.Chart(corr_melt)
        .mark_rect()
        .encode(
            x=alt.X("col1:N", title="", sort=numeric_cols),
            y=alt.Y("col2:N", title="", sort=numeric_cols),
            color=alt.Color(
                "corr:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1], domainMid=0),
                title="Pearson r",
            ),
            tooltip=[
                alt.Tooltip("col1:N", title="kolumna 1"),
                alt.Tooltip("col2:N", title="kolumna 2"),
                alt.Tooltip("corr:Q", title="r", format=".3f"),
            ],
        )
        .properties(width=500, height=500)
    )

    pairs = []
    high_corr_drop_candidates = []
    null_pct_map = info_df.set_index("kolumna")["null_pct"].to_dict()

    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            r_val = float(corr_mat.loc[c1, c2])
            n1 = null_pct_map.get(c1, 0.0)
            n2 = null_pct_map.get(c2, 0.0)
            suggest_drop = c1 if n1 > n2 else c2
            item = {"col1": c1, "col2": c2, "corr": r_val, "suggest_drop": suggest_drop}
            pairs.append(item)
            if abs(r_val) >= threshold:
                high_corr_drop_candidates.append(suggest_drop)

    high_corr_drop_candidates = sorted(list(set(high_corr_drop_candidates)))
    pairs_sorted = sorted(pairs, key=lambda d: abs(d["corr"]), reverse=True)
    return heatmap, pairs_sorted, high_corr_drop_candidates


def _detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    dups_mask = df.duplicated(keep="first")
    duplicates_count = int(dups_mask.sum())
    duplicates_pct = ((100.0 * duplicates_count / len(df)) if len(df) else 0.0)
    return {"duplicates_count": duplicates_count, "duplicates_pct": round(duplicates_pct, 2)}

# ---------- Pomocnicze formaty/liczby ----------

def _fmt_int(x: Any) -> str:
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return str(x)

def _num(x: Any) -> Any:
    try:
        xf = float(x)
        return round(xf, 4) if np.isfinite(xf) else None
    except Exception:
        return None

def slice_df(df: pd.DataFrame, where: dict[str, Any]) -> pd.DataFrame:
    """Prosty builder: where={'Survived': 1, 'Sex': 'female'} → zwróci pocięty df."""
    mask = pd.Series(True, index=df.index)
    for col, val in (where or {}).items():
        if col not in df.columns:
            continue
        if isinstance(val, (list, tuple, set)):
            mask &= df[col].isin(list(val))
        else:
            mask &= (df[col] == val)
    return df[mask].copy()

def hist_kde(df: pd.DataFrame, value_col: str, maxbins: int = 30) -> alt.Chart:
    """Histogram (liczebność) + KDE na jednym układzie."""
    use = pd.DataFrame({value_col: pd.to_numeric(df[value_col], errors="coerce")}).dropna()
    if use.empty:
        return alt.Chart(pd.DataFrame({"msg":["Brak danych"]})).mark_text(size=14).encode(text="msg")
    hist = (
        alt.Chart(use).mark_bar(opacity=0.85)
        .encode(
            x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=maxbins), title=value_col),
            y=alt.Y("count():Q", title="Liczebność"),
            tooltip=[alt.Tooltip("count():Q", title="liczebność")]
        ).properties(height=220)
    )
    kde = (
        alt.Chart(use).transform_density(value_col, as_=[value_col, "density"])
        .mark_line(size=2, opacity=0.9)
        .encode(x=f"{value_col}:Q", y=alt.Y("density:Q", title="Gęstość"))
        .properties(height=220)
    )
    return alt.layer(hist, kde).resolve_scale(y="independent")

# ---------- Czynniki eliminujące (Sekcja 5) ----------

def _col_factors(df: pd.DataFrame, info_df: pd.DataFrame, col: str) -> Dict[str, float]:
    """Wkład czynników: missing, id_like, engineered, variance_inv, outlier_ratio, cardinality, skew_abs."""
    # missing %
    try:
        missing_pct = float(info_df.loc[info_df["kolumna"] == col, "null_pct"].fillna(0).iloc[0]) / 100.0
    except Exception:
        missing_pct = 0.0

    row = info_df.set_index("kolumna").to_dict(orient="index").get(col, {})
    id_like = 1.0 if row.get("logical_type") == "id_like" else 0.0
    engineered = 1.0 if (col.endswith("_year") or col.endswith("_month") or col.endswith("_dow") or col.startswith("is_outlier_")) else 0.0

    if pd.api.types.is_numeric_dtype(df[col]):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        var = float(np.var(s)) if len(s) > 1 else 0.0
        # outlier ratio
        if len(s) > 3:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                outlier_ratio = 0.0
            else:
                lo, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_ratio = float(((s < lo) | (s > up)).mean())
        else:
            outlier_ratio = 0.0
        skew = float(pd.Series(s).skew()) if len(s) > 2 else 0.0
        try:
            s_num = pd.to_numeric(df[col], errors="coerce")
            rng = float(s_num.max() - s_num.min())
            variance_inv = 1.0 - min(1.0, (var / (rng**2 + 1e-9)))
            variance_inv = max(0.0, variance_inv)
        except Exception:
            variance_inv = 0.0
    else:
        variance_inv = 0.0
        outlier_ratio = 0.0
        skew = 0.0

    card = df[col].nunique(dropna=True) / max(len(df[col]), 1)

    return {
        "missing_pct": missing_pct,
        "id_like": id_like,
        "engineered": engineered,
        "variance_inv": variance_inv,
        "outlier_ratio": outlier_ratio,
        "cardinality": float(card),
        "skew_abs": float(abs(skew)),
    }

def _elimination_score(f: Dict[str, float]) -> float:
    """Łączny score (priorytety: braki, ID-like, engineered...)."""
    w = {
        "missing_pct": 0.40,
        "id_like": 0.15,
        "engineered": 0.10,
        "variance_inv": 0.10,
        "outlier_ratio": 0.10,
        "cardinality": 0.05,
        "skew_abs": 0.05,
    }
    return sum(w[k] * f.get(k, 0.0) for k in w)

# ---------- Rozkład kolumn ----------

def _numeric_distribution_details(s: pd.Series, col_name: str) -> Tuple[alt.VConcatChart, pd.DataFrame, str, Dict[str, float]]:
    """Boxplot + histogram + metryki + komentarz
    - Tekst pokazuje REALNE whiskers (jak na wykresie), nie teoretyczne płotki.
    - Tabela metryk zawiera dodatkowo 'liczebność'.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    s_clean = s_num.dropna()

    if len(s_clean) == 0:
        empty_df = pd.DataFrame({"msg": ["brak danych"]})
        empty_chart = (
            alt.Chart(empty_df)
            .mark_text(text="Brak danych do wizualizacji", size=14)
            .encode()
            .properties(height=200)
        )
        stats_table = pd.DataFrame(
            {"metryka": ["liczebność", "min", "Q1 (25%)", "mediana (50%)", "Q3 (75%)", "max", "średnia", "odch.std."],
             "wartość": [0] + [None]*7}
        )
        return empty_chart, stats_table, "Brak danych liczbowych do analizy.", {}

    plot_df = pd.DataFrame({"value": s_clean})
    plot_df["label"] = col_name

    # Podstawowe statystyki
    desc = s_clean.describe(percentiles=[0.25, 0.5, 0.75])
    q1       = float(desc.get("25%",  np.nan))
    q3       = float(desc.get("75%",  np.nan))
    median_v = float(desc.get("50%",  np.nan))
    mean_v   = float(desc.get("mean", np.nan))
    min_v    = float(desc.get("min",  np.nan))
    max_v    = float(desc.get("max",  np.nan))
    std_v    = float(desc.get("std",  np.nan))
    cnt_v    = int(len(s_clean))

    # IQR + płotki Tukeya
    iqr = q3 - q1 if (np.isfinite(q1) and np.isfinite(q3)) else np.nan
    if np.isfinite(iqr) and iqr != 0:
        fence_lo = q1 - 1.5 * iqr
        fence_hi = q3 + 1.5 * iqr
    else:
        fence_lo, fence_hi = q1, q3

    # >>> REALNE WĄSY (whiskers) – to co rysuje boxplot:
    # dolny wąs = najmniejsza wartość >= fence_lo,
    # górny wąs = największa wartość <= fence_hi
    if np.isfinite(fence_lo):
        whisker_lo = float(s_clean[s_clean >= fence_lo].min()) if not s_clean[s_clean >= fence_lo].empty else float(s_clean.min())
    else:
        whisker_lo = float(s_clean.min())

    if np.isfinite(fence_hi):
        whisker_hi = float(s_clean[s_clean <= fence_hi].max()) if not s_clean[s_clean <= fence_hi].empty else float(s_clean.max())
    else:
        whisker_hi = float(s_clean.max())

    # Outliery: wszystko poniżej/ponad płotkami
    plot_df["is_outlier"] = False
    if np.isfinite(fence_lo) and np.isfinite(fence_hi):
        plot_df["is_outlier"] = (plot_df["value"] < fence_lo) | (plot_df["value"] > fence_hi)
        outlier_ratio = (plot_df["is_outlier"].mean() * 100.0) if len(plot_df) else 0.0
    else:
        outlier_ratio = 0.0
    outliers_df = plot_df[plot_df["is_outlier"]]

    # Widok osi X: nie kotwiczymy do 0, tylko do realnego zakresu danych.
    # Dzięki temu kolumny liczbowe o dużej bazie (np. identyfikatory / numery dokumentów)
    # nie "przyklejają się" do prawej strony wykresu.
    span = max_v - min_v if np.isfinite(max_v) and np.isfinite(min_v) else 0.0
    base_pad = max(span * 0.03, (iqr * 0.15 if np.isfinite(iqr) else 0.0), 1.0)
    if np.isfinite(min_v) and min_v >= 0:
        view_lo = max(0.0, min_v - min(base_pad, max(min_v * 0.10, 1.0)))
    else:
        view_lo = min_v - base_pad
    view_hi = max_v + base_pad if np.isfinite(max_v) else max_v
    if not (np.isfinite(view_lo) and np.isfinite(view_hi)) or view_hi <= view_lo:
        view_lo, view_hi = min_v, max_v
    x_scale = alt.Scale(domain=[float(view_lo), float(view_hi)], zero=False, nice=False)
    x_axis = alt.Axis(labelOverlap=False, format=",.0f")

    # Wykresy: boxplot (z outlierami) + histogram
    box_base = (
        alt.Chart(plot_df)
        .mark_boxplot(color="#1f77b4", outliers={"color": "#dc3545"})
        .encode(
            x=alt.X("value:Q", title=col_name, scale=x_scale, axis=x_axis),
            y=alt.Y("label:N", title=""),
        )
    )
    outlier_layer = (
        alt.Chart(outliers_df)
        .mark_point(filled=True, size=50, color="#dc3545")
        .encode(x=alt.X("value:Q", title=col_name, scale=x_scale, axis=x_axis), y=alt.Y("label:N", title=""))
    )
    box_layer = (box_base + outlier_layer).properties(height=70)

    hist_chart = (
        alt.Chart(plot_df)
        .mark_bar(color="#1f77b4")
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=col_name, scale=x_scale, axis=x_axis),
            y=alt.Y("count():Q", title="Liczebność"),
            tooltip=[alt.Tooltip("count():Q", title="liczebność"), alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=160)
    )
    combo_chart = alt.vconcat(box_layer, hist_chart).resolve_scale(x="shared")

    # Komentarz – pokazujemy REALNE WHISKERS (identyczne z wykresem), bez teoretycznych płotków
    phys_hint = ""
    if s_clean.min() >= 0 and np.isfinite(whisker_lo) and whisker_lo < 0:
        phys_hint = " (wartości ujemne mogą być niefizyczne dla tej cechy)"

    comment_text = (
        f"Szacowany bezpieczny zakres (wg IQR, realne wąsy boxplota): "
        f"[{whisker_lo:.2f} ; {whisker_hi:.2f}]{phys_hint}. "
        f"Około {outlier_ratio:.2f}% rekordów poza zakresem.\n\n"
        "Domyślnie **nie usuwamy** odstających; dodamy flagę `is_outlier_<kolumna>`. "
        "Na życzenie możesz przyciąć ekstremy (winsoryzacja) w Sekcji 7."
    )

    # Tabela metryk — dodajemy 'liczebność'
    stats_table = pd.DataFrame(
        {
            "metryka": ["liczebność", "min", "Q1 (25%)", "mediana (50%)", "Q3 (75%)", "max", "średnia", "odch.std."],
            "wartość": [cnt_v, _num(min_v), _num(q1), _num(median_v), _num(q3), _num(max_v), _num(mean_v), _num(std_v)],
        }
    )

    details = {
        "lower": whisker_lo,
        "upper": whisker_hi,
        "outlier_ratio_pct": outlier_ratio,
    }
    return combo_chart, stats_table, comment_text, details


def _numeric_by_category_charts(
    s_num: pd.Series,
    s_cat: pd.Series,
    col_num_name: str,
    col_cat_name: str,
    categories_keep: list[str] | None = None,
    maxbins: int = 30,
    mode: str = "facet",           # "facet" | "overlay"
    scale: str = "count",          # "count" | "share" | "kde"
    opacity: float = 0.55,
) -> tuple[alt.Chart, pd.DataFrame]:
    """
    Rozkład liczb (hist/KDE) wg kategorii z opcją facet/overlay.
    Zmiany:
    • 1A: delikatne odstępy między słupkami histogramu (binSpacing),
          wyrzucenie 'bounds' z properties (by nie wywoływać błędu 'Spec has no parameter named "bounds"').
    • 1B: dla 'share' (Udział w grupie) wymuszamy overlay i używamy stack='normalize' (100%).
    """
    # --- przygotowanie danych / statystyk ---
    v = pd.to_numeric(s_num, errors="coerce")
    g = s_cat.astype(str)
    dfp = pd.DataFrame({col_num_name: v, col_cat_name: g}).dropna()
    if categories_keep:
        dfp = dfp[dfp[col_cat_name].isin(categories_keep)]
    if dfp.empty:
        empty = (
            alt.Chart(pd.DataFrame({"msg": ["Brak danych po filtrze"]}))
            .mark_text(size=14)
            .encode(text="msg:N")
            .properties(height=120, width="container")
        )
        cols = [col_cat_name, "n", "min", "Q1", "median", "Q3", "max", "mean", "std"]
        return empty, pd.DataFrame(columns=cols)

    def _stats(x: pd.Series) -> dict:
        d = x.describe(percentiles=[.25, .5, .75])
        return {"min": float(d.get("min", np.nan)), "Q1": float(d.get("25%", np.nan)),
                "median": float(d.get("50%", np.nan)), "Q3": float(d.get("75%", np.nan)),
                "max": float(d.get("max", np.nan)), "mean": float(d.get("mean", np.nan)),
                "std": float(d.get("std", np.nan)), "n": int(x.notna().count())}

    stats_rows = []
    for k, sub in dfp.groupby(col_cat_name, dropna=False):
        stt = _stats(sub[col_num_name]); stt[col_cat_name] = k; stats_rows.append(stt)
    cols = [col_cat_name, "n", "min", "Q1", "median", "Q3", "max", "mean", "std"]
    stats_df = pd.DataFrame(stats_rows).reindex(columns=cols)
    cat_order = stats_df[col_cat_name].astype(str).tolist()
    if not stats_df.empty and col_cat_name in stats_df.columns:
        stats_df = stats_df.sort_values(col_cat_name, na_position="last")

    base = alt.Chart(dfp)

    # --- inteligentne binowanie / extent ---
    vals = pd.to_numeric(dfp[col_num_name], errors="coerce").dropna()
    unique_vals = np.sort(vals.unique())
    lo, hi = float(vals.min()), float(vals.max())
    bin_param = alt.Bin(maxbins=maxbins, extent=[lo, hi])

    # dyskretne zbiory (0/1 lub małe całkowite) → step=1 z buforem
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
        bin_param = alt.Bin(extent=[-0.5, 1.5], step=1)
    elif (len(unique_vals) <= 12
          and np.all(np.isfinite(unique_vals))
          and np.all(np.mod(unique_vals, 1) == 0)):
        loi, hii = float(unique_vals.min()), float(unique_vals.max())
        bin_param = alt.Bin(extent=[loi - 0.5, hii + 0.5], step=1)

    # --- 1B: „Udział w grupie” wymusza overlay (ma sens tylko jako 100% stacked na jednej osi) ---
    if scale == "share":
        mode = "overlay"

    # --- KDE ---
    if scale == "kde":
        density = base.transform_density(col_num_name, groupby=[col_cat_name],
                                         as_=[col_num_name, "density"])
        tips = [
            alt.Tooltip(f"{col_cat_name}:N", title="kategoria"),
            alt.Tooltip(f"{col_num_name}:Q", title=col_num_name, format=".2f"),
            alt.Tooltip("density:Q", title="gęstość", format=".3f"),
        ]

        if mode == "facet":
            chart = (
                density.mark_line(size=2, opacity=0.95)
                .encode(
                    x=alt.X(f"{col_num_name}:Q", title=col_num_name),
                    y=alt.Y("density:Q", title="Gęstość"),
                    color=alt.Color(f"{col_cat_name}:N", legend=None,
                                    scale=_scale_for_categories(cat_order)),
                    tooltip=tips,
                )
                .properties(height=160, width=FACET_CHART_WIDTH)
                .facet(row=alt.Row(f"{col_cat_name}:N",
                                   header=alt.Header(labelFontWeight="bold")))
            )
        else:
            chart = (
                density.mark_line(size=2, opacity=0.95)
                .encode(
                    x=alt.X(f"{col_num_name}:Q", title=col_num_name),
                    y=alt.Y("density:Q", title="Gęstość"),
                    color=alt.Color(f"{col_cat_name}:N", title=col_cat_name,
                                    scale=_scale_for_categories(cat_order)),
                    tooltip=tips,
                )
                .properties(height=260, width="container")
            )

        return chart, stats_df

    # --- COUNT / SHARE (histogram) — 100% stacked z poprawnie wycentrowanymi etykietami ---
    if scale == "share":
        # 1) policzmy count, total, share oraz środki binów
        share_data = (
            base
            .transform_bin(as_=["bin_start", "bin_end"], field=col_num_name, bin=bin_param)
            .transform_aggregate(
                count="count()", groupby=[col_cat_name, "bin_start", "bin_end"]
            )
            .transform_joinaggregate(
                total="sum(count)", groupby=["bin_start", "bin_end"]
            )
            .transform_calculate(
                share="datum.total == 0 ? 0 : datum.count / datum.total",
                bin_mid="(datum.bin_start + datum.bin_end) / 2"
            )
        )

        # 2) porządek kategorii zgodny z paletą (kluczowe dla poprawnej pozycji etykiet)
        #    – Vega-Lite będzie stackować wg 'order', a my kumulujemy wg tej samej kolejności
        _cat_domain_json = json.dumps(list(cat_order))
        _rank_expr = f"indexof({_cat_domain_json}, datum['{col_cat_name}'])"
        share_data = (
            share_data
            .transform_calculate(cat_rank=_rank_expr)
            .transform_window(
                cum_share="sum(share)",
                groupby=["bin_start", "bin_end"],
                sort=[alt.SortField("cat_rank", order="ascending")]
            )
            .transform_calculate(
                y_mid="datum.cum_share - datum.share/2"
            )
            .transform_filter("isFinite(datum.y_mid)")
        )

        # 3) słupki 100% stacked (z tooltipem także o udziale %)
        bars = (
            share_data.mark_bar(opacity=opacity)
            .encode(
                x=alt.X("bin_start:Q", title=col_num_name, bin="binned"),
                x2=alt.X2("bin_end:Q"),
                y=alt.Y(
                    "count:Q",
                    stack="normalize",
                    title="Udział w binie (100%)",
                    axis=alt.Axis(format=".0%")
                ),
                color=alt.Color(
                    f"{col_cat_name}:N",
                    legend=(None if mode == "facet" else alt.Legend(title=col_cat_name)),
                    scale=_scale_for_categories(cat_order)
                ),
                # ⬇️ wymuszamy identyczny porządek stacku jak w kumulacji
                order=alt.Order("cat_rank:Q", sort="ascending"),
                tooltip=[
                    alt.Tooltip(f"{col_cat_name}:N", title="kategoria"),
                    alt.Tooltip("count:Q", title="liczebność w binie"),
                    alt.Tooltip("share:Q", title="udział %", format=".1%")
                ],
            )
        )

        # 4) etykiety – czarne, wyśrodkowane pionowo w segmencie (y_mid)
        labels = (
            share_data
            .transform_filter("datum.share >= 0.04")  # próg widoczności, dopasuj wg potrzeb
            .mark_text(
                baseline="middle",
                align="center",
                fontSize=11,
                color="black"
            )
            .encode(
                x=alt.X("bin_mid:Q", title=col_num_name),
                y=alt.Y("y_mid:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                # ⬇️ ten sam porządek co stack (gdy wiele etykiet na tej samej x-pozycji)
                order=alt.Order("cat_rank:Q", sort="ascending"),
                text=alt.Text("share:Q", format=".0%")
            )
        )

        # 5) warstwa: słupki + etykiety (reszta – facet/overlay – bez zmian poniżej)
        mark_base = alt.layer(bars, labels)

    else:
        # ── Liczebność (klasyczny histogram) „obok siebie” ───────────────────────────
        # 1) pre-binning + środki i szerokości binów (wspólne dla wszystkich kategorii)
        binned = (
            base
            .transform_bin(
                as_=["bin_start", "bin_end"],
                field=col_num_name,
                bin=bin_param
            )
            .transform_calculate(
                bin_mid="(datum.bin_start + datum.bin_end) / 2",
                bin_w="(datum.bin_end - datum.bin_start)"
            )
        )

        # 2) stały porządek kategorii (rangi) zgodny z domeną palety
        _cat_domain_json = json.dumps(list(cat_order))
        _rank_expr = f"indexof({_cat_domain_json}, datum['{col_cat_name}'])"
        # ile kategorii łącznie (stała liczba; łatwiej niż liczyć w oknach)
        _k_total = len(cat_order)

        binned = (
            binned
            .transform_calculate(
                cat_rank=_rank_expr,
                k_total=str(_k_total),
                # szerokość „paska” zajmowanego przez wszystkie kategorie w jednym binie (0..1 * szerokość binu)
                band_fraction="0.85",
                # szerokość jednej kategorii wewnątrz binu
                cat_w="(toNumber(datum.band_fraction) * datum.bin_w) / toNumber(datum.k_total)",
                # środek paska danej kategorii (lewo/prawo od środka binu)
                x_mid="datum.bin_mid + ( (datum.cat_rank - (toNumber(datum.k_total)-1)/2 ) * datum.cat_w )",
                # krawędzie słupka dla kategorii
                x_left="datum.x_mid - datum.cat_w/2",
                x_right="datum.x_mid + datum.cat_w/2"
            )
        )

        # 3) rysujemy słupki: x/x2 = wyliczone krawędzie, y = count, bez stackowania
        mark_base = (
            binned.mark_bar(opacity=opacity)
            .encode(
                x=alt.X("x_left:Q", title=col_num_name),
                x2=alt.X2("x_right:Q"),
                y=alt.Y("count():Q", title="Liczebność", stack=None),
                color=alt.Color(
                    f"{col_cat_name}:N",
                    legend=(None if mode == "facet" else alt.Legend(title=col_cat_name)),
                    scale=_scale_for_categories(cat_order)
                ),
                order=alt.Order("cat_rank:Q", sort="ascending"),
                tooltip=[
                    alt.Tooltip(f"{col_cat_name}:N", title="kategoria"),
                    alt.Tooltip("count():Q", title="liczebność"),
                    alt.Tooltip("bin_start:Q", title="bin od", format=".2f"),
                    alt.Tooltip("bin_end:Q",   title="bin do", format=".2f"),
                ],
            )
        )

    if mode == "facet":
        chart = (
            mark_base
            .properties(height=160, width=FACET_CHART_WIDTH)
            .facet(row=alt.Row(f"{col_cat_name}:N",
                            header=alt.Header(labelFontWeight="bold")))
        )
    else:
        chart = mark_base.properties(height=260, width="container")

    # 👇 Teraz odstępy między „binami” działają
    chart = chart.configure_bar(binSpacing=4)
    return chart, stats_df



def _categorical_distribution_details(
    s: pd.Series, col_name: str, top_n: int = 20
) -> Tuple[alt.Chart, alt.Chart, pd.DataFrame, str]:
    """Top kategorie, coverage, tabela i komentarz."""
    ser = s.astype(str)
    ser = ser.replace("nan", np.nan).dropna()

    vc = ser.value_counts()
    total = vc.sum() if len(vc) > 0 else 0
    tmp_df = vc.head(top_n).reset_index()

    if tmp_df.shape[1] == 2:
        top_df = tmp_df.rename(columns={tmp_df.columns[0]: "kategoria", tmp_df.columns[1]: "liczebność"})
    else:
        top_df = pd.DataFrame(columns=["kategoria", "liczebność"])

    top_df["udział_%"] = np.where(total > 0, (top_df["liczebność"] / total) * 100.0, 0.0)
    cov_df = top_df.copy()
    cov_df["cum_count"] = cov_df["liczebność"].cumsum()
    cov_df["cum_pct"] = np.where(total > 0, (cov_df["cum_count"] / total) * 100.0, 0.0)
    cov_df["rank"] = np.arange(1, len(cov_df) + 1)

    bar_chart_top = (
        alt.Chart(top_df).mark_bar(color="#1f77b4")
        .encode(
            x=alt.X("liczebność:Q", title="Liczebność"),
            y=alt.Y("kategoria:N", sort="-x", title=col_name),
            tooltip=[alt.Tooltip("kategoria:N", title="kategoria"),
                     alt.Tooltip("liczebność:Q", title="liczebność"),
                     alt.Tooltip("udział_%:Q", title="udział %", format=".1f")],
        )
        .properties(height=250, title="TOP kategorie")
    )
    coverage_chart = (
        alt.Chart(cov_df).mark_line(point=True, color="#1f77b4")
        .encode(
            x=alt.X("rank:Q", title="Top N kategorii"),
            y=alt.Y("cum_pct:Q", title="Skumulowany udział %"),
            tooltip=[alt.Tooltip("rank:Q", title="Top N"),
                     alt.Tooltip("cum_pct:Q", title="Pokrycie [%]", format=".1f")],
        )
        .properties(height=250, title="Jak szybko TOP kategorie pokrywają zbiór?")
    )

    unique_ratio = (ser.nunique(dropna=True) / max(len(ser), 1) if len(ser) > 0 else 0.0)
    if len(top_df) > 0:
        top_cat = top_df.iloc[0]["kategoria"]
        top_share = (top_df.iloc[0]["liczebność"] / total) if total else 0.0
    else:
        top_cat = "(brak danych)"
        top_share = 0.0

    if unique_ratio > 0.9:
        comment_text = (
            "Ta kolumna ma prawie same unikalne wartości. Wygląda jak ID lub wolny opis. "
            "W klasycznym ML raczej niewiele wnosi bez przetwarzania (embeddingi NLP)."
        )
    elif top_share > 0.5:
        comment_text = (f"Wartość '{top_cat}' dominuje (>50%). To prawdopodobnie główny segment. "
                        "Model łatwo się tego nauczy, ale cecha ma ograniczoną różnorodność.")
    else:
        comment_text = "Rozkład kategorii jest dość zrównoważony — dobra, stabilna cecha kategoryczna."

    freq_table = top_df.copy()
    return bar_chart_top, coverage_chart, freq_table, comment_text


# ---------- Automatyczne przygotowanie danych ----------

def _auto_prepare_for_training(
    df_raw: pd.DataFrame,
    info_df: pd.DataFrame,
    decisions: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Czyszczenie i transformacje zgodnie z zasadami + wyborami usera."""
    df_work = df_raw.copy()

    drop_cols = decisions.get("drop_cols", [])
    winsorize_cols = decisions.get("winsorize_cols", [])
    remove_duplicates = decisions.get("remove_duplicates", True)

    df_work = df_work.drop(columns=[c for c in drop_cols if c in df_work.columns], errors="ignore")

    conversions: Dict[str, str] = {}
    imputations: Dict[str, str] = {}
    outlier_flags: List[str] = []
    winsorized: Dict[str, Dict[str, float]] = {}
    new_time_cols_added: List[str] = []

    info_map = info_df.set_index("kolumna").to_dict(orient="index")

    # Konwersje typów
    for col in list(df_work.columns):
        col_info = info_map.get(col, {})
        if col not in df_work.columns:
            continue

        # --- PRÓBY KONWERSJI SPECJALNYCH DLA OBIEKTÓW ---
        if df_work[col].dtype == object:
            s = df_work[col].astype(str)

            # 1) HH:MM:SS / MM:SS  -> <col>_sec
            sec = s.map(_parse_duration_to_seconds)
            if pd.notna(sec).sum() >= max(5, int(0.6*len(s.dropna()))):
                newc = f"{col}_sec"
                df_work[newc] = pd.to_numeric(sec, errors="coerce")
                conversions[newc] = "derived_seconds_from_time_string"
                continue  # nie ruszamy oryginału

            # 2) pace 'min/km' lub 'MM:SS' -> <col>_sec_per_km
            pace = s.map(_parse_pace_min_per_km_to_sec_per_km)
            if pd.notna(pace).sum() >= max(5, int(0.6*len(s.dropna()))):
                newc = f"{col}_sec_per_km"
                df_work[newc] = pd.to_numeric(pace, errors="coerce")
                conversions[newc] = "derived_pace_sec_per_km"
                continue

            # 3) procenty -> <col>_pct (0..1)
            perc = s.map(_parse_percentage)
            if pd.notna(perc).sum() >= max(5, int(0.6*len(s.dropna()))):
                newc = f"{col}_pct"
                df_work[newc] = pd.to_numeric(perc, errors="coerce")
                conversions[newc] = "derived_ratio_from_percent"
                continue

            # 4) km/h -> <col>_kmh
            kmh = s.map(_parse_kmh)
            if pd.notna(kmh).sum() >= max(5, int(0.6*len(s.dropna()))):
                newc = f"{col}_kmh"
                df_work[newc] = pd.to_numeric(kmh, errors="coerce")
                conversions[newc] = "derived_speed_kmh"
                continue

            # 5) waluty/liczby z tysiącami -> <col>_num
            cur = s.map(_parse_currency_or_plain_number)
            if pd.notna(cur).sum() >= max(5, int(0.6*len(s.dropna()))):
                newc = f"{col}_num"
                df_work[newc] = pd.to_numeric(cur, errors="coerce")
                conversions[newc] = "derived_numeric_from_currency/str"
                continue

        # --- ISTNIEJĄCE REGUŁY ---
        if col_info.get("convert_numeric", False):
            df_work[col] = (df_work[col].astype(str)
                            .str.replace(",", ".", regex=False)
                            .str.replace(" ", "", regex=False))
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
            conversions[col] = "text->float"

        if col_info.get("convert_datetime", False):
            parsed = pd.to_datetime(df_work[col], errors="coerce", utc=False)
            df_work[col] = parsed
            conversions[col] = "text->datetime(+year/month/dow)"
            df_work[f"{col}_year"] = parsed.dt.year
            df_work[f"{col}_month"] = parsed.dt.month
            df_work[f"{col}_dow"] = parsed.dt.dayofweek
            new_time_cols_added.extend([f"{col}_year", f"{col}_month", f"{col}_dow"])
                # + liczbowy wariant czasu: sekundy od północy
            try:
                sec = (parsed.dt.hour * 3600 + parsed.dt.minute * 60 + parsed.dt.second).astype("float")
                df_work[f"{col}_sec"] = sec
                conversions[f"{col}_sec"] = "derived_seconds_from_datetime"
            except Exception:
                pass


    # Imputacja braków
    for col in list(df_work.columns):
        s = df_work[col]
        if s.dtype == object:
            s = s.replace("", np.nan)
            df_work[col] = s
        if s.isna().any():
            if pd.api.types.is_numeric_dtype(s):
                med = s.median(skipna=True)
                df_work[col] = s.fillna(med)
                imputations[col] = f"median={med}"
            elif pd.api.types.is_datetime64_any_dtype(s):
                mode_val = s.dropna().mode()
                if not mode_val.empty:
                    fill_val = mode_val.iloc[0]
                elif s.dropna().size > 0:
                    fill_val = s.dropna().min()
                else:
                    fill_val = pd.Timestamp("1970-01-01")
                df_work[col] = s.fillna(fill_val)
                imputations[col] = f"datetime_fill={fill_val}"
            else:
                mode_val = s.dropna().mode()
                fill_val = (mode_val.iloc[0] if not mode_val.empty else "__MISSING__")
                df_work[col] = s.fillna(fill_val)
                imputations[col] = f"mode='{fill_val}'"

    # Outliery + winsoryzacja
    for col in _numeric_measure_candidates(df_work, min_non_null=3):
        s = pd.to_numeric(df_work[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        if pd.isna(q1) or pd.isna(q3):
            continue
        iqr = q3 - q1
        if iqr == 0:
            lower, upper = q1, q3
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

        flag_col = f"is_outlier_{col}"
        df_work[flag_col] = ((s < lower) | (s > upper)).astype(int)
        outlier_flags.append(flag_col)

        if col in winsorize_cols:
            df_work[col] = s.clip(lower, upper)
            winsorized[col] = {"lower": float(lower if pd.notna(lower) else 0.0),
                               "upper": float(upper if pd.notna(upper) else 0.0)}

    # Duplikaty
    duplicates_removed = 0
    if remove_duplicates:
        before = len(df_work)
        df_work = df_work.drop_duplicates(keep="first")
        duplicates_removed = before - len(df_work)

    prep_report = {
        "dropped_columns": drop_cols,
        "type_conversions": conversions,
        "new_time_cols_added": new_time_cols_added,
        "imputations": imputations,
        "outlier_flags": outlier_flags,
        "winsorized": winsorized,
        "duplicates_removed": int(duplicates_removed),
        "n_rows_final": int(df_work.shape[0]),
        "n_cols_final": int(df_work.shape[1]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return df_work, prep_report

def _save_summary_to_artifacts(text_markdown: str, latest_info: Dict[str, Any]) -> str:
    """Zapisuje summary_ai.md obok artefaktów bieżącego runu i zwraca pełną ścieżkę."""
    run_dir = latest_info.get("run_dir")
    if not run_dir:
        parquet_path = latest_info.get("parquet_path") or latest_info.get("csv_path") or "."
        run_dir = os.path.dirname(parquet_path)
    os.makedirs(run_dir, exist_ok=True)

    path = os.path.join(run_dir, "summary_ai.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write((text_markdown or "").strip() + "\n")

    st.session_state["summary_ai_path"] = path
    return path

def _persist_artifacts(
    df_ready: pd.DataFrame,
    prep_report: Dict[str, Any],
    latest_info: Dict[str, Any],
) -> Tuple[str, str]:
    """Zapisuje gotowy zbiór + raport i aktualizuje session_state."""
    run_dir = latest_info.get("run_dir")
    if not run_dir:
        parquet_path = latest_info.get("parquet_path") or latest_info.get("csv_path") or "."
        run_dir = os.path.dirname(parquet_path)
    os.makedirs(run_dir, exist_ok=True)

    ready_path = Path(run_dir) / "ready_for_training.parquet"
    report_path = os.path.join(run_dir, "prep_report.json")

    _df_to_parquet(df_ready, ready_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(prep_report, f, ensure_ascii=False, indent=2)

    base_artifacts = _remember_latest_artifacts(latest_info) or dict(latest_info or {})
    base_artifacts.update(
        {
            # IMPORTANT: store as strings (safe for cache + JSON)
            "ready_parquet_path": str(ready_path),
            "stage2_ready_parquet_path": str(ready_path),
            "prep_report_path": str(report_path),
            "n_rows_ready": prep_report["n_rows_final"],
            "n_cols_ready": prep_report["n_cols_final"],
            "status_ready": "ok",
            "stage2_export_ts": datetime.utcnow().isoformat() + "Z",
            "run_dir": str(Path(run_dir)),
            "ingest_root": str(Path(run_dir).parent),
        }
    )
    st.session_state["latest_artifacts"] = base_artifacts
    st.session_state["_stage2_last_valid_artifacts"] = base_artifacts

    # Disk pointer for Stage 3 (works even after navigation / rerun)
    try:
        st.session_state["latest_handoff_path"] = _write_latest_handoff_pointer(
            run_dir=Path(run_dir),
            ready_parquet_path=ready_path,
            prep_report_path=str(report_path),
            datachat_handoff_path=st.session_state.get("datachat_handoff_path"),
        )
    except Exception:
        pass
    st.session_state["prep_report"] = prep_report
    return ready_path, report_path


def _compute_readiness_score(
    high_null_cols: List[str],
    duplicates_pct: float,
    auto_drop_candidates: List[str],
    pairs_sorted: List[Dict[str, Any]],
) -> int:
    """Indeks gotowości 0–100."""
    score = 100
    if len(high_null_cols) > 0:
        score -= 10
    if duplicates_pct > 1.0:
        score -= 10
    if len(auto_drop_candidates) > 0:
        score -= 10
    if len(pairs_sorted) > 0:
        score -= 10
    return max(0, min(score, 100))


def _estimate_hours_saved(
    n_rows_raw: int,
    n_cols_raw: int,
    high_null_cols: list[str],
    duplicates_count: int,
    auto_drop_candidates: list[str],
    pairs_sorted: list[dict[str, Any]],
) -> float:
    base_hours = 4.0
    size_hours = (math.log10(max(n_rows_raw, 10)) * 2.0) + (n_cols_raw * 0.3)
    issues = 0
    if len(high_null_cols) > 0:
        issues += 1
    if duplicates_count > 0:
        issues += 1
    if len(auto_drop_candidates) > 0:
        issues += 1
    if len(pairs_sorted) > 0:
        issues += 1
    quality_hours = issues * 2.0
    est = base_hours + size_hours + quality_hours
    est = max(est, 4.0)
    est = min(est, 40.0)
    return round(est, 1)

def _estimate_cost_saved_pln(hours_saved: float, hourly_rate_pln: float = 450.0) -> float:
    return round(hours_saved * hourly_rate_pln, 2)

# ---------- Sekcja 7: kontrola widoczności (reset po zmianie danych) ----------
def _sec7_signature(csv_path: str, df: pd.DataFrame) -> str:
    """Lekka sygnatura danych do resetu widoczności sekcji 7 po rerunie/zmianie pliku."""
    try:
        mtime = os.path.getmtime(csv_path) if csv_path and os.path.exists(csv_path) else 0
    except Exception:
        mtime = 0
    cols = ",".join(list(df.columns)[:50])
    return f"{csv_path}|{mtime}|{df.shape[0]}x{df.shape[1]}|{hash(cols)}"

def _reset_sec7_state():
    st.session_state["sec7_revealed"] = False
    st.session_state["latest_summary_text"] = ""
    st.session_state["prep_done_and_summarized"] = False
    st.session_state["play_tts_now"] = False
    _eda_clear_debug_state()

    # 🔄 czyścimy cache wyników winsoryzacji (dla bezpieczeństwa przy zmianie zbioru)
    st.session_state.pop("winsor_cache", None)

# alias kompatybilności wstecznej (stare wywołania bez "_")
def reset_sec7_state():
    """Alias do _reset_sec7_state() – nie usuwać, bo używane w main()."""
    _reset_sec7_state()

# ---------- UI helpers ----------

def _metric_card_html(title: str, value: str, subtitle: str, bg: str = "#f8f9fa", fg: str = "#111", border: str = "#ddd") -> str:
    return f"""
    <div style="
        border:1px solid {border};
        border-radius:0.5rem;
        padding:0.75rem 1rem;
        background:{bg};
        min-height:6.9rem;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        box-sizing:border-box;
    ">
      <div style="font-size:0.75rem; color:#666; line-height:1.2; font-weight:500;">{title}</div>
      <div style="font-size:1.5rem; font-weight:600; color:{fg}; line-height:1.4; margin-top:0.25rem;">{value}</div>
      <div style="font-size:0.8rem; color:#666; line-height:1.2; margin-top:0.25rem;">{subtitle}</div>
    </div>
    """

def _status_banner_html(level: str, text_md: str) -> str:
    """
    level: 'ok' | 'warn' | 'err'
    text_md: prosty markdown w jednej linii (obsługujemy **bold** i `code`)
    """
    import re
    # mini-renderer: **...** -> <strong>...</strong>, `...` -> <code>...</code>
    def _md_to_inline_html(s: str) -> str:
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    styles = {
        "ok":   {"bg":"#f0fff4","bd":"#c6f6d5","fg":"#22543d","icon":"✅"},
        "warn": {"bg":"#fffaf0","bd":"#feebc8","fg":"#7b341e","icon":"⚠️"},
        "err":  {"bg":"#fff5f5","bd":"#fed7d7","fg":"#742a2a","icon":"⛔"},
    }[level]

    text_html = _md_to_inline_html(text_md)

    return f"""
    <div style="border:1px solid {styles['bd']};background:{styles['bg']};
                border-radius:0.6rem;padding:0.9rem 1rem;margin-top:0.6rem;
                color:{styles['fg']}; line-height:1.55;">
      <span style="font-weight:700;margin-right:.35rem;">{styles['icon']}</span>{text_html}
    </div>
    """


def _tech_details_html(csv_path: str, run_dir: str, mask_pii: bool, global_missing_pct: float,
                       duplicates_pct: float, auto_drop_candidates: List[str],
                       predicted_cols_after_drop: int) -> str:
    return f"""
    <ul style="list-style-type:none; padding-left:0; font-size:0.9rem; line-height:1.5;">
      <li>• <b>Ścieżka pliku źródłowego:</b> <code>{csv_path}</code></li>
      <li>• <b>Folder roboczy runu:</b> <code>{run_dir}</code></li>
      <li>• <b>Maskowanie wrażliwych danych (PII):</b> {"tak" if mask_pii else "nie"}</li>
      <li>• <b>Globalne braki:</b> ~{global_missing_pct:.1f}% wszystkich komórek</li>
      <li>• <b>Potencjalne duplikaty:</b> {duplicates_pct:.1f}% rekordów</li>
      <li>• <b>Kolumny do potencjalnego usunięcia / scalania:</b> {", ".join(auto_drop_candidates) if auto_drop_candidates else "(brak)"}</li>
      <li>• <b>Szacowana liczba kolumn po czyszczeniu:</b> {predicted_cols_after_drop} (+ cechy z dat / outlierów)</li>
    </ul>
    """

def _success_hero_box(hours_saved: float, cost_saved: float) -> str:
    return f"""
    <div style="border:1px solid #d4edda;background:#f6fff8;border-radius:0.5rem;padding:1rem 1.25rem;font-size:1rem;line-height:1.5;color:#155724;">
      <div style="font-size:1.25rem; font-weight:600; margin-bottom:0.5rem;">
        ✅ Dane gotowe do trenowania modelu
      </div>
      <div>
        🎉 Szacunkowo zaoszczędzono <b>{hours_saved} godzin</b> pracy analityka (~<b>{cost_saved} PLN</b>),
        bez ręcznego czyszczenia Excela i bez ryzyka pomyłki.
      </div>
      <div style="margin-top:0.75rem;">
        Gotowy zbiór: <code>ready_for_training.parquet</code> · Raport: <code>prep_report.json</code>.
      </div>
    </div>
    """

def _before_after_cards_html(n_cols_raw: int, n_cols_final: int, missing_before_pct: float,
                             duplicates_before: int, duplicates_removed: int, outlier_flags_count: int) -> str:
    return f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:0.75rem; margin-top:1rem;">
      <div style="border:1px solid #ddd;border-radius:0.5rem;padding:0.75rem;">
        <div style="font-size:0.8rem;color:#666;">Kolumny (przed ➜ po)</div>
        <div style="font-size:1.1rem;font-weight:600;">{n_cols_raw} ➜ {n_cols_final}</div>
        <div style="font-size:0.8rem;color:#666;margin-top:0.25rem;">Usunęliśmy śmieciowe/zdublowane kolumny i dodaliśmy cechy z dat oraz flagi outlierów.</div>
      </div>
      <div style="border:1px solid #ddd;border-radius:0.5rem;padding:0.75rem;">
        <div style="font-size:0.8rem;color:#666;">Braki danych</div>
        <div style="font-size:1.1rem;font-weight:600;">~{missing_before_pct:.1f}% ➜ 0.0%</div>
        <div style="font-size:0.8rem;color:#666;margin-top:0.25rem;">Uzupełniliśmy luki medianą/trybem/sensowną datą.</div>
      </div>
      <div style="border:1px solid #ddd;border-radius:0.5rem;padding:0.75rem;">
        <div style="font-size:0.8rem;color:#666;">Duplikaty</div>
        <div style="font-size:1.1rem;font-weight:600;">{duplicates_before} ➜ {duplicates_before - duplicates_removed}</div>
        <div style="font-size:0.8rem;color:#666;margin-top:0.25rem;">Usunęliśmy zdublowane wiersze.</div>
      </div>
      <div style="border:1px solid #ddd;border-radius:0.5rem;padding:0.75rem;">
        <div style="font-size:0.8rem;color:#666;">Wartości odstające</div>
        <div style="font-size:1.1rem;font-weight:600;">Zflagowano {outlier_flags_count} kolumn</div>
        <div style="font-size:0.8rem;color:#666;margin-top:0.25rem;">Nie kasujemy ekstremów — dajemy flagę, model widzi nietypowe przypadki.</div>
      </div>
    </div>
    """

# ---------- Scatter z trendem ----------

def _scatter_with_trend(df: pd.DataFrame, x: str, y: str, height: int = 200, title: str | None = None) -> alt.Chart:
    use_df = df[[x, y]].dropna()
    base = (
        alt.Chart(use_df)
        .mark_point(filled=True, size=40, opacity=0.6, color="#007bff")
        .encode(x=alt.X(f"{x}:Q", title=x), y=alt.Y(f"{y}:Q", title=y), tooltip=[x, y])
        .properties(height=height)
    )
    trend = (
        alt.Chart(use_df)
        .transform_regression(x, y)
        .mark_line(color="#dc3545")
        .encode(x=alt.X(f"{x}:Q", title=x), y=alt.Y(f"{y}:Q", title=y))
    )
    return (base + trend).properties(title=title) if title else (base + trend)

def _pairs_dataframe(pairs_sorted: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{"col1": p["col1"], "col2": p["col2"], "corr": float(p["corr"]), "suggest_drop": p["suggest_drop"]} for p in pairs_sorted])

# ---------- TTS + AI TL;DR ----------

# ───────────────────── Helpery / cache TTS ─────────────────────


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
    return h.hexdigest()

def _run_tts_for_summary(
    text_for_tts: str,
    provider: str,
    openai_tts_model_selected: str,
    openai_voice_selected: str,
    cur_tts_hash: str | None = None,
):
    """
    Generuje audio do podsumowania (AI) i zapisuje je w sesji.
    Odtwarzacz renderujemy osobno, żeby uniknąć równoległych ścieżek audio.
    """
    text_for_tts = _plain_text_for_tts(text_for_tts)
    if not text_for_tts:
        return

    # jeśli nie podano hash-a, policz tutaj (musi być spójny z blokiem niżej)
    if cur_tts_hash is None:
        cur_tts_hash = _hash_key(
            text_for_tts,
            provider or "",
            openai_tts_model_selected or "",
            openai_voice_selected or "",
        )

    # Bezpieczniki na klucze/API
    openai_key = _get_env_or_secret("OPENAI_API_KEY")

    # Trace wspólny dla obu dostawców (opcjonalny – tylko jeśli Langfuse jest dostępny)
    lf_client = None
    if "get_langfuse" in globals():
        try:
            lf_client = get_langfuse()
        except Exception:
            lf_client = None

    tts_trace = (
        lf_client.trace(
            name="eda_tts",
            user_id=st.session_state.get("wf_session_id", "anon"),
            input=text_for_tts[:2000],
            metadata={
                "provider": provider,
                "openai_model": openai_tts_model_selected if provider == "OpenAI" else None,
            },
        )
        if lf_client
        else None
    )

    audio_bytes, err = b"", ""

    try:
        if provider == "OpenAI":
            if not openai_key:
                err = "Brak OPENAI_API_KEY w .env"
            else:
                voice = openai_voice_selected or DEFAULT_FEMALE_VOICE
                audio_bytes, err = _tts_openai_cached(
                    text_for_tts,
                    voice=voice,
                    model=openai_tts_model_selected,
                    api_key=openai_key,
                )
        else:
            err = "Etap 2 obsługuje TTS przez OpenAI."
        if audio_bytes:
            st.session_state["latest_tts_audio_bytes"] = audio_bytes
            st.session_state["latest_tts_audio_mime"] = "audio/mpeg"
            st.session_state["latest_tts_audio_hash"] = cur_tts_hash
            st.session_state["tts_last_hash"] = cur_tts_hash
            if tts_trace:
                tts_trace.update(status="success", metadata={"bytes": len(audio_bytes)})
        else:
            st.warning(f"Nie udało się wygenerować TTS ({provider}): {err}")
            if tts_trace:
                tts_trace.update(status="error", metadata={"error": err})
    finally:
        # resetujemy jednorazowy trigger
        st.session_state["play_tts_now"] = False

@st.cache_data(show_spinner=False, ttl=1800, max_entries=64)
def _tts_openai_cached(text: str, voice: str, model: str, api_key: str):
    # zwraca (audio_bytes, err)
    errors: list[str] = []
    try:
        if _openai is not None:
            client = _openai.OpenAI(api_key=api_key)
            # Speech API (nowy SDK)
            res = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="mp3",
                speed=OPENAI_TTS_SPEED,
            )
            return res.read(), ""
        errors.append("pakiet openai nie jest dostepny")
    except Exception as e:
        errors.append(f"OpenAI SDK TTS: {e}")

    try:
        payload = {
            "model": model,
            "voice": voice,
            "input": (text or "")[:STAGE2_TTS_MAX_CHARS],
            "response_format": "mp3",
            "speed": OPENAI_TTS_SPEED,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=payload,
            timeout=OPENAI_TTS_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400 and "speed" in (response.text or "").lower():
            payload.pop("speed", None)
            response = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload,
                timeout=OPENAI_TTS_TIMEOUT_SECONDS,
            )
        if response.status_code >= 400:
            errors.append(_openai_error_message(response))
            return b"", "OpenAI TTS error: " + "; ".join(errors)
        return response.content, ""
    except Exception as e:
        errors.append(f"OpenAI REST TTS: {e}")
        return b"", "OpenAI TTS error: " + "; ".join(errors)

# ====================   MAIN VIEW   ======================
def _guess_cols_from_info_df(info_df: pd.DataFrame) -> Tuple[str | None, str | None, str | None]:
    """Fast heuristics using precomputed per-column stats (no full-data scans)."""
    try:
        cols = info_df.copy()
        cols["name_l"] = cols["kolumna"].astype(str).str.lower()
        # Outcome (target) — prefer obvious labels
        outcome = None
        for pat in ["target", "y", "label", "churn", "outcome", "default", "fraud"]:
            hit = cols.loc[cols["name_l"].str.contains(pat, na=False), "kolumna"]
            if len(hit):
                outcome = str(hit.iloc[0])
                break

        # Value — numeric with sales/value/amount/qty semantics
        num = cols.loc[cols["logical_type"].isin(["num", "int", "float", "numeric"]) | cols["dtype_raw"].astype(str).str.contains("int|float", case=False, na=False)]
        value = None
        for pat in ["value", "wart", "sales", "sprzed", "amount", "revenue", "price", "qty", "quantity", "wolumen"]:
            hit = num.loc[num["name_l"].str.contains(pat, na=False), "kolumna"]
            if len(hit):
                value = str(hit.iloc[0]); break
        if value is None and len(num):
            value = str(num.iloc[0]["kolumna"])

        # Group — categorical-like (low-ish cardinality)
        cat = cols.loc[cols["logical_type"].isin(["cat", "date"]) == False]  # just to keep structure
        # We treat object/category as group candidates
        grp_cand = cols.loc[
            cols["dtype_raw"].astype(str).str.contains("object|category", case=False, na=False)
            & (cols["n_unique"].fillna(0).astype(float) >= 2)
            & (cols["n_unique"].fillna(0).astype(float) <= 200)
            & (cols["null_pct"].fillna(0).astype(float) <= 90)
        ]
        group = None
        for pat in ["category", "kategoria", "country", "kraj", "segment", "brand", "produkt", "product", "customer", "klient"]:
            hit = grp_cand.loc[grp_cand["name_l"].str.contains(pat, na=False), "kolumna"]
            if len(hit):
                group = str(hit.iloc[0]); break
        if group is None and len(grp_cand):
            group = str(grp_cand.iloc[0]["kolumna"])

        return outcome, group, value
    except Exception:
        return None, None, None


# ─────────────────────────────────────────────────────────────
# Stage2 -> Stage3 handoff pointer (ingest/_latest_handoff.json)
# ─────────────────────────────────────────────────────────────
def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic JSON write (avoid half-written files on rerun/crash)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(path))

def _write_latest_handoff_pointer(
    run_dir: Path,
    ready_parquet_path: Path,
    prep_report_path: str | None,
    datachat_handoff_path: str | None,
) -> str:
    """Write/overwrite ingest/_latest_handoff.json so Stage3 can always pick the newest Stage2 output."""
    ingest_root = run_dir.parent
    pointer_path = ingest_root / "_latest_handoff.json"
    payload: Dict[str, Any] = {
        "version": 1,
        "run_dir": str(run_dir),
        "ingest_root": str(ingest_root),
        "ready_parquet_path": str(ready_parquet_path),
        "prep_report_path": str(prep_report_path) if prep_report_path else None,
        "datachat_handoff_path": str(datachat_handoff_path) if datachat_handoff_path else None,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        payload["ready_parquet_mtime"] = os.path.getmtime(str(ready_parquet_path))
    except Exception:
        payload["ready_parquet_mtime"] = None
    _atomic_write_json(pointer_path, payload)
    return str(pointer_path)


def _save_datachat_handoff(
    latest_info: dict,
    summary_text: str,
    pairs_sorted: List[Dict[str, Any]] | None,
    prep_report_path: str | None,
    info_df: pd.DataFrame | None = None,
) -> str:
    """Zapisuje minimalny handoff do Etapu 3 bez skanowania całego df_ready."""
    parquet_src = latest_info.get("parquet_path") or latest_info.get("csv_path") or "."
    run_dir = Path(os.path.dirname(parquet_src))
    run_dir.mkdir(parents=True, exist_ok=True)

    ready_parquet_path = run_dir / "ready_for_training.parquet"
    handoff_path = run_dir / "datachat_handoff.json"

    outcome_col = group_col = value_col = None
    if info_df is not None and isinstance(info_df, pd.DataFrame) and not info_df.empty:
        outcome_col, group_col, value_col = _guess_cols_from_info_df(info_df)

    handoff = {
        "version": 2,
        "parquet_path_full": str(Path(parquet_src)),
        "ready_parquet_path": str(ready_parquet_path),
        "summary_text": summary_text,
        "pairs_sorted": pairs_sorted or [],
        "prep_report_path": prep_report_path,
        "suggested_columns": {
            "outcome_col": outcome_col,
            "group_col": group_col,
            "value_col": value_col,
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(handoff_path, "w", encoding="utf-8") as f:
        json.dump(handoff, f, ensure_ascii=False, indent=2)

    st.session_state["datachat_handoff_path"] = str(handoff_path)

    # Update ingest/_latest_handoff.json as well (freshness-based handoff for Stage 3)
    try:
        st.session_state["latest_handoff_path"] = _write_latest_handoff_pointer(
            run_dir=run_dir,
            ready_parquet_path=ready_parquet_path,
            prep_report_path=prep_report_path,
            datachat_handoff_path=str(handoff_path),
        )
    except Exception:
        pass

    return str(handoff_path)

def _hero_time_series_overview(
    df: pd.DataFrame,
    time_col: str | None,
    roles: dict,
) -> None:
    st.subheader("Podgląd szeregu czasowego", divider="gray")
    st.markdown("<div style='margin-top:-0.40rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([4, 1.5])

    num_cols = _numeric_measure_candidates(df, min_non_null=2)
    if not num_cols:
        st.info("Brak kolumn liczbowych – nie można narysować szeregu.")
        return

    # zmienne na globalny trend szeregu (wykorzystamy po prawej stronie)
    trend_text: str | None = None

    # -------- lewa kolumna – wybory + wykres linii -----------
    with col_left:
        # oś czasu – kandydaci
        x_options = ["Numer wiersza (pseudo-czas)"]
        if time_col and time_col in df.columns:
            x_options.insert(0, time_col)

        # trzy selectboxy w jednej linii
        c1, c2, c3 = st.columns(3)

        with c1:
            x_choice = st.selectbox(
                "Oś czasu (oś X)",
                x_options,
                key="ts_hero_x",
            )

        with c2:
            y_col = st.selectbox(
                "Kolumna wartości (oś Y)",
                num_cols,
                key="ts_hero_y",
            )

        with c3:
            agg_choice = st.selectbox(
                "Agregacja czasu",
                ["Dzień", "Tydzień", "Miesiąc", "Rok"],
                index=0,
                key="ts_hero_agg",
            )

# --- przygotowanie danych do wykresu ---
        cols_used = [y_col]
        plot_df = df[cols_used].copy()

        # mapowanie agregacji
        rule_map = {"Dzień": "D", "Tydzień": "W", "Miesiąc": "M", "Rok": "Y"}
        x_title_map = {
            "Dzień": "Data (dzień)",
            "Tydzień": "Tydzień",
            "Miesiąc": "Miesiąc",
            "Rok": "Rok",
        }
        rule = rule_map.get(agg_choice)
        x_axis_title = x_title_map.get(agg_choice, x_choice)
        x_axis_format_map = {
            "Dzień": "%Y-%m-%d",
            "Tydzień": "%Y-%m-%d",
            "Miesiąc": "%Y-%m",
            "Rok": "%Y",
        }
        x_axis_format = x_axis_format_map.get(agg_choice, "%Y-%m-%d")

        if x_choice == "Numer wiersza (pseudo-czas)":
            # pseudo-czas → oś numeryczna; agregujemy przez grupowanie co N wierszy
            plot_df["__x__"] = np.arange(len(df))
            plot_df = plot_df.dropna(subset=[y_col])

            if rule is not None and not plot_df.empty:
                # przybliżone okna dla pseudo-czasu (w wierszach)
                pseudo_window_map = {"D": 1, "W": 7, "M": 30, "Y": 365}
                win = int(pseudo_window_map.get(rule, 1))

                plot_df["_grp"] = (plot_df["__x__"] // win).astype(int)
                plot_df = (
                    plot_df.groupby("_grp", as_index=False)[y_col]
                    .mean()
                )
                plot_df["__x__"] = plot_df["_grp"] * win
                plot_df = plot_df.drop(columns=["_grp"])

            x_enc = alt.X("__x__:Q", title="Numer wiersza (pseudo-czas)")

        else:
            # prawdziwy czas
            t = pd.to_datetime(df[x_choice], errors="coerce")
            plot_df["__x__"] = t
            plot_df = plot_df.dropna(subset=["__x__", y_col]).sort_values("__x__")

            # resample tylko dla datetime
            if (
                rule is not None
                and not plot_df.empty
                and pd.api.types.is_datetime64_any_dtype(plot_df["__x__"])
            ):
                plot_df = (
                    plot_df.set_index("__x__")[y_col]
                    .resample(rule)
                    .mean()
                    .dropna()
                    .reset_index()
                    .rename(columns={"__x__": "__x__", y_col: y_col})
                )

            x_enc = alt.X(
                "__x__:T",
                axis=alt.Axis(
                    title=x_axis_title,
                    format=x_axis_format,
                    labelAngle=-35,
                    labelOverlap="greedy",
                    labelLimit=140,
                ),
            )

        # finalne czyszczenie po wcześniejszej agregacji
        plot_df = plot_df.dropna(subset=["__x__", y_col])

        if plot_df.empty:
            st.info("Brak danych po agregacji do wyświetlenia szeregu.")
            return

        # --- globalny trend szeregu (skala -3..0..+3) ---
        plot_df_sorted = plot_df.sort_values("__x__")
        y_vals = plot_df_sorted[y_col].to_numpy(dtype=float)

        if len(y_vals) >= 2 and np.isfinite(y_vals).sum() >= 2:
            x_idx = np.arange(len(y_vals), dtype=float)

            # --- stabilizacja polyfit: tylko wartości skończone ---
            finite_mask = np.isfinite(y_vals)
            x_fit = x_idx[finite_mask]
            y_fit = y_vals[finite_mask]

            if len(y_fit) < 2:
                slope = 0.0
            else:
                # polyfit na czystych danych + bezpieczne obejście błędów LAPACK/SVD
                try:
                    slope = float(np.polyfit(x_fit, y_fit, 1)[0])
                except np.linalg.LinAlgError:
                    slope = 0.0

            y_range = float(np.nanmax(y_vals) - np.nanmin(y_vals)) if np.isfinite(y_vals).any() else 0.0
            if y_range == 0.0:
                norm = 0.0
            else:
                norm = slope / (y_range / max(len(y_vals) - 1, 1))

            abs_norm = abs(norm)

            if abs_norm < 0.02:
                strength = "brak wyraźnego trendu"
                level = 0
            elif abs_norm < 0.07:
                strength = "lekki trend"
                level = 1
            elif abs_norm < 0.15:
                strength = "średni trend"
                level = 2
            else:
                strength = "silny trend"
                level = 3

            if norm > 0:
                direction = "wzrostowy"
                arrow = "↗"
                signed_level = level
            elif norm < 0:
                direction = "spadkowy"
                arrow = "↘"
                signed_level = -level
            else:
                direction = "płaski"
                arrow = "→"
                signed_level = 0

            trend_text = (
                f"{arrow} {strength} {direction} "
                f"(poziom {signed_level:+d}/3, znormalizowane nachylenie ≈ {norm:+.3f})."
            )

        # --- Altair ma limit ~5000 wierszy → rysujemy próbkę równomierną ---
        plot_df_sorted = plot_df.sort_values("__x__")
        plot_df_plot = plot_df_sorted
        MAX_PLOT_POINTS = 5000

        if len(plot_df_sorted) > MAX_PLOT_POINTS:
            idx = np.linspace(0, len(plot_df_sorted) - 1, MAX_PLOT_POINTS).astype(int)
            plot_df_plot = plot_df_sorted.iloc[idx].copy()

        # --- wykres linii (z paddingiem, żeby nie ucinało osi) ---
        chart = (
            alt.Chart(plot_df_plot)
            .mark_line()
            .encode(
                x=x_enc,
                y=alt.Y(f"{y_col}:Q", title=y_col),
                tooltip=[
                    alt.Tooltip(
                        "__x__:T" if x_choice != "Numer wiersza (pseudo-czas)" else "__x__:Q",
                        title=x_axis_title if x_choice != "Numer wiersza (pseudo-czas)" else x_choice,
                    ),
                    alt.Tooltip(f"{y_col}:Q", title=y_col, format=".3g"),
                ],
            )
            .properties(
                height=380,
                padding={"left": 10, "right": 20, "top": 10, "bottom": 45},
            )
            .configure_axis(titlePadding=10, labelPadding=6)
            .configure_view(stroke=None)
        )
        altair_chart_stretch(st, chart, width='stretch')


    # -------- prawa kolumna – karta z oceną + badge trendu -----------
    with col_right:
        st.markdown("**Szybka ocena szeregu**")

        n_raw = len(df)
        st.markdown(f"**Liczba obserwacji:** {n_raw:,}")

        if time_col and time_col in df.columns:
            t = pd.to_datetime(df[time_col], errors="coerce").dropna()
            if not t.empty:
                st.markdown(
                    f"**Okres:** {t.min().date()} – {t.max().date()}"
                )

        st.markdown(f"**Agregacja:** {agg_choice}")
        st.markdown(f"**Punkty po agregacji:** {len(plot_df_sorted):,}")

        if n_raw >= 100:
            st.success("✅ Seria ma wystarczająco dużo punktów do podstawowej analizy.")
        else:
            st.warning("⚠️ Seria ma mało punktów – wnioski mogą być niestabilne.")

        if trend_text:
            st.markdown(
                f"""
                <div style="
                    margin-top:0.75rem;
                    padding:0.6rem 0.75rem;
                    border-radius:0.6rem;
                    background:#eff6ff;
                    border:1px solid #bfdbfe;
                    font-size:0.85rem;
                    line-height:1.4;
                ">
                  <div style="font-weight:600;margin-bottom:0.20rem;">
                    Trend globalny szeregu
                  </div>
                  <div>{trend_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="margin-top:0.75rem;font-size:0.80rem;color:#6b7280;">'
                "Brak wystarczających danych, aby wiarygodnie ocenić trend."
                "</div>",
                unsafe_allow_html=True,
            )

# ============================================================
#  HERO: KLASTERYZACJA – przestrzeń cech
# ============================================================


def _hero_classification_overview(df: pd.DataFrame, y_col: str) -> None:
    """Lightweight classification target sanity-check (fast, safe)."""
    st.markdown("### Cel (klasyfikacja) — szybki sanity check")
    if not y_col or y_col not in df.columns:
        st.info("Wybierz kolumnę celu, aby zobaczyć rozkład klas.")
        return

    s = df[y_col]
    n = int(len(s))
    vc = s.value_counts(dropna=False)
    nunique = int(vc.shape[0])
    top_share = float(vc.iloc[0] / max(n, 1)) if n else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Liczba obserwacji", f"{n:,}".replace(",", " "))
    c2.metric("Liczba klas", f"{nunique:,}".replace(",", " "))
    c3.metric("Największa klasa", f"{top_share*100:.1f}%")

    if nunique <= 1:
        st.error("Kolumna celu ma tylko jedną klasę — nie da się trenować klasyfikacji.")
        return

    if top_share >= 0.9:
        st.warning("Silna nierównowaga klas (≥90% w jednej klasie). Rozważ ważenie klas / resampling / inną metrykę.")
    elif top_share >= 0.7:
        st.info("Umiarkowana nierównowaga klas (≥70% w jednej klasie) — zweryfikuj metryki i strategię walidacji.")

    # Show top classes (keep it light)
    top_k = 20
    vc_top = vc.head(top_k).reset_index()
    vc_top.columns = [y_col, "count"]
    vc_top[y_col] = vc_top[y_col].astype(str).replace({"nan": "(brak)"})

    try:
        import altair as alt
        chart = (
            alt.Chart(vc_top)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Liczba"),
                y=alt.Y(f"{y_col}:N", sort="-x", title="Klasa"),
                tooltip=[alt.Tooltip(f"{y_col}:N", title="Klasa"), alt.Tooltip("count:Q", title="Liczba")],
            )
            .properties(height=420)
        )
        # Streamlit version compatibility: width='stretch' (new) vs use_container_width (old)
        # Streamlit multipage: avoid importing via "app.*" ("app" isn't a package at runtime)
        from core.ui_safe import altair_chart_stretch, dataframe_stretch
        altair_chart_stretch(st, chart)
    except Exception:
        # Streamlit multipage: avoid importing via "app.*" ("app" isn't a package at runtime)
        from core.ui_safe import dataframe_stretch
        dataframe_stretch(st, vc_top)

    if nunique > top_k:
        st.caption(f"Pokazano Top {top_k} klas z {nunique}.")
def _hero_clustering_overview(df: pd.DataFrame, cluster_col: str | None) -> None:
    st.subheader("Przestrzeń cech do klasteryzacji", divider="gray")
    st.markdown("<div style='margin-top:-0.40rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([4, 1.5])

    # -------- LEWA: wybór osi + wykres -----------------
    with col_left:
        # kolumny numeryczne do osi X/Y
        num_cols = _numeric_measure_candidates(df, min_non_null=2)

        if len(num_cols) < 2:
            st.info(
                "Do wizualizacji przestrzeni cech potrzebne są przynajmniej dwie kolumny liczbowe."
            )
            return

        # kolumny kategoryczne do kolorowania
        cat_cols = _categorical_feature_candidates(df)

        # --- trzy selectboxy w jednej linii ---
        c1, c2, c3 = st.columns(3)

        with c1:
            x_col = st.selectbox(
                "Cecha na osi X",
                num_cols,
                index=0,
                key="cluster_x",
            )

        with c2:
            # domyślnie pierwsza inna kolumna niż X
            y_candidates = [c for c in num_cols if c != x_col] or num_cols
            y_col = st.selectbox(
                "Cecha na osi Y",
                y_candidates,
                index=0,
                key="cluster_y",
            )

        with c3:
            color_options = ["(Brak)"]
            # najpierw sugerujemy kolumnę klastrów, jeśli istnieje
            if cluster_col and cluster_col in df.columns:
                color_options.append(cluster_col)
            # potem pozostałe kategoryczne (bez duplikatów)
            for c in cat_cols:
                if c not in color_options:
                    color_options.append(c)

            color_choice = st.selectbox(
                "Kolor punktów (segment / kategoria)",
                color_options,
                index=0,
                key="cluster_color",
            )

        # --- przygotowanie danych do wykresu ---
        cols_for_plot = [x_col, y_col]
        if color_choice != "(Brak)":
            cols_for_plot.append(color_choice)

        df_plot = df[cols_for_plot].copy()
        df_plot[x_col] = _numeric_series_for_candidate(df_plot[x_col])
        df_plot[y_col] = _numeric_series_for_candidate(df_plot[y_col])
        df_plot = df_plot.dropna(subset=[x_col, y_col])
        if color_choice != "(Brak)":
            df_plot = df_plot.dropna(subset=[color_choice])

        if df_plot.empty:
            st.info("Brak danych do wyświetlenia przestrzeni cech dla wybranych kolumn.")
            return
        if len(df_plot) > 5_000:
            df_plot = df_plot.sample(5_000, random_state=42)
            st.caption("Pokazuję próbkę 5,000 punktów, żeby wykres pozostał responsywny.")

        tooltip_fields = [x_col, y_col]
        encodings = {
            "x": alt.X(x_col, title=x_col),
            "y": alt.Y(y_col, title=y_col),
        }

        if color_choice != "(Brak)":
            encodings["color"] = alt.Color(
                color_choice,
                title=color_choice,
                legend=alt.Legend(title=color_choice),
            )
            tooltip_fields.append(color_choice)

        chart = (
            alt.Chart(df_plot)
            .mark_circle(size=50, opacity=0.7)
            .encode(
                tooltip=tooltip_fields,
                **encodings,
            )
            .properties(height=320)
        )

        altair_chart_stretch(st, chart, width='stretch')

    # -------- PRAWA: szybka ocena -----------------------
    with col_right:
        st.markdown("**Szybka ocena przestrzeni cech**")
        st.write(f"Liczba rekordów: **{len(df)}**")
        st.write(f"Liczba cech numerycznych: **{len(num_cols)}**")

        st.info(
            "Użyj kolorowania, żeby zobaczyć wyraźniejsze segmenty. "
            "W kolejnych krokach możemy nazwać i opisać klastry automatycznie."
        )


# ============================================================
#  HERO: REGRESJA – rozkład celu + ocena
# ============================================================
def _hero_regression_overview(df: pd.DataFrame, target_col: str) -> None:
    st.subheader("Rozkład wartości celu (y)", divider="gray")
    st.markdown("<div style='margin-top:-0.40rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([4, 1.5])

    s = pd.to_numeric(df[target_col], errors="coerce").dropna()

    # -------- lewa kolumna – histogram -----------
    with col_left:
        if s.empty:
            st.info("Brak danych liczbowych w kolumnie celu – nie można narysować rozkładu.")
        else:
            hist_df = pd.DataFrame({target_col: s})
            chart = (
                alt.Chart(hist_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{target_col}:Q",
                        bin=alt.Bin(maxbins=40),
                        title=target_col,
                    ),
                    y=alt.Y("count():Q", title="Liczba rekordów"),
                )
                .properties(height=320)
            )
            altair_chart_stretch(st, chart, width='stretch')

    # -------- prawa kolumna – karta z oceną -----------
    with col_right:
        st.markdown("**Szybka ocena rozkładu celu**")

        if s.empty:
            st.info("Brak danych do oceny rozkładu.")
            return

        st.markdown(
            f"**Mediana:** {s.median():.3g}  \n"
            f"**Min / max:** {s.min():.3g} – {s.max():.3g}"
        )

        skew = s.skew()
        if abs(skew) < 0.5:
            st.success("✅ Rozkład jest w przybliżeniu symetryczny – to ułatwia modelowanie.")
        elif skew > 0:
            st.warning("⚠️ Rozkład jest mocno prawoskośny – rozważ transformację (np. log).")
        else:
            st.warning("⚠️ Rozkład jest lewoskośny – sprawdź, czy nie ma efektów obcięcia.")

def main():
    # --- NAWIGACJA ---
    hide_default_multipage_nav()
    render_flow_nav(current_id="02_Automat_EDA")  # aktywny kafel Etapu 2

    st.title("Automat EDA — szybka diagnostyka danych (Etap 2)")
    # Langfuse: identyfikator sesji użytkownika
    if "wf_session_id" not in st.session_state:
        st.session_state["wf_session_id"] = str(uuid4())

    lf = get_langfuse()
    _eda_reset_run_debug_state()

    # --- FLASH MESSAGE po rerunie ---
    _flash = st.session_state.pop("_flash_msg", None)
    if _flash:
        # Możesz wybrać formę: toast (zanika po kilku sekundach) albo klasyczny alert
        if _flash.get("use_toast", True):
            st.toast(_flash.get("text", ""), icon="✅" if _flash.get("type") == "success" else "⚠️")
        else:
            level = _flash.get("type", "info")  # "success" | "warning" | "error" | "info"
            getattr(st, level)(_flash.get("text", ""))

    st.markdown(
        """
        Etap **2 z 4** - automatycznie podsumowuje dane z kroku
        **„Przelicz na całości i zapisz artefakty”** (zakładka **Analiza Danych**).

        **Co dostajesz:**\n
        • diagnozę jakości danych zanim zaczniesz trenować model,  
        • automatyczne czyszczenie (usunięcie śmieci, uzupełnienie braków, rozbicie dat, flagi outlierów),  
        • gotowy zestaw ready_for_training.parquet,  
        • możliwość przejścia do **Trenowanie Modelu** praktycznie bez ręcznej roboty.
        """
    )
    st.subheader("", divider="gray")

    # ───────────────────────── SIDEBAR: Praca na próbce danych (szybsza EDA) ─────────────────────────
    with st.sidebar:
        use_sample = st.checkbox(
            "⚡ Pracuj na próbce danych (szybsza EDA)",
            value=True,
            help=(
                "Przy bardzo dużych zbiorach EDA będzie działać na równomiernej próbie z całego pliku, "
                "a pełny zbiór zostanie wykorzystany później do trenowania modelu."
            ),
            key="eda_use_sample",
        )
        sample_rows_eda = st.number_input(
            "Rozmiar próbki do EDA",
            min_value=5_000,    # minimalna sensowna próbka
            max_value=200_000,  # górny limit, żeby nie „zabić” przeglądarki
            step=5_000,
            value=20_000,       # domyślnie: 20k, dobry kompromis
            format="%d",
            key="eda_sample_rows",
        )

    # ───────────────────────── SIDEBAR: Lektor & TL;DR ─────────────────────────
    with st.sidebar.expander("🎙️ Lektor & TL;DR", expanded=False):
        fast_sidebar = st.checkbox(
            "🧠 Generuj TL;DR (OpenAI)",
            value=True,
            key="eda_generate_tldr",
            help="4–6 zwięzłych zdań na bazie metryk i wniosków",
        )
        openai_tldr_model = st.selectbox(
            "OpenAI TL;DR model",
            ["gpt-4o-mini", "gpt-4o"],
            index=0,
            disabled=not fast_sidebar,
        )
        
        # defensywne domyślne wartości; dzięki temu zmienne zawsze istnieją
        openai_voice_selected = None
        openai_tts_model_selected = OPENAI_TTS_MODELS[0]

        st.markdown("---")
        st.checkbox("✅ Włącz lektora (TTS)", value=True, key="tts_enabled")

        # Dostawca TTS: w Etapie 2 blokujemy wybór (zostawiamy OpenAI).
        provider = 'OpenAI'
        gender = st.radio("Głos", ["Kobieta", "Mężczyzna"], horizontal=True, index=0)

        if provider == "OpenAI":
            gender_key = "female" if gender == "Kobieta" else "male"
            voice_pool = OPENAI_VOICES[gender_key]
            default_voice = DEFAULT_FEMALE_VOICE if gender == "Kobieta" else DEFAULT_MALE_VOICE
            if default_voice not in voice_pool:
                default_voice = voice_pool[0]
            openai_voice_selected = st.selectbox("OpenAI voice", options=voice_pool,
                                                 index=voice_pool.index(default_voice))

        # Gdy użytkownik zmieni dostawcę lub głos – skasuj hash, by wymusić ponowny odczyt
        _sidebar_sig = _hash_key(provider or "", str(openai_voice_selected or ""))
        if st.session_state.get("_sidebar_sig") != _sidebar_sig:
            st.session_state["_sidebar_sig"] = _sidebar_sig
            st.session_state.pop("tts_last_hash", None)
            st.session_state.pop("latest_tts_audio_hash", None)
            st.session_state.pop("latest_tts_audio_bytes", None)
            st.session_state.pop("latest_tts_audio_mime", None)

    # 0) Wczytanie danych
    # Etap 2 na DO/Streamlit CC probkuje bez brania samego poczatku pliku.
    # Dla szeregów czasowych to krytyczne: próbka musi widzieć cały zakres dat.
    load_sample_rows = None
    if "use_sample" in locals() and "sample_rows_eda" in locals() and use_sample:
        load_sample_rows = int(sample_rows_eda)

    df_full, latest_info, err = _load_latest_dataset(max_rows=load_sample_rows)
    if err:
        st.error(err)
        st.stop()

    # W trybie próbki _load_latest_dataset zwraca juz rownomierna próbkę z calego
    # Parquetu, więc nie wykonujemy kosztownego df_full.sample(...) po pelnym loadzie.
    df = df_full
    is_sampled = False

    try:
        n_rows_total = int((latest_info or {}).get("n_rows") or df.shape[0])
        is_sampled = bool(load_sample_rows and n_rows_total > df.shape[0])
    except Exception:
        is_sampled = bool(load_sample_rows and df.shape[0] >= int(load_sample_rows))

    # -------------------------------------------------------------
    # CACHE core obliczeń per-dataset (żeby nie liczyć w kółko)
    # -------------------------------------------------------------
    csv_path = latest_info.get("csv_path", "")
    sig_now = _sec7_signature(csv_path, df_full)

    # 1) jeśli zmienił się dataset → reset jak dotąd + wyczyść cache
    if st.session_state.get("sec7_signature") != sig_now:
        reset_sec7_state()
        st.session_state["sec7_signature"] = sig_now

        for k in [
            # override'y
            "task_override", "target_override", "cluster_override",
            "task_override_radio", "target_override_select", "cluster_override_select",
            "eda_cluster_col_override", "eda_cluster_col_override_manual",
            # cache core
            "_cached_df_aug", "_cached_roles",
            "_cached_df_aug_sig", "_cached_roles_sig",
            # cache sec1
            "_cached_sec1_sig", "_cached_sec1",
            EDA_TEMP_CLUSTER_STATE_KEY,
        ]:
            st.session_state.pop(k, None)

    # 2) policz df_aug i roles tylko raz dla tej sygnatury (+ tryb próbki)
    sample_flag = bool(st.session_state.get("eda_use_sample", False))
    sample_n = int(st.session_state.get("eda_sample_rows", df.shape[0]))
    core_sig = f"{sig_now}|sample={sample_flag}|n={sample_n}"

    if st.session_state.get("_cached_df_aug_sig") != core_sig:
        df_aug_cached = _augment_numeric_derivatives_for_ui(df)
        df_aug_cached = _eda_attach_temp_clusters_to_df(df_aug_cached)
        st.session_state["_cached_df_aug"] = df_aug_cached
        st.session_state["_cached_df_aug_sig"] = core_sig

    if st.session_state.get("_cached_roles_sig") != core_sig:
        st.session_state["_cached_roles"] = _infer_eda_roles(st.session_state["_cached_df_aug"], latest_info)
        st.session_state["_cached_roles_sig"] = core_sig

    df_aug = st.session_state["_cached_df_aug"]
    df = df_aug
    roles  = st.session_state["_cached_roles"]

    # ─────────────────────────────────────────────────────────────
    # UŻYTKOWNIK MOŻE NADPISAĆ TASK / TARGET / CLUSTER (deterministycznie)
    # ─────────────────────────────────────────────────────────────
    override_task    = st.session_state.get("task_override")
    override_target  = st.session_state.get("target_override")
    override_cluster = st.session_state.get("cluster_override")

    if override_task in ("regression", "classification", "time_series", "clustering"):
        roles["task"] = override_task

    if override_target is not None and override_target in df.columns:
        roles["target"] = override_target

    if override_cluster is not None and override_cluster in df.columns:
        roles["cluster_col"] = override_cluster

    st.session_state["eda_roles"] = roles


    # ───────────────────────── Sekcje 1–6 (wracają do main) ─────────────────────────
    run_dir    = latest_info.get("run_dir", "(brak)")
    source_name = latest_info.get("source_name", os.path.basename(csv_path) or "(brak nazwy źródła)")
    n_rows_raw = latest_info.get("n_rows", df_full.shape[0])
    n_cols_raw = latest_info.get("n_cols", df_full.shape[1])
    mask_pii   = latest_info.get("pii_masked", True)
    timestamp  = latest_info.get("timestamp", "(brak)")

    # Informacja dla użytkownika o pracy na próbce
    if "is_sampled" in locals() and is_sampled:
        st.caption(
            f"⚡ EDA pracuje teraz na roboczej próbie **{df.shape[0]:,}** wierszy "
            f"z pełnego zbioru **{n_rows_raw:,}**. "
            "Pełny zbiór zostawiamy do trenowania modelu w kolejnym etapie."
        )

    # Dalej cała logika sekcji 1–6 działa już na df (próbka lub pełny zbiór)
    stats_sig = f"{core_sig}|rows={df.shape[0]}|cols={tuple(df.columns)}"

    if st.session_state.get("_cached_sec1_sig") != stats_sig:
        df_key = _df_cache_fingerprint(df)
        global_missing_pct = stage2_cached('Stage2/global_missing_pct', (df_key,), lambda: _calc_global_missing_pct(df))
        info_df, high_null_cols = stage2_cached('Stage2/analyze_columns', (df_key,), lambda: _analyze_columns(df))

        corr_chart, pairs_sorted, corr_drop_suggestions = _build_correlation_report(
            df, info_df, threshold=0.9
        )
        pairs_df_full = _pairs_dataframe(pairs_sorted)

        dups_info = stage2_cached('Stage2/duplicates', (df_key,), lambda: _detect_duplicates(df))

        st.session_state["_cached_sec1_sig"] = stats_sig
        st.session_state["_cached_sec1"] = dict(
            global_missing_pct=global_missing_pct,
            info_df=info_df,
            high_null_cols=high_null_cols,
            corr_chart=corr_chart,
            pairs_sorted=pairs_sorted,
            corr_drop_suggestions=corr_drop_suggestions,
            pairs_df_full=pairs_df_full,
            dups_info=dups_info,
        )
    else:
        _c = st.session_state["_cached_sec1"]
        global_missing_pct     = _c["global_missing_pct"]
        info_df                = _c["info_df"]
        high_null_cols         = _c["high_null_cols"]
        corr_chart             = _c["corr_chart"]
        pairs_sorted           = _c["pairs_sorted"]
        corr_drop_suggestions  = _c["corr_drop_suggestions"]
        pairs_df_full          = _c["pairs_df_full"]
        dups_info              = _c["dups_info"]

    duplicates_count = dups_info["duplicates_count"]
    duplicates_pct   = dups_info["duplicates_pct"]


    auto_drop_candidates = set(info_df.loc[info_df["drop_candidate"], "kolumna"].tolist())
    auto_drop_candidates.update(corr_drop_suggestions)
    auto_drop_candidates = sorted(list(auto_drop_candidates))
    predicted_cols_after_drop = max(0, n_cols_raw - len(auto_drop_candidates))

    readiness_score = _compute_readiness_score(
        high_null_cols=high_null_cols,
        duplicates_pct=duplicates_pct,
        auto_drop_candidates=auto_drop_candidates,
        pairs_sorted=pairs_sorted,
    )

    quality_flags = []
    if len(high_null_cols) > 0:
        quality_flags.append("⚠️ Występują kolumny z dużą liczbą braków (>30%).")
    if duplicates_count > 0:
        quality_flags.append(f"⚠️ Znaleziono {duplicates_count} potencjalnych duplikatów ({duplicates_pct}%).")
    if len(auto_drop_candidates) > 0:
        quality_flags.append("⚠️ Część kolumn wygląda na zbędne / prawie puste / duplikujące sygnał.")
    if not quality_flags:
        quality_flags.append("✅ Dane wyglądają stabilnie. Możesz praktycznie od razu przejść do trenowania modelu.")

    # 1) Status zbioru
    st.header("1. Status zbioru")

    # — 1A. Rząd metryk
    colA, colB, colC, colD, colE = st.columns(5)

    with colA:
        st.markdown(_metric_card_html(
            "Wiersze × Kolumny",
            f"{_fmt_int(n_rows_raw)} × {_fmt_int(n_cols_raw)}",
            "Rozmiar surowego zbioru"
        ), unsafe_allow_html=True)
    with colB:
        color_missing = "#28a745" if global_missing_pct < 10 else ("#ffc107" if global_missing_pct < 30 else "#dc3545")
        st.markdown(_metric_card_html(
            "Braki danych (globalnie)",
            f"{global_missing_pct:.1f}%",
            "Imputacja medianą / trybem / sensowną datą",
            fg=color_missing
        ), unsafe_allow_html=True)
    with colC:
        color_dup = "#28a745" if duplicates_pct < 1 else ("#ffc107" if duplicates_pct < 5 else "#dc3545")
        st.markdown(_metric_card_html(
            "Potencjalne duplikaty",
            f"{_fmt_int(duplicates_count)} ({duplicates_pct:.1f}%)",
            "Możemy je automatycznie usunąć",
            fg=color_dup
        ), unsafe_allow_html=True)
    with colD:
        risk_cols = len(auto_drop_candidates)
        st.markdown(_metric_card_html(
            "Kolumny ryzykowne",
            _fmt_int(risk_cols),
            f"Po czyszczeniu przewidujemy ok. {_fmt_int(predicted_cols_after_drop)} kolumn",
            fg=("#dc3545" if risk_cols > 0 else "#28a745")
        ), unsafe_allow_html=True)
    with colE:
        color_ready = "#28a745" if readiness_score >= 85 else ("#ffc107" if readiness_score >= 60 else "#dc3545")
        st.markdown(_metric_card_html(
            "Data Readiness Index",
            f"{readiness_score} / 100",
            "Czy można już trenować model bez wstydu",
            fg=color_ready
        ), unsafe_allow_html=True)

    # — 1B. Baner jakości danych + box z typem zadania w jednej linii
    if readiness_score >= 85:
        banner_level = "ok"
        banner_text = (
            "Dane wyglądają stabilnie. "
            "Przejdź do dalszych sekcji i zobacz jak "
            "**automatycznie przygotujemy** dane dla modelu predykcyjnego."
        )
    elif readiness_score >= 60:
        banner_level = "warn"
        banner_text = (
            "Widzimy podwyższone braki/duplikaty/ryzykowne kolumny.<br>"
            "**Naprawimy to automatycznie** i przygotujemy dane dla modelu predykcyjnego "
            "w sekcji **7. Przygotowanie danych**."
        )
    else:
        banner_level = "err"
        banner_text = (
            "Dane nie wyglądają dobrze.<br>"
            "**Nie martw się** – w sekcji **7. Przygotowanie danych** "
            "krok po kroku oczyścimy je automatycznie i przygotujemy bezpieczny zbiór "
            "dla modelu predykcyjnego."
        )

    banner_html = _status_banner_html(banner_level, banner_text)

    # box z typem zadania – niebieski, szerokość jak dwie ostatnie metryki
    roles = st.session_state.get("eda_roles") or {}
    task = (roles.get("task") or "regression").lower()

    task_name_main = {
        "regression": "Regresja",
        "classification": "Klasyfikacja",
        "time_series": "Szereg czasowy",
        "clustering": "Klasteryzacja",
    }.get(task, "Inny typ zadania")

    task_name_sub = {
        "regression": "Przewidywanie liczby ciągłej (np. cena, czas, popyt).",
        "classification": "Przewidywanie klasy / etykiety (np. tak/nie, segment).",
        "time_series": "Przewidywanie wartości w czasie (np. prognoza trendu).",
        "clustering": "Odkrywanie naturalnych grup / segmentów w danych.",
    }.get(task, "")

    task_emoji = {
        "regression": "📈",
        "classification": "🏷️",
        "time_series": "⏱️",
        "clustering": "🧩",
    }.get(task, "🤖")

    task_box_html = f"""
    <div style="
        border:1px solid #bfdbfe;
        background:#eff6ff;
        border-radius:0.6rem;
        padding:0.85rem 1.0rem;
        margin-top:0.6rem;
        line-height:1.45;
        min-height:5.0rem;
        display:flex;
        align-items:center;
        box-sizing:border-box;
    ">
      <div style="display:flex;align-items:center;gap:.55rem;">
        <div style="font-size:1.0rem;line-height:1;">{task_emoji}</div>
        <div style="display:flex;flex-direction:column;">
          <span style="font-size:1.0rem;font-weight:600;color:#0b57d0;">
            Zadanie: {task_name_main}
          </span>
          <span style="font-size:0.85rem;color:#1f2937;">
            {task_name_sub}
          </span>
        </div>
      </div>
    </div>
    """

    # układ 3/5 + 2/5 – jak rząd metryk (3 karty + 2 karty)
    col_msg, col_task = st.columns([3, 2])
    with col_msg:
        st.markdown(banner_html, unsafe_allow_html=True)
    with col_task:
        # --- główna ramka z wykrytym zadaniem ---
        st.markdown(task_box_html, unsafe_allow_html=True)

        # --- ramka-hint PODZIELONA: (1) wybór tasku + (2) wybór targetu ---
        # wariant umiarkowany: pokazujemy tylko sensowne opcje
        if task in ("regression", "classification") or roles.get("time_col") or roles.get("cluster_col"):

            active_task = st.session_state.get("task_override") or task

            # bazowe sygnały
            time_col = roles.get("time_col")
            has_time_col = bool(time_col)
            ts_suspected = bool(roles.get("ts_suspected"))

            # bazowe opcje
            task_options = ["regression", "classification"]

            # time_series pokazujemy:
            # - gdy wykryto prawdziwą kolumnę czasu
            # - LUB gdy meta/source sugerują TS (np. PyCaret time series bez time_col)
            if has_time_col or ts_suspected:
                task_options.append("time_series")

            # jeśli heurystyka widzi klasteryzację → pozwól wybrać clustering
            if task == "clustering" or roles.get("cluster_col"):
                task_options.append("clustering")

            # uniknij duplikatów, ustaw index
            task_options = list(dict.fromkeys(task_options))
            active_task_index = task_options.index(active_task) if active_task in task_options else 0

            # aktywny target = override jeśli jest, inaczej auto-wykryty
            active_target = st.session_state.get("target_override") or roles.get("target")
            target_options = list(df.columns)
            if active_target not in target_options and target_options:
                active_target = target_options[0]

            with st.container(border=True):
                left, right = st.columns(2, gap="small")

                # --- LEWA POŁOWA: task ---
                with left:
                    st.markdown(
                        "<span style='color:#0b57d0;font-weight:600;font-size:0.9rem'>"
                        "👶 Jeśli myślisz, że to inne <b>zadanie</b> – wybierz tutaj."
                        "</span>",
                        unsafe_allow_html=True
                    )

                    chosen_task = st.radio(
                        label="Wybór",
                        options=task_options,
                        index=active_task_index,
                        key="task_override_radio",
                        horizontal=True,
                        label_visibility="collapsed",
                    )

                # mikro-hint zależny od danych
                if has_time_col and time_col:
                    st.caption(
                        f"🕒 Wykryto kolumnę czasu: **{time_col}** → możesz też użyć **time_series**."
                    )
                    st.caption("⏱️ Wybierz time_series, gdy chcesz prognozować po czasie.")
                elif ts_suspected:
                    st.caption("🕒 Źródło sugeruje szereg czasowy, ale nie widzę jawnej kolumny czasu.")
                    st.caption("Możesz użyć **time_series** z pseudo-czasem (numer wiersza) lub wskazać oś czasu w sekcji 2.")

                # --- PRAWA POŁOWA: target ---
                with right:
                    st.markdown(
                        "<span style='color:#0b57d0;font-weight:600;font-size:0.9rem'>"
                        "🎯 Jeśli chcesz przewidywać inną <b>kolumnę (target)</b> – wybierz ją tutaj."
                        "</span>",
                        unsafe_allow_html=True
                    )

                    chosen_target = st.selectbox(
                        label="Wybór",
                        options=target_options,
                        index=target_options.index(active_target) if active_target in target_options else 0,
                        key="target_override_select",
                        label_visibility="collapsed",
                    )

            # --- jeśli user zmienił task → zapisz override + rerun ---
            if chosen_task != active_task:
                st.session_state["task_override"] = chosen_task
                st.rerun()

            # --- jeśli user zmienił target → zapisz override + rerun ---
            if chosen_target != active_target:
                st.session_state["target_override"] = chosen_target
                st.rerun()



    # — 1C. Krótka stopka źródła (zostawiamy)
    st.caption(
        f"Źródło danych: **{source_name}** · Plik: {os.path.basename(csv_path)} · "
        f"Maskowanie PII: {'tak' if mask_pii else 'nie'} · Ostatnia aktualizacja: {timestamp}"
    )

    with st.expander("Szczegóły techniczne (dla zespołu data science)"):
        st.markdown(
            _tech_details_html(
                csv_path=csv_path, run_dir=run_dir, mask_pii=mask_pii,
                global_missing_pct=global_missing_pct, duplicates_pct=duplicates_pct,
                auto_drop_candidates=auto_drop_candidates,
                predicted_cols_after_drop=predicted_cols_after_drop
            ),
            unsafe_allow_html=True,
        )

    # 2) Podgląd danych
    st.header("2. Podgląd danych")
    st.caption("Szybki rzut oka na zawartość zbioru. To są dane już po anonimizacji PII (jeśli była włączona).")

    roles = st.session_state.get("eda_roles") or {}
    task = (roles.get("task") or "regression").lower()
    target_col = roles.get("target")
    time_col = roles.get("time_col")
    cluster_col = roles.get("cluster_col")
    if task == "clustering":
        cluster_col_candidates = _eda_cluster_column_candidates(df)
        cluster_override = st.session_state.get("eda_cluster_col_override")
        if cluster_override in cluster_col_candidates:
            cluster_col = cluster_override
        elif cluster_col not in cluster_col_candidates and cluster_col_candidates:
            cluster_col = cluster_col_candidates[0]
        roles["cluster_col"] = cluster_col
        st.session_state["eda_roles"] = roles

    # ---------- HERO: w zależności od typu zadania ----------
    if task == "classification" and target_col and target_col in df.columns:
        _hero_classification_overview(df, target_col)
    elif task == "time_series":
        _hero_time_series_overview(df, time_col, roles)
    elif task == "clustering":
        _hero_clustering_overview(df, cluster_col)
    else:
        # domyślnie regresja / inne z targetem liczbowym
        if target_col and target_col in df.columns:
            _hero_regression_overview(df, target_col)

    # ---------- klasyczny podgląd head / tail pod spodem ----------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Pierwsze wiersze (head)", divider="gray")
        st_df_safe(_df_preview_for_ui(df, max_rows=10), width="stretch", hide_index=True)
    with c2:
        st.subheader("Ostatnie wiersze (tail)", divider="gray")
        st_df_safe(_df_preview_for_ui(df.tail(10), max_rows=10), width="stretch", hide_index=True)


    # 3) Jakość kolumn
    st.header("3. Jakość kolumn")

    # — przygotuj treść banera (żółty/g zielony), ale NIE wyświetlaj go jeszcze —
    if len(high_null_cols) > 0:
        _null_banner_kind = "warn"
        _null_banner_text = (
            "Kolumny z dużą liczbą braków (>30%): "
            + ", ".join(high_null_cols)
            + ". Przy czyszczeniu uzupełnimy je lub usuniemy, jeśli są praktycznie puste/ID."
        )
    else:
        _null_banner_kind = "ok"
        _null_banner_text = "Brak kolumn z krytycznie dużą liczbą braków (>30%)."

    left3, right3 = st.columns([1, 1])

    # --------- LEWA: tabela z flagami (bez zmian) ---------
    with left3:
        info_view_cols = ["kolumna", "logical_type", "null_pct", "n_unique", "comment"]
        df_show = info_df[info_view_cols].copy()

        icon_map = info_df.set_index("kolumna")["flags"].to_dict()
        df_show["comment"] = [
            ((icon_map.get(r.kolumna, "") + " " if icon_map.get(r.kolumna) else "") + (r.comment or ""))
            for r in df_show.itertuples()
        ]

        ui_lang = (st.session_state.get("ui_lang", "PL") or "PL").upper()
        if ui_lang == "PL":
            header_map = {
                "kolumna": "Kolumna",
                "logical_type": "Typ logiczny",
                "null_pct": "% braków",
                "n_unique": "Liczba unikalnych",
                "comment": "Komentarz",
            }
        else:
            header_map = {
                "kolumna": "column",
                "logical_type": "logical_type",
                "null_pct": "% missing",
                "n_unique": "unique values",
                "comment": "comment",
            }

        df_show_sorted = df_show.sort_values("kolumna")
        df_vis = df_show_sorted.rename(columns=header_map)

        st_df_safe(
            df_vis.style.format({header_map["null_pct"]: "{:.1f}"}),
            width="stretch",
            hide_index=True,
        )

        st.markdown("**Legenda flag:**  \n"
                    "🗑 kandydat do usunięcia (ID/stała/prawie pusta)  \n"
                    "🔢 / 📅 automatyczna konwersja typu (tekst→liczba / tekst→data+cechy)  \n"
                    "📝 długie opisy tekstowe (raczej NLP niż klasyczny ML)")

    # --------- PRAWA: wykres braków + legenda kolorów (bez zmian) ---------
    with right3:
        null_plot_df = info_df[["kolumna", "null_pct"]].copy()
        null_plot_df["null_pct"] = null_plot_df["null_pct"].astype(float)
        bar_missing = (
            alt.Chart(null_plot_df).mark_bar()
            .encode(
                x=alt.X("null_pct:Q", title="% braków"),
                y=alt.Y("kolumna:N", sort="-x", title="kolumna"),
                tooltip=[alt.Tooltip("kolumna:N", title="kolumna"),
                        alt.Tooltip("null_pct:Q", title="% braków", format=".1f")],
                color=alt.condition(alt.datum.null_pct > 30, alt.value("#dc3545"), alt.value("#007bff")),
            )
            .properties(height=320, title="Braki danych w kolumnach (% braków)")
        )
        altair_chart_stretch(st, bar_missing, width='stretch')

        legend_html = """
        <div style="margin-top:.5rem;">
        <div style="font-weight:600; margin-bottom:.25rem;">Objaśnienie kolorów:</div>
        <ul style="list-style:none; padding-left:0; margin:0;">
            <li style="display:flex; align-items:center; gap:.5rem; margin:.15rem 0;">
            <span style="display:inline-block; width:18px; height:10px; border-radius:999px; background:#007bff;"></span>
            <span>kolumna w normie</span>
            </li>
            <li style="display:flex; align-items:center; gap:.5rem; margin:.15rem 0;">
            <span style="display:inline-block; width:18px; height:10px; border-radius:999px; background:#dc3545;"></span>
            <span>kolumna z podwyższonym ryzykiem (&gt;30%)</span>
            </li>
        </ul>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

    # --------- TERAZ wyświetlamy baner z brakami POD legendą ---------
    if _null_banner_kind == "warn":
        st.warning(_null_banner_text)
    else:  # "ok"
        st.success(_null_banner_text)


    st.info("Kolumny prawie puste, stałe lub wyglądające jak ID oznaczyliśmy jako kandydatów do usunięcia 🗑. "
            "Zajmiemy się nimi automatycznie w kroku 7, żeby model nie uczył się śmieci.")

    # 4) Analiza wybranej kolumny
    t4 = _perf('Stage2/4')
    st.header("4. Analiza wybranej kolumny")
    st.caption("Zbadaj konkretną kolumnę: rozkład, odstające wartości, dominujące kategorie, pokrycie TOP segmentów.")

    available_cols = _ordered_analysis_columns(df)
    col_to_plot = st.selectbox(
        "Wybierz kolumnę do analizy:",
        options=available_cols,
        index=0 if available_cols else None,
    )

    # Meta o zadaniu – przyda się do trybów klasyfikacja / klasteryzacja
    roles = st.session_state.get("eda_roles") or {}
    task = (roles.get("task") or "regression").lower()
    target_col = roles.get("target")
    cluster_col = roles.get("cluster_col")

    if col_to_plot:
        s_col_raw = df[col_to_plot]
        s_col = s_col_raw.copy()

        _non_null_cnt = int(s_col_raw.notna().sum())
        _nunique_sel = int(s_col_raw.nunique(dropna=True)) if _non_null_cnt else 0
        _uniq_ratio_sel = (_nunique_sel / max(_non_null_cnt, 1)) if _non_null_cnt else 0.0
        _is_integer_like_numeric = False
        if pd.api.types.is_numeric_dtype(s_col_raw):
            try:
                _sample_num = pd.to_numeric(s_col_raw, errors="coerce").dropna().head(5000)
                if not _sample_num.empty:
                    _is_integer_like_numeric = float((_sample_num - _sample_num.round()).abs().max()) < 1e-9
            except Exception:
                _is_integer_like_numeric = False
        _id_like_numeric = bool(pd.api.types.is_numeric_dtype(s_col_raw) and _is_integer_like_numeric and _uniq_ratio_sel > 0.9)

        if (col_to_plot in auto_drop_candidates) or _id_like_numeric:
            st.warning(
                f"Kolumna **{col_to_plot}** wygląda jak identyfikator lub pole techniczne. "
                "Wykres jest poprawny, ale jego wartość diagnostyczna dla modelu bywa ograniczona. "
                "Takie kolumny zwykle traktujemy jako kandydatów do usunięcia albo wyłączenia z trenowania."
            )

        # jeśli nie jest numeryczna, spróbuj z naszych parserów specjalnych
        if not pd.api.types.is_numeric_dtype(s_col):
            try_coerced = _coerce_to_numeric_special(s_col)
            if pd.notna(try_coerced).mean() > 0.8:  # wystarczająco dużo parsuje się na liczbę
                s_col = try_coerced

        # ───────────────────── PODSTAWOWA ANALIZA KOLUMNY (jak dotychczas) ─────────────────────
        if pd.api.types.is_numeric_dtype(s_col):
            st.caption("Czy to wygląda zdrowo?")
            combo_chart, stats_table, comment_text, details = _numeric_distribution_details(
                s_col, col_to_plot
            )
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                altair_chart_stretch(st, combo_chart, width='stretch')
            with cc2:
                st.subheader("Metryki kolumny", divider="gray")
                st_df_safe(stats_table, hide_index=True, width='stretch')

            outlier_ratio = details.get("outlier_ratio_pct", 0.0)
            if outlier_ratio > 5.0:
                st.warning(
                    "Widzimy wartości odstające (czerwone punkty / słupki). "
                    "Domyślnie nie usuwamy ich automatycznie – zostaną oznaczone flagą `is_outlier_*`. "
                    "Nie skasujemy ich bez pytania."
                )
            else:
                st.success("Ta kolumna wygląda stabilnie — model dostanie czysty sygnał.")

            st.info(comment_text)

            # ───────────────────── POZIOM 2 DLA NUMERYCZNYCH (REGRESJA) ─────────────────────
            if (
                task == "regression"
                and target_col
                and target_col in df.columns
                and col_to_plot != target_col
            ):
                st.subheader("Jak ta cecha wpływa na target?", divider="gray")

                df_num = pd.DataFrame(
                    {
                        "feature": pd.to_numeric(s_col, errors="coerce"),
                        "target": pd.to_numeric(df[target_col], errors="coerce"),
                    }
                ).dropna()

                if df_num.empty:
                    st.info("Brak danych jednocześnie w wybranej kolumnie i w targetcie.")
                else:
                    # ─── próbkujemy dla wydajności ───
                    MAX_PTS = 5000
                    if len(df_num) > MAX_PTS:
                        df_plot = df_num.sample(MAX_PTS, random_state=42)
                        st.caption(f"Pokazuję próbkę {MAX_PTS:,} punktów (dla wydajności).")
                    else:
                        df_plot = df_num

                    # ─── przypisowa legenda (subtelna) ───
                    st.caption(
                        "ℹ️ LOESS = wygładzona krzywa trendu (nieliniowa). "
                        "Rolling mean (linia przerywana) = średnia krocząca po posortowanej cesze."
                    )

                    # ─── scatter + LOESS + rolling mean ───
                    base = alt.Chart(df_plot).mark_point(
                        filled=True, size=22, opacity=0.25
                    ).encode(
                        x=alt.X("feature:Q", title=col_to_plot),
                        y=alt.Y("target:Q", title=target_col),
                        tooltip=[
                            alt.Tooltip("feature:Q", title=col_to_plot, format=".3g"),
                            alt.Tooltip("target:Q", title=target_col, format=".3g"),
                        ],
                    )

                    loess = alt.Chart(df_plot).transform_loess(
                        "feature", "target", bandwidth=0.35
                    ).mark_line(size=3).encode(
                        x="feature:Q",
                        y="target:Q",
                    )

                    rolling = alt.Chart(df_plot).transform_window(
                        rolling_mean="mean(target)",
                        sort=[alt.SortField("feature")],
                        frame=[-200, 200],
                    ).mark_line(size=2, strokeDash=[6, 4]).encode(
                        x="feature:Q",
                        y="rolling_mean:Q",
                    )

                    chart_scatter = (base + rolling + loess).properties(height=340)
                    altair_chart_stretch(st, chart_scatter, width='stretch')

                    # ─── korelacje ───
                    spearman = float(df_num["feature"].corr(df_num["target"], method="spearman"))
                    pearson = float(df_num["feature"].corr(df_num["target"], method="pearson"))

                    # ─── krótki wniosek pod wykresem ───
                    rho_abs = abs(spearman) if np.isfinite(spearman) else 0.0
                    if rho_abs < 0.10:
                        strength_txt = "brak wyraźnej zależności"
                    elif rho_abs < 0.30:
                        strength_txt = "słaba zależność"
                    elif rho_abs < 0.50:
                        strength_txt = "umiarkowana zależność"
                    elif rho_abs < 0.70:
                        strength_txt = "silna zależność"
                    else:
                        strength_txt = "bardzo silna zależność"

                    if spearman > 0:
                        dir_txt = "rosnąca"
                    elif spearman < 0:
                        dir_txt = "malejąca"
                    else:
                        dir_txt = "płaska"

                    nonlin_hint = (
                        np.isfinite(spearman) and np.isfinite(pearson)
                        and (abs(spearman) > abs(pearson) + 0.15)
                    )

                    extra = " z lekką nieliniowością" if nonlin_hint else ""
                    st.markdown(
                        f"✅ **Wniosek:** relacja jest **{dir_txt}**, "
                        f"({strength_txt}{extra})."
                    )


                    # ───────────────────── SUBTELNE KAFELKI + PANELE GRUP (v3) ─────────────────────
                    st.markdown(
                        """
                        <style>
                        .eda-group{
                            border:1px solid #e6e8ef;
                            background:#fbfcff;
                            border-radius:14px;
                            padding:10px 12px;
                            margin-top:4px;   /* mniej pustki pod wykresem */
                            margin-bottom:6px;
                        }
                        .eda-tile{
                            border:1px solid #eef0f3;
                            background:#f9fafb;
                            border-radius:12px;
                            padding:10px 12px;
                            height:108px;
                            display:flex;
                            flex-direction:column;
                            justify-content:space-between;
                            box-sizing:border-box;
                        }
                        .eda-tile-wide{
                            border:1px solid #e6e9ef;
                            background:#ffffff;
                            border-radius:12px;
                            padding:10px 12px;
                            height:108px;
                            display:flex;
                            flex-direction:column;
                            justify-content:space-between;
                            box-sizing:border-box;
                            margin-top:8px;
                            border-left:6px solid var(--vcol);
                        }
                        .eda-tile-title{
                            font-size:0.88rem;
                            font-weight:600;
                            color:#374151;
                        }
                        .eda-tile-value{
                            font-size:1.45rem;
                            font-weight:800;
                            color:#111827;
                            line-height:1.0;
                        }
                        .eda-pill{
                            display:inline-block;
                            font-size:0.80rem;
                            font-weight:700;
                            padding:2px 8px;
                            border-radius:999px;
                            background:rgba(0,0,0,0.04);
                            color:#111827;
                            width:fit-content;
                        }
                        .eda-help{
                            font-size:0.78rem;
                            color:#6b7280;
                            margin-top:4px;
                            line-height:1.25;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    def _level_from_abs(x: float, thresholds):
                        for ub, label, color in thresholds:
                            if x < ub:
                                return label, color
                        return thresholds[-1][1], thresholds[-1][2]

                    def _tile_html(title, value, label, color, help_txt=None, wide=False):
                        help_html = f"<div class='eda-help'>{help_txt}</div>" if help_txt else ""
                        if wide:
                            return f"""
                            <div class="eda-tile-wide" style="--vcol:{color}">
                                <div class="eda-tile-title">{title}</div>
                                <div class="eda-tile-value">{value}</div>
                                <div class="eda-pill" style="color:{color}; background:{color}18;">{label}</div>
                                {help_html}
                            </div>
                            """
                        return f"""
                        <div class="eda-tile">
                            <div class="eda-tile-title">{title}</div>
                            <div class="eda-tile-value">{value}</div>
                            <div class="eda-pill" style="color:{color}; background:{color}18;">{label}</div>
                            {help_html}
                        </div>
                        """

                    CORR_THRESH = [
                        (0.10, "bardzo słaba", "#b91c1c"),
                        (0.30, "słaba",        "#b45309"),
                        (0.50, "umiarkowana",  "#a16207"),
                        (0.70, "silna",        "#15803d"),
                        (10.0, "bardzo silna", "#166534"),
                    ]

                    # słownictwo zgodne z resztą appki
                    SKEW_THRESH = [
                        (0.50, "stabilna",       "#166534"),
                        (1.00, "lekko skośna",   "#15803d"),
                        (2.00, "skośna",         "#a16207"),
                        (3.00, "mocno skośna",   "#b45309"),
                        (10.0, "bardzo skośna",  "#b91c1c"),
                    ]

                    # 50/50 jak w ustaleniach
                    colL, colR = st.columns(2)

                    # ───────────────────── PANEL 1: Korelacje ─────────────────────
                    with colL:
                        st.subheader("Korelacje", divider="gray")

                        sp_val = spearman if np.isfinite(spearman) else 0.0
                        pr_val = pearson  if np.isfinite(pearson)  else 0.0

                        sp_lbl, sp_col = _level_from_abs(abs(sp_val), CORR_THRESH)
                        pr_lbl, pr_col = _level_from_abs(abs(pr_val), CORR_THRESH)

                        rho_abs = abs(sp_val) if np.isfinite(spearman) else 0.0
                        verdict_lbl, verdict_col = _level_from_abs(rho_abs, CORR_THRESH)

                        cA, cB = st.columns(2)
                        with cA:
                            st.markdown(
                                _tile_html(
                                    "Spearman ρ (odporny)",
                                    f"{sp_val:.3f}" if np.isfinite(spearman) else "n/a",
                                    sp_lbl,
                                    sp_col,
                                    "Monotoniczny sygnał, odporny na outliery."
                                ),
                                unsafe_allow_html=True
                            )
                        with cB:
                            st.markdown(
                                _tile_html(
                                    "Pearson r (liniowy)",
                                    f"{pr_val:.3f}" if np.isfinite(pearson) else "n/a",
                                    pr_lbl,
                                    pr_col,
                                    "Siła zależności liniowej."
                                ),
                                unsafe_allow_html=True
                            )

                        # WERDYKT szeroki pod spodem (nadrzędny)
                        st.markdown(
                            _tile_html(
                                "Werdykt (|ρ|)",
                                f"{rho_abs:.3f}",
                                verdict_lbl,
                                verdict_col,
                                "Podsumowanie siły zależności (Spearman + Pearson).",
                                wide=True
                            ),
                            unsafe_allow_html=True
                        )

                    # ───────────────────── PANEL 2: Diagnostyka nieliniowości ─────────────────────
                    with colR:
                        st.subheader("Diagnostyka nieliniowości i transformacje", divider="gray")

                        skew_feat = float(df_num["feature"].skew())
                        skew_tgt  = float(df_num["target"].skew())
                        feat_pos = bool((df_num["feature"] > 0).all())
                        tgt_pos  = bool((df_num["target"] > 0).all())

                        nonlin_hint = False
                        if np.isfinite(spearman) and np.isfinite(pearson):
                            if abs(spearman) > abs(pearson) + 0.15:
                                nonlin_hint = True

                        skf_lbl, skf_col = _level_from_abs(abs(skew_feat), SKEW_THRESH)
                        skt_lbl, skt_col = _level_from_abs(abs(skew_tgt),  SKEW_THRESH)

                        dA, dB = st.columns(2)
                        with dA:
                            st.markdown(
                                _tile_html("Skośność cechy", f"{skew_feat:.2f}", skf_lbl, skf_col),
                                unsafe_allow_html=True
                            )
                        with dB:
                            st.markdown(
                                _tile_html("Skośność targetu", f"{skew_tgt:.2f}", skt_lbl, skt_col),
                                unsafe_allow_html=True
                            )

                        recs = []
                        if nonlin_hint:
                            recs.append(
                                "⚠️ **Sygnał nieliniowości:** Spearman wyraźnie > Pearson → zależność monotoniczna, ale nieliniowa."
                            )

                        if feat_pos and skew_feat > 1.0:
                            recs.append("✅ **Rekomendacja:** rozważ `log1p(feature)` (silna prawoskośność i wartości dodatnie).")
                        elif feat_pos and abs(skew_feat) > 0.8:
                            recs.append("✅ **Rekomendacja:** rozważ **Box-Cox** (cecha dodatnia, umiarkowana/silna skośność).")
                        elif (not feat_pos) and abs(skew_feat) > 0.8:
                            recs.append("✅ **Rekomendacja:** rozważ **Yeo-Johnson** (działa też dla 0/ujemnych).")

                        if tgt_pos and skew_tgt > 1.0:
                            recs.append("ℹ️ Target prawoskośny → w modelu możesz rozważyć `log1p(target)`.")

                        if recs:
                            st.markdown("\n\n".join(recs))
                        else:
                            st.markdown("Brak wyraźnej potrzeby transformacji — sygnał wygląda stabilnie.")

                        st.caption(
                            "Transformacje pomagają modelom liniowym i stabilizują wariancję. "
                            "Drzewa (CatBoost/LightGBM) często poradzą sobie bez nich, ale mogą zyskać na stabilności sygnału."
                        )

                    # ───────────────────── KONIEC PANELI ─────────────────────

            # ───────────────── Dodatkowo: sezonowość i trend dla szeregu czasowego ─────────────────
            roles = st.session_state.get("eda_roles") or {}
            task_ts = (roles.get("task") or "").lower()
            time_col = roles.get("time_col")

            # pokazuj sekcję TS także gdy user wymusi time_series bez kolumny czasu
            if task_ts == "time_series":
                # jeśli inferred time_col jest poprawny, użyj go jako preferowanego,
                # w przeciwnym razie zostaw None i oprzyj się o pseudo-czas
                if not (time_col and time_col in df.columns):
                    time_col = None

                # 1 divider jak w sekcji 2 (bez st.divider)
                st.subheader("Sezonowość i trend (dla szeregu czasowego)", divider="gray")
                st.markdown("<div style='margin-top:-0.60rem'></div>", unsafe_allow_html=True)

                st.caption(
                    "Poniżej możesz zobaczyć, jak średnia lub suma wartości zmienia się w czasie "
                    "dla wybranej agregacji (dzień / tydzień / miesiąc / rok / dzień tygodnia). "
                    "Dodatkowo pokazujemy średnią kroczącą oraz linię trendu od momentu, "
                    "w którym wykrywamy trwały trend. Skala trendu: od -3 (silny spadek) "
                    "przez 0 (brak trendu) do +3 (silny wzrost)."
                )

                chart_col, info_col = st.columns([4, 1.5])

                # ---------- wybór osi czasu + agregacji + miary ----------
                def _datetime_candidates(df_: pd.DataFrame, preferred: str) -> list[str]:
                    out = []
                    if preferred and preferred in df_.columns:
                        out.append(preferred)
                    for c in df_.columns:
                        if c != preferred and pd.api.types.is_datetime64_any_dtype(df_[c]):
                            out.append(c)
                    return out

                x_cands = _datetime_candidates(df, time_col)
                x_options = x_cands + ["Numer wiersza (pseudo-czas)"]
                dow_map = {0: "Pon", 1: "Wt", 2: "Śr", 3: "Czw", 4: "Pt", 5: "Sob", 6: "Nd"}

                with chart_col:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        x_choice = st.selectbox(
                            "Oś czasu (oś X)",
                            x_options,
                            index=0,
                            key="ts_seasonality_x",
                        )
                    with c2:
                        agg_choice = st.selectbox(
                            "Agregacja czasu",
                            ["Dzień", "Tydzień", "Miesiąc", "Rok", "Dzień tygodnia"],
                            index=0,
                            key="ts_seasonality_agg",
                        )
                    with c3:
                        metric_choice = st.selectbox(
                            "Miara wartości",
                            ["Średnia", "Suma"],
                            index=0,
                            key="ts_seasonality_metric",
                        )

# ---------- agregacja danych ----------
                agg_df = pd.DataFrame()
                x_title = ""
                y_title = ""

                try:
                    # 1) PSEUDO-CZAS (brak kolumny czasu)
                    if x_choice == "Numer wiersza (pseudo-czas)":
                        ts_df = df[[col_to_plot]].copy()

                        # zawsze nadpisz pewnym indeksem int
                        ts_df["__idx__"] = np.arange(len(ts_df), dtype=int)
                        ts_df = ts_df.dropna(subset=[col_to_plot])

                        if ts_df.empty:
                            agg_df = pd.DataFrame()
                        else:
                            if agg_choice == "Dzień tygodnia":
                                ts_df["season_key"] = (ts_df["__idx__"] % 7).astype(int)
                                x_title = "Dzień tygodnia (pseudo-czas)"
                            else:
                                step_map = {"Dzień": 1, "Tydzień": 7, "Miesiąc": 30, "Rok": 365}
                                step = int(step_map.get(agg_choice, 1))
                                ts_df["season_key"] = (ts_df["__idx__"] // max(1, step)).astype(int)
                                x_title = f"{agg_choice} (pseudo-czas)"

                            if "season_key" in ts_df.columns:
                                if metric_choice == "Suma":
                                    agg_df = (
                                        ts_df.groupby("season_key", dropna=True)[col_to_plot]
                                        .sum()
                                        .reset_index()
                                    )
                                    y_title = "Suma"
                                else:
                                    agg_df = (
                                        ts_df.groupby("season_key", dropna=True)[col_to_plot]
                                        .mean()
                                        .reset_index()
                                    )
                                    y_title = "Średnia"

                    # 2) PRAWDZIWY CZAS (kolumna datetime)
                    else:
                        ts_df = df[[x_choice, col_to_plot]].copy()

                        # 1) oś czasu próbujemy wymusić na datetime
                        ts_df[x_choice] = pd.to_datetime(ts_df[x_choice], errors="coerce")

                        # 2) wartości muszą być liczbowe do agregacji (mean/sum/median)
                        ts_df[col_to_plot] = pd.to_numeric(ts_df[col_to_plot], errors="coerce")

                        # 3) czyścimy braki po konwersjach
                        ts_df = ts_df.dropna(subset=[x_choice, col_to_plot])

                        # jeśli po konwersjach nic nie zostało – ustaw pusty agg_df
                        # (dalszy guard w PATCH 2 pokaże komunikat zamiast crasha)
                        if ts_df.empty:
                            agg_df = pd.DataFrame()
                            x_enc = None

                        if ts_df.empty:
                            agg_df = pd.DataFrame()
                        else:
                            dt = ts_df[x_choice].dt

                            if agg_choice == "Rok":
                                ts_df["season_key"] = dt.to_period("Y").dt.start_time
                                x_title = "Rok"
                            elif agg_choice == "Kwartal":
                                ts_df["season_key"] = dt.to_period("Q").dt.start_time
                                x_title = "Kwartal"
                            elif agg_choice == "Miesiąc":
                                ts_df["season_key"] = dt.to_period("M").dt.start_time
                                x_title = "Miesiąc"
                            elif agg_choice == "Tydzień":
                                ts_df["season_key"] = dt.to_period("W-SUN").dt.start_time
                                x_title = "Tydzień"
                            elif agg_choice == "Dzień tygodnia":
                                ts_df["season_key"] = dt.dayofweek.astype(int)
                                x_title = "Dzień tygodnia"
                            else:  # "Dzień"
                                ts_df["season_key"] = dt.floor("D")
                                x_title = "Data (dzień)"

                            if "season_key" in ts_df.columns:
                                if metric_choice == "Suma":
                                    agg_df = (
                                        ts_df.groupby("season_key", dropna=True)[col_to_plot]
                                        .sum()
                                        .reset_index()
                                    )
                                    y_title = "Suma"
                                else:
                                    agg_df = (
                                        ts_df.groupby("season_key", dropna=True)[col_to_plot]
                                        .mean()
                                        .reset_index()
                                    )
                                    y_title = "Średnia"

                    # normalizacja kolumny wynikowej -> `value`
                    if not agg_df.empty:
                        agg_df = agg_df.rename(columns={col_to_plot: "value"})
                        agg_df = agg_df.dropna(subset=["season_key", "value"])
                        agg_df = agg_df.sort_values("season_key").reset_index(drop=True)

                        if agg_choice == "Dzień tygodnia":
                            agg_df["season_label"] = agg_df["season_key"].map(dow_map).fillna(
                                agg_df["season_key"].astype(str)
                            )
                            _sort_order = [v for v in dow_map.values() if v in set(agg_df["season_label"])]
                            x_enc = alt.X(
                                "season_label:N",
                                sort=_sort_order,
                                axis=alt.Axis(title=x_title, labelAngle=0),
                            )
                        elif pd.api.types.is_datetime64_any_dtype(agg_df["season_key"]):
                            _fmt_map = {
                                "Dzień": "%Y-%m-%d",
                                "Tydzień": "%Y-%m-%d",
                                "Miesiąc": "%Y-%m",
                                "Kwartal": "%Y-Q%q",
                                "Rok": "%Y",
                            }
                            _fmt = _fmt_map.get(agg_choice, "%Y-%m-%d")
                            _dt_key = pd.to_datetime(agg_df["season_key"], errors="coerce")
                            if agg_choice == "Kwartal":
                                agg_df["season_label"] = _dt_key.dt.to_period("Q").astype(str)
                            else:
                                agg_df["season_label"] = _dt_key.dt.strftime(_fmt)
                            _sort_order = agg_df["season_label"].astype(str).tolist()
                            x_enc = alt.X(
                                "season_label:N",
                                sort=_sort_order,
                                axis=alt.Axis(
                                    title=x_title,
                                    labelAngle=-35 if len(_sort_order) > 8 else 0,
                                ),
                            )
                        else:
                            agg_df["season_label"] = agg_df["season_key"].astype(str)
                            _sort_order = agg_df["season_label"].tolist()
                            x_enc = alt.X(
                                "season_label:N",
                                sort=_sort_order,
                                axis=alt.Axis(
                                    title=x_title,
                                    labelAngle=-35 if len(_sort_order) > 8 else 0,
                                ),
                            )

                except Exception as e:
                    agg_df = pd.DataFrame()
                    st.warning(
                        f"Nie udało się policzyć sezonowości/trendu dla wybranych ustawień: {e}"
                    )


                if agg_df.empty:
                    st.info("Brak danych do analizy po agregacji.")
                else:
                    # ── GUARD: czasem x_enc nie powstaje (np. niestandardowe dane / filtry)
                    _x_title = locals().get("x_title", "Czas")
                    if "x_enc" not in locals() or x_enc is None:
                        x_enc = alt.X(
                            "season_key:O",
                            sort="ascending",
                            axis=alt.Axis(title=_x_title, labelAngle=0),
                        )

                    # ── GUARD: w skrajnych przypadkach season_key może nie istnieć
                    if "season_key" not in agg_df.columns:
                        agg_df = agg_df.reset_index().rename(columns={"index": "season_key"})
                    
                    COLOR_RAW = "#2563eb"
                    COLOR_MA = "#f59e0b"
                    COLOR_TREND = "#10b981"

                    n_points = len(agg_df)
                    can_analyze_ts = n_points >= 3
                    if can_analyze_ts:
                        max_ma = max(2, min(60, n_points))
                        default_ma = max(2, min(6, max_ma))
                    else:
                        max_ma = 1
                        default_ma = 1

                    # --- układ suwak + checkboxy (zgodnie z Twoim layoutem) ---
                    with chart_col:
                        if not can_analyze_ts:
                            st.info(
                                f"Po agregacji „{agg_choice}” są tylko {n_points} punkty. "
                                "Trend, średnia krocząca i sezonowość wymagają minimum 3 punktów. "
                                "Wybierz drobniejszą agregację, np. dzień, albo szerszy zakres danych."
                            )
                            ma_window = 1
                            show_raw = True
                            show_ma = False
                            show_trend = False
                        else:
                            # suwak na szerokość 2 pól wyboru, checkboxy pod 3. polem
                            row = st.columns([2.0, 1.0])

                            with row[0]:
                                # Clamp persisted slider state before rendering. Streamlit keeps
                                # widget values across reruns, while max_ma changes with aggregation.
                                prev_ma = st.session_state.get("ts_ma_window", default_ma)
                                try:
                                    prev_ma = int(prev_ma)
                                except Exception:
                                    prev_ma = int(default_ma)
                                safe_ma = int(max(2, min(max_ma, prev_ma)))
                                st.session_state["ts_ma_window"] = safe_ma
                                ma_window = st.slider(
                                    "Okno średniej kroczącej",
                                    min_value=2, max_value=max_ma,
                                    value=safe_ma, step=1,
                                    key="ts_ma_window",
                                )

                            with row[1]:
                                # lekki spacer, żeby checkboxy były w linii z suwakiem
                                st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)

                                cb1, cb2, cb3 = st.columns(3, gap="small")
                                with cb1:
                                    show_raw = st.checkbox("dane", True, key="ts_show_raw")
                                with cb2:
                                    show_ma = st.checkbox("śr. krocząca", True, key="ts_show_ma")
                                with cb3:
                                    show_trend = st.checkbox("trend", True, key="ts_show_trend")

                    # zabezpieczenia (gdyby UI nie wyrenderowało się w danym rerunie)
                    if can_analyze_ts:
                        ma_window  = st.session_state.get("ts_ma_window", default_ma)
                        ma_window = int(max(2, min(max_ma, ma_window)))
                        show_raw   = st.session_state.get("ts_show_raw", True)
                        show_ma    = st.session_state.get("ts_show_ma", True)
                        show_trend = st.session_state.get("ts_show_trend", True)
                    else:
                        ma_window = 1
                        show_raw = True
                        show_ma = False
                        show_trend = False

                    agg_df = agg_df.reset_index(drop=True)
                    agg_df["ma_value"] = agg_df["value"].rolling(
                        window=ma_window, min_periods=max(1, ma_window // 2)
                    ).mean()

                    # --- trend start (MAD + spójność) ---
                    y_ma = agg_df["ma_value"].to_numpy(dtype=float)
                    n = len(y_ma)
                    trend_start_idx = 0
                    if n >= max(5, ma_window):
                        diffs = np.diff(y_ma)
                        mad = np.nanmedian(np.abs(diffs - np.nanmedian(diffs)))
                        thr = 1.5 * mad if np.isfinite(mad) else 0.0
                        run = 0
                        sign = 0
                        for i_d, d in enumerate(diffs):
                            if not np.isfinite(d) or abs(d) <= thr:
                                run = 0; sign = 0; continue
                            cur_sign = 1 if d > 0 else -1
                            run = run + 1 if cur_sign == sign else 1
                            sign = cur_sign
                            if run >= max(3, ma_window // 2):
                                trend_start_idx = max(0, i_d - run + 1)
                                break

                    seg = agg_df.iloc[trend_start_idx:].copy()
                    x_seg = np.arange(len(seg), dtype=float)
                    y_seg = seg["ma_value"].to_numpy(dtype=float)
                    finite = np.isfinite(y_seg)

                    if finite.sum() >= 2:
                        try:
                            slope = float(np.polyfit(x_seg[finite], y_seg[finite], 1)[0])
                        except np.linalg.LinAlgError:
                            slope = 0.0
                    else:
                        slope = 0.0

                    seg["trend_value"] = slope * x_seg + float(
                        np.nanmean(y_seg[finite]) if finite.any() else 0.0
                    )
                    agg_df["trend_value"] = np.nan
                    agg_df.loc[trend_start_idx:, "trend_value"] = seg["trend_value"].to_numpy()

                    # --- NORMALIZACJA PO PEŁNYM ZAKRESIE (spójna z kątem) ---
                    full_range = float(np.nanmax(y_ma) - np.nanmin(y_ma)) if np.isfinite(y_ma).any() else 0.0
                    denom = (full_range / max(n - 1, 1)) if full_range > 0 else 1.0
                    norm = 0.0 if full_range == 0 else float(slope) / denom

                    abs_norm = abs(norm)
                    if abs_norm < 0.02:
                        strength, level = "brak wyraźnego trendu", 0
                    elif abs_norm < 0.07:
                        strength, level = "lekki trend", 1
                    elif abs_norm < 0.15:
                        strength, level = "średni trend", 2
                    else:
                        strength, level = "silny trend", 3

                    if norm > 0:
                        direction, trend_arrow, signed_level = "wzrostowy", "↗", level
                    elif norm < 0:
                        direction, trend_arrow, signed_level = "spadkowy", "↘", -level
                    else:
                        direction, trend_arrow, signed_level = "płaski", "→", 0

                    # Kąt w skali wykresu (surowy) + kąt spójny z normalizacją
                    angle_deg_raw = float(np.degrees(np.arctan(slope)))
                    angle_deg_norm = float(np.degrees(np.arctan(norm)))

                    # --- zakres osi Y z headroomem (żeby linie nie kleiły się do sufitu) ---
                    y_cols = ["value"]
                    if "ma_value" in agg_df.columns:
                        y_cols.append("ma_value")
                    if "trend_value" in agg_df.columns:
                        y_cols.append("trend_value")

                    y_min = float(np.nanmin(agg_df[y_cols].to_numpy()))
                    y_max = float(np.nanmax(agg_df[y_cols].to_numpy()))
                    y_range = y_max - y_min

                    # % zapasu na górze/dole: 0.08 = 8% (możesz zmienić np. na 0.12)
                    pad = 0.8 * y_range if y_range > 0 else (0.05 * abs(y_max) if y_max != 0 else 1.0)
                    y_scale = alt.Scale(domain=[y_min - pad, y_max + pad], nice=True, zero=True)

                    base = alt.Chart(agg_df).encode(x=x_enc)

                    layers = []
                    if show_raw:
                        layers.append(
                            base.mark_line(strokeWidth=1.7, color=COLOR_RAW, point=True).encode(
                                x=x_enc,  # <-- wymuszenie osi X na warstwie danych
                                y=alt.Y("value:Q", title=y_title, scale=y_scale)
                            )
                        )

                    if show_ma:
                        layers.append(
                            base.mark_line(strokeWidth=2.2, color=COLOR_MA, strokeDash=[5,3]).encode(
                                x=x_enc,  # <-- wymuszenie osi X na warstwie MA
                                y=alt.Y("ma_value:Q", title=y_title, scale=y_scale)
                            )
                        )

                    if show_trend:
                        # linia trendu
                        layers.append(
                            base.mark_line(strokeWidth=2.0, color=COLOR_TREND, strokeDash=[2,2]).encode(
                                x=x_enc,  # <-- wymuszenie osi X na warstwie trendu
                                y=alt.Y("trend_value:Q", title=y_title, scale=y_scale)
                            )
                        )


                        # punkt startu trendu — ten sam X co reszta, ale BEZ osi,
                        # żeby Altair nie dorysował drugiej osi na górze.
                        if agg_choice == "Dzień tygodnia":
                            x_point = alt.X(
                                "season_label:N",
                                sort=list(dow_map.values()),
                                axis=alt.Axis(labels=False, ticks=False, title=None),
                            )
                        elif agg_choice == "Dzień":
                            # datetime lub pseudo-czas — oba są bezpieczne na season_key
                            x_point = alt.X(
                                "season_key:T" if np.issubdtype(agg_df["season_key"].dtype, np.datetime64) else "season_key:Q",
                                sort="ascending",
                                axis=alt.Axis(labels=False, ticks=False, title=None),
                            )
                        else:
                            x_point = alt.X(
                                "season_key:Q",
                                sort="ascending",
                                axis=alt.Axis(labels=False, ticks=False, title=None),
                            )

                        layers.append(
                            alt.Chart(agg_df.iloc[[trend_start_idx]])
                            .mark_point(size=120, filled=True, color=COLOR_TREND, opacity=0.9)
                            .encode(
                                x=x_enc,
                                y=alt.Y("trend_value:Q", title=y_title, scale=y_scale),
                                tooltip=[
                                    alt.Tooltip("season_label:N", title=x_title),
                                    alt.Tooltip("trend_value:Q", format=".2f", title="Start trendu"),
                                ],
                            )
                        )

                    with chart_col:
                        chart = (
                            alt.layer(*layers)
                            .properties(
                                height=365,
                                padding={"left": 5, "right": 5, "top": 0, "bottom": 20},
                            )
                            .configure_axis(titlePadding=10, labelPadding=6)
                            .configure_view(stroke=None)
                        )
                        altair_chart_stretch(st, chart, width='stretch')

                        # expander MAKSYMALNIE blisko wykresu
                        # (zmniejszamy odstęp dodawany przez Streamlit POD wykresem Altair)
                        st.markdown("""
                        <style>
                        /* Streamlit dodaje spory margin pod wykresem (vega-embed). 
                        Ściągamy ten odstęp, żeby expander przykleił się do wykresu. */
                        div[data-testid="stElementContainer"]:has(div.vega-embed) {
                            margin-bottom: -1.8rem !important;
                        }

                        /* dodatkowo lekko podciągamy sam expander */
                        div[data-testid="stExpander"] {
                            margin-top: -0.2rem !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        with st.expander("Jak liczymy trend i co oznacza znormalizowane nachylenie?"):

                            st.markdown(
                                "- **Trend** to prosta dopasowana metodą najmniejszych kwadratów do punktów po agregacji.\n"
                                "- **Początek trendu** wykrywamy automatycznie (MAD): szukamy pierwszego spójnego fragmentu nachylenia.\n"
                                "- **Znormalizowane nachylenie** liczymy względem **pełnego zakresu serii po agregacji**, "
                                "dzięki czemu jest spójne z kątem trendu.\n"
                                "- **Skala 1–3 / 0 / -1–-3:**\n"
                                "  - |norm| < 0.02 → **0 (brak trendu)**\n"
                                "  - 0.02–0.07 → **±1 (lekki trend)**\n"
                                "  - 0.07–0.15 → **±2 (średni trend)**\n"
                                "  - ≥ 0.15 → **±3 (silny trend)**"
                            )

                        # ============================
                        # SEZONOWOŚĆ – METRYKI PROFILU
                        # ============================
                        season_cycle_name = "brak sezonowości w tej skali"
                        season_strength = "brak / śladowa"
                        season_amp = 0.0
                        season_rel_amp = 0.0
                        season_reg = 0.0
                        eta2 = 0.0  # udział wariancji wyjaśniony przez cykl (η²)

                        # Które agregacje traktujemy jako cykl sezonowy?
                        cyclical_map = {
                            "Dzień tygodnia": ("profil tygodniowy (7 dni)", 7, 3),
                            "Miesiąc": ("cykl roczny (12 miesięcy)", 12, 18),
                            "Tydzień": ("cykl roczny-tygodniowy (52 tyg.)", 52, 52),
                        }

                        if agg_choice in cyclical_map:
                            season_cycle_name, season_period, min_points_for_cycle = cyclical_map[agg_choice]

                            if n_points < min_points_for_cycle:
                                season_cycle_name = (
                                    f"{season_cycle_name} — za mało punktów po agregacji "
                                    f"({n_points}/{min_points_for_cycle})"
                                )
                                season_strength = "nie oceniono"
                            else:
                                prof = agg_df[["season_key", "value"]].dropna().copy()
                                if agg_choice == "Dzień tygodnia":
                                    prof["cycle_key"] = prof["season_key"]
                                elif agg_choice == "Miesiąc":
                                    prof["cycle_key"] = pd.to_datetime(
                                        prof["season_key"], errors="coerce"
                                    ).dt.month
                                elif agg_choice == "Tydzień":
                                    prof["cycle_key"] = pd.to_datetime(
                                        prof["season_key"], errors="coerce"
                                    ).dt.isocalendar().week.astype("Int64")
                                else:
                                    prof["cycle_key"] = prof["season_key"]

                                prof = prof.dropna(subset=["cycle_key"])
                            if n_points >= min_points_for_cycle and not prof.empty:
                                overall = float(prof["value"].mean())

                                grp_mean = prof.groupby("cycle_key")["value"].mean()

                                # amplituda sezonowa
                                season_amp = float(grp_mean.max() - grp_mean.min())
                                season_rel_amp = 0.0 if overall == 0 else float(season_amp / abs(overall))

                                # η² (effect size ANOVA) – udział wariancji wyjaśniony przez cykl
                                ss_total = float(((prof["value"] - overall) ** 2).sum())
                                ss_between = float(((grp_mean - overall) ** 2).sum() * (len(prof) / max(grp_mean.shape[0], 1)))
                                eta2 = 0.0 if ss_total == 0 else ss_between / ss_total

                                # regularność profilu: 1 - (within/total)
                                resid = prof["value"] - prof["cycle_key"].map(grp_mean)
                                ss_within = float((resid ** 2).sum())
                                season_reg = 0.0 if ss_total == 0 else max(0.0, 1.0 - ss_within / ss_total)

                                # opis jakościowy siły sezonowości
                                if eta2 < 0.02:
                                    season_strength = "brak / śladowa"
                                elif eta2 < 0.07:
                                    season_strength = "lekka"
                                elif eta2 < 0.15:
                                    season_strength = "średnia"
                                else:
                                    season_strength = "silna"

                    with info_col:
                        st.subheader("Ocena zjawisk", divider="gray")
                        st.markdown("**Ocena trendu:**")
                        if can_analyze_ts:
                            st.markdown(
                                f"""
                                • **Oś czasu:** {x_choice}<br>
                                • **Agregacja:** {agg_choice.lower()}<br>
                                • **Miara:** {metric_choice.lower()}<br>
                                • **Trend:** {trend_arrow} {strength} {direction} (**{signed_level}/3**,
                                  znormalizowane nachylenie ≈ {norm:+.3f},
                                  start trendu od punktu {trend_start_idx + 1} z {n_points} — zaznaczony kropką).<br>
                                • **Kąt nachylenia trendu:** {angle_deg_raw:+.1f}°
                                (znormalizowany: {angle_deg_norm:+.1f}°)
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                • **Oś czasu:** {x_choice}<br>
                                • **Agregacja:** {agg_choice.lower()}<br>
                                • **Miara:** {metric_choice.lower()}<br>
                                • **Trend:** nie oceniono — po agregacji są tylko {n_points} punkty.
                                """,
                                unsafe_allow_html=True,
                            )
                        st.markdown("**Ocena sezonowości:**")
                        st.markdown(
                            f"""
                            • **Cykl:** {season_cycle_name}<br>
                            • **Siła sezonowości (η²):** {eta2:.2f} — {season_strength}
                            (udział wariancji wyjaśniony przez cykl)<br>
                            • **Amplituda cyklu:** {season_amp:.3g} 
                            ({season_rel_amp*100:.1f}% średniej)<br>
                            • **Regularność profilu:** {season_reg}
                            """,
                            unsafe_allow_html=True,
                        )

                        # legenda sklejona i bez rozstrzału
                        st.markdown("**Legenda:**")
                        st.markdown(
                            f"""
                            <div style="margin-top:-0.9rem;font-size:0.95rem;line-height:1.15;">
                              <div><span style="color:{COLOR_RAW};font-weight:700;display:inline-block;width:16px;">●</span>dane</div>
                              <div><span style="color:{COLOR_MA};font-weight:700;display:inline-block;width:16px;">●</span>średnia krocząca</div>
                              <div><span style="color:{COLOR_TREND};font-weight:700;display:inline-block;width:16px;">●</span>trend</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown("")

        else:
            # ============= BLOK DLA KOLUMN NIENUMERYCZNYCH (KATEGORIALNYCH) =============
            st.caption("Rozkład kategorii (TOP kategorie + pokrycie)")

            try:
                # _categorical_distribution_details zwraca:
                # (chart_top, chart_coverage, table_top_df, comment_text)
                cat_top_chart, cat_cov_chart, cat_top_df, cat_comment = _categorical_distribution_details(
                    s_col_raw, col_to_plot
                )

                c1, c2 = st.columns([2, 1])
                with c1:
                    altair_chart_stretch(st, cat_top_chart, width='stretch')
                    altair_chart_stretch(st, cat_cov_chart, width='stretch')
                with c2:
                    st.subheader("Metryki kategorii", divider="gray")
                    st_df_safe(cat_top_df, hide_index=True, width='stretch')

                if cat_comment:
                    st.info(cat_comment)

                n_unique = int(s_col_raw.nunique(dropna=True))
                if n_unique > 50:
                    st.info(
                        f"Kolumna ma dużo unikalnych kategorii ({n_unique}). "
                        "Poniżej pokazujemy TOP 20. W modelu rozważ target encoding lub CatBoost."
                    )

            except Exception as e:
                st.warning(
                    f"Nie udało się policzyć rozkładu kategorii dla '{col_to_plot}'. "
                    f"Szczegóły: {e}"
                )

            # ───────────────────── TRYB REGRESJI: wpływ cechy na target ─────────────────────
            if (
                task == "regression"
                and target_col
                and target_col in df.columns
                and col_to_plot != target_col
            ):
                st.subheader("Jak ta cecha wpływa na target?", divider="gray")

                df_reg = pd.DataFrame(
                    {
                        "feature_raw": s_col_raw.astype(str),
                        "target": pd.to_numeric(df[target_col], errors="coerce"),
                    }
                ).dropna()

                if df_reg.empty:
                    st.info("Brak danych jednocześnie w wybranej kolumnie i w targetcie.")
                else:
                    # --- ograniczamy kardynalność do TOP 20 + '(pozostałe)' ---
                    vc = df_reg["feature_raw"].value_counts(dropna=False)
                    top_k = min(20, len(vc))
                    keep_cats = vc.head(top_k).index.tolist()
                    df_reg["feature"] = np.where(
                        df_reg["feature_raw"].isin(keep_cats),
                        df_reg["feature_raw"],
                        "(pozostałe)",
                    )

                    overall_mean = float(df_reg["target"].mean())

                    def _iqr(x: pd.Series) -> float:
                        return float(x.quantile(0.75) - x.quantile(0.25))

                    def _trim_mean(x: pd.Series, p: float = 0.05) -> float:
                        x = x.sort_values()
                        k = int(len(x) * p)
                        if len(x) <= 2 * k:
                            return float(x.mean())
                        return float(x.iloc[k:-k].mean())

                    grp = df_reg.groupby("feature")["target"]

                    stats_df = grp.agg(
                        n="count",
                        mean="mean",
                        median="median",
                        std="std",
                    ).reset_index()

                    stats_df["iqr"] = grp.apply(_iqr).values
                    stats_df["trim_mean5"] = grp.apply(_trim_mean).values
                    stats_df["delta_%"] = np.where(
                        overall_mean != 0,
                        (stats_df["mean"] / overall_mean - 1.0) * 100.0,
                        0.0,
                    )

                    stats_df = stats_df.sort_values("mean", ascending=False)

                    # --- miary wpływu: η² (ANOVA-like) oraz ε² (Kruskal–Wallis, odporna) ---

                    def _effect_label(v: float) -> str:
                        if not np.isfinite(v):
                            return "brak wpływu"
                        if v < 0.01:
                            return "praktycznie brak wpływu"
                        elif v < 0.06:
                            return "słaby wpływ"
                        elif v < 0.14:
                            return "umiarkowany wpływ"
                        else:
                            return "silny wpływ"

                    # η²
                    ss_total = float(((df_reg["target"] - overall_mean) ** 2).sum())
                    ss_between = float(
                        (stats_df["n"] * (stats_df["mean"] - overall_mean) ** 2).sum()
                    )
                    eta2 = 0.0 if ss_total == 0 else max(0.0, min(1.0, ss_between / ss_total))

                    # ε² (Kruskal–Wallis)
                    eps2 = np.nan
                    try:
                        from scipy.stats import kruskal
                        groups = [g["target"].values for _, g in df_reg.groupby("feature")]
                        k = len(groups)
                        n_total = int(df_reg.shape[0])
                        if k >= 2 and n_total > k and all(len(g) > 0 for g in groups):
                            H, p_kw = kruskal(*groups)
                            eps2 = float((H - k + 1) / (n_total - k))
                            eps2 = max(0.0, min(1.0, eps2))
                    except Exception:
                        eps2 = np.nan

                    # heurystyki wyboru miary (prosto + bezpiecznie)
                    n_total = int(df_reg.shape[0])
                    min_n_eff = max(20, int(0.01 * n_total))  # min 20 lub 1% zbioru
                    small_groups = bool((stats_df["n"] < min_n_eff).any())

                    # szybka ocena "nienormalności / outlierów" przez skośność targetu w grupach
                    try:
                        skew_by_group = df_reg.groupby("feature")["target"].skew().abs()
                        non_normal = bool((skew_by_group > 1.5).any())
                    except Exception:
                        non_normal = False

                    # wybór: jeśli grupy małe lub rozkłady "trudne" → preferuj ε²
                    use_eps2 = (small_groups or non_normal or not np.isfinite(eta2)) and np.isfinite(eps2)

                    score_used = eps2 if use_eps2 else eta2
                    score_lbl = _effect_label(score_used)

                    # krótkie uzasadnienie wyboru
                    why_lines = []
                    if use_eps2:
                        if small_groups:
                            why_lines.append("małe/graniczne liczebności grup")
                        if non_normal:
                            why_lines.append("skośne rozkłady/outliery w grupach")
                        why_txt = "Używamy ε² (Kruskal), bo są " + ", ".join(why_lines) + "."
                    else:
                        why_txt = "Używamy η² (ANOVA-like), bo grupy są wystarczająco liczne i rozkłady stabilne."

                    # zapamiętujemy do późniejszego renderingu
                    effect_score_used = score_used
                    effect_score_lbl = score_lbl
                    effect_use_eps2 = use_eps2
                    effect_why_txt = why_txt


                    # --- wykres wpływu ---
                    n_cats_plot = stats_df.shape[0]
                    colL, colR = st.columns([2, 1])

                    if n_cats_plot <= 12:
                        order_cats = stats_df["feature"].tolist()
                        chart_imp = (
                            alt.Chart(df_reg)
                            .mark_boxplot(extent=1.5)
                            .encode(
                                x=alt.X(
                                    "feature:N",
                                    sort=order_cats,
                                    title=col_to_plot,
                                    axis=alt.Axis(labelAngle=0),
                                ),
                                y=alt.Y("target:Q", title=target_col),
                                tooltip=[
                                    alt.Tooltip("feature:N", title="Kategoria"),
                                    alt.Tooltip("target:Q", title=target_col, format=".3g"),
                                ],
                            )
                            .properties(height=320)
                        )
                    else:
                        stats_df["se"] = stats_df["std"] / np.sqrt(stats_df["n"].clip(lower=1))
                        stats_df["ci95"] = 1.96 * stats_df["se"]
                        stats_df["ci_low"] = stats_df["mean"] - stats_df["ci95"]
                        stats_df["ci_high"] = stats_df["mean"] + stats_df["ci95"]

                        base = alt.Chart(stats_df).encode(
                            y=alt.Y("feature:N", sort="-x", title=col_to_plot),
                            x=alt.X("mean:Q", title=f"Średni {target_col}"),
                            tooltip=[
                                alt.Tooltip("feature:N", title="Kategoria"),
                                alt.Tooltip("n:Q", title="n", format=","),
                                alt.Tooltip("mean:Q", title="Średnia", format=".3g"),
                                alt.Tooltip("median:Q", title="Mediana", format=".3g"),
                                alt.Tooltip("delta_%:Q", title="Î” vs global [%]", format=".1f"),
                            ],
                        )

                        bars = base.mark_bar()
                        err = base.mark_errorbar().encode(
                            x="ci_low:Q",
                            x2="ci_high:Q",
                        )

                        h = max(240, min(620, 22 * n_cats_plot + 80))
                        chart_imp = alt.layer(bars, err).properties(height=h)

                    with colL:
                        altair_chart_stretch(st, chart_imp, width='stretch')

                    with colR:
                        st.subheader("Statystyki targetu w kategoriach", divider="gray")
                        show_df = stats_df.rename(
                            columns={
                                "feature": "kategoria",
                                "mean": "średnia",
                                "median": "mediana",
                                "std": "odch.std.",
                                "trim_mean5": "trim_mean(5%)",
                                "delta_%": "Î” vs global [%]",
                            }
                        )
                        st_df_safe(
                            show_df[
                                ["kategoria", "n", "średnia", "mediana", "odch.std.", "iqr", "trim_mean(5%)", "Î” vs global [%]"]
                            ],
                            hide_index=True,
                            width='stretch',
                        )

                    # --- tekst pod wykresem w 2 kontenerach (UX / Gestalt) ---
                    top_txt, bot_txt = "", ""
                    try:
                        top2 = show_df.head(2)
                        bot2 = show_df.tail(2)
                        top_txt = ", ".join(
                            [f"{r.kategoria} (Î” {r['Î” vs global [%]']:+.1f}%)" for _, r in top2.iterrows()]
                        )
                        bot_txt = ", ".join(
                            [f"{r.kategoria} (Î” {r['Î” vs global [%]']:+.1f}%)" for _, r in bot2.iterrows()]
                        )
                    except Exception:
                        pass

                    # --- tekst pod wykresem MA być w obrębie kolumny z wykresem (colL) ---
                    with colL:
                        left_c, right_c = st.columns([1.15, 1.0], gap="small")

                        with left_c:
                            if top_txt or bot_txt:
                                st.markdown(
                                    f"• Najwyższy target mają: **{top_txt}**.  \n"
                                    f"• Najniższy target mają: **{bot_txt}**."
                                )

                        with right_c:
                            # 1) główny werdykt wpływu (czarny tekst)
                            if effect_use_eps2:
                                main_line = (
                                    f"**Wpływ cechy na target (ε², Kruskal–Wallis): "
                                    f"`{effect_score_used:.3f}` → {effect_score_lbl}.**"
                                )
                                comp_line = f"Dla porównania η² (ANOVA-like): `{eta2:.3f}`." if np.isfinite(eta2) else ""
                            else:
                                main_line = (
                                    f"**Wpływ cechy na target (η², ANOVA-like): "
                                    f"`{effect_score_used:.3f}` → {effect_score_lbl}.**"
                                )
                                comp_line = f"Dla porównania ε² (Kruskal): `{eps2:.3f}`." if np.isfinite(eps2) else ""

                            lines = [main_line]
                            if effect_why_txt:
                                lines.append(effect_why_txt)
                            if comp_line:
                                lines.append(comp_line)

                            # D) skrócone microcopy
                            lines.append("Im wyżej, tym stabilniejszy sygnał.")

                            # E) równe odstępy jak po lewej: bez pustych linii
                            st.markdown("\n".join(lines))

                    # ───────────────────── POZIOM 2: typ cechy + rekomendowane kodowanie ─────────────────────
                    with st.expander("Poziom 2: typ cechy nienumerycznej i rekomendowane kodowanie", expanded=False):

                        # --- heurystyki typu cechy ---
                        s_feat = s_col_raw.dropna().astype(str)
                        n_feat = len(s_feat)
                        n_unique = int(s_feat.nunique(dropna=True))
                        uniq_ratio = (n_unique / n_feat) if n_feat > 0 else 0.0
                        avg_len = float(s_feat.str.len().mean()) if n_feat > 0 else 0.0

                        # ordinal tokens (PL+EN) – prosta lista, bez ryzyka false positive dla tekstu
                        ORD_TOKENS = {
                            # EN
                            "low","medium","high","very high","very_low","very_high",
                            "bad","fair","good","very good","excellent","poor",
                            "small","medium","large","xl","xxl","xs","s","m","l",
                            # PL
                            "niski","średni","wysoki","bardzo wysoki","bardzo niski",
                            "zły","średni","dobry","bardzo dobry","doskonały",
                            "mały","średni","duży","bardzo duży",
                        }

                        # czy wygląda jak liczby zakodowane jako string?
                        num_like = pd.to_numeric(s_feat, errors="coerce")
                        frac_num_like = float(num_like.notna().mean()) if n_feat > 0 else 0.0

                        # czy w wartościach są tokeny porządkujące?
                        low_vals = set(v.strip().lower() for v in s_feat.unique()[:200])
                        has_ordinal_tokens = any(any(tok in v for tok in ORD_TOKENS) for v in low_vals)

                        if uniq_ratio > 0.6 and avg_len > 8:
                            feat_kind = "tekstowa"
                        elif frac_num_like >= 0.8:
                            feat_kind = "ordinalna (liczbowa w stringu)"
                        elif has_ordinal_tokens and n_unique <= 30:
                            feat_kind = "ordinalna (nazwana)"
                        else:
                            feat_kind = "nominalna"

                        st.markdown(f"**Wykryty typ cechy:** `{feat_kind}`")

                        # --- stabilność kategorii ---
                        min_n = max(20, int(0.01 * n_feat))  # min 20 lub 1% zbioru
                        vc_full = s_feat.value_counts()
                        weak_cats = vc_full[vc_full < min_n]

                        if weak_cats.empty:
                            st.success(
                                f"✅ Kategorie są stabilne: każda ma ≥ {min_n} obserwacji."
                            )
                        else:
                            st.warning(
                                f"⚠️ {len(weak_cats)} kategorii ma małą liczebność (< {min_n}). "
                                "Średnie targetu dla nich mogą być niestabilne."
                            )
                            st.caption("Najrzadsze kategorie:")
                            st_df_safe(
                                weak_cats.head(10).rename("liczebność").reset_index().rename(columns={"index":"kategoria"}),
                                hide_index=True,
                                width='stretch',
                            )

                        # --- rekomendacja kodowania ---
                        rec_lines = []

                        if feat_kind.startswith("tekstowa"):
                            rec_lines += [
                                "• **Tekstowa cecha**: rozważ TF-IDF / embeddings. ",
                                "• Jeśli to naprawdę identyfikatory/unikaty, rozważ wyłączenie lub ekstrakcję prostych cech (długość, liczba słów).",
                            ]

                        elif feat_kind.startswith("ordinalna"):
                            rec_lines += [
                                "• **Ordinalna**: najlepiej zachować porządek → **ordinal encoding** (mapowanie na 0..k).",
                                "• Drzewa (CatBoost/LightGBM) poradzą sobie też z kodowaniem kategorycznym, ale kolejność może nie być wtedy w pełni wykorzystana.",
                            ]

                        else:  # nominalna
                            if n_unique <= 10:
                                rec_lines += [
                                    "• **Nominalna o małej kardynalności**: **one-hot encoding** będzie najbezpieczniejszy.",
                                ]
                            elif n_unique <= 50:
                                rec_lines += [
                                    "• **Nominalna średniej kardynalności**: one-hot dla modeli liniowych, ",
                                    "  a dla drzew → **CatBoost encoding / target encoding**.",
                                ]
                            else:
                                rec_lines += [
                                    "• **Nominalna o wysokiej kardynalności**: preferuj **target encoding** lub **CatBoost encoding**.",
                                    "• One-hot da ogromny wymiar i spadek jakości/wydajności.",
                                ]

                        leak_warn = (
                            "⚠️ **Uwaga na leakage:** target encoding stosuj **tylko na train** "
                            "(w CV/foldach), nigdy na pełnym zbiorze przed podziałem."
                        )

                        st.markdown("\n".join(rec_lines))
                        st.info(leak_warn)


        # ───────────────────── TRYB KLASYFIKACJI: rozkład cechy w klasach ─────────────────────
        if (
            task == "classification"
            and target_col
            and target_col in df.columns
            and col_to_plot != target_col
        ):
            st.subheader("Jak ta cecha rozróżnia klasy?", divider="gray")

            # Budujemy ramkę z cechą (po ewentualnym rzutowaniu) i klasą
            df_cls = pd.DataFrame(
                {
                    "feature": s_col,
                    "class": df[target_col],
                }
            ).dropna()

            if df_cls.empty:
                st.info("Brak danych jednocześnie w wybranej kolumnie i w targetcie.")
            else:
                # Limitujemy liczbę punktów do wizualizacji dla wydajności
                max_points = 5000
                if len(df_cls) > max_points:
                    df_vis = df_cls.sample(max_points, random_state=0)
                else:
                    df_vis = df_cls

                if pd.api.types.is_numeric_dtype(df_cls["feature"]):
                    # Gęstość cechy osobno dla każdej klasy (KDE)
                    chart_overlay = (
                        alt.Chart(df_vis)
                        .transform_density(
                            "feature",
                            as_=["feature", "density"],
                            groupby=["class"],
                        )
                        .mark_area(opacity=0.45)
                        .encode(
                            x=alt.X("feature:Q", title=col_to_plot),
                            y=alt.Y("density:Q", title="Gęstość"),
                            color=alt.Color("class:N", title=target_col),
                            tooltip=[
                                alt.Tooltip("class:N", title="Klasa"),
                            ],
                        )
                        .properties(height=260)
                    )
                    altair_chart_stretch(st, chart_overlay, width='stretch')

                    means = df_cls.groupby("class")["feature"].mean()
                    overall_std = float(df_cls["feature"].std() or 0.0)
                    separation = 0.0
                    if overall_std > 0 and math.isfinite(overall_std):
                        separation = float((means.max() - means.min()) / overall_std)

                    if separation >= 1.5:
                        level_txt = "bardzo dobrze rozróżnia klasy"
                    elif separation >= 0.8:
                        level_txt = "całkiem dobrze rozróżnia klasy"
                    elif separation >= 0.3:
                        level_txt = "słabo rozróżnia klasy"
                    else:
                        level_txt = "praktycznie nie rozróżnia klas"

                    st.caption(
                        f"Wskaźnik separacji (zasięg średnich / σ): **{separation:.2f}** – "
                        f"ta cecha {level_txt}."
                    )
                else:
                    # Cechy kategoryczne – udział klas w każdej kategorii
                    df_cat = df_cls.copy()
                    df_cat["feature"] = df_cat["feature"].astype(str)

                    freq = (
                        df_cat.groupby(["feature", "class"])
                        .size()
                        .reset_index(name="count")
                    )
                    if freq.empty:
                        st.info("Brak danych do analizy rozkładu klas w kategoriach.")
                    else:
                        freq["total_cat"] = freq.groupby("feature")["count"].transform("sum")
                        freq["share"] = freq["count"] / freq["total_cat"] * 100.0
                                            
                        chart_cat = (
                            alt.Chart(freq)
                            .mark_bar()
                            .encode(
                                # ⬇⬇⬇ tu dodajemy axis=Alt.Axis(labelAngle=0) – poziome etykiety
                                x=alt.X(
                                    "feature:N",
                                    title=col_to_plot,
                                    axis=alt.Axis(labelAngle=0),
                                ),
                                y=alt.Y("share:Q", title="Udział klasy [%]"),
                                color=alt.Color("class:N", title=target_col),
                                tooltip=[
                                    alt.Tooltip("feature:N", title="Kategoria"),
                                    alt.Tooltip("class:N", title="Klasa"),
                                    alt.Tooltip("share:Q", title="Udział [%]", format=".1f"),
                                    alt.Tooltip("count:Q", title="Liczba rekordów", format=","),
                                ],
                            )
                            .properties(height=555)
                        )

                        # Układ jak w Sekcji 2: wykres po lewej, panel z oceną po prawej
                        col_cls_chart, col_cls_panel = st.columns([4, 1.6])

                        with col_cls_chart:
                            altair_chart_stretch(st, chart_cat, width='stretch')

                        # Wskaźnik „czystości” kategorii – jak bardzo kategorie są jednorodne klasowo
                        pivot = (
                            df_cat.groupby(["feature", "class"])
                            .size()
                            .unstack(fill_value=0)
                        )
                        totals = pivot.sum(axis=1)
                        probs = pivot.div(totals, axis=0)
                        majority = probs.max(axis=1)
                        purity = float((majority * totals).sum() / max(float(totals.sum()), 1.0))

                        if purity >= 0.8:
                            purity_txt = "bardzo dobrze rozróżnia klasy (większość kategorii jest jednorodna)"
                        elif purity >= 0.65:
                            purity_txt = "całkiem nieźle rozróżnia klasy"
                        elif purity >= 0.55:
                            purity_txt = "słabo rozróżnia klasy"
                        else:
                            purity_txt = "praktycznie nie rozróżnia klas"

                        # --- dodatkowy score rozróżniania klas: 1 - ważona entropia (0..1) ---
                        # Entropia per kategoria mierzy "mieszanie się" klas. Normalizujemy przez log(K),
                        # gdzie K = liczba klas. Score = 1 - entropia ważona => im wyżej, tym lepiej rozróżnia.
                        n_classes = probs.shape[1]
                        eps = 1e-12  # stabilność numeryczna

                        entropy_cat = -(probs * np.log(probs + eps)).sum(axis=1)  # entropia w kategoriach
                        entropy_cat_norm = entropy_cat / np.log(max(n_classes, 2))  # normalizacja do 0..1

                        weights = totals / max(float(totals.sum()), 1.0)  # udział kategorii w danych
                        weighted_entropy = float((entropy_cat_norm * weights).sum())
                        discr_score = float(1.0 - weighted_entropy)

                        if discr_score >= 0.80:
                            discr_txt = "bardzo wysokie — kategorie są zwykle jednoznacznie przypisane do jednej klasy"
                        elif discr_score >= 0.65:
                            discr_txt = "wysokie — większość kategorii ma wyraźnie dominującą klasę"
                        elif discr_score >= 0.50:
                            discr_txt = "umiarkowane — część kategorii miesza klasy, ale widać pewne różnice"
                        else:
                            discr_txt = "niskie — rozkłady klas w kategoriach są mocno wymieszane"

                        with col_cls_panel:
                            st.markdown(
                                "<div style='font-size:1.5rem; font-weight:700; margin:0 0 0.35rem 0;'>"
                                "Szybka ocena rozróżniania klas"
                                "</div>",
                                unsafe_allow_html=True,
                            )


                            # ========== 1) CZYSTOŚĆ KATEGORII ==========
                            st.markdown(
                                f"• **Czystość kategorii:** `{purity:.2f}`  \n"
                                f"  {purity_txt.capitalize()}."
                            )
                            st.caption(
                                "Czystość = ważona średnia udziału *najliczniejszej klasy* w każdej kategorii. "
                                "Zakres 0–1: 0 oznacza pełne wymieszanie klas w kategoriach, "
                                "1 oznacza, że niemal każda kategoria należy głównie do jednej klasy."
                            )

                            # ========== 2) SCORE ROZRÓŻNIANIA (ENTROPIA) ==========
                            # U Ciebie ten score jest już policzony wyżej jako discr_score / discr_txt
                            st.markdown(
                                f"• **Score rozróżniania klas:** `{discr_score:.2f}`  \n"
                                f"  {discr_txt.capitalize()}."
                            )
                            st.caption(
                                "Score = 1 − (ważona, znormalizowana entropia rozkładów klas w kategoriach). "
                                "Zakres 0–1: 0 oznacza pełne wymieszanie klas, 1 niemal idealną separację."
                            )

                            # ========== 3) WERDYKT ŁĄCZONY (POJEDYNCZA OCENA) ==========
                            # Średnia obu miar + kara za silną niezgodność
                            disagreement = abs(purity - discr_score)
                            penalty = max(0.0, disagreement - 0.15) * 0.5   # kara dopiero gdy różnica > 0.15
                            combined_score = max(0.0, min(1.0, 0.5 * (purity + discr_score) - penalty))

                            if combined_score < 0.40:
                                combined_txt = "praktycznie nie rozróżnia klas"
                                combined_lvl = "niski"
                            elif combined_score < 0.55:
                                combined_txt = "słabo rozróżnia klasy"
                                combined_lvl = "słaby"
                            elif combined_score < 0.70:
                                combined_txt = "umiarkowanie rozróżnia klasy"
                                combined_lvl = "umiarkowany"
                            elif combined_score < 0.85:
                                combined_txt = "dobrze rozróżnia klasy"
                                combined_lvl = "dobry"
                            else:
                                combined_txt = "bardzo dobrze rozróżnia klasy"
                                combined_lvl = "bardzo dobry"

                            st.markdown(
                                f"• **Werdykt łączony:** `{combined_score:.2f}`  \n"
                                f"  **{combined_txt.capitalize()}** (poziom: *{combined_lvl}*)."
                            )

                            if disagreement > 0.25:
                                st.caption(
                                    "Uwaga: miary czystości i entropii mocno się różnią. "
                                    "Werdykt łączony uwzględnia tę niezgodność, ale warto zajrzeć w wykres."
                                )


    # ───────────────────── TRYB KLASTERYZACJI: profil wybranego klastra ─────────────────────
    if task == "clustering":
        active_df_key = _df_cache_fingerprint(df)
        temp_cluster_state = st.session_state.get(EDA_TEMP_CLUSTER_STATE_KEY) or {}
        temp_cluster_active = (
            temp_cluster_state.get("df_key") == active_df_key
            and str(temp_cluster_state.get("col_name") or "") in df.columns
        )

        cluster_col_candidates = _eda_cluster_column_candidates(df)
        if cluster_col and cluster_col in df.columns and cluster_col not in cluster_col_candidates:
            cluster_col_candidates = [cluster_col] + cluster_col_candidates

        if cluster_col_candidates:
            if cluster_col in cluster_col_candidates:
                _cluster_idx = cluster_col_candidates.index(cluster_col)
            else:
                _cluster_idx = 0
            cluster_col = st.selectbox(
                "Kolumna z etykieta klastra / segmentu:",
                options=cluster_col_candidates,
                index=_cluster_idx,
                key="eda_cluster_col_override",
                help="Wybierz kolumne, ktora identyfikuje klaster lub segment, aby pokazac profil klastra i wygenerowac nazwy AI.",
            )
            roles["cluster_col"] = cluster_col
            st.session_state["eda_roles"] = roles
        else:
            manual_cluster_candidates = [
                c for c in df.columns
                if not pd.api.types.is_datetime64_any_dtype(df[c])
                and _infer_logical_type(df[c]) != "id_like"
                and not str(c).lower().startswith("is_")
            ]
            manual_options = ["(wybierz recznie)"] + manual_cluster_candidates
            selected_manual_cluster = st.selectbox(
                "Kolumna z etykieta klastra / segmentu:",
                options=manual_options,
                index=0,
                key="eda_cluster_col_override_manual",
                help="Nie znalezlismy oczywistej kolumny klastra. Wybierz recznie kolumne, ktora identyfikuje segment.",
            )

            if selected_manual_cluster != "(wybierz recznie)":
                cluster_col = selected_manual_cluster
                roles["cluster_col"] = cluster_col
                st.session_state["eda_roles"] = roles
                st.warning(
                    "Kolumna klastra nie zostala rozpoznana automatycznie. Uzywamy wyboru recznego."
                )
            else:
                _eda_record_checkpoint(
                    "eda.cluster.unavailable",
                    reason="no_cluster_column_candidates",
                )
                st.info(
                    "Nie znalezlismy w danych oczywistej kolumny wygladajacej na etykiete klastra lub segmentu. "
                    "Mozesz wskazac taka kolumne recznie albo utworzyc robocze klastry automatycznie."
                )

                st.markdown("**Opcja awaryjna: robocze klastry automatyczne**")
                st.caption(
                    "Gdy w danych nie ma gotowej etykiety segmentu, zbudujemy tymczasowe klastry z najsensowniejszych cech liczbowych."
                )
                temp_k = st.slider(
                    "Liczba roboczych klastrow",
                    min_value=2,
                    max_value=6,
                    value=int(st.session_state.get("eda_temp_cluster_k", 4)),
                    key="eda_temp_cluster_k",
                )
                temp_run = st.button(
                    "🧪 Utworz robocze klastry automatycznie",
                    key="btn_auto_temp_clusters",
                    width="stretch",
                    type="primary",
                )
                if temp_run:
                    try:
                        state = _eda_create_temp_clusters_and_store(df, k=int(temp_k), max_features=6)
                        cluster_col = str(state.get("col_name") or EDA_TEMP_CLUSTER_COL)
                        roles["cluster_col"] = cluster_col
                        st.session_state["eda_roles"] = roles
                        st.session_state["eda_cluster_col_override"] = cluster_col
                        st.session_state.pop("eda_cluster_col_override_manual", None)
                        st.success(
                            f"Utworzylismy {int(state.get('clusters_count', 0))} robocze klastry "
                            f"na podstawie cech: {', '.join(state.get('features_used', [])[:4])}."
                        )
                        st.rerun()
                    except Exception as exc:
                        _eda_record_checkpoint("eda.cluster.temp.error", error=str(exc))
                        st.warning(f"Nie udalo sie utworzyc roboczych klastrow: {exc}")

        if temp_cluster_active and cluster_col == str(temp_cluster_state.get("col_name")):
            features_used = list(temp_cluster_state.get("features_used") or [])
            preview = ", ".join(features_used[:4])
            if len(features_used) > 4:
                preview += ", ..."
            st.info(
                f"Uzywamy roboczych klastrow automatycznych ({int(temp_cluster_state.get('clusters_count', 0))} segmenty). "
                f"Do ich zbudowania wykorzystano: {preview}."
            )

    if task == "clustering" and cluster_col and cluster_col in df.columns:
        st.subheader("Profil wybranego klastra", divider="gray")

        cluster_series = df[cluster_col]
        unique_clusters = (
            pd.Series(cluster_series.dropna().unique())
            .sort_values()
            .tolist()
        )

        if not unique_clusters:
            st.info("Brak zdefiniowanych klastrów w kolumnie klastra.")
        else:
            option_labels = [str(v) for v in unique_clusters]
            sel_index = st.selectbox(
                "Który klaster chcesz przeanalizować?",
                options=list(range(len(option_labels))),
                index=0,
                format_func=lambda i: option_labels[i],
                key="cluster_profile_id",
            )
            sel_value = unique_clusters[sel_index]
            df_cluster = df[cluster_series == sel_value]

            st.caption(
                f"Liczność klastra: **{len(df_cluster):,}** z **{len(df):,}** wszystkich rekordów."
            )

            num_cols_profile = [c for c in _numeric_measure_candidates(df, min_non_null=3) if c != cluster_col]
            if not num_cols_profile or len(df_cluster) == 0:
                st.info("Brak kolumn liczbowych lub pusty klaster – nie można policzyć profilu.")
            else:
                global_means = df[num_cols_profile].mean()
                cluster_means = df_cluster[num_cols_profile].mean()
                diff = cluster_means - global_means

                profile_df = pd.DataFrame(
                    {
                        "feature": diff.index,
                        "diff": diff.values,
                        "mean_cluster": cluster_means.values,
                        "mean_global": global_means.values,
                    }
                )
                profile_df["abs_diff"] = profile_df["diff"].abs()
                profile_df = profile_df.sort_values("abs_diff", ascending=False)

                top_n = min(12, len(profile_df))
                top_df = profile_df.head(top_n)

                # Uporządkuj oś Y tak, aby cechy były od największej ujemnej do największej dodatniej różnicy
                order = top_df.sort_values("diff")["feature"].tolist()

                chart_profile = (
                    alt.Chart(top_df)
                    .mark_bar()
                    .encode(
                        y=alt.Y("feature:N", sort=order, title="Cecha"),
                        x=alt.X("diff:Q", title="Różnica średniej (klaster − globalnie)"),
                        color=alt.condition(
                            alt.datum.diff > 0,
                            alt.value("#28a745"),
                            alt.value("#dc3545"),
                        ),
                        tooltip=[
                            alt.Tooltip("feature:N", title="Cecha"),
                            alt.Tooltip("mean_cluster:Q", title="Średnia w klastrze", format=".3g"),
                            alt.Tooltip("mean_global:Q", title="Średnia globalnie", format=".3g"),
                            alt.Tooltip("diff:Q", title="Różnica", format=".3g"),
                        ],
                    )
                    .properties(height=320)
                )

                col_chart, col_panel = st.columns([4, 1.6])
                with col_chart:
                    altair_chart_stretch(st, chart_profile, width='stretch')

                # TOP 5 cech opisanych słownie – w panelu po prawej
                top5 = top_df.head(5)
                bullets = []
                for _, row in top5.iterrows():
                    sign = "wyższy" if row["diff"] > 0 else "niższy"
                    bullets.append(
                        f"- **{row['feature']}**: {sign} niż średnio (Î” ≈ {row['diff']:.3g})."
                    )

                with col_panel:
                    if bullets:
                        st.markdown("**Cechy najmocniej odróżniające ten klaster**")
                        # zwykły tekst zamiast caption – większa, czarna czcionka
                        st.markdown("\n".join(bullets))


        st.subheader("AI nazwy i opisy klastrów", divider="gray")

        state_key = f"cluster_ai_labels__{cluster_col}"

        # Krótkie wprowadzenie – własny HTML z mniejszym marginesem pod spodem
        st.markdown(
            """
            <p style="
                font-size: 0.875rem;
                color: var(--text-color-secondary);
                margin-bottom: 0rem;
            ">
            Najpierw podaj o kim lub o czym są te dane (liczba mnoga), 
            a potem poproś AI o zaproponowanie nazw segmentów.
            </p>
            """,
            unsafe_allow_html=True,
        )

        # Prosty CSS, żeby pole i przycisk miały zbliżoną wysokość
        st.markdown(
            """
            <style>
            div[data-testid="stTextInput"] input {
                height: 40px;
                padding-top: 6px;
                padding-bottom: 6px;
            }
            div[data-testid="baseButton-secondary"] button,
            div[data-testid="baseButton-primary"] button {
                height: 40px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Krok 1 i Krok 2 w jednej linii, obie kontrolki w ~pierwszej połowie strony
        col_k1, col_k2, _ = st.columns([1, 1, 2])

        with col_k1:
            # Krok 1 + mała ikonka info w kółku (styl jak help w Streamlit)
            st.markdown(
                '''
                <div style="font-weight:600; margin-bottom:2px; display:flex; align-items:center; gap:4px;">
                    <span>Krok 1:</span>
                    <span title="Przykłady: klienci, produkty, pokemony, miasta…"
                        style="
                            cursor: help;
                            display: inline-flex;
                            align-items: center;
                            justify-content: center;
                            width: 16px;
                            height: 16px;
                            border-radius: 50%;
                            border: 1px solid rgba(0,0,0,0.25);
                            font-size: 10px;
                            line-height: 1;
                            color: rgba(0,0,0,0.6);
                            background-color: rgba(0,0,0,0.02);
                        ">
                        i
                    </span>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            object_label = st.text_input(
                "Nazwa obiektow",
                value=st.session_state.get("cluster_object_label_input", ""),
                placeholder="Tu wpisz nazwę np. Klienci",
                key="cluster_object_label_input",
                label_visibility="collapsed",
            )

        with col_k2:
            # Taki sam ciasny label dla przycisku
            st.markdown(
                '<div style="font-weight:600; margin-bottom:2px;">Krok 2:</div>',
                unsafe_allow_html=True,
            )
            run_ai = st.button(
                "🤖 Wygeneruj nazwy klastrów",
                key="btn_ai_cluster_labels",
                width='stretch',
                type="primary",
            )
        if not run_ai and not st.session_state.get(state_key):
            _eda_record_checkpoint(
                "eda.cluster.skipped",
                cluster_col=cluster_col,
                reason="button_not_clicked",
            )

        # Wywołanie AI po kliknięciu
        if run_ai:
            with st.spinner("Analizuję klastry i generuję nazwy…"):
                try:
                    selected_label = (object_label or "obiekty").strip()
                    labels = _describe_clusters_with_llm(
                        df,
                        cluster_col,
                        object_label=selected_label,
                    )
                    if labels:
                        st.session_state[state_key] = labels
                        st.session_state["cluster_object_label"] = selected_label
                        st.session_state[f"{state_key}__debug"] = st.session_state.get("eda_cluster_debug_v1") or {}
                    else:
                        st.warning(
                            "Nie udało się wygenerować nazw klastrów "
                            "(brak odpowiedzi lub niepoprawny format JSON)."
                        )
                except Exception as e:
                    st.exception(e)

        # Odczyt aktualnych etykiet z sesji – JEDNA tabela
        labels = st.session_state.get(state_key) or {}
        if labels:
            cluster_debug = st.session_state.get(f"{state_key}__debug") or st.session_state.get("eda_cluster_debug_v1") or {}
            if cluster_debug:
                st.session_state["eda_cluster_debug_v1"] = cluster_debug
                _eda_record_checkpoint(
                    "eda.cluster.cache_hit",
                    render_key=cluster_debug.get("render_key"),
                    final_source=cluster_debug.get("final_source"),
                )
                _eda_register_exec_result(
                    "cluster_labels",
                    str(cluster_debug.get("final_source") or "fallback_deterministic_selected"),
                    model=cluster_debug.get("model"),
                    clusters=len(labels),
                )
                if cluster_debug.get("used_fallback"):
                    st.warning("AI nie spelnilo wymagan gate dla nazw klastrow; pokazuje wersje naprawiona lub deterministyczna.")
            rows = []
            for cid in sorted(labels.keys(), key=str):
                info = labels.get(cid) or {}
                rows.append(
                    {
                        "klaster": cid,
                        "nazwa": info.get("name") or "",
                        "skrót": info.get("short_label") or "",
                        "opis": info.get("description") or "",
                    }
                )

            st_df_safe(
                pd.DataFrame(rows),
                hide_index=True,
                width='stretch',
            )
            st.caption(
                "To są pomocnicze nazwy nadane przez AI – możesz je później "
                "dostosować i wykorzystać w dashboardach lub prezentacjach."
            )

    # 5) Korelacje i redundancje
    t5 = _perf('Stage2/5')
    _perf_end('Stage2/4', t4)

    st.header("5. Korelacje i redundancje")
    st.caption("Heatmapa ogólna, podgląd pary i rekomendacja eliminacji redundantnej kolumny.")

    top_left, top_right = st.columns([1, 1])
    with top_left:
        st.subheader("Mapa korelacji (numeryczne ↔ numeryczne)", divider="gray")
        if corr_chart is not None:
            altair_chart_stretch(st, corr_chart, width='stretch')
            st.caption("Każdy kwadrat to siła związku. Czerwone / ciemnoniebieskie pola = bardzo mocny związek.")
        else:
            st.info("Za mało kolumn numerycznych, żeby policzyć korelacje.")
    with top_right:
        st.subheader("Podgląd wybranej pary", divider="gray")
        if not pairs_df_full.empty:
            option_labels = [f"{r['col1']} ↔ {r['col2']} (r={r['corr']:.3f}, usuń: {r['suggest_drop']})" for _, r in pairs_df_full.iterrows()]
            picked_idx = st.selectbox(
                "Wybierz parę do podglądu:", 
                options=list(range(len(option_labels))) or [0], 
                format_func=lambda idx: option_labels[idx] if option_labels else "(brak par)"
            )

            sel_row = pairs_df_full.iloc[picked_idx]
            pair_chart = _scatter_with_trend(df, sel_row["col1"], sel_row["col2"], height=220)
            altair_chart_stretch(st, pair_chart, width='stretch')
            st.markdown(f"**Co to znaczy?**  \n• **{sel_row['col1']}** i **{sel_row['col2']}** są mocno powiązane (r={sel_row['corr']:.2f}).  \n"
                        f"• Najbezpieczniej wyłączyć: **{sel_row['suggest_drop']}** (więcej braków / wtórna).")
        else:
            st.success("Brak par z istotną korelacją — kolumny nie dublują sygnału.")

    bottom_left, bottom_right = st.columns([1, 1])

    with bottom_left:
        # JEDEN nagłówek (z dividerem) – bez duplikatu
        st.subheader("Pary o bardzo wysokiej korelacji (|r| ≥ 0.9)", divider="gray")

        # --- BEZPIECZNY WARUNEK: gdy nie ma żadnych par / brak kolumny 'corr' ---
        if pairs_df_full is None or pairs_df_full.empty or "corr" not in getattr(pairs_df_full, "columns", []):
            # spójny komunikat z tym, co widzisz w prawym panelu
            st.success("Brak par z istotną korelacją — kolumny nie dublują sygnału.")
        else:
            # --- dotychczasowa logika LEWEGO dołu (tabela par do wyłączenia) ---
            pairs_df_local = pairs_df_full.copy()
            pairs_df_local["abs_r"] = pairs_df_local["corr"].abs()
            high_corr_df = (
                pairs_df_local.loc[pairs_df_local["abs_r"] >= 0.9]
                .sort_values("abs_r", ascending=False)
            )

            ui_lang = (st.session_state.get("ui_lang", "PL") or "PL").upper()
            if ui_lang == "PL":
                col_map = {
                    "col1": "Kolumna 1",
                    "col2": "Kolumna 2",
                    "abs_r": "Wartość korelacji (|r|, Pearson)",
                    "suggest_drop": "Propozycja wyłączenia",
                }
                empty_msg = "Brak par z |r| ≥ 0.9. Super – nie duplikujemy informacji."
                why_html = """
<div style="border:1px solid #fff3cd;background:#fffef4;border-radius:0.5rem;
padding:0.75rem 1rem;font-size:0.9rem;line-height:1.4;color:#856404;margin-top:0.75rem;">
  <div style="font-weight:600;margin-bottom:0.4rem;">Dlaczego to ważne i co z tym zrobić?</div>
  • Redundancja zaciemnia interpretację i może destabilizować model.<br>
  • Zostaw jedną z mocno skorelowanych kolumn – zwykle tę z mniejszą liczbą braków lub bardziej zrozumiałą biznesowo.<br>
  • Nasza sugestia jest konserwatywna (patrzymy m.in. na braki danych).
</div>
"""
            else:
                col_map = {
                    "col1": "Column 1",
                    "col2": "Column 2",
                    "abs_r": "|r| (Pearson)",
                    "suggest_drop": "Suggested column to drop",
                }
                empty_msg = "No pairs with |r| ≥ 0.9. Great — no duplicated signal."
                why_html = """
<div style="border:1px solid #fff3cd;background:#fffef4;border-radius:0.5rem;
padding:0.75rem 1rem;font-size:0.9rem;line-height:1.4;color:#856404;margin-top:0.75rem;">
  <div style="font-weight:600;margin-bottom:0.4rem;">Why it matters & what to do</div>
  • Redundancy hides interpretation and may destabilize the model.<br>
  • Keep only one of the highly correlated columns — usually the one with fewer missings or clearer business meaning.<br>
  • Our suggestion is conservative (we also look at missingness).
</div>
"""

            if high_corr_df.empty:
                st.caption(empty_msg)
            else:
                # limit wysokości tabeli (żółty box ma być zaraz pod nią)
                PANE_LIMIT_PX = 700
                WHYBOX_EST_PX = 120
                TABLE_MAX_PX  = max(180, PANE_LIMIT_PX - WHYBOX_EST_PX)
                ROW_H, HDR_H, PAD_H = 36, 38, 10
                nrows = len(high_corr_df)
                needed_px = min(TABLE_MAX_PX, HDR_H + nrows * ROW_H + PAD_H)

                st.markdown(
                    '<div class="hc-scope hc-tight">', unsafe_allow_html=True
                )  # ⟵ usuwa szczelinę pod tabelą

                tbl = high_corr_df[["col1", "col2", "abs_r", "suggest_drop"]].rename(columns=col_map)
                st_df_safe(
                    tbl.style.format({col_map["abs_r"]: "{:.4f}"}),
                    width='stretch',
                    hide_index=True,
                    height=needed_px,
                )

            st.markdown(why_html, unsafe_allow_html=True)

    with bottom_right:
        st.subheader("Którą kolumnę wyłączyć? — wynik i uzasadnienie", divider="gray")
        if not pairs_df_full.empty:
            c1, c2 = sel_row["col1"], sel_row["col2"]
            # 1) Czynniki i łączny score z utili
            f1 = _col_factors(df, info_df, c1)
            f2 = _col_factors(df, info_df, c2)
            total1 = _elimination_score(f1)
            total2 = _elimination_score(f2)
            # spójna paleta dla obu wykresów w całej sekcji
            color_scale = alt.Scale(domain=[c1, c2], range=["#0b5ed7", "#6ea8fe"])

            # 2) OGÓŁ — dwa horyzontalne słupki (więcej miejsca na wartości)
            import pandas as _pd
            totals_df = _pd.DataFrame({"kolumna": [c1, c2], "łączny_score": [total1, total2]})
            order_cols = totals_df.sort_values("łączny_score", ascending=False)["kolumna"].tolist()
            max_x = float(totals_df["łączny_score"].max())
            x_scale = alt.Scale(domain=[0, max_x * 1.18], nice=False)  # +18% zapasu

            # --- Łączny score (paski horyzontalne) ---
            totals_df = pd.DataFrame({"kolumna": [c1, c2], "łączny_score": [total1, total2]})
            order_cols = totals_df.sort_values("łączny_score", ascending=False)["kolumna"].tolist()

            max_x = float(totals_df["łączny_score"].max())
            # 15% zapasu, żeby etykiety NIGDY się nie ucinały
            x_scale = alt.Scale(domain=[0, max_x * 1.15], nice=False)

            # więcej „powietrza” między paskami
            totals_bars = (
                alt.Chart(totals_df)
                .mark_bar(size=26, cornerRadiusEnd=5)
                .encode(
                    y=alt.Y("kolumna:N", sort=order_cols, title=None,
                            axis=alt.Axis(labelFontWeight=600)),
                    x=alt.X("łączny_score:Q", title="Łączny score (wyższy = gorszy)", scale=x_scale),
                    color=alt.Color("kolumna:N", legend=None,
                                    scale=alt.Scale(domain=[c1, c2], range=["#1f77b4", "#ff7f0e"])),
                    tooltip=[alt.Tooltip("kolumna:N"),
                            alt.Tooltip("łączny_score:Q", format=".3f")]
                )
                .properties(height=120)
            )

            # etykiety: normalna czcionka, trochę większa, leciutkie odsunięcie
            totals_labels = (
                alt.Chart(totals_df)
                .mark_text(align="left", dx=10, baseline="middle", fontSize=13, color="black")
                .encode(
                    y=alt.Y("kolumna:N", sort=order_cols, title=None),
                    x=alt.X("łączny_score:Q", scale=x_scale),
                    text=alt.Text("łączny_score:Q", format=".3f")
                )
            )

            totals_layer = alt.layer(totals_bars, totals_labels).properties(padding={"right": 46})
            altair_chart_stretch(st, totals_layer, width='stretch')

            # „Chip” Î” tuż pod wykresem, po prawej stronie
            delta = abs(total1 - total2)
            worse = totals_df.sort_values("łączny_score", ascending=False).iloc[0]["kolumna"]
            cL, cR = st.columns([1, 1.0])
            with cR:
                st.markdown(
                    f"<div style='display:inline-block;padding:.28rem .6rem;border-radius:.6rem;"
                    f"background:#e8f2ff;color:#0b57d0;font-weight:600;float:right;'>Î” = {delta:.3f} · gorsza: {worse}</div>",
                    unsafe_allow_html=True
                )
            st.write("")  # drobny odstęp pod chipem

            # 3) SZCZEGÓŁ — kontrybucje per czynnik (grupowane słupki OBOK SIEBIE)
            def factor_contribs(factors: dict[str, float], colname: str):
                keys = list(factors.keys())
                rows = []
                for k in keys:
                    only = {kk: 0.0 for kk in keys}
                    only[k] = float(factors.get(k, 0.0))
                    rows.append({"kolumna": colname, "czynnik": k, "kontrybucja": _elimination_score(only)})
                return rows

            expl = _pd.DataFrame(factor_contribs(f1, c1) + factor_contribs(f2, c2))
            # ⛔ NIE filtrujemy zer — chcemy widzieć wszystkie 7 czynników zawsze
            # keep_mask = expl.groupby("czynnik")["kontrybucja"].transform("sum") > 0  # ← usuń

            wide = expl.pivot(index="czynnik", columns="kolumna", values="kontrybucja").fillna(0.0)
            if c1 in wide.columns and c2 in wide.columns and not wide.empty:
                wide["abs_delta"] = (wide[c1] - wide[c2]).abs()
                factor_order_all = wide.sort_values("abs_delta", ascending=False).index.tolist()
                top_diff = wide["abs_delta"].idxmax()
            else:
                factor_order_all = list(f1.keys())
                top_diff = None

            # ── Kontrolki w jednej linii: lewo suwak, prawo checkbox ──
            col_left, col_right = st.columns([5, 2], vertical_alignment="center")
            with col_left:
                max_k = max(1, len(factor_order_all))
                # domyślnie pokazuj wszystkie 7 (lub ile jest)
                default_k = max_k
                top_k = st.slider("Pokaż Top-K czynników (wg różnicy) — posortowano wg |Î”|",
                                min_value=1, max_value=max_k, value=default_k)
            with col_right:
                show_pct = st.checkbox("Pokaż czynniki w %", value=False, help="Normalizacja per czynnik")

            factor_keep = factor_order_all[:top_k]

            plot_df = expl[expl["czynnik"].isin(factor_keep)].copy()
            if show_pct:
                plot_df["wartość"] = plot_df.groupby("czynnik")["kontrybucja"].transform(
                    lambda s: (s / s.sum()) if s.sum() else 0.0
                )
                y_title, y_fmt = "Udział w obrębie czynnika", ".0%"
                ref50 = alt.Chart(_pd.DataFrame({"ref": [0.5]})).mark_rule(
                    color="#6c757d", strokeDash=[6, 3], opacity=0.7
                ).encode(y=alt.Y("ref:Q", title=None))
            else:
                plot_df["wartość"] = plot_df["kontrybucja"]
                y_title, y_fmt = "Kontrybucja czynnika do łącznego score’u", ".3f"
                ref50 = None

            # --- Grouped bars: słupki obok siebie, więcej miejsca i czytelne opisy ---
            plot_df = expl[expl["czynnik"].isin(factor_keep)].copy()
            if show_pct:
                plot_df["wartość"] = plot_df.groupby("czynnik")["kontrybucja"].transform(
                    lambda s: (s / s.sum()) if s.sum() else 0.0
                )
                y_title, y_fmt = "Udział w obrębie czynnika", ".0%"
            else:
                plot_df["wartość"] = plot_df["kontrybucja"]
                y_title, y_fmt = "Kontrybucja czynnika do łącznego score’u", ".3f"

            # sam wykres – bez xOffset (Twoja wersja Altaira go nie obsługuje)
            grouped_bars = (
                alt.Chart(plot_df)
                .mark_bar(size=30)
                .encode(
                    x=alt.X(
                        "czynnik:N",
                        title=None,
                        axis=alt.Axis(labelAngle=0, labelLimit=320, labelOverlap=False),
                    ),
                    y=alt.Y("wartość:Q", title=y_title, axis=alt.Axis(format=y_fmt)),
                    color=alt.Color(
                        "kolumna:N",
                        legend=None,
                        scale=alt.Scale(domain=[c1, c2], range=["#1f77b4", "#ff7f0e"]),
                    ),
                    tooltip=[
                        alt.Tooltip("czynnik:N", title="czynnik"),
                        alt.Tooltip("kolumna:N", title="kolumna"),
                        alt.Tooltip("wartość:Q", title="wartość", format=y_fmt),
                    ],
                )
                .properties(height=360)
            )

            # etykiety na szczytach: bez bolda, większa czcionka, czarne – też bez xOffset
            grouped_labels = (
                alt.Chart(plot_df)
                .mark_text(dy=-6, fontSize=13, color="black")
                .encode(
                    x=alt.X("czynnik:N", sort=factor_keep, title=None),
                    y=alt.Y("wartość:Q"),
                    detail="kolumna:N",
                    text=alt.Text("wartość:Q", format=y_fmt),
                )
            )



            layers = [grouped_bars, grouped_labels]
            if show_pct:
                ref50 = (
                    alt.Chart(pd.DataFrame({"ref": [0.5]}))
                    .mark_rule(color="#6c757d", strokeDash=[6, 3], opacity=0.7)
                    .encode(y=alt.Y("ref:Q", title=None))
                )
                layers = [ref50] + layers

            final_chart = (
                alt.layer(*layers)
                .configure_axis(grid=True, gridOpacity=0.2, labelFontSize=12, titleFontSize=12)
                .configure_scale(bandPaddingInner=0.25, bandPaddingOuter=0.2)  # trochę „powietrza” w osi X
                .properties(padding={"left": 0, "right": 0, "top": 12, "bottom": 26})
            )
            altair_chart_stretch(st, final_chart, width='stretch')

            # 4) Werdykt
            suggest_row = totals_df.sort_values("łączny_score", ascending=False).iloc[0]
            suggest = suggest_row["kolumna"]
            delta = abs(total1 - total2)

            if delta < 1e-3:
                st.info("**Rekomendacja:** remis — oba profile są niemal identyczne. Sprawdź kontekst biznesowy lub inne pary.")
            else:
                if top_diff:
                    st.info(
                        f"**Rekomendacja:** wyłącz **`{suggest}`** — największa różnica na czynniku **{top_diff}** "
                        f"(i wyższy łączny score). **Wyłączenie uprości model** (mniej korelacji, mniejsza wariancja cech)."
                    )
                else:
                    st.info(
                        f"**Rekomendacja:** wyłącz raczej **`{suggest}`** — ma wyższy łączny score (gorszy profil czynników). "
                        f"**Wyłączenie uprości model** (mniej korelacji, mniejsza wariancja cech)."
                    )

            # if st.button(f"➕ Dodaj `{suggest}` do listy kolumn do wyłączenia (Sekcja 7)"):
            #     # ...tu Twój kod, który dopisuje kolumnę do listy...
            #     st.toast(f"Dodano `{suggest}` do czyszczenia w Sekcji 7 (możesz cofnąć w formularzu).", icon="✅")

        else:
            st.caption("Brak par do analizy.")

    # 6) Anomalie globalne
    t6 = _perf('Stage2/6')
    _perf_end('Stage2/5', t5)

    st.header("6. Anomalie globalne: duplikaty i odstające wartości")

    # --- Przygotuj treść banerów, ale na razie nic nie wyświetlaj ---
    _outliers_info_text = (
        "Czerwone punkty = wartości odstające wg IQR. Nie usuwamy ich w ciemno: "
        "dodajemy flagi `is_outlier_*`. Możesz opcjonalnie przyciąć ekstremy (winsoryzacja) w sekcji 7."
    )
    if duplicates_count > 0:
        _dup_banner_kind = "warn"
        _dup_banner_text = (
            f"Wykryto potencjalne duplikaty: {_fmt_int(duplicates_count)} "
            f"({duplicates_pct:.1f}% próbek). Domyślnie usuniemy je przy czyszczeniu."
        )
    else:
        _dup_banner_kind = "ok"
        _dup_banner_text = "Nie wykryto zduplikowanych pełnych wierszy."

    numeric_cols = _numeric_measure_candidates(df, min_non_null=3)
    if numeric_cols:
        st.subheader("Podgląd wartości odstających — wybierz kolumnę numeryczną:", divider="gray")
        col_for_anomaly = st.selectbox("Kolumna do analizy anomalii", options=numeric_cols, index=0, label_visibility="collapsed")
        temp_series = pd.to_numeric(df[col_for_anomaly], errors="coerce")
        temp_df = pd.DataFrame({"idx": np.arange(len(temp_series)), "value": temp_series}).dropna()

        if not temp_df.empty:
            q1 = temp_df["value"].quantile(0.25); q3 = temp_df["value"].quantile(0.75)
            iqr = q3 - q1
            lower_o, upper_o = (q1, q3) if iqr == 0 else (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            temp_df["is_outlier"] = ((temp_df["value"] < lower_o) | (temp_df["value"] > upper_o)).astype(int)
            med_val = float(temp_df["value"].median())

            ui_col1, ui_col2 = st.columns([2, 1])
            with ui_col1:
                max_points = st.slider(
                    "Maks. punkty w scatterze",
                    min_value=1000, max_value=20000, value=5000, step=500,
                    help="Sampling dotyczy TYLKO wizualizacji; metryki liczone są na pełnym zbiorze."
                )
            with ui_col2:
                show_tips = st.checkbox(
                    "Pokaż podpowiedzi (tooltips)", value=True,
                    help="Wyłącz, aby przyspieszyć rysowanie."
                )

            temp_df_vis = (
                temp_df.sample(n=max_points, random_state=42).sort_values("idx")
                if len(temp_df) > max_points else temp_df
            )
            eps   = max(1e-6, 0.01 * max(1.0, float(temp_df["value"].max()) - float(temp_df["value"].min())))
            y_min = float(min(0.0, temp_df["value"].min() - eps))
            y_max = float(max(0.0, temp_df["value"].max() + eps))
            CHART_HEIGHT = 280

            y_enc = alt.Y("value:Q", title=col_for_anomaly,
                        scale=alt.Scale(domain=[y_min, y_max], nice=False))

            median_y = float(temp_df["value"].median())
            median_rule = (
                alt.Chart(pd.DataFrame({"median_value": [median_y]}))
                .mark_rule(color="#1f77b4", strokeDash=[6, 3], strokeWidth=1.5, opacity=0.95)
                .encode(y=alt.Y("median_value:Q",
                                scale=alt.Scale(domain=[y_min, y_max], nice=False)))
            )

            base_scatter = (
                alt.Chart(temp_df_vis)
                .mark_point(filled=True, size=32, opacity=0.6)
                .encode(
                    x=alt.X("idx:Q", title="ID rekordu (kolejność w pliku)",
                            axis=alt.Axis(labelOverlap=True, labelLimit=1000)),
                    y=y_enc,
                    color=alt.condition(alt.datum.is_outlier == 1,
                                        alt.value("#dc3545"), alt.value("#007bff"))
                )
            )
            if show_tips:
                base_scatter = base_scatter.encode(
                    tooltip=[
                        alt.Tooltip("idx:Q",    title="rekord"),
                        alt.Tooltip("value:Q",  format=".2f", title="wartość"),
                        alt.Tooltip("is_outlier:N", title="outlier"),
                    ]
                )

            left_panel_final = alt.layer(base_scatter, median_rule).properties(width=900, height=CHART_HEIGHT)
            right_panel_final = alt.layer(
                alt.Chart(temp_df)
                .mark_boxplot(color="#1f77b4", outliers={"color": "#dc3545"})
                .encode(
                    y=alt.Y("value:Q",
                            scale=alt.Scale(domain=[y_min, y_max], nice=False),
                            axis=alt.Axis(title=None, labels=False, ticks=False, domain=False, grid=True))
                ),
                median_rule
            ).properties(width=100, height=CHART_HEIGHT)

            combined = (
                alt.hconcat(left_panel_final, right_panel_final, spacing=28)
                .resolve_scale(y="shared")
                .properties(spacing=10, bounds="flush",
                            padding={"left": 0, "right": 10, "top": 0, "bottom": 0})
            )
            altair_chart_stretch(st, combined, width='content')

    # --- Po wykresach: najpierw informacja o duplikatach, potem info o outlierach ---
    if _dup_banner_kind == "ok":
        st.success(_dup_banner_text)
    else:
        st.warning(_dup_banner_text)

    st.info(_outliers_info_text)

    # 7) Przygotowanie danych do trenowania (z tabami TL;DR)
    t7 = _perf('Stage2/7')
    _perf_end('Stage2/6', t6)

    st.header("7. Przygotowanie danych do trenowania")

    # Delikatne zbicie odstępów tylko dla tej sekcji (bez hacków na DOM)
    st.markdown("""
    <style>
    .section-7-tight { margin-top: 0.25rem !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-7-tight">', unsafe_allow_html=True)

    # ── Stan sekcji 7 ───────────────────────────────────────────────────────────
    if "sec7_revealed" not in st.session_state:
        st.session_state["sec7_revealed"] = False
    if "latest_summary_text" not in st.session_state:
        st.session_state["latest_summary_text"] = ""

    # ── CTA do odsłonięcia sekcji 7 (PRIMARY) ──────────────────────────────────
    if not st.session_state["sec7_revealed"]:
        summary_cta_label = "✨ Zrób podsumowanie (AI)" if fast_sidebar else "✨ Zrób podsumowanie"
        summary_cta_help = (
            "Wygeneruj narrację – odsłoni to całą sekcję 7."
            if fast_sidebar
            else "Pokaż deterministyczne podsumowanie bez wywołania OpenAI."
        )
        if st.button(summary_cta_label, type="primary", help=summary_cta_help):
            st.session_state["sec7_revealed"] = True
            st.session_state["play_tts_now"] = bool(st.session_state.get("tts_enabled", True))

    run_prep = False
    if st.session_state["sec7_revealed"]:
        ai_tldr_enabled = bool(fast_sidebar)
        if (
            ai_tldr_enabled
            and (st.session_state.get("eda_summary_debug_v1") or {}).get("final_source") == "deterministic_no_ai"
        ):
            st.session_state["latest_summary_text"] = ""
            st.session_state.pop("eda_summary_debug_v1", None)

        if not ai_tldr_enabled and not st.session_state.get("latest_summary_text"):
            summary_text = _make_eda_summary_text(
                source_name=source_name,
                readiness_score=readiness_score,
                duplicates_count=duplicates_count,
                global_missing_pct=global_missing_pct,
                auto_drop_candidates=auto_drop_candidates,
                prep_report=None,
            )
            st.session_state["latest_summary_text"] = summary_text
            st.session_state["eda_summary_debug_v1"] = {
                "render_key": "deterministic_no_ai",
                "model": None,
                "raw_text": None,
                "gate_ok": True,
                "gate_reasons": [],
                "used_fallback": False,
                "error": None,
                "postprocessed": False,
                "final_one_sentence": "",
                "final_source": "deterministic_no_ai",
            }

        if (
            ai_tldr_enabled
            and _get_env_or_secret("OPENAI_API_KEY")
            and (st.session_state.get("eda_summary_debug_v1") or {}).get("used_fallback")
            and not st.session_state.get("eda_ai_retry_after_secret_fix_v1")
        ):
            st.session_state["latest_summary_text"] = ""
            st.session_state.pop("eda_summary_debug_v1", None)
            st.session_state["eda_ai_retry_after_secret_fix_v1"] = True

        # ── Przygotowanie promptu do TL;DR (raz) ────────────────────────────────
        if ai_tldr_enabled and not st.session_state.get("latest_summary_text"):
            base_facts = [
                f"Liczba wierszy × kolumn: {n_rows_raw} × {n_cols_raw}",
                f"Braki danych (globalnie): ~{global_missing_pct:.1f}%",
                f"Duplikaty: {duplicates_count} ({duplicates_pct:.1f}%)",
                f"Kandydaci do wyłączenia (heurystyka): {', '.join(auto_drop_candidates) if auto_drop_candidates else 'brak'}",
                f"Pary silnie skorelowane (|r|≥0.9): {sum(1 for _ in [p for p in pairs_sorted if abs(p['corr'])>=0.9])}",
            ]
            example_pair = next((p for p in pairs_sorted if abs(p["corr"]) >= 0.9), None)
            example_txt = f"{example_pair['col1']} ↔ {example_pair['col2']} (r={example_pair['corr']:.2f})" if example_pair else "brak"
            tl_dr_prompt = (
                "Stwórz spójne podsumowanie w języku polskim, w trzech częściach, jako czysty markdown:\n"
                "### Podsumowanie danych\n- Rozmiar, poziom braków, duplikaty.\n- Wspomnij o liczbie kolumn do potencjalnego wyłączenia i mocnych korelacjach.\n"
                "### Insight z analizy\n- 2–4 kluczowe spostrzeżenia...\n"
                f"- Jeden przykład pary skorelowanej: {example_txt}.\n"
                "### Co dalej\n- Zachęta do kliknięcia „Przygotuj dane do trenowania”…\n\n"
                f"Fakty wejściowe: {base_facts}"
            )

        # ── Generowanie TL;DR tylko gdy pusto ───────────────────────────────────
        if ai_tldr_enabled and not st.session_state.get("latest_summary_text"):
            with st.spinner("⏳ Generuję podsumowanie (AI)…"):
                summary_text = ""
                trace = lf.trace(
                    name="eda_tldr",
                    user_id=st.session_state.get("wf_session_id", "anon"),
                    input=tl_dr_prompt,
                    metadata={
                        "source_name": source_name,
                        "csv_path": csv_path,
                        "n_rows": n_rows_raw,
                        "n_cols": n_cols_raw,
                        "model": openai_tldr_model,
                        "module": "02_Automat_EDA",
                    },
                ) if lf else None

                try:
                    # --- LLM: generacja podsumowania ---
                    tl = _openai_chat_completion(
                        model=openai_tldr_model,
                        messages=[{"role": "user", "content": tl_dr_prompt}],
                        temperature=0.3,
                    )
                    summary_text = (tl.choices[0].message.content or "").strip()

                    # Fallback, gdyby LLM zwrócił pusty tekst
                    if not summary_text:
                        summary_text = _make_eda_summary_text(
                            source_name=source_name,
                            readiness_score=readiness_score,
                            duplicates_count=duplicates_count,
                            global_missing_pct=global_missing_pct,
                            auto_drop_candidates=auto_drop_candidates,
                            prep_report=None,           # <-- ważne: żadnego prep_report tutaj
                        )

                    st.session_state["latest_summary_text"] = summary_text

                    # Audio odpalamy dopiero po finalnym gate/repair niżej,
                    # żeby lektor czytał dokładnie ten tekst, który widzi użytkownik.

                    if trace:
                        trace.update(status="success", output=summary_text[:2000])

                except Exception as e:
                    if trace:
                        trace.update(status="error", metadata={"error": str(e)})

                    st.warning(
                        "Nie udało się wygenerować podsumowania przez AI – używam prostego fallbacku."
                    )
                    summary_text = _make_eda_summary_text(
                        source_name=source_name,
                        readiness_score=readiness_score,
                        duplicates_count=duplicates_count,
                        global_missing_pct=global_missing_pct,
                        auto_drop_candidates=auto_drop_candidates,
                        prep_report=None,       # <-- znowu: None, nieprep_report
                    )
                    st.session_state["latest_summary_text"] = summary_text
        else:
            summary_text = st.session_state["latest_summary_text"]

        current_summary_debug = st.session_state.get("eda_summary_debug_v1") or {}
        if ai_tldr_enabled and not current_summary_debug:
            trace = lf.trace(
                name="eda_tldr_gate",
                user_id=st.session_state.get("wf_session_id", "anon"),
                input="stage2_tldr_gate",
                metadata={
                    "source_name": source_name,
                    "csv_path": csv_path,
                    "n_rows": n_rows_raw,
                    "n_cols": n_cols_raw,
                    "model": openai_tldr_model,
                    "module": "02_Automat_EDA",
                    "mode": "post_gate_overlay",
                },
            ) if lf else None

            summary_text, current_summary_debug = _eda_generate_tldr_markdown(
                source_name=source_name,
                readiness_score=readiness_score,
                duplicates_count=duplicates_count,
                duplicates_pct=duplicates_pct,
                global_missing_pct=global_missing_pct,
                auto_drop_candidates=auto_drop_candidates,
                pairs_sorted=pairs_sorted,
                n_rows_raw=n_rows_raw,
                n_cols_raw=n_cols_raw,
                model=openai_tldr_model,
                trace=trace,
            )
            st.session_state["latest_summary_text"] = summary_text
        elif not current_summary_debug:
            current_summary_debug = {
                "render_key": "deterministic_no_ai",
                "model": None,
                "raw_text": None,
                "gate_ok": True,
                "gate_reasons": [],
                "used_fallback": False,
                "error": None,
                "postprocessed": False,
                "final_one_sentence": "",
                "final_source": "deterministic_no_ai",
            }
            st.session_state["eda_summary_debug_v1"] = current_summary_debug
        else:
            _eda_record_checkpoint(
                "eda.summary.cache_hit",
                render_key=current_summary_debug.get("render_key"),
                final_source=current_summary_debug.get("final_source"),
            )
            _eda_register_exec_result(
                "summary_tldr",
                str(current_summary_debug.get("final_source") or "fallback_deterministic_selected"),
                model=current_summary_debug.get("model"),
            )

        if ai_tldr_enabled and current_summary_debug.get("used_fallback"):
            st.warning("AI nie spelnilo wymagan gate dla TL;DR; pokazuje wersje deterministyczna.")

        # ── TL;DR: podgląd i edycja ─────────────────────────────────────────────
        st.subheader("Podsumowanie (AI)" if ai_tldr_enabled else "Podsumowanie", divider="gray")
        tab_view, tab_edit = st.tabs(["📄 Podsumowanie", "✍️ Edytuj tekst i zapisz"])

        with tab_view:
            st.markdown(st.session_state.get("latest_summary_text", ""))

        with tab_edit:
            left, right = st.columns([1, 1], gap="large")
            with left:
                edited = st.text_area(
                    "Tekst narracji (markdown)",
                    value=st.session_state.get("latest_summary_text", ""),
                    key="summary_text_area",
                    height=380,
                    help="Po kliknięciu „Zapisz…” tekst trafi do summary_ai.md i natychmiast zaktualizuje Podsumowanie.",
                )
                if st.button("💾 Zapisz podsumowanie do artefaktów", type="primary"):
                    st.session_state["latest_summary_text"] = edited
                    st.session_state.pop("tts_last_hash", None)
                    st.session_state.pop("latest_tts_audio_hash", None)
                    st.session_state.pop("latest_tts_audio_bytes", None)
                    st.session_state.pop("latest_tts_audio_mime", None)
                    st.session_state["eda_summary_debug_v1"] = {
                        "render_key": "manual_override",
                        "model": None,
                        "raw_text": None,
                        "gate_ok": True,
                        "gate_reasons": [],
                        "used_fallback": False,
                        "error": None,
                        "postprocessed": False,
                        "final_one_sentence": "",
                        "final_source": "manual_override",
                    }
                    _eda_record_checkpoint("eda.summary.manual_override", final_source="manual_override")
                    saved_path = _save_summary_to_artifacts(edited, latest_info)
                    st.toast(f"Zapisano: {saved_path}", icon="✅")
                    st.rerun()

            with right:
                st.caption("Podgląd (markdown):")
                st.markdown(
                    st.session_state.get(
                        "summary_text_area",
                        st.session_state.get("latest_summary_text", ""),
                    )
                )
                st.caption("Podgląd aktualizuje się na bieżąco podczas pisania.")

        # --- TTS automatycznie po kliknięciu podsumowania, bez drugiego przycisku ---
        _text_for_tts = _plain_text_for_tts(
            st.session_state.get("latest_summary_text", "") or "",
            max_chars=0,
        )
        tts_enabled   = st.session_state.get("tts_enabled", True)

        cur_tts_hash = _hash_key(
            _text_for_tts,
            provider or "",
            openai_tts_model_selected or "",
            openai_voice_selected or "",
        )
        explicit_trigger = bool(st.session_state.get("play_tts_now", False))
        cached_audio = st.session_state.get("latest_tts_audio_bytes")
        cached_audio_hash = st.session_state.get("latest_tts_audio_hash")
        audio_mime = st.session_state.get("latest_tts_audio_mime", "audio/mpeg")
        autoplay_audio = False
        # Stage 2 uses browser speech synthesis now. Do not render old MP3 players
        # left in session state from earlier runs, because they can overlap audio.
        cached_audio = None
        cached_audio_hash = None

        if explicit_trigger and ai_tldr_enabled and tts_enabled and _text_for_tts:
            voice_hint = (
                "male"
                if str(openai_voice_selected or "").lower() in set(OPENAI_VOICES.get("male", []))
                else "female"
            )
            _render_browser_tts(_text_for_tts, voice_hint=voice_hint, autoplay=True)
            st.session_state["play_tts_now"] = False
            explicit_trigger = False
            cached_audio = None
            cached_audio_hash = None

        if not (ai_tldr_enabled and tts_enabled and _text_for_tts):
            st.session_state["play_tts_now"] = False
        else:
            should_generate_tts = explicit_trigger and cached_audio_hash != cur_tts_hash

            if should_generate_tts:
                with st.spinner("🔊 Generuję narrację audio…"):
                    _run_tts_for_summary(
                        _text_for_tts,
                        provider,
                        openai_tts_model_selected,
                        openai_voice_selected,
                        cur_tts_hash,
                    )
                cached_audio = st.session_state.get("latest_tts_audio_bytes")
                cached_audio_hash = st.session_state.get("latest_tts_audio_hash")
                audio_mime = st.session_state.get("latest_tts_audio_mime", "audio/mpeg")
                autoplay_audio = bool(cached_audio and cached_audio_hash == cur_tts_hash)
            elif explicit_trigger and cached_audio and cached_audio_hash == cur_tts_hash:
                autoplay_audio = True
                st.session_state["play_tts_now"] = False

            if cached_audio and cached_audio_hash == cur_tts_hash:
                _render_tts_audio_player(
                    cached_audio,
                    mime=audio_mime,
                    autoplay=autoplay_audio,
                )


        # ── Opis akcji czyszczenia ──────────────────────────────────────────────
        st.subheader("Ten krok zbuduje gotowy zbiór treningowy:", divider="gray")
        st.markdown(
            "🧹 usuniemy zbędne kolumny  \n"
            "📅 przekonwertujemy liczby i daty  \n"
            "🧱 uzupełnimy braki  \n"
            "🏷️ dodamy flagi `is_outlier_*` / winsoryzacja (opcjonalnie)  \n"
            "🧹 usuniemy duplikaty  \n"
            "💾 zapiszemy `ready_for_training.parquet` + `prep_report.json`  \n"
            "⚙️ zaktualizujemy artefakty"
        )
        
        with st.form("eda_cleaning_form", border=False):
            with st.container(border=True):
                st.write("### Ustawienia szybkiego czyszczenia")
                remove_duplicates_user = st.checkbox("Usuń zduplikowane rekordy (zalecane)", value=True)
                basic_col_drop_default = sorted(
                    set(auto_drop_candidates) | set(st.session_state.get("sec7_preselected_drop", set()))
                )
    
                # --- STATE: winsor verification ---
                if "winsor_verify_ready" not in st.session_state:
                    st.session_state.winsor_verify_ready = False
                if "winsor_verify_params" not in st.session_state:
                    st.session_state.winsor_verify_params = {}
    
                # 🔓 Czy expander ma startować rozwinięty?
                if "sec7_adv_expanded" not in st.session_state:
                    st.session_state.sec7_adv_expanded = False  # na wejściu do sekcji jest zamknięty
    
                st.write("### Zaawansowane ustawienia (opcjonalnie)")
                with st.expander(
                    "Pokaż / ukryj ustawienia zaawansowane",
                    expanded=st.session_state.get("sec7_adv_expanded", False),
                ):
    
                    drop_cols_user = st.multiselect(
                        "Kolumny do usunięcia:",
                        options=list(df.columns),
                        default=basic_col_drop_default,
                    )
                    numeric_cols_for_winsor = _numeric_measure_candidates(df, min_non_null=3)
                    winsorize_cols_user = st.multiselect(
                        "Kolumny do przycięcia (winsoryzacja):",
                        options=numeric_cols_for_winsor,
                        default=[],
                        key="adv_wins_cols",           # <<— TEN KEY jest ważny
                        help="Podaj kolumny, które chcesz przyciąć (winsoryzacja).",
                    )
    
                    # ✅ JEDYNY przycisk weryfikacji winsoryzacji – widoczny od razu
                    go = st.form_submit_button(
                        "🔎 Uruchom / odśwież weryfikację winsoryzacji",
                        type="secondary",                # szary (bezpieczny); chcesz „blade czerwone”? patrz sekcja 3 (opcjonalny CSS)
                        width='content',
                        key="wins_verify_btn",
                        help="Policz metryki i pokaż wizualizacje ‘przed vs po’ dla wybranej kolumny."
                    )
                    # jeśli użytkownik kliknął przycisk – od tej pory expander ma być zawsze rozwinięty
                    if go:
                        st.session_state.sec7_adv_expanded = True
    
                    # === WERYFIKACJA WINSORYZACJI — KONTYNUACJA SEKCJI „Ustawienia szybkiego czyszczenia” ===
                    DF = locals().get("df_clean", locals().get("df", None))
                    if DF is None:
                        st.warning("Nie znaleziono ramki danych DF (df / df_clean). Upewnij się, że zmienna z danymi istnieje.")
                    else:
                        # pierwszy wybrany element z multiselecta „adv_wins_cols”
                        # próbujemy kilku możliwych nazw key, żeby nie polegać na jednej konkretnej
                        _possible_keys = [
                            "adv_wins_cols",          # moja propozycja
                            "wins_cols_to_clip",      # częsty wariant
                            "wins_cols",              # inny wariant
                            "winsorize_cols",         # inny wariant
                            "wins_cols_preview"       # bywa i tak
                        ]
                        _selected_list = []
                        for _k in _possible_keys:
                            if _k in st.session_state and st.session_state[_k]:
                                _selected_list = st.session_state[_k]
                                break
    
                        _selected = None
                        if isinstance(_selected_list, (list, tuple)) and len(_selected_list) > 0:
                            _selected = _selected_list[0]
                        elif isinstance(_selected_list, str) and _selected_list:
                            _selected = _selected_list
    
                        # ——— Helpers (jedna, zwięzła wersja) ———
                        def _to_num(s: pd.Series) -> pd.Series:
                            return pd.to_numeric(s, errors="coerce")
    
                        def compute_iqr_fences(s: pd.Series, k: float = 1.5):
                            s = _to_num(s).dropna()
                            if s.empty:
                                return np.nan, np.nan, s
                            q1, q3 = s.quantile(0.25), s.quantile(0.75)
                            iqr = q3 - q1
                            return float(q1 - k * iqr), float(q3 + k * iqr), s
    
                        def winsorize_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
                            return _to_num(s).clip(lower=lower, upper=upper)
    
                        def mini_stats(s: pd.Series) -> dict:
                            s = _to_num(s).dropna()
                            if s.empty:
                                return {"liczebność (n)": 0, "min": np.nan, "Q1": np.nan, "mediana": np.nan,
                                        "Q3": np.nan, "max": np.nan, "średnia": np.nan, "odch.std.": np.nan, "IQR": np.nan}
                            d = s.describe(percentiles=[.25, .5, .75])
                            return {
                                "liczebność (n)": int(s.size),
                                "min": float(d["min"]),
                                "Q1": float(d["25%"]),
                                "mediana": float(d["50%"]),
                                "Q3": float(d["75%"]),
                                "max": float(d["max"]),
                                "średnia": float(d["mean"]),
                                "odch.std.": float(d["std"]),
                                "IQR": float(d["75%"] - d["25%"]),
                            }
    
                        # ——— UI: POKAZUJEMY dopiero, gdy wybrano kolumnę ———
                        if _selected:
                            # Sterowanie bez żadnych KPI nad nim
                            c1, c2 = st.columns([1, 1])
                            with c1:
                                viz_mode = st.radio(
                                    "Tryb wizualizacji",
                                    ["Histogram", "ECDF"],
                                    index=0,
                                    horizontal=True,
                                    key="wins_viz_mode_qc",
                                    help="ECDF = Empiryczna Dystrybuanta Skumulowana. Oś Y w [0,1]; dobrze pokazuje ‘ściśnięcie’ ogonów."
                                )
                            with c2:
                                k_iqr = st.slider(
                                    "K (Tukey/IQR)", 1.0, 3.0,
                                    float(st.session_state.get("wins_k_iqr_qc", 1.5)),
                                    0.1, key="wins_k_iqr_qc",
                                    help="K=1.5 (klasyczny Tukey). Większe K = łagodniejsze cięcie. Użyj większego K przy ciężkoogonowych rozkładach."
                                )
    
                            # Cała logika i rysunki – TYLKO po kliknięciu (nie ma "luźnych" KPI nad sterowaniem)
                            if go:
                                col = _selected
    
                                # 1) brak kolumny -> tylko komunikat
                                if not col:
                                    st.warning("Najpierw wybierz kolumnę do winsoryzacji.")
                                else:
                                    # 2) inicjalizacja cache'a, jeśli jeszcze go nie ma
                                    if "winsor_cache" not in st.session_state:
                                        st.session_state["winsor_cache"] = {}
    
                                    cache = st.session_state["winsor_cache"]
    
                                    # 3) klucz cache: (nazwa kolumny, K zaokrąglone do 3 miejsc)
                                    k_key = round(float(k_iqr), 3)
                                    cache_key = (col, k_key)
    
                                    # 4) jeśli mamy w cache – używamy, inaczej liczymy i zapisujemy
                                    if cache_key in cache:
                                        data = cache[cache_key]
                                        lower = data["lower"]
                                        upper = data["upper"]
                                        before = data["before"]
                                        after = data["after"]
                                    else:
                                        lower, upper, before = compute_iqr_fences(DF[col], k=k_iqr)
                                        after = winsorize_series(DF[col], lower, upper)
    
                                        cache[cache_key] = {
                                            "lower": lower,
                                            "upper": upper,
                                            "before": before,
                                            "after":  after,
                                        }
    
                                    # 5) unikalna baza kluczy – zależy od kolumny i K (Tukey/IQR)
                                    chart_key_base = f"wins_{col}_{float(k_iqr):.2f}"
    
                                    # 6) liczebność = liczba NIE-NaN (spójnie z mini_stats / tabelą)
                                    before_num = before                    # 'before' ma już dropna()
                                    n_before = int(before_num.size)
    
                                    after_num = _to_num(after).dropna()
                                    n_after  = int(after_num.size)
    
                                    # 7) outliery PRZED – poniżej L lub powyżej U
                                    below = int((before_num < lower).sum())
                                    above = int((before_num > upper).sum())
                                    out_before = below + above
    
                                    # 8) outliery PO – teoretycznie powinny być ≈ 0, ale liczmy „na wszelki wypadek”
                                    out_after = int(((after_num < lower) | (after_num > upper)).sum())
    
                                    pct_below = 100.0 * below / n_before if n_before else 0.0
                                    pct_above = 100.0 * above / n_before if n_before else 0.0
                                    pct_total = pct_below + pct_above
    
                                    # 9) KPI w „kartach” (cyferki na górze)
                                    kcol = st.columns(6)
                                    with kcol[0]:
                                        st.markdown(
                                            f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px'>"
                                            f"<div style='font-size:12px;color:#666'>n przed &rarr; po</div>"
                                            f"<div style='font-size:18px;font-weight:600'>{n_before:,} &rarr; {n_after:,}</div>"
                                            f"<div style='font-size:11px;color:#888'>liczba obserwacji</div></div>",
                                            unsafe_allow_html=True,
                                        )
                                    with kcol[1]:
                                        st.markdown(
                                            f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px'>"
                                            f"<div style='font-size:12px;color:#666'>n odstające przed &rarr; po</div>"
                                            f"<div style='font-size:18px;font-weight:600'>{out_before:,} &rarr; {out_after:,}</div>"
                                            f"<div style='font-size:11px;color:#888'>liczba wartości odstających ogółem</div></div>",
                                            unsafe_allow_html=True,
                                        )
                                    with kcol[2]:
                                        st.markdown(
                                            f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px'>"
                                            f"<div style='font-size:12px;color:#666'>% &lt; lower</div>"
                                            f"<div style='font-size:18px;font-weight:600'>{pct_below:.2f}%</div>"
                                            f"<div style='font-size:11px;color:#888'>odsetek lewy ogon</div></div>",
                                            unsafe_allow_html=True,
                                        )
                                    with kcol[3]:
                                        st.markdown(
                                            f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px'>"
                                            f"<div style='font-size:12px;color:#666'>% &gt; upper</div>"
                                            f"<div style='font-size:18px;font-weight:600'>{pct_above:.2f}%</div>"
                                            f"<div style='font-size:11px;color:#888'>odsetek prawy ogon</div></div>",
                                            unsafe_allow_html=True,
                                        )
                                    with kcol[4]:
                                        st.markdown(
                                            f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px'>"
                                            f"<div style='font-size:12px;color:#666'>% łącznie</div>"
                                            f"<div style='font-size:18px;font-weight:600'>{pct_total:.2f}%</div>"
                                            f"<div style='font-size:11px;color:#888'>lewy+prawy</div></div>",
                                            unsafe_allow_html=True,
                                        )
                                    with kcol[5]:
                                        st.markdown(
                                            f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px' "
                                            f"title='L = Q1 - K·IQR, U = Q3 + K·IQR; IQR = Q3 - Q1; K ustawiasz suwakiem „K (Tukey/IQR)”.'>"
                                            f"<div style='font-size:12px;color:#666'>Progi [L, U]</div>"
                                            f"<div style='font-size:18px;font-weight:600'>[{lower:.2f}, {upper:.2f}]</div>"
                                            f"<div style='font-size:11px;color:#888'>granice winsoryzacji</div></div>",
                                            unsafe_allow_html=True,
                                        )
    
                                    # mały odstęp pod KPI
                                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                                    
                                # =========================
                                #  HISTOGRAM: słupki OBOK SIEBIE (Altair v5)
                                # =========================
                                base_vals = before.dropna()
    
                                # 🔍 Pusta / degenerowana kolumna:
                                # - same NaN po konwersji
                                # - albo jedna unikalna wartość (brak wariancji)
                                if base_vals.empty or base_vals.nunique() <= 1:
                                    st.info(
                                        "Kolumna nie ma wariancji / brak danych liczbowych "
                                        "(same NaN lub jedna unikalna wartość) – nie można narysować wykresów."
                                    )
                                else:
                                    vmin, vmax = float(base_vals.min()), float(base_vals.max())
    
                                    # JEDEN DataFrame z kolumną 'zestaw' — to KLUCZOWE dla xOffset
                                    plot_df = pd.concat(
                                        [
                                            pd.DataFrame({"value": before, "zestaw": "przed"}),
                                            pd.DataFrame({"value": after,  "zestaw": "po"}),
                                        ],
                                        ignore_index=True
                                    ).dropna(subset=["value"])
    
                                    dom = ["przed", "po"]
                                    rng = ["#1f77b4", "#ff7f0e"]  # stałe kolory
    
                                # =========================
                                #  HISTOGRAM / ECDF
                                # =========================
                                base_vals = before.dropna()
                                if base_vals.empty:
                                    st.info("Brak danych liczbowych po konwersji (NaN) – nie można narysować wykresów.")
                                else:
                                    # 1) jeden DF „przed/po”
                                    plot_df = pd.concat(
                                        [
                                            pd.DataFrame({"value": before, "zestaw": "przed"}),
                                            pd.DataFrame({"value": after,  "zestaw": "po"}),
                                        ],
                                        ignore_index=True,
                                    ).dropna(subset=["value"])
    
                                    vals = pd.to_numeric(plot_df["value"], errors="coerce").dropna()
                                    if vals.empty:
                                        st.info("Brak danych liczbowych po konwersji (NaN) – nie można narysować wykresów.")
                                    else:
                                        vmin, vmax = float(vals.min()), float(vals.max())
                                        unique_vals = np.sort(vals.unique())
    
                                        dom = ["przed", "po"]
                                        rng = ["#1f77b4", "#ff7f0e"]
    
                                        # ---------- A) HISTOGRAM (Altair – jak w _numeric_by_category_charts) ----------
                                        if viz_mode == "Histogram":
                                            # inteligentne binowanie – dokładnie ta sama logika co w 4C
                                            bin_param = alt.Bin(maxbins=40, extent=[vmin, vmax])
                                            if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
                                                bin_param = alt.Bin(extent=[-0.5, 1.5], step=1)
                                            elif (
                                                len(unique_vals) <= 12
                                                and np.all(np.isfinite(unique_vals))
                                                and np.all(np.mod(unique_vals, 1) == 0)
                                            ):
                                                loi, hii = float(unique_vals.min()), float(unique_vals.max())
                                                bin_param = alt.Bin(extent=[loi - 0.5, hii + 0.5], step=1)
    
                                            base_hist = alt.Chart(plot_df)
    
                                            # wspólne biny + środek i szerokość binu
                                            binned = (
                                                base_hist
                                                .transform_bin(
                                                    as_=["bin_start", "bin_end"],
                                                    field="value",
                                                    bin=bin_param,
                                                )
                                                .transform_calculate(
                                                    bin_mid="(datum.bin_start + datum.bin_end) / 2",
                                                    bin_w="(datum.bin_end - datum.bin_start)",
                                                )
                                            )
    
                                            # rozstawiamy słupki kategorii wewnątrz binu – jak w _numeric_by_category_charts
                                            _cat_domain_json = json.dumps(dom)   # ["przed", "po"]
                                            _rank_expr = f"indexof({_cat_domain_json}, datum['zestaw'])"
                                            _k_total = len(dom)
    
                                            binned = (
                                                binned
                                                .transform_calculate(
                                                    cat_rank=_rank_expr,
                                                    k_total=str(_k_total),
                                                    band_fraction="0.85",
                                                    cat_w="(toNumber(datum.band_fraction) * datum.bin_w) / toNumber(datum.k_total)",
                                                    x_mid="datum.bin_mid + ( (datum.cat_rank - (toNumber(datum.k_total)-1)/2 ) * datum.cat_w )",
                                                    x_left="datum.x_mid - datum.cat_w/2",
                                                    x_right="datum.x_mid + datum.cat_w/2",
                                                )
                                            )
    
                                            hist = (
                                                binned
                                                .mark_bar(opacity=0.85)
                                                .encode(
                                                    x=alt.X("x_left:Q", title=col),
                                                    x2=alt.X2("x_right:Q"),
                                                    y=alt.Y("count():Q", title="Liczebność", stack=None),
                                                    color=alt.Color(
                                                        "zestaw:N",
                                                        sort=dom,
                                                        scale=alt.Scale(domain=dom, range=rng),
                                                        title="Zbiór",
                                                    ),
                                                    order=alt.Order("cat_rank:Q", sort="ascending"),
                                                    tooltip=[
                                                        alt.Tooltip("zestaw:N", title="Zbiór"),
                                                        alt.Tooltip("count():Q", title="liczebność"),
                                                        alt.Tooltip("bin_start:Q", title="bin od", format=".2f"),
                                                        alt.Tooltip("bin_end:Q",   title="bin do", format=".2f"),
                                                    ],
                                                )
                                                .properties(height=280)
                                            )
    
                                            # cieniowanie poza [L, U]
                                            shade_df = pd.DataFrame(
                                                {"x": [vmin, upper], "x2": [lower, vmax], "side": ["left", "right"]}
                                            )
                                            shade = (
                                                alt.Chart(shade_df)
                                                .mark_rect(opacity=0.10)
                                                .encode(x="x:Q", x2="x2:Q")
                                            )
    
                                            # linie progów [L, U]
                                            rules = (
                                                alt.Chart(pd.DataFrame({"b": [lower, upper]}))
                                                .mark_rule(strokeDash=[6, 3], opacity=0.9, color="#dc3545")
                                                .encode(x="b:Q")
                                            )
    
                                            chart_layer = alt.layer(shade, hist, rules).resolve_scale(y="independent")
    
                                            altair_chart_stretch(st, 
                                                chart_layer,
                                                width='stretch',
                                                key=f"{chart_key_base}_hist",
                                            )
    
                                        # ---------- B) ECDF (lekki, zostaje jak był) ----------
                                        else:
                                            ecdf_df = plot_df.sort_values("value").assign(
                                                rank=lambda d: d.groupby("zestaw").cumcount() + 1
                                            )
                                            sizes = (
                                                ecdf_df.groupby("zestaw")["rank"]
                                                .transform("max")
                                                .replace(0, np.nan)
                                            )
                                            ecdf_df["frac"] = ecdf_df["rank"] / sizes
    
                                            ecdf = (
                                                alt.Chart(ecdf_df)
                                                .mark_line()
                                                .encode(
                                                    x=alt.X(
                                                        "value:Q",
                                                        title=col,
                                                        scale=alt.Scale(domain=[vmin, vmax]),
                                                    ),
                                                    y=alt.Y(
                                                        "frac:Q",
                                                        title="ECDF",
                                                        scale=alt.Scale(domain=[0, 1]),
                                                    ),
                                                    color=alt.Color(
                                                        "zestaw:N",
                                                        sort=dom,
                                                        scale=alt.Scale(domain=dom, range=rng),
                                                        title="Zbiór",
                                                    ),
                                                    tooltip=[
                                                        alt.Tooltip("zestaw:N", title="Zbiór"),
                                                        alt.Tooltip("value:Q", title="Wartość", format=".2f"),
                                                        alt.Tooltip("frac:Q",  title="Frakcja", format=".2f"),
                                                    ],
                                                )
                                                .properties(height=280)
                                            )
                                            rules = (
                                                alt.Chart(pd.DataFrame({"b": [lower, upper]}))
                                                .mark_rule(strokeDash=[6, 3], opacity=0.9, color="#dc3545")
                                                .encode(x="b:Q")
                                            )
    
                                            altair_chart_stretch(st, 
                                                alt.layer(ecdf, rules),
                                                width='stretch',
                                                key=f"{chart_key_base}_ecdf",
                                            )
    
                                    # 🔹 Boxplot (wąsy 1.5·IQR zgodnie z Tukey)
                                    bL, bR = st.columns([1.3, 1.0])
                                    with bL:
                                        bp = (
                                            alt.Chart(plot_df)
                                            .mark_boxplot(extent=1.5, outliers=True)  # ← tu zmiana: 1.5·IQR (zamiast "min-max")
                                            .encode(
                                                y=alt.Y("zestaw:N", title=None, sort=dom),
                                                x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[vmin, vmax])),
                                                color=alt.Color("zestaw:N", sort=dom, scale=alt.Scale(domain=dom, range=rng), legend=None),
                                            )
                                            .properties(height=150)
                                        )
                                        # Linie median ‘przed’ i ‘po’
                                        med_df = plot_df.groupby("zestaw")["value"].median().reset_index()
                                        med_rule = alt.Chart(med_df).mark_rule(strokeDash=[4,4], color="#6c757d").encode(
                                            x="value:Q", detail="zestaw:N"
                                        )
    
                                        # Warstwa „kropek-outlierów” w kolorze czerwonym (liczone wg progów [L,U])
                                        out_df = plot_df[(plot_df["value"] < lower) | (plot_df["value"] > upper)]
                                        out_layer = (
                                            alt.Chart(out_df)
                                            .mark_point(size=30, opacity=0.9, color="#dc3545")
                                            .encode(
                                                y=alt.Y("zestaw:N", sort=dom, title=None),
                                                x=alt.X("value:Q")
                                            )
                                        )
    
                                        altair_chart_stretch(st, alt.layer(bp, med_rule, out_layer), width='stretch', key=f"{chart_key_base}_box",)
    
    
                                    with bR:
                                        before_stats = mini_stats(before)
                                        after_stats  = mini_stats(after)
                                        rows = ["liczebność (n)", "min", "Q1", "mediana", "Q3", "max", "średnia", "odch.std.", "IQR"]
                                        tbl = []
                                        for r in rows:
                                            b = before_stats.get(r, np.nan)
                                            a = after_stats.get(r, np.nan)
                                            d  = (a - b) if (isinstance(a,(int,float)) and isinstance(b,(int,float))) else np.nan
                                            dp = (d / b * 100.0) if (isinstance(b,(int,float)) and b not in (0, np.nan)) else np.nan
                                            tbl.append([r,
                                                        f"{b:.4g}" if isinstance(b,(int,float)) else b,
                                                        f"{a:.4g}" if isinstance(a,(int,float)) else a,
                                                        f"{d:+.4g}" if isinstance(d,(int,float)) else "",
                                                        f"{dp:+.2f}%" if isinstance(dp,(int,float)) else ""])
                                        stats_df = pd.DataFrame(tbl, columns=["metryka","przed","po","Î”","Î”%"])
                                        st_df_safe(stats_df, width='stretch', hide_index=True)
    
    
                                    with bL:
                                        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
                                        st.caption(
                                            "ℹ️ Liczebność liczona jest po konwersji do typów numerycznych i odrzuceniu NaN. "
                                            "Winsoryzacja ścina ogony wg progów [L,U] (Tukey/IQR). \n\n"
                                            f"▶️ Jeśli % łącznie ({pct_total:.2f}%) jest większe niż ok. 5% i mediana w tabeli po prawej "
                                            "nie przesunęła się istotnie, zwykle możesz zaakceptować winsoryzację. "
                                            "W przeciwnym razie rozważ większe K lub rezygnację z winsoryzacji dla tej kolumny."
                                        )
    
            st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
            run_prep = st.form_submit_button("⚙️ Przygotuj dane do trenowania (automatycznie)", type="primary")

    # ── Submit handler ──────────────────────────────────────────────────────────
    if run_prep and st.session_state.get("sec7_revealed", False):
        if "drop_cols_user" not in locals():      drop_cols_user = basic_col_drop_default
        if "winsorize_cols_user" not in locals(): winsorize_cols_user = []
        decisions = {
            "drop_cols": drop_cols_user,
            "winsorize_cols": winsorize_cols_user,
            "remove_duplicates": remove_duplicates_user
        }

        # Micro-profiler (tabela timingów) — reset na każde uruchomienie
        st.session_state[STAGE2_PREP_TIMINGS_KEY] = []

        with st.spinner("🛠️ Przygotowuję dane do trenowania…"):
            # ⬇⬇⬇ KLUCZOWA ZMIANA: używamy PEŁNEGO zbioru df_full, nie próbki df ⬇⬇⬇
            with stage2_prepare_step("auto_prepare_for_training"):
                df_ready, prep_report = _auto_prepare_for_training(df_full, info_df, decisions)

            with stage2_prepare_step("persist_artifacts"):
                ready_path, report_path = _persist_artifacts(df_ready, prep_report, latest_info)

            with stage2_prepare_step("estimate_savings"):
                hours_saved = _estimate_hours_saved(
                    n_rows_raw,
                    n_cols_raw,
                    high_null_cols,
                    duplicates_count,
                    auto_drop_candidates,
                    pairs_sorted,
                )
                cost_saved  = _estimate_cost_saved_pln(hours_saved)

            with stage2_prepare_step("datachat_handoff"):
                # ➜ Handoff do etapu Data Chat (pakiet startowy)
                handoff_path = _save_datachat_handoff(
                    latest_info=latest_info,
                    summary_text=st.session_state.get("latest_summary_text", ""),
                    pairs_sorted=pairs_sorted,
                    prep_report_path=report_path,
                    info_df=info_df,
                )
                st.toast(f"Pakiet Data Chat zapisany: {handoff_path}", icon="✅")

        st.markdown(_success_hero_box(hours_saved, cost_saved), unsafe_allow_html=True)

        with st.expander("⏱️ Timingi: przygotowanie danych (Stage2)", expanded=False):
            timings_df = stage2_prepare_timing_df()
            if timings_df.empty:
                st.caption("Brak danych timingowych (uruchom przygotowanie ponownie).")
            else:
                st_df_safe(timings_df, width='stretch', hide_index=True)

        outlier_flags_count = len(prep_report["outlier_flags"])
        duplicates_removed  = prep_report["duplicates_removed"]
        n_cols_final        = prep_report["n_cols_final"]

        st.markdown(
            _before_after_cards_html(n_cols_raw, n_cols_final, global_missing_pct,
                                    duplicates_count, duplicates_removed, outlier_flags_count),
            unsafe_allow_html=True,
        )
        st.subheader("Co dokładnie zrobiliśmy:", divider="gray")
        dropped_cols_txt  = ", ".join(prep_report["dropped_columns"])     if prep_report["dropped_columns"]     else "(nic)"
        new_time_cols_txt = ", ".join(prep_report["new_time_cols_added"]) if prep_report["new_time_cols_added"] else "(brak)"
        outlier_flags_txt = ", ".join(prep_report["outlier_flags"])        if prep_report["outlier_flags"]        else "(brak)"
        imputations_lines = [f"✅ {col}: {how}" for col, how in prep_report["imputations"].items()]
        winsor_lines = [f"✅ {col}: [{b['lower']:.2f}, {b['upper']:.2f}]" for col, b in prep_report["winsorized"].items()]
        st.markdown(
            f"**Pliki wynikowe:**  \n✅ `{ready_path}`  \n✅ `{report_path}`  \n\n"
            f"**Zmiany w danych:**  \n✅ Usunięte kolumny: {dropped_cols_txt}  \n"
            f"✅ Dodane cechy z dat: {new_time_cols_txt}  \n✅ Flagi outlierów: {outlier_flags_txt}  \n"
            f"✅ Usuniętych duplikatów: {prep_report['duplicates_removed']}  \n\n"
            f"**Uzupełnianie braków:**  \n" + ("\n".join(imputations_lines) if imputations_lines else "✅ (brak braków)") + "\n\n"
            f"**Winsoryzacja:**  \n" + ("\n".join(winsor_lines) if winsor_lines else "✅ (brak kolumn do przycięcia)")
        )
        with st.expander("Raport techniczny przygotowania (pełne szczegóły)"):
            st.json(prep_report)

    st.markdown('</div>', unsafe_allow_html=True)
    # --- Powtórzony potok na dole strony ---
    render_flow_nav(current_id="02_Automat_EDA", key_prefix="flow_bottom")
    st.markdown("---")

    # --- dla pewności żeby wszystkie zdarzenia trafiły do Langfuse ---
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()  # blokująco czeka aż kolejka wyśle wszystko
        except Exception:
            pass

# Streamlit entrypoint
if __name__ == "__main__":
    main()
