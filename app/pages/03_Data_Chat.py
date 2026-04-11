# app/pages/03_Data_Chat.py
from __future__ import annotations

from pathlib import Path
import sys

# Ensure .../app is on PYTHONPATH (Streamlit pages often run with pages/ on sys.path)
APP_DIR = Path(__file__).resolve().parents[1]  # .../app
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from typing import Any, Dict, Optional
import os
import copy

# --- load .env (local dev) ---
try:
    from dotenv import load_dotenv  # type: ignore
    # próbuj załadować .env z katalogu projektu (jeden poziom wyżej niż /app/pages)
    _here = Path(__file__).resolve()
    for _p in [_here.parent.parent.parent, _here.parent.parent, _here.parent]:
        _env = _p / '.env'
        if _env.exists():
            load_dotenv(_env)
            break
except Exception:
    # brak python-dotenv albo brak .env — OK (np. produkcja / secrets)
    pass

import io
import contextlib
import time
import datetime
import wave
import hashlib
import base64
import json
import html
import re
import math
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

def _df_from_parquet(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load Parquet. If max_rows is set, read only first N rows (fast preview)."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        if not max_rows:
            return pf.read().to_pandas()
        out = []
        remaining = int(max_rows)
        for rg in range(pf.num_row_groups):
            if remaining <= 0:
                break
            tbl = pf.read_row_group(rg)
            df_rg = tbl.to_pandas()
            if len(df_rg) > remaining:
                df_rg = df_rg.iloc[:remaining].copy()
            out.append(df_rg)
            remaining -= len(df_rg)
        if not out:
            return pf.read_row_group(0).to_pandas().head(max_rows)
        return pd.concat(out, ignore_index=True)
    except Exception:
        # fallback: pandas may still read full file
        df = pd.read_parquet(path)
        return df.head(max_rows) if max_rows else df

import streamlit as st
import streamlit.components.v1 as components
try:
    from audio_recorder_streamlit import audio_recorder
except Exception:
    audio_recorder = None

# Optional: Stage 3 disk-based fallback (survives server restart).
try:
    from core.config import load_config, resolve_artifacts_dir  # type: ignore
except Exception:  # pragma: no cover
    load_config = None  # type: ignore
    resolve_artifacts_dir = None  # type: ignore


def _is_ready_for_training_parquet(p: str | Path) -> bool:
    try:
        pp = Path(p)
        name = pp.name.lower()
        if name != "ready_for_training.parquet":
            return False
        # hard rule: never load masked/full dataset in Data Chat
        if "full_masked" in str(pp).lower():
            return False
        return pp.exists()
    except Exception:
        return False

def _safe_mtime(p: str | Path) -> float:
    try:
        return Path(p).stat().st_mtime
    except Exception:
        return 0.0

def _read_latest_handoff_json(pointer_path: Path) -> dict | None:
    try:
        if pointer_path.exists():
            with open(pointer_path, "r", encoding="utf-8") as f:
                v = json.load(f)
            return v if isinstance(v, dict) else None
    except Exception:
        return None
    return None


def _load_df_from_parquet_cached(path: str | Path, max_rows: int | None = None) -> pd.DataFrame | None:
    try:
        pp = Path(path)
        if not pp.exists():
            return None
        cache = st.session_state.get("datachat_df_cache_v2")
        if not isinstance(cache, dict):
            cache = {}
        cache_key = (
            str(pp),
            float(_safe_mtime(pp)),
            None if max_rows is None else int(max_rows),
        )
        cached = cache.get(cache_key)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
        loaded = _df_from_parquet(pp, max_rows=max_rows)
        if isinstance(loaded, pd.DataFrame) and not loaded.empty:
            cache[cache_key] = loaded
            if len(cache) > 4:
                for _old_key in list(cache.keys())[:-4]:
                    cache.pop(_old_key, None)
            st.session_state["datachat_df_cache_v2"] = cache
            if max_rows is None:
                st.session_state["df_ready_for_training"] = loaded
        return loaded
    except Exception:
        return None

def _discover_latest_stage2_ready_parquet(ingest_root: str | Path | None = None) -> Path | None:
    """Fallback C: scan ingest for newest */ready_for_training.parquet.

    Works even if session_state is empty or user navigates back and forth between stages.
    """
    try:
        root = Path(ingest_root) if ingest_root else None
        if root is None or not root.exists():
            return None
        # Only direct subfolders (runs) to keep it fast
        best: Path | None = None
        best_m = 0.0
        for d in root.iterdir():
            if not d.is_dir():
                continue
            cand = d / "ready_for_training.parquet"
            if not cand.exists():
                continue
            mtime = _safe_mtime(cand)
            if mtime > best_m:
                best_m = mtime
                best = cand
        return best
    except Exception:
        return None

def _resolve_stage3_parquet_path() -> Dict[str, Any]:
    """Pick the freshest ready_for_training.parquet from:
    - datachat_handoff (session_state)
    - latest_artifacts (session_state)
    - ingest/_latest_handoff.json (disk pointer)
    - fallback scan ingest
    """
    candidates: list[tuple[str, str]] = []

    # 1) datachat_handoff
    handoff = st.session_state.get("datachat_handoff") or {}
    if isinstance(handoff, dict):
        p = handoff.get("ready_parquet_path")
        if isinstance(p, str) and _is_ready_for_training_parquet(p):
            candidates.append(("datachat_handoff", p))

    # 2) latest_artifacts
    latest = st.session_state.get("latest_artifacts") or {}
    if isinstance(latest, dict):
        for k in ("stage2_ready_parquet_path", "ready_parquet_path"):
            p = latest.get(k)
            if isinstance(p, str) and _is_ready_for_training_parquet(p):
                candidates.append((f"latest_artifacts:{k}", p))

    # 3) disk pointer: ingest/_latest_handoff.json (best effort)
    ingest_root = None
    if isinstance(latest, dict):
        ingest_root = latest.get("ingest_root") if isinstance(latest.get("ingest_root"), str) else None

    # infer ingest_root from any candidate
    if ingest_root is None and candidates:
        try:
            ingest_root = str(Path(candidates[0][1]).parent.parent)
        except Exception:
            ingest_root = None

    pointer_path = Path(ingest_root) / "_latest_handoff.json" if ingest_root else None
    if pointer_path is not None:
        v = _read_latest_handoff_json(pointer_path)
        if isinstance(v, dict):
            p = v.get("ready_parquet_path")
            if isinstance(p, str) and _is_ready_for_training_parquet(p):
                candidates.append(("latest_handoff_json", p))

    # choose freshest by mtime
    best_src = None
    best_p = None
    best_m = 0.0
    for src, p in candidates:
        m = _safe_mtime(p)
        if m > best_m:
            best_m = m
            best_src = src
            best_p = p

    # 4) fallback scan ingest
    if best_p is None and ingest_root:
        disk = _discover_latest_stage2_ready_parquet(ingest_root=ingest_root)
        if disk is not None:
            best_src = "fallback_scan_ingest"
            best_p = str(disk)
            best_m = _safe_mtime(disk)

    return {"source": best_src, "path": best_p, "mtime": best_m, "ingest_root": ingest_root}


try:
    from openai import OpenAI  # ✅ wymagane do: OpenAI(api_key=...)
except Exception:
    OpenAI = None  # type: ignore

# ─────────────────────────────────────────────
# Executive Takeaway — LLM wiring (GLOBAL, MUST)
# ─────────────────────────────────────────────

def _dc_register_exec_takeaway_llm_callable() -> None:
    """
    Ustawia globalnie:
    - st.session_state["dc_llm_text"] jako callable (prompt:str)->str
    - st.session_state["dc_llm_status_v1"] z prawdą o konfiguracji (debug MUST)

    HARD RULE: nie może być sytuacji, że dc_llm_status_v1 jest {} po runie.
    """
    ss = st.session_state

    # status zawsze jako dict + 2 pola MUST: configured + reason
    status = ss.get("dc_llm_status_v1")
    if not isinstance(status, dict):
        status = {}
    status.setdefault("provider", "openai")
    status.setdefault("model", os.getenv("DATA_CHAT_OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
    status["configured"] = False
    status["reason"] = "init"

    # OpenAI import check
    if OpenAI is None:
        ss.pop("dc_llm_text", None)
        status["configured"] = False
        status["reason"] = "openai_import_error"
        ss["dc_llm_status_v1"] = status
        return

    # API key (env or secrets)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            api_key = None

    if not api_key:
        ss.pop("dc_llm_text", None)
        status["configured"] = False
        status["reason"] = "missing_api_key"
        ss["dc_llm_status_v1"] = status
        return

    model = status.get("model") or "gpt-4o-mini"

    # create client once per run
    client = OpenAI(api_key=api_key)

    def _llm_text(prompt: str) -> str:
        # PL-only hard requirement enforced here
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Zwracaj WYŁĄCZNIE treść po polsku. Bez angielskiego. Bez markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        out = (resp.choices[0].message.content or "")
        return out

    ss["dc_llm_text"] = _llm_text
    status["configured"] = True
    status["reason"] = "configured_ok"
    ss["dc_llm_status_v1"] = status

# ─────────────────────────────────────────────
# DC — LLM wiring for Executive Takeaway (global)
# Contract: st.session_state["dc_llm_text"] is callable(prompt:str)->str
# ─────────────────────────────────────────────

def _dc_register_dc_llm_text_callable() -> None:
    """
    Minimal, isolated wiring ONLY for Executive Takeaway engine.
    Does not change other branches.

    Writes:
      - st.session_state["dc_llm_text"] : callable(prompt)->str
      - st.session_state["dc_llm_status_v1"] : non-empty dict (configured/model/provider)
    """
    ss = st.session_state
    status = ss.get("dc_llm_status_v1")
    if not isinstance(status, dict):
        status = {}
        ss["dc_llm_status_v1"] = status

    api_key = os.getenv("OPENAI_API_KEY", "")
    if (not api_key) or (OpenAI is None):
        # leave explicit status for debug panel
        status.update({
            "configured": False,
            "provider": "openai_chat_completions",
            "model": None,
            "reason": "missing_api_key_or_openai_lib",
        })
        ss.pop("dc_llm_text", None)
        return

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # Create client once per run
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        status.update({
            "configured": False,
            "provider": "openai_chat_completions",
            "model": chat_model,
            "reason": f"client_init_failed: {type(e).__name__}: {e}",
        })
        ss.pop("dc_llm_text", None)
        return

    sys_prompt = (
        "Jesteś asystentem analitycznym. Odpowiadasz WYŁĄCZNIE po polsku. "
        "Nie używasz markdown. Nie używasz wielokropków. Nie dopisujesz komentarzy."
    )

    def _llm_text(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=chat_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": str(prompt or "")},
            ],
        )
        return (resp.choices[0].message.content or "")

    ss["dc_llm_text"] = _llm_text
    status.update({
        "configured": True,
        "provider": "openai_chat_completions",
        "model": chat_model,
        "reason": None,
    })

from data_chat_core.cp0_router import (
    cp0_detect_has_time,
    cp0_time_intent_from_question,
    cp0_time_mode,
    cp0_branch_from_intent_and_time,
    cp0_compute_checkpoint0_analysis_struct,
)

from data_chat_branches import (
    distribution,
    composition_static,
    composition_over_time,
)

from core.top_nav import (
    hide_default_multipage_nav,
    render_flow_nav,
    render_sidebar_links,
)

from core.safe_frontend import compute_safe_frontend_config

# NOTE: Ten plik jest routerem. Logika wykresów jest w modułach gałęzi.

def datachat_answer_key(q_txt: str, intent_txt: str, f: Dict[str, Any] | None) -> str:
    """
    Jedno źródło prawdy dla identyfikacji odpowiedzi (rating per odpowiedź).
    Zwraca stabilny string-key (hash).
    """
    payload = {
        "q": (q_txt or "").strip(),
        "intent": (intent_txt or "").strip(),
        "filters": f or {},
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

# ─────────────────────────────────────────────
# DC — Interpretacja (LLM + Quality Gate) — MUST
# ─────────────────────────────────────────────

_DC_INTERP_SYSTEM_PROMPT = """
Jesteś Lead Data Analyst (McKinsey/Bain). Piszesz interpretację wykresu rozkładu (histogram + KDE + IQR fences).
Zasady twarde:
- Język: wyłącznie polski. Ton: rzeczowy, executive, bez marketingu.
- Zero halucynacji: wolno używać WYŁĄCZNIE danych z przekazanego stats_payload.
- Jeśli nie masz podstaw w payload → napisz wprost ograniczenie (bez zgadywania).
- "Odpowiedź w jednym zdaniu" MUSI zawierać ≥2 kotwice liczbowe ze stats_payload.
- Zakazy: "większość", "znaczna część", "przeważnie" bez podania %; "symetryczny" gdy |skewness| nie jest bliskie 0;
  wnioski o outlierach muszą być spójne z outlier_rate_iqr i progami fences.
- Zwróć wynik jako JSON zgodny ze schematem.
""".strip()

# ⬅️ Zmiana tej wersji unieważnia render-once cache interpretacji
_DC_INTERP_VERSION = "v2"

def _dc_interp_user_prompt(stats_payload: Dict[str, Any], question: str) -> str:
    # Uwaga: to jest jedyny kontekst liczbowy. Model ma pisać wyłącznie o tym.
    payload_str = json.dumps(stats_payload, ensure_ascii=False, sort_keys=True, default=str)
    q = (question or "").strip()
    return f"""
Masz stats_payload (jedyna prawda):
{payload_str}

Pytanie użytkownika:
{q}

Zwróć JSON o strukturze:
{{
  "one_sentence": "…",                 # >= 2 kotwice liczbowe z payload
  "what": ["…","…","…"],               # 2–4 punkty
  "insights": ["…","…","…"],           # 3–5 punktów, tylko to co wynika z payload
  "reco": ["…","…","…"],               # 2–4 punkty
  "limits": ["…","…"]                  # 2–4 punktów, jawne ograniczenia
}}

Dodatkowe zasady jakości:
- Jeżeli wspominasz o outlierach: MUSISZ podać outlier_rate_iqr w % i próg fence (dolny/górny), spójne z payload.
- Jeżeli wspominasz o skośności/symetrii: MUSISZ odnieść się do skewness i nie wolno pisać "symetryczny" jeśli |skewness| > 0.2.
- W polu "one_sentence" NIE używaj skewness; lead ma opierać się wyłącznie na medianie, IQR, percentylach, outlier_rate_iqr i/lub fences.
- Skewness wolno używać tylko w polu "insights".
- Nie używaj słowa "większość" bez liczby % (np. "ok. 60%").
- Nie wymyślaj przyczyn biznesowych ("premium", "błędy") jeśli payload tego nie wspiera; zamiast tego dodaj do limits.
""".strip()

def _dc_is_out_of_scope_for_distribution(question: str) -> bool:
    q = (question or "").lower()

    # pytania, które nie mają prawa być odpowiadane z samego histogramu/KDE/IQR
    out_kw = [
        "braki", "brak danych", "missing", "null", "nan",
        "kategorie", "top", "najbardziej", "napędzają", "dlaczego",
        "trend", "w czasie", "sezon", "miesiąc", "rok",
        "korel", "zależ", "porówn", "segment", "profil",
        "duplik", "anomali",  # sanity/quality poza distribution
    ]
    return any(k in q for k in out_kw)

def _dc_build_stats_payload_distribution(
    s: pd.Series,
    dist_col: str,
    filters: Dict[str, Any],
) -> Dict[str, Any]:
    """Buduje stats_payload wyłącznie z danych po filtrach (spójne z histogram/KDE/IQR)."""
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    n = int(s_num.shape[0])

    if n <= 0:
        return {
            "dist_col": dist_col,
            "n": 0,
            "filters": filters or {},
        }

    q1 = float(s_num.quantile(0.25))
    q3 = float(s_num.quantile(0.75))
    iqr = float(q3 - q1)
    lo_fence = float(q1 - 1.5 * iqr)
    hi_fence = float(q3 + 1.5 * iqr)

    out_mask = (s_num < lo_fence) | (s_num > hi_fence)
    outlier_rate = float(out_mask.mean())

    # skewness: pandas definition (Fisher-Pearson); defensywnie
    try:
        skew = float(s_num.skew())
        if math.isnan(skew) or math.isinf(skew):
            skew = None
    except Exception:
        skew = None

    payload = {
        "dist_col": dist_col,
        "n": n,
        "min": float(s_num.min()),
        "max": float(s_num.max()),
        "mean": float(s_num.mean()),
        "median": float(s_num.median()),
        "std": float(s_num.std(ddof=1)) if n >= 2 else None,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "fences": {"lo": lo_fence, "hi": hi_fence},
        "outlier_rate_iqr": outlier_rate,   # 0..1
        "skewness": skew,
        "p90": float(s_num.quantile(0.90)),
        "p95": float(s_num.quantile(0.95)),
        "filters": filters or {},
    }
    return payload

# ─────────────────────────────────────────────────────────────────────────────
# Composition Static — interpretacja (system prompt, stats, gate, fallback, LLM)
# ─────────────────────────────────────────────────────────────────────────────

_DC_CS_SYSTEM_PROMPT = """
Jesteś Lead Data Analyst (McKinsey/Bain). Piszesz interpretację wykresu struktury (treemap + Pareto).
Zasady twarde:
- Język: wyłącznie polski. Ton: rzeczowy, executive, bez marketingu.
- Zero halucynacji: wolno używać WYŁĄCZNIE danych z przekazanego stats_payload.
- Jeśli nie masz podstawy w payload → napisz wprost ograniczenie (bez zgadywania).
- "Odpowiedź w jednym zdaniu" MUSI zawierać ≥2 kotwice liczbowe ze stats_payload (np. udział top-1 %, HHI, liczba grupów).
- Jeśli używasz sformułowań typu "lider", "dominuje", "największy udział", MUSISZ wskazać dokładnie top_labels[0].
- Nie wolno wskazać innej grupy jako największej, jeśli top_labels[0] istnieje.
- Zakazy: "większość" bez podania %; wnioski o koncentracji muszą być spójne z hhi i top_shares_pct.
- Opisuj strukturę (kto dominuje, jak rozproszona jest wartość). NIE opisuj rozkładu (histogram/KDE).
- Zwróć wynik jako JSON zgodny ze schematem.
""".strip()


def _dc_cs_user_prompt(stats_payload: Dict[str, Any], question: str) -> str:
    payload_str = json.dumps(stats_payload, ensure_ascii=False, sort_keys=True, default=str)
    q = (question or "").strip()
    top_labels = list(stats_payload.get("top_labels") or [])
    top_values = list(stats_payload.get("top_values") or [])
    top_shares = list(stats_payload.get("top_shares_pct") or [])
    top1_label = str(top_labels[0]) if top_labels else ""
    top1_value = top_values[0] if top_values else None
    top1_share = top_shares[0] if top_shares else None
    top3_share = round(sum(float(x or 0.0) for x in top_shares[:3]), 2) if top_shares else None
    anchor_summary = {
        "top1_label": top1_label,
        "top1_value": top1_value,
        "top1_share_pct": top1_share,
        "top3_share_pct": top3_share,
        "pareto_80_n": stats_payload.get("pareto_80_n"),
        "hhi": stats_payload.get("hhi"),
        "hhi_class": stats_payload.get("hhi_class"),
        "price_corridor_p80": ((stats_payload.get("price_corridor") or {}).get("p80_price")),
        "price_corridor_share_pct": ((stats_payload.get("price_corridor") or {}).get("corridor_share_pct")),
    }
    return f"""
Masz stats_payload (jedyna prawda):
{payload_str}

Pytanie użytkownika:
{q}

Najważniejsze kotwice zarządcze:
{json.dumps(anchor_summary, ensure_ascii=False, sort_keys=True, default=str)}

Zwróć JSON o strukturze:
{{
  "one_sentence": "…",                 # >= 2 kotwice liczbowe z payload
  "what": ["…","…"],                   # 2–3 punkty: co pokazuje wykres
  "insights": ["…","…","…"],           # 3–4 punkty, tylko to co wynika z payload
  "reco": ["…","…"],                   # 2–3 punkty
  "limits": ["…","…"]                  # 2–3 punkty, jawne ograniczenia
}}

Zasady:
- Jeżeli wspominasz koncentrację: MUSISZ podać HHI lub top-N share % spójne z payload.
- Jeśli używasz pojęcia lider/dominacja/top-1, MUSI to być dokładnie grupa top1_label.
- Priorytet dla executive summary: skala, koncentracja, implikacja zarządcza; bez opisów ogólnikowych.
- Nie wymyślaj przyczyn biznesowych jeśli payload tego nie wspiera — dodaj do limits.
""".strip()


def _dc_build_stats_payload_composition_legacy(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group_col2: str | None = None,
    top_n: int = 10,
    price_col: str | None = None,
) -> Dict[str, Any]:
    """Stats payload dla CS: grupy, udziały, HHI, Pareto-80, top-N."""
    payload: Dict[str, Any] = {
        "group_col": group_col,
        "value_col": value_col,
        "group_col2": group_col2,
        "top_n": top_n,
        "total_rows": int(len(df)),
    }

    if not group_col or group_col not in df.columns or not value_col or value_col not in df.columns:
        payload["error"] = "Brak kolumny grupowej lub wartościowej"
        return payload

    try:
        agg = (
            df[[group_col, value_col]]
            .dropna(subset=[group_col])
            .groupby(group_col, dropna=False)[value_col]
            .sum()
            .reset_index()
            .rename(columns={group_col: "group", value_col: "value"})
        )
        agg["value"] = pd.to_numeric(agg["value"], errors="coerce").fillna(0.0).clip(lower=0.0)
        agg = agg[agg["value"] > 0].sort_values("value", ascending=False).reset_index(drop=True)

        total_value = float(agg["value"].sum())
        n_groups   = int(len(agg))

        if total_value <= 0 or n_groups == 0:
            payload["error"] = "Brak wartości po agregacji"
            return payload

        agg["share"] = agg["value"] / total_value

        # top-N
        agg_topn    = agg.head(top_n)
        top_labels  = [str(g) for g in agg_topn["group"]]
        top_shares  = [round(float(s) * 100, 2) for s in agg_topn["share"]]

        # Pareto 80
        cum = agg["share"].cumsum()
        pareto_80_n = int((cum >= 0.80).idxmax()) + 1 if (cum >= 0.80).any() else n_groups

        # HHI (skala 0-10000, standard DOJ)
        hhi = float((agg["share"] ** 2).sum() * 10000)

        payload.update({
            "n_groups":        n_groups,
            "total_value":     round(total_value, 2),
            "top_labels":      top_labels,
            "top_shares_pct":  top_shares,
            "top1_share_pct":  round(top_shares[0], 2) if top_shares else 0.0,
            "pareto_80_n":     pareto_80_n,
            "hhi":             round(hhi, 1),
            "hhi_class":       "rozproszona" if hhi < 1500 else ("umiarkowana" if hhi < 2500 else "skoncentrowana"),
        })

        # 2-level: ile unikalnych subcategories
        if group_col2 and group_col2 in df.columns and group_col2 != group_col:
            try:
                n_sub = int(
                    df[[group_col, group_col2]]
                    .dropna(subset=[group_col, group_col2])
                    .drop_duplicates()
                    .shape[0]
                )
                payload["n_subcategories"] = n_sub
            except Exception:
                pass

    except Exception as e:
        payload["error"] = f"Stats error: {e}"


    # Price corridor (optional)
    if price_col and price_col in df.columns:
        try:
            s = pd.to_numeric(df[price_col], errors="coerce").dropna()
            if not s.empty:
                # quantile corridor 20-80 by value_col-weighted revenue (if available)
                tmp = df[[price_col, value_col]].copy()
                tmp[price_col] = pd.to_numeric(tmp[price_col], errors="coerce")
                tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                tmp = tmp.dropna(subset=[price_col])
                tmp = tmp[tmp[value_col] > 0]

                # FIXED (equal-width) bins with automatic 'nice' step + robust clipping for outliers
                s = tmp[price_col].dropna()
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

                    target_bins = 20

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
                    step = _nice_step(rng / target_bins)

                    def _n_bins(_step: float) -> int:
                        return int(math.ceil((use_hi - use_lo) / _step))

                    # keep readable #bins
                    while _n_bins(step) > 30:
                        step = _nice_step(step * 1.25)

                    start = math.floor(use_lo / step) * step
                    end = math.ceil(use_hi / step) * step
                    if end <= start:
                        end = start + step

                    edges = np.arange(start, end + step * 0.999999, step)
                    tmp["price_bin"] = pd.cut(tmp[price_col], bins=edges, include_lowest=True).astype(str)

                    corr = (
                        tmp.groupby("price_bin", dropna=False)[value_col]
                        .sum()
                        .reset_index()
                        .rename(columns={value_col: "value"})
                    )
                    corr = corr[corr["value"] > 0].copy()

                    # sort by numeric bin start
                    def _bin_start(lbl: str) -> float:
                        try:
                            m = re.match(r"[\[\(]([^,]+),", str(lbl))
                            return float(m.group(1)) if m else float("inf")
                        except Exception:
                            return float("inf")

                    corr["_start"] = corr["price_bin"].map(_bin_start)
                    corr = corr.sort_values("_start").drop(columns=["_start"]).reset_index(drop=True)

                    total = float(corr["value"].sum() or 1.0)
                    corr["share"] = corr["value"] / total
                    corr["cum_pct"] = (corr["share"].cumsum() * 100.0).clip(0, 100)

                    # 20–80 corridor & P80
                    lo_idx = int((corr["cum_pct"] >= 20.0).idxmax()) if (corr["cum_pct"] >= 20.0).any() else int(corr.index.min())
                    hi_idx = int((corr["cum_pct"] >= 80.0).idxmax()) if (corr["cum_pct"] >= 80.0).any() else int(corr.index.max())
                    corridor_share_pct = float(corr.loc[lo_idx:hi_idx, "share"].sum() * 100.0)

                    def _bin_hi(lbl: str) -> Optional[float]:
                        try:
                            m = re.match(r"[\[\(]([^,]+),\s*([^\]\)]+)[\]\)]", str(lbl))
                            return float(m.group(2)) if m else None
                        except Exception:
                            return None

                    p80_price = _bin_hi(corr.loc[hi_idx, "price_bin"])

                    payload["price_corridor"] = {
                        "bin_method": "fixed",
                        "bin_step": step,
                        "bin_count": int(corr["price_bin"].nunique()),
                        "clip": clip,
                        "p80_price": p80_price,
                        "corridor_share_pct": corridor_share_pct,
                        "bins": corr[["price_bin", "value", "share", "cum_pct"]].to_dict(orient="records"),
                    }
        except Exception:
            pass

    return payload


def _dc_build_stats_payload_composition(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group_col2: str | None = None,
    top_n: int = 10,
    price_col: str | None = None,
) -> Dict[str, Any]:
    """Delegates CS stats building to the branch module so UI and interpretation stay aligned."""
    try:
        return composition_static.build_stats_payload(
            df=df,
            group_col=group_col,
            value_col=value_col,
            group_col2=group_col2,
            top_n=top_n,
            price_col=price_col,
        )
    except Exception as e:
        return {
            "group_col": group_col,
            "value_col": value_col,
            "group_col2": group_col2,
            "top_n": top_n,
            "total_rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
            "error": "Nie udało się przygotować statystyk struktury.",
            "debug_error": f"{type(e).__name__}: {e}",
        }


def _dc_text_has_number(text: str) -> bool:
    return bool(re.search(r"[-+]?\d+(?:[.,]\d+)?", str(text or "")))


def _dc_cs_has_specific_anchor(text: str, stats_payload: Dict[str, Any]) -> bool:
    low = str(text or "").lower()
    top_labels = stats_payload.get("top_labels") or []
    top1_label = str(top_labels[0]).lower() if top_labels else ""
    if top1_label and top1_label in low:
        return True
    if any(token in low for token in ["hhi", "pareto", "top-", "p80", "korytarz", "koncentrac", "udzial"]):
        return True
    if _dc_text_has_number(text) and any(
        token in low for token in ["pln", "warto", "grup", "kategor", "segment", "cena", "przedzial", "udzial", "udział"]
    ):
        return True
    return False


def _dc_cs_is_generic_text(text: str) -> bool:
    low = str(text or "").strip().lower()
    generic_markers = [
        "zaleca sie dalsza analize",
        "warto rozwa",
        "monitorowanie zmian",
        "potencjal wzrostu",
        "moze pomoc",
        "zroznicowana pomiedzy roznymi kategoriami",
        "wartosci sa zorganizowane",
        "pokazuje dominacje kilku kategorii",
        "pokazuje koncentracje wartosci",
        "nie uwzgledniono danych o konkurencji",
        "brak szczegolowych danych",
        "czynnikow zewnetrznych",
    ]
    return any(marker in low for marker in generic_markers)


def _dc_cs_normalize_dimension_terms(text: str, stats_payload: Dict[str, Any]) -> str:
    out = str(text or "").strip()
    if not out:
        return out
    group_col = str(stats_payload.get("group_col") or "").strip().lower()
    if group_col in {"category", "kategoria"}:
        return out
    replacements = [
        (r"\bkategorie\b", "grupy"),
        (r"\bkategoria\b", "grupa"),
        (r"\bkategorii\b", "grup"),
    ]
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def _dc_cs_top_label(stats_payload: Dict[str, Any], idx: int = 0) -> str:
    labels = list(stats_payload.get("top_labels") or [])
    if idx < 0 or idx >= len(labels):
        return ""
    return str(labels[idx] or "").strip()


def _dc_cs_top_value(stats_payload: Dict[str, Any], idx: int = 0) -> float:
    values = list(stats_payload.get("top_values") or [])
    if idx < 0 or idx >= len(values):
        return 0.0
    try:
        return float(values[idx] or 0.0)
    except Exception:
        return 0.0


def _dc_cs_top_share(stats_payload: Dict[str, Any], idx: int = 0) -> float:
    shares = list(stats_payload.get("top_shares_pct") or [])
    if idx < 0 or idx >= len(shares):
        return 0.0
    try:
        return float(shares[idx] or 0.0)
    except Exception:
        return 0.0


def _dc_cs_top_share_sum(stats_payload: Dict[str, Any], n: int) -> float:
    shares = list(stats_payload.get("top_shares_pct") or [])
    try:
        return round(sum(float(x or 0.0) for x in shares[: max(0, int(n))]), 2)
    except Exception:
        return 0.0


def _dc_cs_fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{int(digits)}f}%".replace(".", ",")
    except Exception:
        return "—"


def _dc_cs_fmt_pp(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{int(digits)}f} pp".replace(".", ",")
    except Exception:
        return "—"


def _dc_cs_fmt_pln(value: Any) -> str:
    try:
        return f"{float(value):,.0f} PLN".replace(",", " ")
    except Exception:
        return "—"


def _dc_cs_mentions_wrong_leader(text: str, stats_payload: Dict[str, Any]) -> bool:
    low = str(text or "").lower()
    if not low:
        return False
    dominant_markers = [
        "dominuje", "lider", "najwieksz", "największ", "top-1", "najwyzszy udzial", "najwyższy udział",
    ]
    if not any(marker in low for marker in dominant_markers):
        return False
    top_labels = [str(x or "").strip().lower() for x in (stats_payload.get("top_labels") or [])[:5] if str(x or "").strip()]
    if not top_labels:
        return False
    top1 = top_labels[0]
    mentioned = [lbl for lbl in top_labels if lbl in low]
    return bool(mentioned) and top1 not in mentioned


def _dc_cs_specific_limits(stats_payload: Dict[str, Any]) -> list[str]:
    group_col = str(stats_payload.get("group_col") or "kategoria")
    group_col2 = str(stats_payload.get("group_col2") or "").strip()
    limits = [
        f"Analiza dotyczy tylko biezacego przekroju po grupowaniu '{group_col}' i aktualnych filtrow; ten widok nie pokazuje zmian w czasie.",
        "Wnioski opisuja strukture, koncentracje i korytarz cenowy na podstawie danych z wykresu; nie wyjasniaja przyczyn biznesowych ani wplywu czynnikow zewnetrznych.",
    ]
    if group_col2 and group_col2.lower() not in {"(brak)", "none"}:
        limits.append(
            f"Drugi poziom '{group_col2}' sluzy do podzialu struktury, ale ten widok nie pokazuje pelnych rankingow osobno dla kazdej wartosci tego wymiaru."
        )
    return limits[:3]


def _dc_pick_cs_list(
    value: Any,
    fallback: list[str],
    stats_payload: Dict[str, Any],
    *,
    min_items: int,
    require_anchor: bool,
) -> list[str]:
    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            text = _dc_cs_normalize_dimension_terms(text, stats_payload)
            if _dc_cs_is_generic_text(text):
                continue
            if _dc_cs_mentions_wrong_leader(text, stats_payload):
                continue
            if require_anchor and not _dc_cs_has_specific_anchor(text, stats_payload):
                continue
            if text not in out:
                out.append(text)
    if len(out) < min_items:
        for item in fallback:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
    return out[: max(min_items, min(len(out), 4))]


def _dc_postprocess_cs_interpretation(interp: Dict[str, Any], stats_payload: Dict[str, Any]) -> Dict[str, Any]:
    base = _dc_cs_fallback_deterministic(stats_payload)
    one = str(base.get("one_sentence") or "").strip()
    what = _dc_pick_cs_list(interp.get("what"), list(base.get("what") or []), stats_payload, min_items=2, require_anchor=False)
    insights = _dc_pick_cs_list(interp.get("insights"), list(base.get("insights") or []), stats_payload, min_items=3, require_anchor=True)
    reco = _dc_pick_cs_list(interp.get("reco"), list(base.get("reco") or []), stats_payload, min_items=2, require_anchor=True)

    return {
        "one_sentence": one,
        "what": what[:3],
        "insights": insights[:4],
        "reco": reco[:3],
        "limits": _dc_cs_specific_limits(stats_payload),
    }


def _dc_quality_gate_cs(interp: Dict[str, Any], stats_payload: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Gate dla CS: konkret liczbowy, brak generycznych limits i sekcje oparte na payload."""
    reasons: list[str] = []

    for k in ("one_sentence", "what", "insights", "reco", "limits"):
        if k not in interp:
            reasons.append(f"Brak pola: {k}")

    one = str(interp.get("one_sentence") or "").strip()
    if not one:
        reasons.append("Puste one_sentence")

    # min 2 numeric anchors
    nums = re.findall(r"[-+]?\d+(?:[.,]\d+)?", one)
    if len(nums) < 2:
        reasons.append("one_sentence ma < 2 kotwice liczbowe")
    if not _dc_cs_has_specific_anchor(one, stats_payload):
        reasons.append("one_sentence nie odnosi sie do konkretnych anchorow CS")
    if _dc_cs_mentions_wrong_leader(one, stats_payload):
        reasons.append("one_sentence wskazuje niewlasciwego lidera")

    # PL-only
    low = (one + " " + " ".join(map(str, interp.get("insights") or []))).lower()
    if any(w in low for w in [" therefore ", " however ", " overall ", " basically ", " significantly "]):
        reasons.append("Wykryto fragmenty EN (PL-only violated)")

    # min lengths
    for field in ("what", "insights", "reco", "limits"):
        if not isinstance(interp.get(field), list) or len(interp[field]) < 1:
            reasons.append(f"Pole {field} jest puste")

    insights = [str(x or "").strip() for x in (interp.get("insights") or []) if str(x or "").strip()]
    if sum(1 for item in insights if _dc_cs_has_specific_anchor(item, stats_payload)) < 2:
        reasons.append("insights sa zbyt ogolne lub bez anchorow")
    if any(_dc_cs_mentions_wrong_leader(item, stats_payload) for item in insights):
        reasons.append("insights wskazuja niewlasciwego lidera")

    reco = [str(x or "").strip() for x in (interp.get("reco") or []) if str(x or "").strip()]
    if not any(_dc_cs_has_specific_anchor(item, stats_payload) for item in reco):
        reasons.append("reco nie odnosi sie do konkretnych anchorow")

    limits = [str(x or "").strip().lower() for x in (interp.get("limits") or []) if str(x or "").strip()]
    banned_limit_markers = [
        "konkurenc",
        "brak szczegolowych danych",
        "poszczegolnych kategoriach",
        "poszczegolnych krajach",
    ]
    if any(any(marker in item for marker in banned_limit_markers) for item in limits):
        reasons.append("limits zawieraja generyczne lub falszywe zastrzezenia")

    return len(reasons) == 0, reasons


def _dc_cs_fallback_deterministic(stats_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministyczny fallback CS - zero domyslow, tylko liczby z payload."""
    n_groups = int(stats_payload.get("n_groups") or 0)
    hhi = stats_payload.get("hhi", 0)
    hhi_class = stats_payload.get("hhi_class", "nieznana")
    pareto_80_n = stats_payload.get("pareto_80_n", "?")
    group_col = stats_payload.get("group_col") or "kategoria"
    total_value = stats_payload.get("total_value", 0)
    n_sub = stats_payload.get("n_subcategories")
    price_corr = stats_payload.get("price_corridor") or {}
    p80_price = price_corr.get("p80_price")
    corridor_pct = price_corr.get("corridor_share_pct")
    top1_label = _dc_cs_top_label(stats_payload, 0) or "brak"
    top2_label = _dc_cs_top_label(stats_payload, 1)
    top1 = _dc_cs_top_share(stats_payload, 0)
    top2 = _dc_cs_top_share(stats_payload, 1)
    top1_value = _dc_cs_top_value(stats_payload, 0)
    top3_share = _dc_cs_top_share_sum(stats_payload, 3)
    gap_pp = round(max(0.0, float(top1) - float(top2)), 2) if top2_label else None

    if n_groups == 0:
        return {
            "one_sentence": "Brak danych do analizy struktury po grupowaniu.",
            "what": ["Treemap wymaga co najmniej jednej grupy z wartoscia > 0."],
            "insights": ["Sprawdz wybor kolumny grupowej i wartosciowej w sidebarze."],
            "reco": ["Zmien kolumne grupowa lub wartosciowa."],
            "limits": ["Brak danych po agregacji."],
        }

    one = (
        f"Struktura jest {hhi_class}: '{top1_label}' odpowiada za {_dc_cs_fmt_pct(top1)} z {_dc_cs_fmt_pln(total_value)}, "
        f"a top-{pareto_80_n} grupy buduja 80% wartosci (HHI {float(hhi):.0f})."
    )

    what = [
        f"Treemap pokazuje, jak grupy '{group_col}' dziela {_dc_cs_fmt_pln(total_value)} calkowitej wartosci sprzedaży.",
        "Zakladka Kluczowe insighty rozwija ten sam przekroj o ranking, waterfall, Pareto, korytarz cenowy oraz mix 2-poziomowy.",
    ]
    if n_sub:
        what.append(f"Drugi wymiar rozbija strukture na {int(n_sub)} unikalnych kombinacji grup i skladnikow.")

    insights = [
        f"Liderem jest '{top1_label}' z udzialem {_dc_cs_fmt_pct(top1)} i wartoscia {_dc_cs_fmt_pln(top1_value)}.",
        (
            f"Luka do kolejnej grupy ('{top2_label}', {_dc_cs_fmt_pct(top2)}) wynosi {_dc_cs_fmt_pp(gap_pp)}."
            if top2_label and gap_pp is not None
            else f"Top-{pareto_80_n} grupy odpowiadaja juz za {_dc_cs_fmt_pct(top3_share, 2)}, co pokazuje waski rdzen portfela."
        ),
        f"Top-3 grupy odpowiadaja za {_dc_cs_fmt_pct(top3_share, 2)}, a HHI {float(hhi):.0f} potwierdza strukture {hhi_class}.",
    ]
    if p80_price is not None and corridor_pct is not None:
        insights.append(
            f"Korytarz cenowy do {p80_price} PLN odpowiada za {_dc_cs_fmt_pct(corridor_pct)}, wiec koncentracja dotyczy tez nizszych przedzialow cenowych."
        )

    reco = [
        f"Traktuj '{top1_label}' jako glowny punkt zarzadzania portfelem: odpowiada za {_dc_cs_fmt_pct(top1)} wartosci i wymaga osobnego monitoringu marzy oraz dostepnosci.",
        f"Zarzadzaj top-{pareto_80_n} grupami jako priorytetowym portfelem, bo to one buduja 80% wyniku i najszybciej przesuwaja cala strukture.",
    ]
    if p80_price is not None and corridor_pct is not None:
        reco.append(
            f"Pilnuj dostepnosci i pricingu do {p80_price} PLN, bo ten korytarz odpowiada za ok. {_dc_cs_fmt_pct(corridor_pct)} wartosci."
        )

    return {
        "one_sentence": one,
        "what": what[:3],
        "insights": insights[:4],
        "reco": reco[:3],
        "limits": _dc_cs_specific_limits(stats_payload),
    }


def _dc_llm_generate_cs_interpretation(
    stats_payload: Dict[str, Any],
    question: str,
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    """Parallel do _dc_llm_generate_interpretation_json — dla CS. Reuses _dc_parse_llm_json."""
    debug: Dict[str, Any] = {
        "gate_ok": False, "gate_reasons": [], "raw_text": "",
        "used_fallback": False, "model": None, "error": None,
    }

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or OpenAI is None:
        debug["error"] = "Brak OPENAI_API_KEY lub biblioteki openai — fallback deterministyczny."
        debug["used_fallback"] = True
        return None, debug

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    debug["model"] = chat_model

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=chat_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _DC_CS_SYSTEM_PROMPT},
                {"role": "user",   "content": _dc_cs_user_prompt(stats_payload, question)},
            ],
        )
        raw = ""
        try:
            raw = (resp.choices[0].message.content or "").strip()
        except Exception:
            raw = str(resp)

        debug["raw_text"] = raw
        obj = _dc_parse_llm_json(raw)
        if obj is None:
            debug["gate_reasons"] = ["Nie udało się sparsować JSON z odpowiedzi LLM."]
            return None, debug

        obj = _dc_postprocess_cs_interpretation(obj, stats_payload)
        debug["postprocessed"] = True
        debug["final_one_sentence"] = str(obj.get("one_sentence") or "").strip()
        ok, reasons = _dc_quality_gate_cs(obj, stats_payload)
        debug["gate_ok"] = ok
        debug["gate_reasons"] = reasons
        if not ok:
            return None, debug

        return obj, debug

    except Exception as e:
        debug["error"] = f"LLM CS error: {type(e).__name__}: {e}"
        return None, debug


def _dc_llm_generate_cs_takeaways(
    stats_payload: Dict[str, Any],
    question: str,
) -> tuple[Dict[str, str] | None, Dict[str, Any]]:
    """Generuje Executive takeaways dla bloków CS (McKinsey/Bain-grade, 1 zdanie / blok)."""
    debug: Dict[str, Any] = {
        "gate_ok": False, "gate_reasons": [], "raw_text": "",
        "used_fallback": False, "model": None, "error": None,
    }

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or OpenAI is None:
        debug["error"] = "Brak OPENAI_API_KEY lub biblioteki openai."
        debug["used_fallback"] = True
        return None, debug

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    debug["model"] = chat_model

    sys = (
        "Jesteś konsultantem McKinsey/Bain. Tworzysz *Executive takeaway* (1 zdanie) dla bloków w zakładce "
        "'Kluczowe insighty' gałęzi Composition Static.\n"
        "\n"
        "ZASADY (MUST):\n"
        "1) Każdy takeaway to 2 zdania (max ~350 znaków), bez ogólników.\n"
        "2) MUSI zawierać co najmniej 1 twardą liczbę z wejściowego stats_payload (%, wartość, HHI, P80 itp.).\n"
        "3) Styl: decyzja + fakt liczbowy + implikacja (ryzyko/okazja).\n"
        "4) Nie wymyślaj liczb. Jeśli brakuje danych, napisz: 'Brak danych do wniosku.'\n"
        "\n"
        "SZABLON (stosuj konsekwentnie):\n"
        "• 'Wynik jest <ocena>: <fakt liczbowy>, co <implikacja biznesowa>.'\n"
        "\n"
        "PRZYKŁAD: 'Wynik jest silnie skoncentrowany: 1 segment (RL) generuje 83% sprzedaży, co zwiększa ryzyko zależności portfela.'\n"
        "\n"
        "Zwróć JSON dokładnie w formacie: {ranking, waterfall, pareto, mix, marimekko, price_corridor} "
        "— wartości to stringi (takeaway)."
    ).strip()
    user = {
        "question": question,
        "stats_payload": stats_payload,
        "required_keys": ["ranking","waterfall","pareto","mix","marimekko","price_corridor"],
        "output_contract": {
            "type": "object",
            "properties": {
                "ranking": {"type":"string"},
                "waterfall": {"type":"string"},
                "pareto": {"type":"string"},
                "mix": {"type":"string"},
                "marimekko": {"type":"string"},
                "price_corridor": {"type":"string"},
            },
            "required": ["ranking","waterfall","pareto","mix","marimekko","price_corridor"]
        }
    }

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=chat_model,
            temperature=0.2,
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":json.dumps(user, ensure_ascii=False)}
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        debug["raw_text"] = txt
        obj = _dc_parse_llm_json(txt)

        # hard gate
        if not isinstance(obj, dict):
            debug["gate_reasons"].append("Not a JSON object.")
            return None, debug

        required = ["ranking","waterfall","pareto","mix","marimekko","price_corridor"]
        missing = [k for k in required if not str(obj.get(k,"")).strip()]
        if missing:
            debug["gate_reasons"].append(f"Missing keys: {missing}")
            return None, debug

        # numeric anchor gate (cheap)
        def _has_digit(s: str) -> bool:
            return any(ch.isdigit() for ch in s)

        bad = [k for k in required if not _has_digit(str(obj.get(k,"")))]
        if bad:
            debug["gate_reasons"].append(f"No numeric anchor in: {bad}")
            return None, debug

        debug["gate_ok"] = True
        return {k: str(obj[k]).strip() for k in required}, debug

    except Exception as e:
        debug["error"] = f"{e}"
        return None, debug




def _dc_parse_llm_json(text: str) -> Dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    # czasem model otacza ```json ... ```
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _dc_quality_gate_hard(
    interp: Dict[str, Any],
    stats_payload: Dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Hard gate: odcina „ładne kłamstwa”.
    Zwraca: (ok, reasons)
    """
    reasons: list[str] = []

    # 1) Wymagana struktura
    for k in ("one_sentence", "what", "insights", "reco", "limits"):
        if k not in interp:
            reasons.append(f"Brak pola: {k}")

    one = str(interp.get("one_sentence") or "").strip()
    if not one:
        reasons.append("Puste one_sentence")

    # 2) PL-only (lekki guard): jeśli są typowe angielskie spójniki, odcinamy
    low = (one + " " + " ".join(map(str, interp.get("insights") or []))).lower()
    if any(w in low for w in [" therefore ", " however ", " overall ", " basically ", " significantly "]):
        reasons.append("Wykryto fragmenty EN (PL-only violated)")

    # 3) Min. 2 kotwice liczbowe w one_sentence
    nums = re.findall(r"[-+]?\d+(?:[.,]\d+)?", one)
    if len(nums) < 2:
        reasons.append("one_sentence ma < 2 kotwice liczbowe")

    # 4) Zakaz 'większość' bez %
    if "większo" in low:
        if "%" not in one and "%" not in low:
            reasons.append("Użyto 'większość' bez podania %")

    # 5) Symetria tylko przy skewness≈0
    skew = stats_payload.get("skewness", None)
    if any(w in low for w in ["symetrycz", "symetria"]):
        if skew is None:
            reasons.append("Symetria przy braku skewness w payload")
        else:
            try:
                if abs(float(skew)) > 0.2:
                    reasons.append(f"Symetria mimo |skewness|={float(skew):.3f} > 0.2")
            except Exception:
                reasons.append("Symetria: nie udało się ocenić skewness")
    
    # 5b) SUPER-GUARD: zakaz "długi ogon" bez podstaw w skewness (None lub |skewness|<=0.2)
    if any(w in low for w in ["długi ogon", "dlugi ogon", "ogon", "long tail"]):
        if skew is None:
            reasons.append("Użyto 'długi ogon/ogon' przy braku skewness w payload")
        else:
            try:
                if abs(float(skew)) <= 0.2:
                    reasons.append(f"Użyto 'długi ogon/ogon' mimo |skewness|={abs(float(skew)):.3f} <= 0.2")
            except Exception:
                reasons.append("Użyto 'długi ogon/ogon' — nie udało się ocenić skewness")    

    # 6) Outliery muszą być spójne z outlier_rate_iqr + fences
    out_rate = stats_payload.get("outlier_rate_iqr", None)
    fences = stats_payload.get("fences", {}) if isinstance(stats_payload.get("fences"), dict) else {}
    whole_txt = json.dumps(interp, ensure_ascii=False).lower()

    mentions_outliers = any(w in low for w in ["outlier", "odstając", "odstające", "odstających"])
    if mentions_outliers:
        if out_rate is None:
            reasons.append("Wzmianka o outlierach bez outlier_rate_iqr w payload")
        else:
            # (A) musi pojawić się % gdziekolwiek w JSON
            if "%" not in whole_txt:
                reasons.append("Wzmianka o outlierach bez %")

            # (B) SUPER-TWARDY: outlier % w tekście ≈ out_rate*100 (±2 pp)
            # szukamy liczb z '%'
            perc_vals = []
            for m in re.finditer(r"([-+]?\d+(?:[.,]\d+)?)\s*%", whole_txt):
                try:
                    v = float(m.group(1).replace(",", "."))
                    perc_vals.append(v)
                except Exception:
                    pass
            if not perc_vals:
                reasons.append("Outliery: brak wartości % do porównania z outlier_rate_iqr")
            else:
                target = float(out_rate) * 100.0
                best = min(abs(p - target) for p in perc_vals)
                if best > 2.0:
                    reasons.append(f"Outliery: % w tekście niespójny z payload (min|Δ|={best:.2f} pp, target={target:.2f}%)")

            # fences: jeżeli wspomniane progi, powinny istnieć w payload
            if ("fence" in whole_txt or "próg" in whole_txt or "iqr" in whole_txt):
                if not (isinstance(fences.get("hi"), (int, float)) and isinstance(fences.get("lo"), (int, float))):
                    reasons.append("Wzmianka o progach/fences, ale brak fences w payload")

    # 8) SUPER-TWARDY: kotwice liczbowe w one_sentence muszą pochodzić z konkretnych pól payload
    # Dozwolone: median, q1, q3, fences.lo, fences.hi, outlier_rate_iqr*100
    allowed = []
    def _push(x):
        if isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x)):
            allowed.append(float(x))

    _push(stats_payload.get("median"))
    _push(stats_payload.get("q1"))
    _push(stats_payload.get("q3"))
    _push(stats_payload.get("mean"))
    _push(stats_payload.get("p90"))
    _push(stats_payload.get("p95"))
    _push(stats_payload.get("min"))
    _push(stats_payload.get("max"))
    _push(stats_payload.get("n"))
    if isinstance(fences, dict):
        _push(fences.get("lo"))
        _push(fences.get("hi"))
    if isinstance(out_rate, (int, float)):
        _push(float(out_rate) * 100.0)  # percent anchor

    # wyciągamy liczby z one_sentence wraz z info czy to procent
    anchors = []
    for m in re.finditer(r"([-+]?\d+(?:[.,]\d+)?)\s*(%)?", one):
        try:
            v = float(m.group(1).replace(",", "."))
            is_pct = bool(m.group(2))
            anchors.append((v, is_pct))
        except Exception:
            pass

    # muszą być co najmniej 2 (to już jest wyżej), ale tu weryfikujemy pochodzenie
    def _matches_allowed(v: float, is_pct: bool) -> bool:
        if not allowed:
            return False
        if is_pct:
            # procent: twardo porównuj do outlier_rate_iqr*100 (±2 pp)
            target = float(out_rate) * 100.0 if isinstance(out_rate, (int, float)) else None
            if target is None:
                return False
            return abs(v - target) <= 2.0

        # liczby nie-%: dopasuj do median/q1/q3/fences z tolerancją
        # tolerancja: ±1% wartości lub ±2 jednostki (cokolwiek większe)
        for a in allowed:
            if a == float(out_rate) * 100.0:  # skip percent proxy w nie-% porównaniu
                continue
            tol = max(abs(a) * 0.01, 2.0)
            if abs(v - a) <= tol:
                return True
        return False

    bad = []
    for (v, is_pct) in anchors:
        if not _matches_allowed(v, is_pct):
            bad.append(f"{v}{'%' if is_pct else ''}")

    if bad:
        reasons.append("one_sentence: kotwice liczbowe nie pochodzą z dozwolonych pól payload: " + ", ".join(bad))


    # 7) Minimalna sensowność list (nie puste)
    def _len_list(x: Any) -> int:
        return len(x) if isinstance(x, list) else 0
    if _len_list(interp.get("what")) < 1:
        reasons.append("Pole what jest puste")
    if _len_list(interp.get("insights")) < 1:
        reasons.append("Pole insights jest puste")
    if _len_list(interp.get("limits")) < 1:
        reasons.append("Pole limits jest puste")

    ok = len(reasons) == 0
    return ok, reasons

def _dc_interp_fallback_deterministic(stats_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministyczny fallback: zero domysłów, tylko liczby i bezpieczne sformułowania."""
    dist_col = str(stats_payload.get("dist_col") or "miara")
    n = int(stats_payload.get("n") or 0)

    if n < 8:
        one = "Za mało danych po filtrach, aby wiarygodnie opisać kształt rozkładu i outliery (wymagane ≥ 8 obserwacji)."
        return {
            "one_sentence": one,
            "what": [f"Histogram + KDE dla miary **{dist_col}** na danych po filtrach."],
            "insights": [f"Liczba obserwacji po filtrach: **{n}** (zbyt mało do stabilnych wniosków)."],
            "reco": ["Poszerz zakres filtrów lub usuń filtr kategorii i uruchom analizę ponownie."],
            "limits": ["Wnioski ograniczone przez małą próbę po filtrach."],
        }

    med = float(stats_payload.get("median"))
    q1 = float(stats_payload.get("q1"))
    q3 = float(stats_payload.get("q3"))
    out = float(stats_payload.get("outlier_rate_iqr"))
    hi = float((stats_payload.get("fences") or {}).get("hi"))
    lo = float((stats_payload.get("fences") or {}).get("lo"))
    skew = stats_payload.get("skewness", None)

    one = (
        f"Mediana **{dist_col}** wynosi ~{med:,.0f}, typowy zakres (IQR) to ~{q1:,.0f}–{q3:,.0f}, "
        f"a outliery wg IQR stanowią ~{out*100:.1f}% (fences: ~{lo:,.0f}–{hi:,.0f})."
    ).replace(",", " ")

    skew_txt = "brak" if skew is None else f"{float(skew):.2f}"
    shape_safe = (
        "Nie formułuję wniosku o symetrii, bo wymaga to jednoznacznej podstawy w skośności."
        if skew is None else
        ("Skośność jest bliska 0, więc rozkład jest zbliżony do symetrycznego."
         if abs(float(skew)) <= 0.2 else
         f"Skośność (skewness={float(skew):.2f}) wskazuje na niesymetryczny rozkład.")
    )

    return {
        "one_sentence": one,
        "what": [
            f"Histogram pokazuje liczebność obserwacji w przedziałach wartości **{dist_col}**.",
            "Linia KDE pokazuje kształt rozkładu (gęstość) niezależnie od szerokości binów.",
            "Fences (IQR) wskazują progi typowego zakresu oraz obszary potencjalnych outlierów.",
        ],
        "insights": [
            f"Liczba obserwacji po filtrach: **{n}**.",
            f"Outliery wg IQR: **{out*100:.1f}%** (próg dolny ~{lo:,.0f}, próg górny ~{hi:,.0f}).".replace(",", " "),
            f"Skośność (skewness): **{skew_txt}**. {shape_safe}",
        ],
        "reco": [
            "Jeśli outliery wpływają na decyzje: potwierdź ich pochodzenie (błąd vs realne zdarzenie) na poziomie rekordów.",
            "W analizach porównawczych raportuj medianę oraz IQR (odporne na outliery) obok średniej.",
        ],
        "limits": [
            "Interpretacja dotyczy wyłącznie danych po aktualnych filtrach.",
            "Bez dodatkowych informacji nie wnioskuję o przyczynach biznesowych obserwowanych kształtów.",
        ],
    }


def _dc_postprocess_distribution_interpretation(
    interp: Dict[str, Any],
    stats_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Stabilizuje user-facing output Distribution bez ruszania ET."""
    base = _dc_interp_fallback_deterministic(stats_payload)

    def _pick_list(field: str, max_items: int, *, force_base: bool = False) -> list[str]:
        if force_base:
            return [str(x or "").strip() for x in (base.get(field) or []) if str(x or "").strip()][:max_items]

        out: list[str] = []
        value = interp.get(field)
        if isinstance(value, list):
            for item in value:
                text = re.sub(r"\s+", " ", str(item or "").strip())
                if not text:
                    continue
                low = text.lower()
                if field == "reco" and "w czasie" in low:
                    continue
                if field == "limits" and any(
                    marker in low for marker in ["błę", "nietypow", "przyczyn", "sezon", "trend", "kategor"]
                ):
                    continue
                if text not in out:
                    out.append(text)
        if not out:
            out = [str(x or "").strip() for x in (base.get(field) or []) if str(x or "").strip()]
        return out[:max_items]

    return {
        "one_sentence": str(base.get("one_sentence") or "").strip(),
        "what": _pick_list("what", 3),
        "insights": _pick_list("insights", 4),
        "reco": _pick_list("reco", 3),
        "limits": _pick_list("limits", 3, force_base=True),
    }

def _dc_llm_generate_interpretation_json(
    stats_payload: Dict[str, Any],
    question: str,
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    """
    Zwraca: (interp_json_or_none, debug_meta)
    debug_meta zawiera: gate_ok, reasons, raw_text, model, etc.
    """
    debug: Dict[str, Any] = {
        "gate_ok": False,
        "gate_reasons": [],
        "raw_text": "",
        "used_fallback": False,
        "model": None,
        "error": None,
    }

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or OpenAI is None:
        debug["error"] = "Brak OPENAI_API_KEY lub biblioteki openai — fallback deterministyczny."
        debug["used_fallback"] = True
        return None, debug

    # Model do czatu (nie TTS/STT)
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    debug["model"] = chat_model

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=chat_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _DC_INTERP_SYSTEM_PROMPT},
                {"role": "user", "content": _dc_interp_user_prompt(stats_payload, question)},
            ],
        )
        raw = ""
        try:
            raw = (resp.choices[0].message.content or "").strip()
        except Exception:
            raw = str(resp)

        debug["raw_text"] = raw
        obj = _dc_parse_llm_json(raw)
        if obj is None:
            debug["gate_reasons"] = ["Nie udało się sparsować JSON z odpowiedzi LLM."]
            debug["used_fallback"] = True
            return None, debug

        obj = _dc_postprocess_distribution_interpretation(obj, stats_payload)
        debug["postprocessed"] = True
        debug["final_one_sentence"] = str(obj.get("one_sentence") or "").strip()
        ok, reasons = _dc_quality_gate_hard(obj, stats_payload)
        debug["gate_ok"] = ok
        debug["gate_reasons"] = reasons
        if not ok:
            debug["used_fallback"] = True
            return None, debug

        return obj, debug

    except Exception as e:
        debug["error"] = f"LLM error: {type(e).__name__}: {e}"
        debug["used_fallback"] = True
        return None, debug

# -----------------------
# TTS (lektors) helpers
# -----------------------
@st.cache_data(show_spinner=False, ttl=3600)
def datachat_tts_generate(
    text: str,
    voice: str,
    model: str,
    api_key: str,
    cache_bust: str = "v3",  # <- ważne: zmienia klucz cache i “odcina” stare puste wyniki
) -> tuple[bytes, str | None]:
    """Generuje MP3 z tekstu. Zwraca (bytes_mp3, error_str)."""
    try:
        if not api_key:
            return b"", "Brak OPENAI_API_KEY (sprawdź .env)."
        _t = (text or "").strip()
        if not _t:
            return b"", "Brak tekstu do narracji."

        if OpenAI is None:
            return b"", "Brak biblioteki `openai`. Zainstaluj: pip install openai"

        client = OpenAI(api_key=api_key)

        # Różne wersje SDK obsługują różne nazwy parametru: response_format vs format
        try:
            resp = client.audio.speech.create(
                model=model or "gpt-4o-mini-tts",
                voice=voice or "alloy",
                input=_t,
                response_format="mp3",
            )
        except TypeError:
            resp = client.audio.speech.create(
                model=model or "gpt-4o-mini-tts",
                voice=voice or "alloy",
                input=_t,
                format="mp3",
            )

        # Różne typy odpowiedzi: bytes / obiekt z .read() / itp.
        if isinstance(resp, (bytes, bytearray)):
            audio_bytes = bytes(resp)
        elif hasattr(resp, "read"):
            audio_bytes = resp.read()
        elif hasattr(resp, "content"):
            audio_bytes = resp.content
        else:
            audio_bytes = bytes(resp)

        if not audio_bytes:
            return b"", "TTS zwrócił pusty plik audio."
        return audio_bytes, None

    except Exception as e:
        return b"", f"TTS error: {type(e).__name__}: {e}"

def _datachat_tts_generate_mp3(
    text: str,
    voice: str,
    model: str,
    api_key: str,
    cache_bust: str = "v3",
) -> tuple[bytes, str | None]:
    """
    Kompatybilność wstecz: w kilku patchach UI wołał _datachat_tts_generate_mp3.
    Delegujemy do datachat_tts_generate() (cache + obsługa SDK).
    """
    return datachat_tts_generate(
        text=text,
        voice=voice,
        model=model,
        api_key=api_key,
        cache_bust=cache_bust,
    )

def _data_source_info() -> Dict[str, Any]:
    """Ustala źródło danych dla Data Chat (Stage3) w sposób stabilny i deterministyczny.

    Zasada: zawsze wybieramy NAJŚWIEŻSZY `ready_for_training.parquet` (nigdy *_full_masked.parquet).
    Priorytety:
    - session_state: datachat_handoff / latest_artifacts
    - ingest/_latest_handoff.json
    - fallback: skan ingest po */ready_for_training.parquet
    """
    info = _resolve_stage3_parquet_path()
    p = info.get("path")
    if isinstance(p, str) and p and os.path.exists(p):
        return {
            "source": info.get("source") or "unknown",
            "path": p,
            "mtime": info.get("mtime"),
            "ingest_root": info.get("ingest_root"),
        }
    return {"source": None, "path": None, "ingest_root": info.get("ingest_root")}

def _get_df_from_session(max_rows: int | None = None) -> pd.DataFrame | None:
    """Adapter Etap 2 -> Data Chat (Parquet only).

    Kluczowe:
    - Domyślnie ładujemy PEŁNY zbiór (max_rows=None), bo metryki i zakres czasu muszą być zgodne z parquettami.
    - Jeżeli chcesz próbkę, ustaw max_rows na liczbę (np. 200_000).
    """
    # Restore handoff from JSON pointer on disk (survives full rerun / navigation).
    if not isinstance(st.session_state.get("datachat_handoff"), dict):
        hp = st.session_state.get("datachat_handoff_path")
        if isinstance(hp, str) and hp and os.path.exists(hp):
            try:
                with open(hp, "r", encoding="utf-8") as f:
                    st.session_state["datachat_handoff"] = json.load(f)
            except Exception:
                pass

        # Resolve latest Stage2 output (freshness-based; never load *_full_masked.parquet)
    src_info = _data_source_info()
    st.session_state["datachat_source_info"] = src_info
    p = src_info.get("path")
    if isinstance(p, str) and p and os.path.exists(p):
        st.session_state["datachat_ready_path"] = p
        loaded = _load_df_from_parquet_cached(p, max_rows=max_rows)
        if isinstance(loaded, pd.DataFrame) and not loaded.empty:
            return loaded

    # fallback:

    # fallback: stare klucze w session_state
    df_ready_key = st.session_state.get("df_ready_key") or "df_ready_for_training"
    df_ready = st.session_state.get(df_ready_key)
    if isinstance(df_ready, pd.DataFrame) and not df_ready.empty:
        return df_ready if max_rows is None else df_ready.head(max_rows)

    # Fallback 2: latest_artifacts pointers (Etap 1/2)
    latest = st.session_state.get("latest_artifacts")
    if isinstance(latest, dict):
        for key in (
            "stage2_ready_parquet_path",
            "ready_parquet_path",
            "parquet_path",
            "parquet_path_full",
        ):
            p = latest.get(key)
            if isinstance(p, str) and p and os.path.exists(p):
                st.session_state["datachat_ready_path"] = p
                loaded = _load_df_from_parquet_cached(p, max_rows=max_rows)
                if isinstance(loaded, pd.DataFrame) and not loaded.empty:
                    return loaded

    # 4) As a last resort, discover latest Stage 2 artifacts on disk.
    disk_path = _discover_latest_stage2_ready_parquet()
    if disk_path is not None:
        loaded = _load_df_from_parquet_cached(disk_path, max_rows=max_rows)
        if isinstance(loaded, pd.DataFrame) and not loaded.empty:
            return loaded

    return None


def _get_schema_ctx_from_session() -> Dict[str, Any]:
    v = st.session_state.get("datachat_schema_ctx")
    return v if isinstance(v, dict) else {}

def _get_chart_spec_from_session() -> Dict[str, Any] | None:
    v = st.session_state.get("datachat_last_chart_spec")
    return v if isinstance(v, dict) else None


def _get_persisted_datachat_prompt() -> str:
    for key in ("datachat_prompt_saved", "datachat_prompt_input", "datachat_prompt", "datachat_last_question"):
        v = st.session_state.get(key)
        if isinstance(v, str):
            return v
    return ""


def _set_persisted_datachat_prompt(value: Any) -> str:
    prompt_value = "" if value is None else str(value)
    st.session_state["datachat_prompt_saved"] = prompt_value
    st.session_state["datachat_prompt_input"] = prompt_value
    st.session_state["datachat_prompt"] = prompt_value
    return prompt_value


def _guess_datachat_intent_from_prompt(prompt: Any, fallback_intent: str | None = None) -> str:
    prompt_norm = str(prompt or "").strip().lower()
    fallback = str(fallback_intent or st.session_state.get("datachat_active_intent", "distribution") or "distribution").strip().lower()
    if fallback == "composition":
        fallback = "composition_static"

    cs_keywords = [
        "per kategori", "wg kategor", "struktura", "udział", "share",
        "mix", "pareto", "treemap", "koncentrac", "sprzedaży i per",
        "sprzedaż per", "udziały per", "ranking",
    ]
    dist_keywords = [
        "rozkład", "rozklad", "histogram", "kde", "ecdf", "boxplot",
        "violin", "rug", "dystrybu", "mediana", "percentyl", "kwantyl",
        "outlier", "odstając", "skoś", "skos", "gęstość", "gestosc",
        "iqr", "fence", "logarytm", "wartości (histogram", "wartosci (histogram",
    ]
    cot_time_hints = (
        "w czasie", "trend", "dynamik", "zmienia", "zmieniała",
        "miesiąc", "miesiac", "kwarta", "tydzień", "tydzien",
    )
    has_time_hint = bool(re.search(r"\b(19|20)\d{2}\b", prompt_norm)) or any(h in prompt_norm for h in cot_time_hints)

    if any(kw in prompt_norm for kw in cs_keywords):
        return "composition_over_time" if has_time_hint else "composition_static"
    if any(kw in prompt_norm for kw in dist_keywords):
        return "distribution"
    return fallback or "distribution"


def _sync_prompt_and_sidebar_intent(value: Any | None = None) -> str:
    prompt_value = _set_persisted_datachat_prompt(
        st.session_state.get("datachat_prompt_input") if value is None else value
    )
    chart_spec = _get_chart_spec_from_session() or {}
    base_intent = str(chart_spec.get("intent") or st.session_state.get("datachat_active_intent", "distribution") or "distribution")
    new_intent = _guess_datachat_intent_from_prompt(prompt_value, base_intent)
    st.session_state["datachat_active_intent"] = new_intent
    st.session_state["datachat_analysis_running"] = False
    if new_intent.startswith("composition"):
        st.session_state["datachat_cs_active_view"] = "overview"
        st.session_state["datachat_cs_active_view_label"] = "Obraz całości"
    return prompt_value


def _update_distribution_debug_sidebar_exports() -> None:
    slots = st.session_state.get("_dist_debug_sidebar_slots")
    if not isinstance(slots, dict):
        return

    try:
        _dist_log = distribution.get_debug_checkpoints() if hasattr(distribution, "get_debug_checkpoints") else []
    except Exception:
        _dist_log = []
    _dist_exec_log = [x for x in _dist_log if str((x or {}).get("where") or "").startswith("dist.exec.")]
    _dist_interp_log = [x for x in _dist_log if str((x or {}).get("where") or "").startswith("dist.interp.")]
    _dist_summary = st.session_state.get("dist_exec_summary_v1") or {}
    _dist_interp_debug = (
        st.session_state.get("dist_interp_debug_v1")
        or st.session_state.get("dc_interp_debug")
        or {}
    )

    _count_slot = slots.get("count_slot")
    if _count_slot is not None:
        _count_slot.caption(
            f"Checkpointy w pamięci: {len(_dist_log)} | ET dist.exec.*: {len(_dist_exec_log)} | "
            f"Interpretacja dist.interp.*: {len(_dist_interp_log)}"
        )

    _enabled = bool(st.session_state.get("dist_debug_enabled", False))
    for _name in ("summary_slot", "exec_slot", "interp_slot", "interp_debug_slot", "all_cp_slot"):
        _slot = slots.get(_name)
        if _slot is not None:
            _slot.empty()

    if not _enabled:
        return

    _summary_slot = slots.get("summary_slot")
    if _summary_slot is not None:
        with _summary_slot.container():
            st.markdown("**Distribution / Plan 9.5 - skopiuj ten JSON**")
            st.code(json.dumps(_dist_summary, ensure_ascii=False), language="json")

    _exec_slot = slots.get("exec_slot")
    if _exec_slot is not None:
        with _exec_slot.container():
            st.markdown("**Logi ET / dist.exec.* (kopiuj to do audytu)**")
            st.code(json.dumps(_dist_exec_log, ensure_ascii=False), language="json")

    _interp_slot = slots.get("interp_slot")
    if _interp_slot is not None:
        with _interp_slot.container():
            st.markdown("**Logi Interpretacja / dist.interp.* (kopiuj to do audytu)**")
            st.code(json.dumps(_dist_interp_log, ensure_ascii=False), language="json")

    _interp_debug_slot = slots.get("interp_debug_slot")
    if _interp_debug_slot is not None:
        with _interp_debug_slot.container():
            with st.expander("Interpretacja / stats_payload + quality gate (Distribution)", expanded=False):
                st.markdown("**stats_payload (jedyna prawda)**")
                st.json(_dist_interp_debug.get("stats_payload") or {})
                st.markdown("**LLM / gate debug**")
                st.json(_dist_interp_debug.get("llm_debug") or {})

    _all_cp_slot = slots.get("all_cp_slot")
    if _all_cp_slot is not None:
        with _all_cp_slot.container():
            with st.expander("Pełne checkpointy Distribution (opcjonalnie)", expanded=False):
                st.code(json.dumps(_dist_log, ensure_ascii=False), language="json")


def _make_cp0_cache_key(
    *,
    df: pd.DataFrame,
    prompt: str,
    chart_spec: Dict[str, Any] | None,
    roles: Dict[str, Any] | None,
    filters: Dict[str, Any] | None,
) -> str:
    source_info = st.session_state.get("datachat_source_info") or {}
    payload = {
        "prompt": str(prompt or "").strip(),
        "chart_spec": chart_spec or {},
        "roles": roles or {},
        "filters": filters or {},
        "rows": int(len(df) if isinstance(df, pd.DataFrame) else 0),
        "cols": [str(c) for c in (df.columns.tolist() if isinstance(df, pd.DataFrame) else [])],
        "source_path": str(source_info.get("path") or ""),
        "source_mtime": source_info.get("mtime"),
    }
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _get_roles_from_session() -> Dict[str, Any]:
    # Adapter Etap 2 -> Data Chat:
    # - preferuj 'eda_roles' (spójny kontrakt ról)
    # - uzupełnij o 'datachat_handoff.selected' jeżeli jest dostępne
    roles: Dict[str, Any] = {}

    v = st.session_state.get("eda_roles")
    if isinstance(v, dict):
        roles.update(v)

    handoff = st.session_state.get("datachat_handoff")
    if isinstance(handoff, dict):
        sel = handoff.get("selected")
        if isinstance(sel, dict):
            # nie nadpisuj istniejących ról, tylko uzupełnij
            for k_src, k_dst in (
                ("outcome_col", "outcome_col"),
                ("group_col", "group_col"),
                ("value_col", "value_col"),
                ("target_val", "target_val"),
            ):
                if k_dst not in roles and sel.get(k_src) is not None:
                    roles[k_dst] = sel.get(k_src)

    return roles

def _render_sidebar_filters(df: pd.DataFrame, schema_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Kontrolki filtrów w sidebarze (MVP) — bez ryzyka regresji.
    Zwraca słownik ustawień, które wykorzystujemy do filtrowania danych i do override w gałęzi Distribution.
    """
    if df is None or df.empty:
        return {}

    def _looks_numeric_like(_s: pd.Series, sample_n: int = 5000) -> bool:
        try:
            if pd.api.types.is_numeric_dtype(_s):
                return True
            if pd.api.types.is_datetime64_any_dtype(_s):
                return False
            ss = _s.dropna().head(sample_n)
            if ss.empty:
                return False
            nn = pd.to_numeric(ss, errors="coerce")
            return float(nn.notna().mean()) >= 0.95
        except Exception:
            return False

    def _is_dist_technical_col(_col: str) -> bool:
        low = str(_col or "").strip().lower()
        return (
            low.startswith("is_outlier_")
            or low.startswith("is_")
            or low.startswith("_")
            or "outlier_" in low
            or low.endswith("_outlier")
            or low.startswith("tmp_")
        )

    def _looks_like_dist_id(_col: str) -> bool:
        low = str(_col or "").strip().lower()
        if not low:
            return False
        if low in {"id", "idx", "index"}:
            return True
        if low.endswith("_id"):
            return True
        return ("id" in low) and (len(low) <= 8)

    def _is_distribution_measure_candidate(_c: str) -> bool:
        if not isinstance(_c, str) or _c not in df.columns:
            return False
        if _is_dist_technical_col(_c) or _looks_like_dist_id(_c):
            return False
        _s = df[_c]
        if pd.api.types.is_datetime64_any_dtype(_s) or pd.api.types.is_bool_dtype(_s):
            return False
        if not _looks_numeric_like(_s):
            return False
        try:
            _nn = pd.to_numeric(_s, errors="coerce").dropna().head(50000)
            if _nn.empty:
                return False
            if int(_nn.nunique(dropna=True)) < 3:
                return False
        except Exception:
            return False
        return True

    num_cols = list(schema_ctx.get("num_cols") or [])
    if not num_cols:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    num_cols = [c for c in num_cols if _is_distribution_measure_candidate(c)]

    # Kolumny kategoryczne (w tym niska krotność liczb)
    cat_cols = list(schema_ctx.get("cat_cols") or [])

    def _distribution_segment_candidates(_selected_measure: Optional[str] = None) -> list[str]:
        opts: list[str] = []
        for _c in list(df.columns):
            if not isinstance(_c, str):
                continue
            if _selected_measure and _c == _selected_measure:
                continue
            if _is_dist_technical_col(_c):
                continue
            try:
                _s = df[_c]
                if pd.api.types.is_datetime64_any_dtype(_s):
                    continue
                _nun = int(_s.nunique(dropna=True)) if len(_s) else 0
                if _nun < 2:
                    continue
                if pd.api.types.is_numeric_dtype(_s):
                    _sample = pd.to_numeric(_s.dropna().head(5000), errors="coerce").dropna()
                    if _sample.empty:
                        continue
                    _int_like = float((_sample - _sample.round()).abs().max()) < 1e-9
                    if (not _int_like) or (_nun > 60):
                        continue
                else:
                    if _looks_numeric_like(_s):
                        continue
                    if _nun > 120:
                        continue
            except Exception:
                continue
            opts.append(_c)
        return opts

    # Domyślny dist_col
    preferred_default = st.session_state.get("dc_dist_col")
    if not preferred_default:
        preferred_default = "SalePrice" if "SalePrice" in num_cols else (num_cols[0] if num_cols else None)

    # --- TTS: zmiana ustawień = wyczyść audio cache + bust cache_data
    def _dc_tts_bump_cache():
        try:
            st.session_state["datachat_tts_cache_bust"] = str(
                int(st.session_state.get("datachat_tts_cache_bust", "0") or 0) + 1
            )
        except Exception:
            st.session_state["datachat_tts_cache_bust"] = "1"

        if isinstance(st.session_state.get("datachat_tts_audio"), dict):
            st.session_state["datachat_tts_audio"].clear()

    if "debug_exec_takeaway_global" not in st.session_state:
        st.session_state["debug_exec_takeaway_global"] = False

    with st.sidebar:

        # ── Aktualizuj active_intent PRZED renderem widgetów
        # Na pierwszym uruchomieniu (brak w state) → default "distribution"
        # Po kolejnym rerun routing block zapisuje prawdziwy intent
        _active_intent = st.session_state.get("datachat_active_intent", "distribution")

        # ─────────────────────────────────────────────
        # Filtry danych — panel przełączany wg _active_intent
        # ─────────────────────────────────────────────
        _panel_label = (
            "📊 Filtry — Distribution"
            if _active_intent == "distribution"
            else (
                "📊 Filtry — Composition Over Time"
                if _active_intent in ("composition_over_time", "cot")
                else "📊 Filtry — Composition Static"
            )
        )

        # defaults (needed for return dict even when panel != that panel)
        # --- defaults for Distribution (must exist regardless of active panel) ---
        dist_col = None
        dist_range = None
        color_col = None
        filt_col = None
        filt_values = []

        # time filter defaults (used in return dict)
        date_cols = list(schema_ctx.get("date_cols") or [])

        # FAST sanity-check for date columns (prevents cases like Invoice IDs parsed into years 2049+)
        _date_cols_raw = list(date_cols)
        def _is_valid_date_col_fast(_col: str, sample_n: int = 20000) -> bool:
            try:
                if not _col or _col not in df.columns:
                    return False
                s = df[_col]
                if pd.api.types.is_datetime64_any_dtype(s):
                    ss = s.dropna()
                    if ss.empty:
                        return False
                    y_med = float(ss.dt.year.median())
                else:
                    ss = s.head(sample_n)
                    dt = pd.to_datetime(ss, errors="coerce", infer_datetime_format=True)
                    ok = float(dt.notna().mean()) if len(dt) else 0.0
                    if ok < 0.85:
                        return False
                    years = dt.dropna().dt.year
                    if years.empty:
                        return False
                    y_med = float(years.median())
                now_y = pd.Timestamp.now().year
                if y_med > (now_y + 2) or y_med < 1900:
                    return False
                return True
            except Exception:
                return False

        date_cols = [c for c in date_cols if _is_valid_date_col_fast(c)]
        if _date_cols_raw and not date_cols:
            # fail-safe: never drop everything; keep raw if validation misfired
            date_cols = _date_cols_raw

        date_col = None
        date_range = None
        cs_group_col = None
        cs_group_col2 = "(brak)"
        cs_top_n = int(st.session_state.get("cs__top_n", 10))
        cs_cutoff = float(st.session_state.get("cs__cutoff", 0.80))
        cs_price_col = st.session_state.get("cs__price_col", "(auto)")

        cot_cat_col = st.session_state.get("cot__cat_col", "(auto)")
        cot_top_n = int(st.session_state.get("cot__top_n", 10))
        cot_include_other = bool(st.session_state.get("cot__include_other", True))

        with st.expander(_panel_label, expanded=False):
            # ══════════════════════════════════════════
            # A) DISTRIBUTION panel (keys: dist__*)
            # ══════════════════════════════════════════
            if _active_intent == "distribution":
                dist_col = None
                if num_cols:
                    _dist_state = st.session_state.get("dist__col")
                    if _dist_state not in num_cols:
                        st.session_state["dist__col"] = (
                            preferred_default if preferred_default in num_cols else num_cols[0]
                        )
                    dist_col = st.selectbox(
                        "Miara do rozkładu",
                        options=num_cols,
                        index=num_cols.index(preferred_default) if preferred_default in num_cols else 0,
                        key="dist__col",
                    )
                    
                    # ✅ CRITICAL FIX: Reset zakresu przy zmianie miary
                    _prev_dist_col = st.session_state.get("_dc_prev_dist_col")
                    if _prev_dist_col is not None and dist_col != _prev_dist_col:
                        # Zmiana miary → wyczyść stary zakres i cache interpretacji
                        for key_to_clear in ["dist__range", "dc_dist_range", "datachat_last_render_key", "datachat_interp"]:
                            if key_to_clear in st.session_state:
                                del st.session_state[key_to_clear]
                    st.session_state["_dc_prev_dist_col"] = dist_col
                    st.session_state["dc_dist_col"] = dist_col

                # Zakres dla wybranej miary
                dist_range = None
                if dist_col:
                    s = pd.to_numeric(df[dist_col], errors="coerce").dropna()
                    if not s.empty:
                        lo, hi = float(s.min()), float(s.max())
                        
                        # ✅ Pobierz poprzednią wartość lub użyj pełnego zakresu
                        _prev_range = st.session_state.get("dist__range", (lo, hi))
                        # Walidacja: czy poprzedni zakres mieści się w nowym
                        if not (lo <= _prev_range[0] <= hi and lo <= _prev_range[1] <= hi):
                            _prev_range = (lo, hi)
                        
                        dist_range = st.slider(
                            "Zakres wartości",
                            min_value=lo,
                            max_value=hi,
                            value=_prev_range,
                            key="dist__range",
                        )
                        st.session_state["dc_dist_range"] = dist_range

                # Kolorowanie
                color_col = None
                _segment_candidates = _distribution_segment_candidates(dist_col)
                _color_opts = ["(brak)"] + _segment_candidates
                _default_color = st.session_state.get("dc_color_col")
                if not _default_color:
                    # heurystyka: szukaj "rok"/"year" — bez hardcode nazwy kolumny
                    for _c in _segment_candidates:
                        if "rok" in str(_c).lower() or "year" in str(_c).lower():
                            _default_color = _c
                            break
                _color_i = _color_opts.index(_default_color) if (_default_color in _color_opts) else 0

                color_opt = st.selectbox(
                    "Koloruj słupki wg",
                    options=_color_opts,
                    index=_color_i,
                    key="dist__color_col",
                    help="Kolory są widoczne w legendzie oraz w tooltipie.",
                )
                st.session_state["dc_color_col"] = color_opt
                if color_opt != "(brak)":
                    color_col = color_opt

                # Filtr kategorii (opcjonalny)
                filt_col = st.selectbox(
                    "Filtr kategorii (opcjonalnie)",
                    options=["(brak)"] + _segment_candidates,
                    index=(["(brak)"] + _segment_candidates).index(st.session_state.get("dc_filt_col"))
                    if st.session_state.get("dc_filt_col") in (["(brak)"] + _segment_candidates)
                    else 0,
                    key="dist__filt_col",
                )
                st.session_state["dc_filt_col"] = filt_col

                filt_values = []
                if filt_col != "(brak)" and (filt_col in df.columns):
                    vals = df[filt_col].dropna().unique().tolist()
                    vals = sorted(vals, key=lambda x: str(x))[:200]
                    _prev = st.session_state.get("dc_filt_vals") or []
                    if not isinstance(_prev, (list, tuple, set)):
                        _prev = []
                    _prev = [v for v in list(_prev) if v in set(vals)]
                    filt_values = st.multiselect(
                        "Wartości",
                        options=vals,
                        default=_prev,
                        key="dist__filt_vals",
                    )
                st.session_state["dc_filt_vals"] = list(filt_values) if filt_values else []

                # Filtr czasu (jeśli mamy)
                date_cols = list(schema_ctx.get("date_cols") or [])
                date_col = None
                date_range = None
                if date_cols:
                    date_col = st.selectbox(
                        "Kolumna czasu",
                        options=["(auto)"] + date_cols,
                        index=0,
                        key="dist__date_col",
                    )
                    chosen = date_cols[0] if date_col == "(auto)" else date_col
                    dt = pd.to_datetime(df[chosen].head(200000), errors="coerce")
                    if dt.notna().any():
                        dmin, dmax = dt.min().date(), dt.max().date()
                        date_range = st.date_input(
                            "Zakres dat",
                            value=(dmin, dmax),
                            key="dist__date_range",
                        )
                    st.session_state["dc_date_col"] = date_col
                    st.session_state["dc_date_range"] = date_range

            # ══════════════════════════════════════════
            # ══════════════════════════════════════════

            # ══════════════════════════════════════════
            # B) COMPOSITION OVER TIME panel (keys: cot__*)
            # ══════════════════════════════════════════
            elif _active_intent in ("composition_over_time", "cot"):

                date_cols = list(schema_ctx.get("date_cols") or [])

                # Wybór kategorii (opcjonalnie) — domyślnie auto (COT ma też własną inferencję)
                # Wybór kategorii (opcjonalnie) — domyślnie auto (COT ma też własną inferencję)
                # ✅ Guard: pokazuj WYŁĄCZNIE sensowne kolumny kategoryczne (bez liczbowych/numeric-like i datetime)
                def _looks_numeric_like(_s: pd.Series, sample_n: int = 5000) -> bool:
                    try:
                        if pd.api.types.is_numeric_dtype(_s):
                            return True
                        if pd.api.types.is_datetime64_any_dtype(_s):
                            return False
                        # object/string: try fast numeric coercion on a sample
                        ss = _s.dropna().head(sample_n)
                        if ss.empty:
                            return False
                        nn = pd.to_numeric(ss, errors="coerce")
                        ok = float(nn.notna().mean())
                        # if almost everything parses as number -> treat as numeric-like
                        return ok >= 0.95
                    except Exception:
                        return False

                _cot_cat_opts = ["(auto)"]
                for _c in df.columns:
                    if _c in (date_cols or []):
                        continue
                    try:
                        s = df[_c]
                        # hard reject: datetime, numeric dtype or numeric-like strings
                        if pd.api.types.is_datetime64_any_dtype(s) or _looks_numeric_like(s):
                            continue
                        # reject extremely high-cardinality (usually IDs/descriptions)
                        nun = int(s.nunique(dropna=True)) if len(s) else 0
                        if nun > 5000:
                            continue
                    except Exception:
                        continue
                    _cot_cat_opts.append(_c)

                _cot_cat_default = st.session_state.get("cot__cat_col", "(auto)")
                _cot_cat_i = _cot_cat_opts.index(_cot_cat_default) if _cot_cat_default in _cot_cat_opts else 0
                cot_cat_col = st.selectbox(
                    "Kategoria",
                    options=_cot_cat_opts,
                    index=_cot_cat_i,
                    key="cot__cat_col",
                    help="Wybierz kolumnę kategoryczną. (auto) użyje najlepiej pasującej z danych.",
                )

                cot_top_n = st.slider(
                    "Top-N kategorii",
                    min_value=3,
                    max_value=20,
                    value=int(st.session_state.get("cot__top_n", 10)),
                    step=1,
                    key="cot__top_n",
                )

                cot_include_other = st.checkbox(
                    "Dodaj 'Other' (reszta kategorii)",
                    value=bool(st.session_state.get("cot__include_other", True)),
                    key="cot__include_other",
                )

                # Tryb (wartość vs udziały) wynika z pytania — nie z sidebaru.


            # ══════════════════════════════════════════
            # C) COMPOSITION STATIC panel (keys: cs__*)
            # ══════════════════════════════════════════
            elif _active_intent in ("composition", "composition_static"):

                # scoring: preferuj sensowne kategorie biznesowe
                # (żadne hardcode nazwy kolumn — działa na dowolnych danych)
                def _cs_is_time_like(col: str) -> bool:
                    cl = col.lower()
                    return any(tok in cl for tok in ("date", "data", "time", "rok", "mies"))

                def _cs_score_col(name: str) -> int:
                    n = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
                    score = 0
                    for kw, w in [
                        ("category", 120), ("kategoria", 120),
                        ("subcategory", 100), ("podkategoria", 100),
                        ("segment", 90), ("subcat", 80),
                        ("brand", 70), ("marka", 70),
                        ("productname", 60), ("nazwa", 60),
                        ("product", 40), ("item", 40),
                        ("sku", 30), ("code", 15), ("kod", 15),
                    ]:
                        if kw in n:
                            score += w
                    if any(x in n.split() for x in ["id", "uuid", "guid"]):
                        score -= 20
                    if _cs_is_time_like(name):
                        score -= 200
                    return score

                candidate_cols = composition_static.get_groupable_columns(df)
                if not candidate_cols:
                    candidate_cols = composition_static.get_groupable_columns(df, max_unique=2000)
                if not candidate_cols:
                    candidate_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns[:10]) or list(df.columns[:10])
                _cs_default = candidate_cols[0] if candidate_cols else None

                # ── Kategoria 1 (segment główny)
                _prev_g1 = st.session_state.get("cs__group_col")
                _g1_idx = (
                    candidate_cols.index(_prev_g1)
                    if (_prev_g1 and _prev_g1 in candidate_cols)
                    else (candidate_cols.index(_cs_default) if (_cs_default and _cs_default in candidate_cols) else 0)
                )
                cs_group_col = st.selectbox(
                    "Kategoria 1 (segment główny)",
                    options=candidate_cols,
                    index=_g1_idx,
                    key="cs__group_col",
                )

                # ── Kategoria 2 (opcjonalnie)
                other_cols = ["(brak)"] + [c for c in candidate_cols if c != cs_group_col]
                _prev_g2 = st.session_state.get("cs__group_col2", "(brak)")
                _g2_idx = other_cols.index(_prev_g2) if (_prev_g2 in other_cols) else 0
                cs_group_col2 = st.selectbox(
                    "Kategoria 2 (opcjonalnie)",
                    options=other_cols,
                    index=_g2_idx,
                    key="cs__group_col2",
                )

                # ── TOP-N
                cs_top_n = st.slider(
                    "TOP-N (Treemap / Pareto / Mix)",
                    min_value=5,
                    max_value=50,
                    value=int(st.session_state.get("cs__top_n", 10)),
                    step=1,
                    key="cs__top_n",
                )

                # ── Pareto cutoff
                cs_cutoff = st.slider(
                    "Pareto cutoff (np. 80%)",
                    min_value=0.50,
                    max_value=0.95,
                    value=float(st.session_state.get("cs__cutoff", 0.80)),
                    step=0.05,
                    key="cs__cutoff",
                )

                
                # ── Kolumna ceny (korytarz cenowy) — opcjonalnie
                _cs_price_exclude = {cs_group_col}
                if cs_group_col2 and cs_group_col2 != "(brak)":
                    _cs_price_exclude.add(cs_group_col2)
                price_candidates = composition_static.get_price_candidate_columns(df, exclude=_cs_price_exclude)
                price_opts = ["(auto)"] + price_candidates
                _prev_pc = st.session_state.get("cs__price_col", "(auto)")
                _pc_idx = price_opts.index(_prev_pc) if (_prev_pc in price_opts) else 0
                cs_price_col = st.selectbox(
                    "Kolumna ceny (korytarz cenowy)",
                    options=price_opts,
                    index=_pc_idx,
                    key="cs__price_col",
                    help="Lista pokazuje tylko wiarygodne kandydaty ceny. "
                         "Przy (auto) automatycznie użyjemy najlepszej znalezionej kolumny ceny.",
                )
                _auto_price = None
                if cs_price_col == "(auto)" and price_candidates:
                    _auto_price = price_candidates[0]
                    st.caption(f"Auto: użyjemy kolumny `{_auto_price}`.")
                elif cs_price_col == "(auto)":
                    st.caption("Auto: nie znaleziono wiarygodnej kolumny ceny dla korytarza cenowego.")
                # ── Guardrail: duża kardynalność
                try:
                    _nuniq = int(df[cs_group_col].nunique(dropna=True)) if (cs_group_col and cs_group_col in df.columns) else 0
                    if (len(df) >= 500_000) and (_nuniq >= 50_000):
                        st.warning(
                            f'Kolumna "{cs_group_col}" ma {_nuniq} unikalnych wartości. '
                            "Rozważ mniejsze TOP-N lub inną kolumnę."
                        )
                except Exception:
                    pass

                # CS panel nie generuje row-filtrów (dist_col/filt_col/date_col)
                # → ustawia je na None żeby _apply_df_filters nie mieszała
                dist_col = None
                dist_range = None
                color_col = None
                filt_col = None
                filt_values = []
                date_col = None
                date_range = None
                date_cols = []

            # ══════════════════════════════════════════
            # C) Fallback — jeszcze nie ustalono intent
            # ══════════════════════════════════════════
            else:
                st.info("Filtry zostaną pokazane po pierwszej analizie.")
                dist_col = None
                dist_range = None
                color_col = None
                filt_col = None
                filt_values = []
                date_col = None
                date_range = None
                date_cols = []
                cs_group_col = None
                cs_group_col2 = "(brak)"
                cs_top_n = 10
                cs_cutoff = 0.80

        # CS debug UI removed in production; checkpoints remain internal.

        # ─────────────────────────────────────────────
        # Lektor (Data Chat) — tylko ustawienia lektora
        # ─────────────────────────────────────────────
        with st.expander("🔊 Lektor (Data Chat)", expanded=False):
            st.checkbox(
                "✅ Włącz lektora (TTS)",
                value=bool(st.session_state.get("dc_tts_enabled", False)),
                key="dc_tts_enabled",
            )

            dc_tts_gender = st.radio(
                "Głos",
                ["Kobieta", "Mężczyzna"],
                index=0,
                key="dc_tts_gender",
            )

            # --- TTS voices (tylko wspierane przez API; filtr po płci) ---
            _FEMALE_VOICES = ["shimmer", "nova", "coral", "sage", "marin"]
            _MALE_VOICES   = ["alloy", "onyx", "echo", "verse", "ash", "ballad", "cedar", "fable"]

            allowed_voices = _FEMALE_VOICES if dc_tts_gender == "Kobieta" else _MALE_VOICES

            # jeśli w session_state jest voice spoza allowed (np. po zmianie płci) -> ustawiamy DEFAULT *przed* widgetem
            if st.session_state.get("dc_tts_voice") not in allowed_voices:
                st.session_state["dc_tts_voice"] = allowed_voices[0]

            dc_tts_voice = st.selectbox(
                "OpenAI voice",
                options=allowed_voices,
                index=allowed_voices.index(st.session_state.get("dc_tts_voice", allowed_voices[0])),
                key="dc_tts_voice",
            )

            st.selectbox(
                "OpenAI TTS model",
                options=["gpt-4o-mini-tts"],
                index=0,
                key="dc_tts_model",
                on_change=_dc_tts_bump_cache,
            )

            st.checkbox(
                "☑️ Automatycznie uruchamiaj analizę po nagraniu pytania",
                value=bool(st.session_state.get("dc_tts_auto_run", True)),
                key="dc_tts_auto_run",
            )

        # Distribution debug sidebar hidden in production; checkpoints stay internal only.
        st.session_state.pop("_dist_debug_sidebar_slots", None)

    # ── safe defaults for keys not set by the active panel branch
    if _active_intent == "distribution":
        cs_group_col    = st.session_state.get("cs__group_col")
        cs_group_col2   = st.session_state.get("cs__group_col2", "(brak)")
        cs_top_n        = int(st.session_state.get("cs__top_n", 10))
        cs_cutoff       = float(st.session_state.get("cs__cutoff", 0.80))

    return {
        # ── Distribution
        "dist_col":     dist_col,
        "dist_range":   dist_range,
        "color_col":    color_col,
        "filt_col":     None if (not filt_col or filt_col == "(brak)") else filt_col,
        "filt_values":  filt_values,
        "date_col":     None if (not date_cols) else (date_cols[0] if date_col in (None, "(auto)") else date_col),
        "date_range":   date_range,
        # ── Composition Static (passthrough config, nie row-filtry)
        "cs_group_col":  cs_group_col,
        "cs_group_col2": cs_group_col2 if cs_group_col2 != "(brak)" else None,
        "cs_top_n":      cs_top_n,
        "cs_cutoff":     cs_cutoff,
        "cs_price_col": None if (not cs_price_col or cs_price_col == "(auto)") else cs_price_col,
        # ── Composition Over Time (passthrough config)
        "cot_cat_col": None if (not cot_cat_col or cot_cat_col == "(auto)") else cot_cat_col,
        "cot_top_n": cot_top_n,
        "cot_include_other": bool(cot_include_other),
    }

def _dc_df_fingerprint(df: pd.DataFrame) -> tuple:
    """Lekki fingerprint DF — do resetu cache, gdy użytkownik podmieni dataset."""
    try:
        return (len(df), tuple(df.columns), tuple(str(x) for x in df.dtypes))
    except Exception:
        return (id(df),)


def _dc_filters_signature(filters: Dict[str, Any]) -> tuple:
    """Stabilny podpis filtrów — klucz do cache."""
    if not isinstance(filters, dict):
        return ("_no_filters_",)

    def _t(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return x

    return (
        str(filters.get("filt_col") or ""),
        tuple(filters.get("filt_values") or []),
        str(filters.get("date_col") or ""),
        _t(filters.get("date_range")),
        str(filters.get("dist_col") or ""),
        _t(filters.get("dist_range")),
        # ── Composition Static (cache musi się łamać przy zmianie parametrów CS)
        str(filters.get("cs_group_col") or ""),
        str(filters.get("cs_group_col2") or ""),
        int(filters.get("cs_top_n") or 10),
        float(filters.get("cs_cutoff") or 0.80),
        str(filters.get("cs_price_col") or ""),
    )


def _apply_df_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Szybkie filtrowanie (globalne) z cache:
    - cache maski po (df_fingerprint + filters_signature)
    - cache konwersji datetime dla date_col (robione raz na dataset)
    """
    if df is None or df.empty or not isinstance(filters, dict):
        return df

    # Fast path: Composition Static and część rerunów nie mają realnych filtrów wierszy,
    # tylko ustawienia renderu wykresów. Wtedy nie buduj pełnej maski True ani kopii df.
    fc = filters.get("filt_col")
    fv = filters.get("filt_values")
    dc = filters.get("date_col")
    dr = filters.get("date_range")
    has_category_filter = bool(fc and fv and fc in df.columns)
    has_date_filter = bool(dc and dr and dc in df.columns and isinstance(dr, (list, tuple)) and len(dr) == 2)
    if not has_category_filter and not has_date_filter:
        return df

    # ── init / reset cache, gdy zmienił się dataset
    df_fp = _dc_df_fingerprint(df)
    if st.session_state.get("datachat_df_fp") != df_fp:
        st.session_state["datachat_df_fp"] = df_fp
        st.session_state["datachat_filter_mask_cache"] = {}   # (sig)->mask
        st.session_state["datachat_col_cache"] = {}           # (col,kind)->Series

    mask_cache = st.session_state.get("datachat_filter_mask_cache") or {}
    col_cache = st.session_state.get("datachat_col_cache") or {}

    sig = _dc_filters_signature(filters)

    # ── cache hit
    if sig in mask_cache:
        try:
            m = mask_cache[sig]
            # m może być numpy/pandas bool-array
            return df.loc[m]
        except Exception:
            pass

    # ── budowa maski (bez kopiowania df)
    out_mask = pd.Series(True, index=df.index)

    # 1) filtrowanie kategorii
    if fc and fv and fc in df.columns:
        try:
            out_mask &= df[fc].isin(fv)
        except Exception:
            pass

    # 2) filtrowanie daty (range) — KONWERSJA RAZ, potem reuse
    if dc and dr and dc in df.columns and isinstance(dr, (list, tuple)) and len(dr) == 2:
        try:
            # dtype datetime? użyj bez konwersji
            if pd.api.types.is_datetime64_any_dtype(df[dc]):
                sdt = df[dc]
            else:
                ck = (dc, "datetime")
                sdt = col_cache.get(ck)
                if sdt is None or len(sdt) != len(df):
                    sdt = pd.to_datetime(df[dc], errors="coerce", cache=True)
                    col_cache[ck] = sdt

            lo = pd.to_datetime(dr[0], errors="coerce")
            hi = pd.to_datetime(dr[1], errors="coerce")
            out_mask &= (sdt >= lo) & (sdt <= hi)
        except Exception:
            pass

    # zapisz cache (ważne: values -> szybkie i lekkie w loc)
    try:
        mask_cache[sig] = out_mask.values
        st.session_state["datachat_filter_mask_cache"] = mask_cache
        st.session_state["datachat_col_cache"] = col_cache
    except Exception:
        pass

    return df.loc[out_mask.values]

def _plan_chart_placeholder(question: str) -> Dict[str, Any]:
    # Refactor-only: jeśli w Twojej wersji chart_spec powstaje wcześniej,
    # router go pobiera z session_state. Jeśli nie ma – minimalny fallback.
    return {"intent": "distribution", "primary_chart": {"chart_type": "histogram+kde", "x": None}}

def _transcribe_audio_to_text(audio_bytes: bytes, mime_type: str = "audio/wav") -> tuple[str, str]:
    """
    Bezpieczna transkrypcja audio -> tekst.
    Działa tylko jeśli masz: openai + OPENAI_API_KEY.
    Zwraca: (text, error_msg). Gdy OK: (tekst, "").
    """
    if not audio_bytes:
        return "", "Brak danych audio."
    if OpenAI is None:
        return "", "Brak biblioteki `openai` w środowisku. Zainstaluj: pip install openai"
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "", "Brak OPENAI_API_KEY w zmiennych środowiskowych."

    # szybki guard: nie transkrybuj „kliknięć”
    try:
        with contextlib.closing(wave.open(io.BytesIO(audio_bytes))) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            duration = frames / float(rate)
        if duration < 0.15:
            return "", f"Nagranie zbyt krótkie ({duration:.3f}s)."
    except Exception:
        pass

    stt_model = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    try:
        client = OpenAI(api_key=api_key)
        f = io.BytesIO(audio_bytes)
        f.name = "datachat_input.wav"
        resp = client.audio.transcriptions.create(
            model=stt_model,
            file=f,
            language="pl",  # ✅ wymusza polski (eliminuje “zostawanie” w RU)
            prompt="Transkrybuj po polsku. Zachowaj oryginalne brzmienie i interpunkcję.",
        )
        text = (resp.text or "").strip()
        if not text:
            return "", "Model nie zwrócił tekstu."
        return text, ""
    except Exception as e:
        return "", f"OpenAI STT error: {e}"

def _deterministic_questions(has_time: bool) -> list[str]:
    """
    10 pytań deterministycznych (bez PRO/EXPERT, bez what-if).
    Filtrujemy te 'w czasie', jeśli brak czasu w danych.
    """
    base = [
        # DISTRIBUTION
        "Pokaż rozkład wartości (histogram + statystyki) dla kluczowej miary.",

        # COMPOSITION (statyczne)
        "Pokaż sumaryczną wartość sprzedaży i per kategoria.",

        # COMPOSITION (over time) — tylko jeśli mamy czas
        "Pokaż sprzedaż per kategorię w czasie (cały okres)",
        "Pokaż sprzedaż per kategorię w czasie od początku roku do dziś",
        "Pokaż sprzedaż per kategorię w czasie od ___ do ___",

        "Pokaż udziały per kategorię w czasie (cały okres)",
        "Pokaż udziały per kategorię w czasie od początku roku do dziś",
        "Pokaż udziały per kategorię w czasie od ___ do ___",

        # COMPARISON (among items)
        "Porównaj kategorie: kto jest liderem, kto odstaje?",

        # RELATIONSHIP
        "Czy istnieje zależność między dwiema zmiennymi (korelacja / trend)?",

        # QUALITY / SANITY
        "Jakie są braki danych i które kolumny są najbardziej problematyczne?",
        "Czy są duplikaty / anomalia w kluczowych kolumnach?",

        # SEGMENTATION/CLUSTER (wejście do osobnego zestawu później, ale pytanie musi istnieć)
        "Pokaż wielkość segmentów/klastrów (liczebność) oraz ich udział.",
        "Opisz cechy wyróżniające segmenty/klastry i zaproponuj ich nazwy.",
    ]

    over_time = [
        "Pokaż wartość sprzedaży per kategorię w czasie (miesiąc-rok).",
        "Pokaż udziały per kategorię w czasie (miesiąc-rok).",
    ]

    if has_time:
        # podmieniamy dwa „quality/sanity” na dwa „over time”, żeby nadal było 10
        base = base[:-2] + over_time
    return base


# def _ai_questions_placeholder() -> list[str]:
#     """
#     Na teraz: placeholder pod LLM. Maks 10.
#     UI oznacza tę sekcję jako AI, więc treść pytań NIE powinna mieć prefiksu "AI:".
#     (Implementację generatora zrobimy w kolejnym kroku.)
#     """
#     return [
#         "Jakie 3 kategorie najmocniej napędzają wynik i dlaczego?",
#         "Wskaż nietypowe zmiany w czasie i możliwe przyczyny.",
#         "Co jest największym ryzykiem interpretacji (bias/quality)?",
#         "placeholder question 4",
#         "placeholder question 5",
#         "placeholder question 6",
#         "placeholder question 7",
#         "placeholder question 8",
#         "placeholder question 9",
#         "placeholder question 10",
#     ]


def main() -> None:
    # --- Nawigacja jak w Etapie 1/2 (góra) ---
    hide_default_multipage_nav()
    render_flow_nav(current_id="03_Data_Chat", key_prefix="flow_top")

    # ✅ MUST: wiring LLM dla Executive Takeaway (global, bezwarunkowo, na początku runu)
    _dc_register_exec_takeaway_llm_callable()

    # --- Tytuł + opis Etapu 3 ---
    st.title("Data Chat — inteligentna eksploracja danych (Etap 3)")
    st.markdown(
        """
        <style>
        /* Szare tło expandera "Kolumny" + delikatna ramka (czytelne odseparowanie) */
        div[data-testid="stExpander"] details {
            background: #f4f6f8;
            border: 1px solid #e6e8eb;
            border-radius: 10px;
            padding: 6px 10px;
        }
        div[data-testid="stExpander"] summary {
            padding: 2px 0px;
        }

        /* Zmniejsz pionowe odstępy nagłówków w sekcji dyktowania */
        .dc-tight h4, .dc-tight p, .dc-tight span, .dc-tight b {
            margin-top: 0.15rem !important;
            margin-bottom: 0.25rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────
    # PATCH B1: Executive Takeaway – NEVER truncate with "(...)" in UI
    # Streamlit captions / some containers may apply single-line clamp/ellipsis.
    # We force normal wrapping + disable ellipsis.
    # ─────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* Force wrapping everywhere where Streamlit may clamp text */
        div[data-testid="stCaptionContainer"] p,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stText"] p,
        div[data-testid="stText"] span {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: break-word !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "Etap 3 pozwala zadawać pytania o dane w języku naturalnym i otrzymać spójne wizualizacje. "
        "Zadaj pytanie czego chciałbyś się dowiedzieć, a otrzymasz analizę i wnioski według najwyższych standardów"
    )

    # --- Co dostajesz (bez expandera; jak Etap 2) ---
    st.markdown("**Co dostajesz:**")
    st.markdown(
    """
- szybkie pytania startowe dopasowane do danych (deterministyczne + AI),
- spójne wizualizacje zgodne z zasadami Abeli / Gestalt / McKinsey,
- interpretację i narrację w zakładce „Obraz całości”,
- historię pytań i ocenę odpowiedzi.
""".strip()
)

    st.markdown("---")


    df = _get_df_from_session()

    # ─────────────────────────────────────────────
    # 🛡️ SAFE FRONTEND MODE (Stage3)
    # Prevent sending too much raw data to the browser (Altair/Plotly JSON overload)
    # Works universally for any dataset.
    # ─────────────────────────────────────────────
    _df_rows = int(len(df)) if df is not None else 0
    with st.sidebar.expander("🛡️ SAFE FRONTEND MODE (Stage3)", expanded=False):
        safe_enabled = st.checkbox(
            "Włącz SAFE FRONTEND MODE (auto >100k)",
            value=bool(st.session_state.get("stage3_safe_frontend_enabled", True)),
            key="stage3_safe_frontend_enabled",
        )
        sample_rows = st.number_input(
            "Maks. liczba rekordów do wykresów (sampling)",
            min_value=500,
            max_value=80_000,
            value=int(st.session_state.get("stage3_safe_frontend_sample_rows", 5_000)),
            step=500,
            key="stage3_safe_frontend_sample_rows",
        )

        cfg = compute_safe_frontend_config(
            df_rows=_df_rows,
            enabled=bool(safe_enabled),
            sample_rows=int(sample_rows),
        )
        # Keep the effective config available for all branches.
        st.session_state["stage3_safe_frontend_cfg_v1"] = cfg.to_dict()

        if cfg.effective:
            st.info(
                f"SAFE FRONTEND MODE jest aktywny: **{cfg.sample_rows:,}** rekordów do wykresów.\n\n"
                f"enabled={cfg.enabled} | forced={cfg.forced} | reason={cfg.reason} | rows={cfg.df_rows:,}".replace(",", " ")
            )
        else:
            st.caption(
                f"SAFE FRONTEND MODE wyłączony (rows={cfg.df_rows:,}).".replace(",", " ")
            )

    # Hidden debug flag: handoff diagnostics stay available via session_state / external hooks.
    dbg_handoff = bool(st.session_state.get("dc_debug_handoff_v1", False))
    if dbg_handoff:
        latest = st.session_state.get("latest_artifacts") or st.session_state.get("dc_latest_artifacts") or st.session_state.get("LATEST_ARTIFACTS")
        # Note: Stage2 writes latest_artifacts dict; keep backward-compat with older keys.
        parquet_path = None
        if isinstance(latest, dict):
            parquet_path = latest.get("parquet_path") or latest.get("ready_parquet") or latest.get("ready_path")
        meta_rows = None
        if parquet_path:
            try:
                import pyarrow.parquet as pq  # type: ignore
                meta_rows = int(pq.ParquetFile(parquet_path).metadata.num_rows)
            except Exception:
                meta_rows = None
        # best-effort time span (for Stage2→Stage3 debug)
        # IMPORTANT: do NOT pick numeric IDs like "Invoice" that can be coerced into years 2049+.
        time_span = None
        try:
            schema_ctx = (st.session_state.get("datachat_schema_ctx") or {})
            cand = schema_ctx.get("date_cols") or []
            # fallback: name-based candidates (but WITHOUT privileging invoice IDs)
            if not cand:
                cand = [c for c in df.columns if ("date" in c.lower() or "time" in c.lower())]

            now_y = datetime.datetime.now().year
            best = None
            for dt_col in cand:
                if dt_col not in df.columns:
                    continue
                s = pd.to_datetime(df[dt_col], errors="coerce")
                s = s.dropna()
                if s.empty:
                    continue
                y_med = float(s.dt.year.median())
                if y_med < 1900 or y_med > (now_y + 2):
                    continue
                best = (dt_col, s.min(), s.max())
                break

            if best is not None:
                time_span = {"col": best[0], "min": str(best[1]), "max": str(best[2])}
        except Exception:
            time_span = None
        st.sidebar.markdown("**Handoff diagnostics**")
        st.sidebar.write({"parquet_path": parquet_path, "parquet_rows": meta_rows, "df_rows_loaded": int(len(df)), "time_span": time_span})
    # ── Global debug flag for branches (Stage3)
    # Used across branches to reveal QA/diagnostics (e.g., timing tables) without affecting normal UI.
    if "dc_debug" not in st.session_state:
        st.session_state["dc_debug"] = False
    if "cs_debug_enabled" not in st.session_state:
        st.session_state["cs_debug_enabled"] = False
    if "cs_debug_checkpoints" not in st.session_state:
        st.session_state["cs_debug_checkpoints"] = False
    if "cs_debug_show_fallbacks" not in st.session_state:
        st.session_state["cs_debug_show_fallbacks"] = False
    st.session_state["cs_debug_enabled"] = False
    st.session_state["cs_debug_checkpoints"] = False
    st.session_state["cs_debug_show_fallbacks"] = False
    st.session_state["cs_internal_checkpoints_enabled"] = True
    if "cs_debug_log" not in st.session_state:
        st.session_state["cs_debug_log"] = []
    st.session_state["dist_debug_enabled"] = False
    st.session_state["dist_debug_checkpoints"] = False
    st.session_state["dist_debug_show_fallbacks"] = False
    st.session_state["dist_debug_ui_enabled"] = False
    st.session_state.pop("_dist_debug_sidebar_slots", None)
    st.session_state["dist_internal_checkpoints_enabled"] = True
    if "dist_debug_log" not in st.session_state:
        st.session_state["dist_debug_log"] = []


    
    # ─────────────────────────────────────────────
    # KROK 1 — Status danych + preview kolumn (MUST)
    # ─────────────────────────────────────────────
    src = _data_source_info()

    col_a, col_b = st.columns([2, 1], vertical_alignment="center")
    with col_a:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            st.error("Brak danych do analizy. Uruchom Etap 1–2 lub wskaż poprawny zestaw danych.")
            return
        st.success(f"Dane wczytane ✅  |  wiersze: {len(df):,}  |  kolumny: {df.shape[1]:,}")

    with col_b:
        st.caption(f"Źródło: **{src['source']}**")
        if src.get("path"):
            st.caption(f"Plik: `{src['path']}`")

    with st.expander(f"📚 Kolumny ({df.shape[1]}) — podgląd", expanded=False):
        q = st.text_input("Szukaj kolumny…", value="", key="dc_cols_search")

        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if (c not in dt_cols and c not in num_cols)]

        st.caption(f"🧾 Daty: {len(dt_cols)}  |  🔢 Numeryczne: {len(num_cols)}  |  🏷️ Kategoryczne: {len(cat_cols)}")

        def _filter(cols):
            if not q:
                return cols
            qq = q.lower().strip()
            return [c for c in cols if qq in str(c).lower()]

        st.markdown("**Daty / czas**")
        st.write(_filter(dt_cols) if dt_cols else "—")

        st.markdown("**Numeryczne**")
        st.write(_filter(num_cols) if num_cols else "—")

        st.markdown("**Kategoryczne / tekst**")
        st.write(_filter(cat_cols) if cat_cols else "—")

    st.markdown("")


    schema_ctx = _get_schema_ctx_from_session()

    # ─────────────────────────────────────────────
    # PRE-SIDEBAR: resolvuj intent z tego co mamy w session_state
    # PRZED renderem sidebaru — żeby panel przełączał się od PIERWSZEGO kliknięcia.
    #
    # Logika lustrzana do routing block (dalej w kodzie), ale:
    #   - prompt: z session_state (ustawiony gdy user wpisał/kliknął chip)
    #   - chart_spec: z session_state (z poprzedniej analizy lub placeholder)
    #   - cp0_struct: NIE dostępny jeszcze → pomijamy, nadrobione przez keyword heuristic
    # ─────────────────────────────────────────────
    # --- safety defaults (na wypadek dziwnych rerunów / pustych session keys)
    _pre_prompt = _get_persisted_datachat_prompt()
    _pre_chart_spec = _get_chart_spec_from_session() or {}
    _pre_intent = _guess_datachat_intent_from_prompt(
        _pre_prompt,
        str(_pre_chart_spec.get("intent") or st.session_state.get("datachat_active_intent", "distribution") or "distribution"),
    )

    st.session_state["datachat_active_intent"] = _pre_intent

    # ─────────────────────────────────────────────
    # Sidebar: filtry — panel przełączany wg intent
    df_source = df
    _filters_live = _render_sidebar_filters(df_source, schema_ctx)

    # --- Sugestie pytań (nad polem) ---
    has_time = bool(schema_ctx.get("date_cols") or [])
    if not has_time:
        try:
            has_time = bool(cp0_detect_has_time(df_source.head(5000), schema_ctx))
        except Exception:
            has_time = False

    st.subheader("Sprawdź, o co możesz mnie zapytać", divider="gray")
    st.markdown(
        "Poniżej znajdziesz przykładowe pytania automatycznie dopasowane do struktury danych "
        "(typy kolumn, target, segmenty, itp.). "
        "**Kliknij w dowolne pytanie**, aby przenieść je do pola tekstowego poniżej."
    )

    st.markdown("##### Szybkie pytania (deterministyczne)")
    det_q = _deterministic_questions(has_time=has_time)

    # chipsy: 2 kolumny (czytelnie, bez poziomów)
    chip_cols = st.columns(2)
    for i, q in enumerate(det_q[:10]):
        with chip_cols[i % 2]:
            st.button(
                q,
                key=f"det_q_{i}",
                width='stretch',
                on_click=_sync_prompt_and_sidebar_intent,
                args=(q,),
            )

    # # st.markdown("---")
    # st.markdown("##### ⚡ Pytania od AI")
    # ai_q = _ai_questions_placeholder()
    # ai_cols = st.columns(2)
    # for i, q in enumerate(ai_q[:10]):
    #     q_clean = str(q).strip()
    #     if q_clean.lower().startswith("ai:"):
    #         q_clean = q_clean.split(":", 1)[1].strip()

    #     with ai_cols[i % 2]:
    #         if st.button(q_clean, key=f"ai_q_{i}", width='stretch'):
    #             st.session_state["datachat_analysis_running"] = False
    #             st.session_state["datachat_prompt"] = q_clean
    #             st.rerun()

    st.subheader("Twoje pytanie o dane", divider="gray")

    # --- Mikrofon + textbox (układ jak w dawcy) ---
    if "datachat_prompt_input" not in st.session_state:
        _set_persisted_datachat_prompt(_get_persisted_datachat_prompt())

    analysis_running: bool = bool(st.session_state.get("datachat_analysis_running", False))

    col_mic, col_text = st.columns([1, 9])

    with col_mic:
        st.markdown(
            """
            <span title="Naciśnij mikrofon, powiedz pytanie i przestań mówić.
            Nagrywanie zatrzyma się po chwili ciszy.">
                <b>🎙️ Dyktuj</b>
            </span>
            """,
            unsafe_allow_html=True,
        )

        audio_bytes = None
        if audio_recorder is None:
            st.caption("Brak audio_recorder_streamlit")
        else:
            audio_bytes = audio_recorder(
                text="Kliknij:",
                recording_color="#ff4b4b",
                neutral_color="#dddddd",
                icon_name="microphone",
                icon_size="4x",
            )

    # STT: transkrypcja PRZED textboxem — TYLKO gdy nagranie jest nowe
    if audio_bytes and not analysis_running:
        # sygnatura: długość + początek (wystarczy, żeby wykryć "to samo nagranie")
        sig = (len(audio_bytes), audio_bytes[:24])
        last_sig = st.session_state.get("dc_last_audio_sig")

        if sig != last_sig:
            st.session_state["dc_last_audio_sig"] = sig
            txt, err = _transcribe_audio_to_text(audio_bytes=audio_bytes, mime_type="audio/wav")
            if err:
                st.warning(f"Nie udało się przetranskrybować: {err}")
            elif txt:
                _sync_prompt_and_sidebar_intent(txt)

    with col_text:
        st.markdown("**Zapisz pytanie**")
        st.text_area(
            "Treść pytania",
            key="datachat_prompt_input",
            height=80,
            label_visibility="collapsed",
            placeholder="Wpisz pytanie lub użyj mikrofonu po lewej…",
            on_change=_sync_prompt_and_sidebar_intent,
        )

    # Dolna linia: Run / Stop
    live_prompt = (st.session_state.get("datachat_prompt_input") or "").strip()
    st.session_state["datachat_prompt_saved"] = live_prompt
    st.session_state["datachat_prompt"] = live_prompt
    st.session_state["datachat_last_question"] = live_prompt

    col_run, col_stop = st.columns([7, 3])
    with col_run:
        run_clicked = st.button(
            "🔍 Przeanalizuj pytanie",
            disabled=(not bool(live_prompt)) or analysis_running,
            width='stretch',
            type="primary",
        )
    with col_stop:
        stop_clicked = st.button(
            "⛔ Stop",
            width='stretch',
        )

    if stop_clicked:
        st.session_state["datachat_analysis_running"] = False
        st.rerun()

    # Po rerun'ach Streamlit (np. interakcje z UI) przycisk nie jest "kliknięty".
    # Nie chcemy wtedy chować wyników — jeśli mamy już ostatni chart_spec (czyli analiza była uruchomiona), renderujemy dalej.
    if (not run_clicked) and (not analysis_running) and (st.session_state.get("datachat_last_chart_spec") is None):
        render_flow_nav(current_id="03_Data_Chat", key_prefix="flow_bottom")
        st.markdown("---")
        st.stop()

    committed_prompt = str(st.session_state.get("datachat_committed_prompt") or "").strip()
    committed_filters_raw = st.session_state.get("datachat_committed_filters")
    committed_filters = committed_filters_raw if isinstance(committed_filters_raw, dict) else {}

    if run_clicked:
        st.session_state["datachat_analysis_running"] = True
        st.session_state["datachat_committed_prompt"] = live_prompt
        st.session_state["datachat_committed_filters"] = copy.deepcopy(_filters_live)
        st.session_state["datachat_cs_active_view"] = "overview"
        st.session_state["datachat_cs_active_view_label"] = "Obraz całości"
        prompt = live_prompt
        active_filters = copy.deepcopy(_filters_live)
    else:
        prompt = committed_prompt or live_prompt
        current_intent = str(st.session_state.get("datachat_active_intent") or "").strip().lower()
        if current_intent == "distribution" and prompt:
            active_filters = copy.deepcopy(_filters_live)
        else:
            active_filters = copy.deepcopy(committed_filters or _filters_live)

    st.session_state["datachat_active_filters"] = active_filters
    df = _apply_df_filters(df_source, active_filters)
    roles = _get_roles_from_session()

    chart_spec = _get_chart_spec_from_session()
    if chart_spec is None:
        chart_spec = _plan_chart_placeholder(prompt)
    # (opcjonalnie) zapisz, aby kolejne reruny miały to samo
    st.session_state["datachat_last_chart_spec"] = chart_spec

    cp0_cache_key = _make_cp0_cache_key(
        df=df,
        prompt=prompt,
        chart_spec=chart_spec,
        roles=roles,
        filters=active_filters,
    )
    cp0_cache = st.session_state.get("datachat_cp0_cache")
    if not isinstance(cp0_cache, dict):
        cp0_cache = {}
    cp0_struct = cp0_cache.get(cp0_cache_key)
    if isinstance(cp0_struct, dict):
        try:
            composition_static.record_debug_checkpoint(
                "datachat.cp0.cache_hit",
                cache_key=cp0_cache_key,
            )
        except Exception:
            pass
    else:
        cp0_struct = cp0_compute_checkpoint0_analysis_struct(
            df=df,
            question=prompt,
            chart_spec=chart_spec,
            schema_ctx=schema_ctx,
            roles=roles,
        )
        cp0_cache[cp0_cache_key] = cp0_struct if isinstance(cp0_struct, dict) else {}
        if len(cp0_cache) > 8:
            for _old_key in list(cp0_cache.keys())[:-8]:
                cp0_cache.pop(_old_key, None)
        st.session_state["datachat_cp0_cache"] = cp0_cache
        try:
            composition_static.record_debug_checkpoint(
                "datachat.cp0.cache_store",
                cache_key=cp0_cache_key,
            )
        except Exception:
            pass

    # Gate: Composition Changing Over Time (COT)
    # Cel: jeśli mamy kolumnę czasu i intent = composition, kieruj do gałęzi COT.
    # Uwaga: time_intent bywa klasyfikowany jako RANGE dla zapytań z zakresem dat — to nadal COT.
    _has_time_cols = bool(schema_ctx.get("date_cols") or []) or bool(cp0_struct.get("has_time"))
    _analysis_intent = str(cp0_struct.get("analysis_intent") or "").strip().lower()
    _time_intent = str(cp0_struct.get("time_intent") or "").strip().upper()
    _time_mode = str(cp0_struct.get("time_mode") or "").strip().upper()

    # heurystyka: jeśli w pytaniu jest jawny sygnał czasu (rok / miesiąc / "w czasie" / "trend"),
    # to traktujemy to jako COT nawet gdy CP0 nie ustawił OVER_TIME.
    _q = str(prompt or "").lower()
    _q_has_time_hint = bool(re.search(r"\b(19|20)\d{2}\b", _q)) or any(
        kw in _q for kw in ("w czasie", "trend", "dynamik", "zmienia", "zmieniała", "miesiąc", "miesiac", "kwarta", "tydzień", "tydzien")
    )

    cp1_gate = (
        _has_time_cols
        and _analysis_intent == "composition"
        and (
            _time_intent in {"OVER_TIME", "FULL_PERIOD", "RANGE"}
            or _time_mode in {"OVER_TIME", "FULL_PERIOD", "RANGE"}
            or _q_has_time_hint
        )
    )

    _pre_intent = str((chart_spec or {}).get("intent") or "").strip().lower()
    if not _pre_intent:
        _pre_intent = _analysis_intent
    if _pre_intent == "distribution":
        if any(
            k in _q
            for k in [
                "per kategori",
                "wg kategor",
                "struktura",
                "udział",
                "share",
                "mix",
                "pareto",
                "treemap",
                "koncentrac",
                "sprzedaży i per",
                "sprzedaż per",
                "udziały per",
                "ranking",
            ]
        ):
            _pre_intent = "composition_over_time" if _q_has_time_hint else "composition_static"
    elif (not _pre_intent) and any(
        k in _q for k in ["per kategori", "wg kategor", "struktura", "udział", "share", "mix", "pareto", "treemap", "koncentrac"]
    ):
        _pre_intent = "composition_over_time" if _q_has_time_hint else "composition_static"

    st.subheader("Wizualizacja danych", divider="gray")
    tab_labels = ["Obraz całości", "Kluczowe insighty"]
    _render_overview = True
    _render_insights = True
    _use_cs_single_view = False
    tab_overview, tab_insights = st.tabs(tab_labels)

    # ── UI skeleton (MUST): allows branches to fill consistent slots without breaking layout
    _ui_metric_value = None
    _ui_metric_qty = None
    _ui_metric_txn = None
    _ui_overview_chart = None
    _ui_overview_interp = None
    _ui_overview_audio = None
    _ui_overview_history = None
    _ui_insights_slot = None

    if tab_overview is not None:
        with tab_overview:
            m1, m2, m3 = st.columns(3)
            _metric_value_label = "Suma wartości"
            try:
                _metric_intent = str(st.session_state.get("datachat_active_intent") or "").strip().lower()
                _dist_col_live = st.session_state.get("dist__col")
                if _metric_intent == "distribution" and _dist_col_live:
                    _metric_value_label = f"Suma {_dist_col_live}"
            except Exception:
                pass
            with m1:
                _ui_metric_value = st.empty()
                _ui_metric_value.metric(_metric_value_label, "—")
            with m2:
                _ui_metric_qty = st.empty()
                _ui_metric_qty.metric("Suma ilości", "—")
            with m3:
                _ui_metric_txn = st.empty()
                _ui_metric_txn.metric("Liczba transakcji", "—")

            st.markdown("")
            _ui_overview_chart = st.empty()

            if not _use_cs_single_view:
                st.markdown("")
                st.subheader("Interpretacja", divider="gray")
                _ui_overview_interp = st.empty()
                _ui_overview_audio = st.container()
                _ui_overview_history = st.container()

    if tab_insights is not None:
        with tab_insights:
            if _use_cs_single_view:
                st.subheader("Interpretacja", divider="gray")
                _ui_overview_interp = st.empty()
                _ui_overview_audio = st.container()
                _ui_overview_history = st.container()
            _ui_insights_slot = tab_insights.container()

    tabs = {"overview": tab_overview, "insights": tab_insights}
    ctx = {
        "schema_ctx": schema_ctx,
        "filters": st.session_state.get("datachat_active_filters", {}),
        "dist_col_override": (st.session_state.get("datachat_active_filters", {}) or {}).get("dist_col"),
        "distribution_color_col": (st.session_state.get("datachat_active_filters", {}) or {}).get("color_col"),
        # ── CS passthrough (lądują w ctx["filters"] automatycznie, ale explicit dla czytelności)
        "cs_group_col":  (st.session_state.get("datachat_active_filters", {}) or {}).get("cs_group_col"),
        "cs_group_col2": (st.session_state.get("datachat_active_filters", {}) or {}).get("cs_group_col2"),
        "cs_top_n":      (st.session_state.get("datachat_active_filters", {}) or {}).get("cs_top_n", 10),
        "cs_cutoff":     (st.session_state.get("datachat_active_filters", {}) or {}).get("cs_cutoff", 0.80),
        "cs_price_col": (st.session_state.get("datachat_active_filters", {}) or {}).get("cs_price_col"),
        "cs_takeaways": st.session_state.get("cs_takeaways_cache", {}),
        "chart_spec": chart_spec,
        "cp0_struct": cp0_struct,
        "question": prompt,
        "safe_frontend": st.session_state.get("stage3_safe_frontend_cfg_v1", None),
        "tabs": {"overview": tab_overview, "insights": tab_insights},
        "render_overview": _render_overview,
        "render_insights": _render_insights,
        "ui": {
            "overview": {
                "metric_value": _ui_metric_value,
                "metric_qty": _ui_metric_qty,
                "metric_txn": _ui_metric_txn,
                "chart_slot": _ui_overview_chart,
                "interpretation_slot": _ui_overview_interp,
                "audio_slot": _ui_overview_audio,
                "history_slot": _ui_overview_history,
            },
            "insights": {"slot": _ui_insights_slot},
        },
    }

    # 1) ✅ WYKRES — Branch sam używa chart_slot z ctx
    # ⚠️ FIX: nie pozwól, aby branch wrzucał wykres do "history_slot"
    ctx_render = dict(ctx)
    ctx_render["history_slot"] = None
    try:
        _ui = dict(ctx_render.get("ui") or {})
        _ov = dict((_ui.get("overview") or {}))
        _ov["history_slot"] = None
        _ui["overview"] = _ov
        ctx_render["ui"] = _ui
    except Exception:
        pass

    # ✅ Branch render używa chart_slot z ctx_render["ui"]["overview"]["chart_slot"]
    # --- ensure ctx.question (MUST for Exec Takeaway) ---
    if "question" not in ctx_render:
        ctx_render["question"] = (chart_spec or {}).get("question") or ""

    # ✅ Branch render używa chart_slot z ctx_render["ui"]["overview"]["chart_slot"]
    if cp1_gate:
        composition_over_time.render(df, ctx_render)
        intent = "composition_over_time"
    else:
        # 1) intent z chart_spec (LLM)
        intent = str((chart_spec or {}).get("intent") or "").strip().lower()

        # 2) fallback: intent z cp0_struct (heurystyka/CP0), jeśli LLM puste / niepewne
        if not intent:
            intent = str((cp0_struct or {}).get("analysis_intent") or "").strip().lower()

        # 3) HARD fallback po treści pytania (gdy LLM błędnie daje "distribution")
        q_txt = str(prompt or "").lower()
        # Heurystyka: jeśli pytanie brzmi jak kompozycja ("per kategoria/udział/struktura…") → composition.
        # Jeśli dodatkowo ma sygnał czasu / zakres / rok → preferuj COT.
        _COT_TIME_HINTS = (
            "w czasie", "trend", "dynamik", "zmienia", "zmieniała",
            "miesiąc", "miesiac", "kwarta", "tydzień", "tydzien",
            "rok", "okres", "od", "do", "między", "pomiedzy"
        )
        _has_time_hint = bool(re.search(r"\b(19|20)\d{2}\b", q_txt)) or any(h in q_txt for h in _COT_TIME_HINTS)

        if intent == "distribution":
            if any(k in q_txt for k in ["per kategori", "wg kategor", "struktura", "udział", "share", "mix", "pareto", "treemap", "koncentrac", "sprzedaży i per",
                                        "sprzedaż per", "udziały per", "ranking"]):
                intent = "composition_over_time" if _has_time_hint else "composition_static"

        # ROUTING
        if intent == "distribution":
            distribution.render(df, ctx_render)
            _update_distribution_debug_sidebar_exports()

        elif intent in ("composition_over_time", "cot"):
            composition_over_time.render(df, ctx_render)

        elif intent in ("composition", "composition_static"):
            composition_static.render(df, ctx_render)

        else:
            # HARD fallback — statyczna kompozycja jest najbezpieczniejsza
            intent = "composition_static"
            composition_static.render(df, ctx_render)

    # ✅ PERSIST intent → sidebar na kolejnym rerun wie, który panel pokazać
    st.session_state["datachat_active_intent"] = intent

    # ✅ MUST: po zakończeniu renderu odblokuj przycisk "Przeanalizuj pytanie"
    st.session_state["datachat_analysis_running"] = False

    # ─────────────────────────────────────────────
    # UI: Overview — formatowanie interpretacji + stars rating + historia (pełna)
    # ─────────────────────────────────────────────
    try:
        ui_overview = ((ctx.get("ui") or {}).get("overview") or {}) if isinstance(ctx.get("ui"), dict) else {}
        interp_slot = ui_overview.get("interpretation_slot")
        audio_slot = ui_overview.get("audio_slot")
        history_slot = ui_overview.get("history_slot")

        filters = st.session_state.get("datachat_active_filters", {}) or {}
        dist_col = filters.get("dist_col")

        def _answer_key(q_txt: str, intent_txt: str, f: Dict[str, Any]) -> str:
            """Jedno źródło prawdy dla identyfikacji odpowiedzi (rating + historia)."""
            return datachat_answer_key(q_txt, intent_txt, f)

        # ✅ rating per odpowiedź: {answer_key: int(0..5)}
        if not isinstance(st.session_state.get("datachat_ratings"), dict):
            st.session_state["datachat_ratings"] = {}

        def _h(title: str) -> None:
            st.markdown(
                f"<div style='font-size:18px;font-weight:800;margin:10px 0 6px 0'>{title}</div>",
                unsafe_allow_html=True,
            )
        def _fmt_html(x: object) -> str:
            """Escape text and convert markdown **bold** to <strong> for HTML renderer."""
            s = "" if x is None else str(x)
            s = html.escape(s)
            # Convert markdown bold markers to HTML <strong>
            s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
            return s


        def _tight_bullets(lines: list[str]) -> None:
            for x in lines:
                st.markdown(
                    "<div style='margin:2px 0 2px 0; line-height:1.35;'>• " + _fmt_html(x) + "</div>",
                    unsafe_allow_html=True,
                )
            # subtle gap after each subsection list (match Distribution / CS)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Interpretacja (tylko jeśli jest slot) - Z MECHANIZMEM RENDER ONCE
        interp_md = None
        one_sentence = None

        if interp_slot is not None and intent == "distribution" and dist_col and dist_col in df.columns:
            # ✅ Unikalna klucz dla tej analizy (unikamy duplikacji)
            _prompt_key = hashlib.md5((prompt or "").strip().encode("utf-8")).hexdigest()[:10]

            # ✅ Stabilizator rerenderu: klucz MUSI obejmować też kolorowanie + Top-K policy + SAFE policy
            # (inaczej Streamlit próbuje reużyć stare specy / cache po zmianie kolorów).
            _render_key = (
                _DC_INTERP_VERSION,
                _prompt_key,  # ⬅️ kluczowe: zmiana pytania wymusza regenerację
                intent,
                dist_col,
                tuple(filters.get("dist_range") or []),
                filters.get("color_col"),
                int((filters.get("topk_k") or 0) or 0),
                bool((filters.get("safe") or {}).get("enabled", False)),
                int((filters.get("safe") or {}).get("sample_size", 0) or 0),
                filters.get("filt_col"),
                tuple(filters.get("filt_values") or []),
            )
            distribution.record_debug_checkpoint(
                "dist.interp.resolved",
                prompt=(prompt or "").strip(),
                dist_col=dist_col,
                dist_range=list(filters.get("dist_range") or []),
                color_col=filters.get("color_col"),
                filt_col=filters.get("filt_col"),
                filt_values=list(filters.get("filt_values") or []),
                topk_k=int((filters.get("topk_k") or 0) or 0),
            )

            _last_render_key = st.session_state.get("datachat_last_render_key")

            if _render_key == _last_render_key:
                interp_md = st.session_state.get("datachat_interp")
                one_sentence = st.session_state.get("datachat_one_sentence")

                # ⛔ Guard: jeżeli w cache siedzi stary "głupi" fallback albo struktura jest niekompletna -> regeneruj
                if not isinstance(interp_md, dict):
                    interp_md, one_sentence = None, None
                else:
                    if not all(k in interp_md for k in ("one_sentence", "what", "insights", "reco", "limits")):
                        interp_md, one_sentence = None, None
                    if isinstance(one_sentence, str) and "Brak interpretacji LLM" in one_sentence:
                        interp_md, one_sentence = None, None
                    else:
                        distribution.record_debug_checkpoint(
                            "dist.interp.cache_hit",
                            render_key=str(_render_key),
                        )
                        _cached_dbg = st.session_state.get("dc_interp_debug") or {}
                        if isinstance(_cached_dbg, dict) and _cached_dbg:
                            st.session_state["dist_interp_debug_v1"] = _cached_dbg
            
            # ✅ Jeśli nowa analiza - wygeneruj interpretację
            if interp_md is None or one_sentence is None:
                distribution.record_debug_checkpoint(
                    "dist.interp.cache_miss",
                    render_key=str(_render_key),
                )
                df_work = df
                
                # Zastosuj filtry (zakres)
                rng = filters.get("dist_range")
                if rng and isinstance(rng, (list, tuple)) and len(rng) == 2:
                    lo, hi = rng
                    try:
                        s_num = pd.to_numeric(df_work[dist_col], errors="coerce")
                        df_work = df_work[(s_num >= float(lo)) & (s_num <= float(hi))]
                    except Exception:
                        pass
                
                # Zastosuj filtry (kategorie)
                fcol = filters.get("filt_col")
                fvals = filters.get("filt_values")
                if fcol and isinstance(fvals, (list, tuple)) and len(fvals) > 0 and fcol in df_work.columns:
                    try:
                        df_work = df_work[df_work[fcol].isin(fvals)]
                    except Exception:
                        pass
                
                # ✅ Aktualizuj metryki KPI po zastosowaniu filtrów
                try:
                    _mval = ((ctx.get('ui') or {}).get('overview') or {}).get('metric_value')
                    _mtxn = ((ctx.get('ui') or {}).get('overview') or {}).get('metric_txn')
                    if _mtxn is not None:
                        _mtxn.metric('Liczba transakcji', f"{len(df_work):,}".replace(',', ' '))
                    if _mval is not None:
                        # ✅ NIE nadpisuj dynamicznego KPI statycznym "Suma wartości".
                        # KPI #1 ma zawsze dotyczyć aktualnie wybranej "Miary do rozkładu".
                        if dist_col and dist_col in df_work.columns and pd.api.types.is_numeric_dtype(df_work[dist_col]):
                            _total = float(pd.to_numeric(df_work[dist_col], errors="coerce").sum())
                            _mval.metric(f"Suma {dist_col}", f"{_total:,.0f}".replace(',', ' '))
                        else:
                            _mval.metric('Suma', '—')
                except Exception:
                    pass
                
                s = pd.to_numeric(df_work[dist_col], errors="coerce").dropna()

                # ✅ stats_payload = jedyne źródło prawdy (spójne z histogram/KDE/IQR na danych po filtrach)
                stats_payload = _dc_build_stats_payload_distribution(
                    s=s,
                    dist_col=str(dist_col),
                    filters=filters or {},
                )
                distribution.record_debug_checkpoint(
                    "dist.interp.stats_payload",
                    stats_keys=list(stats_payload.keys()) if isinstance(stats_payload, dict) else [],
                    n=stats_payload.get("n") if isinstance(stats_payload, dict) else None,
                    median=stats_payload.get("median") if isinstance(stats_payload, dict) else None,
                    p95=stats_payload.get("p95") if isinstance(stats_payload, dict) else None,
                    outlier_rate_iqr=stats_payload.get("outlier_rate_iqr") if isinstance(stats_payload, dict) else None,
                    skewness=stats_payload.get("skewness") if isinstance(stats_payload, dict) else None,
                )

                # 1) Guard: jeśli pytanie jest poza zakresem Distribution → nie wołamy LLM
                if _dc_is_out_of_scope_for_distribution(prompt):
                    interp_llm, dbg = None, {"used_fallback": True, "error": "Pytanie poza zakresem widoku Distribution (histogram/KDE/IQR)."}
                else:
                    interp_llm, dbg = _dc_llm_generate_interpretation_json(
                        stats_payload=stats_payload,
                        question=prompt,
                    )

                # 2) Fallback deterministyczny (DOBRY)
                if interp_llm is None:
                    interp_md = _dc_interp_fallback_deterministic(stats_payload)

                    # jeśli out-of-scope, dopisz twarde ograniczenie i rekomendację (bez halucynacji)
                    if isinstance(dbg, dict) and "poza zakresem" in str(dbg.get("error", "")).lower():
                        interp_md = dict(interp_md)
                        interp_md["limits"] = list(interp_md.get("limits") or [])
                        interp_md["limits"].insert(0, "To pytanie wymaga innego widoku niż Distribution; tutaj odpowiadam wyłącznie na podstawie rozkładu (histogram/KDE/IQR).")
                        interp_md["reco"] = list(interp_md.get("reco") or [])
                        interp_md["reco"].insert(0, "Zmień pytanie na dotyczące rozkładu/outlierów albo przejdź do zakładki/gałęzi, która analizuje braki danych / kategorie / czas.")
                else:
                    interp_md = interp_llm

                one_sentence = str(interp_md.get("one_sentence") or "").strip()
                distribution.record_debug_checkpoint(
                    "dist.interp.final_source",
                    src="llm_selected" if interp_llm is not None else "fallback_deterministic_selected",
                    used_fallback=(interp_llm is None) or bool((dbg or {}).get("used_fallback")),
                    error=(dbg or {}).get("error"),
                    gate_reasons=(dbg or {}).get("gate_reasons"),
                    one_sentence=one_sentence,
                )
                st.session_state["dc_interp_debug"] = {
                    "stats_payload": stats_payload,
                    "llm_debug": dbg,
                }
                st.session_state["dist_interp_debug_v1"] = {
                    "stats_payload": stats_payload,
                    "llm_debug": dbg,
                }
                st.session_state["datachat_interp"] = interp_md
                st.session_state["datachat_one_sentence"] = one_sentence
                st.session_state["datachat_last_render_key"] = _render_key
            
            # ✅ Renderowanie (TYLKO RAZ, używając zapisanych danych)
            interp_slot.empty()
            with interp_slot.container():
                _h("✅ Odpowiedź w jednym zdaniu")
                st.markdown(f"<div style='margin:0 0 8px 0'>{_fmt_html(one_sentence)}</div>", unsafe_allow_html=True)
                # 📌 Subtelna informacja, gdy użyto fallback (bez rozwalania one_sentence)
                try:
                    _dd = st.session_state.get("dc_interp_debug") or {}
                    _ld = (_dd.get("llm_debug") or {}) if isinstance(_dd, dict) else {}
                    if isinstance(_ld, dict) and bool(_ld.get("used_fallback", False)):
                        _err = str(_ld.get("error") or "").strip()
                        _reasons = _ld.get("gate_reasons") or []
                        _reason_txt = ""
                        if _err:
                            _reason_txt = _err
                        elif isinstance(_reasons, list) and _reasons:
                            _reason_txt = " | ".join([str(x) for x in _reasons[:2]])
                        if _reason_txt:
                            st.caption(f"⚠️ Fallback: {_reason_txt}")
                except Exception:
                    pass
                
                _h("📊 Co pokazuje wykres")
                _tight_bullets(interp_md.get("what", []))
                
                _h("💡 Kluczowe insighty")
                _tight_bullets(interp_md.get("insights", []))
                
                _h("🎯 Rekomendacje działań")
                _tight_bullets(interp_md.get("reco", []))
                
                _h("⚠️ Ograniczenia / zastrzeżenia")
                _tight_bullets(interp_md.get("limits", []))

                # Distribution debug UI hidden in production; checkpoints stay internal only.
                if bool(st.session_state.get("dist_debug_ui_enabled", False)):
                    with st.expander("🧪 DEBUG: stats_payload + quality gate (Interpretacja)", expanded=False):
                        dd = st.session_state.get("dc_interp_debug") or {}
                        st.markdown("**stats_payload (jedyna prawda)**")
                        st.json(dd.get("stats_payload") or {})
                        st.markdown("**LLM / gate debug**")
                        st.json(dd.get("llm_debug") or {})

            _update_distribution_debug_sidebar_exports()

        # ── Interpretacja: COMPOSITION STATIC ──────────────────────────────
        if interp_slot is not None and intent in ("composition", "composition_static"):
            # ── sidebar params ──────────────────────────────────────────────
            _cs_group_col  = filters.get("cs_group_col")
            _cs_group_col2 = filters.get("cs_group_col2")
            _cs_top_n      = int(filters.get("cs_top_n") or 10)
            _cs_price_col  = filters.get("cs_price_col")

            _cs_sidebar_signature = (_cs_group_col, _cs_group_col2, _cs_price_col, _cs_top_n)
            _cs_render_state = st.session_state.get("datachat_cs_render_state_v3") or {}
            _cs_render_state_match = (
                isinstance(_cs_render_state, dict)
                and _cs_render_state.get("sidebar_signature") == _cs_sidebar_signature
            )

            if _cs_render_state_match:
                _cs_group_col = _cs_render_state.get("group_col")
                _cs_group_col2 = _cs_render_state.get("group_col2")
                _cs_price_col = _cs_render_state.get("price_col")
                _cs_value_col = _cs_render_state.get("value_col")
                _cs_price_source = _cs_render_state.get("price_source")
                _cs_stats_from_render = (
                    _cs_render_state.get("stats_payload")
                    if isinstance(_cs_render_state.get("stats_payload"), dict)
                    else None
                )
            else:
                _cs_resolved = composition_static.resolve_grouping_selection(
                    df=df,
                    group_col_sel=_cs_group_col,
                    group_col2_sel=_cs_group_col2,
                    price_col_sel=_cs_price_col,
                )
                _cs_group_col = _cs_resolved.get("group_col")
                _cs_group_col2 = _cs_resolved.get("group_col2")
                _cs_price_col = _cs_resolved.get("price_col")
                _cs_price_source = _cs_resolved.get("price_source")
                _cs_value_candidates = composition_static.get_value_candidate_columns(
                    df,
                    exclude={_cs_price_col} if _cs_price_col else None,
                )
                _cs_value_col = _cs_value_candidates[0] if _cs_value_candidates else None
                if not _cs_value_col:
                    _cs_value_col = composition_static._infer_value_col(df)
                _cs_stats_from_render = None
            composition_static.record_debug_checkpoint(
                "cs.interp.resolved",
                prompt=(prompt or "").strip(),
                group_col=_cs_group_col,
                group_col2=_cs_group_col2,
                price_col=_cs_price_col,
                value_col=_cs_value_col,
                price_source=_cs_price_source,
                top_n=_cs_top_n,
                reused_render_state=_cs_render_state_match,
            )

            # fallback group_col (heurystyka identyczna jak w composition_static.py)
            if not _cs_group_col or _cs_group_col not in df.columns:
                _cat_cs = [c for c in df.columns if str(df[c].dtype) in ("object", "category")]
                for _pref in ["sector", "segment", "category", "kategoria", "mszoning", "subclass", "class"]:
                    for _cc in _cat_cs:
                        if _pref in _cc.lower() and 2 <= df[_cc].nunique(dropna=True) <= 30:
                            _cs_group_col = _cc
                            break
                    if _cs_group_col:
                        break

            if _cs_group_col and _cs_value_col:
                # ── render-once key ──────────────────────────────────────
                _cs_prompt_key = hashlib.md5((prompt or "").strip().encode("utf-8")).hexdigest()[:10]
                _cs_render_key = (
                    _DC_INTERP_VERSION, "cs",
                    _cs_prompt_key, intent,
                    _cs_group_col, _cs_group_col2, _cs_price_col, _cs_top_n,
                )

                _cs_last_key = st.session_state.get("datachat_cs_last_render_key")
                interp_md    = None
                one_sentence = None

                # ── cache hit ──────────────────────────────────────────
                if _cs_render_key == _cs_last_key:
                    interp_md    = st.session_state.get("datachat_cs_interp")
                    one_sentence = st.session_state.get("datachat_cs_one_sentence")
                    if not isinstance(interp_md, dict) or not all(
                        k in interp_md for k in ("one_sentence", "what", "insights", "reco", "limits")
                    ):
                        interp_md, one_sentence = None, None
                    else:
                        composition_static.record_debug_checkpoint(
                            "cs.interp.cache_hit",
                            render_key=str(_cs_render_key),
                        )

                # ── cache miss → generate ──────────────────────────────
                if interp_md is None or one_sentence is None:
                    composition_static.record_debug_checkpoint(
                        "cs.interp.cache_miss",
                        render_key=str(_cs_render_key),
                    )
                    if _cs_render_state_match and isinstance(_cs_stats_from_render, dict):
                        stats_payload_cs = dict(_cs_stats_from_render)
                        composition_static.record_debug_checkpoint(
                            "cs.interp.stats_reused_from_render",
                            stats_keys=list(stats_payload_cs.keys()),
                            error=stats_payload_cs.get("error"),
                        )
                    else:
                        stats_payload_cs = _dc_build_stats_payload_composition(
                            df=df,
                            group_col=_cs_group_col,
                            value_col=_cs_value_col,
                            group_col2=_cs_group_col2,
                            top_n=_cs_top_n,
                            price_col=_cs_price_col,
                        )
                    composition_static.record_debug_checkpoint(
                        "cs.interp.stats_payload",
                        stats_keys=list(stats_payload_cs.keys()) if isinstance(stats_payload_cs, dict) else [],
                        error=stats_payload_cs.get("error") if isinstance(stats_payload_cs, dict) else None,
                        price_corridor_ready=bool((stats_payload_cs.get("price_corridor") or {}).get("bins")) if isinstance(stats_payload_cs, dict) else False,
                    )

                    if "error" not in stats_payload_cs:
                        interp_llm_cs, dbg_cs = _dc_llm_generate_cs_interpretation(
                            stats_payload=stats_payload_cs,
                            question=prompt,
                        )
                    else:
                        interp_llm_cs = None
                        dbg_cs = {"used_fallback": True, "error": stats_payload_cs.get("error")}

                    interp_md = interp_llm_cs if interp_llm_cs else _dc_cs_fallback_deterministic(stats_payload_cs)
                    if interp_llm_cs is None:
                        dbg_cs["used_fallback"] = True
                    one_sentence = str(interp_md.get("one_sentence") or "").strip()
                    composition_static.record_debug_checkpoint(
                        "cs.interp.final_source",
                        src="llm_selected" if interp_llm_cs else "fallback_deterministic_selected",
                        used_fallback=bool(dbg_cs.get("used_fallback")),
                        error=dbg_cs.get("error"),
                        gate_reasons=dbg_cs.get("gate_reasons"),
                        one_sentence=one_sentence,
                    )

                    
                    # Executive takeaways (ET v1.0) — cached
                    # Switch CS to the same engine as Distribution:
                    # 2 sentences, >=2 numbers, decision in sentence 2, PL-only
                    tw_cs = st.session_state.get("cs_takeaways_cache", {})
                    tw_ver = "et_v1"

                    # Rebuild if missing or from legacy generator (removes ~160-char limit + ellipsis issues)
                    if False and ((not isinstance(tw_cs, dict)) or (tw_ver != "et_v1")):
                        try:
                            from data_chat_core.exec_takeaway import get_exec_takeaway as _get_exec_takeaway
                        except Exception:
                            _get_exec_takeaway = None  # type: ignore

                        required_keys = ["ranking", "waterfall", "pareto", "mix", "marimekko", "price_corridor"]
                        tw_cs = {}
                        meta_map = {}

                        if callable(_get_exec_takeaway):
                            for k in required_keys:
                                # minimal block stub; real numbers come from stats_payload_cs
                                _block_stub = {"key": k, "title": k}
                                txt, meta = _get_exec_takeaway(
                                    intent=f"composition_static:{k}",
                                    block=_block_stub,
                                    stats=stats_payload_cs,
                                    question=prompt,
                                    session_state=st.session_state,
                                    llm_fn=None,
                                    force_refresh=False,
                                )
                                tw_cs[k] = txt
                                meta_map[k] = meta
                            dbg_tw = {"engine": "exec_takeaway_v1", "meta_by_block": meta_map}
                        else:
                            # last-resort fallback (should be rare) — keep empty dict to avoid crash
                            dbg_tw = {"engine": "exec_takeaway_v1", "error": "get_exec_takeaway import failed"}
                            tw_cs = {}

                        st.session_state["datachat_cs_takeaways"] = tw_cs
                        st.session_state["datachat_cs_takeaways_debug"] = dbg_tw
                        st.session_state["datachat_cs_takeaways_version"] = "et_v1"
                        
                    st.session_state["datachat_cs_interp"]          = interp_md
                    st.session_state["datachat_cs_one_sentence"]    = one_sentence
                    st.session_state["datachat_cs_last_render_key"] = _cs_render_key
                    st.session_state["dc_interp_debug"] = {
                        "stats_payload": stats_payload_cs,
                        "llm_debug":     dbg_cs,
                    }
                    # persist for history (same keys Distribution uses)
                    st.session_state["datachat_interp"]       = interp_md
                    st.session_state["datachat_one_sentence"] = one_sentence

                # ── render into interp_slot (identycznie jak Distribution) ─
                interp_slot.empty()
                with interp_slot.container():
                    _h("✅ Odpowiedź w jednym zdaniu")
                    st.markdown(f"<div style='margin:0 0 8px 0'>{_fmt_html(one_sentence)}</div>", unsafe_allow_html=True)

                    # fallback badge
                    try:
                        _dd_cs = st.session_state.get("dc_interp_debug") or {}
                        _ld_cs = (_dd_cs.get("llm_debug") or {}) if isinstance(_dd_cs, dict) else {}
                        if isinstance(_ld_cs, dict) and bool(_ld_cs.get("used_fallback", False)):
                            _err_cs   = str(_ld_cs.get("error") or "").strip()
                            _rsn_cs   = _ld_cs.get("gate_reasons") or []
                            _txt_cs   = _err_cs or (" | ".join([str(x) for x in _rsn_cs[:2]]) if _rsn_cs else "")
                            if _txt_cs and bool(
                                st.session_state.get("cs_debug_show_fallbacks", False)
                                or st.session_state.get("dc_debug", False)
                            ):
                                st.caption(f"⚠️ Fallback: {_txt_cs}")
                    except Exception:
                        pass

                    _h("📊 Co pokazuje wykres")
                    _tight_bullets(interp_md.get("what", []))

                    _h("💡 Kluczowe insighty")
                    _tight_bullets(interp_md.get("insights", []))

                    _h("🎯 Rekomendacje działań")
                    _tight_bullets(interp_md.get("reco", []))

                    _h("⚠️ Ograniczenia / zastrzeżenia")
                    _tight_bullets(interp_md.get("limits", []))

                    if bool(st.session_state.get("cs_debug_enabled", False) or st.session_state.get("dc_debug", False)):
                        with st.expander("🧪 DEBUG: stats_payload + quality gate (CS)", expanded=False):
                            _dd_dbg = st.session_state.get("dc_interp_debug") or {}
                            st.markdown("**stats_payload (jedyna prawda)**")
                            st.json(_dd_dbg.get("stats_payload") or {})
                            st.markdown("**LLM / gate debug**")
                            st.json(_dd_dbg.get("llm_debug") or {})

        # CS debug sidebar export removed in production; checkpoints stay internal only.

        # ── Audio + rating (gwiazdki po prawej w jednej linii) - BEZ IFRAME
        st.markdown(
            """
        <style>
        /* rating pill: stała wysokość => brak skoku layoutu */
        .dc-pill-wrap { min-height: 24px; margin-top: 6px; }
        .dc-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        background: #e8f5e9;
        color: #1b5e20;
        font-size: 12px;
        font-weight: 650;
        line-height: 18px;
        }
        .dc-pill--empty { background: transparent; color: transparent; }

        /* delikatne dopasowanie odstępów w historii */
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p { margin: 0.15rem 0 0.45rem 0; }
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] ul { margin: 0.15rem 0 0.55rem 1.2rem; }
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h3 { margin: 0.55rem 0 0.25rem 0; }
        </style>
        """,
            unsafe_allow_html=True,
        )

        if audio_slot is not None:
            with audio_slot:
                st.subheader("Narracja audio i ocena odpowiedzi", divider="gray")
                left, right = st.columns([7, 3], vertical_alignment="center")


                # --- TTS state z sidebaru (już u Ciebie istnieje) ---
                dc_tts_enabled = bool(st.session_state.get("dc_tts_enabled", False))
                dc_tts_voice = str(st.session_state.get("dc_tts_voice", "shimmer") or "shimmer")
                dc_tts_model = str(st.session_state.get("dc_tts_model", os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")) or "gpt-4o-mini-tts")

                # API KEY z .env / env
                api_key = os.getenv("OPENAI_API_KEY", "")

                # klucz odpowiedzi (żeby audio było per odpowiedź + per głos/model)
                # (używamy tego samego klucza co historia)
                _ans_key = str(st.session_state.get("datachat_current_key") or "").strip()
                if not _ans_key:
                    hist = st.session_state.get("datachat_history") or []
                    if isinstance(hist, list) and hist:
                        _ans_key = str(hist[-1].get("key") or hist[-1].get("_key") or "").strip()

                # ostateczny fallback (żeby nigdy nie było pusto)
                if not _ans_key:
                    _ans_key = hashlib.md5(
                        repr((prompt, intent, filters)).encode("utf-8", errors="ignore")
                    ).hexdigest()[:10]

                # tekst do narracji: najpierw one_sentence, fallback: krótkie streszczenie interp_md
                tts_text = ""
                try:
                    if isinstance(one_sentence, str) and one_sentence.strip():
                        tts_text = one_sentence.strip()
                    elif isinstance(interp_md, str) and interp_md.strip():
                        tts_text = (interp_md.strip().replace("\n", " "))[:800]
                except Exception:
                    tts_text = ""

                # cache per-odpowiedź + per głos/model (żeby zmiana voice REALNIE zmieniała audio)
                if not isinstance(st.session_state.get("datachat_tts_audio"), dict):
                    st.session_state["datachat_tts_audio"] = {}
                tts_audio_map = st.session_state["datachat_tts_audio"]

                audio_key = f"{_ans_key}|{dc_tts_voice}|{dc_tts_model}"

                # zawsze zainicjuj err (żeby debug nie wywalał 'unbound local')
                err = None
                generated_now = False

                with left:
                    if not dc_tts_enabled:
                        st.caption("🔇 Lektor: wyłączony.")
                    else:
                        # generuj jeśli brak audio dla tej kombinacji (odpowiedź+voice+model)
                        if (audio_key not in tts_audio_map) and tts_text:
                            cache_bust = str(st.session_state.get("datachat_tts_cache_bust", 0) or 0)
                            with st.spinner("🎙️ Generuję nagranie lektora..."):
                                mp3_bytes, err = datachat_tts_generate(
                                    text=tts_text,
                                    voice=dc_tts_voice,
                                    model=dc_tts_model,
                                    api_key=api_key,
                                    cache_bust=cache_bust,
                                )
                            if not err and mp3_bytes:
                                tts_audio_map[audio_key] = mp3_bytes
                                st.session_state["datachat_tts_audio"] = tts_audio_map
                                generated_now = True

                        mp3 = tts_audio_map.get(audio_key, b"")

                        # 🔊 Player (zawsze full width, bez “pustej” przerwy)
                        if mp3:
                            st.caption("🔊 Lektor: wygenerowany dla tej odpowiedzi.")

                            import base64 as _b64
                            b64 = _b64.b64encode(mp3).decode("ascii")

                            # autoplay tylko gdy świeżo wygenerowane (po kliknięciu "Przeanalizuj pytanie")
                            autoplay_attr = "autoplay" if generated_now else ""

                            components.html(
                                f"""
                                <audio controls {autoplay_attr} style="width:100%; height:44px;">
                                <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
                                </audio>
                                """,
                                height=54,
                            )
                        else:
                            if err:
                                st.error(err)
                            else:
                                st.caption("🎧 Lektor: brak audio (brak tekstu lub jeszcze nie wygenerowano).")


                        # # ✅ DEBUG ZAWSZE POD PLAYEREM ----------------------------------
                        # # --- SAFETY DEFAULTS (żeby nie było "not defined") ---
                        # dc_tts_gender = st.session_state.get("dc_tts_gender", "Kobieta")
                        # dc_tts_voice = st.session_state.get("dc_tts_voice", "shimmer")
                        # err = locals().get("err", None)
                        # with st.expander("🧪 DEBUG: audio player (TTS)", expanded=True):
                        #     st.write(
                        #         {
                        #             "tts_enabled": dc_tts_enabled,
                        #             "ans_key": _ans_key,
                        #             "audio_key": audio_key,
                        #             "tts_text_len": len(tts_text or ""),
                        #             "gender": dc_tts_gender,
                        #             "voice": dc_tts_voice,
                        #             "model": dc_tts_model,
                        #             "api_key_loaded": bool(api_key),
                        #             "mp3_len": len(mp3 or b""),
                        #             "generated_now": generated_now,
                        #             "err": err,
                        #         }
                        #     )
                        # # ------------------------------------------------------------ 

                with right:
                    # ✅ klucz bieżącej odpowiedzi (rating per odpowiedź)
                    _k = st.session_state.get('datachat_current_key') or _answer_key(prompt, intent, filters)

                    if not isinstance(st.session_state.get("datachat_ratings"), dict):
                        st.session_state["datachat_ratings"] = {}
                    ratings = st.session_state["datachat_ratings"]

                    # ✅ Unikalny, stabilny key widgetu PER ODPOWIEDŹ (to jest główna naprawa 1-gwiazdki)
                    _k_id = hashlib.md5(repr(_k).encode("utf-8")).hexdigest()[:10]
                    _slider_key = f"datachat_rating_slider_{_k_id}"

                    # ✅ bieżąca wartość ratingu dla tej odpowiedzi
                    current_rating = int(ratings.get(_k, 0) or 0)

                    # ✅ Layout: skróć i przesuń w prawo "umiarkowanie" + zachowaj tekst w 1 linii
                    _spacer, _box = st.columns([1, 9], vertical_alignment="center")
                    with _box:
                        st.markdown(
                            """
                            <style>
                            div[data-testid="stSelectSlider"] label { display:none !important; }
                            /* delikatne odsunięcie od prawej + brak "rozjeżdżania" */
                            div[data-testid="stSelectSlider"] { padding-right: 10px !important; }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            "<div style='font-weight:500; margin-bottom:6px; white-space:nowrap;'>"
                            "Oceń tę odpowiedź (1 ⭐ – najsłabiej, 5 ⭐ – najlepiej):"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                        stars = ["☆", "★☆☆☆☆", "★★☆☆☆", "★★★☆☆", "★★★★☆", "★★★★★"]

                        # Callback który wykona się TYLKO przy zmianie przez użytkownika
                        def on_rating_change():
                            new_rating = st.session_state[_slider_key]
                            
                            # 1) ✅ zapis per-odpowiedź
                            if not isinstance(st.session_state.get("datachat_ratings"), dict):
                                st.session_state["datachat_ratings"] = {}
                            st.session_state["datachat_ratings"][_k] = new_rating

                            # 2) ✅ aktualizacja historii (zapis do datachat_history)
                            hist = st.session_state.get("datachat_history") or []
                            if isinstance(hist, list):
                                # Znajdź odpowiedni wpis w historii i zaktualizuj rating
                                found = False
                                for it in reversed(hist):
                                    if (it.get("key") == _k) or (it.get("_key") == _k):
                                        it["rating"] = new_rating
                                        found = True
                                        break
                                
                                # Jeśli nie znaleziono - dodaj do ostatniego wpisu
                                if not found and hist:
                                    hist[-1]["rating"] = new_rating
                                
                                st.session_state["datachat_history"] = hist

                            # 3) ✅ Toast NATYCHMIAST
                            if hasattr(st, "toast"):
                                st.toast("✅ Zapisano ✓", icon="✅")
                            
                            # 4) ✅ feedback data (dla potencjalnego pill - ale nie działa)
                            st.session_state["datachat_rating_feedback"] = {
                                "k_id": _k_id,
                                "msg": "Zapisano ✓",
                                "ts": time.time(),
                                "show_for": 3.0,
                                "toast_shown": True,
                            }
                            
                            # NIE ROBIMY RERUNU - pill nie działa, a rerun powoduje problemy

                        selected_idx = st.select_slider(
                            "Rating",
                            options=list(range(6)),
                            value=current_rating,
                            format_func=lambda x: stars[x],
                            key=_slider_key,
                            label_visibility="collapsed",
                            on_change=on_rating_change,
                        )
                        
                        # ✅ Dodaj tekstowe labele POD sliderem dla każdej opcji
                        st.markdown(
                            "<div style='display:flex; justify-content:space-between; margin-top:4px; font-size:13px; color:#666;'>"
                            "<span>0☆</span><span>1★</span><span>2★</span><span>3★</span><span>4★</span><span>5★</span>"
                            "</div>",
                            unsafe_allow_html=True
                        )

        # ── Historia (pytanie + pełna interpretacja)
        if history_slot is not None:
            with history_slot:
                st.subheader("Historia rozmowy (kliknij, aby rozwinąć)", divider="gray")
                hist = st.session_state.get("datachat_history") or []

                with st.expander("🧾 Historia rozmowy (kliknij, aby rozwinąć)", expanded=False):
                    if not hist:
                        st.caption("Brak historii w tej sesji.")
                    else:
                        def _stars_str(r: int | None) -> str:
                            if not isinstance(r, int) or r < 1 or r > 5:
                                return "—"
                            return ("★" * r) + ("☆" * (5 - r))

                        for i, item in enumerate(reversed(hist[-20:]), start=1):
                            st.markdown(f"**{i}. {item.get('q','')}**")
                            st.caption(f"Ocena: {_stars_str(item.get('rating'))}")

                            saved = item.get("interp")

                            if isinstance(saved, dict):
                                # mniejszy odstęp tytuł→treść, większy między sekcjami
                                st.markdown(
                                    f"<div style='margin:0 0 10px 0;'>"
                                    f"<b>✅ Odpowiedź w jednym zdaniu</b><br>"
                                    f"{saved.get('one_sentence','')}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                                if saved.get("what"):
                                    st.markdown("<div style='margin:10px 0 4px 0;'><b>📊 Co pokazuje wykres</b></div>", unsafe_allow_html=True)
                                    _tight_bullets(saved.get("what"))

                                if saved.get("insights"):
                                    st.markdown("<div style='margin:10px 0 4px 0;'><b>💡 Kluczowe insighty</b></div>", unsafe_allow_html=True)
                                    _tight_bullets(saved.get("insights"))

                                if saved.get("reco"):
                                    st.markdown("<div style='margin:10px 0 4px 0;'><b>🎯 Rekomendacje działań</b></div>", unsafe_allow_html=True)
                                    _tight_bullets(saved.get("reco"))

                                if saved.get("limits"):
                                    st.markdown("<div style='margin:10px 0 4px 0;'><b>⚠️ Ograniczenia / zastrzeżenia</b></div>", unsafe_allow_html=True)
                                    _tight_bullets(saved.get("limits"))

                            st.markdown("<hr style='margin:12px 0 12px 0;'>", unsafe_allow_html=True)

        # dopisz do historii tylko raz per analiza (spójny klucz z ratingiem)
        try:
            _q = prompt or ""
            _k = _answer_key(_q, str(intent or ""), filters or {})

            last_key = st.session_state.get("datachat_current_key")
            if _k != last_key:
                hist = st.session_state.get("datachat_history") or []
                if not isinstance(hist, list):
                    hist = []

                # co zapisujemy do historii jako "pełna interpretacja"
                _interp_saved = st.session_state.get("datachat_interp")
                _one_sentence_saved = st.session_state.get("datachat_one_sentence")

                # zachowaj tylko ostatnie 30 wpisów
                hist.append(
                    {
                        "q": _q,
                        "intent": str(intent or ""),
                        "key": _k,                     # ✅ jeden standard: "key"
                        "interp": _interp_saved,        # ✅ historia renderuje item.get("interp")
                        "one_sentence": _one_sentence_saved or "",
                        "rating": None,
                    }
                )
                st.session_state["datachat_history"] = hist[-30:]
                st.session_state["datachat_current_key"] = _k

            # MUST: aktywna odpowiedź do oceny (rating per odpowiedź)
            st.session_state["datachat_current_key"] = _k

        except Exception:
            pass

    except Exception as e:
        st.error(f"Data Chat UI error: {e}")
        with st.expander("🧨 DEBUG: wyjątek UI (traceback)", expanded=False):
            import traceback
            st.code(traceback.format_exc())

    # # DEBUG CP0 ────────────────────────────────
    # with st.expander("🧪 DEBUG CP0", expanded=False):
    #     st.json(cp0_struct)
    #     st.write("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))
    # # ─────────────────────────────────────────────

    # --- Nawigacja (dół) ---

    # ── Debug (global) — Executive Takeaway
    if st.session_state.get("debug_exec_takeaway_global"):
        try:
            from data_chat_core.exec_takeaway_debug import render_exec_takeaway_debug
            render_exec_takeaway_debug(st.session_state)
        except Exception as _e:
            st.warning(f"Debug Executive Takeaway: nie udało się wyrenderować ({_e}).")

    render_flow_nav(current_id="03_Data_Chat", key_prefix="flow_bottom")
    st.markdown("---")

main()

