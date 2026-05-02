from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

from time import perf_counter
import datetime
import re
import hashlib
import html
import json
import uuid
import unicodedata

import math
import numpy as np
import pandas as pd
import streamlit as st


def _safe_json(v: Any) -> Any:
    """Best-effort, size-safe serializer for debug checkpoints (never raise)."""
    try:
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        # Common data containers
        if isinstance(v, (list, tuple)):
            return {"type": type(v).__name__, "len": len(v), "head": [_safe_json(x) for x in list(v)[:5]]}
        if isinstance(v, dict):
            out: Dict[str, Any] = {"type": "dict", "len": len(v), "keys": list(v.keys())[:20]}
            # keep only a few small fields
            for k in list(v.keys())[:10]:
                out[str(k)] = _safe_json(v.get(k))
            return out
        # Pandas / numpy
        if hasattr(v, "shape"):
            sh = getattr(v, "shape", None)
            return {"type": type(v).__name__, "shape": tuple(sh) if sh is not None else None}
        # Fallback to string, truncated
        s = str(v)
        return s if len(s) <= 800 else (s[:800] + "…")
    except Exception:
        return "<unserializable>"


# --- Debug checkpoints (ET flow) ---------------------------------------------
# Lightweight, safe to keep in production (no-op unless __cot_exec_dbg_on == True)
_EXEC_CP_KEY = "__cot_exec_cp_v1"

def dbg_cp(where: str, **fields: Any) -> None:
    """Append a lightweight debug checkpoint for ET flow.

    Writes only when st.session_state['__cot_exec_dbg_on'] is truthy.
    Never raises (debug must not crash the app).
    """
    try:
        if not bool(st.session_state.get("__cot_exec_dbg_on")):
            return
        lst = st.session_state.get(_EXEC_CP_KEY)
        if not isinstance(lst, list):
            lst = []
            st.session_state[_EXEC_CP_KEY] = lst
        # Keep a smaller rolling window to avoid debug-induced memory growth.
        if len(lst) > 300:
            del lst[: max(0, len(lst) - 240)]
        payload: Dict[str, Any] = {"where": where}
        for k, v in fields.items():
            payload[k] = _safe_json(v)
        lst.append(payload)
    except Exception:
        return


def _debug_df_safe(rows: Any):
    """Make debug rows Arrow-safe for st.dataframe by normalizing mixed/object columns."""
    try:
        import pandas as _pd
        df = _pd.DataFrame(rows or [])
        if df.empty:
            return df
        for col in df.columns:
            s = df[col]
            try:
                if str(s.dtype) == "object":
                    non_null = s.dropna()
                    kinds = {type(x).__name__ for x in non_null.head(100).tolist()}
                    if len(kinds) > 1:
                        df[col] = s.map(lambda x: "" if x is None else str(x))
                elif str(s.dtype).startswith("mixed"):
                    df[col] = s.map(lambda x: "" if x is None else str(x))
            except Exception:
                df[col] = s.map(lambda x: "" if x is None else str(x))
        return df
    except Exception:
        return rows


from core.ui_safe import altair_chart_stretch
import altair as alt

from data_chat_core.ui_contract import render_exec_takeaway, render_guidance
from data_chat_core.exec_takeaway import get_exec_takeaway

# ------------------------------------------------------------------
# Compatibility + Debug (minimal patch)
# ------------------------------------------------------------------
# In some call-sites we pass only text to render_exec_takeaway() or
# three text parts to render_guidance(). The core helpers expect
# (slot, text, ...) signatures. Provide thin wrappers that:
# - accept legacy call styles used in this file
# - never crash
# - optionally collect per-block meta into session_state for debug
#
# NOTE: We keep names render_exec_takeaway/render_guidance/get_exec_takeaway
# in module scope to avoid touching the rest of the file.

from typing import Iterable  # noqa: E402

# --- Pylance forward declarations (runtime-safe no-ops; real defs later in file) ---
# These helpers are implemented later in the module. Declaring them here prevents
# editor false-positives when they are referenced before their full definitions.
def _canonical_interp_topic(mode: Any, sem_topic: Any = None) -> str:
    mode_l = str(mode or "").strip().lower()
    topic = str(sem_topic or "").strip().lower()

    if mode_l in ("shares_pp", "share", "mix"):
        if topic in ("shares_pp", "share", "mix", "generic", ""):
            return "shares_pp"
        return topic

    if mode_l in ("value", "sprzedaz", "sales"):
        if topic in ("value", "value_pln", "generic", ""):
            return "value_pln"
        return topic

    if topic:
        return topic
    return mode_l

_STATUS_KEY = "dc_llm_status_v4"
_EXEC_CACHE_KEY = "exec_takeaway_cache"
_EXEC_META_KEY = "exec_takeaway_meta_v1"
_EXEC_ERR_KEY = "exec_takeaway_errors_v1"
_EXEC_LAST_RUN_MODE_KEY = "__cot_exec_last_run_mode_v1"
_EXEC_FORCE_COLD_REQ_KEY = "__cot_exec_force_cold_request_id_v1"
_EXEC_FORCE_COLD_DONE_KEY = "__cot_exec_force_cold_consumed_id_v1"

def _status_set(key: str, meta: Dict[str, Any]) -> None:
    try:
        d = st.session_state.setdefault(_STATUS_KEY, {})
        d[key] = meta
    except Exception:
        return

def _errors_add(msg: str) -> None:
    try:
        st.session_state.setdefault(_EXEC_ERR_KEY, []).append(msg)
    except Exception:
        return


def _set_exec_last_run_mode(mode: str, **meta: Any) -> None:
    try:
        st.session_state[_EXEC_LAST_RUN_MODE_KEY] = {
            "mode": str(mode or ""),
            **dict(meta or {}),
        }
    except Exception:
        return


def _get_exec_force_cold_request_id() -> int:
    try:
        return int(st.session_state.get(_EXEC_FORCE_COLD_REQ_KEY) or 0)
    except Exception:
        return 0


def _get_exec_force_cold_consumed_id() -> int:
    try:
        return int(st.session_state.get(_EXEC_FORCE_COLD_DONE_KEY) or 0)
    except Exception:
        return 0


def _exec_force_cold_pending() -> bool:
    try:
        return _get_exec_force_cold_request_id() > _get_exec_force_cold_consumed_id()
    except Exception:
        return False


def _mark_exec_force_cold_requested() -> int:
    try:
        _next_id = _get_exec_force_cold_request_id() + 1
        st.session_state[_EXEC_FORCE_COLD_REQ_KEY] = int(_next_id)
        return int(_next_id)
    except Exception:
        return 0


def _mark_exec_force_cold_consumed() -> None:
    try:
        st.session_state[_EXEC_FORCE_COLD_DONE_KEY] = int(_get_exec_force_cold_request_id())
    except Exception:
        return

def _hash_short(x: Any) -> str:
    import hashlib, json
    try:
        s = json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        s = str(x)
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

def _cache_exec_takeaway(block_id: Optional[str], meta: Dict[str, Any], text_out: str, allow_empty: bool = True) -> None:
    """Store exec takeaway into session_state for debug table."""
    try:
        cache = st.session_state.setdefault(_EXEC_CACHE_KEY, {})
        # Backward/forward compatible alias used by the UI debug panel
        if "exec_takeaway_cache" not in st.session_state:
            st.session_state["exec_takeaway_cache"] = cache
        else:
            # Keep both keys pointing to the same dict instance
            if st.session_state["exec_takeaway_cache"] is not cache:
                # merge then re-point
                try:
                    cache.update(st.session_state["exec_takeaway_cache"])
                except Exception:
                    pass
                st.session_state["exec_takeaway_cache"] = cache
        try:
            st.session_state.exec_takeaway_cache = cache  # type: ignore[attr-defined]
        except Exception:
            pass

        # Prefer stable key: block_id + stats_hash (or derived) + text_hash
        stats_hash = meta.get("stats_hash") or _hash_short(meta.get("stats") or meta.get("anchors") or {})
        text_hash = meta.get("text_hash") or _hash_short(text_out)
        cache_key = f"{block_id or meta.get('block_id') or 'unknown'}:{stats_hash}:{text_hash}"
        # Guard: do not poison cache with empty text (common failure mode: text->empty).
        if not (text_out or "").strip():
            if not allow_empty:
                dbg_cp(
                    "exec_takeaway.state_write_skip_empty",
                    block_id=block_id,
                    cache_key=cache_key,
                    state_key=_EXEC_CACHE_KEY,
                    written_len=0,
                    src=(meta or {}).get("src"),
                    gate_reason=(meta or {}).get("gate_reason"),
                )
                return
            # If we already have non-empty text for this key, do not overwrite with empty.
            try:
                prev = cache.get(cache_key) or {}
                prev_text = (prev.get("text") if isinstance(prev, dict) else "") or ""
                if prev_text.strip():
                    dbg_cp(
                        "exec_takeaway.state_write_keep_previous",
                        block_id=block_id,
                        cache_key=cache_key,
                        state_key=_EXEC_CACHE_KEY,
                        written_len=len(prev_text),
                        src=(meta or {}).get("src"),
                        gate_reason=(meta or {}).get("gate_reason"),
                    )
                    return
            except Exception:
                pass
        cache[cache_key] = {"text": text_out, "meta": meta}
        # DEBUG checkpoint: state write (after cache write)
        dbg_cp("exec_takeaway.state_write",
               block_id=block_id,
               cache_key=cache_key,
               state_key=_EXEC_CACHE_KEY,
               written_len=len(text_out or ""),
               src=(meta or {}).get("src"))

        rows = st.session_state.setdefault(_EXEC_META_KEY, [])
        rows.append({
            "block_id": block_id or meta.get("block_id") or "unknown",
            "source": meta.get("src") or meta.get("source") or "unknown",
            "stats_hash": stats_hash,
            "text_hash": text_hash,
            "gate": meta.get("gate") or meta.get("passed"),
            "preview": (text_out or "")[:140].replace("\n", " "),
        })
        if len(rows) > 120:
            del rows[: max(0, len(rows) - 90)]
    except Exception:
        return

def _read_cached_exec_takeaway(block_id: Optional[str], meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Best-effort read-through cache for ET. Returns (text, meta) or ("", {})."""
    try:
        cache = st.session_state.get(_EXEC_CACHE_KEY, {})
        if not isinstance(cache, dict):
            return "", {}

        stats_hash = meta.get("stats_hash") or _hash_short(meta.get("stats") or meta.get("anchors") or {})
        block_key = str(block_id or meta.get("block_id") or "unknown")

        for key, payload in cache.items():
            if not str(key).startswith(f"{block_key}:{stats_hash}:"):
                continue
            if not isinstance(payload, dict):
                continue

            txt = str(payload.get("text") or "").strip()
            meta_out = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            if txt:
                dbg_cp(
                    "exec_takeaway.cache_hit",
                    block_id=block_key,
                    cache_key=key,
                    text_len=len(txt),
                    src=(meta_out or {}).get("src"),
                )
                return txt, meta_out

    except Exception:
        return "", {}

    return "", {}

# --- wrap core get_exec_takeaway to always return (text, meta) and feed debug ---
_core_get_exec_takeaway = get_exec_takeaway

def get_exec_takeaway(*args, **kwargs):  # type: ignore[override]
    """Safe wrapper around core get_exec_takeaway.

    Expected return: (text, meta_dict). If core returns None or raises, we
    return deterministic fallback and record debug.
    """
    block_id = kwargs.get("block_id") or kwargs.get("block") or kwargs.get("blockName")
    intent = kwargs.get("intent")
    # Clear accumulated exec-takeaway errors when request context changes (prevents stale 'numbers_s1<2')
    run_sig = (
        intent,
        str(kwargs.get("stats_hash") or ""),
        str(kwargs.get("text_hash") or ""),
        str(block_id or ""),
    )
    if st.session_state.get("_cot_exec_run_sig") != run_sig:
        st.session_state[_EXEC_ERR_KEY] = []
        st.session_state["_cot_exec_run_sig"] = run_sig
    try:
        # Inject a per-call id into stats so validators can persist the initial gate reason
        if len(args) >= 2 and isinstance(args[1], dict):
            _st = dict(args[1])
            _cid = _st.get('_dbg_call_id') or uuid.uuid4().hex[:10]
            _st['_dbg_call_id'] = _cid
            args = (args[0], _st) + tuple(args[2:])

        _pre_meta = {
            "block_id": block_id,
            "intent": intent,
        }
        if len(args) >= 2 and isinstance(args[1], dict):
            _pre_meta["stats"] = args[1]
            _pre_meta["stats_hash"] = _hash_short(args[1])

        cached_txt, cached_meta = _read_cached_exec_takeaway(block_id, _pre_meta)
        if cached_txt:
            _status_set(f"exec:{block_id or 'unknown'}", cached_meta or _pre_meta)
            return cached_txt, (cached_meta or _pre_meta)

        out = _core_get_exec_takeaway(*args, **kwargs)
    except Exception as e:
        _errors_add(f"get_exec_takeaway error: {type(e).__name__}: {e}")
        meta = {"block_id": block_id, "intent": intent, "src": "error", "passed": False, "reasons": [str(e)]}
        txt = ""
        _status_set(f"exec:{block_id or 'unknown'}", meta)
        _cache_exec_takeaway(str(block_id or ''), meta, txt, allow_empty=False)
        return txt, meta

    if out is None:
        meta = {"block_id": block_id, "intent": intent, "src": "none", "passed": False, "reasons": ["none"]}
        txt = ""
        _status_set(f"exec:{block_id or 'unknown'}", meta)
        _cache_exec_takeaway(str(block_id or ''), meta, txt, allow_empty=False)
        return txt, meta

    if isinstance(out, tuple) and len(out) == 2:
        txt, meta = out
        meta = meta if isinstance(meta, dict) else {"block_id": block_id, "src": "unknown"}
        meta.setdefault("block_id", block_id)
        meta.setdefault("intent", intent)
        # Recover the initial gate reason (e.g., zero_percent_placeholder) even if a repair later succeeded.
        try:
            if not meta.get('gate_reason'):
                _cid = None
                if len(args) >= 2 and isinstance(args[1], dict):
                    _cid = args[1].get('_dbg_call_id')
                _reasons = _MBB_FAIL_REASONS_BY_CALL_ID.get(str(_cid), []) if _cid else []
                t2 = str(txt or '').lower()
                if ('0,0%' in t2) or ('0.0%' in t2):
                    meta['gate_reason'] = 'zero_percent_placeholder'
                elif any('zero_percent_placeholder' in r for r in _reasons):
                    meta['gate_reason'] = 'zero_percent_placeholder'
                elif _reasons:
                    meta['gate_reason'] = _reasons[0]
        except Exception:
            pass

        dbg_cp("exec_takeaway.candidate_ready_for_selection",
               block_id=block_id,
               src=meta.get("src"),
               gate_reason=meta.get("gate_reason"),
               text_len=len(str(txt or "")),
               preview=str(txt or "")[:160])
        _status_set(f"exec:{block_id or 'unknown'}", meta)
        _cache_exec_takeaway(block_id, meta, str(txt or ""), allow_empty=False)
        return str(txt or ""), meta

    # Some implementations may return dict-like
    if isinstance(out, dict):
        txt = str(out.get("text") or out.get("takeaway") or "")
        if not str(txt or "").strip():
            _props = out.get("properties")
            if isinstance(_props, dict):
                txt = str(
                    _props.get("takeaway")
                    or _props.get("text")
                    or _props.get("content")
                    or ""
                )

        meta = out.get("meta") if isinstance(out.get("meta"), dict) else {}
        meta = dict(meta)
        meta.setdefault("block_id", block_id)
        meta.setdefault("intent", intent)
        meta.setdefault("src", out.get("src") or "dict")
        dbg_cp("exec_takeaway.candidate_ready_for_selection",
               block_id=block_id,
               src=meta.get("src"),
               gate_reason=meta.get("gate_reason"),
               text_len=len(txt or ""),
               preview=str(txt or "")[:160])
        _status_set(f"exec:{block_id or 'unknown'}", meta)
        _cache_exec_takeaway(block_id, meta, txt, allow_empty=False)
        return txt, meta

    # Fallback: stringify
    txt = str(out)
    meta = {"block_id": block_id, "intent": intent, "src": "coerced", "passed": True}
    dbg_cp("exec_takeaway.candidate_ready_for_selection",
           block_id=block_id,
           src=meta.get("src"),
           gate_reason=meta.get("gate_reason"),
           text_len=len(txt or ""),
           preview=(txt or "")[:160])
    _status_set(f"exec:{block_id or 'unknown'}", meta)
    _cache_exec_takeaway(block_id, meta, txt, allow_empty=False)
    return txt, meta

def _mark_exec_rendered(block: Optional[Dict[str, Any]]) -> bool:
    """Return True when ET for this block was already rendered in current pass."""
    if not isinstance(block, dict):
        return False
    if block.get("_exec_rendered"):
        return True
    block["_exec_rendered"] = True
    return False

def _force_cold_audit_run_enabled() -> bool:
    try:
        return bool(st.session_state.get("__cot_force_cold_audit_run"))
    except Exception:
        return False


def _force_cold_exec_run_enabled() -> bool:
    try:
        return bool(
            st.session_state.get("__cot_force_cold_audit_run")
            or st.session_state.get("__cot_force_cold_exec_run")
            or _exec_force_cold_pending()
        )
    except Exception:
        return False


def _force_cold_overview_run_enabled() -> bool:
    try:
        return bool(
            st.session_state.get("__cot_force_cold_audit_run")
            or st.session_state.get("__cot_force_cold_overview_run")
        )
    except Exception:
        return False


def _clear_force_cold_run_flags() -> None:
    try:
        for key in [
            "__cot_force_cold_audit_run",
            "__cot_force_cold_exec_run",
            "__cot_force_cold_overview_run",
            _EXEC_FORCE_COLD_REQ_KEY,
            _EXEC_FORCE_COLD_DONE_KEY,
        ]:
            st.session_state.pop(key, None)
    except Exception:
        return


def _consume_force_cold_exec_run_flag() -> None:
    try:
        st.session_state.pop("__cot_force_cold_exec_run", None)
        _mark_exec_force_cold_consumed()
    except Exception:
        return


def _consume_force_cold_overview_run_flag() -> None:
    try:
        st.session_state.pop("__cot_force_cold_overview_run", None)
    except Exception:
        return


def _clear_exec_runtime_caches(clear_debug_meta: bool = True) -> None:
    try:
        keys = [
            "__cot_exec_batch_cache_v1",
            "__cot_exec_final_cache_v1",
            "__cot_exec_cold_audit_block_stats_v1",
        ]
        if clear_debug_meta:
            keys.extend([
                _EXEC_CACHE_KEY,
                _EXEC_META_KEY,
                _EXEC_ERR_KEY,
                "exec_takeaway_cache",
            ])
        for key in keys:
            st.session_state.pop(key, None)
    except Exception:
        return


def _clear_overview_interp_runtime_cache() -> None:
    try:
        for key in [
            "__cot_overview_interp_cache_v1",
            "__cot_overview_last_render_key",
            "__cot_overview_last_repeat_log_key",
        ]:
            st.session_state.pop(key, None)
    except Exception:
        return


def _request_exec_cold_run() -> None:
    try:
        _clear_exec_runtime_caches(clear_debug_meta=True)
        st.session_state[_EXEC_CP_KEY] = []
        st.session_state["__cot_exec_cache_cleared_notice"] = True
        st.session_state["__cot_force_cold_exec_run"] = True
        _req_id = _mark_exec_force_cold_requested()
        _set_exec_last_run_mode("pending_force_cold", request_id=_req_id)
    except Exception:
        return


def _request_overview_cold_run() -> None:
    try:
        _clear_overview_interp_runtime_cache()
        st.session_state[_EXEC_CP_KEY] = []
        st.session_state["__cot_overview_cache_cleared_notice"] = True
        st.session_state["__cot_force_cold_overview_run"] = True
    except Exception:
        return

# --- wrap rendering helpers for backward-compatible call styles ---
_core_render_exec_takeaway = render_exec_takeaway
_EXEC_FINAL_CACHE_ENGINE_VERSION = "v18_exec_start_end_universal_fix_1"

def _set_exec_final(
    block: Optional[Dict[str, Any]],
    *,
    text: str,
    src: str = "unknown",
    gate_reason: str = "",
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the single ET source-of-truth into block['_exec_final'].

    Runtime-safe helper introduced by controlled rewrite:
    - keeps one canonical payload dict,
    - synchronizes legacy convenience fields still used in parts of this file,
    - does not render anything.
    """
    _meta_extra = dict(meta_extra or {})

    if not isinstance(block, dict):
        return {
            "text": str(text or ""),
            "src": str(src or "unknown"),
            "path": "block._exec_final",
            "gate_reason": str(gate_reason or ""),
            "meta": {"block_id": None, "src": str(src or "unknown"), **_meta_extra},
            **_meta_extra,
        }

    payload = {
        "text": str(text or ""),
        "src": str(src or "unknown"),
        "path": "block._exec_final",
        "gate_reason": str(gate_reason or ""),
        "meta": {
            "block_id": block.get("id"),
            "src": str(src or "unknown"),
            **_meta_extra,
        },
        **_meta_extra,
    }

    block["_exec_final"] = payload

    # compatibility fields kept only as soft fallback
    block["exec"] = payload["text"]
    block["_exec_source"] = payload["meta"]["src"]

    # legacy duplicates removed — source-of-truth stays in _exec_final
    block.pop("_exec_path", None)
    block.pop("_exec_gate_reason", None)

    return payload


def _exec_final_meta_extra_from_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    keys = [
        "origin_stage",
        "contract_mode",
        "used_repair",
        "hard_reasons",
        "soft_reasons",
        "selected_by",
        "selection_reason",
        "score_llm",
        "score_det",
        "llm_bonus",
        "det_bonus",
    ]
    out: Dict[str, Any] = {}
    for key in keys:
        val = payload.get(key, meta.get(key))
        if val is not None and val != "":
            out[key] = val
    return out


_upgrade_value_interp_to_executive_base = None


def _early_upgrade_value_interp_to_executive_v2(out: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    out = _upgrade_value_interp_to_executive_base(out, stats)
    out = dict(out or {})
    stats = stats if isinstance(stats, dict) else {}

    peak_month_raw = _first_not_none(stats.get("peak_month"), stats.get("peak_month_label"))
    peak_month = _format_month_pl(str(peak_month_raw)) if peak_month_raw else None
    peak_window = f"przed szczytem w {peak_month}" if peak_month else "przed szczytem"

    recs = [str(x or "").strip() for x in list(out.get("recommendations") or []) if str(x or "").strip()]

    def _is_soft_value_recommendation(row: str) -> bool:
        rl = str(row or "").lower()
        return not any(
            token in rl
            for token in [
                "zabezpiecz",
                "ogranicz",
                "przenieś",
                "przenies",
                "ekspozycj",
                "dostępno",
                "dostepno",
                "fill-rate",
                "fill rate",
                "aktywacj",
                "m/m",
                "rotacj",
            ]
        )

    if (len(recs) < 2) or all(_is_soft_value_recommendation(r) for r in recs):
        recs = [
            f"Zabezpiecz dostępność i ekspozycję {peak_window}, monitorując odchylenie m/m oraz fill-rate w tygodniach poprzedzających pik.",
            "Poza oknem szczytu ogranicz szeroką aktywację i przenieś wsparcie do miesięcy z najwyższą rotacją, aby nie rozpraszać budżetu.",
        ]

    out["recommendations"] = recs[:3]
    if not str(out.get("recommendation") or "").strip():
        out["recommendation"] = out["recommendations"][0]

    return out


_early_upgrade_value_interp_to_executive_v2_ref = _early_upgrade_value_interp_to_executive_v2


def _exec_llm_target_for_block(block_id: str, contract_mode: str = "full") -> float:
    bid = str(block_id or "")
    mode = str(contract_mode or "full")
    if bid in {"cot__seasonality", "cot__start_end", "cot__concentration"}:
        return 0.85
    if bid == "cot__winners_losers":
        return 0.80
    if bid == "cot__mix_share_topN":
        return 0.60 if mode in {"reduced", "sparse"} else 0.80
    return 0.80


def _exec_audit_row_from_final_payload(
    block_id: str,
    payload: Optional[Dict[str, Any]],
    stats: Optional[Dict[str, Any]] = None,
    audit_path: str = "cold_path",
) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    stats = stats if isinstance(stats, dict) else {}
    src = str(
        payload.get("src")
        or ((payload.get("meta") or {}).get("src") if isinstance(payload.get("meta"), dict) else "")
        or "unknown"
    )
    profile = _build_exec_contract_profile(str(block_id or ""), stats)
    contract_mode = str(
        payload.get("contract_mode")
        or ((payload.get("meta") or {}).get("contract_mode") if isinstance(payload.get("meta"), dict) else "")
        or profile.get("contract_mode")
        or "full"
    )
    used_repair = bool(
        payload.get("used_repair")
        if payload.get("used_repair") is not None
        else src.startswith("llm_repair")
    )
    hard_reasons = payload.get("hard_reasons")
    soft_reasons = payload.get("soft_reasons")
    return {
        "block_id": str(block_id or ""),
        "final_src": src,
        "is_llm": _is_llm_exec_source(src),
        "is_det": src.startswith("deterministic_"),
        "contract_mode": contract_mode,
        "used_repair": used_repair,
        "hard_reasons": list(hard_reasons or []),
        "soft_reasons": list(soft_reasons or []),
        "selected_by": str(payload.get("selected_by") or payload.get("selection_reason") or ""),
        "audit_path": str(audit_path or "cold_path"),
    }


def _inspect_exec_final_cache_payload_map(payload_map: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload_map = payload_map if isinstance(payload_map, dict) else {}
    out: Dict[str, Any] = {
        "poisoned": False,
        "reasons": [],
        "blocks_count": 0,
        "llm_count": 0,
        "det_count": 0,
        "final_src_by_block": {},
    }
    reasons: List[str] = []

    for raw_block_id, raw_payload in payload_map.items():
        if not isinstance(raw_payload, dict):
            continue
        text = _normalize_exec_takeaway_text(raw_payload.get("text"))
        if not text:
            continue

        block_id = str(raw_block_id or "")
        meta = raw_payload.get("meta") if isinstance(raw_payload.get("meta"), dict) else {}
        src = str(raw_payload.get("src") or meta.get("src") or "unknown")
        origin_stage = str(raw_payload.get("origin_stage") or meta.get("origin_stage") or "")
        selected_by = str(raw_payload.get("selected_by") or raw_payload.get("selection_reason") or meta.get("selected_by") or meta.get("selection_reason") or "")
        gate_reason = str(raw_payload.get("gate_reason") or "")

        out["blocks_count"] = int(out.get("blocks_count") or 0) + 1
        out["final_src_by_block"][block_id] = src
        if _is_llm_exec_source(src):
            out["llm_count"] = int(out.get("llm_count") or 0) + 1
        if src.startswith("deterministic_"):
            out["det_count"] = int(out.get("det_count") or 0) + 1

        if origin_stage == "global_empty_exec_results":
            reasons.append("origin_stage_global_empty_exec_results")
        if selected_by == "global_empty_exec_results":
            reasons.append("selected_by_global_empty_exec_results")
        if gate_reason == "global_empty_exec_results":
            reasons.append("gate_reason_global_empty_exec_results")

    out["reasons"] = sorted(set(reasons))
    out["poisoned"] = bool(out["reasons"])
    return out

def _fallback_from_stats(block: Optional[Dict[str, Any]]) -> str:
    """Deterministic ET fallback from block stats.

    This is the only hard-fallback used by final ET rendering.
    It never reads legacy payload shadows like block["exec"].
    """
    if not isinstance(block, dict):
        return ""
    try:
        _bid = str(block.get("id") or block.get("label") or "")
        _stats = block.get("stats") if isinstance(block.get("stats"), dict) else {}
        _fallback_fn = globals().get("_force_exec_takeaway")
        if callable(_fallback_fn):
            return str(_fallback_fn(_bid, _stats) or "").strip()
    except Exception:
        return ""
    return ""

def _with_dimension_stat_aliases(stats: Any) -> Dict[str, Any]:
    src = dict(stats) if isinstance(stats, dict) else {}
    if not src:
        return {}

    def _present(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return bool(v.strip())
        return True

    def _coalesce(*keys: str) -> Any:
        for key in keys:
            val = src.get(key)
            if _present(val):
                return val
        return None

    def _mirror(keys: List[str], value: Any) -> None:
        if not _present(value):
            return
        for key in keys:
            src.setdefault(key, value)

    leader_value = _coalesce("leader_dimension_value", "top1_dimension_value", "driver_dimension_value", "primary_dimension_value", "leader_category", "top1_category", "driver_category_name", "primary_category")
    winner_value = _coalesce("winner_dimension_value", "winner_dimension_value_non_other", "winner_category", "winner_category_non_other", "biggest_gainer_category", "driver_dimension_value", "driver_category_name")
    loser_value = _coalesce("loser_dimension_value", "loser_dimension_value_non_other", "loser_category", "loser_category_non_other", "biggest_loser_category")
    focus_value = _coalesce("focus_dimension_value", "seasonality_focus_dimension_value", "focus_category", "seasonality_focus_category")
    second_value = _coalesce("second_dimension_value", "second_category")
    generic_value = _coalesce("dimension_value", "category", "label", "name", "segment", "dimension_label", "dimension_member")
    primary_value = _coalesce("primary_dimension_value", "primary_category", "driver_dimension_value", "driver_category_name", "leader_dimension_value", "leader_category", "top1_dimension_value", "top1_category")

    _mirror(["leader_dimension_value", "leader_category"], leader_value)
    _mirror(["top1_dimension_value", "top1_category"], leader_value)
    _mirror(["driver_dimension_value", "driver_category_name"], primary_value or leader_value)
    _mirror(["primary_dimension_value", "primary_category"], primary_value or leader_value)
    _mirror(["winner_dimension_value", "winner_category"], winner_value)
    _mirror(["winner_dimension_value_non_other", "winner_category_non_other"], _coalesce("winner_dimension_value_non_other", "winner_category_non_other", winner_value))
    _mirror(["loser_dimension_value", "loser_category"], loser_value)
    _mirror(["loser_dimension_value_non_other", "loser_category_non_other"], _coalesce("loser_dimension_value_non_other", "loser_category_non_other", loser_value))
    _mirror(["focus_dimension_value", "focus_category", "seasonality_focus_category"], focus_value)
    _mirror(["second_dimension_value", "second_category"], second_value)
    _mirror(["dimension_value", "category"], generic_value or primary_value or leader_value or winner_value or loser_value or focus_value)

    n_values = _coalesce("n_dimension_values", "n_categories")
    if _present(n_values):
        src.setdefault("n_dimension_values", n_values)
        src.setdefault("n_categories", n_values)

    return src


def _pick_dimension_value(stats: dict, fallback: str = "elementu") -> str:
    stats = _with_dimension_stat_aliases(stats)
    for key in [
        "leader_dimension_value",
        "top1_dimension_value",
        "driver_dimension_value",
        "primary_dimension_value",
        "winner_dimension_value",
        "winner_dimension_value_non_other",
        "loser_dimension_value",
        "loser_dimension_value_non_other",
        "focus_dimension_value",
        "second_dimension_value",
        "dimension_value",
        "leader_category",
        "top1_category",
        "driver_category_name",
        "winner_category",
        "winner_category_non_other",
        "biggest_gainer_category",
        "loser_category",
        "loser_category_non_other",
        "biggest_loser_category",
        "seasonality_focus_category",
        "focus_category",
        "category",
    ]:
        val = stats.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return fallback


def _pick_category(stats: dict, fallback: str = "kategorii") -> str:
    return _pick_dimension_value(stats, fallback=fallback)

def _fmt_pct(v: Any) -> str:
    try:
        if v is None:
            return "—"
        fv = float(v)
        if not np.isfinite(fv):
            return "—"
        return f"{fv:.2f}%"
    except Exception:
        return "—"

def _fmt_pp(v: Any) -> str:
    try:
        if v is None:
            return "—"
        fv = float(v)
        if not np.isfinite(fv):
            return "—"
        return f"{fv:.2f} pp"
    except Exception:
        return "—"

def _force_exec_takeaway(block_id: str, stats: dict) -> str:
    stats = _with_dimension_stat_aliases(stats)
    cat = _pick_dimension_value(stats, fallback="elementu")
    if block_id == "cot__seasonality":
        return _format_exec_takeaway_seasonality(stats)
    if block_id == "cot__concentration":
        hhi_start = _first_not_none(stats.get("hhi_start"), stats.get("hhi_t0"), stats.get("hhi_first"))
        hhi_end = _first_not_none(stats.get("hhi_end"), stats.get("hhi_t1"), stats.get("hhi_last"))
        top5_delta_pp = _first_not_none(
            stats.get("top5_delta_pp"),
            stats.get("top5_share_delta_pp"),
            stats.get("top5_delta"),
        )

        s1 = (
            f"Struktura portfela staje się bardziej zależna od rdzenia: HHI zmienił się z {_fmt_float(hhi_start, 2)} "
            f"do {_fmt_float(hhi_end, 2)}, a udział Top-5 przesunął się o {_fmt_pp(top5_delta_pp)}, "
            f"co zwiększa ryzyko nadmiernej koncentracji wyniku w kilku pozycjach portfela."
        )
        s2 = (
            f"Decyzja: w ciągu 1–2 cykli wzmocnij challengery tam, gdzie rdzeń traci udział, "
            f"aby poprawić odporność portfela i ograniczyć ryzyko utraty udziału."
        )
        return f"{s1} {s2}"

    if block_id == "cot__mix_share_topN":
        _mix_stats = _canonical_mix_share_stats(stats)
        topn = int(_first_not_none(_mix_stats.get("topN"), 10) or 10)
        topn_start = _mix_stats.get("topN_start_pct")
        topn_end = _mix_stats.get("topN_end_pct")
        topn_delta = _mix_stats.get("topN_delta_pp")
        leader = str(_first_not_none(_mix_stats.get("leader_category"), _pick_category(stats, fallback="lider")) or "lider")
        leader_delta = _mix_stats.get("leader_delta_pp")

        if leader_delta is not None:
            _leader_clause = f'a lider „{leader}” osłabił się o {_fmt_pp(leader_delta)}, '
        else:
            _leader_clause = f'a lider „{leader}” traci pozycję w miksie, '

        s1 = (
            f"Top-{topn} zmienił się z {_fmt_pct(topn_start)} do {_fmt_pct(topn_end)} "
            f"({_fmt_pp(topn_delta)}), {_leader_clause}"
            f"co oznacza erozję rdzenia portfela i rosnącą presję na obronę lidera."
        )
        s2 = (
            f"Decyzja: w 1–2 cyklach przesuń wsparcie na pozycje wzmacniające rdzeń, "
            f"a dla „{leader}” sprawdź cenę, ekspozycję i rolę asortymentową zanim zwiększysz budżet."
        )
        return f"{s1} {s2}"
    if block_id == "cot__winners_losers":
        winner = _first_not_none(
            stats.get("winner_category"),
            stats.get("winner_category_non_other"),
            stats.get("biggest_gainer_category"),
            stats.get("driver_category_name"),
            "pozycja wzrostowa",
        )
        winner_delta = _first_not_none(
            stats.get("winner_delta_pp"),
            stats.get("winner_delta_pp_non_other"),
            stats.get("biggest_gainer_delta_pp"),
            stats.get("driver_delta_pp"),
        )
        loser = _first_not_none(
            stats.get("loser_category"),
            stats.get("loser_category_non_other"),
            stats.get("biggest_loser_category"),
            "pozycja tracąca",
        )
        loser_delta = _first_not_none(
            stats.get("loser_delta_pp"),
            stats.get("loser_delta_pp_non_other"),
            stats.get("biggest_loser_delta_pp"),
        )

        s1 = (
            f"„{winner}” zyskał {_fmt_pp(winner_delta)}, a „{loser}” stracił {_fmt_pp(loser_delta)}, "
            f"co wskazuje na transfer udziału i przejęcie części popytu przez nowego drivera wzrostu."
        )
        s2 = (
            f"Decyzja: w 1–2 miesiącach zwiększ availability, ekspozycję i wsparcie promocyjne dla „{winner}”, "
            f"a dla „{loser}” sprawdź, czy spadek wynika z ceny, sezonowości czy słabszej obecności na półce, "
            f"aby wykorzystać okno przejęcia popytu i ograniczyć koszt utraty udziału."
        )
        return f"{s1} {s2}"
    if block_id == "cot__start_end":
        winner = _first_not_none(
            stats.get("winner_category"),
            stats.get("winner_category_non_other"),
            stats.get("biggest_gainer_category"),
            stats.get("driver_category_name"),
            "pozycja wzrostowa",
        )
        winner_delta = _first_not_none(
            stats.get("winner_delta_pp"),
            stats.get("winner_delta_pp_non_other"),
            stats.get("biggest_gainer_delta_pp"),
            stats.get("driver_delta_pp"),
        )
        loser = _first_not_none(
            stats.get("loser_category"),
            stats.get("loser_category_non_other"),
            stats.get("biggest_loser_category"),
            "pozycja tracąca",
        )
        loser_delta = _first_not_none(
            stats.get("loser_delta_pp"),
            stats.get("loser_delta_pp_non_other"),
            stats.get("biggest_loser_delta_pp"),
        )

        s1 = (
            f"Między startem i końcem okresu „{winner}” zyskał {_fmt_pp(winner_delta)}, "
            f"a „{loser}” stracił {_fmt_pp(loser_delta)}, co wskazuje na trwałe przesunięcie struktury popytu, "
            f"a nie tylko krótkoterminowy szum."
        )
        s2 = (
            f"Decyzja: zaktualizuj bazową alokację wsparcia i ekspozycji pod nowy układ udziałów, "
            f"zamiast opierać kolejny cykl na historycznym liderze, aby ograniczyć ryzyko utraty udziału przy trwałym przesunięciu popytu."
        )
        return f"{s1} {s2}"
    # Generic safe fallback
    s1 = f"Struktura udziałów zmienia się w czasie: dla „{cat}” widoczna jest mierzalna zmiana na poziomie {_fmt_pct(stats.get('share_start'))} → {_fmt_pct(stats.get('share_end'))}."
    s2 = "Decyzja: Ustal 1–2 priorytety dla rosnących i tracących pozycji oraz monitoruj ich udział/pp w stałej częstotliwości raportowania."
    return f"{s1} {s2}"

def _render_exec_takeaway_final(block: Optional[Dict[str, Any]]) -> None:
    """Render ET only from block['_exec_final'] using canonical payload + hard stats fallback."""
    if not isinstance(block, dict):
        return

    payload = block.get("_exec_final")
    if not isinstance(payload, dict):
        payload = _set_exec_final(
            block,
            text="",
            src="missing_exec_final",
            gate_reason="missing_exec_final",
        )

    _final_text = str(payload.get("text") or "").strip()
    _final_src = str((payload.get("meta") or {}).get("src") or payload.get("src") or "unknown")

    if not _final_text:
        dbg_cp(
            "exec_takeaway.ERROR_empty_final_render",
            block_id=block.get("id"),
            final_src=_final_src,
        )

        _fallback_text = _fallback_from_stats(block)
        if _fallback_text:
            payload = _set_exec_final(
                block,
                text=_fallback_text,
                src="deterministic_fallback",
                gate_reason="empty_final_payload",
            )
            _final_text = str(payload.get("text") or "").strip()
            _final_src = str((payload.get("meta") or {}).get("src") or payload.get("src") or "deterministic_fallback")

            dbg_cp(
                "exec_takeaway.render_safety_fill",
                block_id=block.get("id"),
                final_src=_final_src,
                text_len=len(_final_text),
                preview=_final_text[:160],
            )

    if not _mark_exec_rendered(block):
        render_exec_takeaway(payload)

# ------------------------------------------------------------------
# CONTROLLED REWRITE — EXECUTION LAYER SOURCE OF TRUTH
# ------------------------------------------------------------------

def _build_exec_results(
    exec_by_label: Dict[str, str],
    exec_source_by_id: Optional[Dict[str, str]] = None,
    raw_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    exec_source_by_id = exec_source_by_id or {}
    raw_map = raw_map if isinstance(raw_map, dict) else {}

    for label, txt in (exec_by_label or {}).items():
        text = _normalize_exec_takeaway_text(txt)
        src = str(exec_source_by_id.get(str(label)) or ("llm" if text else ""))
        raw_payload = raw_map.get(str(label)) if isinstance(raw_map, dict) else {}
        raw_payload = raw_payload if isinstance(raw_payload, dict) else {}
        meta = raw_payload.get("meta") if isinstance(raw_payload.get("meta"), dict) else {}

        if not text:
            dbg_cp(
                "exec_takeaway.ERROR_empty_before_set_final",
                block_id=str(label),
                source_state=src,
            )

        results[str(label)] = {
            "text": text,
            "text_len": len(text),
            "src": src,
            "meta": dict(meta),
            "gate_reason": str(meta.get("gate_reason") or ""),
        }

    return results

def render_exec_takeaway(*args, **kwargs):  # type: ignore[override]
    """Thin compatibility wrapper for legacy call sites.

    Uwaga:
    - nie jest source-of-truth dla ET,
    - nie prowadzi własnego payload/debug path,
    - finalny render ET w tym module ma przechodzić przez _render_exec_takeaway_final(...).
    """
    if len(args) == 1 and isinstance(args[0], dict):
        payload = args[0]
        slot = kwargs.get("slot")
        text = str(payload.get("text") or payload.get("takeaway") or "")
        try:
            return _core_render_exec_takeaway(slot, text)
        except TypeError:
            return _core_render_exec_takeaway(text)

    if len(args) == 1 and isinstance(args[0], str) and not kwargs:
        text = args[0]
        try:
            return _core_render_exec_takeaway(None, text)
        except TypeError:
            return _core_render_exec_takeaway(text)

    if len(args) >= 2:
        slot, text = args[0], args[1]
        try:
            return _core_render_exec_takeaway(slot, text)
        except TypeError:
            return _core_render_exec_takeaway(text)

    if "text" in kwargs and isinstance(kwargs.get("text"), str):
        text = kwargs.get("text") or ""
        try:
            return _core_render_exec_takeaway(kwargs.get("slot"), text)
        except TypeError:
            return _core_render_exec_takeaway(text)

    return None

_core_render_guidance = render_guidance

def render_guidance(*args, **kwargs):  # type: ignore[override]
    """Compatibility wrapper.

    Supports:
    - render_guidance(slot, text, label="Guidance")  (core)
    - render_guidance(sens, interpretacja, best_practice)  -> merged text
    """
    label = kwargs.get("label", "Guidance")

    # Legacy call-style used in this file: render_guidance(sens, interpretacja, best_practice)
    if len(args) == 3 and all(isinstance(a, (str, type(None))) for a in args):
        sens, interpretacja, best_practice = args
        parts: list[str] = []
        if isinstance(sens, str) and sens.strip():
            parts.append(f"**Sens / co robi wykres?**\n\n{sens.strip()}")
        if isinstance(interpretacja, str) and interpretacja.strip():
            parts.append(f"**Interpretacja**\n\n{interpretacja.strip()}")
        if isinstance(best_practice, str) and best_practice.strip():
            parts.append(f"**Best practice**\n\n{best_practice.strip()}")
        merged = "\n\n---\n\n".join(parts)

        # Core signatures vary across versions:
        # 1) render_guidance(slot, text, label=...)
        # 2) render_guidance(slot, text)
        # 3) render_guidance(text)
        try:
            return _core_render_guidance(None, merged, label=label)  # type: ignore[arg-type]
        except TypeError:
            try:
                return _core_render_guidance(None, merged)  # type: ignore[arg-type]
            except TypeError:
                return _core_render_guidance(merged)

    # Pass-through to core for other signatures, with safe fallbacks for kw/slot differences
    try:
        return _core_render_guidance(*args, **kwargs)
    except TypeError:
        # drop unsupported kwargs like 'label'
        kw = dict(kwargs)
        kw.pop("label", None)
        try:
            return _core_render_guidance(*args, **kw)
        except TypeError:
            # if core expects text-only, but we got (slot, text, ...)
            if len(args) >= 2:
                try:
                    return _core_render_guidance(args[1], **kw)
                except TypeError:
                    return _core_render_guidance(args[1])
            if len(args) == 1:
                return _core_render_guidance(args[0])
            return None


from data_chat_core.exec_takeaway_llm import get_exec_takeaways_llm
from data_chat_core.llm import llm_fn


# ------------------------------------------------------------------
# LLM Layer Upgrade (MBB grade) — prompts + hard gate only (no metrics/UI logic changes)
# ------------------------------------------------------------------
_core_get_exec_takeaways_llm = get_exec_takeaways_llm  # keep original

_MBB_STRATEGIC_WORDS = [
    "alokacja", "priorytet", "koncentracja", "ryzyko", "szansa", "ekspozycja", "momentum",
    "rentowność", "marża", "mix", "portfel", "trade-off", "dźwignia",
    "transfer udziału", "trwałe przesunięcie", "erozja rdzenia", "obrona lidera",
    "przejęcie popytu", "okno przejęcia popytu", "zależność od rdzenia",
    "dywersyfikacja", "challenger", "zatowarowanie", "szczyt sezonu",
    "błąd planowania", "koszt błędu planowania", "niedoszacowanie popytu",
    "niedostępność", "availability", "rola kategorii", "ekspozycja bazowa",
    "realokacja wsparcia", "plan wsparcia", "rdzeń portfela", "odporność portfela"
]

# --- Minimal debug memory for gate reasons (per-call) ---
_MBB_FAIL_REASONS_BY_CALL_ID: Dict[str, List[str]] = {}


_MBB_BANNED_PHRASES = [
    "może sugerować", "wydaje się", "warto rozważyć", "prawdopodobnie", "należy przeanalizować",
    "ten wykres pokazuje", "w danych widać", "analiza pokazuje"
]
_MBB_DECISION_MARKERS = [

    "dlatego", "więc", "rekomendacja", "decyzja", "priorytet", "alokuj", "zwiększ", "ogranicz",
    "wstrzymaj", "skaluj", "testuj", "zabezpiecz", "przenieś", "utrzymaj"
]

def _mbb_json_dumps_safe(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        try:
            return str(x)
        except Exception:
            return "{}"

def _et_is_nonzero_number(v: Any) -> bool:
    try:
        return bool(np.isfinite(float(v))) and abs(float(v)) > 1e-12
    except Exception:
        return False


def _et_pick_first(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        return v
    return None


def _normalize_exec_takeaway_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _format_exec_delta_pp(value: Any) -> str:
    try:
        value_f = float(value)
    except Exception:
        return ""
    txt = f"{value_f:.2f}".rstrip("0").rstrip(".")
    return txt if txt else str(value_f)


def _postprocess_exec_takeaway_by_block(text: Any, block_id: str, stats: Any) -> str:
    t = _normalize_exec_takeaway_text(text)
    bid = str(block_id or "")
    compact_stats = _compact_exec_stats(bid, stats)
    if not t:
        return t

    if bid == "cot__mix_share_topN":
        mix_stats = _canonical_mix_share_stats(stats)
        leader = _first_not_none(
            mix_stats.get("leader_category"),
            compact_stats.get("leader_category"),
            compact_stats.get("driver_category_name"),
        )
        leader_delta_pp = _first_not_none(
            mix_stats.get("leader_delta_pp"),
            compact_stats.get("leader_delta_pp"),
            compact_stats.get("driver_delta_pp"),
        )
        has_explicit_leader_delta = bool(
            re.search(r"\bo\s*[+-]?\d+(?:[\.,]\d+)?(?:\s*(?:pp|%))?", t, flags=re.IGNORECASE)
        )

        if leader_delta_pp is not None and not has_explicit_leader_delta:
            leader_name = str(leader or "lider").strip()
            delta_txt = _format_exec_delta_pp(leader_delta_pp)
            if delta_txt:
                leader_clause = f'a lider "{leader_name}" zmienil udzial o {delta_txt} pp'
                _parts = re.split(r"(?<=[\.\!\?])\s+", t, maxsplit=1)
                _first = str((_parts[0] if _parts else t) or "").strip()
                _rest = str((_parts[1] if len(_parts) > 1 else "") or "").strip()
                if _first:
                    _first = re.sub(r"[\.\!\?]+\s*$", "", _first).rstrip(",;: ")
                    _first = f"{_first}, {leader_clause}."
                    t = f"{_first} {_rest}".strip() if _rest else _first

    if bid == "cot__start_end" and ("pp" not in t.lower()) and ("%" not in t):
        winner_delta_pp = compact_stats.get("winner_delta_pp")
        loser_delta_pp = compact_stats.get("loser_delta_pp")

        if winner_delta_pp is not None:
            winner_txt = _format_exec_delta_pp(winner_delta_pp)
            t = re.sub(
                r"((?:zyskał|zyskała|zyskało|zyskal|zyskala|zyskalo)\s+)([+-]?\d+(?:[\.,]\d+)?)\b(?!\s*(?:pp|%))",
                rf"\1{winner_txt} pp",
                t,
                count=1,
                flags=re.IGNORECASE,
            )
        if loser_delta_pp is not None:
            loser_txt = _format_exec_delta_pp(loser_delta_pp)
            t = re.sub(
                r"((?:stracił|straciła|straciło|stracil|stracila|stracilo)\s+)([+-]?\d+(?:[\.,]\d+)?)\b(?!\s*(?:pp|%))",
                rf"\1{loser_txt} pp",
                t,
                count=1,
                flags=re.IGNORECASE,
            )

    if bid == "cot__seasonality":
        tn = _norm_pl_token(t)
        has_tradeoff = any(
            token in tn
            for token in [
                "w porownaniu do kategorii",
                "w porownaniu do",
                "wobec kategorii",
                "wobec",
                "vs",
                "wzgledem kategorii",
                "ma wyzsza amplitude niz",
                "ma wieksza amplitude niz",
                "jest wyzsza niz",
                "jest wieksza niz",
                "podczas gdy",
            ]
        )
        if not has_tradeoff:
            focus_cat = str(compact_stats.get("seasonality_focus_category") or "ta kategoria").strip()
            second_cat = str(compact_stats.get("seasonality_second_category") or "").strip()
            share_gap = compact_stats.get("seasonality_share_gap")
            amp_gap = compact_stats.get("seasonality_amplitude_gap")
            top_mode = str(compact_stats.get("seasonality_top_mode") or "").strip().lower()

            tradeoff_clause = ""
            if second_cat:
                try:
                    share_gap_txt = _fmt_seasonality_share(abs(float(share_gap))) if share_gap is not None else ""
                except Exception:
                    share_gap_txt = ""
                try:
                    amp_gap_txt = _fmt_seasonality_amp(abs(float(amp_gap))) if amp_gap is not None else ""
                except Exception:
                    amp_gap_txt = ""

                if top_mode == "cluster" and share_gap_txt and amp_gap_txt:
                    tradeoff_clause = (
                        f'W porownaniu do "{second_cat}" "{focus_cat}" ma wieksza amplitude '
                        f'({amp_gap_txt}) przy zblizonej sile sezonowosci ({share_gap_txt} roznicy), '
                        "wiec koszt bledu planowania jest wyzszy."
                    )
                elif amp_gap_txt:
                    tradeoff_clause = (
                        f'W porownaniu do "{second_cat}" "{focus_cat}" ma wieksza amplitude o {amp_gap_txt}, '
                        "wiec priorytet zatowarowania przed szczytem sezonu powinien byc wyzszy."
                    )
                else:
                    tradeoff_clause = (
                        f'W porownaniu do "{second_cat}" sezonowosc "{focus_cat}" jest silniejsza, '
                        "wiec priorytet zatowarowania przed szczytem sezonu powinien byc wyzszy."
                    )

            if tradeoff_clause:
                t = f"{t} {tradeoff_clause}".strip()

    return _normalize_exec_takeaway_text(t)


def _canonical_mix_share_stats(stats: Any) -> Dict[str, Any]:
    src = _with_dimension_stat_aliases(stats)
    if not src:
        return {}

    def pick(*keys: str) -> Any:
        return _first_not_none(*[src.get(k) for k in keys])

    topn_start_pct = _share_to_pct(pick("topN_share_start_pct", "topN_start_pct", "top10_start_pct", "start_pct", "share_start_pct"))
    topn_end_pct = _share_to_pct(pick("topN_share_end_pct", "topN_end_pct", "top10_end_pct", "end_pct", "share_end_pct"))
    topn_delta_pp = _first_not_none(pick("topN_share_delta_pp", "topN_delta_pp", "top10_delta_pp", "delta_pp"))

    leader_start_pct = _share_to_pct(pick("top1_start_pct", "leader_share_start_pct", "driver_start_pct", "primary_start_pct", "leader_start_share"))
    leader_end_pct = _share_to_pct(pick("top1_end_pct", "leader_share_end_pct", "driver_end_pct", "primary_end_pct", "leader_end_share"))
    leader_delta_pp = _first_not_none(pick("leader_delta_pp", "top1_delta_pp", "leader_share_delta_pp", "driver_delta_pp", "primary_delta_pp"))

    if leader_delta_pp is None and leader_start_pct is not None and leader_end_pct is not None:
        try:
            leader_delta_pp = float(leader_end_pct) - float(leader_start_pct)
        except Exception:
            leader_delta_pp = None

    if topn_delta_pp is None and topn_start_pct is not None and topn_end_pct is not None:
        try:
            topn_delta_pp = float(topn_end_pct) - float(topn_start_pct)
        except Exception:
            topn_delta_pp = None

    return {
        "topN": pick("topN", "top_n", "n_top", "top_n_share"),
        "topN_start_pct": topn_start_pct,
        "topN_end_pct": topn_end_pct,
        "topN_delta_pp": topn_delta_pp,
        "leader_category": pick("leader_dimension_value", "top1_dimension_value", "driver_dimension_value", "primary_dimension_value", "leader_category", "top1_category", "driver_category_name", "primary_category"),
        "leader_dimension_value": pick("leader_dimension_value", "top1_dimension_value", "driver_dimension_value", "primary_dimension_value", "leader_category", "top1_category", "driver_category_name", "primary_category"),
        "leader_start_pct": leader_start_pct,
        "leader_end_pct": leader_end_pct,
        "leader_delta_pp": leader_delta_pp,
    }


def _et_pick_compact_rows(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(stats, dict):
        return rows

    candidate_lists: List[Any] = []
    for k, v in stats.items():
        kl = str(k).lower()
        if isinstance(v, list) and v and isinstance(v[0], dict):
            if any(token in kl for token in ["scorecard", "season", "ranking", "table", "rows"]):
                candidate_lists.append(v)

    for arr in candidate_lists:
        local: List[Dict[str, Any]] = []
        for row in arr[:20]:
            if not isinstance(row, dict):
                continue

            cat = _et_pick_first(
                row.get("category"),
                row.get("Category"),
                row.get("label"),
                row.get("name"),
                row.get("segment"),
            )

            share = _et_pick_first(
                row.get("seasonality_share"),
                row.get("share"),
                row.get("seasonality_share_norm"),
            )

            strength = _et_pick_first(
                row.get("seasonality_strength"),
                row.get("seasonality_strength_alt"),
                row.get("seasonality_weight"),
                row.get("weight"),
                row.get("waga"),
                row.get("score"),
                row.get("seasonality_score"),
            )

            amp = _et_pick_first(
                row.get("seasonality_amplitude"),
                row.get("amplitude"),
                row.get("amp"),
                row.get("avg_amplitude"),
                row.get("amplitude_mean"),
            )

            verdict = _et_pick_first(
                row.get("verdict"),
                row.get("status"),
                row.get("seasonality_verdict"),
                row.get("seasonality_label"),
            )

            if cat is None and share is None and strength is None and amp is None:
                continue

            local.append({
                "category": cat,
                "share": share,
                "weight": share if share is not None else strength,
                "strength": strength,
                "amplitude": amp,
                "verdict": verdict,
            })

        if local:
            def _sort_key(r: Dict[str, Any]) -> float:
                try:
                    return abs(float(r.get("share") if r.get("share") is not None else r.get("weight") or 0.0))
                except Exception:
                    return 0.0

            local.sort(key=_sort_key, reverse=True)
            rows = local
            break

    return rows


def _compact_exec_stats(block_id: str, stats: Any) -> Dict[str, Any]:
    src = _with_dimension_stat_aliases(stats)
    bid = str(block_id or "")
    if not src:
        return {}

    def pick(*keys: str) -> Any:
        return _et_pick_first(*[src.get(k) for k in keys])

    compact: Dict[str, Any] = {}

    if bid == "cot__mix_share_topN":
        compact = _canonical_mix_share_stats(src)
    elif bid == "cot__winners_losers":
        compact = {
            "winner_dimension_value": pick("winner_dimension_value_non_other", "winner_dimension_value", "winner_category_non_other", "biggest_gainer_category", "winner_category", "driver_dimension_value", "driver_category_name"),
            "winner_category": pick("winner_dimension_value_non_other", "winner_dimension_value", "winner_category_non_other", "biggest_gainer_category", "winner_category", "driver_dimension_value", "driver_category_name"),
            "winner_delta_pp": pick("winner_delta_pp_non_other", "biggest_gainer_delta_pp", "winner_delta_pp", "driver_delta_pp"),
            "loser_dimension_value": pick("loser_dimension_value_non_other", "loser_dimension_value", "loser_category_non_other", "biggest_loser_category", "loser_category"),
            "loser_category": pick("loser_dimension_value_non_other", "loser_dimension_value", "loser_category_non_other", "biggest_loser_category", "loser_category"),
            "loser_delta_pp": pick("loser_delta_pp_non_other", "biggest_loser_delta_pp", "loser_delta_pp"),
        }
    elif bid == "cot__start_end":
        compact = {
            "winner_dimension_value": pick("winner_dimension_value_non_other", "winner_dimension_value", "winner_category_non_other", "biggest_gainer_category", "winner_category", "driver_dimension_value", "driver_category_name"),
            "winner_category": pick("winner_dimension_value_non_other", "winner_dimension_value", "winner_category_non_other", "biggest_gainer_category", "winner_category", "driver_dimension_value", "driver_category_name"),
            "winner_delta_pp": pick("winner_delta_pp_non_other", "biggest_gainer_delta_pp", "winner_delta_pp", "driver_delta_pp"),
            "loser_dimension_value": pick("loser_dimension_value_non_other", "loser_dimension_value", "loser_category_non_other", "biggest_loser_category", "loser_category"),
            "loser_category": pick("loser_dimension_value_non_other", "loser_dimension_value", "loser_category_non_other", "biggest_loser_category", "loser_category"),
            "loser_delta_pp": pick("loser_delta_pp_non_other", "biggest_loser_delta_pp", "loser_delta_pp"),
            "period_start": pick("period_start", "start_label", "date_start"),
            "period_end": pick("period_end", "end_label", "date_end"),
        }
    elif bid == "cot__concentration":
        top3_start = pick("top3_start", "top3_share_start", "top3_t0", "top3_share_start_pct")
        top3_end = pick("top3_end", "top3_share_end", "top3_t1", "top3_share_end_pct")
        top5_start = pick("top5_start", "top5_share_start", "top5_t0", "top5_share_start_pct")
        top5_end = pick("top5_end", "top5_share_end", "top5_t1", "top5_share_end_pct")
        if (not _et_is_nonzero_number(top3_start)) and (not _et_is_nonzero_number(top3_end)) and (
            _et_is_nonzero_number(top5_start) or _et_is_nonzero_number(top5_end)
        ):
            top3_start, top3_end = top5_start, top5_end
        compact = {
            "hhi_start": pick("hhi_start", "hhi_t0", "hhi_first", "hhi"),
            "hhi_end": pick("hhi_end", "hhi_t1", "hhi_last", "hhi"),
            "top3_start_pct": top3_start,
            "top3_end_pct": top3_end,
            "top5_delta_pp": pick("top5_delta_pp", "top5_share_delta_pp", "top5_delta"),
        }
    elif bid == "cot__seasonality":
        rows = _et_pick_compact_rows(src)
        top1 = rows[0] if rows else {}
        top2 = rows[1] if len(rows) > 1 else {}

        focus_share = _et_pick_first(
            pick("seasonality_focus_share", "seasonality_share"),
            top1.get("share"),
            top1.get("weight"),
        )
        second_share = _et_pick_first(
            pick("seasonality_second_share"),
            top2.get("share"),
            top2.get("weight"),
        )

        focus_strength = _et_pick_first(
            pick("seasonality_focus_strength", "seasonality_strength", "seasonality_weight"),
            top1.get("strength"),
            top1.get("weight"),
        )
        second_strength = _et_pick_first(
            pick("seasonality_second_strength"),
            top2.get("strength"),
            top2.get("weight"),
        )

        focus_amp = _et_pick_first(
            pick(
                "seasonality_focus_amplitude",
                "seasonality_amplitude",
                "seasonality_amplitude_mean",
                "amplitude_mean",
                "amp_mean",
            ),
            top1.get("amplitude"),
        )
        second_amp = _et_pick_first(
            pick("seasonality_second_amplitude"),
            top2.get("amplitude"),
        )

        focus_cat = pick("seasonality_focus_category", "focus_category") or top1.get("category")
        second_cat = pick("seasonality_second_category") or top2.get("category")

        focus_verdict = pick("seasonality_focus_verdict", "seasonality_verdict", "verdict") or top1.get("verdict") or "sygnał sezonowy"
        second_verdict = pick("seasonality_second_verdict") or top2.get("verdict") or "sygnał sezonowy"

        share_gap = _et_pick_first(
            pick("seasonality_share_gap"),
            (float(focus_share) - float(second_share)) if (focus_share is not None and second_share is not None) else None,
        )
        amplitude_gap = _et_pick_first(
            pick("seasonality_amplitude_gap"),
            (float(focus_amp) - float(second_amp)) if (focus_amp is not None and second_amp is not None) else None,
        )

        top_mode = _et_pick_first(
            pick("seasonality_top_mode"),
            "cluster" if (share_gap is not None and abs(float(share_gap)) < 0.03 and second_cat) else ("leader" if focus_cat else "amplitude_only"),
        )

        compact = {
            "seasonality_focus_category": focus_cat,
            "seasonality_focus_share": focus_share,
            "seasonality_focus_weight": focus_share,
            "seasonality_focus_strength": focus_strength,
            "seasonality_focus_amplitude": focus_amp,
            "seasonality_focus_verdict": focus_verdict,

            "seasonality_second_category": second_cat,
            "seasonality_second_share": second_share,
            "seasonality_second_weight": second_share,
            "seasonality_second_strength": second_strength,
            "seasonality_second_amplitude": second_amp,
            "seasonality_second_verdict": second_verdict,

            "seasonality_share_gap": share_gap,
            "seasonality_amplitude_gap": amplitude_gap,
            "seasonality_top_mode": top_mode,

            "seasonality_weight_max": pick("seasonality_weight_max", "weight_max") or focus_share,
            "seasonality_weight_min": pick("seasonality_weight_min", "weight_min"),
            "seasonality_metric_col": pick("seasonality_metric_col") or "seasonality_share",
            "seasonality_rows_top2": rows[:2],
        }

    else:
        allow = [
            "total_value", "peak_value", "peak_month", "total_txn", "n_categories",
            "n_dimension_values", "share_start_pct", "share_end_pct", "delta_pp",
            "driver_dimension_value", "driver_category_name", "leader_dimension_value", "dimension_label"
        ]
        compact = {k: src.get(k) for k in allow if k in src}

    cleaned: Dict[str, Any] = {}
    for k, v in compact.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        cleaned[str(k)] = v
    return cleaned


def _build_exec_contract_profile(block_id: str, stats: Any) -> Dict[str, Any]:
    bid = str(block_id or "")
    compact = _compact_exec_stats(bid, stats)
    available: List[str] = []
    missing: List[str] = []
    hard_required: List[str] = []
    mode = "insufficient"

    def _present(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return bool(v.strip())
        try:
            if pd.isna(v):
                return False
        except Exception:
            pass
        return True

    def _mark(name: str, v: Any) -> bool:
        if _present(v):
            available.append(name)
            return True
        missing.append(name)
        return False

    if bid == "cot__mix_share_topN":
        topn_start = compact.get("topN_start_pct")
        topn_end = compact.get("topN_end_pct")
        topn_delta = compact.get("topN_delta_pp")
        leader = compact.get("leader_category")
        leader_delta = compact.get("leader_delta_pp")

        _has_topn_range = _mark("topN_range", topn_start) and _mark("topN_range_end", topn_end)
        _has_topn_delta = _mark("topN_delta_pp", topn_delta)
        _has_leader = _mark("leader_category", leader)
        _has_leader_delta = _mark("leader_delta_pp", leader_delta)

        if (_has_topn_delta or _has_topn_range) and _has_leader and _has_leader_delta:
            mode = "full"
            hard_required = ["topN_delta_or_range", "leader_category"]
        elif (_has_topn_delta or _has_topn_range) and _has_leader:
            mode = "reduced"
            hard_required = ["topN_delta_or_range", "leader_category"]
        elif _has_leader or _has_topn_range or _has_topn_delta:
            mode = "sparse"
            hard_required = ["leader_or_topN_signal"]

    elif bid == "cot__winners_losers":
        _has_winner = _mark("winner_category", compact.get("winner_category"))
        _has_loser = _mark("loser_category", compact.get("loser_category"))
        _has_winner_delta = _mark("winner_delta_pp", compact.get("winner_delta_pp"))
        _has_loser_delta = _mark("loser_delta_pp", compact.get("loser_delta_pp"))

        if _has_winner and _has_loser and _has_winner_delta and _has_loser_delta:
            mode = "full"
            hard_required = ["winner_category", "loser_category"]
        elif _has_winner and _has_loser and (_has_winner_delta or _has_loser_delta):
            mode = "reduced"
            hard_required = ["winner_category", "loser_category"]
        elif (_has_winner or _has_loser) and (_has_winner_delta or _has_loser_delta):
            mode = "sparse"
            hard_required = ["named_shift_signal"]

    elif bid == "cot__start_end":
        _has_winner = _mark("winner_category", compact.get("winner_category"))
        _has_loser = _mark("loser_category", compact.get("loser_category"))
        _has_winner_delta = _mark("winner_delta_pp", compact.get("winner_delta_pp"))
        _has_loser_delta = _mark("loser_delta_pp", compact.get("loser_delta_pp"))
        _has_period_start = _mark("period_start", compact.get("period_start"))
        _has_period_end = _mark("period_end", compact.get("period_end"))

        if _has_winner and _has_loser and _has_winner_delta and _has_loser_delta:
            mode = "full"
            hard_required = ["winner_or_loser_names", "numeric_shift"]
        elif (_has_winner or _has_loser) and (_has_winner_delta or _has_loser_delta):
            mode = "reduced"
            hard_required = ["winner_or_loser_names", "numeric_shift"]
        elif (_has_winner or _has_loser or _has_period_start or _has_period_end):
            mode = "sparse"
            hard_required = ["structural_shift_signal"]

    elif bid == "cot__seasonality":
        _has_focus_cat = _mark("focus_category", compact.get("seasonality_focus_category"))
        _has_focus_amp = _mark("focus_amplitude", compact.get("seasonality_focus_amplitude"))
        _has_second_cat = _mark("second_category", compact.get("seasonality_second_category"))
        _has_second_amp = _mark("second_amplitude", compact.get("seasonality_second_amplitude"))
        _has_focus_strength = _mark("focus_strength", compact.get("seasonality_focus_strength"))

        if _has_focus_cat and _has_focus_amp and _has_second_cat and _has_second_amp:
            mode = "full"
            hard_required = ["focus_category", "focus_amplitude"]
        elif _has_focus_cat and _has_focus_amp and (_has_second_cat or _has_focus_strength):
            mode = "reduced"
            hard_required = ["focus_category", "focus_amplitude"]
        elif _has_focus_cat and (_has_focus_amp or _has_focus_strength):
            mode = "sparse"
            hard_required = ["focus_category", "seasonality_signal"]

    elif bid == "cot__concentration":
        _has_hhi_start = _mark("hhi_start", compact.get("hhi_start"))
        _has_hhi_end = _mark("hhi_end", compact.get("hhi_end"))
        _has_top5_delta = _mark("top5_delta_pp", compact.get("top5_delta_pp"))
        _has_top3_start = _mark("top3_start_pct", compact.get("top3_start_pct"))
        _has_top3_end = _mark("top3_end_pct", compact.get("top3_end_pct"))

        if _has_hhi_start and _has_hhi_end and _has_top5_delta:
            mode = "full"
            hard_required = ["hhi_change"]
        elif (_has_hhi_start and _has_hhi_end) or _has_top5_delta:
            mode = "reduced"
            hard_required = ["hhi_or_top5_signal"]
        elif _has_top3_start or _has_top3_end:
            mode = "sparse"
            hard_required = ["portfolio_concentration_signal"]

    if not compact:
        mode = "insufficient"

    return {
        "block_id": bid,
        "contract_mode": mode,
        "available_anchors": sorted(list(dict.fromkeys(available))),
        "missing_anchors": sorted(list(dict.fromkeys(missing))),
        "hard_required_anchors": sorted(list(dict.fromkeys(hard_required))),
    }


def _exec_contract_prompt_guidance(block_id: str, profile: Optional[Dict[str, Any]]) -> str:
    profile = profile if isinstance(profile, dict) else {}
    mode = str(profile.get("contract_mode") or "full")
    available = ", ".join([str(x) for x in (profile.get("available_anchors") or [])]) or "none"
    missing = ", ".join([str(x) for x in (profile.get("missing_anchors") or [])]) or "none"

    if mode == "full":
        return (
            f"Profil danych: FULL. Dostepne kotwice: {available}. "
            "Uzyj pelnego kontraktu bloku i podaj wszystkie kluczowe liczby bez ostroznosciowego zastrzezenia."
        )
    if mode == "reduced":
        return (
            f"Profil danych: REDUCED. Dostepne kotwice: {available}. Brakujace: {missing}. "
            "Uzyj tylko liczb obecnych w JSON. Nie zgaduj brakujacych kotwic. "
            "Jesli brak jednej liczby wymaganej w wersji full, nazwij mechanizm i decyzje ostroznie, ale nadal executive-style."
        )
    if mode == "sparse":
        return (
            f"Profil danych: SPARSE. Dostepne kotwice: {available}. Brakujace: {missing}. "
            "Oprzyj takeaway na najsilniejszym dostepnym sygnale. Nie inventuj brakujacych liczb. "
            "Wolno dodac ostrozne zdanie typu 'sygnal jest ograniczony' lub 'nie eskaluj budzetu bez potwierdzenia', "
            "ale nadal podaj konkretna decyzje biznesowa."
        )
    return (
        f"Profil danych: INSUFFICIENT. Dostepne kotwice: {available}. Brakujace: {missing}. "
        "Jesli sygnal jest zbyt slaby, napisz ostrozny executive takeaway bez zgadywania brakujacych liczb "
        "i nie eskaluj rekomendacji ponad dostepny sygnal."
    )


def _is_llm_exec_source(src: Any) -> bool:
    s = str(src or "").strip().lower()
    return s.startswith("llm_")


def _has_exec_numeric_placeholder(text: str) -> bool:
    txt = _normalize_exec_takeaway_text(text)
    if not txt:
        return False
    return bool(re.search(r"\bo\s*\u2014(?=[,\s])", txt))

_MBB_ET_SYSTEM = """Pisz po polsku, liczbowo i bardzo rygorystycznie.
Zwróć wyłącznie 2 krótkie zdania jako czysty tekst.
Nie zwracaj JSON.
Nie zwracaj {}.
Nie zwracaj listy.
Nie zwracaj nagłówków ani markdown.
Jeśli model ma zwrócić pusty obiekt, niepewny format albo brak treści, ma mimo to zwrócić 2 krótkie zdania jako czysty tekst.
Nigdy nie zwracaj pustego obiektu zamiast odpowiedzi tekstowej.

WYMAGANIA:
1) Zdanie 1 musi zawierać minimum 2 liczby z przekazanego JSON.
2) Zdanie 1 musi nazwać mechanizm biznesowy właściwy dla bloku.
3) Jeśli blok dotyczy kategorii / winner-loser / seasonality, zdanie 1 musi zawierać nazwę kategorii w cudzysłowie.
4) Jeśli blok dotyczy koncentracji, zdanie 1 ma zaczynać się od „Portfel” albo „Struktura portfela” i nie może używać placeholderów typu „kategoria”, „konsolidacja”, „sprzedaż” jako nazwy drivera.
5) Zdanie 2 musi zaczynać się od 'Decyzja:' i zawierać konkretne działanie operacyjne lub portfelowe.
6) Zdanie 2 musi zawierać why-now: ryzyko utraty udziału / koszt zaniechania / okno przejęcia popytu / koszt błędu planowania / trwałe przesunięcie popytu.
7) Nie zostawiaj pustej odpowiedzi.
8) Jeżeli odpowiedź jest niepewna, nadal zwróć najlepszą możliwą wersję tekstową zamiast {}.
9) Nie używaj odpowiedzi technicznych ani meta-komentarzy o formacie.

DODATKOWE WYMAGANIA BLOKOWE:
- Dla cot__mix_share_topN używaj języka: „erozja rdzenia portfela”, „obrona lidera”, „presja na rdzeń”, „rola asortymentowa”, „przesunięcie wsparcia”.
- Dla cot__mix_share_topN nie kończ rekomendacji ogólnym „zwiększyć ekspozycję w asortymencie”; wskaż kierunek: obrona lidera / przesunięcie wsparcia / sprawdzenie ceny, ekspozycji i roli asortymentowej.
- Dla cot__winners_losers używaj języka: „transfer udziału”, „przejęcie popytu”, „winner”, „loser”, „availability”, „ekspozycja”, „wsparcie promocyjne”, „obecność na półce”.
- Dla cot__winners_losers nie kończ rekomendacji ogólną „realokacją wsparcia”; wskaż winnera i losera oraz nazwij dźwignię po obu stronach.
- Dla cot__winners_losers zdanie 2 ma zawierać dwie konkretne dźwignie: jedną dla winnera (np. availability / ekspozycja / promo), a drugą dla losera (np. cena / sezonowość / obecność na półce / ograniczenie wsparcia).
- Dla cot__start_end używaj języka: „trwałe przesunięcie struktury”, „nowy układ popytu”, „bazowa alokacja”, „obrona udziału”, „przeniesienie wsparcia”.
- Dla cot__start_end nie kończ rekomendacji ogólnym „zwiększyć ekspozycję”; wskaż kierunek: bazowa ekspozycja / realokacja wsparcia / ograniczenie wsparcia przegrywającej kategorii.
- Dla cot__concentration podkreślaj konsekwencję portfelową: „zależność od rdzenia”, „odporność portfela”, „rola challengerów”, „dywersyfikacja”.
- Dla cot__seasonality zawsze wiąż decyzję z kalendarzem szczytu, zapasem lub kosztem błędu planowania.
- Dla cot__seasonality zdanie 1 ma zawierać wyraźny trade-off między dwiema kategoriami albo między amplitudą i ryzykiem niedostępności.
- Dla cot__seasonality zdanie 2 ma wskazać konkretną decyzję operacyjną: kalendarz aktywacji / zwiększenie zapasu / priorytet zatowarowania przed szczytem sezonu.
- Nie używaj ogólników: „tę kategorię”, „asortyment”, „zwiększyć działania marketingowe”, „optymalizacja kanałów”.
- Zamiast tego używaj nazwanej decyzji: „obrona lidera”, „realokacja wsparcia”, „zwiększenie ekspozycji bazowej”, „priorytet zatowarowania”, „wzmocnienie challengerów”."""

def _cot_extract_r2(stats: Any) -> Optional[float]:
    """Best-effort: extract a representative R² from stats dict for tone differentiation."""
    try:
        if isinstance(stats, dict):
            # common patterns: r2, r2_top5, r2_hhi, etc.
            vals = []
            for k, v in stats.items():
                kl = str(k).lower()
                if "r2" in kl:
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            vals.append(fv)
                    except Exception:
                        pass
                # nested
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        kkl = str(kk).lower()
                        if "r2" in kkl:
                            try:
                                fv = float(vv)
                                if np.isfinite(fv):
                                    vals.append(fv)
                            except Exception:
                                pass
            if vals:
                # conservative: take min as "weakest link" confidence proxy
                return float(min(vals))
    except Exception:
        return None
    return None

def _cot_confidence_bucket(r2: Optional[float]) -> str:
    if r2 is None:
        return "unknown"
    try:
        r2 = float(r2)
    except Exception:
        return "unknown"
    if r2 >= 0.70:
        return "high"
    if r2 >= 0.45:
        return "medium"
    return "low"


def _fmt_seasonality_share(v: Any) -> str:
    try:
        if v is None:
            return "—"
        fv = float(v)
        if not np.isfinite(fv):
            return "—"
        return f"{fv:.2f}"
    except Exception:
        return "—"


def _fmt_seasonality_amp(v: Any) -> str:
    try:
        if v is None:
            return "—"
        fv = abs(float(v))
        if not np.isfinite(fv):
            return "—"
        if fv >= 1_000_000:
            return f"{fv/1_000_000:.1f} mln"
        if fv >= 1_000:
            return f"{fv/1_000:.1f} tys."
        return f"{fv:.0f}"
    except Exception:
        return "—"


def _seasonality_activation_scale(v: Any) -> str:
    try:
        if v is None:
            return "umiarkowaną"
        fv = abs(float(v))
        if not np.isfinite(fv):
            return "umiarkowaną"
        if fv >= 120_000:
            return "wysoką"
        if fv >= 60_000:
            return "średnią"
        return "umiarkowaną"
    except Exception:
        return "umiarkowaną"


def _format_exec_takeaway_seasonality(stats: Dict[str, Any]) -> str:
    s = stats or {}
    cat1 = str(_first_not_none(s.get("seasonality_focus_category"), s.get("focus_category"), "kategoria 1") or "kategoria 1").strip()
    cat2 = str(_first_not_none(s.get("seasonality_second_category"), "") or "").strip()
    share1 = _first_not_none(s.get("seasonality_focus_share"), s.get("seasonality_focus_weight"), s.get("seasonality_weight_max"))
    share2 = _first_not_none(s.get("seasonality_second_share"), s.get("seasonality_second_weight"))
    amp1 = _first_not_none(s.get("seasonality_focus_amplitude"), s.get("seasonality_amplitude_mean"))
    amp2 = _first_not_none(s.get("seasonality_second_amplitude"))
    share_gap = _first_not_none(s.get("seasonality_share_gap"))
    amp_gap = _first_not_none(s.get("seasonality_amplitude_gap"))
    mode = str(_first_not_none(s.get("seasonality_top_mode"), "leader") or "leader").strip().lower()

    try:
        share_gap_f = abs(float(share_gap)) if share_gap is not None else None
    except Exception:
        share_gap_f = None
    try:
        amp_gap_f = abs(float(amp_gap)) if amp_gap is not None else None
    except Exception:
        amp_gap_f = None

    share1_txt = _fmt_seasonality_share(share1)
    share2_txt = _fmt_seasonality_share(share2)
    amp1_txt = _fmt_seasonality_amp(amp1)
    amp2_txt = _fmt_seasonality_amp(amp2)
    amp_gap_txt = _fmt_seasonality_amp(amp_gap_f)
    activation_scale = _seasonality_activation_scale(amp1)

    if mode == "cluster" and cat2:
        s1 = (
            f"„{cat1}” i „{cat2}” mają podobnie silny sygnał sezonowy "
            f"({share1_txt} vs {share2_txt}), ale „{cat1}” generuje większą amplitudę popytu "
            f"({amp1_txt} vs {amp2_txt}), więc błąd planowania będzie tam kosztował więcej."
        )
        s2 = (
            f"Decyzja: prowadź wspólny kalendarz aktywacji dla obu kategorii, "
            f"ale większy zapas, ekspozycję i wsparcie promocyjne ustaw dla „{cat1}” przed szczytem sezonu."
        )
        return f"{s1} {s2}"

    if mode == "leader":
        s1 = (
            f"„{cat1}” ma najsilniejszy sygnał sezonowy ({share1_txt}) i amplitudę {amp1_txt}, "
            f"co oznacza, że właśnie tam sezonowość najsilniej steruje skalą popytu."
        )
        s2 = (
            f"Decyzja: ustaw „{cat1}” jako priorytet kalendarza promo, ekspozycji i zatowarowania, "
            f"aby ograniczyć ryzyko niedoszacowania szczytu sezonu."
        )
        return f"{s1} {s2}"

    s1 = f'Najsilniejszy sygnał sezonowy ma „{cat1}” ({share1_txt}), a amplituda wynosi {amp1_txt}.'
    s2 = (
        f'Decyzja: traktuj tę kategorię jako punkt odniesienia dla kalendarza aktywacji i skaluj budżet oraz zapas zgodnie z amplitudą, '
        f'aby ograniczyć ryzyko niedoszacowania szczytu popytu.'
    )
    return f'{s1} {s2}'

def _mbb_has_percent_or_pp(s: str) -> bool:
    sl = (s or "").lower()
    return ("%" in sl) or ("pp" in sl) or bool(re.search(r"\bpp\b", sl))

def _mbb_has_strategic_word(s: str) -> bool:
    sl = (s or "").lower()
    return any(w in sl for w in _MBB_STRATEGIC_WORDS)

def _mbb_contains_banned(s: str) -> bool:
    sl = (s or "").lower()
    return any(p in sl for p in _MBB_BANNED_PHRASES)

def _mbb_stats_expect_percent(stats: Any) -> bool:
    """Heuristic: require %/pp when the block is about shares/mix or when stats contain share-like keys."""
    try:
        js = _mbb_json_dumps_safe(stats).lower()
        return any(k in js for k in ["share", "udz", "pp", "mix", "hhi", "top-3", "top3", "top-5", "top5"])
    except Exception:
        return False

def _mbb_count_numbers(s: str) -> int:
    try:
        return len(re.findall(r"\d+(?:[\.,]\d+)?", s or ""))
    except Exception:
        return 0

def _mbb_validate_exec_takeaway(text_out: str, stats: Any = None) -> Tuple[bool, List[str]]:
    t = (text_out or "").strip()
    reasons: List[str] = []
    if not t:
        return False, ["empty"]

    _block_id = ""
    try:
        if isinstance(stats, dict):
            _block_id = str(
                stats.get("block_id")
                or stats.get("label")
                or stats.get("id")
                or ""
            )
    except Exception:
        _block_id = ""
    _contract_profile = _build_exec_contract_profile(_block_id, stats) if _block_id else {}
    _contract_mode = str((_contract_profile or {}).get("contract_mode") or "full")

    # Guardrail: block obvious placeholder-like percentages that tend to come from a broken numeric payload.
    # We fail validation (→ repair/fallback) instead of crashing the UI.
    if re.search(r"\b0[\.,]0\s*%\b", t):
        reasons.append("zero_percent_placeholder")
    if _mbb_contains_banned(t):
        reasons.append("contains_banned_phrase")

    # Forbid referencing raw 'strength' in seasonality narratives (we standardize on weight+verdict).
    try:
        if isinstance(stats, dict):
            has_strength_key = any(('strength' in str(k).lower()) and ('seasonality' in str(k).lower()) for k in stats.keys())
            has_seasonality_ctx = has_strength_key or ('seasonality_weight' in stats) or ('seasonality_verdict' in stats) or ('seasonality_score' in stats)
            if has_seasonality_ctx and (('siła sezonowości' in t.lower()) or ('sila sezonowosci' in t.lower()) or ('seasonality strength' in t.lower()) or (' strength' in t.lower())):
                reasons.append('seasonality_strength_forbidden')
    except Exception:
        pass

    # 2–3 sentences target
    parts = [p for p in re.split(r"(?<=[\.!\?])\s+", t) if p.strip()]
    if _contract_mode in {"reduced", "sparse"}:
        if len(parts) < 1 or len(parts) > 3:
            reasons.append("sentence_count_not_1_to_3")
    elif len(parts) < 2 or len(parts) > 3:
        reasons.append("sentence_count_not_2_to_3")

    # Sentence 1: must contain at least one numeric anchor
    s1 = parts[0] if parts else t
    if _mbb_count_numbers(s1) < 1:
        reasons.append("numbers_s1<1")

    # Entire ET: must contain at least 2 numeric anchors overall
    _numbers_total = _mbb_count_numbers(t)
    if _contract_mode == "full" and _numbers_total < 2:
        reasons.append("numbers_total<2")
    elif _contract_mode in {"reduced", "sparse"} and _numbers_total < 1:
        reasons.append("numbers_total<1")

    # Require %/pp only when the content is share/mix-like (avoid blocking value-mode narratives)
    _skip_percent_gate = _block_id in {"cot__seasonality"}

    _strategic_gate_exceptions = [
        "transfer udziału",
        "trwałe przesunięcie struktury",
        "okno przejęcia popytu",
        "ryzyko utraty udziału",
        "ryzyka utraty udziału",
        "z powodu ryzyka utraty udziału",
        "aby zminimalizować ryzyko utraty udziału",
        "aby uniknąć ryzyka utraty udziału",
        "ekspozycję bazową",
        "bazową alokację",
        "erozję rdzenia",
        "obrona lidera",
        "rosnącą zależność od rdzenia",
        "koszt błędu planowania",
        "priorytet zatowarowania",
        "niedostępności",
        "przesunięcie popytu",
        "realokacja wsparcia",
        "plan wsparcia",
        "rola kategorii w miksie",
        "rola elementu w miksie",
        "odporność portfela",
    ]

    if _mbb_stats_expect_percent(stats) and not _skip_percent_gate:
        if not _mbb_has_percent_or_pp(t):
            reasons.append("missing_percent_or_pp")

    _reduced_mechanism_proxy = any(
        token in str(t or "").lower()
        for token in [
            "transfer udzia",
            "przejecie popytu",
            "oddawanie udzia",
            "erozj",
            "obrona lidera",
            "zaleznosc od rdzenia",
            "koszt bledu planowania",
            "trwale przesuniecie",
            "odpornosc portfela",
            "dywersyfik",
        ]
    )
    _reduced_action_proxy = any(
        token in str(t or "").lower()
        for token in [
            "decyzja:",
            "dlatego:",
            "rekomendacja:",
            "realokacja",
            "plan wsparcia",
            "alokacja",
            "ekspozyc",
            "availability",
            "zapas",
            "zatowarowanie",
            "sprawdz cene",
            "sprawdz ekspozycje",
            "wzmocnij",
            "utrzymaj",
            "ogranicz",
        ]
    )
    _reduced_caution_proxy = any(
        token in str(t or "").lower()
        for token in [
            "sygnal jest ograniczony",
            "ostroznie",
            "bez eskalacji budzetu",
            "dopiero po potwierdzeniu",
            "najpierw sprawdz",
            "na probe",
            "pilota",
        ]
    )

    if not _mbb_has_strategic_word(t):
        if not any(
            phrase in str(t or "").lower()
            for phrase in _strategic_gate_exceptions
        ):
            if not (_contract_mode in {"reduced", "sparse"} and (_reduced_mechanism_proxy or _reduced_caution_proxy)):
                reasons.append("missing_strategic_word")

    # decision marker
    if not any(m in (t or "").lower() for m in _MBB_DECISION_MARKERS):
        if not (_contract_mode in {"reduced", "sparse"} and _reduced_action_proxy):
            reasons.append("missing_decision_marker")

    # Persist primary fail reasons (per call) so the debug collector can show why a repaired ET was gated.
    try:
        if isinstance(stats, dict):
            _cid = stats.get('_dbg_call_id')
            if _cid and reasons:
                _MBB_FAIL_REASONS_BY_CALL_ID[str(_cid)] = list(reasons)
    except Exception:
        pass

    return (len(reasons) == 0), reasons


def _validate_exec_takeaway_by_block(text: str, block_id: str, stats: Dict[str, Any]) -> Tuple[bool, List[str]]:
    stats = _with_dimension_stat_aliases(stats)
    if block_id == "cot__mix_share_topN":
        stats = _canonical_mix_share_stats(stats)
    elif block_id in {"cot__winners_losers", "cot__start_end"}:
        stats = _compact_exec_stats(block_id, stats)

    t = _normalize_exec_takeaway_text(text)
    tl = t.lower()
    tn = _norm_pl_token(t)
    reasons: List[str] = []

    def _has_num() -> bool:
        return bool(re.search(r"\d", t))

    def _has_any(words: List[str]) -> bool:
        return any(w in tl for w in words)

    def _has_any_norm(words: List[str]) -> bool:
        return any(w in tn for w in words)

    def _pick_stat(*keys: str) -> Any:
        for k in keys:
            v = stats.get(k)
            if v is not None and v != "":
                return v
        return None

    if not t:
        return False, ["empty_text"]

    if block_id == "cot__winners_losers" and bool(st.session_state.get("__cot_exec_dbg_on")):
        dbg_cp(
            "exec_takeaway.block_validator_input",
            block_id=block_id,
            from_path="_validate_exec_takeaway_by_block",
            text_len=len(t),
            preview=t[:220],
            stats_keys=list((stats or {}).keys())[:12],
        )

    if block_id == "cot__mix_share_topN":
        leader = _pick_stat("leader_category", "top1_category", "driver_category_name")
        topn = int(_pick_stat("topN", "top_n", "n_top", "top_n_share") or 10)
        topn_delta = _pick_stat("topN_delta_pp", "topN_share_delta_pp", "top10_delta_pp")
        leader_delta = _pick_stat("leader_delta_pp", "top1_delta_pp", "driver_delta_pp")
        has_placeholder_delta = bool(re.search("\\bo\\s*\u2014(?=[,\\s])", t))
        has_explicit_leader_delta = bool(re.search(r"\bo\s*[+-]?\d", t))
        has_topn_anchor = any(
            token in tl
            for token in [
                f"top-{topn}",
                f"top {topn}",
                "top-n",
                "top n",
            ]
        )
        has_topn_delta_text = has_topn_anchor and bool(
            re.search(r"top[\s-]*(?:\d+|n).{0,80}(?:pp|%)", tl)
        )
        has_leader_position_phrase = any(
            token in tl
            for token in [
                "lider traci pozycje w miksie",
                "lider traci pozycje",
                "traci pozycje w miksie",
                "oslabienie lidera",
                "slabnie w miksie",
                "traci pozycje lidera",
            ]
        )

        has_numbers = bool(re.search(r"\d", t or ""))

        has_driver = any(
            k in (t or "").lower()
            for k in [
                "kategoria",
                "segment",
                "rdzeń",
                "lider",
                "udział",
            ]
        )
        if leader and str(leader).lower() in tl:
            has_driver = True

        has_decision = any(
            k in (t or "").lower()
            for k in [
                "obrona lidera",
                "realokacja wsparcia",
                "realokuj wsparcie",
                "przesuń wsparcie",
                "plan wsparcia dla rdzenia",
                "rola kategorii w miksie",
                "rola elementu w miksie",
                "sprawdź cenę, ekspozycję i rolę asortymentową",
                "sprawdź cenę",
                "sprawdź ekspozycję",
                "rola asortymentowa",
                "wzmocnienie challengerów",
                "wzmocnij challengerów",
                "ograniczenie wsparcia ogona",
                "ogranicz wsparcie ogona",
                "zwiększenie ekspozycji bazowej",
                "zwiększ ekspozycję bazową"
            ]
        ) or any(
            k in tn
            for k in [
                "obrona lidera",
                "obrony lidera",
                "obrone lidera",
                "przeprowadz obrone lidera",
                "realokacj wspar",
                "realokuj wspar",
                "przesun wspar",
                "rola kategorii w miksie",
                "rola elementu w miksie",
                "rola asortymentow",
                "ekspozycj bazow",
            ]
        )

        has_soft_generic_mix_decision = any(
            k in (t or "").lower()
            for k in [
                "wzmocnienie ekspozycji",
                "wzmocnij ekspozycję",
                "zwiększenie ekspozycji",
                "zwiększ ekspozycję",
                "należy zwiększyć ekspozycję",
                "ekspozycja w asortymencie",
                "zwiększyć ekspozycję w asortymencie",
                "zwiększenie ekspozycji w asortymencie",
                "wzmocnienie asortymentu",
                "wzmocnij asortyment"
            ]
        )

        has_mechanism = any(
            k in (t or "").lower()
            for k in [
                "co sugeruje", "co wskazuje", "co oznacza", "mechanizm",
                "erozja rdzenia", "osłabienie lidera", "fragmentacja",
                "presja na obronę lidera", "obronę lidera", "rdzeń portfela"
            ]
        )

        has_why_now = any(
            k in (t or "").lower()
            for k in [
                "ryzyko utraty udziału",
                "ryzyka utraty udziału",
                "z powodu ryzyka utraty udziału",
                "aby przeciwdziałać ryzyku utraty udziału",
                "aby zminimalizować ryzyko utraty udziału",
                "aby uniknąć ryzyka utraty udziału",
                "koszt zaniechania",
                "koszt oddawania udziału",
                "presja na obronę lidera",
                "erozję rdzenia portfela",
                "erozja rdzenia portfela"
            ]
        )

        if not has_why_now:
            reasons.append("mix_missing_why_now")
        if not has_mechanism:
            has_mechanism = _has_any(["erozja rdzenia", "osłabienie lidera", "fragmentacja", "presję na obronę", "obronę lidera", "rdzeń portfela"])
        if not has_numbers:
            reasons.append("mix_missing_numeric_anchor")
        if not has_driver:
            reasons.append("mix_missing_leader_name")
        if topn_delta is None and leader_delta is None:
            reasons.append("mix_missing_stats_contract")
        if topn_delta is not None and not has_topn_delta_text:
            reasons.append("mix_missing_topn_delta_text")
        if has_placeholder_delta:
            reasons.append("mix_placeholder_delta")
        if leader_delta is not None and not has_explicit_leader_delta:
            reasons.append("mix_missing_leader_delta_text")
        if leader_delta is None and not has_leader_position_phrase:
            reasons.append("mix_missing_leader_position_phrase")
        if not has_mechanism:
            reasons.append("mix_missing_mechanism")
        if not has_decision:
            reasons.append("mix_missing_decision_direction")
        if has_soft_generic_mix_decision:
            reasons.append("mix_soft_generic_decision")

    elif block_id == "cot__winners_losers":
        winner = _pick_stat("winner_category", "winner_category_non_other", "biggest_gainer_category", "driver_category_name")
        loser = _pick_stat("loser_category", "loser_category_non_other", "biggest_loser_category")
        winner_delta = _pick_stat("winner_delta_pp", "winner_delta_pp_non_other", "biggest_gainer_delta_pp", "driver_delta_pp")
        loser_delta = _pick_stat("loser_delta_pp", "loser_delta_pp_non_other", "biggest_loser_delta_pp")
        _wl_has_mechanism = _has_any([
            "transfer udziału",
            "transferu udziału",
            "transfer udziałów",
            "przejęcie popytu",
            "przejęciu popytu",
            "przejęciem popytu",
            "realokacja wsparcia",
            "okno przejęcia popytu",
            "koszt utraty udziału",
            "ryzyko utraty udziału",
            "ryzyka utraty udziału",
            "sezonowy driver",
            "trwałe przesunięcie popytu",
        ])

        if not _has_num():
            reasons.append("wl_missing_numeric_anchor")
        if winner and str(winner).lower() not in tl:
            reasons.append("wl_missing_winner_name")
        if loser and str(loser).lower() not in tl:
            reasons.append("wl_missing_loser_name")
        if winner_delta is None or loser_delta is None:
            reasons.append("wl_missing_winner_loser_delta")
        if not _wl_has_mechanism:
            reasons.append("wl_missing_mechanism")
        _wl_has_winner_lever = _has_any([
            "availability",
            "dostępn",
            "dostepn",
            "promo",
            "promoc",
            "ekspozycj",
            "wsparcie promocyjne",
            "obecność na półce",
            "realokacja wsparcia",
            "realokuj wsparcie",
            "alokacja wsparcia",
            "przesuń wsparcie",
            "plan wsparcia"
        ])

        _wl_has_loser_lever = _has_any([
            "cena",
            "cen",
            "obniż",
            "sezonowość",
            "sezonow",
            "obecność na półce",
            "availability",
            "dostępn",
            "dostepn",
            "ograniczenie wsparcia",
            "ogranicz wsparcie",
            "rola kategorii",
            "rola asortymentowa",
            "price-pack",
            "price pack"
        ])

        if not (_wl_has_winner_lever and _wl_has_loser_lever):
            reasons.append("wl_missing_decision_direction")

    elif block_id == "cot__seasonality":
        if not _has_num():
            reasons.append("seasonality_missing_numeric_anchor")

        has_seasonality_mechanism = _has_any([
            "koszt błędu planowania",
            "błąd planowania",
            "ryzyko niedoszacowania popytu",
            "ryzyko niedostępności",
            "napięcie dostępności",
            "priorytet zatowarowania przed szczytem sezonu",
            "zatowarowanie przed szczytem sezonu",
            "zapas przed szczytem",
            "szczyt sezonu"
        ])

        has_seasonality_tradeoff = _has_any([
            "w porównaniu do kategorii",
            "wobec kategorii",
            "vs",
            "względem kategorii",
            "ma wyższą amplitudę niż",
            "ma większą amplitudę niż",
            "niż kategoria",
            "podczas gdy",
            "w porównaniu do",
            "wobec",
            "jest wyższa niż",
            "jest większa niż",
            "niż \"",
            "priorytetem zatowarowania przed szczytem sezonu"
        ])

        has_seasonality_tradeoff = has_seasonality_tradeoff or _has_any_norm([
            "w porownaniu do kategorii",
            "w porownaniu do",
            "wobec kategorii",
            "wobec",
            "vs",
            "wzgledem kategorii",
            "ma wyzsza amplitude niz",
            "ma wieksza amplitude niz",
            "niz kategoria",
            "podczas gdy",
            "jest wyzsza niz",
            "jest wieksza niz",
            'niz "',
            "priorytet zatowarowania przed szczytem sezonu",
        ])

        if not has_seasonality_mechanism:
            reasons.append("seasonality_missing_mechanism")

        if not has_seasonality_tradeoff:
            reasons.append("seasonality_missing_tradeoff")

        if not _has_any([
            "kalendarz aktywacji",
            "zapas",
            "zatowarowanie",
            "priorytet zatowarowania",
            "zwiększyć zapas",
            "zwiększenie zapasu",
            "zwiększamy zapas",
            "przed szczytem sezonu",
            "priorytet zatowarowania przed szczytem sezonu",
            "zwiększyć zapas dla kategorii",
            "zwiększamy zapas dla kategorii",
            "zwiększenie zapasu dla kategorii",
            "zwiększyć zapas w kategorii",
            "zwiększamy zapas w kategorii",
            "zwiększenie zapasu w kategorii",
            "zwiększyć zatowarowanie",
            "zwiększamy zatowarowanie",
            "zwiększenie zatowarowania"
        ]):
            reasons.append("seasonality_missing_decision_direction")

    elif block_id == "cot__start_end":
        if not _has_num():
            reasons.append("start_end_missing_numeric_anchor")
        if not _has_any([
            "trwał", "regime shift", "nowy układ kategorii",
            "przesunięcie struktury", "nie tylko szum", "trwałe przesunięcie popytu"
        ]):
            reasons.append("start_end_missing_mechanism")
        if not _exec_has_start_end_direction_signal(t):
            reasons.append("start_end_missing_decision_direction")

    elif block_id == "cot__concentration":
        if not _has_num():
            reasons.append("concentration_missing_numeric_anchor")
        if not _has_any([
            "zależność od rdzenia", "odporność portfela", "dywersyfik",
            "challenger", "równowaga rdzenia",
            "bardziej zależna od rdzenia",
            "staje się bardziej zależna od rdzenia",
            "rosnąca zależność od rdzenia",
            "konsolidacja portfela",
            "konsoliduje się",
            "dywersyfikuje się",
            "koncentracja portfela",
            "rdzeń portfela"
        ]):
            reasons.append("concentration_missing_mechanism")
        if not _has_any([
            "challenger",
            "challengerów",
            "challengery",
            "ograniczyć ryzyko",
            "ograniczenie zależności",
            "ogranicz zależność od rdzenia",
            "dywersyfik",
            "dywersyfikację portfela",
            "zwiększ dywersyfikację portfela",
            "realokacja wsparcia na challengerów",
            "realokuj wsparcie na challengerów",
            "realokacja wsparcia w kierunku challengerów",
            "wzmocnij challengerów",
            "zwiększ ekspozycję challengerów",
            "zwiększ ekspozycję na challengerów",
            "wzmocnienie challengerów",
            "rola challengerów",
            "odporność portfela",
            "zmniejszyć zależność od rdzenia",
            "przesunięcie wsparcia na challengerów",
            "w celu obrony lidera",
            "w celu poprawy odporności portfela",
            "w celu ograniczenia zależności od rdzenia",
            "dla poprawy odporności portfela",
            "dla ograniczenia zależności od rdzenia",
            "dla wzmocnienia challengerów"
        ]):
            reasons.append("concentration_missing_decision_direction")
        if _has_any([
            "\"kategoria\"",
            "\"sprzedaż\"",
            "\"konsolidacja\"",
            "\"koncentracja\""
        ]):
            reasons.append("concentration_placeholder_driver")

    reasons = list(dict.fromkeys(reasons))
    return (len(reasons) == 0, reasons)


_EXEC_BLOCK_SOFT_REASONS = {
    "mix_missing_why_now",
    "mix_missing_mechanism",
    "mix_missing_decision_direction",
    "mix_soft_generic_decision",
    "wl_missing_mechanism",
    "wl_missing_decision_direction",
    "seasonality_missing_mechanism",
    "seasonality_missing_tradeoff",
    "seasonality_missing_decision_direction",
    "start_end_missing_mechanism",
    "start_end_missing_decision_direction",
    "concentration_missing_mechanism",
    "concentration_missing_decision_direction",
}


def _validate_exec_takeaway_by_block_detail(text: str, block_id: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    stats = _with_dimension_stat_aliases(stats)
    strict_ok, reasons = _validate_exec_takeaway_by_block(text, block_id, stats)
    profile = _build_exec_contract_profile(block_id, stats)
    contract_mode = str(profile.get("contract_mode") or "full")
    text_norm = _norm_pl_token(_normalize_exec_takeaway_text(text))

    dynamic_soft_reasons = set(_EXEC_BLOCK_SOFT_REASONS)
    if block_id == "cot__mix_share_topN":
        if contract_mode in {"reduced", "sparse"}:
            dynamic_soft_reasons.add("mix_missing_leader_position_phrase")
        if contract_mode == "sparse":
            dynamic_soft_reasons.add("mix_missing_topn_delta_text")
    elif block_id == "cot__winners_losers":
        if contract_mode in {"reduced", "sparse"}:
            dynamic_soft_reasons.add("wl_missing_winner_loser_delta")
    elif block_id == "cot__seasonality":
        if contract_mode == "sparse":
            dynamic_soft_reasons.add("seasonality_missing_tradeoff")
    elif block_id == "cot__concentration":
        if contract_mode == "sparse":
            dynamic_soft_reasons.add("concentration_missing_decision_direction")

    hard_reasons = [r for r in (reasons or []) if r not in dynamic_soft_reasons]
    soft_reasons = [r for r in (reasons or []) if r in dynamic_soft_reasons]

    if block_id == "cot__mix_share_topN" and text_norm:
        mix_contextual_direction = any(
            token in text_norm
            for token in [
                "obrona lidera",
                "obrony lidera",
                "obrone lidera",
                "przeprowadz obrone lidera",
                "w celu obrony lidera",
                "dla obrony lidera",
                "realokacja wsparcia",
                "realokuj wsparcie",
                "przesun wsparcie",
                "rdzen portfela",
                "rola kategorii w miksie",
                "rola elementu w miksie",
                "rola asortymentowa",
                "ekspozycja bazowa",
                "ekspozycji bazowej",
            ]
        )
        mix_named_mechanism = any(
            token in text_norm
            for token in [
                "traci pozycje w miksie",
                "traci pozycje lidera",
                "utrata pozycji lidera",
                "top-n spadl",
                "spadek top-n",
                "spadek top n",
                "rdzen portfela",
                "obrony lidera",
            ]
        )
        mix_topn_anchor = any(
            token in text_norm
            for token in [
                "top-",
                "top ",
                "top-n",
                "top n",
            ]
        )
        if mix_contextual_direction:
            hard_reasons = [r for r in hard_reasons if r != "mix_missing_decision_direction"]
            soft_reasons = [
                r for r in soft_reasons
                if r not in {"mix_missing_decision_direction", "mix_soft_generic_decision"}
            ]
        if mix_named_mechanism:
            hard_reasons = [r for r in hard_reasons if r != "mix_missing_mechanism"]
            soft_reasons = [r for r in soft_reasons if r != "mix_missing_mechanism"]
        if mix_contextual_direction and mix_named_mechanism and mix_topn_anchor:
            hard_reasons = [r for r in hard_reasons if r != "mix_missing_leader_delta_text"]
            if "mix_missing_leader_delta_text" in (reasons or []) and "mix_missing_leader_delta_text" not in soft_reasons:
                soft_reasons.append("mix_missing_leader_delta_text")
        reasons = list(dict.fromkeys(hard_reasons + soft_reasons))

    strict_ok = not bool(hard_reasons or soft_reasons)

    if contract_mode == "insufficient" and not hard_reasons:
        hard_reasons.append("insufficient_stats_contract")
        strict_ok = False

    return {
        "strict_ok": bool(strict_ok),
        "hard_ok": not bool(hard_reasons),
        "soft_ok": not bool(soft_reasons),
        "hard_reasons": hard_reasons,
        "soft_reasons": soft_reasons,
        "reasons": list(reasons or []),
        "contract_profile": profile,
    }


def _mbb_postprocess_exec_takeaway(txt: str, stats: dict) -> str:
    """Deterministic safety net for Executive Takeaway: enforce >=2 numbers in first sentence."""
    stats = stats or {}
    if not isinstance(txt, str):
        txt = ""
    t = txt.strip()

    # Replace placeholder category labels
    t = re.sub(r"\b(kategoria|category)\s+[A-Z]\b", "kategoria", t, flags=re.IGNORECASE)

    first_sentence = re.split(r"(?<=[\.!\?])\s+", t, maxsplit=1)[0] if t else ""
    if len(re.findall(r"\d+(?:[\.,]\d+)?", first_sentence)) >= 2:
        return t

    # Prefer share anchors when available
    cat = stats.get("top1_category") or stats.get("winner") or stats.get("leader_category") or stats.get("top_category")
    start = stats.get("top1_start_pct") or stats.get("start_pct") or stats.get("share_start_pct")
    end = stats.get("top1_end_pct") or stats.get("end_pct") or stats.get("share_end_pct")
    delta = stats.get("top1_delta_pp") or stats.get("delta_pp")
    if (start is not None) and (end is not None) and (delta is not None):
        label = f'"{cat}"' if cat else "lider"
        return f"{label}: {float(start):.2f}% → {float(end):.2f}% (Δ {float(delta):.2f} pp)."

    # Value anchors
    total = stats.get("total_value") or stats.get("sum_value") or stats.get("total_sales")
    peak = stats.get("peak_value") or stats.get("max_value")
    if (total is not None) and (peak is not None):
        return f"Suma: {float(total):,.0f}; szczyt: {float(peak):,.0f}."

    # Count anchors
    n_cat = stats.get("n_categories")
    n_txn = stats.get("total_txn") or stats.get("n_txn")
    if (n_cat is not None) and (n_txn is not None):
        return f"Liczba kategorii: {int(n_cat)}; liczba transakcji: {int(n_txn):,}."

    return t


def _et_trace_preview(obj: Any, limit: int = 400) -> str:
    try:
        s = repr(obj)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            s = "<unrepr>"
    s = " ".join(str(s).split())
    return s[:limit]


def _et_extract_text_like_overview(resp: Any) -> str:
    """
    Normalizacja odpowiedzi ET w duchu overview:
    - akceptuj string
    - akceptuj dict z content/text/output_text/answer/one_sentence
    - akceptuj OpenAI-like shapes: choices/message/content, output_text, output[...].content[...].text
    - akceptuj listy i sklejaj elementy tekstowe
    """
    if resp is None:
        return ""

    if isinstance(resp, str):
        return " ".join(resp.strip().split())

    if isinstance(resp, dict):
        # rescue path for ET returns {"takeaway": "..."} so keep it first
        for key in ["takeaway", "content", "text", "output_text", "answer", "final_text", "one_sentence"]:
            val = resp.get(key)
            if isinstance(val, str) and val.strip():
                return " ".join(val.strip().split())

        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            ch0 = choices[0] or {}
            if isinstance(ch0, dict):
                msg = ch0.get("message") or {}
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return " ".join(content.strip().split())
                txt = ch0.get("text")
                if isinstance(txt, str) and txt.strip():
                    return " ".join(txt.strip().split())

        output = resp.get("output")
        if isinstance(output, list):
            parts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            t = c.get("text") or c.get("content")
                            if isinstance(t, str) and t.strip():
                                parts.append(t.strip())
            if parts:
                return " ".join(" ".join(parts).split())

        # extra defensive fallback for ET rescue objects
        try:
            _tk = resp.get("takeaway")
            if isinstance(_tk, str) and _tk.strip():
                return " ".join(_tk.strip().split())
        except Exception:
            pass

        return ""

    if isinstance(resp, list):
        parts = []
        for item in resp:
            txt = _et_extract_text_like_overview(item)
            if txt:
                parts.append(txt)
        return " ".join(" ".join(parts).split())

    return ""

def _get_exec_block_vocabulary(block_id: str) -> str:
    if block_id == "cot__mix_share_topN":
        return (
            "Preferowane słowa: rdzeń portfela, obrona lidera, fragmentacja, erozja rdzenia, "
            "presja na obronę lidera, koszt oddawania udziału, erozja rdzenia portfela."
        )
    if block_id == "cot__winners_losers":
        return (
            "Preferowane słowa: transfer udziału, przejęcie popytu, realokacja wsparcia, "
            "okno przejęcia popytu, koszt utraty udziału."
        )
    if block_id == "cot__seasonality":
        return (
            "Preferowane słowa: kalendarz aktywacji, zapas, błąd planowania, szczyt sezonu, "
            "koszt błędu planowania, ryzyko niedostępności."
        )
    if block_id == "cot__start_end":
        return (
            "Preferowane słowa: trwałe przesunięcie struktury, nowy układ kategorii, "
            "trwałe przesunięcie popytu, ryzyko błędnej alokacji bazowej."
        )
    if block_id == "cot__concentration":
        return (
            "Preferowane słowa: zależność od rdzenia, challenger, odporność portfela, "
            "dywersyfikacja, ryzyko zależności od rdzenia, rola challengerów."
        )
    return (
        "Preferowane słowa: mechanizm biznesowy, priorytet operacyjny, decyzja portfelowa, "
        "ryzyko, koszt zaniechania, presja, trwałe przesunięcie."
    )


def _mbb_exec_takeaway_user_prompt(block: Dict[str, Any], stats: Any, question: str) -> str:
    bid = (block or {}).get("id") or (block or {}).get("label") or "unknown"
    title = (block or {}).get("title") or ""
    stats = _with_dimension_stat_aliases(stats)
    compact_stats = _compact_exec_stats(str(bid), stats)
    contract_profile = _build_exec_contract_profile(str(bid), compact_stats)

    vocab_hint = _get_exec_block_vocabulary(bid)
    _contract_mode = str(contract_profile.get("contract_mode") or "full")
    _numbers_rule = "minimum 2 liczby z JSON"
    if _contract_mode == "reduced":
        _numbers_rule = "1-2 liczby z JSON, preferencyjnie 2, bez zgadywania brakujacej kotwicy"
    elif _contract_mode in {"sparse", "insufficient"}:
        _numbers_rule = "1-2 liczby z JSON; jesli masz tylko 1 wiarygodna liczbe, nie dopisuj drugiej na sile"
    block_extra = ""
    stats_for_prompt = compact_stats
    if bid == "cot__seasonality":

        src_stats = compact_stats if isinstance(compact_stats, dict) and compact_stats else (
            stats if isinstance(stats, dict) else {}
        )

        stats_for_prompt = {
            "focus_dimension_value": src_stats.get("focus_dimension_value") or src_stats.get("seasonality_focus_dimension_value") or src_stats.get("seasonality_focus_category") or src_stats.get("focus_category"),
            "focus_category": src_stats.get("seasonality_focus_category") or src_stats.get("focus_category"),
            "focus_strength": src_stats.get("seasonality_focus_strength") or src_stats.get("focus_strength"),
            "focus_amplitude": src_stats.get("seasonality_focus_amplitude") or src_stats.get("focus_amplitude"),
            "second_dimension_value": src_stats.get("second_dimension_value") or src_stats.get("seasonality_second_dimension_value") or src_stats.get("seasonality_second_category") or src_stats.get("second_category"),
            "second_category": src_stats.get("seasonality_second_category") or src_stats.get("second_category"),
            "second_strength": src_stats.get("seasonality_second_strength") or src_stats.get("second_strength"),
            "second_amplitude": src_stats.get("seasonality_second_amplitude") or src_stats.get("second_amplitude"),
            "top_mode": src_stats.get("seasonality_top_mode") or src_stats.get("top_mode"),
            "share_gap": src_stats.get("seasonality_share_gap") or src_stats.get("share_gap"),
            "amplitude_gap": src_stats.get("seasonality_amplitude_gap") or src_stats.get("amplitude_gap"),
        }

        block_extra = (
            "Zdanie 1 MUSI porównać dokładnie dwa elementy wymiaru: focus i second. "
            "MUSI podać amplitude obu elementów oraz – jeśli dostępne – także strength lub weight. "
            "MUSI zawierać dosłownie konstrukcję porównawczą jednego z typów: "
            "'X ma wyższą amplitudę niż Y', 'X ma większą amplitudę niż Y', "
            "'X ... podczas gdy Y ...', albo 'X jest priorytetem zatowarowania przed szczytem sezonu w porównaniu do Y'. "
            "MUSI nazwać mechanizm biznesowy dosłownie przez co najmniej jedno z wyrażeń: "
            "'koszt błędu planowania', 'ryzyko niedoszacowania popytu', 'ryzyko niedoszacowania szczytu', "
            "'ryzyko niedostępności', 'priorytet zatowarowania przed szczytem sezonu'. "
            "MUSI wskazać, który element jest priorytetem zatowarowania przed szczytem sezonu. "
            "Zdanie 2 MUSI zaczynać się od 'Decyzja:' i zawierać działanie przede wszystkim na zapasie / zatowarowaniu / dostępności przed szczytem sezonu. "
            "Możesz dodać ekspozycję lub kalendarz aktywacji tylko jako element wtórny, ale główny kierunek ma dotyczyć zapasu i ryzyka błędu planowania. "
            "Why-now MUSI wynikać z kosztu błędu planowania, ryzyka niedostępności albo ryzyka niedoszacowania popytu. "
            "Zwróć wyłącznie 2 krótkie zdania jako czysty tekst. Nie zwracaj JSON. Nie zwracaj {}. Nie zostawiaj pustej odpowiedzi."
        )
    elif bid == "cot__mix_share_topN":

        src_stats = compact_stats if isinstance(compact_stats, dict) and compact_stats else (
            stats if isinstance(stats, dict) else {}
        )

        stats_for_prompt = _canonical_mix_share_stats(src_stats)

        block_extra = (
            "Zdanie 1 MUSI podać Top-N start → end oraz delta pp dla Top-N. "
            "Jeśli w JSON istnieje leader_delta_pp / top1_delta_pp / driver_delta_pp, MUSI podać także liczbową zmianę lidera w pp. "
            "Jeśli delta lidera nie istnieje w JSON, wolno napisać tylko: 'lider traci pozycję w miksie', ale bez placeholdera '—'. "
            "Zdanie 2 MUSI zaczynać się od 'Decyzja:' albo 'Dlatego:' i zawierać kierunek typu obrona lidera / rdzeń portfela / realokacja wsparcia / rola elementu w miksie. "
            "Fraza 'obrona lidera' jest poprawna także w odmianie typu 'przeprowadzić obronę lidera'. "
            "Why-now MUSI wynikać z presji na obronę lidera, erozji rdzenia portfela albo ryzyka utraty udziału. "
            "Zwróć wyłącznie 2 krótkie zdania jako czysty tekst. Nie zwracaj JSON. Nie zwracaj {}. Nie zostawiaj pustej odpowiedzi."
        )
    elif bid == "cot__winners_losers":

        src_stats = compact_stats if isinstance(compact_stats, dict) and compact_stats else (
            stats if isinstance(stats, dict) else {}
        )

        stats_for_prompt = {
            "winner_dimension_value": src_stats.get("winner_dimension_value") or src_stats.get("winner_dimension_value_non_other") or src_stats.get("winner_category") or src_stats.get("winner_category_non_other") or src_stats.get("biggest_gainer_category") or src_stats.get("driver_dimension_value") or src_stats.get("driver_category_name"),
            "winner_category": src_stats.get("winner_category") or src_stats.get("winner_category_non_other") or src_stats.get("biggest_gainer_category") or src_stats.get("driver_category_name"),
            "winner_delta_pp": src_stats.get("winner_delta_pp") or src_stats.get("winner_delta_pp_non_other") or src_stats.get("biggest_gainer_delta_pp") or src_stats.get("driver_delta_pp"),
            "loser_dimension_value": src_stats.get("loser_dimension_value") or src_stats.get("loser_dimension_value_non_other") or src_stats.get("loser_category") or src_stats.get("loser_category_non_other") or src_stats.get("biggest_loser_category"),
            "loser_category": src_stats.get("loser_category") or src_stats.get("loser_category_non_other") or src_stats.get("biggest_loser_category"),
            "loser_delta_pp": src_stats.get("loser_delta_pp") or src_stats.get("loser_delta_pp_non_other") or src_stats.get("biggest_loser_delta_pp"),
        }

        block_extra = (
            "Zdanie 1 MUSI podać winner delta i loser delta oraz nazwać mechanizm przez 'transfer udziału' albo 'przejęcie popytu'. "
            "Zdanie 2 MUSI zaczynać się od 'Decyzja:' i zawierać jednocześnie: "
            "(1) jeden lever dla winnera: availability / ekspozycja / wsparcie promocyjne, "
            "(2) jedną diagnozę albo lever dla loser: cena / sezonowość / obecność na półce / rola elementu. "
            "Why-now MUSI wynikać z okna przejęcia popytu albo kosztu utraty udziału. "
            "Zwróć wyłącznie 2 krótkie zdania jako czysty tekst. Nie zwracaj JSON. Nie zwracaj {}. Nie zostawiaj pustej odpowiedzi."
        )
    elif bid == "cot__start_end":

        compact_stats = {
            "winner_dimension_value": (
                stats.get("winner_dimension_value")
                or stats.get("winner_category")
                or stats.get("start_end_winner_dimension_value")
                or stats.get("start_end_winner_category")
            ),
            "winner_category": (
                stats.get("winner_category")
                or stats.get("start_end_winner_category")
            ),
            "winner_delta_pp": (
                stats.get("winner_delta_pp")
                or stats.get("start_end_winner_delta_pp")
            ),
            "loser_dimension_value": (
                stats.get("loser_dimension_value")
                or stats.get("loser_category")
                or stats.get("start_end_loser_dimension_value")
                or stats.get("start_end_loser_category")
            ),
            "loser_category": (
                stats.get("loser_category")
                or stats.get("start_end_loser_category")
            ),
            "loser_delta_pp": (
                stats.get("loser_delta_pp")
                or stats.get("start_end_loser_delta_pp")
            ),
        }

        block_extra = (
            "Zdanie 1 MUSI podać winner delta i loser delta w pp oraz wskazać, czy to trwałe przesunięcie struktury czy szum. "
            "Zdanie 2 MUSI zaczynać się od 'Decyzja:' i zawierać dosłownie jeden z kierunków: "
            "bazowa alokacja / plan wsparcia / ekspozycja bazowa / realokacja wsparcia / rola elementu w miksie. "
            "Zdanie 2 MUSI zawierać także horyzont działania: 'w ciągu 1–2 cykli' albo 'w ciągu 1–2 miesięcy'. "
            "Why-now MUSI wynikać z trwałego przesunięcia popytu albo ryzyka utraty udziału w nowym układzie udziałów. "
            "Zwróć wyłącznie 2 krótkie zdania jako czysty tekst. Nie zwracaj JSON. Nie zwracaj {}. Nie zostawiaj pustej odpowiedzi."
        )

        stats_for_prompt = compact_stats
    elif bid == "cot__concentration":

        compact_stats = {
            "hhi_start": stats.get("hhi_start"),
            "hhi_end": stats.get("hhi_end"),
            "top5_delta_pp": (
                stats.get("top5_delta_pp")
                or stats.get("topN_delta_pp")
            ),
            "topN": stats.get("topN"),
        }

        block_extra = (
            "Zdanie 1 MUSI zaczynać się od 'Portfel' albo 'Struktura portfela' "
            "i nie może zaczynać się od 'W kategorii'. "
            "MUSI zawierać zmianę HHI oraz – jeśli dostępne – zmianę Top-5 w pp. "
            "MUSI używać języka biznesowego: zależność od rdzenia / dywersyfikacja / odporność portfela / rola challengerów. "
            "Zdanie 2 MUSI zaczynać się od 'Decyzja:' i zawierać dosłownie jeden z kierunków: "
            "'realokacja wsparcia dla challengerów' / 'wzmocnienie challengerów' / "
            "'ograniczenie zależności od rdzenia' / 'poprawa odporności portfela'. "
            "Nie używaj ogólnika: 'realokacja wsparcia' bez adresata decyzji. "
            "Zwróć wyłącznie 2 krótkie zdania jako czysty tekst. Nie zwracaj JSON. Nie zwracaj {}. Nie zostawiaj pustej odpowiedzi. "
            "Nie używaj nazw drivera typu: 'kategoria', 'sprzedaż', 'konsolidacja', 'koncentracja' ani 'kategoria \"...\"'. "
            "To zdanie ma mówić o strukturze portfela, nie o nazwie kategorii. "
            "Preferowany początek: 'Portfel konsoliduje się...' albo 'Struktura portfela staje się bardziej zależna od rdzenia...'."
        )

        stats_for_prompt = compact_stats

    user_prompt = (
        f"Pytanie: {question}\n"
        f"Blok: {bid}\n"
        f"Tytuł: {title}\n"
        f"Contract profile: {_mbb_json_dumps_safe(contract_profile)}\n"
        f"JSON: {_mbb_json_dumps_safe(stats_for_prompt)}\n\n"
        "Napisz dokładnie 2 krótkie zdania po polsku.\n"
        "Zdanie 1 MUSI zawierać:\n"
        f"- {_numbers_rule},\n"
        "- mechanizm biznesowy (np. erozja rdzenia, transfer udziału, przejęcie popytu, trwała zmiana struktury, rosnąca zależność od rdzenia, sezonowy driver).\n"
        "- oryginalną etykietę z danych w cudzysłowie tylko wtedy, gdy blok dotyczy konkretnego elementu wymiaru; "
        "dla cot__concentration zdanie 1 ma mówić o strukturze portfela, nie o nazwie kategorii.\n"
        "Zdanie 2 MUSI:\n"
        "- zaczynać się od 'Decyzja:' albo 'Dlatego:',\n"
        "- zawierać konkretne działanie operacyjne lub portfelowe,\n"
        "- wskazywać kierunek: availability / ekspozycja / cena / asortyment / zapas / kalendarz aktywacji / alokacja wsparcia,\n"
        "- zawierać horyzont działania (np. 1–2 cykle / 1–2 miesiące) albo KPI kontrolny.\n"
        "- zawierać why-now DOSŁOWNIE przez co najmniej jedno z wyrażeń:\n"
        "  - ryzyko utraty udziału\n"
        "  - ryzyka utraty udziału\n"
        "  - koszt zaniechania\n"
        "  - presja na obronę lidera\n"
        "  - okno przejęcia popytu\n"
        "  - trwałe przesunięcie popytu\n"
        "  - koszt błędu planowania\n"
        "Unikaj sformułowań: 'zwiększyć marketing', 'poprawić pozycję', 'wesprzeć kategorię', 'monitorować trendy', 'warto rozważyć', 'należy przeanalizować'.\n"
        f"{vocab_hint}\n"
        f"{_exec_contract_prompt_guidance(str(bid), contract_profile)}\n"
        "Brzmienie: executive summary dla CMO / zarządu, nie opis BI ani komentarz do wykresu.\n"
        "Bez bulletów, bez nagłówków, bez placeholderów."
    )

    if block_extra:
        user_prompt = f"{user_prompt}\n{block_extra}"

    return user_prompt

def get_exec_takeaways_llm(ctx: Dict[str, Any], intent: str, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Executive Takeaway generator with:
    - raw response tracing
    - response extraction aligned with overview-like normalization
    - diagnostic single-call path for 1-2 critical blocks
    """
    try:
        if bool((ctx or {}).get("disable_mbb_llm")):
            return _core_get_exec_takeaways_llm(ctx=ctx, intent=intent, blocks=blocks) or {}
    except Exception:
        pass

    out: Dict[str, str] = {}
    runtime_meta: Dict[str, Dict[str, Any]] = {}
    question = str((ctx or {}).get("question") or (ctx or {}).get("user_question") or "").strip()
    escalation_mode = bool((ctx or {}).get("_exec_escalation_mode"))
    diagnostic_direct_blocks = set()

    def _build_exec_repair_prompt(_blk: Dict[str, Any], _txt: str, _soft_reasons: List[str]) -> str:
        _bid = str(_blk.get("id") or _blk.get("label") or "unknown")
        _stats = _compact_exec_stats(_bid, _blk.get("stats") or {})
        _contract_profile = _build_exec_contract_profile(_bid, _stats)
        _reason_txt = ", ".join([str(x) for x in (_soft_reasons or []) if str(x or "").strip()]) or "soft_gate"
        _fix_hint = (
            "Popraw executive takeaway tak, by byl bardziej kontraktowy i bardziej board-ready. "
            "Zachowaj liczby z oryginalu i nie zmieniaj faktow z JSON. "
            "Zwroc dokladnie 2 krotkie zdania jako czysty tekst."
        )
        if _bid == "cot__mix_share_topN":
            _fix_hint = (
                "Popraw executive takeaway. Zdanie 1 ma podac Top-N start/end i delta pp, a jesli leader delta nie istnieje, "
                "ma nazwac utrate pozycji lidera bez placeholdera. "
                "Zdanie 2 ma zaczynac sie od 'Decyzja:' i wskazac obrone lidera, rdzen portfela albo realokacje wsparcia. "
                "Fraza 'przeprowadzic obrone lidera' jest poprawna i traktowana jak kierunek decyzji. "
                "Nie zgaduj brakujacych liczb."
            )
        elif _bid == "cot__winners_losers":
            _fix_hint = (
                "Popraw executive takeaway. Zdanie 1 musi nazwac mechanizm przez 'transfer udzialu' albo 'przejecie popytu'. "
                "Zdanie 2 ma zaczynac sie od 'Decyzja:' i zawierac osobny lever dla winnera oraz osobna diagnoze albo lever dla loser. "
                "Winner lever moze byc nazwany przez dostepnosc / availability / ekspozycje / wsparcie promocyjne. "
                "Loser musi dostac diagnoze albo lever typu cena / sezonowosc / rola elementu / ograniczenie wsparcia. "
                "Why-now MUSI zostac zachowane doslownie przez 'okno przejecia popytu' albo 'koszt utraty udzialu'. "
                "Zachowaj liczby i oryginalne etykiety z danych."
            )
        elif _bid == "cot__start_end":
            _fix_hint = (
                "Popraw executive takeaway. Dodaj pp przy obu deltach winner/loser i why-now wynikajace z trwalego przesuniecia popytu albo ryzyka blednej alokacji bazowej. "
                "Zdanie 2 ma zaczynac sie od 'Decyzja:' i mowic o bazowej alokacji, planie wsparcia albo ekspozycji bazowej. "
                "Zdanie 2 ma zawierac tez horyzont 1-2 cykli albo 1-2 miesiecy. "
                "Zachowaj liczby i oryginalne etykiety z danych."
            )
        return (
            f"Blok: {_bid}\n"
            f"Soft reasons: {_reason_txt}\n"
            f"Contract profile: {_mbb_json_dumps_safe(_contract_profile)}\n"
            f"JSON: {_mbb_json_dumps_safe(_stats)}\n"
            f"Obecny tekst: {_txt}\n\n"
            f"{_fix_hint}\n"
            f"{_exec_contract_prompt_guidance(_bid, _contract_profile)}\n"
            "Nie zwracaj JSON. Nie zwracaj listy. Nie zwracaj markdown."
        )

    def _repair_exec_takeaway_single_direct(_blk: Dict[str, Any], _txt: str, _soft_reasons: List[str]) -> str:
        _label = str(_blk.get("label") or _blk.get("id") or "unknown")
        _model = str(((ctx or {}).get("openai_model") or "gpt-4o-mini"))
        _prompt = _build_exec_repair_prompt(_blk, _txt, _soft_reasons)
        _resp = _call_llm_with_trace(
            ctx=ctx,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Pisz po polsku, konkretnie i executive-style. "
                        "Zwracaj tylko gotowy tekst."
                    ),
                },
                {"role": "user", "content": _prompt},
            ],
            model=_model,
            temperature=0.0,
            max_tokens=180,
            payload={"kind": "cot_exec_takeaway_repair_direct", "block_id": _label},
            trace_where="llm.wrapper.exec_takeaway_repair_direct",
        )
        _txt_raw = _extract_text_from_llm_response(_resp)
        if (not _txt_raw) and isinstance(_resp, dict):
            _txt_raw = _et_extract_text_like_overview(_resp)
        _txt_norm = " ".join(str(_txt_raw or "").strip().split())
        dbg_cp(
            "exec_takeaway.repair_candidate",
            block_id=_label,
            soft_reasons=_soft_reasons,
            text_len=len(_txt_norm or ""),
            preview=(_txt_norm or "")[:220],
        )
        return _txt_norm

    def _normalize_exec_batch_result(_raw_map: Any) -> Dict[str, str]:
        _out: Dict[str, str] = {}
        if not isinstance(_raw_map, dict):
            return _out
        for _k, _v in _raw_map.items():
            if isinstance(_v, dict):
                _txt_val = (
                    _v.get("text")
                    or _v.get("takeaway")
                    or (( _v.get("meta") or {}).get("text") if isinstance(_v.get("meta"), dict) else "")
                    or ""
                )
            else:
                _txt_val = _v
            _txt_norm = _normalize_exec_takeaway_text(_txt_val)
            if _txt_norm:
                _out[str(_k)] = _txt_norm
        return _out

    def _normalize_exec_batch_source_map(_raw_map: Any, default_src: str = "llm_batch_direct") -> Dict[str, str]:
        _out: Dict[str, str] = {}
        if not isinstance(_raw_map, dict):
            return _out
        for _k, _v in _raw_map.items():
            _bid = str(_k)
            if isinstance(_v, dict):
                _src_val = str(
                    _v.get("src")
                    or (( _v.get("meta") or {}).get("src") if isinstance(_v.get("meta"), dict) else "")
                    or default_src
                )
            else:
                _src_val = default_src
            if _normalize_exec_takeaway_text(_v if not isinstance(_v, dict) else (_v.get("text") or _v.get("takeaway") or "")):
                _out[_bid] = _src_val
        return _out

    def _build_exec_batch_response_format(_blks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        _properties: Dict[str, Any] = {}
        _required: List[str] = []
        for _blk in (_blks or []):
            _bid = str(_blk.get("id") or _blk.get("label") or "").strip()
            if (not _bid) or (_bid in _properties):
                continue
            _properties[_bid] = {
                "type": "string",
                "description": f"Final executive takeaway for {_bid} in Polish.",
            }
            _required.append(_bid)
        if not _properties:
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "exec_takeaways_batch",
                "schema": {
                    "type": "object",
                    "properties": _properties,
                    "required": _required,
                    "additionalProperties": False,
                },
            },
        }

    def _build_exec_batch_prompt(_blks: List[Dict[str, Any]]) -> str:
        _parts: List[str] = []
        for _blk in (_blks or []):
            _bid = str(_blk.get("id") or _blk.get("label") or "unknown")
            _stats = _compact_exec_stats(_bid, (_blk or {}).get("stats") or {})
            if isinstance(_stats, dict):
                _stats["block_id"] = _bid
            _contract_profile = _build_exec_contract_profile(_bid, _stats)
            _parts.append(
                f"BLOCK_ID: {_bid}\n"
                f"Contract profile: {_mbb_json_dumps_safe(_contract_profile)}\n"
                f"{_mbb_exec_takeaway_user_prompt(block=_blk, stats=_stats, question=question)}"
            )
        return (
            "Zwroc WYLACZNIE poprawny JSON zgodny ze schema.\n"
            "Kazdy klucz ma byc dokladnie rowny BLOCK_ID.\n"
            "Kazda wartosc ma byc niepustym executive takeaway po polsku dla danego bloku.\n"
            "Zachowaj liczby, oryginalne etykiety z danych i kierunek decyzji. Nie dodawaj komentarza poza JSON.\n\n"
            + "\n\n---\n\n".join(_parts)
        )

    _batch_prefill: Dict[str, str] = {}
    _batch_prefill_src: Dict[str, str] = {}
    _batch_candidate_blocks = [
        _blk for _blk in (blocks or [])
        if str((_blk or {}).get("label") or (_blk or {}).get("id") or "") not in diagnostic_direct_blocks
    ]

    if (not escalation_mode) and len(_batch_candidate_blocks) >= 2:
        try:
            _batch_prompt = _build_exec_batch_prompt(_batch_candidate_blocks)
            _batch_schema = _build_exec_batch_response_format(_batch_candidate_blocks)
            if _batch_prompt and _batch_schema:
                dbg_cp(
                    "exec_takeaway.batch_first_request",
                    blocks_count=len(_batch_candidate_blocks),
                    block_ids=[
                        str((_blk or {}).get("id") or (_blk or {}).get("label") or "")
                        for _blk in _batch_candidate_blocks
                    ],
                )
                _batch_resp = _call_llm_with_trace(
                    ctx=ctx,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                _MBB_ET_SYSTEM
                                + "\n\nBATCH-FIRST MODE:\n"
                                + "Zwracasz jeden JSON mapujacy block_id -> finalny executive takeaway. "
                                + "Kazdy takeaway ma byc gotowym tekstem dla jednego bloku. "
                                + "Nie wolno zwrocic pustych wartosci ani dodatkowych kluczy."
                            ),
                        },
                        {"role": "user", "content": _batch_prompt},
                    ],
                    model=str((ctx.get("openai_model") or "gpt-4o-mini")),
                    temperature=0.0,
                    payload={"kind": "cot_exec_takeaway_batch_direct", "block_id": "__all__"},
                    response_format=_batch_schema,
                    trace_where="llm.wrapper.exec_takeaway_batch_direct",
                )
                _batch_prefill = _normalize_exec_batch_result(_batch_resp)
                _batch_prefill_src = _normalize_exec_batch_source_map(_batch_resp, default_src="llm_batch_direct")
                dbg_cp(
                    "exec_takeaway.batch_first_response",
                    blocks_count=len(_batch_candidate_blocks),
                    filled_blocks=sorted(list(_batch_prefill.keys())),
                    missing_blocks=[
                        str((_blk or {}).get("id") or (_blk or {}).get("label") or "")
                        for _blk in _batch_candidate_blocks
                        if not str(
                            _batch_prefill.get(
                                str((_blk or {}).get("id") or (_blk or {}).get("label") or "")
                            )
                            or ""
                        ).strip()
                    ],
                )
        except Exception as _batch_err:
            _errors_add(f"mbb_exec_takeaway batch_first error: {type(_batch_err).__name__}: {_batch_err}")
            dbg_cp(
                "exec_takeaway.batch_first_exception",
                blocks_count=len(_batch_candidate_blocks),
                error=f"{type(_batch_err).__name__}: {_batch_err}",
            )
            _batch_prefill = {}
            _batch_prefill_src = {}

    for b in (blocks or []):
        label = (b or {}).get("label") or (b or {}).get("id") or "unknown"
        stats = (b or {}).get("stats") or {}
        compact_stats = _compact_exec_stats(str(label), stats)
        if isinstance(compact_stats, dict):
            compact_stats["block_id"] = str(label)
        prompt = ""

        meta = {"block_id": label, "intent": intent, "src": "llm", "gate": None, "passed": False, "reasons": []}
        txt = ""

        try:
            _prefilled_txt = str(_batch_prefill.get(str(label)) or "").strip()
            if _prefilled_txt and str(label) not in diagnostic_direct_blocks:
                raw_resp = {"takeaway": _prefilled_txt}
                meta["src"] = str(_batch_prefill_src.get(str(label)) or "llm_batch_direct")
                txt = _prefilled_txt
                if bool(st.session_state.get("__cot_exec_dbg_on")):
                    dbg_cp(
                        "exec_takeaway.batch_candidate_loaded",
                        block_id=str(label),
                        src=meta.get("src"),
                        text_len=len(txt or ""),
                        preview=(txt or "")[:220],
                    )
            elif str(label) in diagnostic_direct_blocks:
                prompt = prompt or _mbb_exec_takeaway_user_prompt(block=b, stats=compact_stats, question=question)
                raw_resp = _call_llm_with_trace(
                    ctx=ctx,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                ("Pisz po polsku, liczbowo i bardzo rygorystycznie. "
                                "Zwróć wyłącznie 2 krótkie zdania jako czysty tekst. "
                                "Nie zwracaj JSON. Nie zwracaj {}. Nie zwracaj listy. "
                                "Nie zwracaj nagłówków ani markdown. "
                                "Podaj minimum 2 liczby i zacznij drugie zdanie od 'Decyzja:'. "
                                "Jeśli nie możesz spełnić formatu, zwróć 1 niepuste zdanie zamiast pustego obiektu.")
                                if escalation_mode
                                else
                                ("Pisz po polsku, liczbowo i zwięźle. "
                                "Zwróć wyłącznie 2 krótkie zdania jako czysty tekst. "
                                "Nie zwracaj JSON. Nie zwracaj {}. Nie zwracaj listy. "
                                "Nie zwracaj nagłówków ani markdown. "
                                "Jeśli nie możesz spełnić formatu, zwróć 1 niepuste zdanie zamiast pustego obiektu.")
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=str((ctx.get("openai_model") or "gpt-4o-mini")),
                    temperature=0.0,
                    max_tokens=(220 if escalation_mode else 140),
                    payload={
                        "kind": (
                            "cot_exec_takeaway_escalation"
                            if escalation_mode
                            else "cot_exec_takeaway_single_direct"
                        ),
                        "block_id": str(label),
                    },
                    trace_where=(
                        "llm.wrapper.exec_takeaway_escalation"
                        if escalation_mode
                        else "llm.wrapper.exec_takeaway_single_direct"
                    ),
                )
                meta["src"] = "llm_escalation" if escalation_mode else "llm_single_direct"
            else:
                prompt = prompt or _mbb_exec_takeaway_user_prompt(block=b, stats=compact_stats, question=question)
                raw_resp = _call_llm_with_trace(
                    ctx=ctx,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                _MBB_ET_SYSTEM
                                + "\n\nDODATKOWA REGUŁA FIRST-PASS:\n"
                                + "W tej ścieżce nie wolno zwrócić pustego dict, schema-like dict ani {}. "
                                + "Jeżeli model nie jest pewny formatu, ma zwrócić zwykły tekst w 2 zdaniach, "
                                + "a jeśli to niemożliwe — 1 niepuste zdanie po polsku zamiast pustego obiektu. "
                                + "Nie wolno zwrócić JSON, nawet częściowego."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=str((ctx.get("openai_model") or "gpt-4o-mini")),
                    temperature=0.0,
                    max_tokens=180,
                    payload={"kind": "cot_exec_takeaway_single_direct", "block_id": str(label)},
                    trace_where="llm.wrapper.exec_takeaway_single_direct",
                )
                meta["src"] = "llm_single_direct"

            if bool(st.session_state.get("__cot_exec_dbg_on")):
                dbg_cp(
                    "exec_takeaway.candidate_raw_response",
                    block_id=str(label),
                    response_type=type(raw_resp).__name__,
                    preview=_et_trace_preview(raw_resp, 500),
                )

                txt = _extract_text_from_llm_response(raw_resp)
                txt = " ".join(str(txt or "").strip().split())

                if (not txt) and isinstance(raw_resp, dict):
                    _tk = raw_resp.get("takeaway")
                    if isinstance(_tk, str) and _tk.strip():
                        txt = " ".join(_tk.strip().split())

                if (not txt) and isinstance(raw_resp, dict):
                    _props = raw_resp.get("properties")
                    if isinstance(_props, dict):
                        _tk = _props.get("takeaway")
                        if isinstance(_tk, str) and _tk.strip():
                            txt = " ".join(_tk.strip().split())

                if (not txt) and isinstance(raw_resp, dict):
                    _type = str(raw_resp.get("type") or "").strip().lower()
                    _props = raw_resp.get("properties")
                    if _type == "object" and isinstance(_props, dict):
                        for _k in ["takeaway", "text", "content"]:
                            _candidate = _props.get(_k)
                            if isinstance(_candidate, str) and _candidate.strip():
                                txt = " ".join(_candidate.strip().split())
                                break

            _mix_placeholder_detected = False
            if str(label) == "cot__mix_share_topN":
                _tl = str(txt or "").lower()
                _has_soft_generic_mix_decision = any(
                    _k in _tl
                    for _k in [
                        "wzmocnienie ekspozycji",
                        "wzmocnij ekspozycję",
                        "zwiększenie ekspozycji",
                        "zwiększ ekspozycję",
                        "należy zwiększyć ekspozycję",
                        "ekspozycja w asortymencie",
                        "zwiększyć ekspozycję w asortymencie",
                        "zwiększenie ekspozycji w asortymencie",
                        "wzmocnienie asortymentu",
                        "wzmocnij asortyment",
                    ]
                )
                _has_hard_mix_direction = any(
                    _k in _tl
                    for _k in [
                        "obrona lidera",
                        "realokacja wsparcia",
                        "realokuj wsparcie",
                        "przesuń wsparcie",
                        "plan wsparcia dla rdzenia",
                        "rola kategorii w miksie",
                        "rola elementu w miksie",
                        "sprawdź cenę, ekspozycję i rolę asortymentową",
                        "sprawdź cenę",
                        "sprawdź ekspozycję",
                        "rola asortymentowa",
                    ]
                )
                _has_mix_placeholder = any(
                    _k in str(txt or "")
                    for _k in [
                        " o —",
                        "— pp",
                        "—%",
                        "— %",
                        "osłabił się o —",
                    ]
                )
                _mix_placeholder_detected = bool(_has_mix_placeholder)

                if bool(st.session_state.get("__cot_exec_dbg_on")):
                    dbg_cp(
                        "exec_takeaway.candidate_parsed_response",
                        block_id=str(label),
                        parsed_len=len(txt or ""),
                        preview=(txt or "")[:220],
                    )

            txt = _postprocess_exec_takeaway_by_block(txt, str(label), compact_stats)
            ok, reasons = _mbb_validate_exec_takeaway(txt, compact_stats)
            block_audit = _validate_exec_takeaway_by_block_detail(txt, str(label), compact_stats)
            block_ok = bool(block_audit.get("strict_ok"))
            block_reasons = list(block_audit.get("reasons") or [])
            _force_by_block_validator_ids = {
                "cot__mix_share_topN",
            }
            _repairable_block_ids = {
                "cot__mix_share_topN",
                "cot__winners_losers",
                "cot__start_end",
            }
            _block_hard_reason_set = {str(r or "") for r in list(block_audit.get("hard_reasons") or [])}
            _mix_repair_first = bool(
                str(label) == "cot__mix_share_topN"
                and txt
                and (
                    _mix_placeholder_detected
                    or (bool(_block_hard_reason_set) and "insufficient_stats_contract" not in _block_hard_reason_set)
                )
            )
            if (
                str(label) in _repairable_block_ids
                and txt
                and (
                    (
                        (not block_audit.get("strict_ok"))
                        and bool(block_audit.get("soft_reasons"))
                        and not bool(block_audit.get("hard_reasons"))
                    )
                    or _mix_repair_first
                )
            ):
                _orig_txt = str(txt or "")
                _orig_ok = bool(ok)
                _orig_reasons = list(reasons or [])
                _orig_block_audit = dict(block_audit or {})
                _orig_block_ok = bool(block_ok)
                _orig_block_reasons = list(block_reasons or [])
                _orig_score_obj = _quick_narrative_score(
                    _orig_txt,
                    block_id=str(label),
                    mode="exec_takeaway",
                    stats=compact_stats if isinstance(compact_stats, dict) else {},
                )
                _orig_score = int((_orig_score_obj or {}).get("score") or 0)
                _orig_soft_count = len(_orig_block_audit.get("soft_reasons") or [])
                _repaired_txt = _repair_exec_takeaway_single_direct(
                    _blk=b,
                    _txt=txt,
                    _soft_reasons=list(block_audit.get("soft_reasons") or []) + list(block_audit.get("hard_reasons") or []),
                )
                if _repaired_txt:
                    _repaired_txt_post = _postprocess_exec_takeaway_by_block(_repaired_txt, str(label), compact_stats)
                    _repair_ok, _repair_reasons = _mbb_validate_exec_takeaway(_repaired_txt_post, compact_stats)
                    _repair_block_audit = _validate_exec_takeaway_by_block_detail(_repaired_txt_post, str(label), compact_stats)
                    _repair_block_ok = bool(_repair_block_audit.get("strict_ok"))
                    _repair_block_reasons = list(_repair_block_audit.get("reasons") or [])
                    _repair_score_obj = _quick_narrative_score(
                        _repaired_txt_post,
                        block_id=str(label),
                        mode="exec_takeaway",
                        stats=compact_stats if isinstance(compact_stats, dict) else {},
                    )
                    _repair_score = int((_repair_score_obj or {}).get("score") or 0)
                    _repair_soft_count = len(_repair_block_audit.get("soft_reasons") or [])
                    _accept_repair = False
                    if not list(_repair_block_audit.get("hard_reasons") or []):
                        if _repair_block_ok and (not _orig_block_ok) and _repair_score >= _orig_score:
                            _accept_repair = True
                        elif _repair_score > _orig_score and _repair_soft_count <= _orig_soft_count:
                            _accept_repair = True
                        elif _repair_score == _orig_score and _repair_soft_count < _orig_soft_count:
                            _accept_repair = True

                    dbg_cp(
                        "exec_takeaway.repair_decision",
                        block_id=str(label),
                        accepted=_accept_repair,
                        original_score=_orig_score,
                        repair_score=_repair_score,
                        original_soft_reasons=_orig_block_audit.get("soft_reasons"),
                        repair_soft_reasons=_repair_block_audit.get("soft_reasons"),
                        original_preview=_orig_txt[:160],
                        repair_preview=_repaired_txt_post[:160],
                    )

                    if _accept_repair:
                        txt = _repaired_txt_post
                        meta["src"] = "llm_repair_direct"
                        ok, reasons = _repair_ok, _repair_reasons
                        block_audit = _repair_block_audit
                        block_ok = _repair_block_ok
                        block_reasons = _repair_block_reasons
                        if str(label) == "cot__mix_share_topN":
                            _mix_placeholder_detected = bool(_has_exec_numeric_placeholder(txt))
                    else:
                        txt = _orig_txt
                        ok, reasons = _orig_ok, _orig_reasons
                        block_audit = _orig_block_audit
                        block_ok = _orig_block_ok
                        block_reasons = _orig_block_reasons
            _force_gate_reasons = list(
                dict.fromkeys(
                    list(block_audit.get("hard_reasons") or [])
                    + (["placeholder_numeric"] if _mix_placeholder_detected else [])
                )
            )
            _mix_force_severe_reasons = {
                "insufficient_stats_contract",
                "mix_missing_stats_contract",
                "mix_missing_numeric_anchor",
            }
            _force_by_block_validator = False
            if str(label) == "cot__mix_share_topN":
                _force_hard_reason_set = {str(r or "") for r in list(block_audit.get("hard_reasons") or [])}
                _force_by_block_validator = bool(_force_hard_reason_set.intersection(_mix_force_severe_reasons))
            if (
                str(label) in _force_by_block_validator_ids
                and txt
                and _force_by_block_validator
            ):
                _forced_txt = str(_force_exec_takeaway(str(label), compact_stats) or "").strip()
                if _forced_txt:
                    dbg_cp(
                        "exec_takeaway.block_force_fallback",
                        block_id=str(label),
                        previous_preview=str(txt or "")[:220],
                        gate_reasons=_force_gate_reasons,
                        forced_preview=_forced_txt[:220],
                    )
                    txt = _forced_txt
                    meta["src"] = "deterministic_forced_by_block_validator"
                    meta["forced_by_block_validator"] = True
                    meta["forced_gate_reasons"] = list(_force_gate_reasons or [])
                    meta["selected_by"] = "block_validator_forced_fallback"
                    meta["selection_reason"] = "block_validator_forced_fallback"
                    ok, reasons = _mbb_validate_exec_takeaway(txt, compact_stats)
                    block_audit = _validate_exec_takeaway_by_block_detail(txt, str(label), compact_stats)
                    block_ok = bool(block_audit.get("strict_ok"))
                    block_reasons = list(block_audit.get("reasons") or [])
            if bool(block_audit.get("hard_reasons")):
                ok = False
                reasons = list(reasons or []) + list(block_audit.get("hard_reasons") or [])
            elif bool(block_audit.get("soft_reasons")):
                reasons = list(reasons or []) + list(block_audit.get("soft_reasons") or [])
            meta.update({
                "passed": ok,
                "gate": ok and not bool(block_audit.get("hard_reasons")),
                "reasons": reasons,
                "confidence": _cot_confidence_bucket(_cot_extract_r2(compact_stats)),
            })

            eval_obj = _quick_narrative_score(txt, block_id=str(label), mode=str(intent or ""), stats=compact_stats)
            matched = _detect_exec_why_now_tokens(txt)
            if bool(st.session_state.get("__cot_exec_dbg_on")):
                dbg_cp(
                    "exec_takeaway.quality_audit",
                    block_id=label,
                    candidate_score=eval_obj.get("score") if isinstance(eval_obj, dict) else None,
                    candidate_reasons=eval_obj.get("reasons") if isinstance(eval_obj, dict) else None,
                    candidate_penalties=eval_obj.get("penalties") if isinstance(eval_obj, dict) else None,
                    block_hard_reasons=block_audit.get("hard_reasons"),
                    block_soft_reasons=block_audit.get("soft_reasons"),
                    why_now_present=bool(matched),
                    why_now_tokens_matched=matched,
                    risk_or_opportunity_present=any(w in str(txt or "").lower() for w in [
                        "ryzyko",
                        "szansa",
                        "okno przejęcia",
                        "utrata",
                        "przejęcie",
                        "presja",
                        "ryzyko utraty udziału",
                        "ryzyka utraty udziału",
                        "z powodu ryzyka utraty udziału",
                        "aby zminimalizować ryzyko utraty udziału",
                        "aby uniknąć ryzyka utraty udziału",
                        "koszt zaniechania",
                        "koszt utraty udziału",
                        "koszt oddawania udziału",
                        "presja na obronę lidera",
                        "okno przejęcia popytu",
                        "trwałe przesunięcie popytu",
                        "trwałe przesunięcie struktury",
                        "koszt błędu planowania",
                        "ryzyko niedostępności",
                        "ryzyko niedoszacowania popytu",
                        "niedostępności",
                        "niedoszacowania popytu"
                    ]),
                    mechanism_present=_exec_has_named_mechanism_signal(str(txt or ""), block_id=str(label)),
                    decision_present=_exec_has_decision_signal(str(txt or "")),
                    horizon_present=any(w in str(txt or "").lower() for w in [
                        "1–2",
                        "1-2",
                        "najbliższych 1–2",
                        "najbliższych 1-2",
                        "miesiąc",
                        "miesiące",
                        "cykl",
                        "cykle",
                        "przed szczytem sezonu",
                        "na szczyt sezonu",
                        "w ciągu 1–2 miesięcy",
                        "w ciągu 1-2 miesięcy",
                        "w ciągu 1–2 cykli",
                        "w ciągu 1-2 cykli"
                    ]),
                )

            if bool(st.session_state.get("__cot_exec_dbg_on")):
                dbg_cp(
                    "exec_takeaway.quality_audit_aligned",
                    block_id=label,
                    candidate_score=eval_obj.get("score") if isinstance(eval_obj, dict) else None,
                    candidate_reasons=eval_obj.get("reasons") if isinstance(eval_obj, dict) else None,
                    candidate_penalties=eval_obj.get("penalties") if isinstance(eval_obj, dict) else None,
                    mechanism_present=_exec_has_named_mechanism_signal(str(txt or ""), block_id=str(label)),
                    decision_present=_exec_has_decision_signal(str(txt or "")),
                    why_now_present=bool(matched),
                    why_now_tokens_matched=matched,
                )

            candidate_score = eval_obj.get("score") if isinstance(eval_obj, dict) else None

            _txt_l = str(txt or "").lower()
            _risk_or_opportunity_present = any(w in _txt_l for w in [
                "ryzyko", "okno przejęcia", "utrata", "przejęcie", "presja",
                "ryzyko utraty udziału", "koszt zaniechania", "presja na obronę lidera",
                "okno przejęcia popytu", "trwałe przesunięcie popytu", "koszt błędu planowania",
                "ryzyko niedostępności", "ryzyko niedoszacowania", "niedostępności",
                "koszt oddawania udziału"
            ])
            _why_now_present = bool(matched)
            _contract_profile_direct = block_audit.get("contract_profile") if isinstance(block_audit, dict) else {}
            _contract_mode_direct = str((_contract_profile_direct or {}).get("contract_mode") or "full")
            _block_hard_fail = bool(block_audit.get("hard_reasons"))
            _block_soft_only = (not _block_hard_fail) and bool(block_audit.get("soft_reasons"))
            _allowed_reduced_generic_reasons = {
                "numbers_total<2",
                "sentence_count_not_2_to_3",
            }
            _generic_ok_for_contract = bool(ok)
            if (not _generic_ok_for_contract) and _contract_mode_direct in {"reduced", "sparse"}:
                _generic_ok_for_contract = all(
                    str(r or "") in _allowed_reduced_generic_reasons
                    for r in list(reasons or [])
                )
            _mechanism_present = any(w in _txt_l for w in [
                "erozj",
                "transfer udziału",
                "trwałe przesunięcie",
                "przejęcie popytu",
                "zależność od rdzenia",
                "obrona lidera",
                "błąd planowania",
                "koszt błędu planowania",
                "oddawanie udziału",
                "osłabienie pozycji",
                "odporność portfela",
                "dywersyfik",
            ])
            _decision_present = any(w in _txt_l for w in [
                "decyzja:",
                "dlatego:",
                "rekomendacja:",
                "realokacja",
                "przesuń wsparcie",
                "plan wsparcia",
                "alokacja",
                "ekspozyc",
                "availability",
                "zapas",
                "zatowarowanie",
                "sprawdź cenę",
                "sprawdź ekspozycję",
                "wzmocnij",
                "utrzymaj",
                "ogranicz",
            ])
            _mechanism_present = _exec_has_named_mechanism_signal(str(txt or ""), block_id=str(label))
            _decision_present = _exec_has_decision_signal(str(txt or ""))
            _horizon_present = any(w in _txt_l for w in [
                "1-2",
                "1–2",
                "miesiąc",
                "miesiące",
                "cykl",
                "cykle",
                "przed szczytem sezonu",
                "na szczyt sezonu",
            ])
            _caution_present = any(w in _txt_l for w in [
                "sygnał jest ograniczony",
                "sygnal jest ograniczony",
                "ostrożnie",
                "ostroznie",
                "bez eskalacji budżetu",
                "bez eskalacji budzetu",
                "dopiero po potwierdzeniu",
                "najpierw sprawdź",
                "najpierw sprawdz",
                "na próbę",
                "na probe",
                "pilota",
            ])
            _directional_gate_failures = {
                "mix_soft_generic_decision",
                "mix_missing_decision_direction",
                "wl_missing_decision_direction",
                "start_end_missing_decision_direction",
                "concentration_missing_decision_direction",
                "seasonality_missing_decision_direction",
            }
            _directional_gate_ok = not any(
                str(r or "") in _directional_gate_failures
                for r in list(reasons or []) + list(block_reasons or [])
            )
            _min_candidate_score = 4 if _contract_mode_direct in {"reduced", "sparse"} else 5

            _direct_ok = bool(
                str(txt or "").strip()
                and _generic_ok_for_contract
                and (bool(block_audit.get("hard_ok")) if isinstance(block_audit, dict) else block_ok)
                and not _block_hard_fail
                and isinstance(candidate_score, (int, float))
                and float(candidate_score) >= _min_candidate_score
                and _directional_gate_ok
                and not str(meta.get("src") or "").startswith("deterministic_")
                and (
                    (
                        _contract_mode_direct == "full"
                        and _risk_or_opportunity_present
                        and _why_now_present
                    )
                    or (
                        _contract_mode_direct in {"reduced", "sparse"}
                        and (_block_soft_only or _generic_ok_for_contract)
                        and _decision_present
                        and (_mechanism_present or _risk_or_opportunity_present or _caution_present)
                        and (_why_now_present or _horizon_present or _caution_present)
                    )
                )
            )

            if _direct_ok:
                meta["src"] = meta.get("src") or "llm_single_direct"
                meta["passed"] = True
                meta["gate"] = True
                meta["reasons"] = []
                out[str(label)] = str(txt or "")
                runtime_meta[str(label)] = dict(meta)

                dbg_cp(
                    "exec_takeaway.direct_pass_selected",
                    block_id=str(label),
                    src=meta.get("src"),
                    candidate_score=candidate_score,
                    text_len=len(str(txt or "")),
                    preview=str(txt or "")[:220],
                )
                continue

            out[str(label)] = txt

            dbg_cp(
                "exec_takeaway.candidate_ready_for_selection",
                block_id=str(label),
                src=(meta or {}).get("src"),
                gate_reason=((meta or {}).get("gate_reason") or ((meta or {}).get("reasons") or [None])[0]),
                text_len=len(txt or ""),
                preview=(txt or "")[:160],
            )

        except Exception as e:
            _errors_add(f"mbb_exec_takeaway llm error ({label}): {type(e).__name__}: {e}")
            try:
                core = _core_get_exec_takeaways_llm(ctx=ctx, intent=intent, blocks=[b]) or {}
                out.update(core)
                meta.update({"src": "core_fallback", "passed": False, "gate": False, "reasons": [f"llm_error:{type(e).__name__}"]})
            except Exception as ee:
                _errors_add(f"mbb_exec_takeaway core fallback error ({label}): {type(ee).__name__}: {ee}")
                out[str(label)] = ""
                meta.update({"src": "error", "passed": False, "gate": False, "reasons": [str(ee)]})

        label_s = str(label)
        try:
            _status_set(f"exec_mbb:{label_s}", meta, allow_empty=True)
            _cache_exec_takeaway(label_s, meta, out.get(label_s, ""), allow_empty=True)
        except Exception:
            pass
        runtime_meta[label_s] = dict(meta or {})
        runtime_meta[label_s]["text_len"] = len(str(out.get(label_s, "") or ""))

    final_results = runtime_meta or {}
    rich_out: Dict[str, Any] = {}
    for label_s, txt in (out or {}).items():
        meta_obj = final_results.get(str(label_s)) if isinstance(final_results, dict) else {}
        meta_obj = meta_obj if isinstance(meta_obj, dict) else {}
        rich_out[str(label_s)] = {
            "text": str(txt or ""),
            "src": str(meta_obj.get("src") or "llm_batch_raw"),
            "meta": dict(meta_obj),
        }

    return rich_out

CHART_BLOCK_HEIGHT = 360


# -----------------------------
# Column inference (robust)
# -----------------------------

_VALUE_HINTS = [
    "value", "sales", "sprzeda", "revenue", "amount", "totalprice", "total_price",
    "przych", "obrót", "turnover", "net", "gross",
]
_QTY_HINTS = ["qty", "quantity", "ilo", "szt", "units", "wolumen"]
_PRICE_HINTS = ["price", "unitprice", "unit_price", "cena"]
_TXN_HINTS = ["invoice", "order", "transaction", "txn", "paragon", "receipt", "id"]

# Category label used for the explicit "Other" bucket in many retail datasets.
# We keep it robust for arbitrary user uploads by detecting common variants.
_OTHER_LABEL_CANDIDATES = [
    "Other",
    "Others",
    "Inne",
    "Pozostałe",
    "Pozostale",
]

OTHER_LABEL = "Other"


def _infer_other_label(df_or_series, cat_col: str | None = None) -> str:
    """Infer the label used for the explicit 'Other' category.

    Supports two calling conventions:
    - _infer_other_label(series)
    - _infer_other_label(df, cat_col)

    This keeps backwards compatibility and allows branch code to work across
    arbitrary datasets where the explicit 'Other' category may be named
    differently (e.g., 'Other', 'Others', 'Pozostałe').
    """
    # Resolve to a Series of category values
    if cat_col is None:
        series = df_or_series
    else:
        try:
            series = df_or_series[cat_col]
        except Exception:
            series = df_or_series

    # Unique, non-null string values
    try:
        vals = list(pd.Series(series).dropna().astype(str).unique())
    except Exception:
        vals = [str(v) for v in series if v is not None]

    # exact match (prefer canonical capitalization present in data)
    for cand in _OTHER_LABEL_CANDIDATES:
        if cand in vals:
            return cand
    # last resort: case-insensitive match
    low_map = {str(v).strip().lower(): v for v in vals}
    for cand in _OTHER_LABEL_CANDIDATES:
        key = cand.lower()
        if key in low_map:
            return str(low_map[key])
    return "Other"

def _first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    low_map = {c.lower(): c for c in cols}
    for cand in candidates:
        for k, orig in low_map.items():
            if cand in k:
                return orig
    return None


def _infer_time_col(df: pd.DataFrame, schema_ctx: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    FAST + robust datetime column inference.

    Previous implementation attempted `pd.to_datetime` over full columns, which can take 10s–100s on large DF.
    This version:
    - Prefers schema_ctx["date_cols"] if present.
    - Restricts candidates by name hints first.
    - Parses only a sample (not full DF) to score candidates.
    """
    import pandas as pd
    from datetime import datetime

    if df is None or df.empty:
        return None

    schema_ctx = schema_ctx or {}
    cols = list(df.columns)

    # 0) If schema_ctx already provides date cols, trust it (fast path).
    date_cols = schema_ctx.get("date_cols") or []
    if isinstance(date_cols, (list, tuple)) and date_cols:
        for c in date_cols:
            if isinstance(c, str) and c in cols:
                return c

    now_year = datetime.now().year

    # 1) If any column is already datetime dtype, pick the first sensible one
    for c in cols:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        except Exception:
            continue

    # 2) Candidate selection by name (avoid trying to parse everything)
    preferred = []
    fallback = []
    for c in cols:
        cl = str(c).lower()
        if any(k in cl for k in ["date", "time", "invoicedate", "invoice date", "data", "czas"]):
            preferred.append(c)
        else:
            fallback.append(c)

    # keep candidate list small (speed): preferred first, then a few fallbacks
    candidates = preferred + fallback[:8]

    # 3) Score candidates on a SAMPLE, not on full DF
    # Use deterministic head for speed (no shuffle), but cap to avoid huge cost.
    sample_n = int(min(20000, max(1000, len(df))))
    best = None
    best_score = -1.0

    for c in candidates:
        try:
            s = df[c]
        except Exception:
            continue

        # Skip obvious numerics unless name suggests date/time
        try:
            if pd.api.types.is_numeric_dtype(s) and (c not in preferred):
                continue
        except Exception:
            pass

        try:
            s_sample = s.head(sample_n)
            parsed = pd.to_datetime(s_sample, errors="coerce", infer_datetime_format=True, utc=False)
        except Exception:
            continue

        ok = float(parsed.notna().mean())
        if ok < 0.85:
            continue

        years = parsed.dropna().dt.year
        if years.empty:
            continue

        y_med = float(years.median())
        y_min = int(years.min())
        y_max = int(years.max())

        # Reject "future-ID parsed as date" (e.g. 2049–2058)
        if y_med > (now_year + 2):
            continue

        span_years = (y_max - y_min)
        name_bonus = 0.25 if c in preferred else 0.0

        score = ok + name_bonus - (0.01 * span_years)
        if score > best_score:
            best_score = score
            best = c

    return best
def _infer_cat_col(df: pd.DataFrame, schema_ctx: Dict[str, Any]) -> Optional[str]:
    cols = list(df.columns)
    c = schema_ctx.get("group_col")
    if isinstance(c, str) and c in cols:
        return c
    cat_cols = schema_ctx.get("cat_cols") or []
    for c in cat_cols:
        if isinstance(c, str) and c in cols:
            return c
    # fallback: low-cardinality non-numeric
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        nunique = s.nunique(dropna=True)
        if 2 <= nunique <= 80:
            return c
    return None


def _dimension_display_label(cat_col: Any, fallback: str = "Wymiar") -> str:
    label = re.sub(r"\s+", " ", str(cat_col or "")).strip()
    return label or fallback


def _dimension_entity_label(cat_col: Any, fallback: str = "pozycja") -> str:
    raw = _dimension_display_label(cat_col, fallback="")
    if not raw:
        return fallback

    key = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii").lower()
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")

    aliases = {
        "category": "kategoria",
        "categories": "kategoria",
        "country": "kraj",
        "country_name": "kraj",
        "channel": "kanał",
        "sales_channel": "kanał",
        "brand": "marka",
        "segment": "segment",
        "region": "region",
        "market": "rynek",
        "store": "sklep",
        "customer": "klient",
    }
    if key in aliases:
        return aliases[key]
    if "country" in key:
        return "kraj"
    if "channel" in key:
        return "kanał"
    if "brand" in key:
        return "marka"
    if "segment" in key:
        return "segment"
    if "region" in key:
        return "region"
    if "market" in key:
        return "rynek"
    if "store" in key:
        return "sklep"
    if "customer" in key or "client" in key:
        return "klient"
    if "category" in key or "group" in key:
        return "kategoria"
    return fallback


def _dimension_axis_title(cat_col: Any, fallback: str = "Category") -> str:
    return _dimension_display_label(cat_col, fallback=fallback)


def _overview_chart_title(mode: str, cat_col: Any) -> str:
    dim = _dimension_display_label(cat_col)
    if mode == "share":
        return f'Udział według „{dim}” w czasie'
    return f'Sprzedaż według „{dim}” w czasie'


def _overview_chart_desc(mode: str, cat_col: Any) -> str:
    dim = _dimension_display_label(cat_col)
    if mode == "share":
        return (
            f'Stacked area 100% pokazujący zmianę udziałów procentowych '
            f'kolejnych wartości wymiaru „{dim}” w kolejnych miesiącach.'
        )
    return (
        f'Stacked area wartości sprzedaży dla kolejnych wartości wymiaru „{dim}” '
        f'z linią łącznej sumy po wierzchu.'
    )

def _infer_value_col(df: pd.DataFrame, schema_ctx: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (value_col, qty_col, price_col).
    If value_col missing but qty+price exist, value will be computed.
    """
    cols = list(df.columns)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    value_col = _first_existing(num_cols, _VALUE_HINTS)
    qty_col = _first_existing(num_cols, _QTY_HINTS)
    price_col = _first_existing(num_cols, _PRICE_HINTS)

    # If no explicit value, but qty+price exist → compute later
    if value_col is None and (qty_col and price_col):
        return None, qty_col, price_col

    # fallback: pick a numeric col that is not an obvious id/flag
    if value_col is None and num_cols:
        candidates = []
        for c in num_cols:
            low = c.lower()
            if any(h in low for h in ["id", "index", "flag", "code"]):
                continue
            candidates.append(c)
        value_col = candidates[0] if candidates else num_cols[0]

    return value_col, qty_col, price_col

def _infer_txn_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return _first_existing(cols, _TXN_HINTS)

# -----------------------------
# Shared helpers
# -----------------------------

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _df_fingerprint(df: pd.DataFrame) -> Tuple[int, Tuple[str, ...], Tuple[str, ...]]:
    """Lightweight DF fingerprint for per-dataset session_state memoization (NOT a cache of results)."""
    try:
        cols = tuple(str(c) for c in df.columns)
        dts = tuple(str(t) for t in df.dtypes)
        return (int(len(df)), cols, dts)
    except Exception:
        return (int(len(df)) if df is not None else 0, (str(id(df)),), ())



def _is_valid_time_col(df: pd.DataFrame, col: str, *, sample_n: int = 20000) -> bool:
    """
    Fast sanity-check for a datetime column (prevents cases like Invoice IDs parsed into years 2049+).
    - Works on a sample (head) for performance.
    - Accepts true datetime dtypes quickly.
    """
    if not col or col not in df.columns:
        return False
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        # still guard against absurd future dates
        ss = s.dropna()
        if ss.empty:
            return False
        years = ss.dt.year
        if years.empty:
            return False
        now_y = datetime.datetime.now().year
        return float(years.median()) <= (now_y + 2)

    # sample parse
    try:
        ss = s.head(sample_n)
        dt = pd.to_datetime(ss, errors="coerce", infer_datetime_format=True)
    except Exception:
        return False

    ok = float(dt.notna().mean()) if len(dt) else 0.0
    if ok < 0.85:
        return False

    years = dt.dropna().dt.year
    if years.empty:
        return False
    now_y = datetime.datetime.now().year
    y_med = float(years.median())
    if y_med > (now_y + 2):
        return False
    if y_med < 1900:
        return False
    return True


def _to_month_start(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    # normalize to month start (for deterministic axis)
    return dt.dt.to_period("M").dt.to_timestamp()

def _mode_from_question(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ["udział", "share", "procent", "%", "mix"]):
        return "share"
    return "value"
# --- period parsing (from question) ---
_MONTHS_PL = {
    "styczeń": 1, "stycznia": 1, "jan": 1, "january": 1,
    "luty": 2, "lutego": 2, "feb": 2, "february": 2,
    "marzec": 3, "marca": 3, "mar": 3, "march": 3,
    "kwiecień": 4, "kwietnia": 4, "apr": 4, "april": 4,
    "maj": 5, "maja": 5, "may": 5,
    "czerwiec": 6, "czerwca": 6, "jun": 6, "june": 6,
    "lipiec": 7, "lipca": 7, "jul": 7, "july": 7,
    "sierpień": 8, "sierpnia": 8, "aug": 8, "august": 8,
    "wrzesień": 9, "września": 9, "sep": 9, "september": 9,
    "październik": 10, "października": 10, "oct": 10, "october": 10,
    "listopad": 11, "listopada": 11, "nov": 11, "november": 11,
    "grudzień": 12, "grudnia": 12, "dec": 12, "december": 12,
}


def _norm_pl_token(s: str) -> str:
    """Normalize Polish tokens for fuzzy month matching (diacritics + ł)."""
    if not s:
        return ""
    s = s.strip().lower()
    # 'ł' does not decompose under NFKD/NFD on some builds, normalize manually
    s = s.replace("ł", "l").replace("Ł", "l")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

# Normalized month map to accept inputs without Polish diacritics and common typos.
_MONTHS_PL_NORM = {_norm_pl_token(k): v for k, v in _MONTHS_PL.items()}

def _parse_month_range_from_question(question: str) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Parse month range like:
      - "styczeń-czerwiec 2010"
      - "styczeń – grudzień 2010"
      - "jan 2010" (single month)
    Returns (start_month, end_month) as month-start timestamps (inclusive).
    """
    q = (question or "").lower()
    # normalize dashes
    q = q.replace("–", "-").replace("—", "-")

    # range: "od <m1> do <m2> <year>" (PL)
    m = re.search(
        r"\bod\s+(?P<m1>[a-ząćęłńóśźż]{3,})\s+do\s+(?P<m2>[a-ząćęłńóśźż]{3,})\s*(?P<y>\d{4})\b",
        q,
    )
    if m:
        m1 = _MONTHS_PL_NORM.get(_norm_pl_token(m.group("m1"))) or _MONTHS_PL.get(m.group("m1"))
        m2 = _MONTHS_PL_NORM.get(_norm_pl_token(m.group("m2"))) or _MONTHS_PL.get(m.group("m2"))
        y = int(m.group("y"))
        if m1 and m2:
            s = pd.Timestamp(year=y, month=m1, day=1)
            e = pd.Timestamp(year=y, month=m2, day=1)
            if e < s:
                s, e = e, s
            return s, e


    # range: <m1> - <m2> <year>
    m = re.search(r"(?P<m1>[a-ząćęłńóśźż]{3,})\s*-\s*(?P<m2>[a-ząćęłńóśźż]{3,})\s*(?P<y>\d{4})", q)
    if m:
        m1 = _MONTHS_PL_NORM.get(_norm_pl_token(m.group("m1"))) or _MONTHS_PL.get(m.group("m1"))
        m2 = _MONTHS_PL_NORM.get(_norm_pl_token(m.group("m2"))) or _MONTHS_PL.get(m.group("m2"))
        y = int(m.group("y"))
        if m1 and m2:
            s = pd.Timestamp(year=y, month=m1, day=1)
            e = pd.Timestamp(year=y, month=m2, day=1)
            if e < s:
                s, e = e, s
            return s, e

    # range (fallback): <m1> <m2> <year>  (np. ASR bez myślnika: "kwiecień wrzesień 2010")
    m = re.search(r"\b(?P<m1>[a-ząćęłńóśźż]{3,})\s+(?P<m2>[a-ząćęłńóśźż]{3,})\s*(?P<y>\d{4})\b", q)
    if m:
        m1 = _MONTHS_PL_NORM.get(_norm_pl_token(m.group("m1"))) or _MONTHS_PL.get(m.group("m1"))
        m2 = _MONTHS_PL_NORM.get(_norm_pl_token(m.group("m2"))) or _MONTHS_PL.get(m.group("m2"))
        y = int(m.group("y"))
        if m1 and m2:
            s = pd.Timestamp(year=y, month=m1, day=1)
            e = pd.Timestamp(year=y, month=m2, day=1)
            if e < s:
                s, e = e, s
            return s, e

    # single month: <m> <year>
    m = re.search(r"(?P<m1>[a-ząćęłńóśźż]{3,})\s+(?P<y>\d{4})", q)
    if m:
        m1 = _MONTHS_PL_NORM.get(_norm_pl_token(m.group("m1"))) or _MONTHS_PL.get(m.group("m1"))
        y = int(m.group("y"))
        if m1:
            s = pd.Timestamp(year=y, month=m1, day=1)
            return s, s

    return None, None



def _is_other_label(x: str) -> bool:
    low = (x or "").strip().lower()
    return low in {"other", "inne", "pozostałe", "pozostale"}

def _fixed_palette(n: int) -> List[str]:
    # spójne z resztą app (PALETTE_MAIN w routerze), ale lokalnie, bez importu
    base = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
            "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    if n <= len(base):
        return base[:n]
    k = int(math.ceil(n / len(base)))
    return (base * k)[:n]

def _build_fixed_color_scale(domain: List[str], top_k: int) -> alt.Scale:
    """
    First top_k get palette colors, rest are shades of grey (deterministic).
    """
    dom = list(domain)
    palette = _fixed_palette(top_k)
    rest_n = max(0, len(dom) - top_k)
    # deterministic greys (light→dark)
    if rest_n > 0:
        greys = []
        for i in range(rest_n):
            # 0.82 .. 0.45
            v = 0.82 - (0.37 * (i / max(1, rest_n - 1)))
            g = int(round(v * 255))
            greys.append(f"#{g:02x}{g:02x}{g:02x}")
    else:
        greys = []
    rng = palette + greys
    return alt.Scale(domain=dom, range=rng)



def _build_fixed_color_map(domain: List[str], top_k: int) -> Dict[str, str]:
    """Return deterministic color per category (Top-K colored, rest grayscale).

    Uses the same palette logic as `_build_fixed_color_scale()` to keep colors consistent.
    This intentionally decouples legend order from ranking order without changing color mapping.
    """
    dom = [str(x) for x in list(domain)]
    top_k = max(0, int(top_k))

    # Top-K palette
    palette = _fixed_palette(min(top_k, len(dom)))

    # Greys for the rest (light -> dark)
    rest_n = max(0, len(dom) - top_k)
    if rest_n <= 0:
        greys: List[str] = []
    elif rest_n == 1:
        greys = ["#c7c7c7"]
    else:
        greys = [
            f"#{v:02x}{v:02x}{v:02x}"
            for v in [
                int(235 - (235 - 120) * i / (rest_n - 1))
                for i in range(rest_n)
            ]
        ]

    rng = palette + greys
    return {k: v for k, v in zip(dom, rng)}


# -----------------------------
# Interpretation (LLM JSON, reused contract)
# -----------------------------

_INTERP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "one_sentence": {"type": "string"},
        "what_chart_shows": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3},
        "key_insights": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
        "recommendations": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "segments": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 3},
        "limitations": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
    },
    "required": ["one_sentence", "what_chart_shows", "key_insights", "recommendations", "limitations"],
}


_INTERP_SYSTEM = """Jesteś Senior Strategy Consultant (McKinsey / Bain). Twoim zadaniem jest napisać sekcję „Interpretacja” do wykresu.
Piszesz po polsku, krótko, liczbowo i decyzyjnie.

ZASADY NIEPRZEKRACZALNE:
1) Korzystasz WYŁĄCZNIE z danych/statystyk podanych w polu `stats`. Nie wymyślasz budżetów, kwartałów (np. Q1 2024), prognoz, YoY, ani faktów spoza okresu.
2) Nazwy elementów analizowanego wymiaru zawsze podajesz dokładnie tak jak w danych i ujmujesz w cudzysłów, np. "Bags & Purses" albo "United Kingdom". Zakaz ogólników typu „kategoria lider”, „top kategoria”, „segment” bez nazwy.

REGUŁA TWARDA (driver insight — obowiązkowo w `one_sentence`):
- Jeśli w `stats` dostępne są: `driver_category_name` oraz delta (PLN lub pp), to **musisz** oprzeć `one_sentence` o te dane (nie o ogólny peak/total).
- `one_sentence` ma zawierać: KPI + nazwę drivera w cudzysłowie + zmianę (Δ) + wartości start/koniec + „dlatego …” + konkretne KPI do monitorowania.
- Dla udziałów używaj **pp** oraz **%** (z X% do Y%), dla wartości używaj **PLN**.

DODATKOWY ZAKAZ: w `one_sentence` nie używaj słów „kategoria”, „segment”, „lider”, „Top” jako substytutu etykiety wymiaru — zawsze podawaj wyłącznie nazwę w cudzysłowie (np. "Decorations" albo "United Kingdom").

3) Każdy punkt ma mieć kotwicę liczbową (PLN / % / pp / liczba transakcji / miesiąc). Zero pustych truizmów.

WYMAGANIA FORMATU:
- `one_sentence`: dokładnie 1 zdanie, maks. 28 słów. MUSI zawierać co najmniej 2 liczby.
  • Jeśli temat dotyczy udziałów (stats zawiera pola z `_pp` lub `%` udziału): MUSI zawierać % oraz pp (np. „z 12,4% do 15,1% (+2,7 pp)”).
  • Jeśli temat dotyczy wartości/ilości: MUSI zawierać PLN (lub ilość) oraz 2. kotwicę (np. miesiąc szczytu / Δ vs start / liczba transakcji).
- `key_insights`: dokładnie 4 punkty. Każdy punkt = 1 zdanie, zawiera co najmniej 1 liczbę. Co najmniej 2 z 4 punktów muszą wymieniać konkretny element analizowanego wymiaru.
- `recommendations`: dokładnie 3 punkty. Działania „co zrobić teraz”. Nie podawaj sztucznych % budżetu (np. +20%) ani terminów typu „w kolejnych 3 miesiącach”, chyba że takie liczby są w `stats`.
- `limitations`: 2–3 punkty. Tylko realne ograniczenia wynikające z danych (np. krótki horyzont, agregacja miesięczna, brak kanałów).

CEL: wynik ma brzmieć jak consulting-grade insight: konkret + liczby + decyzja, bez lania wody."""


def _build_interp_system_prompt(stats: Optional[Dict[str, Any]] = None) -> str:
    stats = stats if isinstance(stats, dict) else {}
    dim_label = _dimension_display_label(
        stats.get("dimension_label") or stats.get("cat_col"),
        fallback="Wymiar",
    )
    dim_entity = _dimension_entity_label(
        stats.get("dimension_label") or stats.get("cat_col"),
        fallback="pozycja",
    )
    return (
        _INTERP_SYSTEM
        + "\n\n"
        + f"ANALIZOWANY_WYMIAR: {dim_label}.\n"
        + f"Jeśli ten wymiar nie jest kategorią, nie używaj słowa 'kategoria' jako domyślnego rzeczownika. "
        + f"Używaj neutralnie '{dim_entity}' albo po prostu oryginalnej etykiety z danych."
    )

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items or []:
        t = re.sub(r"\s+", " ", str(it)).strip()
        key = t.lower()
        if not t or key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _ensure_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    # split on newlines or bullet-like separators
    parts = [p.strip(" •\t-") for p in re.split(r"[\n\r]+", s) if p.strip(" •\t-")]
    return parts if parts else [s]


def _enforce_one_sentence_contract(one_sentence: str, anchors: List[str], fallback_action: str) -> str:
    """
    Enforce MBB-style constraint for 'one_sentence':
    - max 2 sentences
    - must contain: fact+anchors + mechanism + decision/implication
    We allow either:
      A) single sentence with '; dlatego:' (preferred), or
      B) 2 sentences where the 2nd starts with 'Dlatego:' and contains decision language.
    """
    s = (one_sentence or "").strip().replace("**", "")
    s = re.sub(r"\s+", " ", s).strip()

    # Split into sentences, keep at most 2
    parts = [p.strip() for p in re.split(r"(?<=[\.!\?])\s+", s) if p.strip()]
    if len(parts) > 2:
        parts = parts[:2]
    s = " ".join(parts).strip()

    # Ensure a decision/implication connector exists
    sl = s.lower()
    has_connector = ("; dlatego" in sl) or ("; więc" in sl) or (" dlatego:" in sl) or (" dlatego " in sl)
    if not has_connector:
        # If we already have 2 sentences, convert second into 'Dlatego: ...'
        if len(parts) >= 2:
            # keep first as fact+mechanism, second as decision
            second = parts[1]
            if not second.lower().startswith("dlatego"):
                second = "Dlatego: " + second.lstrip("—-:;,. ").strip()
            s = parts[0].rstrip(".") + ". " + second
        else:
            # single sentence: append '; dlatego:' with fallback action
            s = s.rstrip(".") + f"; dlatego: {fallback_action}".strip()

    # Ensure ≥2 anchors appear early (prefer first sentence / fact clause)
    anchors = [a for a in (anchors or []) if isinstance(a, str) and a.strip()]
    if anchors:
        # define "fact region": before '; dlatego' if present, else first sentence
        lower = s.lower()
        cut = lower.find("; dlatego")
        if cut == -1:
            cut = lower.find("; więc")
        if cut == -1:
            # up to first sentence end (.)
            dot = s.find(".")
            cut = dot if dot != -1 else len(s)
        fact = s[:cut]
        found = sum(1 for a in anchors if a in fact)
        if found < 2:
            add = ", ".join(anchors[: max(2, min(4, len(anchors)))])
            # add parenthetical before connector / end of first sentence
            if ";" in fact:
                fact2 = fact.rstrip(".") + f" ({add})"
                s = fact2 + s[len(fact):]
            else:
                if "." in s:
                    first, rest = s.split(".", 1)
                    first = first.strip().rstrip(".") + f" ({add})"
                    s = first + "." + rest
                else:
                    s = s.rstrip(".") + f" ({add})"

    # Normalize connector for readability (prefer '; dlatego:' if single sentence)
    # Do not force into one sentence; just normalize spacing.
    s = re.sub(r"\s*;\s*", "; ", s)
    s = re.sub(r"\s*:\s*", ": ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def _postprocess_interp_dict(interp: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(interp or {})
    anchors = _ensure_list((payload or {}).get("anchors"))
    # Fallback action: take first recommendation or a safe generic
    recs = _ensure_list(out.get("recommendations"))
    fallback_action = recs[0] if recs else "skoncentruj działania na głównych driverach i monitoruj efekt co miesiąc"

    out["one_sentence"] = _enforce_one_sentence_contract(str(out.get("one_sentence") or ""), anchors, fallback_action)

    # Sanitize placeholder-like multiplier notation that often trips semantic validation ("2x", "2 x").
    def _sanitize_x_multiplier(s: str) -> str:
        if not s:
            return s
        return re.sub(r"(\d)\s*x\b", r"\1×", s)

    out["one_sentence"] = _sanitize_x_multiplier(out["one_sentence"])
    out["one_sentence"] = re.sub(r'^KPI:\s*udzia[łl]\s*\(%\)\s*—\s*', '', out["one_sentence"], flags=re.IGNORECASE).strip()
    out["one_sentence"] = re.sub(r"\bw\s+w\b", "w", out["one_sentence"], flags=re.IGNORECASE)
    out["one_sentence"] = re.sub(r"\s{2,}", " ", out["one_sentence"]).strip()
    out["what_chart_shows"] = _dedup_keep_order(_ensure_list(out.get("what_chart_shows")))[:3]
    out["what_chart_shows"] = [_sanitize_x_multiplier(w) for w in out["what_chart_shows"]]
    out["what_chart_shows"] = [w[:110].rstrip() + ("…" if len(w) > 110 else "") for w in out["what_chart_shows"]]
    if len(out["what_chart_shows"]) < 2 and anchors:
        # ensure minimal 2 lines
        out["what_chart_shows"] = (out["what_chart_shows"] + [f"Kluczowe liczby (anchors): {', '.join(anchors[:3])}."])[:2]

    
    # Key insights: dedupe -> score/rank -> keep Top-N (McKinsey-style compression)
    kis = _dedup_keep_order(_ensure_list(out.get("key_insights")))
    kis = [_sanitize_x_multiplier(k) for k in kis]
    kis = [k[:220].rstrip() + ("…" if len(k) > 220 else "") for k in kis]

    primary_cat = _extract_primary_category_from_text(out.get("one_sentence", ""))
    os_lower = out["one_sentence"].lower()

    def _ki_score(t: str) -> float:
        tl = (t or "").lower()
        score = 0.0
        # numeric density
        score += 0.6 * _count_numbers(tl)
        # pp / % / PLN anchors
        if "pp" in tl:
            score += 1.2
        if "%" in tl:
            score += 0.8
        if "pln" in tl:
            score += 0.8
        # business anchor terms
        for kw in ["top 10", "top10", "szczyt", "minimum", "max", "min", "hhi"]:
            if kw in tl:
                score += 0.6
        # category anchor
        if primary_cat and primary_cat.lower() in tl:
            score += 1.0
        # penalty: generic filler / mirrors one_sentence
        if tl in os_lower:
            score -= 2.0
        if any(p in tl for p in ["analiza pokazuje", "różnice między kategoriami", "wybierz", "ustaw mierzalny cel"]):
            score -= 1.5
        return score

    kis = sorted(kis, key=_ki_score, reverse=True)
    out["key_insights"] = kis[:4]
    out["recommendations"] = _dedup_keep_order(recs)[:3]
    out["recommendations"] = [_sanitize_x_multiplier(r) for r in out["recommendations"]]
    out["limitations"] = _dedup_keep_order(_ensure_list(out.get("limitations")))[:4]
    out["limitations"] = [_sanitize_x_multiplier(l) for l in out["limitations"]]
    out["segments"] = _dedup_keep_order(_ensure_list(out.get("segments")))[:3]
    return out

# ============================
# HARD VALIDATOR v2 (COT)
# Enforces: number + percent + KPI + implication/decision
# and blocks generic recommendations (Distribution-like gating).
# ============================

_GENERIC_PATTERNS = [
    r"warto\s+rozwa(ż|z)y(ć|c)",
    r"mo(ż|z)na\s+rozwa(ż|z)y(ć|c)",
    r"mo(ż|z)e\s+sugerowa(ć|c)",
    r"wydaje\s+si(ę|e)",
    r"prawdopodobnie",
    r"nale(ż|z)y\s+przeanalizowa(ć|c)",
    r"ten\s+wykres\s+pokazuje",
    r"w\s+danych\s+wida(ć|c)",
    r"analiza\s+pokazuje",
    r"zaleca\s+si(ę|e)\s+dalsz",
    r"dalsz(a|e)\s+analiz",
    r"zidentyfikuj\s+kategorie\s+rosn",
    r"monitorowa(ć|c)\s+zmiany",
    r"skup\s+si(ę|e)\s+na\s+analizie",
]

_KPI_KEYWORDS = [
    "sprzeda", "warto", "wolumen", "ilo", "transakc", "udzia", "mar", "koszt", "przych",
    "zysk", "roi", "roas", "hhi", "koszyk", "konwers", "cena", "rabat", "promo",
]

_DECISION_MARKERS = [
    "dlatego", "rekomend", "nale", "trzeba", "powinni", "zwi", "zmniejsz", "przenie",
    "alokuj", "zainwestuj", "wstrzymaj", "ogranicz", "skoncentruj", "priorytetyz",
]

def _has_number(s: str) -> bool:
    # absolute number (with optional thousands separators / decimals)
    return bool(re.search(r"\b\d{1,3}(?:[\s.,]\d{3})*(?:[\.,]\d+)?\b", s))

def _has_percent(s: str) -> bool:
    # % or pp
    return bool(re.search(r"(\d+[\.,]?\d*\s*%|\d+[\.,]?\d*\s*pp)\b", s.lower()))

def _has_kpi(s: str) -> bool:
    sl = s.lower()
    return any(k in sl for k in _KPI_KEYWORDS)

def _has_decision(s: str) -> bool:
    sl = (s or "").lower()

    # Accept either:
    # A) explicit inline connector ('; dlatego:' etc.), or
    # B) 2-sentence form where 2nd sentence starts with 'Dlatego:' (board-style decision line).
    has_inline = ("; dlatego" in sl) or ("; więc" in sl) or (" dlatego:" in sl) or (" więc:" in sl)
    has_second = bool(re.search(r"\.\s*dlatego\s*:", sl))

    has_marker = any(m in sl for m in _MBB_DECISION_MARKERS)
    return (has_inline or has_second) and has_marker

def _is_generic(s: str) -> bool:
    sl = s.lower()
    return any(re.search(p, sl) for p in _GENERIC_PATTERNS)

def _normalize_anchor(s: str) -> str:
    # for dedup: remove whitespace, unify decimal separators, strip words
    sl = s.lower()
    sl = re.sub(r"\s+", " ", sl).strip()
    sl = sl.replace(",", ".")
    sl = re.sub(r"[^0-9a-z%\. ]", "", sl)
    return sl

def _interp_expect_percent(payload: Dict[str, Any]) -> bool:
    """Heuristic: require %/pp for share/mix narratives when the underlying stats support it."""
    try:
        title = str((payload or {}).get("chart_title") or "").lower()
        desc = str((payload or {}).get("chart_desc") or "").lower()
        js = json_dumps_safe((payload or {}).get("stats") or {}).lower()
        return any(k in title + " " + desc + " " + js for k in ["udz", "share", "pp", "mix", "hhi", "koncentr", "top-3", "top3", "top-5", "top5"])
    except Exception:
        return False

def _count_numbers(s: str) -> int:
    try:
        return len(re.findall(r"\d+(?:[\.,]\d+)?", s or ""))
    except Exception:
        return 0

def _has_placeholder_text(s: str) -> bool:
    """Hard blocker for placeholder tokens.

    Any placeholder-like category token is a hard failure (trust/polish).
    """
    if not s:
        return False

    sl = s.lower()

    # Explicit placeholders (Polish)
    if "kategoria-" in sl:
        return True
    if re.search(r"\bkategoria\s*[-–—]\s*\d+\b", sl):
        return True
    if re.search(r"\bkategoria\s*[-_]?\s*\d+\b", sl):
        return True

    # Explicit placeholders (English)
    if "category-" in sl or "category_" in sl:
        return True
    if re.search(r"\bcategory\s*[-–—]\s*\d+\b", sl):
        return True

    # Template tokens
    if "{category" in sl or "<category" in sl or "[category" in sl:
        return True
    if "{kategoria" in sl or "<kategoria" in sl or "[kategoria" in sl:
        return True

    return False

# =========================
# Formatting helpers (PL)
# =========================

_PL_MONTHS = {
    1: "styczniu", 2: "lutym", 3: "marcu", 4: "kwietniu", 5: "maju", 6: "czerwcu",
    7: "lipcu", 8: "sierpniu", 9: "wrześniu", 10: "październiku", 11: "listopadzie", 12: "grudniu",
}

def _format_month_pl(date_like: str) -> str:
    """Return Polish 'w <miesiącu> <roku>' label from 'YYYY-MM-DD' or 'YYYY-MM'."""
    if not date_like:
        return ""
    s = str(date_like).strip()
    try:
        if len(s) >= 10:
            y = int(s[0:4]); m = int(s[5:7])
        elif len(s) >= 7:
            y = int(s[0:4]); m = int(s[5:7])
        else:
            return s
        mm = _PL_MONTHS.get(m, str(m))
        return f"w {mm} {y} roku"
    except Exception:
        return s

def _cot_fmt_int(x: Any) -> str:
    """Integer formatting with thin/space thousands separators (PL-friendly)."""
    try:
        if x is None:
            return "—"
        n = int(round(float(x)))
        return f"{n:,}".replace(",", " ")
    except Exception:
        return str(x)

def _cot_fmt_pln(x: Any) -> str:
    """Format PLN with Polish-style spacing; robust for None."""
    try:
        if x is None:
            return "— PLN"
        v = float(x)
        s = f"{v:,.0f}".replace(",", " ")
        return f"{s} PLN"
    except Exception:
        return "— PLN"


def _cot_fmt_pct(x: Any, digits: int = 1) -> str:
    try:
        if x is None:
            return "—%"
        return f"{float(x):.{digits}f}%"
    except Exception:
        return "—%"


def _cot_fmt_pp(x: Any, digits: int = 1) -> str:
    try:
        if x is None:
            return "— pp"
        return f"{float(x):+.{digits}f} pp"
    except Exception:
        return "— pp"

def _fmt_float(x: Any, digits: int = 2) -> str:
    try:
        if x is None:
            return "n/a"
        xv = float(x)
        return f"{xv:.{digits}f}".replace(".", ",")
    except Exception:
        return "n/a"

def _first_not_none(*vals: Any) -> Any:
    """Return the first argument that is not None / NaN.

    Important: unlike plain `or`, this keeps legitimate zeros (0 / 0.0).
    """
    for v in vals:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        return v
    return None


def _share_to_pct(v: Any) -> Any:
    """Normalize share values to percentage points expected by *_pct formatters.

    Internal builders sometimes pass fractional shares (0.101) while UI formatters
    expect percentages (10.1). This helper keeps already-percent values unchanged
    and converts only fraction-like values.
    """
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return v
    if not np.isfinite(fv):
        return None
    return fv * 100.0 if abs(fv) <= 1.000001 else fv


def _enforce_share_one_sentence_driver_delta(out: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    """Post-processor to make shares_pp one_sentence validator-safe.

    Guarantees (when stats provide values):
    - concrete category label in quotes, e.g. "Decorations"
    - pp delta in numeric form, e.g. -1.2 pp (never standalone 'pp')
    This is dataset-agnostic: it only uses provided stats and falls back gracefully.
    """
    try:
        if not isinstance(out, dict):
            return out
        s = (out.get("one_sentence") or "").strip()
        if not s:
            return out

        st = stats or {}

        # --- driver category (prefer explicit driver pack, then common aliases) ---
        cat = (
            st.get("driver_category_name")
            or st.get("primary_category")
            or st.get("leader_category")
            or st.get("leader")
            or st.get("top_category")
            or st.get("category")
        )
        if cat is not None:
            cat = str(cat).strip()
            # avoid placeholders / empty
            if cat and not _has_placeholder_text(cat):
                if f'"{cat}"' not in s:
                    # replace generic labels if present
                    s2 = re.sub(r"\b(wiodąca|dominująca|główna|kluczowa)\s+kategoria\b", f'kategoria "{cat}"', s, flags=re.IGNORECASE)
                    s2 = re.sub(r"\bkategoria\s+(odniesienia|lider|driver)\b", f'kategoria "{cat}"', s2, flags=re.IGNORECASE)
                    if s2 == s:
                        # inject only the category label; avoid noisy KPI prefixes in final UI text
                        if s.lower().startswith("kpi:"):
                            s2 = re.sub(r"^(KPI:\s*udzia[łl].*?—\s*)", r"\1" + f'kategoria "{cat}" ', s, flags=re.IGNORECASE)
                            if s2 == s:
                                s2 = f'kategoria "{cat}": {s}'
                        else:
                            s2 = f'kategoria "{cat}": {s}'
                    s = s2

        # --- pp delta (prefer explicit driver delta, then aliases, then compute from start/end) ---
        delta_pp = _first_not_none(
            st.get("driver_delta_pp"),
            st.get("primary_delta_pp"),
            st.get("leader_delta_pp"),
            st.get("delta_pp"),
            st.get("share_delta_pp"),
        )

        start_pct = _share_to_pct(_first_not_none(
            st.get("driver_start_pct"),
            st.get("primary_start_pct"),
            st.get("leader_start_pct"),
            st.get("driver_start_share"),
            st.get("leader_start_share"),
        ))
        end_pct = _share_to_pct(_first_not_none(
            st.get("driver_end_pct"),
            st.get("primary_end_pct"),
            st.get("leader_end_pct"),
            st.get("driver_end_share"),
            st.get("leader_end_share"),
        ))

        if delta_pp is None:
            try:
                if start_pct is not None and end_pct is not None:
                    delta_pp = float(end_pct) - float(start_pct)
            except Exception:
                delta_pp = None

        # Fix standalone 'pp' token (no number) if present
        if re.search(r"\bpp\b", s) and not re.search(r"[+-]?\d+[\d,\.]*\s*pp\b", s):
            if delta_pp is not None:
                s = re.sub(r"\bpp\b", _cot_fmt_pp(delta_pp), s)

        # Ensure numeric pp delta exists somewhere in the sentence (if we have it)
        if delta_pp is not None:
            if not re.search(r"[+-]?\d+[\d,\.]*\s*pp\b", s):
                phrase = f" zmieniła udział o {_cot_fmt_pp(delta_pp)}"
                if start_pct is not None and end_pct is not None:
                    phrase += f" (z {_cot_fmt_pct(start_pct)} do {_cot_fmt_pct(end_pct)})"
                # insert before '; dlatego' if present to keep structure
                if ";" in s:
                    left, right = s.split(";", 1)
                    s = left.rstrip(". ") + phrase + ";" + right.lstrip()
                else:
                    s = s.rstrip(". ") + phrase + "."

        out["one_sentence"] = (s or "").strip()
        return out
    except Exception:
        return out



def _validate_overview_share_exec_sentence(out: Dict[str, Any], stats: Dict[str, Any]) -> Tuple[bool, List[str]]:
    out = dict(out or {})
    stats = stats if isinstance(stats, dict) else {}
    txt = " ".join(str(out.get("one_sentence") or "").strip().split())
    tl = txt.lower()
    reasons: List[str] = []

    if not txt:
        return False, ["overview_share_empty_one_sentence"]

    if not re.search(r"\d", txt):
        reasons.append("overview_share_missing_numeric_anchor")

    driver = (
        stats.get("driver_category_name")
        or stats.get("leader_category")
        or stats.get("top1_category")
    )
    if driver and str(driver).lower() not in tl:
        reasons.append("overview_share_missing_driver_name")

    if not any(w in tl for w in [
        "oddawanie udziału", "osłabienie pozycji", "transfer udziału",
        "trwałe przesunięcie", "erozj", "wzmacnianie pozycji",
        "umocnienie pozycji", "przejęcie dodatkowego udziału",
    ]):
        reasons.append("overview_share_missing_mechanism")

    if not any(w in tl for w in [
        "cena", "alokacja wsparcia",
        "przesuń", "dlatego:", "decyzja:",
    ]):
        reasons.append("overview_share_missing_decision")

    if any(w in tl for w in [
        "przegląd strategii", "zainwestować w działania marketingowe",
        "należy monitorować", "warto rozważyć"
    ]):
        reasons.append("overview_share_generic_language")

    return (len(reasons) == 0, reasons)


def _review_shares_one_sentence_quality(text: str) -> List[str]:
    t = str(text or "").strip().lower()
    reasons: List[str] = []
    if not t:
        return ["empty"]
    if any(p in t for p in [
        "monitorowanie dalszych trendów",
        "monitorować dalsze trendy",
        "należy monitorować",
        "warto monitorować",
        "obserwować zmiany",
        "analizować dalej",
    ]):
        reasons.append("soft_decision")
    if not any(w in t for w in [
        "oddawanie udziału",
        "osłabienie pozycji",
        "transfer udziału",
        "trwałe przesunięcie",
        "erozj",
        "wzmacnianie pozycji",
        "umocnienie pozycji",
        "przejęcie dodatkowego udziału",
    ]):
        reasons.append("missing_business_mechanism")
    if not any(w in t for w in [
        "alok", "portfel", "ekspozycja", "priorytet",
        "przenieś", "zwiększ", "ogranicz", "utrzymaj", "skaluj",
    ]):
        reasons.append("missing_managerial_decision")
    return reasons


def _review_interp_recommendations_quality(
    recs: List[str],
    *,
    mode: str,
    stats: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []
    rows = [str(x or "").strip().lower() for x in (recs or []) if str(x or "").strip()]
    if len(rows) < 2:
        reasons.append("too_few_recommendations")
    soft_hits = 0
    strong_hits = 0
    for r in rows:
        if any(p in r for p in ["monitorować", "obserwować", "analizować", "warto rozważyć"]):
            soft_hits += 1
        if any(w in r for w in [
            "zwiększ", "ogranicz", "utrzymaj", "przenieś", "skaluj",
            "alok", "portfel", "ekspozycja", "kpi", "w ciągu", "m/m", "pp",
        ]):
            strong_hits += 1
    if rows and soft_hits >= len(rows):
        reasons.append("recommendations_too_soft")
    if strong_hits == 0:
        reasons.append("recommendations_not_actionable")
    return reasons


def _upgrade_what_chart_shows(
    out: Dict[str, Any],
    *,
    mode: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    o = dict(out or {})
    if mode in ("shares_pp", "share", "mix"):
        o["what_chart_shows"] = [
            "Wykres pokazuje, jak zmienia się udział kategorii w miksie sprzedaży miesiąc po miesiącu.",
            "Pozwala odróżnić trwałą zmianę struktury popytu od krótkiego szumu.",
            "Ułatwia ocenę, czy lider portfela broni pozycji, czy oddaje udział konkurującym kategoriom.",
        ]
        return o
    if mode in ("value", "sprzedaz", "sales"):
        o["what_chart_shows"] = [
            "Wykres pokazuje poziom sprzedaży w czasie oraz momenty kumulacji popytu.",
            "Pozwala odróżnić stabilny trend od pików sezonowych i zdarzeń jednorazowych.",
            "Ułatwia ocenę, kiedy sprzedaż wymaga wsparcia dostępnością, ekspozycją lub promocją.",
        ]
        return o
    return o


def _build_overview_det_premium_value(stats: Dict[str, Any], chart_title: str, chart_desc: str) -> Dict[str, Any]:
    payload = {"chart_title": chart_title, "chart_desc": chart_desc, "mode": "value", "expect_pct": False, "stats": stats}
    out = _cot_build_interp_payload_strict(stats or {}, mode_hint="value")
    out = _postprocess_interp_dict(out, payload)
    out = _postprocess_interp_json(out, stats or {}, mode="value")
    out = _upgrade_what_chart_shows(out, mode="value", stats=stats or {})
    return out


def _build_overview_det_premium_shares(stats: Dict[str, Any], chart_title: str, chart_desc: str) -> Dict[str, Any]:
    payload = {"chart_title": chart_title, "chart_desc": chart_desc, "mode": "shares_pp", "expect_pct": True, "stats": stats}
    out = _cot_build_interp_payload_strict(stats or {}, mode_hint="shares_pp")
    out = _postprocess_interp_dict(out, payload)
    out = _postprocess_interp_json(out, stats or {}, mode="shares_pp")
    out = _enforce_share_one_sentence_driver_delta(out, stats or {})
    out = _upgrade_what_chart_shows(out, mode="shares_pp", stats=stats or {})
    return out


def _quick_interp_score(
    interp: Dict[str, Any],
    *,
    mode: str,
    chart_title: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    one = str((interp or {}).get("one_sentence") or "").strip()
    shows = list((interp or {}).get("what_chart_shows") or [])
    insights = list((interp or {}).get("key_insights") or [])
    recs = list((interp or {}).get("recommendations") or [])
    limits = list((interp or {}).get("limitations") or [])

    score = 0
    reasons: List[str] = []
    penalties: List[str] = []

    if len(re.findall(r"\d+(?:[.,]\d+)?", one)) >= 2:
        score += 1
        reasons.append("one_sentence_numbers")
    if any(q in one for q in ['"', '„', '”', "'"]):
        score += 1
        reasons.append("named_driver")
    if any(w in one.lower() for w in [
        "oznacza", "wskazuje", "sygnalizuje", "implikuje",
        "presja", "szansa", "ryzyko", "koncentracja", "dywersyfikacja",
        "koncentracja przychodu", "szczyt", "okresach szczytu", "peak", "pik sprzedaży"
    ]):
        score += 1
        reasons.append("business_mechanism")

    if mode in ("value", "sprzedaz", "sales"):
        if any(m in one.lower() for m in [
            "stycz", "lut", "mar", "kwi", "maj", "cze", "lip", "sie", "wrze", "paź", "listop", "grud",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]):
            score += 1
            reasons.append("named_peak_period")

    if recs and any(
        any(w in str(r).lower() for w in [
            "alok", "priorytet", "przenieś", "zwiększ", "ogranicz",
            "utrzymaj", "skaluj", "testuj", "monitoring", "ekspozycja", "portfel",
        ])
        for r in recs
    ):
        score += 1
        reasons.append("actionable_recommendation")
    if len(insights) >= 2 and len(limits) >= 1 and len(shows) >= 2:
        score += 1
        reasons.append("complete_structure")

    if mode in ("shares_pp", "share", "mix"):
        if "%" not in one and "pp" not in one.lower():
            score -= 1
            penalties.append("missing_share_anchor")
        if any(p in one.lower() for p in [
            "monitorowanie dalszych trendów",
            "warto monitorować",
            "należy monitorować",
            "obserwować dalsze zmiany",
        ]):
            score -= 1
            penalties.append("soft_decision")

    score = max(0, min(5, score))
    return {"score": score, "reasons": reasons, "penalties": penalties}


def _select_best_overview_interp(
    *,
    mode: str,
    chart_title: str,
    stats: Dict[str, Any],
    llm_interp: Dict[str, Any],
    llm_valid: bool,
    llm_src: str,
    det_interp: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    det_eval = _quick_interp_score(det_interp, mode=mode, chart_title=chart_title, stats=stats)

    if not llm_valid or not isinstance(llm_interp, dict):
        return det_interp, "fallback_deterministic_selected", {
            "score_llm": 0,
            "score_det": det_eval["score"],
            "selected_candidate": "deterministic",
            "selection_reason": "llm_validation_failed",
            "llm_eval": {"score": 0, "reasons": [], "penalties": ["validation_failed"]},
            "det_eval": det_eval,
        }

    llm_eval = _quick_interp_score(llm_interp, mode=mode, chart_title=chart_title, stats=stats)

    score_llm = int(llm_eval.get("score") or 0)
    score_det = int(det_eval.get("score") or 0)

    if score_llm > score_det:
        return llm_interp, llm_src, {
            "score_llm": score_llm,
            "score_det": score_det,
            "selected_candidate": "llm",
            "selection_reason": "llm_higher_score",
            "llm_eval": llm_eval,
            "det_eval": det_eval,
        }

    if score_llm == score_det and llm_valid:
        return llm_interp, llm_src, {
            "score_llm": score_llm,
            "score_det": score_det,
            "selected_candidate": "llm",
            "selection_reason": "llm_on_tie_all_modes",
            "llm_eval": llm_eval,
            "det_eval": det_eval,
        }

def _legacy_detect_exec_why_now_tokens(text: str):
    tl = (text or "").lower()

    tokens = [
        "ryzyko utraty udziału",
        "ryzyka utraty udziału",
        "z powodu ryzyka utraty udziału",
        "aby przeciwdziałać ryzyku utraty udziału",
        "aby zminimalizować ryzyko utraty udziału",
        "aby uniknąć ryzyka utraty udziału",
        "presja na obronę lidera",
        "presję na obronę lidera",
        "okno przejęcia popytu",
        "trwałe przesunięcie popytu",
        "koszt zaniechania",
        "koszt błędu planowania",
        "ryzyko niedostępności",
        "ryzykiem niedostępności",
        "ryzyko niedoszacowania popytu",
        "ryzykiem niedoszacowania popytu",
        "niedoszacowania popytu",
        "koszt utraty udziału",
    ]

    return [w for w in tokens if w in tl]


def _exec_has_named_mechanism_signal(text: str, block_id: str = "") -> bool:
    t = " ".join(str(text or "").strip().split())
    tn = _norm_pl_token(t)

    if any(
        token in tn
        for token in [
            "transfer udzial",
            "przejecie popytu",
            "oddawanie udzial",
            "erozj",
            "obrona lidera",
            "trwale przesuniecie popytu",
            "trwale przesuniecie struktury",
            "koszt bledu planowania",
            "zaleznosc od rdzenia",
            "bardziej zalezna od rdzenia",
            "rosnaca zaleznosc od rdzenia",
            "rdzen portfela",
            "odpornosc portfela",
            "priorytet zatowarowania",
            "ryzyko niedoszacowania popytu",
            "niedoszacowania popytu",
        ]
    ):
        return True

    if block_id == "cot__mix_share_topN":
        return any(
            token in tn
            for token in [
                "traci pozycje w miksie",
                "traci pozycje lidera",
                "utrata pozycji lidera",
                "top-n spadl",
                "spadek top-n",
                "spadek top n",
                "obrony lidera",
            ]
        )

    return False


def _exec_has_concentration_exec_signal(text: str) -> bool:
    t = " ".join(str(text or "").strip().split())
    tn = _norm_pl_token(t)

    if any(
        token in tn
        for token in [
            "zaleznosc od rdzenia",
            "bardziej zalezna od rdzenia",
            "rosnaca zaleznosc od rdzenia",
            "odpornosc portfela",
            "dywersyfik",
            "koncentrac",
            "challenger",
            "challengery",
            "wzmocnienie challenger",
        ]
    ):
        return True

    return any(
        re.search(pattern, tn)
        for pattern in [
            r"\brealokac\w*\s+wspar\w*\s+(?:dla|na)\s+challenger\w*",
            r"\bogranicz\w*\s+zalezn\w*\s+od\s+rdzenia",
            r"\bwzmocn\w*\s+challenger\w*",
        ]
    )


def _legacy_exec_has_winners_losers_exec_signal(text: str) -> bool:
    t = " ".join(str(text or "").strip().split())
    tn = _norm_pl_token(t)

    has_mechanism = any(
        token in tn
        for token in [
            "transfer udzial",
            "przejecie popytu",
            "okno przejecia popytu",
        ]
    )
    has_winner_lever = any(
        token in tn
        for token in [
            "availability",
            "dostepn",
            "ekspozyc",
            "promo",
            "promoc",
            "wsparcie promocyjne",
            "zwieksz ekspozyc",
            "zwieksz availability",
            "wzmocnij ekspozyc",
        ]
    )
    has_loser_lever_or_diagnosis = any(
        token in tn
        for token in [
            "cena",
            "price pack",
            "price-pack",
            "sezonow",
            "ograniczenie wsparcia",
            "ogranicz wspar",
            "rola kategor",
            "rola asortyment",
            "obecnosc na polce",
            "sprawdz cen",
            "rewizj strategii cen",
        ]
    )

    return has_mechanism and has_winner_lever and has_loser_lever_or_diagnosis


def _exec_has_start_end_direction_signal(text: str) -> bool:
    t = " ".join(str(text or "").strip().split())
    tn = _norm_pl_token(t)

    if any(
        token in tn
        for token in [
            "bazowa alokacja",
            "alokacj bazow",
            "zaktualizuj bazow",
            "zaktualizuj alokacj",
            "dostosuj bazow",
            "dostosowac bazow",
            "alokacja wsparcia",
            "plan wspar",
            "ekspozycja bazowa",
            "zwieksz ekspozyc",
            "realokacja wsparcia",
            "realokuj wspar",
            "przesun wspar",
            "przeniesienie wsparcia",
            "w kierunku kategorii",
            "w kierunku rosnacej kategorii",
            "w kierunku zwyciezcy",
            "ograniczenie wsparcia",
            "ogranicz wspar",
            "obrona udzialu",
            "obrona lidera",
            "rola kategor",
            "rola asortyment",
            "wzmocnienie roli",
            "wzmocnij role",
            "dostosuj role",
            "dostosowac role",
        ]
    ):
        return True

    return any(
        re.search(pattern, tn)
        for pattern in [
            r"\b(?:wzmocn\w*|dostos\w*|zaktualiz\w*|realok\w*|przesun\w*|przenies\w*|ogranicz\w*|obron\w*|ustaw\w*)\b.*\b(?:rol\w*|alokac\w*|ekspozyc\w*|wspar\w*|udzial\w*)\b",
            r"\b(?:rol\w*|alokac\w*|ekspozyc\w*|wspar\w*|udzial\w*)\b.*\b(?:w\s+miksie|w\s+portfelu|w\s+ukladzie|bazow\w*)\b",
            r"\bw\s+kierunku\s+(?:zwyciezc\w*|rosnac\w*|silniejsz\w*|driver\w*)\b",
        ]
    )


def _exec_has_start_end_exec_signal(text: str) -> bool:
    t = " ".join(str(text or "").strip().split())
    tn = _norm_pl_token(t)

    has_mechanism = any(
        token in tn
        for token in [
            "trwale przesuniecie struktury",
            "trwale przesuniecie popytu",
            "trwale przesuniecie",
            "strukturalne przesuniecie",
            "zmiana struktury popytu",
            "zmiana ukladu kategorii",
            "nowy uklad popytu",
            "nowy uklad kategorii",
            "nowy uklad",
        ]
    )
    has_direction = _exec_has_start_end_direction_signal(t)
    return has_mechanism and has_direction


def _exec_has_business_lever_signal(text: str) -> bool:
    t = " ".join(str(text or "").strip().split())
    tn = _norm_pl_token(t)

    if any(
        token in tn
        for token in [
            "availability",
            "dostepn",
            "ekspozyc",
            "zapas",
            "zatowarowan",
            "cena",
            "challenger",
        ]
    ):
        return True

    return any(
        re.search(pattern, tn)
        for pattern in [
            r"\bdostepn\w*",
            r"\bprice(?:\s+|-)?pack\b",
            r"\bplan\w*\s+wspar\w*",
            r"\brealokac\w*\s+wspar\w*",
            r"\brealokuj\s+wspar\w*",
            r"\bprzesun\w*\s+wspar\w*",
            r"\bprzesunieci\w*\s+wspar\w*",
            r"\brola\w*\s+kategor\w*",
            r"\brola\w*\s+asortyment\w*",
            r"\bkalendarz\w*\s+aktywac\w*",
            r"\balokac\w*\s+bazow\w*",
            r"\bbazow\w*\s+alokac\w*",
            r"\bekspozyc\w*\s+bazow\w*",
            r"\bobron\w*\s+lider\w*",
            r"\bwzmocn\w*\s+rol\w*",
            r"\bdostos\w*\s+rol\w*",
            r"\brola\w*\s+(?:\w+\s+){0,4}(?:w\s+miksie|w\s+portfelu|w\s+ukladzie)\b",
        ]
    )


def _exec_has_decision_signal(text: str) -> bool:
    t = " ".join(str(text or "").strip().split())
    tl = t.lower()
    tn = _norm_pl_token(t)

    if any(
        token in tl
        for token in [
            "decyzja:",
            "dlatego:",
            "rekomendacja:",
            "wzmocnij",
            "utrzymaj",
            "ogranicz",
            "availability",
        ]
    ):
        return True

    if _exec_has_business_lever_signal(t):
        return True

    return any(
        token in tn
        for token in [
            "decyzja",
            "dlatego",
            "rekomendacja",
            "wzmocnij",
            "utrzymaj",
            "ogranicz",
            "zwieksz",
            "przenies",
            "przesun wspar",
            "realokacj wspar",
            "plan wspar",
        ]
    )

def _quick_narrative_score(text: str, block_id: str, mode: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    t = " ".join(str(text or "").strip().split())
    tl = t.lower()
    tn = _norm_pl_token(t)
    stats = stats if isinstance(stats, dict) else {}

    score = 0
    reasons: List[str] = []
    penalties: List[str] = []

    if re.search(r"\d", t):
        score += 1
        reasons.append("numerical_evidence")

    names: List[str] = []
    for k, v in stats.items():
        kl = str(k).lower()
        if any(tok in kl for tok in ["category", "leader", "winner", "loser", "driver"]) and isinstance(v, str):
            vv = v.strip()
            if vv and vv.lower() not in {"other", "unknown", "kategorii", "lider", "winner", "loser"}:
                names.append(vv)

    if any(n.lower() in tl for n in names) or any(q in t for q in ['"', '„', '”']):
        score += 1
        reasons.append("category_or_driver_reference")

    if any(w in tl for w in [
        "oznacza", "wskazuje na", "sygnalizuje", "implikuje", "oddawanie udziału",
        "trwałą zmianę struktury", "transfer udziału", "przejęcie popytu",
        "erozję rdzenia", "osłabienie pozycji", "rosnącą zależność", "ryzyko", "szansa", "presję"
    ]):
        score += 1
        reasons.append("business_mechanism")

    if any(w in tl for w in [
        "rdzeń portfela", "obrona lidera", "realok", "alok", "priorytet", "portfel",
        "availability", "ekspozycj", "zapas", "kalendarz", "aktywac", "challenger",
        "dywersyfik", "koncentrac", "strukturaln", "przesunięcie popytu",
        "selektywna aktywacja", "szczytowych miesiącach", "okresach szczytu",
        "okresach szczytowych", "kanałów sprzedaży", "przychodu w okresach szczytu"
    ]):
        score += 1
        reasons.append("strategic_implication")

    if any(w in tl for w in [
        "decyzja:", "dlatego:", "zwiększ", "ogranicz", "utrzymaj", "przenieś",
        "wzmocnij", "skaluj", "zaktualizuj", "sprawdź", "alokuj", "wstrzymaj"
    ]):
        score += 1
        reasons.append("decision_or_recommendation")

    if any(p in tl for p in [
        "zwiększyć inwestycje w marketing",
        "monitorować trendy",
        "obserwować kategorię",
        "należy monitorować",
        "należy przeanalizować",
        "warto rozważyć"
    ]):
        penalties.append("generic_action")
        score = max(0, score - 1)

    if any(p in tl for p in [
        "praktycznie remis",
        "gap udziału",
        "metryka sugeruje",
        "amplituda powinna decydować"
    ]):
        penalties.append("technical_engine_language")
        score = max(0, score - 1)

    if re.search("\\bo\\s*\u2014(?=[,\\s])", t):
        penalties.append("placeholder_numeric")
        score = max(0, score - 2)

    if mode in ("shares_pp", "share", "mix") and ("%" not in t and "pp" not in tl):
        penalties.append("missing_share_anchor")
        score = max(0, score - 1)

    block_bonus = 0
    if block_id == "cot__mix_share_topN" and any(w in tl for w in ["rdzeń portfela", "obrona lidera", "erozj", "fragmentac"]):
        block_bonus = 1
    elif block_id == "cot__winners_losers" and any(w in tl for w in ["transfer udziału", "przejęcie popytu", "driver wzrostu", "realok"]):
        block_bonus = 1
    elif block_id == "cot__seasonality" and any(w in tl for w in ["kalendarz", "zapas", "ekspozycj", "szczyt sezonu", "błąd planowania"]):
        block_bonus = 1
    elif block_id == "cot__start_end" and any(w in tl for w in ["trwał", "strukturaln", "nowy układ", "przesunięcie popytu"]):
        block_bonus = 1
    elif block_id == "cot__concentration" and any(w in tl for w in ["zależność od rdzenia", "odporność portfela", "challenger", "koncentrac", "dywersyfik"]):
        block_bonus = 1

    if (not block_bonus) and block_id == "cot__mix_share_topN" and any(
        token in tn
        for token in [
            "traci pozycje w miksie",
            "traci pozycje lidera",
            "utrata pozycji lidera",
            "top-n spadl",
            "spadek top-n",
            "spadek top n",
            "obrony lidera",
        ]
    ):
        block_bonus = 1

    if (not block_bonus) and block_id == "cot__concentration" and _exec_has_concentration_exec_signal(t):
        block_bonus = 1

    score += block_bonus
    if block_bonus:
        reasons.append("block_specific_executive_language")

    has_mechanism = any(w in tl for w in [
        "oznacza",
        "wskazuje na",
        "sygnalizuje",
        "implikuje",
        "oddawanie udziału",
        "trwałą zmianę struktury",
        "transfer udziału",
        "przejęcie popytu",
        "erozję rdzenia",
        "osłabienie pozycji",
        "rosnącą zależność",
        "zależność od rdzenia",
        "bardziej zależna od rdzenia",
        "rdzeń portfela",
        "ryzyko",
        "presję",
        "koszt błędu planowania",
        "błąd planowania",
        "trwałe przesunięcie",
        "ryzyko niedostępności",
        "priorytet zatowarowania",
        "priorytetowego zatowarowania",
        "wymaga priorytetowego zatowarowania",
        "przed szczytem sezonu"
    ])
    if (not has_mechanism) and block_id == "cot__mix_share_topN":
        has_mechanism = any(
            token in tn
            for token in [
                "traci pozycje w miksie",
                "traci pozycje lidera",
                "utrata pozycji lidera",
                "top-n spadl",
                "spadek top-n",
                "spadek top n",
                "obrony lidera",
            ]
        )
        if has_mechanism and "business_mechanism" not in reasons:
            score += 1
            reasons.append("business_mechanism")

    has_action = any(w in tl for w in [
        "decyzja:", "dlatego:", "zwiększ", "ogranicz", "utrzymaj", "przenieś",
        "wzmocnij", "skaluj", "zaktualizuj", "sprawdź", "alokuj", "wstrzymaj",
        "availability", "ekspozycj", "zapas", "kalendarz aktywacji"
    ])
    WHY_NOW_TOKENS = [
        "ryzyko utraty udziału",
        "ryzyka utraty udziału",
        "z powodu ryzyka utraty udziału",
        "aby przeciwdziałać ryzyku utraty udziału",
        "aby zminimalizować ryzyko utraty udziału",
        "aby uniknąć ryzyka utraty udziału",
        "presja na obronę lidera",
        "presję na obronę lidera",
        "okno przejęcia popytu",
        "okna przejęcia popytu",
        "trwałe przesunięcie popytu",
        "koszt zaniechania",
        "koszt błędu planowania"
    ]

    has_why_now = any(w in tl for w in WHY_NOW_TOKENS)

    if has_mechanism and has_action and has_why_now:
        reasons.append("strategic_implication")

    if re.search(r"\d", t) and not has_mechanism and not has_action:
        penalties.append("descriptive_only")
        score = max(0, score - 1)

    if re.search(r"\d", t) and has_mechanism and not has_why_now and not has_action:
        penalties.append("descriptive_with_no_why_now")
        score = max(0, score - 1)

    if has_action and not has_why_now:
        penalties.append("no_tradeoff_or_why_now")
        score = max(0, score - 2)

    if not t:
        penalties.append("empty")
        score = 0

    if block_id == "cot__concentration":
        if any(w in tl for w in [
            "challenger", "challengerów", "challengery",
            "odporność portfela",
            "zależność od rdzenia",
            "ograniczenia zależności od rdzenia",
            "ograniczyć zależność od rdzenia",
            "dywersyfikacja portfela",
            "wzmocnienie challengerów",
            "realokacja wsparcia dla challengerów",
            "realokacja wsparcia na challengerów",
        ]):
            score += 1
            reasons.append("block_specific_exec_direction")

        if _exec_has_concentration_exec_signal(t) and "block_specific_exec_direction" not in reasons:
            score += 1
            reasons.append("block_specific_exec_direction")

        if "realokacja wsparcia" in tl and not any(w in tl for w in [
            "challenger", "challengerów", "obrona lidera",
            "odporność portfela", "zależność od rdzenia", "dywersyfik"
        ]):
            score = max(0, score - 1)
            penalties.append("generic_decision_direction")

        if _exec_has_concentration_exec_signal(t) and "generic_decision_direction" in penalties:
            penalties = [p for p in penalties if p != "generic_decision_direction"]
            score = min(5, score + 1)

    if block_id == "cot__winners_losers" and _exec_has_winners_losers_exec_signal(t):
        if "block_specific_exec_direction" not in reasons:
            score += 1
            reasons.append("block_specific_exec_direction")

    if block_id == "cot__start_end" and _exec_has_start_end_exec_signal(t):
        if "block_specific_exec_direction" not in reasons:
            score += 1
            reasons.append("block_specific_exec_direction")

    return {"score": max(0, min(5, score)), "reasons": reasons, "penalties": penalties}


def _quick_narrative_score_v2(text: str, block_id: str, mode: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    t = " ".join(str(text or "").strip().split())
    tl = t.lower()
    tn = _norm_pl_token(t)
    stats = stats if isinstance(stats, dict) else {}

    score = 0
    reasons: List[str] = []
    penalties: List[str] = []

    if re.search(r"\d", t):
        score += 1
        reasons.append("numerical_evidence")

    names: List[str] = []
    for k, v in stats.items():
        kl = str(k).lower()
        if any(tok in kl for tok in ["category", "leader", "winner", "loser", "driver"]) and isinstance(v, str):
            vv = v.strip()
            if vv and vv.lower() not in {"other", "unknown", "kategorii", "lider", "winner", "loser"}:
                names.append(vv)

    if any(n.lower() in tl for n in names) or any(q in t for q in ['"', '„', '”']):
        score += 1
        reasons.append("category_or_driver_reference")

    has_mechanism = _exec_has_named_mechanism_signal(t, block_id=block_id) or any(
        w in tl for w in ["oznacza", "wskazuje na", "sygnalizuje", "implikuje", "ryzyko", "szansa", "presję"]
    )
    if has_mechanism:
        score += 1
        reasons.append("business_mechanism")

    if any(w in tl for w in [
        "rdzeń portfela", "obrona lidera", "realok", "alok", "priorytet", "portfel",
        "availability", "ekspozycj", "zapas", "kalendarz", "aktywac", "challenger",
        "dywersyfik", "koncentrac", "strukturaln", "przesunięcie popytu",
        "selektywna aktywacja", "szczytowych miesiącach", "okresach szczytu",
        "okresach szczytowych", "kanałów sprzedaży", "przychodu w okresach szczytu"
    ]) or _exec_has_business_lever_signal(t):
        score += 1
        reasons.append("strategic_implication")

    has_action = _exec_has_decision_signal(t)
    if has_action:
        score += 1
        reasons.append("decision_or_recommendation")

    if any(p in tl for p in [
        "zwiększyć inwestycje w marketing",
        "monitorować trendy",
        "obserwować kategorię",
        "należy monitorować",
        "należy przeanalizować",
        "warto rozważyć"
    ]):
        penalties.append("generic_action")
        score = max(0, score - 1)

    if any(p in tl for p in [
        "praktycznie remis",
        "gap udziału",
        "metryka sugeruje",
        "amplituda powinna decydować"
    ]):
        penalties.append("technical_engine_language")
        score = max(0, score - 1)

    if re.search("\\bo\\s*\u2014(?=[,\\s])", t):
        penalties.append("placeholder_numeric")
        score = max(0, score - 2)

    if mode in ("shares_pp", "share", "mix") and ("%" not in t and "pp" not in tl):
        penalties.append("missing_share_anchor")
        score = max(0, score - 1)

    block_bonus = 0
    if block_id == "cot__mix_share_topN" and (
        any(w in tl for w in ["rdzeń portfela", "obrona lidera", "erozj", "fragmentac"])
        or _exec_has_named_mechanism_signal(t, block_id=block_id)
    ):
        block_bonus = 1
    elif block_id == "cot__winners_losers" and any(w in tl for w in ["transfer udziału", "przejęcie popytu", "driver wzrostu", "realok"]):
        block_bonus = 1
    elif block_id == "cot__winners_losers" and _exec_has_winners_losers_exec_signal(t):
        block_bonus = 1
    elif block_id == "cot__seasonality" and any(
        w in tl for w in ["kalendarz", "zapas", "ekspozycj", "szczyt sezonu", "błąd planowania"]
    ):
        block_bonus = 1
    elif block_id == "cot__start_end" and (
        any(w in tl for w in ["trwał", "strukturaln", "nowy układ", "przesunięcie popytu"])
        or ("realokacj wspar" in tn)
        or _exec_has_start_end_exec_signal(t)
    ):
        block_bonus = 1
    elif block_id == "cot__concentration" and any(
        w in tl for w in ["zależność od rdzenia", "odporność portfela", "challenger", "koncentrac", "dywersyfik"]
    ):
        block_bonus = 1

    score += block_bonus
    if block_bonus:
        reasons.append("block_specific_executive_language")

    has_why_now = bool(_detect_exec_why_now_tokens(t))

    if has_mechanism and has_action and has_why_now and "strategic_implication" not in reasons:
        reasons.append("strategic_implication")

    if re.search(r"\d", t) and not has_mechanism and not has_action:
        penalties.append("descriptive_only")
        score = max(0, score - 1)

    if re.search(r"\d", t) and has_mechanism and not has_why_now and not has_action:
        penalties.append("descriptive_with_no_why_now")
        score = max(0, score - 1)

    if has_action and not has_why_now:
        penalties.append("no_tradeoff_or_why_now")
        score = max(0, score - 2)

    if not t:
        penalties.append("empty")
        score = 0

    if block_id == "cot__concentration":
        if any(w in tl for w in [
            "challenger", "challengerów", "challengery",
            "odporność portfela",
            "zależność od rdzenia",
            "ograniczenia zależności od rdzenia",
            "ograniczyć zależność od rdzenia",
            "dywersyfikacja portfela",
            "wzmocnienie challengerów",
            "realokacja wsparcia dla challengerów",
            "realokacja wsparcia na challengerów",
        ]):
            score += 1
            reasons.append("block_specific_exec_direction")

        if "realokacja wsparcia" in tl and not any(w in tl for w in [
            "challenger", "challengerów", "obrona lidera",
            "odporność portfela", "zależność od rdzenia", "dywersyfik"
        ]):
            score = max(0, score - 1)
            penalties.append("generic_decision_direction")

    if block_id == "cot__concentration":
        if _exec_has_concentration_exec_signal(t) and "block_specific_exec_direction" not in reasons:
            score += 1
            reasons.append("block_specific_exec_direction")
        if _exec_has_concentration_exec_signal(t) and "generic_decision_direction" in penalties:
            penalties = [p for p in penalties if p != "generic_decision_direction"]
            score = min(5, score + 1)

    if block_id == "cot__winners_losers" and _exec_has_winners_losers_exec_signal(t):
        if "block_specific_exec_direction" not in reasons:
            score += 1
            reasons.append("block_specific_exec_direction")

    if block_id == "cot__start_end" and _exec_has_start_end_exec_signal(t):
        if "block_specific_exec_direction" not in reasons:
            score += 1
            reasons.append("block_specific_exec_direction")

    return {"score": max(0, min(5, score)), "reasons": reasons, "penalties": penalties}


_quick_narrative_score = _quick_narrative_score_v2


def _exec_selector_quality_bonus(text: str, block_id: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    t = " ".join(str(text or "").strip().split())
    tl = t.lower()
    tn = _norm_pl_token(t)
    stats = stats if isinstance(stats, dict) else {}
    profile = _build_exec_contract_profile(block_id, stats)
    contract_mode = str(profile.get("contract_mode") or "full")

    bonus = 0
    reasons: List[str] = []

    has_named_mechanism = any(
        token in tl
        for token in [
            "transfer udzia",
            "przejecie popytu",
            "oddawanie udzia",
            "erozj",
            "obrona lidera",
            "trwale przesuniecie popytu",
            "koszt bledu planowania",
            "zaleznosc od rdzenia",
            "odpornosci portfela",
        ]
    )
    if (not has_named_mechanism) and block_id == "cot__mix_share_topN":
        has_named_mechanism = any(
            token in tn
            for token in [
                "traci pozycje w miksie",
                "traci pozycje lidera",
                "utrata pozycji lidera",
                "top-n spadl",
                "spadek top-n",
                "spadek top n",
                "obrony lidera",
            ]
        )
    if has_named_mechanism:
        bonus += 1
        reasons.append("named_mechanism")

    if _detect_exec_why_now_tokens(t):
        bonus += 1
        reasons.append("explicit_why_now")

    has_business_lever = any(
        token in tl
        for token in [
            "availability",
            "ekspozyc",
            "zapas",
            "zatowarowanie",
            "cena",
            "price-pack",
            "price pack",
            "plan wsparcia",
            "realokacja wsparcia",
            "challenger",
            "rola kategorii",
            "rola asortymentowa",
            "kalendarz aktywacji",
        ]
    )
    if has_business_lever:
        bonus += 1
        reasons.append("explicit_business_lever")

    has_horizon_or_kpi = bool(
        re.search(r"\b\d+\s*[-–]\s*\d+\s*(?:cykl|cykle|miesiac|miesiace|miesiecy)\b", tl)
        or ("kpi" in tl)
        or ("pp" in tl)
    )
    if has_horizon_or_kpi:
        bonus += 1
        reasons.append("horizon_or_kpi")

    if block_id == "cot__mix_share_topN":
        topn = int(_first_not_none(stats.get("topN"), 10) or 10)
        if any(token in tl for token in [f"top-{topn}", f"top {topn}", "top-n", "top n"]):
            bonus += 1
            reasons.append("mix_dual_anchor")

    if block_id == "cot__winners_losers" and _exec_has_winners_losers_exec_signal(t):
        bonus += 1
        reasons.append("wl_dual_lever")

    if contract_mode in {"reduced", "sparse"} and any(
        token in tl
        for token in [
            "sygnal jest ograniczony",
            "ostroznie",
            "bez eskalacji budzetu",
            "dopiero po potwierdzeniu",
            "po potwierdzeniu sygnalu",
            "najpierw sprawdz",
            "na probe",
            "pilota",
        ]
    ):
        bonus += 1
        reasons.append("appropriate_caution_for_sparse_data")

    return {"bonus": bonus, "reasons": reasons}


def _legacy_exec_selector_quality_bonus_v2(text: str, block_id: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    t = " ".join(str(text or "").strip().split())
    tl = t.lower()
    stats = stats if isinstance(stats, dict) else {}
    profile = _build_exec_contract_profile(block_id, stats)
    contract_mode = str(profile.get("contract_mode") or "full")

    bonus = 0
    reasons: List[str] = []

    if _exec_has_named_mechanism_signal(t, block_id=block_id):
        bonus += 1
        reasons.append("named_mechanism")

    if _detect_exec_why_now_tokens(t):
        bonus += 1
        reasons.append("explicit_why_now")

    if _exec_has_business_lever_signal(t):
        bonus += 1
        reasons.append("explicit_business_lever")

    if re.search(r"\b\d+\s*[-–]\s*\d+\s*(?:cykl|cykle|miesiac|miesiace|miesiecy)\b", tl) or ("kpi" in tl) or ("pp" in tl):
        bonus += 1
        reasons.append("horizon_or_kpi")

    if block_id == "cot__mix_share_topN":
        topn = int(_first_not_none(stats.get("topN"), 10) or 10)
        if any(token in tl for token in [f"top-{topn}", f"top {topn}", "top-n", "top n"]):
            bonus += 1
            reasons.append("mix_dual_anchor")

    if block_id == "cot__start_end" and _exec_has_start_end_exec_signal(t):
        bonus += 1
        reasons.append("start_end_structural_signal")

    if contract_mode in {"reduced", "sparse"} and any(
        token in tl
        for token in [
            "sygnal jest ograniczony",
            "ostroznie",
            "bez eskalacji budzetu",
            "dopiero po potwierdzeniu",
            "po potwierdzeniu sygnalu",
            "najpierw sprawdz",
            "na probe",
            "pilota",
        ]
    ):
        bonus += 1
        reasons.append("appropriate_caution_for_sparse_data")

    return {"bonus": bonus, "reasons": reasons}


_legacy_exec_selector_quality_bonus = _legacy_exec_selector_quality_bonus_v2


def _detect_exec_why_now_tokens(text: str):
    tn = _norm_pl_token(" ".join(str(text or "").strip().split()))

    token_map = {
        "ryzyko utraty udziału": [
            "ryzyko utraty udzialu",
            "ryzyka utraty udzialu",
            "z powodu ryzyka utraty udzialu",
            "z uwagi na ryzyko utraty udzialu",
            "aby przeciwdzialac ryzyku utraty udzialu",
            "aby zminimalizowac ryzyko utraty udzialu",
            "aby uniknac ryzyka utraty udzialu",
        ],
        "koszt utraty udziału": [
            "koszt utraty udzialu",
            "kosztu utraty udzialu",
            "z powodu kosztu utraty udzialu",
            "z uwagi na koszt utraty udzialu",
            "aby ograniczyc koszt utraty udzialu",
        ],
        "presja na obronę lidera": [
            "presja na obrone lidera",
            "presje na obrone lidera",
        ],
        "okno przejęcia popytu": [
            "okno przejecia popytu",
            "okna przejecia popytu",
            "aby wykorzystac okno przejecia popytu",
            "z powodu okna przejecia popytu",
        ],
        "trwałe przesunięcie popytu": [
            "trwale przesuniecie popytu",
        ],
        "koszt zaniechania": [
            "koszt zaniechania",
        ],
        "koszt błędu planowania": [
            "koszt bledu planowania",
        ],
        "ryzyko niedostępności": [
            "ryzyko niedostepnosci",
            "ryzykiem niedostepnosci",
        ],
        "ryzyko niedoszacowania popytu": [
            "ryzyko niedoszacowania popytu",
            "ryzykiem niedoszacowania popytu",
            "niedoszacowania popytu",
        ],
    }

    matched: List[str] = []
    for label, variants in token_map.items():
        if any(v in tn for v in variants):
            matched.append(label)
    return matched


def _exec_has_winners_losers_exec_signal(text: str) -> bool:
    t = " ".join(str(text or "").strip().split())
    tn = _norm_pl_token(t)

    has_mechanism = any(
        token in tn
        for token in [
            "transfer udzial",
            "przejecie popytu",
            "okno przejecia popytu",
        ]
    )
    has_winner_lever = any(
        token in tn
        for token in [
            "availability",
            "dostepn",
            "ekspozyc",
            "promo",
            "promoc",
            "wsparcie promocyjne",
            "zwieksz ekspozyc",
            "zwieksz availability",
            "wzmocnij ekspozyc",
        ]
    )
    has_loser_lever_or_diagnosis = any(
        token in tn
        for token in [
            "cena",
            "price pack",
            "price-pack",
            "sezonow",
            "ograniczenie wsparcia",
            "ogranicz wspar",
            "ograniczyc wspar",
            "ogranicz cene",
            "ograniczyc cene",
            "rola kategor",
            "rola asortyment",
            "obecnosc na polce",
            "sprawdz cen",
            "sprawdz ekspozyc",
            "rewizj strategii cen",
        ]
    )

    return has_mechanism and has_winner_lever and has_loser_lever_or_diagnosis


def _exec_selector_quality_bonus_v2(text: str, block_id: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    t = " ".join(str(text or "").strip().split())
    tl = t.lower()
    stats = stats if isinstance(stats, dict) else {}
    profile = _build_exec_contract_profile(block_id, stats)
    contract_mode = str(profile.get("contract_mode") or "full")

    bonus = 0
    reasons: List[str] = []

    if _exec_has_named_mechanism_signal(t, block_id=block_id):
        bonus += 1
        reasons.append("named_mechanism")

    if _detect_exec_why_now_tokens(t):
        bonus += 1
        reasons.append("explicit_why_now")

    if _exec_has_business_lever_signal(t):
        bonus += 1
        reasons.append("explicit_business_lever")

    if re.search(r"\b\d+\s*[-–]\s*\d+\s*(?:cykl|cykle|miesiac|miesiace|miesiecy)\b", _norm_pl_token(tl)) or ("kpi" in tl) or ("pp" in tl):
        bonus += 1
        reasons.append("horizon_or_kpi")

    if block_id == "cot__mix_share_topN":
        topn = int(_first_not_none(stats.get("topN"), 10) or 10)
        if any(token in tl for token in [f"top-{topn}", f"top {topn}", "top-n", "top n"]):
            bonus += 1
            reasons.append("mix_dual_anchor")

    if block_id == "cot__start_end" and _exec_has_start_end_exec_signal(t):
        bonus += 1
        reasons.append("start_end_structural_signal")

    if block_id == "cot__winners_losers" and _exec_has_winners_losers_exec_signal(t):
        bonus += 1
        reasons.append("wl_dual_lever")

    if contract_mode in {"reduced", "sparse"} and any(
        token in tl
        for token in [
            "sygnal jest ograniczony",
            "ostroznie",
            "bez eskalacji budzetu",
            "dopiero po potwierdzeniu",
            "po potwierdzeniu sygnalu",
            "najpierw sprawdz",
            "na probe",
            "pilota",
        ]
    ):
        bonus += 1
        reasons.append("appropriate_caution_for_sparse_data")

    return {"bonus": bonus, "reasons": reasons}


_exec_selector_quality_bonus = _exec_selector_quality_bonus_v2


def _select_best_exec_takeaway(
    *,
    block_id: str,
    mode: str,
    stats: Dict[str, Any],
    llm_text: str,
    llm_valid: bool,
    llm_src: str,
    det_text: str,
) -> Dict[str, Any]:
    det_eval = _quick_narrative_score(det_text, block_id=block_id, mode=mode, stats=stats)

    def _exec_superiority_flags(txt: str, block_id: str) -> int:
        tl = str(txt or "").lower()
        flags = 0

        if _detect_exec_why_now_tokens(txt):
            flags += 1

        if any(w in tl for w in [
            "1–2", "1-2", "miesiąc", "miesiące", "cykl", "cykle", "kpi", "%", "pp"
        ]):
            flags += 1

        block_words = []
        if block_id == "cot__mix_share_topN":
            block_words = ["erozja rdzenia", "obrona lidera", "presja na obronę lidera"]
        elif block_id == "cot__winners_losers":
            block_words = ["transfer udziału", "przejęcie popytu", "okno przejęcia popytu"]
        elif block_id == "cot__seasonality":
            block_words = ["błąd planowania", "szczyt sezonu", "koszt błędu planowania"]
        elif block_id == "cot__concentration":
            block_words = ["zależność od rdzenia", "challenger", "odporność portfela"]
        elif block_id == "cot__start_end":
            block_words = ["trwałe przesunięcie", "nowy układ kategorii", "trwałe przesunięcie popytu"]

        if any(w in tl for w in block_words):
            flags += 1

        return flags

    if not llm_valid or not str(llm_text or "").strip():
        return {
            "text": str(det_text or ""),
            "src": "deterministic_last_resort",
            "score_llm": 0,
            "score_det": det_eval["score"],
            "selected_candidate": "deterministic_last_resort",
            "selection_reason": "llm_validation_failed",
            "llm_eval": {"score": 0, "reasons": [], "penalties": ["validation_failed"]},
            "det_eval": det_eval,
        }

    llm_eval = _quick_narrative_score(llm_text, block_id=block_id, mode=mode, stats=stats)
    score_llm = int(llm_eval.get("score") or 0)
    score_det = int(det_eval.get("score") or 0)
    llm_ok = bool(llm_valid and str(llm_text or "").strip())
    det_ok = bool(str(det_text or "").strip())

    if score_llm > score_det:
        return {
            "text": str(llm_text or ""),
            "src": str(llm_src or "llm"),
            "score_llm": score_llm,
            "score_det": score_det,
            "selected_candidate": "llm",
            "selection_reason": "llm_better",
            "llm_eval": llm_eval,
            "det_eval": det_eval,
        }

    if score_llm == score_det and llm_ok and det_ok:
        llm_flags = _exec_superiority_flags(llm_text, block_id)
        det_flags = _exec_superiority_flags(det_text, block_id)
        if block_id == "cot__mix_share_topN" and llm_valid:
            dbg_cp(
                "exec_takeaway.tie_break_audit",
                block_id=block_id,
                llm_flags=llm_flags,
                det_flags=det_flags,
                score_llm=score_llm,
                score_det=score_det,
                llm_text=llm_text,
                det_text=det_text,
                selection_reason="llm_better_executive_on_tie",
            )
            return {
                "text": str(llm_text or ""),
                "src": str(llm_src or "llm"),
                "score_llm": score_llm,
                "score_det": score_det,
                "selected_candidate": "llm",
                "selection_reason": "llm_better_executive_on_tie",
                "llm_eval": llm_eval,
                "det_eval": det_eval,
            }
        if llm_flags >= 2 and llm_flags > det_flags:
            dbg_cp(
                "exec_takeaway.tie_break_audit",
                block_id=block_id,
                llm_flags=llm_flags,
                det_flags=det_flags,
                score_llm=score_llm,
                score_det=score_det,
                llm_text=llm_text,
                det_text=det_text,
                selection_reason="llm_better_executive_on_tie",
            )
            return {
                "text": str(llm_text or ""),
                "src": str(llm_src or "llm"),
                "score_llm": score_llm,
                "score_det": score_det,
                "selected_candidate": "llm",
                "selection_reason": "llm_better_executive_on_tie",
                "llm_eval": llm_eval,
                "det_eval": det_eval,
            }

    return {
        "text": str(det_text or ""),
        "src": "deterministic_last_resort",
        "score_llm": score_llm,
        "score_det": score_det,
        "selected_candidate": "deterministic_last_resort",
        "selection_reason": "deterministic_safer",
        "llm_eval": llm_eval,
        "det_eval": det_eval,
    }

def _cot_build_one_sentence_strict(mode: str, stats: Dict[str, Any]) -> str:
    """Hard one-sentence template (deterministic).

    This is the last-resort output used by strict fallbacks.
    """
    mode = (mode or "").strip().lower()
    stats = stats or {}

    if mode in ("value", "sprzedaz", "sales"):
        total_v = stats.get("total_value")
        peak_v = stats.get("peak_value")
        peak_m_raw = stats.get("peak_month") or stats.get("peak_month_label")
        peak_m = _format_month_pl(str(peak_m_raw)) if peak_m_raw else "okresie szczytu"
        return (
            f"Całkowita wartość sprzedaży wyniosła {_cot_fmt_pln(total_v)}, a szczyt osiągnęła {_cot_fmt_pln(peak_v)} {peak_m}, "
            f"co oznacza, że istotna część wyniku koncentruje się w krótkim oknie popytowym; dlatego priorytetem powinno być "
            f"zabezpieczenie dostępności, ekspozycji i wsparcia handlowego przed miesiącami szczytowymi, zamiast równomiernej alokacji budżetu w całym okresie."
        )

    if mode in ("quantity", "qty", "ilosc", "volume"):
        total_q = stats.get("total_qty")
        peak_q = stats.get("peak_qty")
        peak_m_raw = stats.get("peak_month") or stats.get("peak_month_label")
        peak_m = _format_month_pl(str(peak_m_raw)) if peak_m_raw else ""
        return (
            f"KPI: wolumen (szt.) — Całkowity wolumen wyniósł {_cot_fmt_int(total_q)}, a szczyt {_cot_fmt_int(peak_q)} {peak_m}; "
            f"dlatego zwiększ dostępność/ekspozycję w miesiącu szczytowym i monitoruj wolumen m/m (KPI: wolumen, szt.)."
        )

    # shares (% / pp)
    primary_cat = stats.get("primary_category") or stats.get("leader_category") or "(brak)"
    start_pct = stats.get("primary_start_pct")
    end_pct = stats.get("primary_end_pct")
    delta_pp = stats.get("primary_delta_pp")
    topN_end = stats.get("topN_end_pct")
    topN_delta = stats.get("topN_delta_pp")

    return (
        f"KPI: udział (%) — \"{primary_cat}\" zmienił udział o {_cot_fmt_pp(delta_pp)} (z {_cot_fmt_pct(start_pct)} do {_cot_fmt_pct(end_pct)}), "
        f"a łączny udział Top 10 wyniósł {_cot_fmt_pct(topN_end)} ({_cot_fmt_pp(topN_delta)}); "
        f"dlatego przesuń 1,0 pp wsparcia do pozycji rosnących i monitoruj udział m/m (KPI: udział, %)."
    )

def _cot_build_interp_payload_strict(stats: Dict[str, Any], mode_hint: str = "auto") -> Dict[str, Any]:
    """Deterministic interpretation payload for strict fallback/postprocess.

    Ensures: hard one_sentence template + non-generic, consistent bullets.
    """
    stats = stats or {}
    inferred = _infer_interp_mode("", stats) if (mode_hint == "auto" or not mode_hint) else mode_hint
    inferred = (inferred or "").lower().strip()

    # shares
    if inferred in ("share", "shares_pp", "mix") or (stats.get("leader_share_start_pct") is not None) or (stats.get("driver_start_share") is not None):
        ns = dict(stats)
        ns.update(
            {
                "primary_category": _first_not_none(stats.get("driver_category_name"), stats.get("primary_category"), stats.get("leader_category"), stats.get("top_category"), "Other"),
                "primary_start_pct": _share_to_pct(_first_not_none(stats.get("driver_start_pct"), stats.get("primary_start_pct"), stats.get("leader_share_start_pct"), stats.get("driver_start_share"))),
                "primary_end_pct": _share_to_pct(_first_not_none(stats.get("driver_end_pct"), stats.get("primary_end_pct"), stats.get("leader_share_end_pct"), stats.get("driver_end_share"))),
                "primary_delta_pp": _first_not_none(stats.get("driver_delta_pp"), stats.get("primary_delta_pp"), stats.get("leader_share_delta_pp"), stats.get("delta_pp")),
                "topN": int(_first_not_none(stats.get("topN"), stats.get("top_n"), 10)),
                "topN_end_pct": _share_to_pct(_first_not_none(stats.get("top10_end_pct"), stats.get("topN_end_pct"), stats.get("topN_share_end_pct"), stats.get("top10_end_share"), stats.get("topN_end_share"))),
                "topN_delta_pp": _first_not_none(stats.get("top10_delta_pp"), stats.get("topN_delta_pp"), stats.get("topN_share_delta_pp")),
                "leader_category": _first_not_none(stats.get("driver_category_name"), stats.get("primary_category"), stats.get("leader_category"), stats.get("top_category"), "Other"),
                "leader_start_share": _share_to_pct(_first_not_none(stats.get("driver_start_pct"), stats.get("primary_start_pct"), stats.get("leader_share_start_pct"), stats.get("driver_start_share"))),
                "leader_end_share": _share_to_pct(_first_not_none(stats.get("driver_end_pct"), stats.get("primary_end_pct"), stats.get("leader_share_end_pct"), stats.get("driver_end_share"))),
                "leader_delta_pp": _first_not_none(stats.get("driver_delta_pp"), stats.get("primary_delta_pp"), stats.get("leader_share_delta_pp"), stats.get("delta_pp")),
                "topN_end_share": _share_to_pct(_first_not_none(stats.get("top10_end_pct"), stats.get("topN_end_pct"), stats.get("topN_share_end_pct"), stats.get("top10_end_share"), stats.get("topN_end_share"))),
            }
        )
        leader = str(ns.get("leader_category") or "Other").strip()
        if _has_placeholder_text(leader):
            leader = "Other"
            ns["leader_category"] = leader
        one = _cot_build_one_sentence_strict("shares", ns)
        start = ns.get("leader_start_share")
        end = ns.get("leader_end_share")
        dpp = ns.get("leader_delta_pp")
        topN = int(ns.get("topN") or 10)
        topN_end = ns.get("topN_end_share")
        topN_dpp = ns.get("topN_delta_pp")
        return {
            "one_sentence": one,
            "what_chart_shows": [
                "Wykres pokazuje zmianę udziałów elementów wymiaru w czasie (w %), po zastosowaniu aktualnych filtrów i agregacji.",
                f'Pozycja odniesienia: "{leader}" (start {_cot_fmt_pct(start)}, koniec {_cot_fmt_pct(end)}, zmiana {_cot_fmt_pp(dpp)}).',
            ],
            "key_insights": [
                f'Łączny udział Top {topN} na końcu okresu to {_cot_fmt_pct(topN_end)} ({_cot_fmt_pp(topN_dpp)} vs start).',
                f'Pozycja "{leader}" zmieniła udział o {_cot_fmt_pp(dpp)} (z {_cot_fmt_pct(start)} do {_cot_fmt_pct(end)}).',
            ],
            "recommendations": [
                f'Przesuń 1,0 pp wsparcia (promo/ekspozycja) do rosnących pozycji; KPI: udział (%) dla "{leader}" ≥ {_cot_fmt_pct(end)}.',
                'Ustaw alert: jeśli udział pozycji zmieni się o >1,0 pp m/m, uruchom przegląd ceny/asortymentu.',
            ],
            "limitations": [
                "Wnioski dotyczą wyłącznie danych po aktualnych filtrach oraz przyjętej agregacji (np. miesięcznej).",
                "Porównania udziałów są wrażliwe na zmiany koszyka kategorii (np. łączenie/rozbijanie kategorii, Top-K+Other).",
            ],
        }

    # value/quantity
    simple = "quantity" if inferred in ("quantity", "qty", "ilosc") else "value"
    one = _cot_build_one_sentence_strict(simple, stats)
    peak_m_raw = stats.get("peak_month") or stats.get("peak_month_label")
    peak_m = _format_month_pl(str(peak_m_raw)) if peak_m_raw else ""
    if simple == "value":
        return {
            "one_sentence": one,
            "what_chart_shows": [
                "Wykres prezentuje wartości sprzedaży w czasie dla kategorii, agregowany zgodnie z wybranym krokiem czasu.",
            ],
            "key_insights": [
                f'Łączna sprzedaż w okresie to {_cot_fmt_pln(stats.get("total_value"))}.',
                f'Szczyt sprzedaży to {_cot_fmt_pln(stats.get("peak_value"))} w {peak_m}.',
            ],
            "recommendations": [
                f'Zwiększ alokację działań (media/promo) w {peak_m}; KPI: sprzedaż (PLN) m/m.',
                'Ustaw alert: jeśli sprzedaż m/m odchyli się o >10%, uruchom przegląd ceny, dostępności i promocji.',
            ],
            "limitations": [
                "Wnioski dotyczą wyłącznie danych po aktualnych filtrach i przyjętej agregacji.",
            ],
        }

    tq = stats.get("total_qty")
    pq = stats.get("peak_qty")
    return {
        "one_sentence": one,
        "what_chart_shows": [
            "Wykres prezentuje wolumen sprzedaży w czasie dla kategorii, agregowany zgodnie z wybranym krokiem czasu.",
        ],
        "key_insights": [
            f'Łączny wolumen w okresie to {int(tq) if tq is not None else "—"} szt.',
            f'Szczyt wolumenu to {int(pq) if pq is not None else "—"} szt. w {peak_m}.',
        ],
        "recommendations": [
            f'Zwiększ dostępność/ekspozycję w {peak_m}; KPI: wolumen (szt.) m/m.',
            'Ustaw alert: jeśli wolumen m/m odchyli się o >10%, uruchom przegląd dostępności i promocji.',
        ],
        "limitations": [
            "Wnioski dotyczą wyłącznie danych po aktualnych filtrach i przyjętej agregacji.",
        ],
    }

def _extract_primary_category_from_text(s: str) -> Optional[str]:
    if not s:
        return None
    # Prefer explicit category phrasing, but accept multiple quote styles and both singular/plural forms.
    patterns = [
        r'kategori(?:a|i|ę|ą)?\s+[„"]([^"”„]+)[”"]',
        r'dla\s+[„"]([^"”„]+)[”"]',
        r'udział\s+w\s+rynku\s+[„"]([^"”„]+)[”"]',
        r"kategori(?:a|i|ę|ą)?\s+'([^']+)'",
        r"dla\s+'([^']+)'",
        r"udział\s+w\s+rynku\s+'([^']+)'",
        r'[„"]([^"”„]+)[”"]',
        r"'([^']+)'",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            val = m.group(1).strip()
            if val:
                return val
    return None

def _narrative_consistency_check(payload: Dict[str, Any]) -> List[str]:
    """Ensure one_sentence, key_insights and recommendations tell the same story.

    Controlled patch: keep this check lightweight.
    - For shares/pp we no longer require exact duplication of primary category / pp delta / Top 10 in key_insights.
    - For value we keep only a soft peak-month consistency check.
    """
    reasons: List[str] = []
    os = str((payload or {}).get("one_sentence") or "")
    expect_pct = bool((payload or {}).get("expect_pct"))

    if not (payload or {}).get("expect_pct"):
        q = str((payload or {}).get("question") or "").lower()
        if ("udzia" in q) or ("share" in q) or ("%" in os) or (" pp" in os) or ("pp" in os):
            expect_pct = True

    ki = (payload or {}).get("key_insights") or []
    ki_blob = " ".join(map(str, ki)).lower()

    if expect_pct:
        if not re.search(r'([+\-]?\d+(?:[\.,]\d+)?)\s*pp', os, flags=re.I):
            reasons.append("consistency: one_sentence missing pp delta (cannot verify)")
    else:
        m = re.search(r'szczyt.*?w\s+([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)', os, flags=re.I)
        if m:
            month_pl = m.group(1).lower()
            pl_to_mm = {
                "styczniu": "01","lutym": "02","marcu": "03","kwietniu": "04","maju": "05","czerwcu": "06",
                "lipcu": "07","sierpniu": "08","wrześniu": "09","wrzesniu": "09",
                "październiku": "10","pazdzierniku": "10","listopadzie": "11","grudniu": "12",
            }
            mm = pl_to_mm.get(month_pl)
            has_pl = month_pl in ki_blob
            has_iso = bool(mm) and (f"-{mm}-" in ki_blob or f"-{mm}" in ki_blob)
            if not (has_pl or has_iso):
                reasons.append("consistency: key_insights missing peak month anchor from one_sentence")

    return reasons

def _value_one_sentence_has_peak_anchor(s: str) -> bool:
    """Accept natural peak wording for value-mode overview narratives.

    Requires a money anchor + a month/year anchor + a peak/max reference,
    but does not force one exact phrasing.
    """
    if not s:
        return False
    s = str(s)
    sl = s.lower()
    has_money = ("pln" in sl) or bool(re.search(r"\b\d+[\d\s]*(?:[\.,]\d+)?\s*(?:zł|zl|pln|tys\.?|mln)\b", s, flags=re.I))
    month_names = (
        "stycz", "lut", "mar", "kwi", "maj", "czerw", "lip", "sierp",
        "wrze", "paź", "paz", "listop", "grud",
        "jan", "feb", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    )
    has_month = any(m in sl for m in month_names) and bool(re.search(r"\b20\d{2}\b", s))
    has_iso_month = bool(re.search(r"\b20\d{2}-\d{2}\b", s))
    has_month_anchor = has_month or has_iso_month
    peak_patterns = [
        r"najwyższ\w*\s+sprzeda",
        r"wartoś\w*\s+szczytow",
        r"szczyt\w*\s+sprzeda",
        r"\bszczytem\b",
        r"maksimum",
        r"maksymaln\w*\s+sprzeda",
        r"maksymaln\w*\s+wynik",
        r"maksymaln\w*\s+wartoś",
        r"wartoś\w*\s+maksymaln",
        r"wynik\w*\s+szczytow",
        r"najwyższ\w*\s+wartoś",
        r"szczyt\w*\s+wartoś",
        r"najwyższ\w*\s+sprzeda\w+.*\bw\s+[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+",
        r"\bpeak\b",
        r"\bpeak value\b",
    ]
    has_peak_ref = any(re.search(p, sl, flags=re.I) for p in peak_patterns)
    return bool(has_money and has_month_anchor and has_peak_ref)

def _shares_one_sentence_is_strong(s: str) -> bool:
    if not s:
        return False
    s = str(s)
    return bool(
        re.search(r'["„][^"”]+["”]', s)
        and re.search(r'[+\-]?\d+(?:[\.,]\d+)?\s*pp', s, flags=re.I)
        and re.search(r'z\s*\(?\s*\d+(?:[\.,]\d+)?%\s*\)?\s*do\s*\(?\s*\d+(?:[\.,]\d+)?%\s*\)?', s, flags=re.I)
    )

def _validate_one_sentence_hard_v2(one_sentence: str, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Controlled validator: strict on placeholders, light on formatting.

    Safe gating patch:
    - value mode no longer requires literal KPI/KPI: labels;
    - shares_pp accepts quoted categories in multiple natural forms and does not require Top 10 in one_sentence.
    """
    reasons: List[str] = []
    s = (one_sentence or "").strip()
    sl = s.lower()

    if _has_placeholder_text(s):
        reasons.append("one_sentence: contains placeholder token")
    if len(s) < 40:
        reasons.append("one_sentence: too short")
    if "dlatego" not in sl:
        reasons.append("one_sentence: missing 'dlatego' causal link")

    expect_pct = bool(payload.get("expect_pct"))
    if expect_pct:
        if not re.search(r'["„][^"”]+["”]', s):
            reasons.append('one_sentence: missing concrete category in quotes (e.g., "Decorations")')
        if not re.search(r'[+\-]?\d+(?:[\.,]\d+)?\s*pp', s, flags=re.I):
            reasons.append("one_sentence: missing pp delta (e.g., -1.2 pp)")
        if not re.search(r'z\s*\(?\s*\d+(?:[\.,]\d+)?%\s*\)?\s*do\s*\(?\s*\d+(?:[\.,]\d+)?%\s*\)?', s, flags=re.I):
            reasons.append("one_sentence: missing start→end % pattern (z X% do Y%)")
    else:
        if "pln" not in sl and not re.search(r"\b\d+[\d\s]*(?:[\.,]\d+)?\s*(?:zł|zl|pln|tys\.?|mln)\b", s, flags=re.I):
            reasons.append("one_sentence: missing PLN currency")
        if not re.search(r'(całkowit\w+\s+sprzeda\w+|sprzeda\w+\s+wynios\w+|łączna\s+sprzeda\w+|total\s+sales).*\d', s, flags=re.I):
            reasons.append("one_sentence: missing total sales anchor")
        if not _value_one_sentence_has_peak_anchor(s):
            reasons.append("one_sentence: missing peak month/value anchor")

    return (len(reasons) == 0), reasons

def _validate_key_insights_v2(key_insights: Any) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not isinstance(key_insights, list):
        return False, ["key_insights: not a list"]
    items = [str(x).strip() for x in key_insights if str(x).strip()]
    # Hard guard: forbid exact duplicate bullets (after lower/strip)
    items_norm = [re.sub(r"\s+", " ", it.strip()).lower() for it in items]
    if len(set(items_norm)) < len(items_norm):
        reasons.append("key_insights: exact duplicate bullets")
    # Hard guard: forbid placeholder category tokens like 'kategoria-...'
    if any("kategoria-" in it.lower() for it in items):
        reasons.append("key_insights: placeholder category label")
    if len(items) < 2:
        return False, ["key_insights: need at least 2 bullets"]
    if len(items) > 4:
        reasons.append("key_insights: too many bullets (max 4)")

    # Prefer numeric bullets, but do not over-penalize structurally good executive insights.
    numeric_missing = 0
    for i, it in enumerate(items, 1):
        if not (_has_number(it) or _has_percent(it)):
            numeric_missing += 1
            reasons.append(f"key_insights[{i}]: missing numeric anchor")

    # Hard fail only when too many bullets are non-numeric.
    if numeric_missing >= 2 and len(items) >= 3:
        pass
    else:
        reasons = [r for r in reasons if not str(r).startswith("key_insights[")]

    # Dedup based on normalized anchors
    norm = [_normalize_anchor(it) for it in items]
    if len(set(norm)) < len(norm):
        reasons.append("key_insights: duplicate bullets detected")

    return (len(reasons) == 0), reasons

def _validate_interp_hard_v2(out: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    meta: Dict[str, Any] = {"hard_validator_v2": {"passed": True, "reasons": []}}
    reasons: List[str] = []

    ok1, r1 = _validate_one_sentence_hard_v2(str(out.get("one_sentence", "")), payload)
    if not ok1:
        reasons.extend(r1)

    ok2, r2 = _validate_key_insights_v2(out.get("key_insights", []))
    if not ok2:
        if bool(payload.get("expect_pct")):
            reasons.extend(r2)
        else:
            # Value-mode: nie fallbackuj tylko dlatego, że 1 bullet w key_insights
            # nie ma jawnej liczby, jeśli one_sentence już ma poprawne total + peak anchor.
            relaxed_r2 = [x for x in r2 if not str(x).startswith("key_insights[")]
            reasons.extend(relaxed_r2)

    # Recommendations must not be generic-only; enforce at least one KPI+number in recommendations block
    recs = out.get("recommendations", [])
    if isinstance(recs, list):
        rec_txt = " ".join(str(x) for x in recs)
    else:
        rec_txt = str(recs or "")
    if rec_txt.strip():
        _one_sentence = str(out.get("one_sentence", "") or "")
        _share_override = bool(payload.get("expect_pct")) and _shares_one_sentence_is_strong(_one_sentence)
        if (not _share_override) and _is_generic(rec_txt) and not (_has_number(rec_txt) and _has_kpi(rec_txt)):
            reasons.append("recommendations: too generic (needs KPI+number)")
    # Hard placeholder ban across the entire output (prevents 'kategoria-1' etc.)
    try:
        blob = json.dumps(out, ensure_ascii=False)
    except Exception:
        blob = str(out)
    if _has_placeholder_text(blob):
        reasons.append("hard: placeholder token detected in interpretation output")

    # Narrative Consistency Engine: one_sentence, key_insights, recommendations must tell the same story
    reasons.extend(_narrative_consistency_check({**out, **{"expect_pct": payload.get("expect_pct")}}))

    # Recommendations: no duplicates
    if isinstance(recs, list):
        recs_norm = [re.sub(r"\\s+", " ", str(x).strip().lower()) for x in recs if str(x).strip()]
        if len(recs_norm) != len(set(recs_norm)):
            reasons.append("recommendations: duplicate bullets detected")


    if reasons:
        meta["hard_validator_v2"]["passed"] = False
        meta["hard_validator_v2"]["reasons"] = reasons
        return False, meta

    return True, meta

def _validate_interp_semantic_v3(interp: Dict[str, Any], stats: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    """Semantic validator v3 for COT Interpretation.

    Enforces anchoring to *available* stats:
    - shares / pp (share deltas)
    - HHI (concentration)
    - seasonality metric
    Also blocks placeholder tokens like '%' or 'pp' without a number.
    Returns: (ok, topic, reasons)
    """
    reasons: List[str] = []

    # defensive
    if not isinstance(interp, dict):
        return False, "unknown", ["interp: not a dict"]

    one = str(interp.get("one_sentence", "") or "")
    insights = interp.get("key_insights", []) or []
    recs = interp.get("recommendations", []) or []
    wcs = interp.get("what_chart_shows", []) or []

    text_blob = " ".join([one] + [str(x) for x in insights] + [str(x) for x in recs] + [str(x) for x in wcs])

    # --- helper regexes
    has_percent_num = bool(re.search(r"\b\d+[\.,]?\d*\s*%", text_blob))
    has_pp_num = bool(re.search(r"\b\d+[\.,]?\d*\s*(pp|p\.p\.)\b|\bpunkt(?:ów)?\s+procent", text_blob, flags=re.IGNORECASE))
    has_hhi_num = bool(re.search(r"\bHHI\b", text_blob, flags=re.IGNORECASE) and re.search(r"\b\d+[\.,]?\d*\b", text_blob))
    has_season_num = bool(re.search(r"sezon", text_blob, flags=re.IGNORECASE) and re.search(r"\b\d+[\.,]?\d*\b", text_blob))

    # placeholders (no digits near %/pp)
    # Do not emit false positives when the text already contains a valid share pattern
    # such as: z 12,9% do 12,4% (-0,5 pp).
    has_share_span = bool(re.search(
        r"z\s*\d+[\.,]?\d*%\s*do\s*\d+[\.,]?\d*%.*?[\(\[]?[-+]?\d+[\.,]?\d*\s*(pp|p\.p\.)",
        text_blob,
        flags=re.IGNORECASE | re.DOTALL,
    ))
    if (not has_percent_num) and re.search(r"(^|\s)%($|\s)", text_blob):
        reasons.append("placeholder: standalone '%' token")
    # "pp" is valid only when clearly attached to a numeric delta (e.g., "6,3 pp").
    # Flag it only if we find a truly standalone token and there is no valid pp anchor.
    if not has_pp_num:
        for _m in re.finditer(r"\bpp\b", text_blob, flags=re.IGNORECASE):
            _left = text_blob[max(0, _m.start()-8):_m.start()]
            _right = text_blob[_m.end():min(len(text_blob), _m.end()+4)]
            if (not re.search(r"\d", _left)) and (not re.search(r"\d", _right)):
                reasons.append("placeholder: standalone 'pp' token")
                break
    if re.search(r"\bx%\b", text_blob, flags=re.IGNORECASE) and not has_share_span:
        reasons.append("placeholder: 'x%'")
    # Flag obvious "0%" placeholder if it appears without any other meaningful percentage.
    if re.search(r"\b0\s*%\b", text_blob) and not re.search(r"\b[1-9]\d*\s*%\b", text_blob):
        reasons.append("zero_percent_placeholder")

    if re.search(r"\b\d+\s*x\b", text_blob, flags=re.IGNORECASE):
        # often from templates
        reasons.append("placeholder: 'x' multiplier")
    if re.search(r"\b\d+\s*%\s*%\b", text_blob):
        reasons.append("format: duplicated percent")

    # --- detect available stats -> required anchors
    skeys = {str(k).lower() for k in (stats or {}).keys()}

    # --- defensive defaults (fix UnboundLocalError) ---
    wants_hhi = False
    wants_season = False
    wants_shares = False

    # Reject references to future years/quarters beyond the data range (avoid hallucinated 2024/Q1, etc.).
    _end_month = (stats or {}).get("end_month") or (stats or {}).get("EndMonth") or (stats or {}).get("END_MONTH")
    _max_year = None
    if isinstance(_end_month, str) and re.match(r"^\d{4}-\d{2}$", _end_month.strip()):
        try:
            _max_year = int(_end_month.strip()[:4])
        except Exception:
            _max_year = None
    if _max_year is not None:
        for _y in re.findall(r"\b(20\d{2})\b", text_blob):
            try:
                if int(_y) > _max_year:
                    reasons.append("v3:future_year_out_of_range")
                    break
            except Exception:
                pass

    wants_hhi = any("hhi" in k for k in skeys)
    wants_season = any("season" in k or "sezon" in k for k in skeys)
    wants_shares = any("share" in k or "udz" in k or "pp" in k for k in skeys)

    # Reject generic placeholder-like category labels only in category-driven topics.
    # Do not penalize generic/value narratives that do not need a quoted category.
    if (wants_shares or wants_hhi or wants_season):
        if re.search(r"\b(kategoria|kategorii|kategorię|kategorią)\s+[A-Z]\b", text_blob, flags=re.IGNORECASE) and not re.search(r'[\"„][^\"”]+[\"”]', text_blob):
            reasons.append("v3:placeholder: generic category label")

    # topic selection (for debug)
    if wants_season:
        topic = "seasonality"
    elif wants_hhi:
        topic = "hhi"
    elif wants_shares:
        topic = "shares_pp"
    else:
        topic = "generic"

    # require at least one of the available anchor families
    required_hits = 0
    if wants_shares:
        if not (has_percent_num or has_pp_num):
            reasons.append("v3: missing share anchor (% or pp) while stats include share/pp")
        else:
            required_hits += 1
    if wants_hhi:
        if not has_hhi_num:
            reasons.append("v3: missing HHI anchor while stats include HHI")
        else:
            required_hits += 1
    if wants_season:
        if not has_season_num:
            reasons.append("v3: missing seasonality anchor while stats include seasonality")
        else:
            required_hits += 1

    # If no specific stats: still require at least one % or pp OR at least 2 numbers total (to avoid generic)
    if not (wants_shares or wants_hhi or wants_season):
        nums = re.findall(r"\b\d+[\.,]?\d*\b", text_blob)
        if not (has_percent_num or has_pp_num or len(nums) >= 2):
            reasons.append("v3: too generic (no %/pp and <2 numbers)")

    # also: recommendations should not be generic verbs without anchors
    generic_verbs = ["zidentyfikuj", "skup", "dostosuj", "monitoruj", "zwiększ", "przenieś", "ustal", "przeanalizuj"]
    _shares_rec_gate_ok = _shares_one_sentence_is_strong(one)
    _value_mode = not (wants_shares or wants_hhi or wants_season)

    for r in recs[:3]:
        r_str = str(r).lower()

        if _shares_rec_gate_ok:
            continue

        if _value_mode:
            if any(v in r_str for v in generic_verbs):
                if not (
                    re.search(r"\b\d", r_str)
                    or any(tok in r_str for tok in [
                        "szczyt", "dołek", "peak", "low",
                        "miesiąc", "okres", "okno",
                        "koncentrację przychodu",
                        "krótkim oknie popytu",
                        "alokację wsparcia",
                        "ekspozycję bazową",
                        "plan wsparcia"
                    ])
                ):
                    reasons.append("v3: value recommendation too generic")
            continue

        if any(v in r_str for v in generic_verbs) and not re.search(r"\b\d", r_str):
            reasons.append("v3: recommendation missing numeric anchor")

    ok = (len(reasons) == 0)
    
    # Reject unsupported "uplift" forecasts (e.g., "zwiększy przychody o 15% w przyszłym roku") unless explicitly anchored.
    if re.search(r"(zwiększ\w*|podnie\w*|popraw\w*).{0,40}\b(\d{1,3}[\.,]?\d*)\s*%\b.{0,40}(przychod|przychody|sprzedaż|revenue)", text_blob, flags=re.IGNORECASE) and re.search(r"(przysz\w*|w\s+kolejn\w*|w\s+następn\w*)", text_blob, flags=re.IGNORECASE):
        reasons.append("v3:unsupported_forecast")

        # In seasonality contexts, forbid the outdated metric wording "siła/strength sezonowości" (use waga/Seasonality Share instead).
    if re.search(r"(siła\s+sezonowości|seasonality\s+strength|strength\s+sezonowości)", text_blob, flags=re.IGNORECASE):
        reasons.append("v3:wrong_seasonality_metric_wording")

    # Hard guard: forbid placeholder category tokens like "kategoria-..."
    if "kategoria-" in text_blob.lower():
        reasons.append("v3:placeholder: category placeholder id")

    # Anti-generic guard: if text uses generic filler phrases but lacks numeric anchors, reject.
    generic_phrases = [
        "warto monitorować", "należy monitorować", "warto analizować", "należy analizować",
        "wskazuje na potrzebę", "sugeruje potrzebę", "rekomenduje się", "zwiększyć alokację"
    ]
    if any(p in text_blob.lower() for p in generic_phrases):
        num_hits = len(re.findall(r"\d+[\d\s]*([\.,]\d+)?\s*(pp|%|pln)?", text_blob, flags=re.IGNORECASE))
        if num_hits < 2:
            reasons.append("v3:generic: low numeric density")

    return ok, topic, reasons

def _emit_interp_meta(
    out: Dict[str, Any],
    src_label: str,
    chart_title: str = "",
    *,
    ok_hard: bool,
    reasons_hard: List[str],
    ok_sem: bool,
    reasons_sem: List[str],
    repair_used: bool,
    repair_passed: Optional[bool] = None,
    fallback_used: bool,
    sem_topic: Optional[str] = None,
    topic_sem: Optional[str] = None,
    **_ignored: Any,
) -> Tuple[Dict[str, Any], str]:
    """Attach meta to interpretation payload and push to dc_llm_status_v1.

    Robust to signature drift: accepts extra keyword args and never raises.
    """
    sem_topic = (sem_topic if sem_topic is not None else topic_sem) or "unknown"
    status_key = f"exec:v4:composition_over_time_interpretation:{chart_title}"

    meta: Dict[str, Any] = {
        "stage": "validate",
        "passed": bool(ok_hard and ok_sem),
        "reasons": (list(reasons_hard or []) if not ok_hard else [])
        + (["v3:" + str(r) for r in (reasons_sem or [])] if not ok_sem else []),
        "repair_used": bool(repair_used),
        "repair_passed": repair_passed,
        "fallback_used": bool(fallback_used),
        "src": str(src_label or "unknown"),
        "hard_validator_v2": {"passed": bool(ok_hard), "reasons": list(reasons_hard or [])},
        "semantic_validator_v3": {"passed": bool(ok_sem), "topic": str(sem_topic), "reasons": list(reasons_sem or [])},
        "topic": str(sem_topic),
        "v2": bool(ok_hard),
        "v3": bool(ok_sem),
    }

    try:
        if isinstance(out, dict):
            out["__meta"] = meta
    except Exception:
        pass

    try:
        import streamlit as st  # type: ignore

        ss = getattr(st, "session_state", None)
        if ss is not None:
            bucket = ss.get("dc_llm_status_v1")
            if not isinstance(bucket, dict):
                bucket = {}
                ss["dc_llm_status_v1"] = bucket
            bucket[status_key] = meta
    except Exception:
        # Never crash rendering because of debug/meta wiring.
        pass

    return out, src_label


def _postprocess_interp_json(payload: dict, stats: dict, mode: str = "auto") -> dict:
    """Deterministic safety net to satisfy hard (v2) + semantic (v3) validators.
    Uses only provided stats; remains dataset-agnostic.
    """
    if not isinstance(payload, dict):
        return payload
    stats = stats or {}
    mode = (mode or "").strip().lower()

    def _first_sentence(s: str) -> str:
        if not isinstance(s, str):
            return ""
        parts = re.split(r"(?<=[\.!\?])\s+", s.strip(), maxsplit=1)
        return parts[0].strip()

    def _count_numbers(s: str) -> int:
        if not isinstance(s, str):
            return 0
        return len(re.findall(r"\d+(?:[\.,]\d+)?", s))

    def _has_pct_or_pp(s: str) -> bool:
        if not isinstance(s, str):
            return False
        return bool(re.search(r"(\bpp\b|punkt(?:u|ów)?\s+procentow|%)", s, flags=re.IGNORECASE))

    leader = stats.get("leader_category") or stats.get("top_category") or stats.get("top_category_name")
    if isinstance(leader, str):
        leader = leader.strip().strip('"').strip("'")
    leader = leader or None

    def _replace_placeholders(s: str) -> str:
        if not isinstance(s, str):
            return s
        # Remove placeholder labels like "kategoria A" (v3 placeholder gate). Prefer real leader_category if available.
        repl = f'kategoria "{leader}"' if leader else "największa kategoria"
        s = re.sub(r"\b(kategoria|category)\s+[A-Z]\b", repl, s)
        return s

    payload["one_sentence"] = _replace_placeholders(payload.get("one_sentence", ""))
    payload["recommendation"] = _replace_placeholders(payload.get("recommendation", ""))
    kis = payload.get("key_insights", [])
    if isinstance(kis, list):
        payload["key_insights"] = [_replace_placeholders(x) for x in kis]

    payload.setdefault("what_chart_shows", payload.get("what_chart_shows") or [])
    payload.setdefault("key_insights", payload.get("key_insights") or [])
    payload.setdefault("recommendation", payload.get("recommendation") or "")
    payload.setdefault("limitations", payload.get("limitations") or [])

    # ------------------------------------------------------------------
    # FIX 1 — hard template for one_sentence (deterministic)
    # ------------------------------------------------------------------
    try:
        inferred = _infer_interp_mode("", stats) if mode == "auto" else mode
        inferred = (inferred or "").lower().strip()
        if inferred in ("share", "shares_pp", "mix"):
            # normalize share keys expected by strict builder
            ns = dict(stats or {})
            ns.update(
                {
                    "primary_category": _first_not_none(stats.get("driver_category_name"), stats.get("primary_category"), stats.get("leader_category"), stats.get("top_category"), stats.get("top_category_name"), "Other"),
                    "primary_start_pct": _share_to_pct(_first_not_none(stats.get("driver_start_pct"), stats.get("primary_start_pct"), stats.get("leader_share_start_pct"), stats.get("leader_start_share"), stats.get("driver_start_share"))),
                    "primary_end_pct": _share_to_pct(_first_not_none(stats.get("driver_end_pct"), stats.get("primary_end_pct"), stats.get("leader_share_end_pct"), stats.get("leader_end_share"), stats.get("driver_end_share"))),
                    "primary_delta_pp": _first_not_none(stats.get("driver_delta_pp"), stats.get("primary_delta_pp"), stats.get("leader_share_delta_pp"), stats.get("leader_delta_pp"), stats.get("delta_pp")),
                    "leader_category": _first_not_none(stats.get("driver_category_name"), stats.get("primary_category"), stats.get("leader_category"), stats.get("top_category"), stats.get("top_category_name"), "Other"),
                    "leader_start_share": _share_to_pct(_first_not_none(stats.get("driver_start_pct"), stats.get("primary_start_pct"), stats.get("leader_share_start_pct"), stats.get("leader_start_share"), stats.get("driver_start_share"))),
                    "leader_end_share": _share_to_pct(_first_not_none(stats.get("driver_end_pct"), stats.get("primary_end_pct"), stats.get("leader_share_end_pct"), stats.get("leader_end_share"), stats.get("driver_end_share"))),
                    "leader_delta_pp": _first_not_none(stats.get("driver_delta_pp"), stats.get("primary_delta_pp"), stats.get("leader_share_delta_pp"), stats.get("leader_delta_pp"), stats.get("delta_pp")),
                    "topN": int(_first_not_none(stats.get("topN"), stats.get("top_n"), 10)),
                    "topN_end_pct": _share_to_pct(_first_not_none(stats.get("top10_end_pct"), stats.get("topN_end_pct"), stats.get("topN_share_end_pct"), stats.get("top10_end_share"), stats.get("topN_end_share"))),
                    "topN_end_share": _share_to_pct(_first_not_none(stats.get("top10_end_pct"), stats.get("topN_end_pct"), stats.get("topN_share_end_pct"), stats.get("top10_end_share"), stats.get("topN_end_share"))),
                    "topN_delta_pp": _first_not_none(stats.get("top10_delta_pp"), stats.get("topN_delta_pp"), stats.get("topN_share_delta_pp")),
                }
            )
            payload["one_sentence"] = _cot_build_one_sentence_strict("shares", ns)
        else:
            # value / quantity
            inferred_simple = "quantity" if inferred in ("quantity", "qty", "ilosc") else "value"
            payload["one_sentence"] = _cot_build_one_sentence_strict(inferred_simple, stats)
    except Exception:
        # do not block rendering
        pass

    # ------------------------------------------------------------------
    # FIX 2 — hard placeholder block (postprocess-level)
    # ------------------------------------------------------------------
    try:
        joined = "\n".join(
            [
                str(payload.get("one_sentence") or ""),
                str(payload.get("recommendation") or ""),
                "\n".join([str(x) for x in (payload.get("key_insights") or [])]),
                "\n".join([str(x) for x in (payload.get("what_chart_shows") or [])]),
            ]
        )
        if _has_placeholder_text(joined):
            # Force deterministic strict payload without placeholders.
            payload.update(_cot_build_interp_payload_strict(stats, mode_hint=mode))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Narrative Consistency Engine — ensure one story across sections
    # ------------------------------------------------------------------
    try:
        # De-duplicate bullets (exact) – requested hard rule.
        for k in ("what_chart_shows", "key_insights", "limitations"):
            if isinstance(payload.get(k), list):
                items = [str(x).strip() for x in payload.get(k) if str(x).strip()]
                payload[k] = list(dict.fromkeys(items))

        # If recommendation is empty or generic, rebuild from fallback (keeps same story as one_sentence).
        rec = str(payload.get("recommendation") or "").strip()
        if (not rec) or ("wybierz" in rec.lower() and "cel" in rec.lower()):
            fb = _cot_build_interp_payload_strict(stats, mode_hint=mode)
            payload.setdefault("recommendations", fb.get("recommendations", []))
            if not payload.get("recommendation"):
                payload["recommendation"] = (fb.get("recommendations") or [""])[0]
    except Exception:
        pass

    os = payload.get("one_sentence", "")
    fs = _first_sentence(os)

    if mode == "share":
        end_pct = stats.get("leader_share_end_pct")
        delta_pp = stats.get("leader_share_delta_pp")
        topn_end = stats.get("topN_share_end_pct")
        topn_delta = stats.get("topN_share_delta_pp")
        leader_phrase = f'kategoria "{leader}"' if leader else "lider kategorii"

        anchor = None
        if end_pct is not None and delta_pp is not None:
            anchor = f"{leader_phrase} ma {float(end_pct):.2f}% udziału (zmiana {float(delta_pp):.2f} pp)"
        elif topn_end is not None and topn_delta is not None:
            anchor = f"TOP‑N łącznie to {float(topn_end):.2f}% (zmiana {float(topn_delta):.2f} pp)"

        if anchor and (not _has_pct_or_pp(os) or _count_numbers(fs) < 2):
            payload["one_sentence"] = (os.strip() + ("" if os.strip().endswith(".") else ".") + f" {anchor}.").strip()
    else:
        total_value = stats.get("total_value")
        peak_value = stats.get("peak_value")
        peak_month = stats.get("peak_month")
        if (total_value is not None) and (peak_value is not None) and (_count_numbers(fs) < 2):
            pm = f" w {peak_month}" if peak_month else ""
            payload["one_sentence"] = (os.strip() + ("" if os.strip().endswith(".") else ".") +
                                      f" Suma to {float(total_value):,.0f}, a szczyt {float(peak_value):,.0f}{pm}.").strip()

    kis = payload.get("key_insights", [])
    if not isinstance(kis, list):
        kis = []
    while len(kis) < 4:
        kis.append("")
    kis = kis[:4]

    for i, s in enumerate(kis):
        if _count_numbers(str(s)) == 0:
            if mode == "share":
                if i == 0 and stats.get("topN_share_end_pct") is not None and stats.get("topN_share_delta_pp") is not None:
                    kis[i] = f"TOP‑N: {float(stats['topN_share_end_pct']):.2f}% (Δ {float(stats['topN_share_delta_pp']):.2f} pp)."
                elif stats.get("leader_share_end_pct") is not None and stats.get("leader_share_delta_pp") is not None:
                    lp = f'"{leader}"' if leader else "lider"
                    kis[i] = f"Udział {lp}: {float(stats['leader_share_end_pct']):.2f}% (Δ {float(stats['leader_share_delta_pp']):.2f} pp)."
                else:
                    kis[i] = "Zmienność udziałów różni się między elementami wymiaru (brak pełnych kotwic liczbowych w danych wejściowych)."
            else:
                if i == 0 and stats.get("peak_value") is not None and stats.get("peak_month"):
                    kis[i] = f"Szczyt sprzedaży: {float(stats['peak_value']):,.0f} w {stats['peak_month']}."
                elif stats.get("trough_value") is not None and stats.get("trough_month"):
                    kis[i] = f"Minimum sprzedaży: {float(stats['trough_value']):,.0f} w {stats['trough_month']}."
                elif stats.get("total_txn") is not None:
                    kis[i] = f"Liczba transakcji: {int(stats['total_txn']):,}."
                else:
                    kis[i] = "Występują różnice między elementami wymiaru w czasie (brak pełnych kotwic liczbowych w danych wejściowych)."

    payload["key_insights"] = kis

    rec = payload.get("recommendation", "")
    if _count_numbers(rec) == 0:
        if mode == "share" and stats.get("leader_share_delta_pp") is not None:
            payload["recommendation"] = (rec.strip() + ("" if rec.strip().endswith(".") else ".") +
                                       f" Ustal cel +{abs(float(stats['leader_share_delta_pp'])):.1f} pp w 90 dni i monitoruj weekly.").strip()
        elif mode != "share":
            payload["recommendation"] = (rec.strip() + ("" if rec.strip().endswith(".") else ".") +
                                       " Zaplanuj +10% capacity na miesiące szczytu i monitoruj odchylenia m/m.").strip()

    if not isinstance(payload.get("limitations"), list):
        payload["limitations"] = [str(payload.get("limitations"))] if payload.get("limitations") else []

    # --- McKinsey-style insight compression & ranking (deterministic post-process) ---
    def _norm_bullet(b: str) -> str:
        b = re.sub(r"\s+", " ", str(b or "").strip().lower())
        return re.sub(r"[^\w%\.\-]+", "", b)

    def _score_bullet(b: str) -> float:
        # Prefer pp deltas, then percentages, then large absolute numbers (PLN / counts)
        s = str(b or "")
        best = 0.0
        for m in re.finditer(r"(-?\d+(?:[\.,]\d+)?)\s*pp", s, flags=re.IGNORECASE):
            try: best = max(best, abs(float(m.group(1).replace(',', '.')))*100.0)
            except Exception: pass
        for m in re.finditer(r"(-?\d+(?:[\.,]\d+)?)\s*%", s):
            try: best = max(best, abs(float(m.group(1).replace(',', '.')))*10.0)
            except Exception: pass
        for m in re.finditer(r"(\d{1,3}(?:[\s,]\d{3})+|\d+)(?:[\.,]\d+)?", s):
            try:
                v = float(m.group(1).replace(" ", "").replace(",", ""))
                best = max(best, min(v/1000.0, 50.0))
            except Exception:
                pass
        return float(best)

    def _compress_list(items: Any, max_n: int) -> List[str]:
        if not isinstance(items, list):
            return []
        cleaned: List[str] = [str(x).strip() for x in items if str(x).strip()]
        # de-dupe (preserve the strongest signal)
        bucket: Dict[str, str] = {}
        for it in cleaned:
            k = _norm_bullet(it)
            if k not in bucket or _score_bullet(it) > _score_bullet(bucket[k]):
                bucket[k] = it
        ranked = sorted(bucket.values(), key=_score_bullet, reverse=True)
        # keep only the top N and trim very long lines (slide-like density)
        out: List[str] = []
        for it in ranked[:max_n]:
            it2 = it.split(" ; ")[0].split(";")[0].strip()
            if len(it2) > 160:
                it2 = it2[:157].rstrip() + "…"
            out.append(it2)
        return out

    payload["key_insights"] = _compress_list(payload.get("key_insights", []), max_n=4) or payload.get("key_insights", [])
    payload["limitations"] = _compress_list(payload.get("limitations", []), max_n=3) or payload.get("limitations", [])

    return payload

def _infer_interp_mode(chart_title: str, stats: dict) -> str:
    """Infer interpretation mode (value vs share/pp) in a dataset-agnostic way.

    Returns:
        "shares_pp" when the chart is about shares / pp changes,
        otherwise "value".
    """
    title = (chart_title or "").lower()
    # prefer explicit signals from title first (UI labels are stable)
    if any(k in title for k in ["udział", "udzial", "share", "pp", "punkt", "punkty procent"]):
        return "shares_pp"
    # fallback: inspect stats keys that typically exist for share-based charts
    keys = set((stats or {}).keys())
    if any(k for k in keys if "pp" in str(k).lower() or "share" in str(k).lower() or "udzial" in str(k).lower()):
        return "shares_pp"
    return "value"

def _interp_llm_json(*, ctx: Dict[str, Any], chart_title: str, chart_desc: str, stats: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    block_id = "cot__overview_interpretation"
    """Overview interpretation with lean selection: validate -> compare -> select."""
    max_attempts = int((ctx or {}).get('llm_max_attempts', (ctx or {}).get('max_attempts', 3)) or 3)
    if max_attempts < 1:
        max_attempts = 1

    mode = _infer_interp_mode(chart_title, stats or {})
    expect_pct = (mode in ("shares_pp", "share", "mix"))
    payload = {"chart_title": chart_title, "chart_desc": chart_desc, "mode": mode, "expect_pct": expect_pct, "stats": stats}

    det_payload = (
        _build_overview_det_premium_shares(stats or {}, chart_title, chart_desc)
        if mode in ("shares_pp", "share", "mix")
        else _build_overview_det_premium_value(stats or {}, chart_title, chart_desc)
    )

    def _run_quality_reviews(out: Dict[str, Any], *, mode: str) -> List[str]:
        extra: List[str] = []
        if mode in ("shares_pp", "share", "mix"):
            extra.extend([f"shares_one_sentence:{r}" for r in _review_shares_one_sentence_quality(str((out or {}).get("one_sentence") or ""))])
        extra.extend([f"recommendations:{r}" for r in _review_interp_recommendations_quality(list((out or {}).get("recommendations") or []), mode=mode, stats=stats or {})])
        return extra

    def _validate_candidate(out: Dict[str, Any], *, src: str) -> Tuple[bool, List[str], bool, str, List[str]]:
        ok_hard, meta_hard = _validate_interp_hard_v2(out, payload)
        reasons_hard = list((meta_hard.get("hard_validator_v2") or {}).get("reasons") or [])
        reasons_hard.extend(_run_quality_reviews(out, mode=mode))
        if mode == "value":
            one_sentence = str((out or {}).get("one_sentence") or "")
            has_named_category = bool(re.search(r"„[^”]+”|\"[^\"]+\"", one_sentence))
            has_named_peak_period = bool(
                re.search(r"\b(stycze|lut|mar|kwie|maj|czer|lip|sier|wrze|paź|listop|grud|q[1-4]|20\d{2})", one_sentence.lower())
            )
            if not (has_named_category or has_named_peak_period):
                reasons_hard.append("value_missing_named_driver")
        if reasons_hard:
            ok_hard = False

        value_named_driver_only_fail = (
            mode == "value"
            and bool(reasons_hard)
            and set(reasons_hard) == {"value_missing_named_driver"}
        )

        ok_sem, sem_topic, reasons_sem = _validate_interp_semantic_v3(out, stats)
        dbg_cp(
            "overview.interp.before_fallback_decision",
            block_id="cot__overview_interpretation",
            from_path=f"_interp_llm_json.validate_{src}",
            chart_title=chart_title,
            mode=mode,
            src=src,
            topic=(sem_topic or ("shares_pp" if mode in ("shares_pp", "share", "mix") else "value_pln")),
            fallback_used=not (ok_hard and ok_sem),
            repair_used=(src == "llm_repair"),
            v2=bool(ok_hard),
            v3=bool(ok_sem),
            reasons_hard=reasons_hard[:6],
            reasons_sem=reasons_sem[:6],
            preview=str((out or {}).get("one_sentence") or "")[:180],
        )
        return bool(ok_hard and ok_sem), reasons_hard, bool(ok_sem), (sem_topic or ""), reasons_sem

    def _finalize(selected_out: Dict[str, Any], src_label: str, sel_meta: Dict[str, Any], *,
                  ok_hard: bool, reasons_hard: List[str], ok_sem: bool, sem_topic: str,
                  reasons_sem: List[str], repair_used: bool, repair_passed: Optional[bool], fallback_used: bool) -> Tuple[Dict[str, Any], str]:
        final_topic = _canonical_interp_topic(mode, sem_topic)
        dbg_cp(
            "overview.interp.selector_decision",
            block_id="cot__overview_interpretation",
            chart_title=chart_title,
            mode=mode,
            score_llm=int(sel_meta.get("score_llm") or 0),
            score_det=int(sel_meta.get("score_det") or 0),
            selected_candidate=str(sel_meta.get("selected_candidate") or ""),
            selection_reason=str(sel_meta.get("selection_reason") or ""),
            llm_eval=sel_meta.get("llm_eval") or {},
            det_eval=sel_meta.get("det_eval") or {},
        )
        dbg_cp(
            "overview.interp.final_source",
            block_id="cot__overview_interpretation",
            from_path="_interp_llm_json.return_selected",
            chart_title=chart_title,
            mode=mode,
            src=src_label,
            topic=final_topic,
            fallback_used=fallback_used,
            repair_used=repair_used,
            text_len=len(str((selected_out or {}).get("one_sentence") or "")),
            preview=str((selected_out or {}).get("one_sentence") or "")[:180],
        )
        selected_src = src_label if isinstance(src_label, str) else ""

        dbg_cp(
            "overview.interp.runtime_quality_summary",
            block_id="cot__overview_interpretation",
            final_src=selected_src,
            llm_validation_passed=bool(ok_hard and ok_sem),
            fallback_used=selected_src.startswith("fallback"),
        )
        return _emit_interp_meta(
            selected_out,
            src_label,
            chart_title,
            ok_hard=ok_hard,
            reasons_hard=reasons_hard,
            ok_sem=ok_sem,
            sem_topic=final_topic,
            reasons_sem=reasons_sem,
            repair_used=repair_used,
            repair_passed=repair_passed,
            fallback_used=fallback_used,
        )

    try:
        messages = [
            {"role": "system", "content": _build_interp_system_prompt(stats or {})},
                        {"role": "user", "content": (
                "Wygeneruj JSON interpretacji zgodny ze schema.\n\n"
                f"Tytuł: {chart_title}\n"
                f"Opis: {chart_desc}\n\n"
                f"WYMIAR: {_dimension_display_label((stats or {}).get('dimension_label') or (stats or {}).get('cat_col'), fallback='Wymiar')}\n"
                f"STATYSTYKI (JSON): {json_dumps_safe(payload)}\n\n"
                "WYMAGANIA STRATEGICZNE (MUST):\n"
                "- 'one_sentence': max 2 zdania. Struktura: (1) driver + liczby, (2) mechanizm biznesowy + decyzja.\n"
                "  Dla mode=shares_pp MUSI zawierać: nazwę drivera / elementu wymiaru w cudzysłowie, wzorzec start→end oraz delta pp, mechanizm biznesowy i decyzję zarządczą / portfelową / alokacyjną.\n"
                "  NIEDOZWOLONE jako decyzja końcowa: 'monitorować trendy', 'obserwować kategorię', 'analizować dalej', 'warto rozważyć', 'należy przeanalizować'.\n"
                "- 'what_chart_shows': 2–3 krótkie linie, ale zarządcze — nie definicje BI-grade.\n"
                "- 'key_insights': 2–4 unikalne punkty, każdy z innym anchorem liczbowym.\n"
                "- 'recommendations': 2–3 działania operacyjne / portfelowe; KPI + horyzont tylko gdy wynikają z danych lub są użyte jako kontrola operacyjna, bez halucynowanych budżetów.\n"
                "- 'limitations': 1–3 konkretne ograniczenia danych/filtrów/miary.\n\n"
                "Brzmienie: jak slajd zarządczy / komitet inwestycyjny.\n"
                "Zwróć WYŁĄCZNIE JSON."
            )},
        ]
        out = _call_llm_with_trace(
            ctx=ctx,
            messages=messages,
            model=str((ctx.get("openai_model") or "gpt-4o-mini")),
            temperature=0.2,
            response_format={"type": "json_schema", "json_schema": {"name": "interpretation", "schema": _INTERP_SCHEMA}},
            payload={"kind": "cot_interpretation"},
            trace_where="llm.wrapper.overview",
        )

        if isinstance(out, dict) and all(k in out for k in _INTERP_SCHEMA["required"]):
            dbg_cp("interp.llm.raw", block_id="cot__overview_interpretation", from_path="_interp_llm_json.raw", chart_title=chart_title, mode=mode, src="llm", text_len=len(str((out or {}).get("one_sentence") or "")), preview=str((out or {}).get("one_sentence") or "")[:180])
            out = _postprocess_interp_dict(out, payload)
            out = _upgrade_what_chart_shows(out, mode=mode, stats=stats or {})
            if mode in ("shares_pp", "share", "mix"):
                out = _enforce_share_one_sentence_driver_delta(out, stats)
                out = _upgrade_share_interp_to_executive(out, stats or {})
            if mode in ("value", "sprzedaz", "sales"):
                out = _upgrade_value_interp_to_executive(out, stats or {})
            if mode in ("shares_pp", "share", "mix"):
                ov_ok, ov_reasons = _validate_overview_share_exec_sentence(out, stats or {})
                dbg_cp(
                    "overview.validator_contract_audit",
                    block_id=block_id,
                    one_sentence=out.get("one_sentence") if isinstance(out, dict) else "",
                    uplift_mechanism_vocab=[
                        "oddawanie udziału",
                        "osłabienie pozycji",
                        "transfer udziału",
                        "trwałe przesunięcie",
                        "erozj",
                        "wzmacnianie pozycji",
                        "umocnienie pozycji",
                        "przejęcie dodatkowego udziału",
                    ],
                    matched_mechanism_tokens=[
                        w for w in [
                            "oddawanie udziału",
                            "osłabienie pozycji",
                            "transfer udziału",
                            "trwałe przesunięcie",
                            "erozj",
                            "wzmacnianie pozycji",
                            "umocnienie pozycji",
                            "przejęcie dodatkowego udziału",
                        ] if w in str(out.get("one_sentence") or "").lower()
                    ],
                    exec_validator_failed_reason=ov_reasons if not ov_ok else [],
                )
                _ov_txt = str(out.get("one_sentence") or "")
                _ov_tl = _ov_txt.lower()
                dbg_cp(
                    "overview.interp.exec_quality_audit",
                    block_id=block_id,
                    mechanism_present=any(w in _ov_tl for w in [
                        "oddawanie udziału", "osłabienie pozycji",
                        "transfer udziału", "trwałe przesunięcie", "erozj",
                        "wzmacnianie pozycji", "umocnienie pozycji",
                        "przejęcie dodatkowego udziału"
                    ]),
                    decision_present=any(w in _ov_tl for w in [
                        "cena", "alokacja wsparcia", "decyzja:", "dlatego:", "przesuń",
                        "utrzymaj", "ekspozycję", "ekspozycje", "alokację", "alokacje"
                    ]),
                    driver_present=bool(
                        _first_not_none(
                            (stats or {}).get("driver_dimension_value"),
                            (stats or {}).get("driver_category_name"),
                        )
                        and str(_first_not_none(
                            (stats or {}).get("driver_dimension_value"),
                            (stats or {}).get("driver_category_name"),
                        )).lower() in _ov_tl
                    ),
                    numeric_anchor_present=bool(re.search(r"\d", _ov_txt)),
                    validator_failed_reason=ov_reasons if not ov_ok else [],
                )
                if not ov_ok:
                    dbg_cp(
                        "overview.interp.share_exec_gate_failed",
                        block_id=block_id,
                        reasons=ov_reasons,
                        preview=str(out.get("one_sentence", ""))[:200],
                    )
                    out = {}
            dbg_cp("overview.interp.candidate_postprocess", block_id="cot__overview_interpretation", from_path="_interp_llm_json.postprocess", chart_title=chart_title, mode=mode, src="llm", text_len=len(str((out or {}).get("one_sentence") or "")), preview=str((out or {}).get("one_sentence") or "")[:180])

            ok_all, reasons_hard, ok_sem, sem_topic, reasons_sem = _validate_candidate(out, src="llm")
            if ok_all:
                selected_out, selected_src, sel_meta = _select_best_overview_interp(
                    mode=mode,
                    chart_title=chart_title,
                    stats=stats or {},
                    llm_interp=out,
                    llm_valid=True,
                    llm_src="llm_selected",
                    det_interp=det_payload,
                )
                if selected_src == "llm_selected":
                    return _finalize(selected_out, selected_src, sel_meta, ok_hard=True, reasons_hard=[], ok_sem=True, sem_topic=sem_topic, reasons_sem=[], repair_used=False, repair_passed=None, fallback_used=False)
                return _finalize(selected_out, selected_src, sel_meta, ok_hard=True, reasons_hard=reasons_hard, ok_sem=True, sem_topic=sem_topic, reasons_sem=reasons_sem, repair_used=False, repair_passed=None, fallback_used=True)
            value_named_driver_only_fail = (
                mode == "value"
                and bool(reasons_hard)
                and set(reasons_hard) == {"value_missing_named_driver"}
            )

            shares_key_insights_only_fail = (
                mode in ("shares_pp", "share", "mix")
                and bool(reasons_hard)
                and all(str(r).startswith("key_insights[") for r in reasons_hard)
            )

            if value_named_driver_only_fail:
                selected_out, selected_src, sel_meta = _select_best_overview_interp(
                    mode=mode,
                    chart_title=chart_title,
                    stats=stats or {},
                    llm_interp=out,
                    llm_valid=True,
                    llm_src="llm_selected",
                    det_interp=det_payload,
                )
                return _finalize(
                    selected_out,
                    selected_src,
                    sel_meta,
                    ok_hard=True,
                    reasons_hard=[],
                    ok_sem=True,
                    sem_topic="value_pln",
                    reasons_sem=[],
                    repair_used=False,
                    repair_passed=False,
                    fallback_used=(selected_src != "llm_selected"),
                )

            if shares_key_insights_only_fail:
                selected_out, selected_src, sel_meta = _select_best_overview_interp(
                    mode=mode,
                    chart_title=chart_title,
                    stats=stats or {},
                    llm_interp=out,
                    llm_valid=True,
                    llm_src="llm_selected",
                    det_interp=det_payload,
                )
                return _finalize(
                    selected_out,
                    selected_src,
                    sel_meta,
                    ok_hard=True,
                    reasons_hard=[],
                    ok_sem=ok_sem,
                    sem_topic=sem_topic,
                    reasons_sem=reasons_sem,
                    repair_used=False,
                    repair_passed=False,
                    fallback_used=(selected_src != "llm_selected"),
                )

            schema_txt = _mbb_json_dumps_safe(_INTERP_SCHEMA)
            raw_txt = _mbb_json_dumps_safe(out)

            repair_prompt = f"""NAPRAWA JSON interpretacji.
Zwróć WYŁĄCZNIE poprawny JSON zgodny ze schematem.

Wymagania jakościowe:
- Zachowaj schema i klucze.
- Popraw tylko treść pól.
- one_sentence ma brzmieć jak executive summary, nie opis wykresu.

Jeśli mode należy do shares_pp/share/mix:
- one_sentence MUSI zawierać:
  - nazwę kategorii/drivera,
  - start %,
  - end %,
  - delta pp,
  - mechanizm dosłownie jednym z wyrażeń:
    - oddawanie udziału
    - osłabienie pozycji
    - transfer udziału
    - erozja rdzenia
  - decyzję dosłownie zawierającą co najmniej jedno z:
    - cena
    - ekspozycja
    - asortyment
    - alokacja wsparcia

Jeśli mode należy do value/sprzedaz/sales:
- one_sentence MUSI zawierać:
  - total_value,
  - peak_value,
  - peak_month,
  - mechanizm dosłownie: koncentracja przychodu w szczycie,
  - decyzję operacyjną zawierającą co najmniej jeden konkretny kierunek:
    - zabezpiecz dostępność przed pikiem
    - zwiększ ekspozycję przed szczytem
    - ogranicz aktywację poza szczytem
    - traktuj miesiące poza szczytem jako selektywną aktywację
- one_sentence NIE MOŻE zawierać decyzji generycznej typu:
  - optymalizacja kanałów sprzedaży
  - optymalizacja portfela produktów
  - zwiększenie sprzedaży
  - zwiększenie inwestycji w marketing
  - zwiększenie alokacji zasobów
- decyzja NIE MOŻE być ogólnikiem typu:
  - optymalizacja kanałów sprzedaży
  - optymalizacja portfela produktów
  - zwiększenie sprzedaży
  - zwiększenie alokacji zasobów

Unikaj fraz generycznych:
- należy monitorować
- warto rozważyć
- zwiększyć marketing
- poprawić pozycję
- wesprzeć kategorię

SCHEMA:
{json.dumps(_INTERP_SCHEMA, ensure_ascii=False)}

JSON_WEJSCIOWY:
{json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=False)}
"""
            repaired = _call_llm_with_trace(
                ctx=ctx,
                messages=[{"role": "system", "content": _build_interp_system_prompt(stats or {})}, {"role": "user", "content": repair_prompt}],
                model=str((ctx.get("openai_model") or "gpt-4o-mini")),
                temperature=0.1,
                response_format={"type": "json_schema", "json_schema": {"name": "interpretation", "schema": _INTERP_SCHEMA}},
                payload={"kind": "cot_interpretation_repair"},
                trace_where="llm.wrapper.overview_repair",
            )
            if isinstance(repaired, dict) and all(k in repaired for k in _INTERP_SCHEMA["required"]):
                dbg_cp("interp.llm.repair", block_id="cot__overview_interpretation", from_path="_interp_llm_json.repair_raw", chart_title=chart_title, mode=mode, src="llm_repair", text_len=len(str((repaired or {}).get("one_sentence") or "")), preview=str((repaired or {}).get("one_sentence") or "")[:180])
                repaired = _postprocess_interp_dict(repaired, payload)
                repaired = _upgrade_what_chart_shows(repaired, mode=mode, stats=stats or {})
                if mode in ("shares_pp", "share", "mix"):
                    repaired = _enforce_share_one_sentence_driver_delta(repaired, stats)
                    repaired = _upgrade_share_interp_to_executive(repaired, stats or {})
                if mode in ("value", "sprzedaz", "sales"):
                    repaired = _upgrade_value_interp_to_executive(repaired, stats or {})
                dbg_cp("interp.llm.repair", block_id="cot__overview_interpretation", from_path="_interp_llm_json.repair_postprocess", chart_title=chart_title, mode=mode, src="llm_repair", text_len=len(str((repaired or {}).get("one_sentence") or "")), preview=str((repaired or {}).get("one_sentence") or "")[:180])
                ok_all2, reasons_hard2, ok_sem2, sem_topic2, reasons_sem2 = _validate_candidate(repaired, src="llm_repair")
                if ok_all2:
                    selected_out, selected_src, sel_meta = _select_best_overview_interp(
                        mode=mode,
                        chart_title=chart_title,
                        stats=stats or {},
                        llm_interp=repaired,
                        llm_valid=True,
                        llm_src="llm_repair_selected",
                        det_interp=det_payload,
                    )

                    if selected_src == "llm_repair_selected":
                        return _finalize(
                            selected_out,
                            selected_src,
                            sel_meta,
                            ok_hard=True,
                            reasons_hard=[],
                            ok_sem=True,
                            sem_topic=sem_topic2,
                            reasons_sem=[],
                            repair_used=True,
                            repair_passed=True,
                            fallback_used=False,
                        )

                    return _finalize(
                        selected_out,
                        selected_src,
                        sel_meta,
                        ok_hard=True,
                        reasons_hard=reasons_hard2,
                        ok_sem=True,
                        sem_topic=sem_topic2,
                        reasons_sem=reasons_sem2,
                        repair_used=True,
                        repair_passed=True,
                        fallback_used=True,
                    )
    except Exception:
        pass

    sel_out, sel_src, sel_meta = _select_best_overview_interp(
        mode=mode,
        chart_title=chart_title,
        stats=stats or {},
        llm_interp={},
        llm_valid=False,
        llm_src="llm_selected",
        det_interp=det_payload,
    )
    return _finalize(sel_out, sel_src, sel_meta, ok_hard=True, reasons_hard=[], ok_sem=True, sem_topic=("shares_pp" if mode in ("shares_pp", "share", "mix") else "value_pln"), reasons_sem=[], repair_used=False, repair_passed=False, fallback_used=True)


def json_dumps_safe(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"

def _llm_trace_preview(obj: Any, limit: int = 500) -> str:
    try:
        s = repr(obj)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            s = "<unrepr>"
    s = " ".join(str(s).split())
    return s[:limit]


def _extract_text_from_llm_response(resp: Any) -> str:
    """
    Wspólny extractor odpowiedzi tekstowej:
    - string
    - dict: content/text/output_text/answer/final_text/one_sentence/takeaway
    - dict schema-like: properties.takeaway / properties.text / properties.content
    - choices[0].message.content / choices[0].text
    - output[].content[].text
    - listy
    """
    if resp is None:
        return ""

    if isinstance(resp, str):
        return " ".join(resp.strip().split())

    if isinstance(resp, dict):
        for key in ["content", "text", "output_text", "answer", "final_text", "one_sentence", "takeaway"]:
            val = resp.get(key)
            if isinstance(val, str) and val.strip():
                return " ".join(val.strip().split())

        props = resp.get("properties")
        if isinstance(props, dict):
            for key in ["takeaway", "text", "content"]:
                val = props.get(key)
                if isinstance(val, str) and val.strip():
                    return " ".join(val.strip().split())

        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            ch0 = choices[0] or {}
            if isinstance(ch0, dict):
                msg = ch0.get("message") or {}
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return " ".join(content.strip().split())
                txt = ch0.get("text")
                if isinstance(txt, str) and txt.strip():
                    return " ".join(txt.strip().split())

        output = resp.get("output")
        if isinstance(output, list):
            parts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            t = c.get("text") or c.get("content")
                            if isinstance(t, str) and t.strip():
                                parts.append(t.strip())
            if parts:
                return " ".join(" ".join(parts).split())

        return ""

    if isinstance(resp, list):
        parts = []
        for item in resp:
            txt = _extract_text_from_llm_response(item)
            if txt:
                parts.append(txt)
        return " ".join(" ".join(parts).split())

    return ""

def _call_llm_with_trace(
    *,
    ctx: Dict[str, Any],
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 180,
    payload: Optional[Dict[str, Any]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    trace_where: str = "llm.wrapper",
) -> Any:
    """
    Wspólny wrapper trace dla overview + ET.
    Nie zmienia kontraktu llm_fn — tylko dodaje trace wejścia/wyjścia.
    """
    _payload = payload if isinstance(payload, dict) else {}
    _payload = {
        "kind": str(_payload.get("kind") or ""),
        "block_id": str(_payload.get("block_id") or ""),
    }
    _schema_first_text_takeaway = {
        "type": "json_schema",
        "json_schema": {
            "name": "text_takeaway",
            "schema": {
                "type": "object",
                "properties": {
                    "takeaway": {"type": "string", "description": "Final short business takeaway in Polish"}
                },
                "required": ["takeaway"],
                "additionalProperties": False,
            },
        },
    }
    _schema_first_et = (
        response_format is None
        and str(_payload.get("kind") or "") in {"cot_exec_takeaway_single_direct"}
    )
    _effective_response_format = response_format if response_format is not None else (_schema_first_text_takeaway if _schema_first_et else None)

    try:
        dbg_cp(
            f"{trace_where}.request",
            payload_kind=str(_payload.get("kind") or ""),
            block_id=str(_payload.get("block_id") or ""),
            model=str(model or ""),
            temperature=temperature,
            max_tokens=max_tokens,
            msg_count=len(messages or []),
            has_response_format=bool(_effective_response_format),
            user_preview=str((messages or [{}])[-1].get("content") or "")[:220],
        )
    except Exception:
        pass

    try:
        if _effective_response_format is not None:
            resp = llm_fn(
                messages=messages,
                model=model,
                temperature=temperature,
                response_format=_effective_response_format,
                payload=_payload,
            )
        else:
            resp = llm_fn(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                ctx=ctx,
                payload=_payload,
            )
    except Exception as e:
        try:
            dbg_cp(
                f"{trace_where}.exception",
                payload_kind=str(_payload.get("kind") or ""),
                block_id=str(_payload.get("block_id") or ""),
                error=f"{type(e).__name__}: {e}",
            )
        except Exception:
            pass
        raise

    try:
        dbg_cp(
            f"{trace_where}.response",
            payload_kind=str(_payload.get("kind") or ""),
            block_id=str(_payload.get("block_id") or ""),
            response_type=type(resp).__name__,
            preview=_llm_trace_preview(resp, 500),
        )
    except Exception:
        pass

    # Text branch rescue:
    # for ET single-direct we use schema-first; rescue remains only when takeaway is still empty.
    # for plain-text branches we preserve the old "retry once with schema" behavior.
    try:
        _needs_rescue = False
        if _schema_first_et:
            _tk_probe = ""
            if isinstance(resp, dict):
                _tk_probe = str(resp.get("takeaway") or "").strip()
            if not _tk_probe:
                _txt_probe = _extract_text_from_llm_response(resp)
                if not str(_txt_probe or "").strip():
                    _needs_rescue = True
        elif response_format is None:
            if resp is None:
                _needs_rescue = True
            elif isinstance(resp, dict) and len(resp) == 0:
                _needs_rescue = True
            elif isinstance(resp, list) and len(resp) == 0:
                _needs_rescue = True
            elif isinstance(resp, str) and not resp.strip():
                _needs_rescue = True
            else:
                _txt_probe = _extract_text_from_llm_response(resp)
                if not str(_txt_probe or "").strip():
                    _needs_rescue = True

        if _needs_rescue:
            dbg_cp(
                f"{trace_where}.text_rescue_request",
                payload_kind=str(_payload.get("kind") or ""),
                block_id=str(_payload.get("block_id") or ""),
                reason=("empty_schema_takeaway" if _schema_first_et else "empty_plain_text_response"),
            )

            resp2 = llm_fn(
                messages=messages,
                model=model,
                temperature=temperature,
                response_format=_schema_first_text_takeaway,
                payload={**_payload, "kind": str(_payload.get("kind") or "") + "_json_rescue"},
            )

            dbg_cp(
                f"{trace_where}.text_rescue_response",
                payload_kind=str(_payload.get("kind") or ""),
                block_id=str(_payload.get("block_id") or ""),
                response_type=type(resp2).__name__,
                preview=_llm_trace_preview(resp2, 500),
            )

            if isinstance(resp2, dict) and ("takeaway" in resp2):
                return resp2
            return resp2
    except Exception as _rescue_e:
        try:
            dbg_cp(
                f"{trace_where}.text_rescue_exception",
                payload_kind=str(_payload.get("kind") or ""),
                block_id=str(_payload.get("block_id") or ""),
                error=f"{type(_rescue_e).__name__}: {_rescue_e}",
            )
        except Exception:
            pass

    return resp

def _render_interpretation_ui(target, data: dict) -> None:
    """Render 'Interpretacja' block with the same pixel spacing as Distribution / CS.

    NOTE: we intentionally use HTML wrappers for tight spacing, but we convert markdown **bold**
    markers into <strong> so the UI stays clean (no literal ** in text).
    """
    if not isinstance(data, dict):
        return

    # Ensure Distribution-like typography for ALL subsections (avoid smaller fonts after the first block)
    target.markdown(
        """
        <style>
        .cot-interp {font-family: inherit;}
        .cot-interp .interp-h {font-size:16px !important; font-weight:700 !important; line-height:1.25 !important; margin:0 !important;}
        .cot-interp .interp-b, .cot-interp ul, .cot-interp li {font-size:15px !important; line-height:1.35 !important;}
        </style>
        <div class="cot-interp">
        """,
        unsafe_allow_html=True,
    )

    # IMPORTANT: target may be a placeholder (st.empty()). Using .markdown() repeatedly would overwrite.
    # Normalize to a container so we can render multiple lines consistently (same as Distribution/CS).
    try:
        target = target.container()
    except Exception:
        pass

    def _fmt_html(x: object) -> str:
        s = "" if x is None else str(x)
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        return s

    def _h(title: str) -> None:
        # Match Distribution: clear hierarchy and consistent typography
        target.markdown(
            f"<div style='font-weight:700; margin:14px 0 6px 0; font-size:16px; line-height:1.25;'>{_fmt_html(title)}</div>",
            unsafe_allow_html=True,
        )

    def _tight_bullets(lines: list[str]) -> None:
        for x in (lines or []):
            target.markdown(
                "<div style='margin:3px 0 3px 0; line-height:1.45; font-size:15px;'>• " + _fmt_html(x) + "</div>",
                unsafe_allow_html=True,
            )
        target.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    one = data.get("one_sentence") or data.get("one_sentence_answer") or ""
    # Support both our COT schema keys and legacy keys used elsewhere
    what = (
        data.get("what_chart_shows")
        or data.get("what")
        or data.get("co_pokazuje")
        or []
    )
    insights = (
        data.get("key_insights")
        or data.get("insights")
        or data.get("kluczowe")
        or []
    )
    reco = (
        data.get("recommendations")
        or data.get("reco")
        or data.get("rekomendacje")
        or []
    )
    segments = data.get("segments") or data.get("segmenty") or None
    limits = (
        data.get("limitations")
        or data.get("limits")
        or data.get("ograniczenia")
        or []
    )

    # One sentence
    _h("✅ Odpowiedź w jednym zdaniu")
    if one:
        target.markdown(
            "<div style='margin:0 0 4px 0; line-height:1.35;'>" + _fmt_html(one) + "</div>",
            unsafe_allow_html=True,
        )
        target.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Sections
    if what:
        _h("📊 Co pokazuje wykres")
        _tight_bullets(list(what) if isinstance(what, (list, tuple)) else [str(what)])

    if insights:
        _h("💡 Kluczowe insighty")
        _tight_bullets(list(insights) if isinstance(insights, (list, tuple)) else [str(insights)])

    if reco:
        _h("🎯 Rekomendacje działań")
        _tight_bullets(list(reco) if isinstance(reco, (list, tuple)) else [str(reco)])

    if segments:
        _h("🧩 Segmenty / klastry (jeśli dotyczy)")
        _tight_bullets(list(segments) if isinstance(segments, (list, tuple)) else [str(segments)])

    if limits:
        _h("⚠️ Ograniczenia / zastrzeżenia")
        _tight_bullets(list(limits) if isinstance(limits, (list, tuple)) else [str(limits)])
    target.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Charts (Tab 1 + Tab 2 blocks)
# -----------------------------

def _safe_series(df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    """Return a Series for a column label even if df has duplicate column names.
    If the label resolves to multiple columns, take the first one deterministically."""
    if not col:
        return None
    if col not in df.columns:
        return None
    v = df.loc[:, col]
    if isinstance(v, pd.DataFrame):
        # duplicate column names -> take first
        if v.shape[1] == 0:
            return None
        return v.iloc[:, 0]
    return v

def _prep_time_df(
    df: pd.DataFrame,
    time_col: str,
    cat_col: str,
    value_col: Optional[str],
    qty_col: Optional[str],
    price_col: Optional[str],
) -> pd.DataFrame:
    x = df[[time_col, cat_col] + ([c for c in [value_col, qty_col, price_col] if c and c in df.columns])].copy()
    x["__month"] = _to_month_start(x[time_col])
    x = x.dropna(subset=["__month", cat_col])
    x[cat_col] = x[cat_col].astype(str)
    # __value: prefer explicit value_col; otherwise qty*price; fallback = 1.0 (count)
    s_value = _safe_series(x, value_col) if value_col else None
    if s_value is not None:
        x["__value"] = pd.to_numeric(s_value, errors="coerce")
    else:
        s_qty = _safe_series(x, qty_col) if qty_col else None
        s_price = _safe_series(x, price_col) if price_col else None
        if (s_qty is not None) and (s_price is not None):
            x["__value"] = pd.to_numeric(s_qty, errors="coerce") * pd.to_numeric(s_price, errors="coerce")
        else:
            x["__value"] = 1.0

    x["__value"] = x["__value"].fillna(0.0)
    return x

def _densify_month_grid(
    df_g: pd.DataFrame,
    *,
    month_col: str = "__month",
    cat_col: str,
    value_col: str = "__value",
    months: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Ensure full monthly grid so stacked areas don't create diagonal 'bridges'.

    For each category we create one row per month; missing values are filled with 0.0.
    """
    if df_g.empty:
        return df_g
    if months is None:
        mn = pd.to_datetime(df_g[month_col]).min()
        mx = pd.to_datetime(df_g[month_col]).max()
        if pd.isna(mn) or pd.isna(mx):
            return df_g
        months = pd.date_range(mn, mx, freq="MS")
    cats = sorted(df_g[cat_col].dropna().astype(str).unique().tolist())
    if not cats:
        return df_g
    idx = pd.MultiIndex.from_product([months, cats], names=[month_col, cat_col])
    out = (
        df_g.assign(**{cat_col: df_g[cat_col].astype(str)})
        .set_index([month_col, cat_col])
        .reindex(idx)
        .reset_index()
    )
    if value_col in out.columns:
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)
    return out
def _kpis(df_time: pd.DataFrame, txn_col: Optional[str], raw_df: pd.DataFrame) -> Tuple[float, int]:
    total_value = float(df_time["__value"].sum())
    # Consistency across branches: show number of loaded rows (records).
    # (Other branches label this as "Liczba transakcji" but use df length.)
    txn = int(len(raw_df))
    return total_value, txn

def _main_chart_share(
    df_time_main: pd.DataFrame,
    cat_col: str,
    top_k: int = 10,
    *,
    include_other: bool = True,
    months_override: Optional[pd.DatetimeIndex] = None,
    df_time_total: Optional[pd.DataFrame] = None,
) -> Tuple[alt.Chart, Dict[str, Any]]:
    """Main (Tab: Obraz całości) share chart.

    Truth rule: shares are ALWAYS computed vs total sales (including explicit 'Other' category),
    even if 'Other' is hidden on the chart.

    Params kept backward-compatible with earlier code (top_k, months_override).
    """
    if df_time_total is None:
        df_time_total = df_time_main

    other_label = _infer_other_label(df_time_main[cat_col] if cat_col in df_time_main.columns else pd.Series(dtype=str))

    # Totals over ALL categories (incl. explicit Other)
    totals_all = (
        df_time_total.groupby("__month", as_index=False)["__value"]
        .sum()
        .rename(columns={"__value": "__total_all"})
    )

    # Determine TopK across total data (excluding explicit Other)
    top = _top_categories(df_time_total, cat_col, top_n=top_k)

    # Display categories depend on include_other
    if include_other:
        cats = list(top)
        if other_label in set(df_time_total[cat_col].astype(str)):
            cats = cats + [other_label]
    else:
        cats = list(top)

    # Aggregate values for displayed categories
    g = (
        df_time_total[df_time_total[cat_col].astype(str).isin([str(x) for x in cats])]
        .groupby(["__month", cat_col], as_index=False)["__value"]
        .sum()
    )

    # Ensure complete month x category grid to avoid implicit interpolation artifacts
    if months_override is None:
        months_override = pd.date_range(df_time_total["__month"].min(), df_time_total["__month"].max(), freq="MS")

    if not g.empty and months_override is not None:
        idx = pd.MultiIndex.from_product([months_override, [str(x) for x in cats]], names=["__month", cat_col])
        g = (
            g.assign(**{cat_col: g[cat_col].astype(str)})
            .set_index(["__month", cat_col])["__value"]
            .reindex(idx, fill_value=0.0)
            .reset_index()
        )

    m = g.merge(totals_all, on="__month", how="left")
    m["share"] = np.where(m["__total_all"] > 0, m["__value"] / m["__total_all"], 0.0)

    # Domain order (bottom->top layers). Legend should be reversed (top->bottom).
    dom = list(top)
    if include_other and other_label in set(m[cat_col].astype(str)):
        dom = dom + [other_label]
    dom_legend = list(reversed(dom))

    # Explicit stack order matching dom
    order_map = {c: i for i, c in enumerate(dom)}
    m["__order"] = m[cat_col].astype(str).map(order_map).fillna(999).astype(int)

    scale_base = _build_fixed_color_scale(dom, top_k=min(5, len(dom)))
    scale = alt.Scale(
        domain=list(reversed(scale_base.domain)),
        range=list(reversed(scale_base.range)),
    )
    legend_order = list(scale.domain) if getattr(scale, "domain", None) else []

    dim_title = _dimension_axis_title(cat_col)

    ch = (
        alt.Chart(m)
        .mark_area(opacity=0.85, strokeWidth=0)
        .encode(
            x=alt.X(
                "__month:T",
                title="Miesiąc",
                axis=alt.Axis(
                    format="%b %Y",
                    labelAngle=45,
                    labelAlign="left",
                    labelBaseline="middle",
                    tickCount={"interval": "month", "step": 1},
                    labelOverlap=False,
                ),
            ),
            y=alt.Y("share:Q", axis=alt.Axis(format="%", title="Udział sprzedaży (%)")),
            color=alt.Color(
                f"{cat_col}:N",
                scale=scale,
                sort=None,
                legend=alt.Legend(title=dim_title, symbolType="circle"),
            ),
            order=alt.Order("__order:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("__month:T", title="Miesiąc", format="%b %Y"),
                alt.Tooltip(f"{cat_col}:N", title=dim_title),
                alt.Tooltip("share:Q", title="Udział (vs Total)", format=".1%"),
                alt.Tooltip("__value:Q", title="Wartość"),
            ],
        )
        .properties(height=max(int(CHART_BLOCK_HEIGHT * 1.35), 420))
    )

    # --- ET anchors (to satisfy validator: >=2 numbers in sentence 1) ---
    _m_min = m["__month"].min()
    _m_max = m["__month"].max()
    _top = [str(x) for x in top]
    _top1 = _top[0] if _top else None

    def _pct_share(_month, _cats):
        try:
            return float(m.loc[(m["__month"] == _month) & (m[cat_col].isin(_cats)), "share"].sum() * 100.0)
        except Exception:
            return float("nan")

    _topN_start_pct = _pct_share(_m_min, _top) if _top else float("nan")
    _topN_end_pct = _pct_share(_m_max, _top) if _top else float("nan")
    _top1_start_pct = _pct_share(_m_min, [_top1]) if _top1 else float("nan")
    _top1_end_pct = _pct_share(_m_max, [_top1]) if _top1 else float("nan")
    _topN_delta_pp = (_topN_end_pct - _topN_start_pct) if _top else float("nan")
    _top1_delta_pp = (_top1_end_pct - _top1_start_pct) if _top1 else float("nan")

    stats = {
        "top_categories": _top,
        "legend_order": legend_order,
        "include_other": bool(include_other),
        "share_denominator": "total_all_including_other",
        "start_month": str(_m_min) if _m_min is not None else None,
        "end_month": str(_m_max) if _m_max is not None else None,
        "top1_dimension_value": _top1,
        "leader_dimension_value": _top1,
        "primary_dimension_value": _top1,
        "top1_category": _top1,
        "top1_start_pct": _top1_start_pct,
        "top1_end_pct": _top1_end_pct,
        "top1_delta_pp": _top1_delta_pp,
        "topN_start_pct": _topN_start_pct,
        "topN_end_pct": _topN_end_pct,
        "topN_delta_pp": _topN_delta_pp,
    }
    return ch, stats

def _main_chart_value(df_time: pd.DataFrame, cat_col: str, *, top_k: int = 10, months_override: Optional[pd.DatetimeIndex] = None) -> alt.LayerChart:
    # full monthly grid (prevents diagonal 'bridges' when some categories appear sporadically)
    mn = pd.to_datetime(df_time["__month"]).min()
    mx = pd.to_datetime(df_time["__month"]).max()
    months = (
        months_override
        if months_override is not None
        else (pd.date_range(mn, mx, freq="MS") if (pd.notna(mn) and pd.notna(mx)) else None)
    )

    g = df_time.groupby(["__month", cat_col], as_index=False)["__value"].sum()
    g = _densify_month_grid(g, cat_col=cat_col, months=months)

    total = df_time.groupby("__month", as_index=False)["__value"].sum().rename(columns={"__value": "total"})
    if months is not None and not total.empty:
        total = (
            total.set_index("__month")
            .reindex(months)
            .fillna(0.0)
            .reset_index()
            .rename(columns={"index": "__month"})
        )

    # --- stack & legend ordering (best practice) ---
    cat_tot = g.groupby(cat_col, as_index=False)["__value"].sum().sort_values("__value", ascending=False)
    dom_stack = cat_tot[cat_col].astype(str).tolist()

    # keep 'Other' on TOP (last), if present
    other_vals = [c for c in dom_stack if _is_other_label(c)]
    if other_vals:
        dom_stack = [c for c in dom_stack if not _is_other_label(c)] + [other_vals[0]]

    legend_order = list(dom_stack)
    order_map = {c: i for i, c in enumerate(dom_stack)}  # 0 = largest -> bottom
    g["__order"] = g[cat_col].astype(str).map(order_map).fillna(len(order_map)).astype(int)
    total = total.copy()
    # dashed line: keep a bit above the stacked area
    total["total_line"] = total["total"] * 1.02

    scale_base = _build_fixed_color_scale(dom_stack, top_k=min(int(top_k or 10), len(dom_stack)))
    scale = alt.Scale(domain=list(reversed(scale_base.domain)), range=list(reversed(scale_base.range)))

    dim_title = _dimension_axis_title(cat_col)

    area = (
        alt.Chart(g)
        .mark_area(opacity=0.85, strokeWidth=0)
        .encode(
            x=alt.X(
                "__month:T",
                title="Miesiąc",
                axis=alt.Axis(format="%b %Y", labelAngle=45, labelAlign="left", labelBaseline="middle", tickCount={"interval":"month","step":1}, labelOverlap=False),
            ),
            y=alt.Y("__value:Q", title="Wartość sprzedaży (suma)"),
            color=alt.Color(
                f"{cat_col}:N",
                scale=scale,
                title=dim_title,
            ),
            order=alt.Order("__order:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("__month:T", title="Miesiąc", format="%b %Y"),
                alt.Tooltip(f"{cat_col}:N", title=dim_title),
                alt.Tooltip("__value:Q", title="Wartość", format=",.0f"),
            ],
        )
        .properties(height=max(int(CHART_BLOCK_HEIGHT * 1.35), 420))
    )

    line = (
        alt.Chart(total)
        .mark_line(color="#6e6e6e", strokeDash=[4, 4], strokeWidth=2)
        .encode(
            x=alt.X("__month:T"),
            y=alt.Y("total_line:Q", title=""),
            tooltip=[
                alt.Tooltip("__month:T", title="Miesiąc", format="%b %Y"),
                alt.Tooltip("total:Q", title="TOTAL", format=",.0f"),
            ],
        )
    )

    # in-chart label for TOTAL (left side, inside the plot) — keep inside plot and point DOWN to the dashed line
    _label_df = total.head(1).copy()
    try:
        _y_pad = float(total["total"].max()) * 0.04
    except Exception:
        _y_pad = 0.0
    _label_df["y_label"] = _label_df["total_line"] + _y_pad
    _label_df["label"] = "TOTAL"
    _label_df["arrow"] = "↓"

    label_txt = (
        alt.Chart(_label_df)
        .mark_text(align="left", baseline="bottom", dx=6, dy=-2, color="#6e6e6e")
        .encode(x=alt.X("__month:T"), y=alt.Y("y_label:Q"), text=alt.Text("label:N"))
    )
    label_arrow = (
        alt.Chart(_label_df)
        .mark_text(align="left", baseline="top", dx=16, dy=0, color="#6e6e6e")
        .encode(x=alt.X("__month:T"), y=alt.Y("y_label:Q"), text=alt.Text("arrow:N"))
    )

    return alt.layer(area, line, label_txt, label_arrow)

def _top_categories(df_time: pd.DataFrame, cat_col: str, top_n: int = 10) -> List[str]:
    s = df_time.groupby(cat_col)["__value"].sum().sort_values(ascending=False)
    dom = [c for c in s.index.astype(str).tolist() if not _is_other_label(c)]
    return dom[:max(1, top_n)]

def _apply_topn_other(df_time: pd.DataFrame, cat_col: str, top_n: int, include_other: bool) -> pd.DataFrame:
    """Return df_time filtered to Top-N categories, with optional 'Other' bucket.

    Rules:
    - If df_time ALREADY contains an 'Other' label:
        * include_other=True  -> keep it (do NOT re-bucket)
        * include_other=False -> drop it
    - If df_time does NOT contain 'Other':
        * include_other=True  -> create 'Other' as sum of the rest per month
        * include_other=False -> keep ONLY Top-N (drop the rest)

    This prevents the regression where a pre-existing 'Other' was silently removed.
    """
    if df_time is None or df_time.empty or not cat_col:
        return df_time

    s_cat = df_time[cat_col].astype(str)
    has_other = bool(s_cat.map(_is_other_label).any())

    top = _top_categories(df_time, cat_col, top_n=max(1, int(top_n or 10)))
    if not top:
        # if we cannot infer top, at least drop/keep explicit Other depending on the checkbox
        if include_other:
            return df_time.copy()
        return df_time[~s_cat.map(_is_other_label)].copy()

    if include_other and has_other:
        # df_time is already bucketed upstream: keep Top-N + existing Other
        keep = df_time[cat_col].isin(top) | s_cat.map(_is_other_label)
        return df_time[keep].copy()

    df_top = df_time[df_time[cat_col].isin(top)].copy()

    if include_other:
        df_rest = df_time[~df_time[cat_col].isin(top)].copy()
        # If we are building 'Other' ourselves, avoid double-counting any explicit 'Other'
        df_rest = df_rest[~df_rest[cat_col].astype(str).map(_is_other_label)]
        if not df_rest.empty:
            other = (
                df_rest.groupby("__month", as_index=False)["__value"].sum()
                .assign(**{cat_col: "Other"})
            )
            df_top = pd.concat([df_top, other], ignore_index=True)
        return df_top

    # include_other=False: keep ONLY Top-N and drop explicit Other labels if present
    df_top = df_top[~df_top[cat_col].astype(str).map(_is_other_label)]
    return df_top

def _block1_mix_share_topN(
    df_time: pd.DataFrame,
    cat_col: str,
    top_n: int = 10,
    include_other: bool = False,
    df_time_total: pd.DataFrame | None = None,
) -> Tuple[alt.Chart, Dict[str, Any]]:
    df_time_display = df_time
    df_time_total = df_time if df_time_total is None else df_time_total
    """
    Udziały (share) liczone ZAWSZE względem całkowitej sprzedaży w danym miesiącu (w tym "Other"/reszta).
    Na wykresie pokazujemy:
      - TopN kategorii (TopN wybierane bez "Other")
      - + opcjonalnie warstwę "Other" jako resztę (wtedy suma = 100%)
    """
    if df_time_total.empty:
        return alt.Chart(pd.DataFrame()), {"top_categories": [], "note": "no_data"}

    # --- Denominator: TOTAL across ALL categories (incl. explicit Other if present) ---
    totals_all = (
        df_time_total.groupby("__month", as_index=False)["__value"]
        .sum()
        .rename(columns={"__value": "__total_all"})
    )
    other_label = _infer_other_label(df_time_total, cat_col)
    explicit_other = other_label in set(df_time_total[cat_col].astype(str))


    # --- TopN selection excluding explicit Other labels ---
    df_rank = df_time_total.copy()
    df_rank = df_rank[~df_rank[cat_col].astype(str).apply(_is_other_label)]
    top = _top_categories(df_rank, cat_col, top_n=top_n)

    # --- Numerator values for TopN (from TOTAL data) ---
    g_top = (
        df_time_total[df_time_total[cat_col].astype(str).isin([str(x) for x in top])]
        .groupby(["__month", cat_col], as_index=False)["__value"]
        .sum()
    )

    if include_other:
        if explicit_other:
            g_other = (
                df_time_total[df_time_total[cat_col].astype(str) == other_label]
                .groupby(["__month", cat_col], as_index=False)["__value"]
                .sum()
            )
            g = pd.concat([g_top, g_other], ignore_index=True)
            dom = list(top)
            if other_label not in dom:
                dom = dom + [other_label]
        else:
            # residual 'Other' = total_all - sum(TopN) per month
            sum_top = (
                g_top.groupby("__month", as_index=False)["__value"]
                .sum()
                .rename(columns={"__value": "__sum_top"})
            )
            other = totals_all.merge(sum_top, on="__month", how="left")
            other["__sum_top"] = other["__sum_top"].fillna(0.0)
            other["__value"] = (other["__total_all"] - other["__sum_top"]).clip(lower=0.0)
            other[cat_col] = other_label
            g = pd.concat([g_top, other[["__month", cat_col, "__value"]]], ignore_index=True)
            dom = list(top) + [other_label]
    else:
        g = g_top
        dom = list(top)
    # --- Share vs TOTAL (always) ---
    m = g.merge(totals_all, on="__month", how="left")
    m["share"] = np.where(m["__total_all"] > 0, m["__value"] / m["__total_all"], 0.0)


    # Y scale: if we hide 'Other', show only TopN share vs total (sum<100%), so auto-scale to max.
    if include_other:
        y_max = 1.0
    else:
        _sum_share = m.groupby("__month", as_index=False)["share"].sum()["share"].max()
        y_max = float(_sum_share) if _sum_share is not None else 1.0
        y_max = max(min(y_max, 1.0), 0.0)
        # avoid degenerate 0..0 domain
        if y_max <= 0.0:
            y_max = 1.0

    # Ensure complete month x category grid (stable stacking)
    if not m.empty:
        months = pd.date_range(df_time_total["__month"].min(), df_time_total["__month"].max(), freq="MS")
        cats = [str(x) for x in dom]
        idx = pd.MultiIndex.from_product([months, cats], names=["__month", cat_col])
        m = (
            m.set_index(["__month", cat_col])[["__value", "__total_all", "share"]]
            .reindex(idx)
            .fillna({"__value": 0.0, "share": 0.0})
            .reset_index()
        )

    # Order bottom→top = dom; legend we display reversed to match previous UX
    order_map = {str(v): i for i, v in enumerate([str(x) for x in dom])}
    m["__order"] = m[cat_col].astype(str).map(order_map).fillna(10_000).astype(int)
    # Keep color mapping stable (TopK colored), but reverse legend order to match stack layers.
    colors_by_cat = _build_fixed_color_map(dom, top_k=min(5, len(dom)))
    legend_dom = list(reversed(dom))
    scale = alt.Scale(domain=legend_dom, range=[colors_by_cat[c] for c in legend_dom])
    subtitle = (
        "Stacked 100%: TopN + 'Other' (udział vs całość)."
        if include_other
        else "Udział TopN względem całości (reszta = 'Other' / pozostałe kategorie)."
    )


    # Y-scale: with 'Other' hidden, auto-scale to max total share of TopN (vs TOTAL) for readability.
    if include_other:
        y_scale = alt.Scale(domain=[0, 1])
    else:
        topn_sum = float(m.groupby('__month', as_index=False)['share'].sum()['share'].max()) if not m.empty else 0.0
        topn_sum = max(0.0, min(1.0, topn_sum))
        y_scale = alt.Scale(domain=[0, min(1.0, topn_sum * 1.05 if topn_sum > 0 else 1.0)])

    dim_title = _dimension_axis_title(cat_col)

    ch = (
        alt.Chart(m)
        .mark_area(opacity=0.85, strokeWidth=0)
        .encode(
            x=alt.X(
                "__month:T",
                title="Miesiąc",
                axis=alt.Axis(
                    format="%b %Y",
                    labelAngle=45,
                    labelAlign="left",
                    labelBaseline="middle",
                    tickCount={"interval": "month", "step": 1},
                    labelOverlap=False,
                ),
            ),
            y=alt.Y(
                "share:Q",
                axis=alt.Axis(format="%", title="Udział sprzedaży (%)"),
                scale=y_scale,
                stack="zero",
            ),
            color=alt.Color(
                f"{cat_col}:N",
                scale=scale,
                legend=alt.Legend(title=dim_title, symbolType="circle"),
            ),
            order=alt.Order("__order:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("__month:T", title="Miesiąc", format="%b %Y"),
                alt.Tooltip(f"{cat_col}:N", title=dim_title),
                alt.Tooltip("share:Q", title="Udział", format=".1%"),
                alt.Tooltip("__value:Q", title="Wartość", format=",.0f"),
            ],
        )
        .properties(height=max(int(CHART_BLOCK_HEIGHT * 1.35), 420))
    )
    # --- Top-N share diagnostics (needed by ET templates)
    try:
        df_base2 = df_time_total if (df_time_total is not None and len(df_time_total) > 0) else df_time_display
        df_base2 = df_base2.copy()
        df_base2["__month"] = pd.to_datetime(df_base2["__month"])
        m_sorted = df_base2["__month"].sort_values()
        m_start = m_sorted.min()
        m_end = m_sorted.max()

        total_by_m = df_base2.groupby("__month", as_index=True)["__value"].sum()
        top_set = [str(x) for x in top]
        top_by_m = df_base2[df_base2[cat_col].astype(str).isin(top_set)].groupby("__month", as_index=True)["__value"].sum()

        def _safe_share(num, den):
            return float(num) / float(den) if (den is not None and float(den) > 0) else 0.0

        topn_start_share = _safe_share(top_by_m.get(m_start, 0.0), total_by_m.get(m_start, 0.0))
        topn_end_share = _safe_share(top_by_m.get(m_end, 0.0), total_by_m.get(m_end, 0.0))
        topn_delta_pp = (topn_end_share - topn_start_share) * 100.0
    except Exception:
        topn_start_share = 0.0
        topn_end_share = 0.0
        topn_delta_pp = 0.0

    top1_category = str(top[0]) if top else None
    top1_start_share = _safe_share(
        df_base2[(df_base2["__month"] == m_start) & (df_base2[cat_col].astype(str) == str(top1_category))]["__value"].sum() if top1_category else 0.0,
        total_by_m.get(m_start, 0.0),
    ) if top1_category else 0.0
    top1_end_share = _safe_share(
        df_base2[(df_base2["__month"] == m_end) & (df_base2[cat_col].astype(str) == str(top1_category))]["__value"].sum() if top1_category else 0.0,
        total_by_m.get(m_end, 0.0),
    ) if top1_category else 0.0
    top1_delta_pp = (top1_end_share - top1_start_share) * 100.0

    stats = {
        "top_categories": [str(x) for x in top],
        "top_n": int(top_n),
        "topN": int(top_n),
        "top_names": [str(x) for x in top],
        "top1_dimension_value": top1_category,
        "leader_dimension_value": top1_category,
        "primary_dimension_value": top1_category,
        "top1_category": top1_category,
        "top1_start_share": float(top1_start_share),
        "top1_end_share": float(top1_end_share),
        "top1_start_pct": float(top1_start_share * 100.0),
        "top1_end_pct": float(top1_end_share * 100.0),
        "top1_delta_pp": float(top1_delta_pp),
        "topn_start_share": float(topn_start_share),
        "topn_end_share": float(topn_end_share),
        "topn_delta_pp": float(topn_delta_pp),
        "topN_start_share": float(topn_start_share),
        "topN_end_share": float(topn_end_share),
        "topN_start_pct": float(topn_start_share * 100.0),
        "topN_end_pct": float(topn_end_share * 100.0),
        "topN_delta_pp": float(topn_delta_pp),
        "topN_share_start_pct": float(topn_start_share * 100.0),
        "topN_share_end_pct": float(topn_end_share * 100.0),
        "topN_share_delta_pp": float(topn_delta_pp),
        "include_other": bool(include_other),
        "share_denominator": "total_all_including_other",
    }
    return ch, stats

def _audit_share_table_compare_top3(
    df_time: pd.DataFrame,
    cat_col: str,
    top: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return a month-by-month comparison for Top-3 categories:
    - share_topN: value / (sum of TopN values that month)
    - share_total: value / (sum of ALL categories that month, incl. explicit Other)
    Window is inclusive and clamped to available months.
    """
    if df_time.empty:
        return pd.DataFrame()

    months = pd.to_datetime(df_time["__month"]).sort_values()
    mmin, mmax = months.min(), months.max()
    start = max(pd.Timestamp(start), pd.Timestamp(mmin))
    end = min(pd.Timestamp(end), pd.Timestamp(mmax))

    win = df_time[(df_time["__month"] >= start) & (df_time["__month"] <= end)].copy()
    if win.empty:
        return pd.DataFrame()

    # Candidate set = TopN list (strings), but keep only present categories
    top_set = [str(x) for x in top if str(x) in set(win[cat_col].astype(str))]
    if not top_set:
        return pd.DataFrame()

    # Identify Top-3 by total value in the window (within TopN candidates)
    sums = (
        win[win[cat_col].astype(str).isin(top_set)]
        .groupby(cat_col, as_index=True)["__value"]
        .sum()
        .sort_values(ascending=False)
    )
    top3 = [str(x) for x in sums.head(3).index.tolist()]
    if not top3:
        return pd.DataFrame()

    # Totals by month
    total_all = win.groupby("__month", as_index=False)["__value"].sum().rename(columns={"__value": "__total_all"})
    total_topN = (
        win[win[cat_col].astype(str).isin(top_set)]
        .groupby("__month", as_index=False)["__value"]
        .sum()
        .rename(columns={"__value": "__total_top"})
    )

    g = (
        win[win[cat_col].astype(str).isin(top3)]
        .groupby(["__month", cat_col], as_index=False)["__value"]
        .sum()
    )
    m = g.merge(total_all, on="__month", how="left").merge(total_topN, on="__month", how="left")
    m["share_total"] = np.where(m["__total_all"] > 0, m["__value"] / m["__total_all"], 0.0)
    m["share_topN"] = np.where(m["__total_top"] > 0, m["__value"] / m["__total_top"], 0.0)

    # Wide comparison table: month rows; for each cat -> (vs Total, vs TopN)
    m["month"] = pd.to_datetime(m["__month"]).dt.strftime("%Y-%m")
    wide_total = m.pivot_table(index="month", columns=cat_col, values="share_total", aggfunc="sum")
    wide_top = m.pivot_table(index="month", columns=cat_col, values="share_topN", aggfunc="sum")
    # Build final df with paired columns
    out = pd.DataFrame(index=wide_total.index)
    for c in top3:
        out[f"{c} — share vs Total"] = wide_total.get(c)
        out[f"{c} — share vs TopN"] = wide_top.get(c)
    out = out.reset_index().rename(columns={"month": "Miesiąc"})

    # Format as percent-friendly floats (keep raw, Streamlit can format)
    return out

def _block2_winners_losers(
    df_time: pd.DataFrame,
    cat_col: str,
    top_n: int = 10,
    include_other: bool = False,
    df_time_total: Optional[pd.DataFrame] = None,
) -> Tuple[alt.Chart, Dict[str, Any]]:
    """Trend lines of category share over time.

    Truth model:
    - share is ALWAYS computed vs TOTAL sales (incl. all categories)
    - optional "Other" line = remainder (TOTAL - sum(TopN)) per month
    - robust to datasets with an explicit "Other" category (it will be part of remainder)
    """
    top_k = max(1, int(top_n))
    df_time_total = df_time if df_time_total is None else df_time_total

    # Totals over ALL categories (incl. any explicit "Other")
    totals_all = (
        df_time_total.groupby(["__month"], as_index=False)["__value"]
        .sum()
        .rename(columns={"__value": "__total_all"})
    )

    # Pick TopN across the whole period (exclude label that we reserve for "Other" line)
    other_label = _infer_other_label(df_time_total)
    df_for_top = df_time_total.copy()
    df_for_top = df_for_top[df_for_top[cat_col].astype(str) != str(other_label)]
    top = _top_categories(df_for_top, cat_col, top_n=top_k)

    # Aggregate TopN series
    g_top = (
        df_time_total[df_time_total[cat_col].astype(str).isin([str(x) for x in top])]
        .groupby(["__month", cat_col], as_index=False)["__value"]
        .sum()
    )

    # Ensure complete month x category grid (avoid implicit interpolation artefacts)
    if not g_top.empty:
        months = pd.date_range(df_time_total["__month"].min(), df_time_total["__month"].max(), freq="MS")
        cats = [str(x) for x in top]
        idx = pd.MultiIndex.from_product([months, cats], names=["__month", cat_col])
        g_top = (
            g_top.set_index(["__month", cat_col])["__value"]
            .reindex(idx, fill_value=0.0)
            .reset_index()
        )

    dom: List[str] = [str(x) for x in top]
    g = g_top

    if include_other:
        # Other = TOTAL - sum(TopN) per month
        sum_top = (
            g_top.groupby(["__month"], as_index=False)["__value"]
            .sum()
            .rename(columns={"__value": "__sum_top"})
        )
        other = totals_all.merge(sum_top, on="__month", how="left")
        other["__sum_top"] = other["__sum_top"].fillna(0.0)
        other["__value"] = (other["__total_all"] - other["__sum_top"]).clip(lower=0.0)
        other[cat_col] = other_label
        other = other[["__month", cat_col, "__value"]]
        g = pd.concat([g_top, other], ignore_index=True)
        dom = dom + [str(other_label)]

    # Share vs TOTAL (always)
    m = g.merge(totals_all, on="__month", how="left")
    m["share"] = np.where(m["__total_all"] > 0, m["__value"] / m["__total_all"], 0.0)

    # Y-axis: when "Other" is off, zoom to TopN sum for readability (same logic as share chart)
    if include_other:
        y_scale = alt.Scale(domain=(0, 1.0))
    else:
        y_max = float(m["share"].max()) if len(m) else 0.0
        y_scale = alt.Scale(domain=(0, min(1.0, max(0.01, y_max * 1.05))))

    # Deterministic colors (Top-5 colored, rest greys, Other = darkest grey)
    colors_by_cat = _build_fixed_color_map(dom, top_k=5)
    if include_other:
        # Use a fixed, very dark grey for the residual "Other" band
        colors_by_cat[str(other_label)] = "#6b6b6b"

    legend_dom = list(reversed(dom))  # legend top->bottom; bottom should be the "base" category
    color_scale = alt.Scale(domain=legend_dom, range=[colors_by_cat[c] for c in legend_dom])


    # Axis/legend styling to match the reference share chart ("🔎 Jak wygląda dynamika udziałów w czasie?")
    x_axis = alt.Axis(
        format="%b %Y",
        labelAngle=45,
        labelAlign="left",
        labelBaseline="middle",
        tickCount={"interval": "month", "step": 1},
        labelOverlap=False,
    )
    if include_other:
        # With 'Other' visible we are on a 0–100% scale; show 10pp grid (0%,10%,...,100%)
        y_axis = alt.Axis(format=".0%", values=[i / 10 for i in range(0, 11)])
    else:
        y_axis = alt.Axis(format=".0%", tickCount=6)

    # Legend: show thick line swatches instead of point/ring symbols
    dim_title = _dimension_axis_title(cat_col)
    legend = alt.Legend(title=dim_title, symbolType="stroke", symbolStrokeWidth=4, symbolSize=200)
    chart = (
        alt.Chart(m)
        .mark_line()
        .encode(
            x=alt.X(
                "__month:T",
                title="Miesiąc",
                axis=x_axis,
            ),
            y=alt.Y(
                "share:Q",
                title="Udział sprzedaży (%)",
                axis=y_axis,
                scale=y_scale,
            ),
            color=alt.Color(f"{cat_col}:N", scale=color_scale, legend=legend),
            tooltip=[
                alt.Tooltip("__month:T", title="Miesiąc", format="%b %Y"),
                alt.Tooltip(f"{cat_col}:N", title=dim_title),
                alt.Tooltip("share:Q", title="Udział", format=".2%"),
            ],
        )
        .properties(height=max(int(CHART_BLOCK_HEIGHT * 1.35), 420))
    )

    # --- ET anchors (to satisfy validator: >=2 numbers in sentence 1) ---
    _m_min = m["__month"].min()
    _m_max = m["__month"].max()
    _winner = None
    _loser = None
    _winner_delta_pp = float("nan")
    _loser_delta_pp = float("nan")
    try:
        _endpoints = (
            m.loc[m["__month"].isin([_m_min, _m_max]), [cat_col, "__month", "share"]]
            .pivot_table(index=cat_col, columns="__month", values="share", aggfunc="sum")
        )
        if (_m_min in _endpoints.columns) and (_m_max in _endpoints.columns):
            _delta_pp = (_endpoints[_m_max] - _endpoints[_m_min]) * 100.0
            if len(_delta_pp) > 0:
                _winner = str(_delta_pp.idxmax())
                _loser = str(_delta_pp.idxmin())
                _winner_delta_pp = float(_delta_pp.loc[_delta_pp.idxmax()])
                _loser_delta_pp = float(_delta_pp.loc[_delta_pp.idxmin()])
    except Exception:
        pass
    _winner_non_other = None
    _loser_non_other = None
    _winner_non_other_delta_pp = None
    _loser_non_other_delta_pp = None
    _delta_rows = []
    try:
        if len(_delta_pp) > 0:
            _delta_rows = [{"category": str(idx), "delta_pp": float(val)} for idx, val in _delta_pp.sort_values(ascending=False).items()]
            _non_other = _delta_pp[~_delta_pp.index.astype(str).map(_is_other_label)]
            _non_other_nz = _non_other[_non_other.abs() > 1e-9]
            _base = _non_other_nz if len(_non_other_nz) else _non_other
            if len(_base) > 0:
                _winner_non_other = str(_base.idxmax())
                _loser_non_other = str(_base.idxmin())
                _winner_non_other_delta_pp = float(_base.loc[_winner_non_other])
                _loser_non_other_delta_pp = float(_base.loc[_loser_non_other])
    except Exception:
        pass

    stats = {
        "top_categories": [str(x) for x in top],
        "other_label": str(other_label),
        "include_other": bool(include_other),
        "share_basis": "total",
        "start_month": str(_m_min) if _m_min is not None else None,
        "end_month": str(_m_max) if _m_max is not None else None,
        "winner_dimension_value": _winner,
        "winner_category": _winner,
        "winner_delta_pp": _winner_delta_pp,
        "loser_dimension_value": _loser,
        "loser_category": _loser,
        "loser_delta_pp": _loser_delta_pp,
        "winner_dimension_value_non_other": _winner_non_other,
        "winner_category_non_other": _winner_non_other,
        "winner_delta_pp_non_other": _winner_non_other_delta_pp,
        "loser_dimension_value_non_other": _loser_non_other,
        "loser_category_non_other": _loser_non_other,
        "loser_delta_pp_non_other": _loser_non_other_delta_pp,
        "delta_rows": _delta_rows,
    }
    return chart, stats

def _block3_seasonality(
    df_time: pd.DataFrame,
    cat_col: str,
    top_n: int = 10,
) -> Tuple[alt.Chart | None, pd.DataFrame, Dict[str, Any]]:
    """
    🧩 Heatmapa czystej sezonowości (seasonal z STL) + waga sezonowości (0–1)
    🧾 Scorecard metryk sezonowości (porównanie kategorii)

    Zasada: nie ruszamy globalnego pipeline; operujemy na df_time, które już jest miesięczne (__month).
    Analiza sezonowości jest liczona na WARTOŚCI SPRZEDAŻY (absolute value), nie na udziałach.
    """
    import numpy as np
    import pandas as pd

    try:
        from statsmodels.tsa.seasonal import STL
    except Exception:
        STL = None  # no STL available

    stats: Dict[str, Any] = {"topN": 0}

    # ---- guardrails
    if df_time is None or df_time.empty or (cat_col not in df_time.columns):
        return None, pd.DataFrame(), stats
    if "__month" not in df_time.columns or "__value" not in df_time.columns:
        return None, pd.DataFrame(), stats

    df_time = df_time.copy()
    df_time["__month"] = pd.to_datetime(df_time["__month"], errors="coerce")
    df_time = df_time.dropna(subset=["__month"])
    if df_time.empty:
        return None, pd.DataFrame(), stats

    # ---- TOP-N po wartości sprzedaży w CAŁYM okresie
    total_by_cat = (
        df_time.groupby(cat_col, as_index=False)["__value"]
        .sum()
        .rename(columns={cat_col: "Category", "__value": "total_value"})
        .sort_values("total_value", ascending=False)
    )
    selected = total_by_cat["Category"].astype(str).head(int(top_n)).tolist()
    if not selected:
        return None, pd.DataFrame(), stats

    stats["topN"] = int(len(selected))
    stats["topN_rule"] = "total_value_over_whole_period"

    # ---- miesięczna agregacja wartości per kategoria
    m = (
        df_time[df_time[cat_col].astype(str).isin(selected)]
        .groupby(["__month", cat_col], as_index=False)["__value"]
        .sum()
        .rename(columns={cat_col: "Category", "__value": "value"})
    )

    # pełny indeks miesięcy do STL (bez ingerencji w źródłowe dane)
    full_idx = pd.date_range(m["__month"].min(), m["__month"].max(), freq="MS")
    min_periods = 18  # robust: 1.5 roku

    def _seasonality_metrics_for_cat(cat: str):
        s = (
            m.loc[m["Category"].astype(str) == str(cat), ["__month", "value"]]
            .set_index("__month")
            .sort_index()
            .reindex(full_idx)
        )

        y = pd.to_numeric(s["value"], errors="coerce")
        if y.isna().all():
            return None, {"skip": True, "skip_reason": "Brak serii wartości."}

        # STL-safe (bez dopisywania sprzedaży poza zakresem):
        # - nie zmieniamy danych źródłowych
        # - uzupełniamy tylko luki *wewnątrz* obserwowanego zakresu (interpolacja czasowa)
        # - nie ekstrapolujemy przed pierwszą ani po ostatniej obserwacji
        nonnull = y.dropna()
        if nonnull.empty:
            return None, {"skip": True, "skip_reason": "Brak serii wartości."}

        first_idx = nonnull.index[0]
        last_idx = nonnull.index[-1]

        y_trim = y.loc[first_idx:last_idx].astype(float)
        # Interpolacja tylko w środku luk; końce pozostają nietknięte (tu i tak są obserwacje)
        y_trim = y_trim.interpolate(method="time", limit_area="inside")

        # Gdyby zostały NaN (np. pojedyncze skrajne braki), domykamy wyłącznie wewnątrz zakresu:
        y_trim = y_trim.ffill().bfill()

        var_y = float(np.nanvar(y_trim.values))
        if len(y_trim) < 1:
            return None, {"skip": True, "skip_reason": "Za krótki lub stały szereg."}

        decomp_method = "stl"
        if STL is not None and len(y_trim) >= min_periods and var_y > 0.0:
            try:
                res = STL(y_trim, period=12, robust=True).fit()
                seasonal_trim = pd.Series(res.seasonal, index=y_trim.index)
                trend_trim = pd.Series(res.trend, index=y_trim.index)
                resid_trim = pd.Series(res.resid, index=y_trim.index)
            except Exception:
                decomp_method = "detrended_rolling_fallback"
        else:
            decomp_method = "detrended_rolling_fallback"

        if decomp_method == "detrended_rolling_fallback":
            # Short ranges on Streamlit Cloud often have only one sales year, so STL
            # cannot estimate a stable 12-month component. Keep the heatmap useful by
            # showing de-trended monthly deviation instead of hiding the whole block.
            vals = y_trim.to_numpy(dtype=float)
            x = np.arange(len(vals), dtype=float)
            ok = np.isfinite(vals)
            if int(ok.sum()) >= 2:
                coeff = np.polyfit(x[ok], vals[ok], 1)
                trend_vals = np.polyval(coeff, x)
            else:
                trend_vals = np.full(len(vals), float(np.nanmean(vals)))

            trend_trim = pd.Series(trend_vals, index=y_trim.index)
            deviation = pd.Series(vals - trend_vals, index=y_trim.index)
            win = 3 if len(deviation) < min_periods else 5
            seasonal_trim = deviation.rolling(window=win, center=True, min_periods=1).mean()
            seasonal_trim = seasonal_trim - float(np.nanmean(seasonal_trim.values))
            resid_trim = deviation - seasonal_trim

        # Dla heatmapy trzymamy wspólną oś czasu, ale bez ekstrapolacji:
        seasonal = seasonal_trim.reindex(full_idx)
        trend = trend_trim.reindex(full_idx)
        resid = resid_trim.reindex(full_idx)

        # --- Variance audit (for Strength diagnostics)
        var_original = float(np.nanvar(y.values))
        var_seasonal = float(np.nanvar(seasonal.values))
        var_resid = float(np.nanvar(resid.values))

        # Strength (current): share of variance explained by seasonal vs (seasonal + resid)
        denom = float(np.nanvar((seasonal + resid).values))
        strength = 0.0 if denom == 0.0 else float(max(0.0, min(1.0, 1.0 - (var_resid / denom))))

        # Strength_alt (diagnostic): share of variance explained by seasonal vs ORIGINAL (incl. trend)
        strength_alt = 0.0 if var_original == 0.0 else float(max(0.0, min(1.0, 1.0 - (var_resid / var_original))))
        amp = float(np.nanmax(seasonal.values) - np.nanmin(seasonal.values))
        noise_cover = float(np.nanstd(resid.values) / (np.nanstd(seasonal.values) + 1e-9))

        # stability slope: trend nachylenia rocznej amplitudy sezonowości
        df_year = pd.DataFrame({"period": full_idx, "seasonal": seasonal.values})
        df_year["_year"] = pd.to_datetime(df_year["period"]).dt.year
        amp_year = df_year.groupby("_year")["seasonal"].agg(lambda x: float(np.nanmax(x) - np.nanmin(x))).reset_index()
        if len(amp_year) >= 2:
            x = np.arange(len(amp_year), dtype=float)
            y_amp = amp_year["seasonal"].values.astype(float)
            slope = float(np.polyfit(x, y_amp, 1)[0])
        else:
            slope = 0.0

        # peak drift (mies.): odchylenie std miesiąca szczytu per rok
        df_peak = pd.DataFrame({"period": full_idx, "seasonal": seasonal.values})
        df_peak["_year"] = pd.to_datetime(df_peak["period"]).dt.year
        df_peak["_month"] = pd.to_datetime(df_peak["period"]).dt.month
        if len(df_peak) > 0:
            def _peak_month_for_year(d: pd.DataFrame):
                d2 = d.dropna(subset=["seasonal"])
                if d2.empty:
                    return np.nan
                return float(d2.loc[d2["seasonal"].idxmax(), "_month"])
            peak_months = (
                df_peak.groupby("_year")
                .apply(_peak_month_for_year)
                .dropna()
                .values
            )
        else:
            peak_months = []
        peak_drift = float(np.std(np.array(peak_months, dtype=float))) if len(peak_months) >= 2 else 0.0

        panel_df = pd.DataFrame(
            {
                "period": full_idx,
                "Category": str(cat),
                "seasonal": seasonal.values,
                "trend": trend.values,
                "noise": resid.values,
                "original": y.values,
            }
        )
        metrics = {
            "Category": str(cat),
            "seasonality_strength": strength,
            "seasonality_strength_alt": strength_alt,
            "var_original": var_original,
            "var_seasonal": var_seasonal,
            "var_residual": var_resid,
            "seasonality_amplitude": amp,
            "stability_slope": slope,
            "peak_drift": peak_drift,
            "noise_cover_ratio": noise_cover,
            "seasonality_method": decomp_method,
        }
        return panel_df, metrics

    panels_all: list[pd.DataFrame] = []
    metrics_all: list[Dict[str, Any]] = []

    for c in selected:
        panel_df, met = _seasonality_metrics_for_cat(str(c))
        if isinstance(met, dict) and met.get("skip"):
            continue
        if panel_df is not None:
            panels_all.append(panel_df)
            metrics_all.append(met)

    if not metrics_all:
        return None, pd.DataFrame(), {"topN": int(len(selected)), "reason": "Brak kategorii z poprawnym szeregiem do STL."}

    panels_df = pd.concat(panels_all, ignore_index=True)
    scorecard_df = pd.DataFrame(metrics_all)


    # ---- DEBUG: audyt Strength (wariancje i rozkład) — tylko pod flagą dc_debug
    debug_perf = bool(st.session_state.get("dc_debug", False))
    if debug_perf and scorecard_df is not None and not scorecard_df.empty:
        with st.expander("🔬 Audyt Strength: var(original) / var(seasonal) / var(residual) + rozkład"):
            diag_cols = [
                "Category",
                "seasonality_strength",
                "seasonality_strength_alt",
                "var_original",
                "var_seasonal",
                "var_residual",
                "stability_slope",
                "noise_cover_ratio",
            ]
            diag_cols = [c for c in diag_cols if c in scorecard_df.columns]
            diag = scorecard_df[diag_cols].copy()
            # ratios (safe)
            if "var_original" in diag.columns and "var_seasonal" in diag.columns:
                diag["ratio_var_seasonal/original"] = diag["var_seasonal"] / diag["var_original"].replace({0.0: np.nan})
            if "var_original" in diag.columns and "var_residual" in diag.columns:
                diag["ratio_var_residual/original"] = diag["var_residual"] / diag["var_original"].replace({0.0: np.nan})

            # uporządkuj i zaokrąglij do czytelności
            for c in diag.columns:
                if c != "Category":
                    diag[c] = pd.to_numeric(diag[c], errors="coerce")
            show = diag.sort_values("seasonality_strength", ascending=False, na_position="last")
            st.dataframe(show, width='stretch', hide_index=True)

            # Rozkład Strength (czy ~0.999 to artefakt?)
            def _dist(s: pd.Series) -> dict:
                s = pd.to_numeric(s, errors="coerce").dropna()
                if s.empty:
                    return {}
                return {
                    "count": int(s.shape[0]),
                    "min": float(s.min()),
                    "p05": float(s.quantile(0.05)),
                    "median": float(s.median()),
                    "p95": float(s.quantile(0.95)),
                    "max": float(s.max()),
                    "share_ge_0.99": float((s >= 0.99).mean()),
                    "share_ge_0.999": float((s >= 0.999).mean()),
                }

            dist_rows = []
            if "seasonality_strength" in scorecard_df.columns:
                dist_rows.append({"metric": "Strength (current)", **_dist(scorecard_df["seasonality_strength"])})
            if "seasonality_strength_alt" in scorecard_df.columns:
                dist_rows.append({"metric": "Strength_alt (vs original)", **_dist(scorecard_df["seasonality_strength_alt"])})
            if dist_rows:
                st.markdown("**Rozkład Strength (w TOP10):**")
                st.dataframe(pd.DataFrame(dist_rows), width='stretch', hide_index=True)

            st.caption(
                "Jeśli Strength (current) jest blisko 1.0 dla prawie wszystkich kategorii, a Strength_alt ma większą rozdzielczość — "
                "to znak, że bieżąca definicja Strength może być 'zbyt łatwa' (denom=var(seasonal+residual) pomija trend). "
                "Ten blok jest tylko audytem — definicję Strength zmienimy dopiero po Twojej akceptacji."
            )
    # ---- twarda kolejność (TOPN) dla A1 i B1
    sort_order = [str(c) for c in selected]
    stats["sort_order"] = sort_order
    stats["selected"] = sort_order

    def _label(cat: str) -> str:
        return f"{cat} (TOP{len(sort_order)})"

    # ---- heatmap data
    df_hm = panels_df[panels_df["Category"].isin(sort_order)].copy()
    df_hm["CategoryLabel"] = df_hm["Category"].astype(str).map(_label)

    df_hm["period_dt"] = pd.to_datetime(df_hm["period"], errors="coerce")
    df_hm = df_hm.dropna(subset=["period_dt", "Category", "seasonal"]).sort_values(["Category", "period_dt"])
    df_hm["period_iso"] = df_hm["period_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # symetryczny domain kolorów (WINSORYZACJA wizualna, bez zmiany danych):
    # - do heatmapy ustawiamy domain na P95(|seasonal|), żeby pojedyncze ekstremum nie "wypłaszczało" reszty
    # - dane (df_hm['seasonal']) pozostają niezmienione
    v = pd.to_numeric(df_hm["seasonal"], errors="coerce").dropna()
    if v.empty:
        absmax = 0.1
    else:
        abs_raw = float(max(abs(v.min()), abs(v.max())))
        abs_p95 = float(np.nanquantile(np.abs(v.to_numpy(dtype=float)), 0.95))
        absmax = abs_p95 if abs_p95 > 0 else abs_raw
        # awaryjnie
        absmax = absmax if absmax > 0 else 0.1
    stats["heatmap_color_domain"] = {"type": "winsor_p95_abs", "absmax": float(absmax)}


    # ---- waga sezonowości (0–1) na podstawie metryk (normalizacja)
    scw = scorecard_df.copy()
    scw["Category"] = scw["Category"].astype(str)
    scw = scw[scw["Category"].isin(sort_order)].copy()

    def _minmax(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - mn) / (mx - mn)

    strength = pd.to_numeric(scw.get("seasonality_strength"), errors="coerce").fillna(0.0).clip(0, 1)
    amp_n = _minmax(scw.get("seasonality_amplitude"))
    peak_n = _minmax(scw.get("peak_drift"))
    noise_n = _minmax(scw.get("noise_cover_ratio"))
    stab_abs_n = _minmax(pd.to_numeric(scw.get("stability_slope"), errors="coerce").abs())

    # Seasonality Strength is ~1.0 across categories in this dataset -> not differentiating.
    # Recalibration: weight based on amplitude + stability + peak drift + noise (0..1, higher = more decision-relevant seasonality).
    weight = (
    0.35 * amp_n +
    0.25 * (1 - peak_n) +
    0.20 * (1 - noise_n) +
    0.20 * (1 - stab_abs_n)
    ).clip(0, 1)

    scw["seasonality_weight"] = weight
    scw["CategoryLabel"] = scw["Category"].map(_label)
    w_vals = scw[["CategoryLabel", "seasonality_weight"]].to_dict("records")

    vals = df_hm[["period_iso", "Category", "CategoryLabel", "seasonal"]].to_dict("records")

    # ---- P90 ramki per kategoria (abs seasonal)
    # wyliczamy na danych heatmapy, bez ingerencji w metryki STL.
    df_thr = df_hm.copy()
    df_thr["_abs"] = pd.to_numeric(df_thr["seasonal"], errors="coerce").abs()
    thr = (
        df_thr.groupby("Category")["_abs"]
        .quantile(0.90)
        .to_dict()
    )
    for k, t in list(thr.items()):
        if not np.isfinite(t):
            thr[k] = np.nan

    boxed_vals = []
    for rec in vals:
        cat = str(rec["Category"])
        try:
            a = abs(float(rec["seasonal"]))
        except Exception:
            a = np.nan
        t = thr.get(cat, np.nan)
        rec2 = dict(rec)
        rec2["is_peak"] = bool(np.isfinite(a) and np.isfinite(t) and (a >= float(t)) and (float(t) > 0))
        boxed_vals.append(rec2)

    # ---- układ A1: bar + heatmap (board-ready):
    # Cel: komórki możliwie KWADRATOWE + brak zbędnej białej przestrzeni.
    n_periods = int(df_hm.get("period_iso").nunique()) if "period_iso" in df_hm.columns else 1
    n_periods = max(1, n_periods)

    # dobieramy "cell size" w px w zależności od liczby miesięcy
    # (w praktyce: im więcej miesięcy, tym mniejsze komórki)
    cell = int(np.clip(round(980 / n_periods), 18, 30))
    heat_w = int(np.clip(n_periods * cell, 720, 1100))

    # kwadraty: wysokość wiersza ~= szerokość komórki
    row_h = int(cell)
    hm_height = int(np.clip(row_h * len(sort_order), 240, 360))

    # dynamiczne ticki osi X (żeby nie robić "grzebienia")
    if n_periods <= 14:
        tick_step = 1
    elif n_periods <= 26:
        tick_step = 2
    else:
        tick_step = 3

    bar = {
        "data": {"values": w_vals},
        "width": 240,  # ~1.5x vs previous 160
        "height": hm_height,
        "mark": {"type": "bar", "color": "#c7cbd1"},
        "encoding": {
            "y": {
                "field": "CategoryLabel",
                "type": "nominal",
                "sort": [_label(c) for c in sort_order],
                "title": "",
                "axis": {"labels": False, "ticks": False, "domain": False},
            },
            "x": {
                "field": "seasonality_weight",
                "type": "quantitative",
                "title": "Seasonality weight",
                "scale": {"domain": [0, 1], "nice": False},
                "axis": {"format": ".1f", "values": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            },
            "tooltip": [
                {"field": "CategoryLabel", "type": "nominal", "title": "Kategoria"},
                {"field": "seasonality_weight", "type": "quantitative", "title": "Seasonality weight", "format": ".3f"}
            ],
            "color": {"value": "#c7cbd1"},  # jasny szary
        },
    }

    # heatmap base
    heat_base = {
        "data": {"values": boxed_vals},
        "transform": [{"calculate": "toDate(datum.period_iso)", "as": "period_dt"}],
        "width": heat_w,
        "height": hm_height,
        "mark": {"type": "rect", "stroke": "rgba(0,0,0,0.18)", "strokeWidth": 0.6},
        "encoding": {
            "x": {
                "field": "period_dt",
                "type": "temporal",
                "timeUnit": "yearmonth",
                "title": "Miesiąc–rok",
                "axis": {"format": "%b %Y", "labelAngle": 45, "labelOverlap": "greedy", "tickCount": {"interval": "month", "step": tick_step}, "labelLimit": 140},
                "scale": {"type": "band"},
            },
            "y": {
                "field": "CategoryLabel",
                "type": "nominal",
                "title": "",
                "sort": [_label(c) for c in sort_order],
                "axis": {"labelLimit": 500, "labelFontSize": 12},
            },
            "color": {
                "field": "seasonal",
                "type": "quantitative",
                "title": "",
                "scale": {"domain": [-absmax, absmax], "scheme": {"name": "redblue", "extent": [0.08, 0.92]}},
                "legend": {"orient": "right", "gradientThickness": 12, "gradientLength": 220, "format": ",.0f"},
            },
            "tooltip": [
                {"field": "CategoryLabel", "type": "nominal"},
                {"field": "period_dt", "type": "temporal", "title": "Miesiąc", "format": "%b %Y"},
                {"field": "seasonal", "type": "quantitative", "format": ",.0f", "title": "Seasonal"},
            ],
        },
    }

    # overlay ramki na peakach (subtelne)
    heat_peaks = {
        "data": {"values": boxed_vals},
        "transform": [
            {"filter": "datum.is_peak == true"},
            {"calculate": "toDate(datum.period_iso)", "as": "period_dt"},
        ],
        "width": heat_w,
        "height": hm_height,
        "mark": {"type": "rect", "fillOpacity": 0.0, "stroke": "#444", "strokeWidth": 0.7},
        "encoding": {
            "x": {"field": "period_dt", "type": "temporal", "timeUnit": "yearmonth", "scale": {"type": "band"}},
            "y": {"field": "CategoryLabel", "type": "nominal", "sort": [_label(c) for c in sort_order]},
            "tooltip": [
                {"field": "CategoryLabel", "type": "nominal"},
                {"field": "period_dt", "type": "temporal", "title": "Miesiąc", "format": "%b %Y"},
                {"field": "seasonal", "type": "quantitative", "format": ",.0f", "title": "Seasonal"},
            ],
        },
    }

    heatmap = {
        "layer": [heat_base, heat_peaks]
    }

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "hconcat": [bar, heatmap],
        "autosize": {"type": "fit", "contains": "padding"},
"config": {
    "view": {"stroke": "transparent"},
    "concat": {"spacing": 14},
    "axis": {"labelFontSize": 12, "titleFontSize": 12},
    "legend": {"titleFontSize": 12, "labelFontSize": 12, "titleLimit": 1000},
},
    }

    a1 = alt.Chart.from_dict(spec)

    _seasonality_focus_category = None
    _seasonality_focus_weight = None
    _seasonality_focus_amplitude = None
    _seasonality_focus_verdict = None
    _seasonality_focus_share = None
    _seasonality_focus_strength = None

    _seasonality_second_category = None
    _seasonality_second_weight = None
    _seasonality_second_share = None
    _seasonality_second_strength = None
    _seasonality_second_amplitude = None
    _seasonality_second_verdict = None

    _seasonality_metric_col = None
    _seasonality_share_gap = None
    _seasonality_amplitude_gap = None
    _seasonality_top_mode = "amplitude_only"
    _seasonality_rows_top2: list[dict] = []

    try:
        if isinstance(scorecard_df, pd.DataFrame) and len(scorecard_df):
            _sc_pick = scorecard_df.copy()
            _cat_col_sc = "CategoryLabel" if "CategoryLabel" in _sc_pick.columns else ("Category" if "Category" in _sc_pick.columns else None)
            if _cat_col_sc is not None:
                _sc_pick["__cat"] = _sc_pick[_cat_col_sc].astype(str)
                _sc_pick = _sc_pick[~_sc_pick["__cat"].map(_is_other_label)].copy()

            _ver_col = "verdict" if "verdict" in _sc_pick.columns else None

            # ranking column for ET
            if "seasonality_share" in _sc_pick.columns:
                _w_col = "seasonality_share"
            elif "seasonality_weight" in _sc_pick.columns:
                _w_col = "seasonality_weight"
            elif "seasonality_strength" in _sc_pick.columns:
                _w_col = "seasonality_strength"
            elif "seasonality_strength_alt" in _sc_pick.columns:
                _w_col = "seasonality_strength_alt"
            else:
                _w_col = None

            _share_col = "seasonality_share" if "seasonality_share" in _sc_pick.columns else _w_col
            _strength_col = (
                "seasonality_strength"
                if "seasonality_strength" in _sc_pick.columns
                else ("seasonality_strength_alt" if "seasonality_strength_alt" in _sc_pick.columns else _w_col)
            )
            _a_col = "seasonality_amplitude" if "seasonality_amplitude" in _sc_pick.columns else None

            _seasonality_metric_col = "seasonality_share" if _share_col is not None else _w_col

            if len(_sc_pick) and _w_col is not None:
                if _ver_col is not None:
                    _rank_map = {
                        "Sezonowość kalendarzowa": 0,
                        "Sezonowość eventowa": 1,
                        "Niestabilność sezonowości": 2,
                    }
                    _sc_pick["__rank"] = _sc_pick[_ver_col].map(_rank_map).fillna(9)
                else:
                    _sc_pick["__rank"] = 9

                _sc_pick["__w"] = pd.to_numeric(_sc_pick[_w_col], errors="coerce")
                _sc_pick["__share"] = pd.to_numeric(_sc_pick[_share_col], errors="coerce") if _share_col is not None else np.nan
                _sc_pick["__strength"] = pd.to_numeric(_sc_pick[_strength_col], errors="coerce") if _strength_col is not None else np.nan
                _sc_pick["__a"] = pd.to_numeric(_sc_pick[_a_col], errors="coerce") if _a_col is not None else np.nan

                _sc_pick = _sc_pick.sort_values(
                    ["__rank", "__w", "__a"],
                    ascending=[True, False, False],
                ).reset_index(drop=True)

                if len(_sc_pick):
                    _row = _sc_pick.iloc[0]
                    _seasonality_focus_category = str(_row.get("__cat") or _row.get(_cat_col_sc) or "")
                    _seasonality_focus_weight = float(_row.get("__w")) if pd.notna(_row.get("__w")) else None
                    _seasonality_focus_share = float(_row.get("__share")) if pd.notna(_row.get("__share")) else _seasonality_focus_weight
                    _seasonality_focus_strength = float(_row.get("__strength")) if pd.notna(_row.get("__strength")) else _seasonality_focus_weight
                    _seasonality_focus_amplitude = float(_row.get("__a")) if pd.notna(_row.get("__a")) else None
                    _seasonality_focus_verdict = str(_row.get(_ver_col) or "") if _ver_col is not None else None

                if len(_sc_pick) > 1:
                    _row2 = _sc_pick.iloc[1]
                    _seasonality_second_category = str(_row2.get("__cat") or _row2.get(_cat_col_sc) or "")
                    _seasonality_second_weight = float(_row2.get("__w")) if pd.notna(_row2.get("__w")) else None
                    _seasonality_second_share = float(_row2.get("__share")) if pd.notna(_row2.get("__share")) else _seasonality_second_weight
                    _seasonality_second_strength = float(_row2.get("__strength")) if pd.notna(_row2.get("__strength")) else _seasonality_second_weight
                    _seasonality_second_amplitude = float(_row2.get("__a")) if pd.notna(_row2.get("__a")) else None
                    _seasonality_second_verdict = str(_row2.get(_ver_col) or "") if _ver_col is not None else None

                for _, _r in _sc_pick.head(2).iterrows():
                    _row_share = float(_r.get("__share")) if pd.notna(_r.get("__share")) else (float(_r.get("__w")) if pd.notna(_r.get("__w")) else None)
                    _row_strength = float(_r.get("__strength")) if pd.notna(_r.get("__strength")) else (float(_r.get("__w")) if pd.notna(_r.get("__w")) else None)
                    _seasonality_rows_top2.append({
                        "category": str(_r.get("__cat") or _r.get(_cat_col_sc) or ""),
                        "share": _row_share,
                        "weight": float(_r.get("__w")) if pd.notna(_r.get("__w")) else None,
                        "strength": _row_strength,
                        "amplitude": float(_r.get("__a")) if pd.notna(_r.get("__a")) else None,
                        "verdict": str(_r.get(_ver_col) or "") if _ver_col is not None else None,
                    })

                try:
                    if (_seasonality_focus_share is not None) and (_seasonality_second_share is not None):
                        _seasonality_share_gap = float(_seasonality_focus_share) - float(_seasonality_second_share)
                    if (_seasonality_focus_amplitude is not None) and (_seasonality_second_amplitude is not None):
                        _seasonality_amplitude_gap = float(_seasonality_focus_amplitude) - float(_seasonality_second_amplitude)
                except Exception:
                    pass

                if (_seasonality_focus_share is not None) and (_seasonality_second_share is not None):
                    _seasonality_top_mode = "cluster" if abs(float(_seasonality_focus_share) - float(_seasonality_second_share)) < 0.03 else "leader"
                elif _seasonality_focus_amplitude is not None:
                    _seasonality_top_mode = "amplitude_only"
    except Exception:
        pass

    stats.update({
        "seasonality_weight_max": float(pd.to_numeric(scorecard_df[_w_col], errors="coerce").max()) if ("_w_col" in locals() and _w_col in scorecard_df.columns and len(scorecard_df)) else None,
        "seasonality_weight_min": float(pd.to_numeric(scorecard_df[_w_col], errors="coerce").min()) if ("_w_col" in locals() and _w_col in scorecard_df.columns and len(scorecard_df)) else None,
        "seasonality_verdict_counts": (scorecard_df["verdict"].value_counts(dropna=False).to_dict() if "verdict" in scorecard_df.columns and len(scorecard_df) else {}),

        "seasonality_focus_category": _seasonality_focus_category,
        "seasonality_focus_share": _seasonality_focus_share,
        "seasonality_focus_weight": _seasonality_focus_weight,
        "seasonality_focus_strength": _seasonality_focus_strength,
        "seasonality_focus_amplitude": _seasonality_focus_amplitude,
        "seasonality_focus_verdict": _seasonality_focus_verdict,

        "seasonality_second_category": _seasonality_second_category,
        "seasonality_second_share": _seasonality_second_share,
        "seasonality_second_weight": _seasonality_second_weight,
        "seasonality_second_strength": _seasonality_second_strength,
        "seasonality_second_amplitude": _seasonality_second_amplitude,
        "seasonality_second_verdict": _seasonality_second_verdict,

        "seasonality_share_gap": _seasonality_share_gap,
        "seasonality_amplitude_gap": _seasonality_amplitude_gap,
        "seasonality_top_mode": _seasonality_top_mode,
        "seasonality_metric_col": _seasonality_metric_col,
        "seasonality_rows_top2": _seasonality_rows_top2,
        "seasonality_methods": (scorecard_df["seasonality_method"].value_counts(dropna=False).to_dict() if "seasonality_method" in scorecard_df.columns and len(scorecard_df) else {}),
    })

    return a1, scorecard_df, stats

def _fallback_seasonality_heatmap(
    df_time: pd.DataFrame,
    cat_col: str,
    top_n: int = 10,
) -> Optional[alt.Chart]:
    """Last-resort heatmap for short/flat ranges where STL-style scoring returns no chart."""
    if df_time is None or df_time.empty:
        return None
    if "__month" not in df_time.columns or "__value" not in df_time.columns or cat_col not in df_time.columns:
        return None

    df = df_time[["__month", cat_col, "__value"]].copy()
    df["__month"] = pd.to_datetime(df["__month"], errors="coerce")
    df["__value"] = pd.to_numeric(df["__value"], errors="coerce")
    df = df.dropna(subset=["__month", cat_col, "__value"])
    if df.empty:
        return None

    total_by_cat = (
        df.groupby(cat_col, dropna=False)["__value"]
        .sum()
        .sort_values(ascending=False)
    )
    selected = [str(c) for c in total_by_cat.index.astype(str).tolist()[: max(1, int(top_n or 1))]]
    if not selected:
        return None

    grouped = (
        df[df[cat_col].astype(str).isin(selected)]
        .groupby(["__month", cat_col], as_index=False)["__value"]
        .sum()
        .rename(columns={cat_col: "Category", "__value": "value"})
    )
    if grouped.empty:
        return None

    full_idx = pd.date_range(grouped["__month"].min(), grouped["__month"].max(), freq="MS")
    rows: list[dict] = []
    for cat in selected:
        s = (
            grouped[grouped["Category"].astype(str) == str(cat)]
            .set_index("__month")["value"]
            .reindex(full_idx)
        )
        y = pd.to_numeric(s, errors="coerce")
        if y.notna().sum() == 0:
            continue
        y = y.interpolate(method="time", limit_area="inside").ffill().bfill().astype(float)
        vals = y.to_numpy(dtype=float)
        x = np.arange(len(vals), dtype=float)
        ok = np.isfinite(vals)
        if int(ok.sum()) >= 2 and float(np.nanvar(vals)) > 0.0:
            coeff = np.polyfit(x[ok], vals[ok], 1)
            trend = np.polyval(coeff, x)
            seasonal = vals - trend
        else:
            seasonal = np.zeros(len(vals), dtype=float)
        if len(seasonal):
            seasonal = seasonal - float(np.nanmean(seasonal))
        for period, seasonal_val in zip(full_idx, seasonal):
            rows.append(
                {
                    "period": period,
                    "Category": str(cat),
                    "seasonal": float(seasonal_val) if np.isfinite(seasonal_val) else 0.0,
                }
            )

    hm = pd.DataFrame(rows)
    if hm.empty:
        return None
    hm["CategoryLabel"] = hm["Category"].astype(str)
    v = pd.to_numeric(hm["seasonal"], errors="coerce").dropna()
    absmax = float(np.nanquantile(np.abs(v.to_numpy(dtype=float)), 0.95)) if not v.empty else 0.0
    if not np.isfinite(absmax) or absmax <= 0:
        absmax = 1.0

    n_periods = max(1, int(hm["period"].nunique()))
    height = int(np.clip(34 * max(1, len(selected)), 120, 360))
    tick_step = 1 if n_periods <= 14 else (2 if n_periods <= 26 else 3)

    return (
        alt.Chart(hm)
        .mark_rect(stroke="rgba(0,0,0,0.18)", strokeWidth=0.6)
        .encode(
            x=alt.X(
                "period:T",
                timeUnit="yearmonth",
                title="Miesiąc-rok",
                axis=alt.Axis(
                    format="%b %Y",
                    labelAngle=45,
                    labelOverlap="greedy",
                    tickCount={"interval": "month", "step": tick_step},
                ),
            ),
            y=alt.Y("CategoryLabel:N", title="", sort=selected, axis=alt.Axis(labelLimit=500)),
            color=alt.Color(
                "seasonal:Q",
                title="Odchylenie",
                scale=alt.Scale(domain=[-absmax, absmax], scheme="redblue"),
                legend=alt.Legend(orient="right", format=",.0f"),
            ),
            tooltip=[
                alt.Tooltip("CategoryLabel:N", title="Kategoria"),
                alt.Tooltip("period:T", title="Miesiąc", format="%b %Y"),
                alt.Tooltip("seasonal:Q", title="Odchylenie", format=",.0f"),
            ],
        )
        .properties(height=height)
        .configure_view(strokeOpacity=0)
    )

def _block4_slope_start_end(
    df_time: pd.DataFrame,
    cat_col: str,
    top_n: int = 10,
    include_other: bool = False,
    winners_k: int = 3,
    losers_k: int = 3,
    df_time_total: Optional[pd.DataFrame] = None,
) -> Tuple[alt.Chart, Dict[str, Any]]:
    """Start vs End slope chart for share changes.

    IMPORTANT: Shares are always computed vs TOTAL (all categories in df_time_total),
    regardless of whether 'Other' is displayed. When include_other=False, we only
    hide the 'Other' line, but shares of named categories must remain unchanged.
    """
    if df_time_total is None:
        df_time_total = df_time

    if df_time_total is None or df_time_total.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line(), {"topN": 0}

    # Display set: Top-N categories + optional 'Other' bucket (for DISPLAY only)
    df_disp = _apply_topn_other(df_time_total, cat_col=cat_col, top_n=top_n, include_other=include_other)

    # Aggregate values per month & category for DISPLAY
    g = (
        df_disp.groupby(["__month", cat_col], as_index=False)["__value"]
        .sum()
        .rename(columns={"__value": "__value_cat"})
    )

    # TOTAL per month is computed on the FULL (unbucketed) dataset
    totals_all = (
        df_time_total.groupby("__month", as_index=False)["__value"]
        .sum()
        .rename(columns={"__value": "__total_all"})
    )

    m = g.merge(totals_all, on="__month", how="left")
    m["share"] = np.where(m["__total_all"] > 0, m["__value_cat"] / m["__total_all"], 0.0)

    # Determine Start/End months (after filters)
    start_m = pd.to_datetime(m["__month"]).min()
    end_m = pd.to_datetime(m["__month"]).max()
    if pd.isna(start_m) or pd.isna(end_m):
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line(), {"topN": int(top_n)}

    p = m[m["__month"].isin([start_m, end_m])].copy()
    if p.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line(), {"topN": int(top_n)}

    p["point"] = np.where(p["__month"] == start_m, "Start", "Koniec")

    # Delta (pp) per category
    pivot = (
        p.pivot_table(index=cat_col, columns="point", values="share", aggfunc="sum")
        .fillna(0.0)
    )
    if "Start" not in pivot.columns:
        pivot["Start"] = 0.0
    if "Koniec" not in pivot.columns:
        pivot["Koniec"] = 0.0
    pivot["delta_pp"] = (pivot["Koniec"] - pivot["Start"]) * 100.0

    # Select Top winners/losers among DISPLAY categories (includes 'Other' when enabled)
    winners = (
        pivot.sort_values("delta_pp", ascending=False)
        .head(int(max(0, winners_k)))
        .index.astype(str)
        .tolist()
    )
    losers = (
        pivot.sort_values("delta_pp", ascending=True)
        .head(int(max(0, losers_k)))
        .index.astype(str)
        .tolist()
    )

    p[cat_col] = p[cat_col].astype(str)

    def _group(cat: str) -> str:
        if cat in winners:
            return "Winners"
        if cat in losers:
            return "Losers"
        return "Context"

    p["group"] = p[cat_col].map(_group)

    # Dynamic Y domain based on visible points (Start+Koniec) for DISPLAY categories
    y_min = float(p["share"].min())
    y_max = float(p["share"].max())
    pad = max(0.01, (y_max - y_min) * 0.08)  # min ~1pp padding
    y0 = max(0.0, y_min - pad)
    y1 = min(1.0, y_max + pad)

    # Label anti-overlap: only for top winners+losers at "Koniec"
    labels = p[(p["point"] == "Koniec") & (p["group"].isin(["Winners", "Losers"]))].copy()
    labels = labels.merge(pivot[["delta_pp"]], left_on=cat_col, right_index=True, how="left")
    labels = labels.sort_values(["share", "delta_pp"], ascending=[False, False]).reset_index(drop=True)

    min_gap = 0.006  # ~0.6pp
    y_vals = labels["share"].astype(float).tolist()
    y_adj = []
    prev = None
    for y in y_vals:
        if prev is None:
            ya = y
        else:
            ya = min(y, prev - min_gap)
        y_adj.append(ya)
        prev = ya
    # If we pushed too far down, re-pack upward within [y0,y1]
    if y_adj:
        # ensure within bounds
        y_adj = [min(y1, max(y0, y)) for y in y_adj]
        # second pass bottom-up
        for i in range(len(y_adj) - 2, -1, -1):
            y_adj[i] = max(y_adj[i], y_adj[i + 1] + min_gap)
        y_adj = [min(y1, max(y0, y)) for y in y_adj]

    labels["y_label"] = y_adj if len(y_adj) == len(labels) else labels["share"]

    # Layers:
    base = alt.Chart(p)

    context_lines = (
        base.transform_filter(alt.datum.group == "Context")
        .mark_line(point=True)
        .encode(
            x=alt.X("point:N", title="", sort=["Start", "Koniec"]),
            y=alt.Y("share:Q", title="Udział w sprzedaży (%)", axis=alt.Axis(format="%", grid=True), scale=alt.Scale(domain=[y0, y1])),
            detail=alt.Detail(f"{cat_col}:N"),
            color=alt.value("#B0B0B0"),
            tooltip=[
                alt.Tooltip(f"{cat_col}:N", title=_dimension_axis_title(cat_col)),
                alt.Tooltip("point:N", title="Punkt"),
                alt.Tooltip("share:Q", title="Udział", format=".1%"),
            ],
        )
    )

    focus_lines = (
        base.transform_filter(alt.datum.group != "Context")
        .mark_line(point=True)
        .encode(

            x=alt.X(
                "point:N",
                title="",
                sort=["Start", "Koniec"],
                axis=alt.Axis(labelAngle=0, labelPadding=10),
            ),
            y=alt.Y(
                "share:Q",
                title="Udział w sprzedaży (%)",
                axis=alt.Axis(format="%", grid=True),
                scale=alt.Scale(domain=[y0, y1]),
            ),
            detail=alt.Detail(f"{cat_col}:N"),
            color=alt.Color(
                "group:N",
                title="",
                sort=["Winners", "Losers"],
                scale=alt.Scale(domain=["Winners", "Losers"], range=["#1B8E3E", "#D92D20"]),
                legend=alt.Legend(orient="top", direction="horizontal", labelFontSize=13, labelFontWeight="bold"),
            ),
            strokeWidth=alt.value(2.4),
            tooltip=[
                alt.Tooltip(f"{cat_col}:N", title=_dimension_axis_title(cat_col)),
                alt.Tooltip("group:N", title=""),
                alt.Tooltip("point:N", title="Punkt"),
                alt.Tooltip("share:Q", title="Udział", format=".1%"),
            ],
        )
    )

    label_text = (
        alt.Chart(labels)
        .mark_text(align="left", dx=6, fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X("point:N", sort=["Start", "Koniec"]),
            y=alt.Y("y_label:Q", scale=alt.Scale(domain=[y0, y1])),
            text=alt.Text(f"{cat_col}:N"),
            color=alt.Color("group:N", scale=alt.Scale(domain=["Winners", "Losers"], range=["#1B8E3E", "#D92D20"]), legend=None),
        )
    )

    chart = (
        (context_lines + focus_lines + label_text)
        .configure_view(strokeOpacity=0)
        # Keep height consistent with other blocks (avoids excess whitespace and helps keep X labels visible).
        .properties(height=max(int(CHART_BLOCK_HEIGHT * 1.35), 420))
        # Reserve room for horizontal X labels and keep a compact top margin.
        .properties(padding={"top": 6, "bottom": 28, "left": 6, "right": 6})
        # Stronger horizontal grid for easier reading (closer to the main time-series chart).
        .configure_axis(gridOpacity=0.45, gridWidth=1, gridColor="#D0D7E2")
    )

    # stats for ET / debug
    biggest_gainer = str(pivot["delta_pp"].idxmax()) if not pivot.empty else ""
    biggest_loser = str(pivot["delta_pp"].idxmin()) if not pivot.empty else ""
    _winner_non_other = None
    _loser_non_other = None
    _winner_non_other_delta_pp = None
    _loser_non_other_delta_pp = None
    _winner_non_other_start_pct = None
    _winner_non_other_end_pct = None
    _loser_non_other_start_pct = None
    _loser_non_other_end_pct = None
    _delta_rows = []
    try:
        _pivot_non_other = pivot.loc[~pivot.index.astype(str).map(_is_other_label)].copy()
        _pivot_non_other_nz = _pivot_non_other[_pivot_non_other["delta_pp"].abs() > 1e-9]
        _base = _pivot_non_other_nz if len(_pivot_non_other_nz) else _pivot_non_other
        if len(_base):
            _winner_non_other = str(_base["delta_pp"].idxmax())
            _loser_non_other = str(_base["delta_pp"].idxmin())
            _winner_non_other_delta_pp = round(float(_base.loc[_winner_non_other, "delta_pp"]), 1)
            _loser_non_other_delta_pp = round(float(_base.loc[_loser_non_other, "delta_pp"]), 1)
            _winner_non_other_start_pct = round(float(_base.loc[_winner_non_other, "Start"] * 100.0), 1) if "Start" in _base.columns else None
            _winner_non_other_end_pct = round(float(_base.loc[_winner_non_other, "Koniec"] * 100.0), 1) if "Koniec" in _base.columns else None
            _loser_non_other_start_pct = round(float(_base.loc[_loser_non_other, "Start"] * 100.0), 1) if "Start" in _base.columns else None
            _loser_non_other_end_pct = round(float(_base.loc[_loser_non_other, "Koniec"] * 100.0), 1) if "Koniec" in _base.columns else None
        _delta_rows = [{"category": str(idx), "start_pct": round(float(row.get("Start",0.0)*100.0),1), "end_pct": round(float(row.get("Koniec",0.0)*100.0),1), "delta_pp": round(float(row.get("delta_pp",0.0)),1)} for idx,row in pivot.sort_values("delta_pp", ascending=False).iterrows()]
    except Exception:
        pass

    stats = {
        "period_start": str(start_m)[:10],
        "period_end": str(end_m)[:10],
        "winners_top": winners,
        "losers_top": losers,
        "winner_dimension_value": biggest_gainer,
        "biggest_gainer_category": biggest_gainer,
        "biggest_gainer_delta_pp": round(float(pivot.loc[biggest_gainer, "delta_pp"]), 1) if biggest_gainer in pivot.index else None,
        "loser_dimension_value": biggest_loser,
        "biggest_loser_category": biggest_loser,
        "biggest_loser_delta_pp": round(float(pivot.loc[biggest_loser, "delta_pp"]), 1) if biggest_loser in pivot.index else None,
        "winner_dimension_value_non_other": _winner_non_other,
        "winner_category_non_other": _winner_non_other,
        "winner_delta_pp_non_other": _winner_non_other_delta_pp,
        "loser_dimension_value_non_other": _loser_non_other,
        "loser_category_non_other": _loser_non_other,
        "loser_delta_pp_non_other": _loser_non_other_delta_pp,
        "winner_start_pct_non_other": _winner_non_other_start_pct,
        "winner_end_pct_non_other": _winner_non_other_end_pct,
        "loser_start_pct_non_other": _loser_non_other_start_pct,
        "loser_end_pct_non_other": _loser_non_other_end_pct,
        "delta_rows": _delta_rows,
        "topN": int(top_n),
        "include_other": bool(include_other),
        "y_domain_pp": [round(y0 * 100, 2), round(y1 * 100, 2)],
    }
    return chart, stats


def _block5_concentration(
    df_time: pd.DataFrame,
    cat_col: str,
    *,
    top_n: int,
    include_other: bool,
) -> Tuple[alt.Chart, Dict[str, Any]]:
    """Tab 2 — 🧲 Konsolidacja/Dywersyfikacja: koncentracja Top-3 i Top-5 w czasie.

    Checkbox "Dodaj 'Other'" steruje *granularnością* (ogon scalony do "Other" vs rozbity).
    Ten blok MUSI reagować na ten przełącznik: jeśli "Other" staje się największą kategorią,
    powinna wpływać na Top-3/Top-5.
    """
    if df_time is None or df_time.empty:
        empty = pd.DataFrame({"__month": [], "series": [], "share": []})
        return alt.Chart(empty).mark_line(), {"status": "fail", "reason": "Brak danych."}

    df_disp = _apply_topn_other(df_time, cat_col=cat_col, top_n=int(top_n), include_other=bool(include_other))
    if df_disp is None or df_disp.empty:
        empty = pd.DataFrame({"__month": [], "series": [], "share": []})
        return alt.Chart(empty).mark_line(), {"status": "fail", "reason": "Brak danych po transformacji Top-N + Other."}

    work = df_disp[["__month", cat_col, "__value"]].copy().dropna(subset=["__month", cat_col])
    work[cat_col] = work[cat_col].astype(str)
    work["__month"] = pd.to_datetime(work["__month"], errors="coerce")
    work = work.dropna(subset=["__month"])
    if work.empty:
        empty = pd.DataFrame({"__month": [], "series": [], "share": []})
        return alt.Chart(empty).mark_line(), {"status": "fail", "reason": "Brak poprawnych miesięcy."}

    g = work.groupby(["__month", cat_col], as_index=False)["__value"].sum()
    totals = g.groupby("__month", as_index=False)["__value"].sum().rename(columns={"__value": "__total"})
    m = g.merge(totals, on="__month", how="left")
    m["share"] = np.where(pd.to_numeric(m["__total"], errors="coerce").fillna(0).values > 0, m["__value"] / m["__total"], 0.0)

    m = m.sort_values(["__month", "share"], ascending=[True, False]).copy()
    m["_rank"] = m.groupby("__month")["share"].rank(method="first", ascending=False)

    top3 = m[m["_rank"] <= 3].groupby("__month", as_index=False)["share"].sum().rename(columns={"share": "top3"})
    top5 = m[m["_rank"] <= 5].groupby("__month", as_index=False)["share"].sum().rename(columns={"share": "top5"})
    x = pd.merge(top3, top5, on="__month", how="outer").sort_values("__month").reset_index(drop=True)
    x["top3"] = pd.to_numeric(x["top3"], errors="coerce").fillna(0.0)
    x["top5"] = pd.to_numeric(x["top5"], errors="coerce").fillna(0.0)

    # HHI (Herfindahl–Hirschman Index): suma kwadratów udziałów kategorii w danym miesiącu
    hhi = (
        m.groupby("__month")["share"]
        .apply(lambda s: float(np.sum(np.square(pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float).values))))
        .reset_index(name="hhi")
    )
    x = x.merge(hhi, on="__month", how="left")
    x["hhi"] = pd.to_numeric(x["hhi"], errors="coerce").fillna(0.0)


    dd = x.melt(id_vars=["__month"], value_vars=["top3", "top5", "hhi"], var_name="series", value_name="share")
    dd["series"] = dd["series"].map({"top3": "Top-3", "top5": "Top-5", "hhi": "HHI"}).fillna(dd["series"].astype(str))

    def _trend_pp_per_year(sdf: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        try:
            ss = sdf.dropna(subset=["__month", "share"]).sort_values("__month").copy()
            if len(ss) < 2:
                return None, None
            t0 = pd.to_datetime(ss["__month"].iloc[0])
            t = (pd.to_datetime(ss["__month"]) - t0).dt.days.astype(float) / 365.25
            y_pp = pd.to_numeric(ss["share"], errors="coerce").fillna(0.0).astype(float).values * 100.0
            if float(np.nanvar(y_pp)) == 0.0:
                return 0.0, 0.0
            slope, intercept = np.polyfit(t.values, y_pp, 1)
            y_hat = slope * t.values + intercept
            ss_res = float(np.sum((y_pp - y_hat) ** 2))
            ss_tot = float(np.sum((y_pp - float(np.mean(y_pp))) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            return float(slope), float(max(0.0, min(1.0, r2)))
        except Exception:
            return None, None

    def _trend_unit_per_year(sdf: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        try:
            ss = sdf.dropna(subset=["__month", "share"]).sort_values("__month").copy()
            if len(ss) < 2:
                return None, None
            t0 = pd.to_datetime(ss["__month"].iloc[0])
            t = (pd.to_datetime(ss["__month"]) - t0).dt.days.astype(float) / 365.25
            y = pd.to_numeric(ss["share"], errors="coerce").fillna(0.0).astype(float).values
            if float(np.nanvar(y)) == 0.0:
                return 0.0, 0.0
            slope, intercept = np.polyfit(t.values, y, 1)
            y_hat = slope * t.values + intercept
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            return float(slope), float(max(0.0, min(1.0, r2)))
        except Exception:
            return None, None

    slope5, r2_5 = _trend_pp_per_year(dd[dd["series"] == "Top-5"])
    slope3, r2_3 = _trend_pp_per_year(dd[dd["series"] == "Top-3"])
    slope_hhi, r2_hhi = _trend_unit_per_year(dd[dd["series"] == "HHI"])

    def _delta_pp(col: str) -> Optional[float]:
        try:
            if x.empty:
                return None
            s = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
            return float((s.iloc[-1] - s.iloc[0]) * 100.0)
        except Exception:
            return None

    d5 = _delta_pp("top5")
    d3 = _delta_pp("top3")

    def _delta_unit(col: str) -> Optional[float]:
        try:
            if x.empty:
                return None
            s = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
            return float(s.iloc[-1] - s.iloc[0])
        except Exception:
            return None

    d_hhi = _delta_unit("hhi")
    hhi_last = float(pd.to_numeric(x["hhi"], errors="coerce").fillna(0.0).iloc[-1]) if not x.empty else None


    n_months = int(x["__month"].nunique()) if not x.empty else 0
    r2_ref = max([v for v in [r2_5, r2_3, r2_hhi] if v is not None], default=0.0)
    if n_months >= 18 and r2_ref >= 0.30:
        conf = "wysoka"
    elif n_months >= 12 and r2_ref >= 0.15:
        conf = "średnia"
    else:
        conf = "niska"

    def _small(v: Optional[float], thr: float) -> bool:
        try:
            return v is not None and abs(float(v)) <= float(thr)
        except Exception:
            return False

    trend5 = float(slope5) if slope5 is not None else 0.0
    trend3 = float(slope3) if slope3 is not None else 0.0
    if _small(d5, 2.0) and _small(d3, 2.0) and _small(trend5, 0.5) and _small(trend3, 0.5):
        badge = "Stabilna struktura"
        badge_desc = "Koncentracja Top-3/Top-5 nie zmienia się istotnie w czasie."
    else:
        direction = 0
        if d5 is not None and abs(d5) >= 1.0:
            direction = 1 if d5 > 0 else -1
        elif abs(trend5) >= 0.2:
            direction = 1 if trend5 > 0 else -1
        elif d3 is not None and abs(d3) >= 1.0:
            direction = 1 if d3 > 0 else -1
        elif abs(trend3) >= 0.2:
            direction = 1 if trend3 > 0 else -1

        # Doprecyzowanie typu konsolidacji przez porównanie Top-5 vs HHI:
        # - "Konsolidacja przez lidera": Top-5 rośnie i HHI rośnie istotnie (dominacja 1–2 kategorii).
        # - "Konsolidacja szeroka": Top-5 rośnie, ale HHI pozostaje ~ stabilne (liderzy rosną równomiernie).
        hhi_delta = d_hhi if d_hhi is not None else 0.0
        hhi_slope = slope_hhi if slope_hhi is not None else 0.0
        hhi_stable = (abs(hhi_delta) <= 0.02) and (abs(hhi_slope) <= 0.01)
        hhi_up = (hhi_delta >= 0.03) or (hhi_slope >= 0.015)

        if direction > 0:
            if hhi_stable:
                badge = "Konsolidacja szeroka"
                badge_desc = "Top‑5 rośnie, ale HHI jest blisko stabilne — liderzy rosną równomiernie."
            elif hhi_up:
                badge = "Konsolidacja przez lidera"
                badge_desc = "Top‑5 rośnie i HHI rośnie — rośnie dominacja 1–2 kategorii."
            else:
                badge = "Konsolidacja"
                badge_desc = "Udział największych kategorii rośnie — struktura się konsoliduje."
        elif direction < 0:
            badge = "Dywersyfikacja"
            badge_desc = "Udział największych kategorii spada — struktura się dywersyfikuje."
        else:
            badge = "Stabilna struktura"
            badge_desc = "Koncentracja Top-3/Top-5 nie ma jednoznacznego trendu."

    # --- Skala (umiarkowana / silna) + kolor Badge (minimal patch, bez wpływu na pozostałą logikę) ---
    # Heurystyka odporna na różne datasety: porównujemy skalę zmiany (Δ) do typowej zmienności (IQR).
    # Dodatkowe progi bezpieczeństwa: bardzo duże Δ -> "silna" niezależnie od IQR.
    strength_label: Optional[str] = None
    badge_icon = "🟢"  # domyślnie (stabilna / brak kwalifikatora)
    badge_display = badge
    try:
        if badge != "Stabilna struktura":
            top5_s = pd.to_numeric(x["top5"], errors="coerce").dropna() if "top5" in x.columns else pd.Series(dtype=float)
            hhi_s = pd.to_numeric(x["hhi"], errors="coerce").dropna() if "hhi" in x.columns else pd.Series(dtype=float)

            top5_iqr_pp = float((top5_s.quantile(0.75) - top5_s.quantile(0.25)) * 100.0) if not top5_s.empty else 1.0
            hhi_iqr = float(hhi_s.quantile(0.75) - hhi_s.quantile(0.25)) if not hhi_s.empty else 0.02

            # minimalne progi ochronne (żeby nie wystrzeliwać na datasetach o bardzo małej wariancji)
            top5_iqr_pp = max(top5_iqr_pp, 1.0)  # pp
            hhi_iqr = max(hhi_iqr, 0.02)         # HHI w [0,1]

            s_top5 = abs(float(d5)) / top5_iqr_pp if d5 is not None else 0.0
            s_hhi = abs(float(d_hhi)) / hhi_iqr if d_hhi is not None else 0.0
            strength_score = 0.6 * s_top5 + 0.4 * s_hhi

                        # Konserwatywny warunek „silna” — ma nie przeszacowywać:
            # - wymaga min. jakości trendu (R²) i nie może przejść przy niskiej pewności
            # - wymaga istotnej skali zmian (Top‑5 / HHI) lub bardzo wysokiego score
            min_r2_strong = 0.20
            is_strong = (
                conf != "niska"
                and float(r2_ref) >= min_r2_strong
                and (
                    (
                        d5 is not None
                        and r2_5 is not None
                        and abs(float(d5)) >= 12.0
                        and float(r2_5) >= 0.25
                    )
                    or (
                        d_hhi is not None
                        and r2_hhi is not None
                        and abs(float(d_hhi)) >= 0.12
                        and float(r2_hhi) >= 0.25
                    )
                    or (
                        strength_score >= 3.0
                        and d5 is not None
                        and abs(float(d5)) >= 6.0
                        and d_hhi is not None
                        and abs(float(d_hhi)) >= 0.06
                    )
                )
            )
            strength_label = "silna" if is_strong else "umiarkowana"
            badge_icon = "🔴" if is_strong else "🟠"

            # Prepend kwalifikator skali do nazwy badge (np. "Silna konsolidacja szeroka")
            if isinstance(badge, str) and badge:
                badge_display = f"{strength_label.capitalize()} {badge[:1].lower() + badge[1:]}"
    except Exception:
        # w razie jakichkolwiek problemów: zachowujemy neutralne renderowanie
        strength_label = None
        badge_icon = "🟢"
        badge_display = badge

    def _fmt(v: Optional[float], nd: int = 1) -> str:
        if v is None or not np.isfinite(float(v)):
            return "—"
        return f"{float(v):+0.{nd}f}"

    def _fmt_r2(v: Optional[float]) -> str:
        if v is None or not np.isfinite(float(v)):
            return "—"
        return f"{float(v):0.2f}"

    evidence = (
        f"Top-5 Δ = {_fmt(d5)} pp, trend = {_fmt(slope5)} pp/rok (R²={_fmt_r2(r2_5)}, pewność: {conf}); "
        f"HHI Δ = {(_fmt(d_hhi, 3) if d_hhi is not None else '—')}, trend = {(_fmt(slope_hhi, 3) if slope_hhi is not None else '—')} /rok (R²={_fmt_r2(r2_hhi)}), "
        f"ostatni HHI = {(f'{hhi_last:0.2f}' if hhi_last is not None else '—')}."
    )

    metric_order = ["Top-3", "Top-5", "HHI"]

    color = alt.Color(
        "series:N",
        title="",
        sort=metric_order,
        scale=alt.Scale(domain=metric_order, range=["#1F77B4", "#9EC7EA", "#6B7280"]),
        legend=alt.Legend(orient="top", direction="horizontal"),
    )

    base = alt.Chart(dd).encode(
        x=alt.X("__month:T", title="Miesiąc", axis=alt.Axis(labelAngle=-45, labelPadding=10, format="%b %Y")),
        color=color,
    )

    share_layer = (
        base.transform_filter(alt.datum.series != "HHI")
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=55, opacity=0.95), strokeWidth=2.8)
        .encode(
            y=alt.Y(
                "share:Q",
                title="Koncentracja udziałów (Top-N)",
                axis=alt.Axis(format="%", grid=True),
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("__month:T", title="Miesiąc", format="%b %Y"),
                alt.Tooltip("series:N", title="Seria"),
                alt.Tooltip("share:Q", title="Wartość", format=".1%"),
            ],
        )
    )

    hhi_layer = (
        base.transform_filter(alt.datum.series == "HHI")
        .mark_line(strokeWidth=2.2, strokeDash=[6, 4])
        .encode(
            y=alt.Y(
                "share:Q",
                title="HHI (koncentracja)",
                axis=alt.Axis(format=".2f", grid=False),
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("__month:T", title="Miesiąc", format="%b %Y"),
                alt.Tooltip("share:Q", title="HHI", format=".2f"),
            ],
        )
    )

    chart = (
        alt.layer(share_layer, hhi_layer)
        .resolve_scale(y="independent")
        .properties(height=max(int(CHART_BLOCK_HEIGHT * 1.35), 420))
        .configure_view(strokeOpacity=0)
        .configure_axis(gridOpacity=0.45, gridWidth=1, gridColor="#D0D7E2")
    )

    stats = {
        "badge": badge,
        "badge_desc": badge_desc,
        "badge_icon": badge_icon,
        "badge_display": badge_display,
        "evidence": evidence,
        "confidence": conf,
        "include_other": bool(include_other),
        "top_n": int(top_n),
        "top5_delta_pp": d5,
        "top3_delta_pp": d3,
        "top5_trend_ppy": slope5,
        "top3_trend_ppy": slope3,
        "top5_r2": r2_5,
        "top3_r2": r2_3,
        "hhi_last": hhi_last,
        "hhi_delta": d_hhi,
        "hhi_trend": slope_hhi,
        "hhi_r2": r2_hhi,
    }
    return chart, stats

# -----------------------------
# Guidance (ported 1:1 spirit from donor)
# -----------------------------

def _guidance_for(block_id: str, *, dimension_label: str = "wymiar", dimension_entity: str = "element") -> Dict[str, str]:
    if block_id == "cot__mix_share_topN":
        return {
            "sens": "Ten wykres pokazuje, jak zmienia się struktura (udziały %) między elementami wymiaru w czasie.",
            "interpretacja": "Szukaj elementów wymiaru, które systematycznie rosną (zyskują pp) oraz tych, które tracą udział — to sygnał realnej zmiany miksu.",
            "best_practice": "Ogranicz liczbę serii (TopN) i utrzymuj stałą kolejność legendy, aby porównania między miesiącami były czytelne.",
        }
    if block_id == "cot__winners_losers":
        return {
            "sens": "Linie ułatwiają zauważenie trendu udziałów w czasie dla każdego elementu wymiaru.",
            "interpretacja": "Najważniejsze są zmiany nachylenia oraz trwałe przesunięcia poziomu udziału (nie pojedyncze skoki).",
            "best_practice": "Pokazuj TopN elementów wymiaru i zawsze podawaj różnicę start→koniec w pp dla zwycięzców i przegranych.",
        }
    if block_id == "cot__seasonality":
        return {
            "sens": "Heatmapa i scorecard wskazują, czy udziały mają sezonowe piki (powtarzalne miesiące wysokie/niskie).",
            "interpretacja": "Jeśli ten sam element wymiaru ma powtarzalny miesiąc szczytu, to sygnał sezonowości; jeśli wzór jest losowy — sezonowość słaba.",
            "best_practice": "Łącz wizualny wzór (heatmapa) z metrykami siły i stabilności sezonowości, aby uniknąć 'pattern hunting'.",
        }
    if block_id == "cot__start_end":
        return {
            "sens": "Porównanie start vs koniec pokazuje, które elementy wymiaru realnie zmieniły pozycję w strukturze.",
            "interpretacja": "Patrz na Δ udziału w pp: małe zmiany mogą być szumem, duże zmiany wymagają wyjaśnienia (popyt, ceny, dostępność).",
            "best_practice": "Zestaw ten widok z wykresem „📈 Kto zyskuje, a kto traci udział w czasie?”, aby sprawdzić, czy zmiana wynika z trwałego trendu, czy z jednorazowego skoku na początku/końcu okresu.",
        }
    return {
        "sens": "Miary koncentracji pokazują, czy sprzedaż skupia się w kilku elementach wymiaru czy rozkłada szerzej.",
        "interpretacja": "Wzrost TopK share i HHI → konsolidacja; spadek → dywersyfikacja.",
        "best_practice": "Używaj równolegle TopK share (intuicyjne) i HHI (wrażliwe na ogon), a wnioski opieraj o trend, nie jeden punkt.",
    }

def _render_timing_ui(target: Any, timings_ms: Dict[str, float]) -> None:
    """Render minimal timing table (debug only)."""
    if not timings_ms:
        return
    target.markdown("⏱️ **COT timing**")
    target.markdown("---")
    order = [
        "infer_cols_filters",
        "infer_time_col",
        "infer_cat_col",
        "tab1_overview_total",
        "tab2_insights_total",
        "_prep_time_df",
        "_main_chart_value",
        "_main_chart_share",
        "render_chart",
        "llm_exec_takeaways_batch",
        "TOTAL_render",
    ]
    rows: List[Tuple[str, float]] = []
    for k in order:
        if k in timings_ms:
            rows.append((k, float(timings_ms[k])))
    for k, v in timings_ms.items():
        if k not in {r[0] for r in rows}:
            rows.append((k, float(v)))
    lines = [f"{k:<18} {v:>7.0f} ms" for k, v in rows]
    target.code("\n".join(lines))

# -----------------------------
# Branch render (public API)
# -----------------------------

def _short_repr(obj: Any, limit: int = 360) -> str:
    """Best-effort short repr for debug messages (anti-crash)."""
    try:
        s = repr(obj)
    except Exception:
        try:
            s = f"<{type(obj).__name__}>"
        except Exception:
            s = "<unrepr>"
    if len(s) > limit:
        return s[:limit] + "…"
    return s

def _is_altair_chart_like(obj: Any) -> bool:
    """Heuristic check: Altair charts expose a callable .to_dict()."""
    try:
        to_dict = getattr(obj, "to_dict", None)
        return callable(to_dict)
    except Exception:
        return False

def _unwrap_altair_chart(obj: Any, *, _depth: int = 2) -> Optional[Any]:
    """Return the first Altair-like chart found in obj (supports tuple/list nesting)."""
    if obj is None:
        return None
    if _is_altair_chart_like(obj):
        return obj
    if _depth <= 0:
        return None
    if isinstance(obj, (tuple, list)):
        for it in obj:
            ch = _unwrap_altair_chart(it, _depth=_depth - 1)
            if ch is not None:
                return ch
    return None

# ------------------------------
# LLM safety helpers (module-scope)
# ------------------------------
def _count_numbers_first_sentence(text: str) -> int:
    """Count numeric tokens in the first sentence (or first line if no sentence end).
    Used by hard validators / gates to enforce '2 numbers' rule."""
    if not text:
        return 0
    s = str(text).strip()
    # First sentence boundary: ., !, ?, or newline
    m = re.split(r"(?<=[\.!\?])\s+|\n", s, maxsplit=1)
    first = m[0] if m else s
    # Count numbers like 12, 12.3, 12,3, 1 234, 1,234.56 etc.
    nums = re.findall(r"[-+]?\d{1,3}(?:[\s\u00A0]?\d{3})*(?:[\.,]\d+)?", first)
    return len(nums)

def _fallback_exec_takeaway(block_id: str, stats: dict) -> str:
    """Deterministic fallback used when LLM output is missing or fails numeric validator.
    Must be universal across datasets; relies only on provided stats keys (if present)."""
    try:
        stats = _with_dimension_stat_aliases(stats)
        if block_id == "cot__mix_share_topN":
            cats = stats.get("top_categories") or []
            n = len(cats) if cats else int(stats.get("topN", 10) or 10)
            top1 = stats.get("top1_category") or (cats[0] if cats else "lider")
            sN = float(stats.get("topN_start_pct", 0.0))
            eN = float(stats.get("topN_end_pct", 0.0))
            dN = float(stats.get("topN_delta_pp", eN - sN))
            d1 = float(stats.get("top1_delta_pp", 0.0))
            return (
                f"Udział Top-{n} zmienił się z {sN:.1f}% do {eN:.1f}% (Δ {dN:.1f} pp); lider {top1} zmienił udział o {d1:.1f} pp. "
                "Rekomendacja: zweryfikuj driver(y) zmiany (ceny, miks, dostępność) i dopasuj alokację budżetu/ekspozycji w kanałach."
            )
        if block_id == "cot__winners_losers":
            w = stats.get("winner_category") or "zwycięzca"
            l = stats.get("loser_category") or "przegrany"
            wd = float(stats.get("winner_delta_pp", 0.0))
            ld = float(stats.get("loser_delta_pp", 0.0))
            return (
                f"„{w}” zyskał +{wd:.1f} pp, a „{l}” stracił {ld:.1f} pp, co sygnalizuje transfer udziału, przejęcie popytu i zmianę drivera wzrostu. "
                f"Decyzja: w 1–2 miesiącach wzmocnij availability, ekspozycję i budżet dla „{w}”, a dla „{l}” uruchom test price-pack/promo/asortyment w kanałach największej utraty."
            )
    except Exception:
        pass
    return "Wynik jest niejednoznaczny – sprawdź dane wejściowe i zakres filtrów."

def render(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
    schema_ctx = ctx.get("schema_ctx") or {}
    question = _safe_str(ctx.get("question") or "")
    tabs = (ctx.get("tabs") or {})
    ui = (ctx.get("ui") or {})
    filters = ctx.get("filters") or {}

    debug_perf = bool(st.session_state.get("dc_debug", False))
    _timings: Dict[str, float] = {}
    _t0_total = perf_counter()
    _t0 = perf_counter()

    # Debug checkpoints must be enabled early in render(), otherwise checkpointy z pierwszych faz
    # nie zapiszą się dla bieżącego pytania. Trzymaj stan sticky w session_state.
    try:
        _dbg_early = bool(ctx.get("debug_exec_takeaway") or ctx.get("debug") or ctx.get("debug_mode"))
    except Exception:
        _dbg_early = False
    try:
        _cp_dbg_prev = bool(st.session_state.get("cot__debug_interp_checkpoints"))
    except Exception:
        _cp_dbg_prev = False
    try:
        st.session_state["__cot_exec_dbg_on"] = bool(_dbg_early or _cp_dbg_prev or st.session_state.get("__cot_exec_dbg_on"))
    except Exception:
        pass

    # ── Overrides z sidebaru (ctx["filters"]) + memoization per dataset (perf)
    ss = st.session_state
    df_fp = _df_fingerprint(df)

    # time_col: prefer sidebar/global date_col, else memoized inference, else infer
    _t_inf_time = perf_counter()
    time_col = (filters.get("date_col") or None)
    # Guard: sometimes Stage2->Stage3 passes a wrong "date_col" (e.g., Invoice IDs parsed as 2049+)
    if time_col and (time_col in df.columns) and (not _is_valid_time_col(df, time_col)):
        st.warning(
            f"⚠️ Wykryto nieprawidłową kolumnę czasu: **{time_col}** (np. dziwne lata typu 2049+). "
            "Używam automatycznie wykrytej kolumny daty."
        )
        time_col = None
    if not time_col:
        memo = ss.get("cot__memo_timecol")
        if isinstance(memo, dict) and memo.get("df_fp") == df_fp:
            time_col = memo.get("time_col")
        if not time_col:
            time_col = _infer_time_col(df, schema_ctx)
            ss["cot__memo_timecol"] = {"df_fp": df_fp, "time_col": time_col}
    _timings["infer_time_col"] = (perf_counter() - _t_inf_time) * 1000.0

    # cat_col: prefer sidebar cat, else memoized inference, else infer
    _t_inf_cat = perf_counter()
    cat_col = (filters.get("cot_cat_col") or None)
    if not cat_col:
        memo = ss.get("cot__memo_catcol")
        if isinstance(memo, dict) and memo.get("df_fp") == df_fp:
            cat_col = memo.get("cat_col")
        if not cat_col:
            cat_col = _infer_cat_col(df, schema_ctx)
            ss["cot__memo_catcol"] = {"df_fp": df_fp, "cat_col": cat_col}
    _timings["infer_cat_col"] = (perf_counter() - _t_inf_cat) * 1000.0

    dimension_label = _dimension_display_label(cat_col) if cat_col else "Wymiar"
    dimension_entity = _dimension_entity_label(cat_col)

    cot_top_n = int(filters.get("cot_top_n") or 10)
    # Single source of truth for including 'Other' (resta kategorii): sidebar checkbox.
    cot_include_other = filters.get("cot_include_other")
    if cot_include_other is None:
        cot_include_other = ss.get("cot_include_other", True)
    include_other = bool(cot_include_other)
# Tryb (wartość vs udziały) wynika z pytania — nie z sidebaru.

    # ── Sanity check: user may incorrectly pick numeric/datetime column as category → do not crash
    try:
        if cat_col and (pd.api.types.is_numeric_dtype(df[cat_col]) or pd.api.types.is_datetime64_any_dtype(df[cat_col])):
            st.info("Wybrana kolumna nie jest poprawnym wymiarem analizy, lecz miarą liczbową lub datą. Wybierz kolumnę tekstową albo kategoryczną.")
            return {"chart_meta": {"kind": "cot"}, "chart_context": {"cot": False}}
    except Exception:
        st.info("Wybrana kolumna nie jest poprawnym wymiarem analizy. Wybierz kolumnę tekstową albo kategoryczną.")
        return {"chart_meta": {"kind": "cot"}, "chart_context": {"cot": False}}

    _timings["infer_cols_filters"] = (perf_counter() - _t0) * 1000.0

    if not time_col or not cat_col:
        st.warning("Brak wymaganych kolumn do analizy 'Composition Over Time' (czas + wymiar).")
        return {"chart_meta": {"kind": "cot"}, "chart_context": {"cot": False}}

    value_col, qty_col, price_col = _infer_value_col(df, schema_ctx)
    # ── Cache ciężkiego prep'u w session_state (szybciej reaguje na zmiany filtrów)
    data_id = _safe_str(
        ctx.get("data_id")
        or ctx.get("parquet_path")
        or ctx.get("data_path")
        or ctx.get("source_path")
        or ""
    )
    cache_key = f"cot_df_time::{data_id}::{time_col}::{cat_col}::{value_col}::{qty_col}::{price_col}"
    ss = st.session_state
    _t_prep = perf_counter()
    if cache_key in ss and isinstance(ss.get(cache_key), pd.DataFrame):
        df_time = ss[cache_key]
    else:
        df_time = _prep_time_df(df, time_col, cat_col, value_col, qty_col, price_col)
        ss[cache_key] = df_time
    _timings["_prep_time_df"] = (perf_counter() - _t_prep) * 1000.0
    # --- Question-driven period filter (month range) ---
    # If the user asks for a specific month range (e.g. "styczeń–czerwiec 2010"),
    # the main chart + KPIs must adapt to that period (regression guard).
    q_start_m, q_end_m = _parse_month_range_from_question(question)
    df_time_q = df_time
    df_q = df
    if q_start_m is not None and q_end_m is not None:
        try:
            # filter aggregated df_time (fast)
            _m = pd.to_datetime(df_time["__month"], errors="coerce")
            df_time_q = df_time[(_m >= q_start_m) & (_m <= q_end_m)].copy()

            # filter raw df only if we have a valid time_col (for KPI: n_rows)
            if time_col and time_col in df.columns:
                _raw_m = _to_month_start(df[time_col])
                df_q = df[(_raw_m >= q_start_m) & (_raw_m <= q_end_m)].copy()
        except Exception:
            df_time_q = df_time
            df_q = df

    txn_col = _infer_txn_col(df_q)
    total_value, total_txn = _kpis(df_time_q, txn_col, df_q)

    mode = _mode_from_question(question)

    # Metric label for LLM narratives (value vs share; value defaults to PLN unless question indicates quantity/transactions)
    _q_l = (question or "").lower()
    share_stats = None
    if mode == "share":
        metric_label = "udział sprzedaży (%)"
    else:
        if any(k in _q_l for k in ["wolumen", "ilość", "ilosc", "szt", "quantity", "volume"]):
            metric_label = "wolumen sprzedaży (szt.)"
        elif any(k in _q_l for k in ["transakc", "paragon", "zamów", "zamow", "orders", "order count", "transactions"]):
            metric_label = "liczba transakcji"
        else:
            metric_label = "wartość sprzedaży (PLN)"

    tab_overview = tabs.get("overview") or st
    tab_insights = tabs.get("insights") or st
    timing_slot = None  # debug only

    # ---- Tab 1: Obraz całości
    with tab_overview:
        _t_tab1 = perf_counter()
        # KPI slots from router (if present)
        metric_value_slot = ((ui.get("overview") or {}).get("metric_value")) or None
        metric_qty_slot = ((ui.get("overview") or {}).get("metric_qty")) or None
        metric_txn_slot = ((ui.get("overview") or {}).get("metric_txn")) or None
        chart_slot = ((ui.get("overview") or {}).get("chart_slot")) or None

        (metric_value_slot or st).metric("Suma wartości", f"{total_value:,.0f}")
        
        # ✅ NOWE: Suma ilości (dopasowana do okresu z pytania, jeśli podano)
        _qty_total = None
        if qty_col and qty_col in df_q.columns and pd.api.types.is_numeric_dtype(df_q[qty_col]):
            try:
                _qty_total = float(pd.to_numeric(df_q[qty_col], errors="coerce").fillna(0).clip(lower=0).sum())
            except Exception:
                _qty_total = None

        (metric_qty_slot or st).metric("Suma ilości", "—" if _qty_total is None else f"{_qty_total:,.0f}")

        (metric_txn_slot or st).metric("Liczba transakcji", f"{int(total_txn):,}")

        st.markdown("")  # small breathing room

        df_time_main = _apply_topn_other(df_time_q, cat_col, cot_top_n, cot_include_other)

        def _is_other(_x: Any) -> bool:
            s = str(_x).strip().lower()
            return s in {"other", "inne"} or s.startswith("other") or s.startswith("inne")

        _t_build = perf_counter()
        if mode == "share":
            chart_title = _overview_chart_title("share", cat_col)
            chart_desc = _overview_chart_desc("share", cat_col)
            months_override = pd.date_range(q_start_m, q_end_m, freq="MS") if (q_start_m is not None and q_end_m is not None) else None
            chart, share_stats = _main_chart_share(df_time_main, cat_col, top_k=cot_top_n, include_other=cot_include_other, months_override=months_override, df_time_total=df_time_q)
            _timings["_main_chart_share"] = (perf_counter() - _t_build) * 1000.0
            stats_for_interp = {
                "mode": "share",
                "total_value": round(total_value, 2),
                "total_txn": int(total_txn),
                "n_categories": int(df_time_main[cat_col].nunique()),
            }

            # --- Driver insight (share): detect driver category + Δ pp and Top-10 share ---
            try:
                val_col = "__value"  # internal aggregated value column from _prep_time_df
                _dfv = df_time_main[[time_col, cat_col, val_col]].copy()
                _dfv = _dfv.dropna(subset=[time_col, cat_col])
                _dfv[val_col] = pd.to_numeric(_dfv[val_col], errors="coerce").fillna(0.0)
                _dfv = _dfv.sort_values(time_col)

                _by_cat_total = _dfv.groupby(cat_col, dropna=False)[val_col].sum().sort_values(ascending=False)
                _by_time_cat = _dfv.groupby([time_col, cat_col], dropna=False)[val_col].sum().reset_index()
                _by_time_total = _by_time_cat.groupby(time_col, dropna=False)[val_col].sum().rename("total").reset_index()
                _m = _by_time_cat.merge(_by_time_total, on=time_col, how="left")
                _m["share"] = _m[val_col] / _m["total"].replace(0, pd.NA)
                _m["share"] = _m["share"].fillna(0.0)

                _pivot = _m.pivot_table(index=time_col, columns=cat_col, values="share", aggfunc="sum", fill_value=0.0).sort_index()
                _cols = [c for c in _pivot.columns.tolist() if not _is_other(c)]
                _cols = _cols or _pivot.columns.tolist()

                if len(_pivot) > 0 and len(_cols) > 0:
                    _first = _pivot.iloc[0]
                    _last = _pivot.iloc[-1]
                    _drv = max(_cols, key=lambda c: abs(float(_last.get(c, 0.0)) - float(_first.get(c, 0.0))))
                    _s0 = float(_first.get(_drv, 0.0))
                    _s1 = float(_last.get(_drv, 0.0))

                    _top10 = [c for c in _by_cat_total.index.tolist() if not _is_other(c)][:10]
                    _top10 = _top10 or _cols
                    _top10_s0 = float(_first[_top10].sum()) if len(_top10) else 0.0
                    _top10_s1 = float(_last[_top10].sum()) if len(_top10) else 0.0

                    stats_for_interp.update({
                        "driver_dimension_value": str(_drv),
                        "driver_category_name": str(_drv),
                        "driver_start_share": _s0,
                        "driver_end_share": _s1,
                        "driver_delta_pp": (_s1 - _s0) * 100.0,
                        "top10_share_start": _top10_s0,
                        "top10_share_end": _top10_s1,
                        "top10_delta_pp": (_top10_s1 - _top10_s0) * 100.0,
                    })
            except Exception:
                pass
            # --- Interpretacja: dodatkowe kotwice udziałowe (dla walidatora i jakości narracji)
            # Uwaga: share_stats istnieje tylko w trybie udziałów (mode == 'share').
            _tmp_share_stats = locals().get('share_stats', None)
            _share_stats = _tmp_share_stats if isinstance(_tmp_share_stats, dict) else {}
            if stats_for_interp.get('mode') == 'share' and _share_stats:
                _top1_cat = _share_stats.get('top1_category')
                _top1_s = _first_not_none(_share_stats.get('top1_start_pct'))
                _top1_e = _first_not_none(_share_stats.get('top1_end_pct'))
                _top1_d = _first_not_none(_share_stats.get('top1_delta_pp'))
                _topN_s = _first_not_none(_share_stats.get('topN_start_pct'))
                _topN_e = _first_not_none(_share_stats.get('topN_end_pct'))
                _topN_d = _first_not_none(_share_stats.get('topN_delta_pp'))
                stats_for_interp.update({
                    'leader_dimension_value': _top1_cat,
                    'top1_dimension_value': _top1_cat,
                    'driver_dimension_value': _top1_cat,
                    'primary_dimension_value': _top1_cat,
                    'leader_category': _top1_cat,
                    'leader_share_start_pct': float(_top1_s) if _top1_s is not None else None,
                    'leader_share_end_pct': float(_top1_e) if _top1_e is not None else None,
                    'leader_share_delta_pp': float(_top1_d) if _top1_d is not None else None,
                    'topN_share_start_pct': float(_topN_s) if _topN_s is not None else None,
                    'topN_share_end_pct': float(_topN_e) if _topN_e is not None else None,
                    'topN_share_delta_pp': float(_topN_d) if _topN_d is not None else None,
                    # Canonical aliases for strict fallback / validators
                    'driver_category_name': _top1_cat,
                    'driver_start_pct': float(_top1_s) if _top1_s is not None else None,
                    'driver_end_pct': float(_top1_e) if _top1_e is not None else None,
                    'driver_delta_pp': float(_top1_d) if _top1_d is not None else None,
                    'top10_start_pct': float(_topN_s) if _topN_s is not None else None,
                    'top10_end_pct': float(_topN_e) if _topN_e is not None else None,
                    'top10_delta_pp': float(_topN_d) if _topN_d is not None else None,
                })
        else:
            chart_title = _overview_chart_title("value", cat_col)
            chart_desc = _overview_chart_desc("value", cat_col)
            months_override = pd.date_range(q_start_m, q_end_m, freq="MS") if (q_start_m is not None and q_end_m is not None) else None
            chart = _main_chart_value(df_time_main, cat_col, top_k=cot_top_n, months_override=months_override)
            _timings["_main_chart_value"] = (perf_counter() - _t_build) * 1000.0
            # find peak/trough months for total
            total_by_month = df_time_main.groupby("__month")["__value"].sum()
            peak_month = str(total_by_month.idxmax())[:10] if not total_by_month.empty else None
            trough_month = str(total_by_month.idxmin())[:10] if not total_by_month.empty else None
            stats_for_interp = {
                "mode": "value",
                "total_value": round(total_value, 2),
                "total_txn": int(total_txn),
                "peak_month": peak_month,
                "peak_value": float(total_by_month.max()) if not total_by_month.empty else None,
                "trough_month": trough_month,
                "trough_value": float(total_by_month.min()) if not total_by_month.empty else None,
                "n_categories": int(df_time_main[cat_col].nunique()),
            }

            # --- Driver insight (value): detect driver category + Δ PLN ---
            try:
                _dfv = df_time_main[[time_col, cat_col, val_col]].copy()
                _dfv = _dfv.dropna(subset=[time_col, cat_col])
                _dfv[val_col] = pd.to_numeric(_dfv[val_col], errors="coerce").fillna(0.0)
                _dfv = _dfv.sort_values(time_col)

                _by_cat_total = _dfv.groupby(cat_col, dropna=False)[val_col].sum().sort_values(ascending=False)
                _cats = [c for c in _by_cat_total.index.tolist() if not _is_other(c)]
                _drv = _cats[0] if _cats else (_by_cat_total.index[0] if len(_by_cat_total) else None)

                if _drv is not None:
                    _drv_df = _dfv[_dfv[cat_col] == _drv].groupby(time_col, dropna=False)[val_col].sum().reset_index().sort_values(time_col)
                    _v0 = float(_drv_df.iloc[0][val_col]) if len(_drv_df) else 0.0
                    _v1 = float(_drv_df.iloc[-1][val_col]) if len(_drv_df) else 0.0
                    stats_for_interp.update({
                        "driver_dimension_value": str(_drv),
                        "driver_category_name": str(_drv),
                        "driver_start_value": _v0,
                        "driver_end_value": _v1,
                        "driver_delta_pln": (_v1 - _v0),
                        "driver_total_value_period": float(_by_cat_total.loc[_drv]) if _drv in _by_cat_total.index else None,
                    })
            except Exception:
                pass

        stats_for_interp.update({
            "cat_col": str(cat_col),
            "dimension_label": dimension_label,
            "dimension_entity_label": dimension_entity,
            "n_dimension_values": int(df_time_main[cat_col].nunique()),
        })

        # Streamlit version compatibility: width='stretch' (new) vs width (old)
        # NOTE: do NOT import via "app.core" because the Streamlit multipage runtime
        # executes scripts in a way where "app" is not a Python package.
        from core.ui_safe import altair_chart_stretch
        _t_render_chart = perf_counter()
        # Guard: Streamlit expects an Altair chart-like object (with .to_dict()).
        _raw_chart = chart
        chart = _unwrap_altair_chart(chart)
        if chart is None:
            if debug_perf:
                st.error(
                    "❌ Nieprawidłowy obiekt wykresu (Tab 1 / overview) — pomijam render, aby uniknąć crasha."
                    f"\n\n• type: `{type(_raw_chart)}`\n• repr: `{_short_repr(_raw_chart)}`"
                )
            else:
                st.warning(
                    "⚠️ Wykres chwilowo niedostępny dla aktualnego zakresu dat. "
                    "Spróbuj odświeżyć lub zmienić zakres."
                )
        else:
            altair_chart_stretch((chart_slot or st), chart)
        _timings["render_chart"] = (perf_counter() - _t_render_chart) * 1000.0

        if debug_perf:
            timing_slot = st.empty()

        _overview_interp_cache = st.session_state.setdefault("__cot_overview_interp_cache_v1", {})

        _overview_interp_key = hashlib.sha1(
            json.dumps(
                {
                    "chart_title": chart_title,
                    "chart_desc": chart_desc,
                    "mode": _infer_interp_mode(chart_title, stats_for_interp or {}),
                    "stats": stats_for_interp or {},
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8", errors="ignore")
        ).hexdigest()

        dbg_cp(
            "overview.interp.cache_lookup",
            block_id="cot__overview_interpretation",
            from_path="overview._render_interpretation_ui",
            chart_title=chart_title,
            mode=_infer_interp_mode(chart_title, stats_for_interp or {}),
            stats_keys=list((stats_for_interp or {}).keys())[:12],
            cache_key=_overview_interp_key,
        )

        _force_cold_audit_run = _force_cold_overview_run_enabled()
        _cached_interp_payload = None if _force_cold_audit_run else _overview_interp_cache.get(_overview_interp_key)
        if isinstance(_cached_interp_payload, dict):
            interp = dict(_cached_interp_payload.get("interp") or {})
            _src = str(_cached_interp_payload.get("src") or "overview_cache")
            dbg_cp(
                "overview.interp.cache_hit",
                block_id="cot__overview_interpretation",
                chart_title=chart_title,
                src=_src,
                cache_key=_overview_interp_key,
                text_len=len(str(interp.get("one_sentence") or "")),
            )
        else:
            dbg_cp(
                "overview.interp.cold_start",
                block_id="cot__overview_interpretation",
                from_path="overview._interp_llm_json",
                chart_title=chart_title,
                mode=_infer_interp_mode(chart_title, stats_for_interp or {}),
                stats_keys=list((stats_for_interp or {}).keys())[:12],
                cache_key=_overview_interp_key,
            )
            interp, _src = _interp_llm_json(
                ctx=ctx,
                chart_title=chart_title,
                chart_desc=chart_desc,
                stats=stats_for_interp,
            )
            if (
                (not _force_cold_audit_run)
                and isinstance(interp, dict)
                and str(interp.get("one_sentence") or "").strip()
            ):
                _overview_interp_cache[_overview_interp_key] = {
                    "interp": dict(interp),
                    "src": _src,
                }
            if _force_cold_audit_run:
                _consume_force_cold_overview_run_flag()

        interp_meta = (interp.get("__meta") if isinstance(interp, dict) else None)

        dbg_cp(
            "overview.interp.return",
            block_id="cot__overview_interpretation",
            from_path="overview._interp_llm_json",
            chart_title=chart_title,
            src=_src,
            topic=((interp_meta or {}).get("topic") if isinstance(interp_meta, dict) else None),
            text_len=len(str((interp or {}).get("one_sentence") or "")),
            preview=str((interp or {}).get("one_sentence") or "")[:180],
        )
        interp_status_key = f"exec:v4:composition_over_time_interpretation:{chart_title}"
        interp_slot = ((ui.get("overview") or {}).get("interpretation_slot"))
        target = interp_slot or st

        _interp_render_key = hashlib.sha1(
            json.dumps(
                {
                    "chart_title": chart_title,
                    "src": _src,
                    "topic": ((interp_meta or {}).get("topic") if isinstance(interp_meta, dict) else None),
                    "one_sentence": str((interp or {}).get("one_sentence") or ""),
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8", errors="ignore")
        ).hexdigest()

        _last_interp_render_key = st.session_state.get("__cot_overview_last_render_key")
        _last_interp_repeat_log_key = st.session_state.get("__cot_overview_last_repeat_log_key")

        if _last_interp_render_key == _interp_render_key:
            if _last_interp_repeat_log_key != _interp_render_key:
                dbg_cp(
                    "overview.interp.render_repeat",
                    block_id="cot__overview_interpretation",
                    from_path="overview._render_interpretation_ui",
                    chart_title=chart_title,
                    src=_src,
                    topic=((interp_meta or {}).get("topic") if isinstance(interp_meta, dict) else None),
                    text_len=len(str((interp or {}).get("one_sentence") or "")),
                )
                st.session_state["__cot_overview_last_repeat_log_key"] = _interp_render_key
            _render_interpretation_ui(target, interp)
        else:
            dbg_cp(
                "overview.interp.render",
                block_id="cot__overview_interpretation",
                from_path="overview._render_interpretation_ui",
                chart_title=chart_title,
                src=_src,
                topic=((interp_meta or {}).get("topic") if isinstance(interp_meta, dict) else None),
                text_len=len(str((interp or {}).get("one_sentence") or "")),
            )
            _render_interpretation_ui(target, interp)
            st.session_state["__cot_overview_last_render_key"] = _interp_render_key
            st.session_state["__cot_overview_last_repeat_log_key"] = None

        _timings["tab1_overview_total"] = (perf_counter() - _t_tab1) * 1000.0

    # ---- Tab 2: Kluczowe insighty (5 bloków)
    with tab_insights:
        _t_tab2 = perf_counter()
        top_n = int(filters.get("cot_top_n") or 10)
        # keep the same defaulting behavior as in Tab 1 (session_state fallback)
        _inc = filters.get("cot_include_other")
        if _inc is None:
            _inc = st.session_state.get("cot_include_other", True)
        include_other = bool(_inc)

        blocks: List[Dict[str, Any]] = []

        _t_b = perf_counter()
        ch1, st1 = _block1_mix_share_topN(df_time_q, cat_col, top_n=top_n, include_other=include_other, df_time_total=df_time_q)
        _timings["block_mix_share_build"] = (perf_counter() - _t_b) * 1000.0
        blocks.append({"id": "cot__mix_share_topN", "title": "🔎 Jak wygląda dynamika udziałów w czasie?", "desc": ("Udział TopN względem całości (vs TOTAL). " + ("Z włączonym 'Other' wykres sumuje się do 100%." if include_other else "Po wyłączeniu 'Other' oś Y auto-skaluje się do sumy udziałów TopN.")), "chart": ch1, "stats": st1})

        _t_b = perf_counter()
        ch2, st2 = _block2_winners_losers(df_time_q, cat_col, top_n=top_n, include_other=include_other, df_time_total=df_time_q)
        _timings["block_winners_losers_build"] = (perf_counter() - _t_b) * 1000.0
        blocks.append({"id": "cot__winners_losers", "title": "📈 Kto zyskuje, a kto traci udział w czasie?", "desc": f"Linie trendu udziałów (%) dla TopN elementów wymiaru {dimension_label}.", "chart": ch2, "stats": st2})

        _t_b = perf_counter()
        ch3, score_df, st3 = _block3_seasonality(df_time, cat_col, top_n=top_n)
        _timings["block_seasonality_build"] = (perf_counter() - _t_b) * 1000.0
        blocks.append({"id": "cot__seasonality", "title": "🧭 Czy jest sezonowość sprzedaży i czy jest ona stabilna?", "desc": f"Analizujemy czystą sezonowość wartości sprzedaży dla Top10 elementów wymiaru {dimension_label} wg wartości sprzedaży w całym okresie. Dla dłuższych serii używany jest STL, a dla krótszych zakresów bezpieczny fallback odchylenia po usunięciu trendu. Heatmapa pokazuje kierunek i siłę odchyleń sezonowych w czasie, a scorecard poniżej porównuje siłę i stabilność sezonowości między elementami wymiaru.", "chart": ch3, "stats": st3, "score_df": score_df})

        _t_b = perf_counter()
        ch4, st4 = _block4_slope_start_end(df_time_q, cat_col, top_n=top_n, include_other=include_other, winners_k=3, losers_k=3, df_time_total=df_time_q)
        _timings["block_slope_build"] = (perf_counter() - _t_b) * 1000.0
        blocks.append({"id": "cot__start_end", "title": "🏁 Kto najbardziej zyskuje, a kto traci udział między startem i końcem okresu?", "desc": "Ten blok porównuje udział **Start ➜ Koniec** w wybranym okresie i pokazuje największych **winners** (Δ udział ↑) oraz **losers** (Δ udział ↓).", "chart": ch4, "stats": st4})

        _t_b = perf_counter()
        ch5, st5 = _block5_concentration(df_time_q, cat_col, top_n=top_n, include_other=include_other)
        _timings["block_concentration_build"] = (perf_counter() - _t_b) * 1000.0
        blocks.append({
            "id": "cot__concentration",
            "title": "🧲 Czy struktura się konsoliduje czy dywersyfikuje w czasie?",
            "desc": "Wykres pokazuje udział **Top-3** i **Top-5** elementów wymiaru w czasie. Wzrost linii oznacza **konsolidację** (większa koncentracja), spadek — **dywersyfikację**.",
            "chart": ch5,
            "stats": st5,
        })

        # LLM: batch first, then direct single-block fallback (not the same batch adapter)
        _exec_by_label: Dict[str, str] = {}
        _exec_blocks = []
        _exec_source_by_id: Dict[str, str] = {}
        _stats_by_label: Dict[str, Dict[str, Any]] = {}

        def _strip_seasonality_strength(_obj):
            """Remove seasonality *strength* fields from LLM payload (keep for visuals/scorecard)."""
            if isinstance(_obj, dict):
                out = {}
                for k, v in _obj.items():
                    kl = str(k).lower()
                    if ("seasonality" in kl) and ("strength" in kl):
                        continue
                    out[k] = _strip_seasonality_strength(v)
                return out
            if isinstance(_obj, list):
                return [_strip_seasonality_strength(x) for x in _obj]
            return _obj

        def _normalize_exec_map(_raw_map: Any) -> Dict[str, str]:
            out: Dict[str, str] = {}
            if isinstance(_raw_map, dict):
                for k, v in _raw_map.items():
                    if isinstance(v, dict):
                        _text_val = (
                            v.get("text")
                            or v.get("takeaway")
                            or ((v.get("meta") or {}).get("text") if isinstance(v.get("meta"), dict) else "")
                            or ""
                        )
                        out[str(k)] = _normalize_exec_takeaway_text(_text_val)
                    else:
                        out[str(k)] = _normalize_exec_takeaway_text(v)
            return out

        def _normalize_exec_source_map(_raw_map: Any, default_src: str = "llm_batch_raw") -> Dict[str, str]:
            out: Dict[str, str] = {}
            if isinstance(_raw_map, dict):
                for k, v in _raw_map.items():
                    _bid = str(k)
                    if isinstance(v, dict):
                        _text_val = _normalize_exec_takeaway_text(
                            v.get("text")
                            or v.get("takeaway")
                            or ((v.get("meta") or {}).get("text") if isinstance(v.get("meta"), dict) else "")
                            or ""
                        )
                        _src_val = str(
                            v.get("src")
                            or ((v.get("meta") or {}).get("src") if isinstance(v.get("meta"), dict) else "")
                            or (default_src if _text_val else "")
                        )
                    else:
                        _text_val = _normalize_exec_takeaway_text(v)
                        _src_val = default_src if _text_val else ""
                    if _text_val:
                        out[_bid] = _src_val
            return out

        def _build_exec_batch_cache_payload(_exec_map: Dict[str, str], _src_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
            out: Dict[str, Dict[str, Any]] = {}
            for _bid, _txt in (_exec_map or {}).items():
                _txt_norm = _normalize_exec_takeaway_text(_txt)
                if not _txt_norm:
                    continue
                out[str(_bid)] = {
                    "text": _txt_norm,
                    "src": str((_src_map or {}).get(str(_bid)) or "llm_batch_raw"),
                }
            return out

        def _exec_block_id(_blk: Dict[str, Any]) -> str:
            return str(_blk.get("id") or _blk.get("label") or "")

        def _get_missing_exec_block_ids(_blks: List[Dict[str, Any]], _exec_map: Dict[str, str]) -> List[str]:
            missing: List[str] = []
            for _blk in (_blks or []):
                _bid = _exec_block_id(_blk)
                if not _bid:
                    continue
                if not str((_exec_map or {}).get(_bid) or "").strip():
                    missing.append(_bid)
            return missing

        def _is_exec_map_complete(_blks: List[Dict[str, Any]], _exec_map: Dict[str, str]) -> bool:
            return len(_get_missing_exec_block_ids(_blks, _exec_map)) == 0

        def _build_exec_single_prompt(_blk: Dict[str, Any]) -> str:
            _bid = str(_blk.get("id") or _blk.get("label") or "unknown")
            _title = str(_blk.get("title") or "")
            _desc = str(_blk.get("desc") or "")
            _stats = _compact_exec_stats(_bid, _blk.get("stats") or {})
            _contract_profile = _build_exec_contract_profile(_bid, _stats)
            _contract_mode = str(_contract_profile.get("contract_mode") or "full")
            _numbers_rule = "minimum 2 liczby z JSON"
            if _contract_mode == "reduced":
                _numbers_rule = "1-2 liczby z JSON, preferencyjnie 2, bez zgadywania brakujacej kotwicy"
            elif _contract_mode in {"sparse", "insufficient"}:
                _numbers_rule = "1-2 liczby z JSON; jesli masz tylko 1 wiarygodna liczbe, nie dopisuj drugiej na sile"
            _extra = ""
            if _bid == "cot__mix_share_topN":
                _extra = (
                    "\nDla cot__mix_share_topN zdanie 1 MUSI podać: Top-N start→end, delta pp Top-N oraz liczbową zmianę lidera w pp, jeśli istnieje w JSON."
                    "\nJeśli delta lidera nie istnieje w JSON, wolno napisać tylko: 'lider traci pozycję w miksie', ale bez placeholdera '—'."
                    "\nZdanie 2 MUSI zawierać obronę lidera / rdzeń portfela / realokację wsparcia / rolę kategorii w miksie."
                )
            elif _bid == "cot__winners_losers":
                _extra = (
                    "\nDla cot__winners_losers zdanie 1 MUSI nazwać mechanizm przez 'transfer udziału' albo 'przejęcie popytu'."
                    "\nZdanie 2 MUSI zawierać jednocześnie: jeden lever dla winnera oraz jedną diagnozę albo lever dla loser."
                    "\nWhy-now MUSI wynikać z okna przejęcia popytu albo kosztu utraty udziału."
                )
            elif _bid == "cot__start_end":
                _extra = (
                    "\nDla cot__start_end why-now MUSI wynikać z trwałego przesunięcia popytu albo ryzyka utraty udziału w nowym układzie kategorii."
                )
            return (
                f"Blok: {_bid}\n"
                f"Tytuł: {_title}\n"
                f"Contract profile: {_mbb_json_dumps_safe(_contract_profile)}\n"
                f"JSON: {_mbb_json_dumps_safe(_stats)}\n\n"
                "Zwróć wyłącznie czysty tekst.\n"
                "Nie zwracaj JSON.\n"
                "Nie zwracaj {}.\n"
                "Nie zwracaj listy.\n"
                "Nie zwracaj markdown.\n"
                "Napisz dokładnie 2 krótkie zdania po polsku.\n"
                f"Zdanie 1: podaj {_numbers_rule} i nazwij mechanizm biznesowy.\n"
                "Zdanie 2: zacznij od 'Decyzja:' albo 'Dlatego:' i podaj jedno kierunkowe działanie biznesowe.\n"
                "Jeśli nie możesz napisać 2 zdań, zwróć 1 niepuste zdanie zamiast pustego obiektu.\n"
                f"{_exec_contract_prompt_guidance(_bid, _contract_profile)}\n"
                "Bez bulletów, bez nagłówków, bez placeholderów."
                f"{_extra}"
            )

        def _get_exec_takeaway_single_direct(_blk: Dict[str, Any]) -> str:
            _label = str(_blk.get("label") or _blk.get("id") or "unknown")
            _model = str(((ctx or {}).get("openai_model") or "gpt-4o-mini"))

            def _call(_prompt: str, _attempt: str = "") -> str:
                if _label == "cot__seasonality":
                    dbg_cp(
                        "seasonality.llm.request",
                        block_id="cot__seasonality",
                        attempt=_attempt,
                        prompt_len=len(str(_prompt or "")),
                        prompt_preview=str(_prompt or "")[:800],
                    )

                _resp = llm_fn(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Jesteś analitykiem biznesowym. "
                                "Pisz po polsku, konkretnie, liczbowo i executive-style. "
                                "Zwracaj tylko gotowy tekst."
                            ),
                        },
                        {"role": "user", "content": _prompt},
                    ],
                    model=_model,
                    temperature=0.1,
                    max_tokens=180,
                    ctx=ctx,
                    payload={"kind": "cot_exec_takeaway_single_direct", "block_id": _label},
                )

                if _label == "cot__seasonality":
                    dbg_cp(
                        "seasonality.llm.response",
                        block_id="cot__seasonality",
                        attempt=_attempt,
                        response_type=type(_resp).__name__,
                        is_none=_resp is None,
                        response_preview=str(_resp)[:800],
                    )

                _txt_raw = _extract_text_from_llm_response(_resp)
                if (not _txt_raw) and isinstance(_resp, dict):
                    _txt_raw = _et_extract_text_like_overview(_resp)
                _txt_raw = str(_txt_raw or "")

                if _label == "cot__seasonality":
                    dbg_cp(
                        "seasonality.llm.parse",
                        block_id="cot__seasonality",
                        attempt=_attempt,
                        parsed_len=len(str(_txt_raw or "")),
                        parsed_preview=str(_txt_raw or "")[:400],
                        parsed_is_empty=(not bool(str(_txt_raw or "").strip())),
                    )

                _txt = " ".join(str(_txt_raw or "").strip().split())

                if _label == "cot__seasonality":
                    dbg_cp(
                        "seasonality.llm.parse_normalized",
                        block_id="cot__seasonality",
                        attempt=_attempt,
                        normalized_len=len(_txt),
                        normalized_preview=_txt[:400],
                        normalized_is_empty=(not bool(_txt)),
                    )

                return _txt

            _prompt1 = _build_exec_single_prompt(_blk)
            _txt1 = _call(_prompt1, _attempt="prompt1")

            if _label == "cot__seasonality":
                dbg_cp(
                    "seasonality.llm.gate_after_prompt1",
                    block_id="cot__seasonality",
                    text_len=len(str(_txt1 or "")),
                    preview=str(_txt1 or "")[:300],
                    passed=bool(str(_txt1 or "").strip()),
                )

            if _txt1:
                return _txt1

            _stats = _blk.get("stats") or {}
            _profile2 = _build_exec_contract_profile(_label, _stats)
            _numbers_rule2 = "minimum 2 liczby"
            if str(_profile2.get("contract_mode") or "full") == "reduced":
                _numbers_rule2 = "1-2 liczby, preferencyjnie 2"
            elif str(_profile2.get("contract_mode") or "full") in {"sparse", "insufficient"}:
                _numbers_rule2 = "1-2 liczby; jesli masz tylko 1 wiarygodna liczbe, nie dopisuj drugiej"
            _prompt2 = (
                "Zwróć wyłącznie czysty tekst po polsku.\n"
                "Nie zwracaj JSON.\n"
                "Nie zwracaj {}.\n"
                "Nie zwracaj listy.\n"
                "Napisz dokładnie 2 krótkie zdania wyłącznie na podstawie JSON.\n"
                f"Zdanie 1: podaj {_numbers_rule2} i nazwij mechanizm biznesowy.\n"
                "Zdanie 2: zacznij od 'Decyzja:' i podaj jedno kierunkowe działanie biznesowe wraz z why-now.\n"
                "Jeśli nie możesz napisać 2 zdań, zwróć 1 niepuste zdanie zamiast pustego obiektu.\n"
                f"Contract profile: {_mbb_json_dumps_safe(_profile2)}\n"
                f"JSON: {_mbb_json_dumps_safe(_stats)}"
            )

            _txt2 = _call(_prompt2, _attempt="prompt2")

            if _label == "cot__seasonality":
                dbg_cp(
                    "seasonality.llm.gate_after_prompt2",
                    block_id="cot__seasonality",
                    text_len=len(str(_txt2 or "")),
                    preview=str(_txt2 or "")[:300],
                    passed=bool(str(_txt2 or "").strip()),
                )

            return _txt2

        for _b in blocks:
            _desc = _b.get("desc", "")
            _desc = (
                _desc
                + "\n\nWYMÓG DLA EXECUTIVE TAKEAWAY:"
                + "\n- Zwróć wyłącznie czysty tekst."
                + "\n- Nie zwracaj JSON."
                + "\n- Nie zwracaj {}."
                + "\n- Nie zwracaj listy."
                + "\n- Napisz dokładnie 2 krótkie zdania po polsku."
                + "\n- Zdanie 1: podaj 2–3 najważniejsze liczby wynikające bezpośrednio ze statystyk poniżej i nazwij mechanizm biznesowy."
                + "\n- Zdanie 2: zacznij od 'Dlatego:' albo 'Decyzja:' i podaj jedną konkretną rekomendację biznesową wraz z why-now."
                + "\n- Nie używaj placeholderów typu 'kategoria A' / 'category A'."
                + "\n- Jeśli nie możesz spełnić formatu, zwróć 1 niepuste zdanie zamiast pustego obiektu."
                + "\n- Jeśli dane sezonowości są słabe, napisz ostrożny wniosek zamiast zgadywać."
            ).strip()

            _label = str(_b.get("id") or _b.get("label") or "")
            _stats = _compact_exec_stats(_label, _b.get("stats") or {})
            _stats_by_label[_label] = _stats if isinstance(_stats, dict) else {}
            if _b.get("id") == "cot__seasonality":
                _desc = (
                    _desc
                    + "\n- Dla sezonowości użyj kategorii top-1/top-2, wagi sezonowości, amplitudy i werdyktu, jeśli są dostępne."
                )

            _exec_blocks.append({
                "label": _label,
                "id": _label,
                "title": str(_b.get("title") or ""),
                "desc": _desc,
                "stats": _stats_by_label[_label],
            })

        if not _exec_blocks:
            dbg_cp("exec_takeaway.no_exec_blocks_guard", total_exec_blocks=0)
            _exec_by_label = {}
            _exec_source_by_id = {}
            _exec_results = {}
        else:
            _exec_cache_payload = {
                "engine_version": _EXEC_FINAL_CACHE_ENGINE_VERSION,
                "blocks": [
                    {
                        "id": str(b.get("id") or ""),
                        "title": str(b.get("title") or ""),
                        "desc": str(b.get("desc") or ""),
                        "stats": b.get("stats") or {},
                    }
                    for b in (_exec_blocks or [])
                ],
            }
            _exec_cache_key = hashlib.sha1(
                json.dumps(_exec_cache_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

            _exec_batch_cache = st.session_state.setdefault("__cot_exec_batch_cache_v1", {})
            _exec_final_cache = st.session_state.setdefault("__cot_exec_final_cache_v1", {})
            _force_cold_audit_run = _force_cold_exec_run_enabled()
            _final_cache_present = bool(_exec_cache_key in _exec_final_cache)
            _batch_cache_present = bool(_exec_cache_key in _exec_batch_cache)

            if _force_cold_audit_run:
                _set_exec_last_run_mode(
                    "cold_requested",
                    cache_key=_exec_cache_key,
                    engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                    blocks_count=len(_exec_blocks),
                )
                dbg_cp(
                    "exec_takeaway.cold_start",
                    cache_key=_exec_cache_key,
                    blocks_count=len(_exec_blocks),
                    engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                    final_cache_present=_final_cache_present,
                    batch_cache_present=_batch_cache_present,
                )
                if _final_cache_present:
                    dbg_cp(
                        "exec_takeaway.final_cache_bypassed_force_cold",
                        cache_key=_exec_cache_key,
                        blocks_count=len(_exec_blocks),
                        engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                    )
                if _batch_cache_present:
                    dbg_cp(
                        "exec_takeaway.batch_cache_bypassed_force_cold",
                        cache_key=_exec_cache_key,
                        blocks_count=len(_exec_blocks),
                        engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                    )

            _batch_raw = {}
            _exec_by_label = {}
            _exec_source_by_id = {}
            _missing_exec_ids: List[str] = []
            _exec_final_by_id: Dict[str, Dict[str, str]] = {}
            _final_cache_hit = False

            _cached_final_raw = None if _force_cold_audit_run else _exec_final_cache.get(_exec_cache_key)
            if isinstance(_cached_final_raw, dict):
                for _k, _v in _cached_final_raw.items():
                    if not isinstance(_v, dict):
                        continue
                    _txt = _normalize_exec_takeaway_text(_v.get("text"))
                    if not _txt:
                        continue
                    _payload_norm = dict(_v)
                    _payload_norm["text"] = _txt
                    _payload_norm["src"] = str(
                        _v.get("src")
                        or ((_v.get("meta") or {}).get("src") if isinstance(_v.get("meta"), dict) else "")
                        or "unknown"
                    )
                    _payload_norm["gate_reason"] = str(_v.get("gate_reason") or "")
                    _exec_final_by_id[str(_k)] = _payload_norm

                _missing_final_ids = _get_missing_exec_block_ids(
                    _exec_blocks,
                    {str(k): str((v or {}).get("text") or "") for k, v in _exec_final_by_id.items()},
                )
                if _missing_final_ids:
                    dbg_cp(
                        "exec_takeaway.final_cache_partial_miss",
                        cache_key=_exec_cache_key,
                        blocks_count=len(_exec_blocks),
                        missing_blocks=_missing_final_ids,
                        cached_non_empty=len(_exec_final_by_id),
                    )
                    _exec_final_cache.pop(_exec_cache_key, None)
                    _exec_final_by_id = {}
                else:
                    _final_cache_audit = _inspect_exec_final_cache_payload_map(_exec_final_by_id)
                    if bool(_final_cache_audit.get("poisoned")):
                        dbg_cp(
                            "exec_takeaway.final_cache_poison_detected",
                            cache_key=_exec_cache_key,
                            blocks_count=len(_exec_blocks),
                            reasons=_final_cache_audit.get("reasons"),
                            final_src_by_block=_final_cache_audit.get("final_src_by_block"),
                        )
                        _exec_final_cache.pop(_exec_cache_key, None)
                        _exec_final_by_id = {}
                    else:
                        _final_cache_hit = True
                        _set_exec_last_run_mode(
                            "warm_cache",
                            cache_key=_exec_cache_key,
                            engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                            blocks_count=len(_exec_blocks),
                        )
                        dbg_cp(
                            "exec_takeaway.final_cache_hit",
                            cache_key=_exec_cache_key,
                            blocks_count=len(_exec_blocks),
                        )
                        _exec_by_label = {
                            str(k): str((v or {}).get("text") or "")
                            for k, v in _exec_final_by_id.items()
                        }
                        _exec_source_by_id = {
                            str(k): str((v or {}).get("src") or "unknown")
                            for k, v in _exec_final_by_id.items()
                        }
                        _exec_results = _build_exec_results(_exec_by_label, _exec_source_by_id, raw_map=_batch_raw)
                        _exec_results_built = True

            if (not _final_cache_hit) and (not _force_cold_audit_run) and (_exec_cache_key in _exec_batch_cache):
                _batch_raw = _exec_batch_cache.get(_exec_cache_key) or {}
                _set_exec_last_run_mode(
                    "warm_batch_cache",
                    cache_key=_exec_cache_key,
                    engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                    blocks_count=len(_exec_blocks),
                )
                dbg_cp(
                    "exec_takeaway.batch_cache_hit",
                    cache_key=_exec_cache_key,
                    blocks_count=len(_exec_blocks),
                )
                _exec_by_label = _normalize_exec_map(_batch_raw)
                _exec_source_by_id = _normalize_exec_source_map(_batch_raw, default_src="llm_batch_raw")
                _missing_exec_ids = _get_missing_exec_block_ids(_exec_blocks, _exec_by_label)

                if _missing_exec_ids:
                    dbg_cp(
                        "exec_takeaway.batch_cache_partial_miss",
                        cache_key=_exec_cache_key,
                        blocks_count=len(_exec_blocks),
                        missing_blocks=_missing_exec_ids,
                        cached_non_empty=len(_exec_source_by_id),
                    )
                    if len(_missing_exec_ids) == len(_exec_blocks):
                        dbg_cp(
                            "exec_takeaway.batch_cache_invalidate_empty_after_normalize",
                            cache_key=_exec_cache_key,
                            blocks_count=len(_exec_blocks),
                        )
                        _exec_batch_cache.pop(_exec_cache_key, None)
                        _batch_raw = {}
                        _exec_by_label = {}
                        _exec_source_by_id = {}
                else:
                    dbg_cp(
                        "exec_takeaway.batch_cache_complete",
                        cache_key=_exec_cache_key,
                        blocks_count=len(_exec_blocks),
                    )

            if (not _final_cache_hit) and (not _exec_by_label):
                _t_llm = perf_counter()
                _batch_raw = get_exec_takeaways_llm(
                    ctx=ctx,
                    intent="composition_over_time_key_insight",
                    blocks=_exec_blocks,
                ) or {}
                _timings["llm_exec_takeaways_batch"] = (perf_counter() - _t_llm) * 1000.0

                _exec_by_label = _normalize_exec_map(_batch_raw)
                _exec_source_by_id = _normalize_exec_source_map(_batch_raw, default_src="llm_batch_raw")

                if _force_cold_audit_run:
                    _set_exec_last_run_mode(
                        "cold_confirmed",
                        cache_key=_exec_cache_key,
                        engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                        blocks_count=len(_exec_blocks),
                        batch_ms=round(float(_timings.get("llm_exec_takeaways_batch") or 0.0), 2),
                        filled_blocks=sorted(list(_exec_by_label.keys())),
                    )
                    dbg_cp(
                        "exec_takeaway.cold_confirmed",
                        cache_key=_exec_cache_key,
                        blocks_count=len(_exec_blocks),
                        engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                        filled_blocks=sorted(list(_exec_by_label.keys())),
                        batch_ms=round(float(_timings.get("llm_exec_takeaways_batch") or 0.0), 2),
                    )

                if not _exec_by_label:
                    dbg_cp(
                        "exec_takeaway.batch_raw_empty_after_fresh_call",
                        cache_key=_exec_cache_key,
                        blocks_count=len(_exec_blocks),
                    )
                    if _force_cold_audit_run:
                        _set_exec_last_run_mode(
                            "cold_empty",
                            cache_key=_exec_cache_key,
                            engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                            blocks_count=len(_exec_blocks),
                            batch_ms=round(float(_timings.get("llm_exec_takeaways_batch") or 0.0), 2),
                        )
                        dbg_cp(
                            "exec_takeaway.cold_empty_after_fresh_call",
                            cache_key=_exec_cache_key,
                            blocks_count=len(_exec_blocks),
                            engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                            batch_ms=round(float(_timings.get("llm_exec_takeaways_batch") or 0.0), 2),
                        )

            _missing_exec_ids = _get_missing_exec_block_ids(_exec_blocks, _exec_by_label) if not _final_cache_hit else []

            if (not _final_cache_hit) and _missing_exec_ids:
                _missing_exec_ids_set = set(_missing_exec_ids)
                _refilled_block_ids: List[str] = []
                dbg_cp(
                    "exec_takeaway.partial_refill_start",
                    cache_key=_exec_cache_key,
                    missing_blocks=_missing_exec_ids,
                    blocks_count=len(_exec_blocks),
                )

                for _blk in (_exec_blocks or []):
                    _blk_id = _exec_block_id(_blk)
                    if _blk_id not in _missing_exec_ids_set:
                        continue

                    _single_text = " ".join(str(_get_exec_takeaway_single_direct(_blk) or "").strip().split())
                    if _single_text:
                        _exec_by_label[_blk_id] = _single_text
                        _exec_source_by_id[_blk_id] = "llm_single_refill"
                        _refilled_block_ids.append(_blk_id)
                    else:
                        dbg_cp(
                            "exec_takeaway.partial_refill_empty_block",
                            block_id=_blk_id,
                            cache_key=_exec_cache_key,
                        )

                _missing_exec_ids = _get_missing_exec_block_ids(_exec_blocks, _exec_by_label)
                dbg_cp(
                    "exec_takeaway.partial_refill_done",
                    cache_key=_exec_cache_key,
                    refilled_blocks=_refilled_block_ids,
                    still_missing=_missing_exec_ids,
                )

            if (not _final_cache_hit) and (not _force_cold_audit_run) and _is_exec_map_complete(_exec_blocks, _exec_by_label):
                _exec_batch_cache[_exec_cache_key] = _build_exec_batch_cache_payload(_exec_by_label, _exec_source_by_id)
            elif (not _final_cache_hit) and (not _force_cold_audit_run):
                _exec_batch_cache.pop(_exec_cache_key, None)
                dbg_cp(
                    "exec_takeaway.batch_cache_not_stored_incomplete",
                    cache_key=_exec_cache_key,
                    missing_blocks=_get_missing_exec_block_ids(_exec_blocks, _exec_by_label),
                    blocks_count=len(_exec_blocks),
                )

            if bool(st.session_state.get("__cot_exec_dbg_on")) and (not _final_cache_hit):
                dbg_cp(
                    "seasonality.batch_after_normalize",
                    block_id="cot__seasonality",
                    batch_keys=sorted(list(_exec_by_label.keys())) if isinstance(_exec_by_label, dict) else [],
                    has_entry=("cot__seasonality" in _exec_by_label) if isinstance(_exec_by_label, dict) else False,
                    text_len=len(str((_exec_by_label or {}).get("cot__seasonality") or "")),
                    preview=str((_exec_by_label or {}).get("cot__seasonality") or "")[:240],
                )

            _exec_results = _build_exec_results(_exec_by_label, _exec_source_by_id, raw_map=_batch_raw)
            _exec_results_built = True

        _all_exec_results_empty = not any(
            str((v or {}).get("text") or "").strip()
            for v in (_exec_results or {}).values()
        )

        if _all_exec_results_empty and _exec_results_built:
            dbg_cp(
                "exec_takeaway.global_empty_exec_results_fallback",
                empty_labels=list((_exec_results or {}).keys()),
            )

        _exec_llm_share_rows: List[Dict[str, Any]] = []
        _audit_path = "warm_cache" if _final_cache_hit else "cold_path"
        for _b in blocks:
            _bid = str(_b.get("id") or "")
            if not _bid:
                continue

            _cached_final_payload = _exec_final_by_id.get(_bid) if _final_cache_hit else None
            if isinstance(_cached_final_payload, dict) and str(_cached_final_payload.get("text") or "").strip():
                _cached_meta_extra = _exec_final_meta_extra_from_payload(_cached_final_payload)
                _set_exec_final(
                    _b,
                    text=str(_cached_final_payload.get("text") or ""),
                    src=str(_cached_final_payload.get("src") or "unknown"),
                    gate_reason=str(_cached_final_payload.get("gate_reason") or ""),
                    meta_extra=_cached_meta_extra,
                )
                _cached_row = _exec_audit_row_from_final_payload(
                    _bid,
                    _cached_final_payload,
                    _stats_by_label.get(_bid) or {},
                    audit_path=_audit_path,
                )
                _exec_llm_share_rows.append(_cached_row)
                if _force_cold_audit_run or bool(st.session_state.get("__cot_exec_dbg_on")):
                    dbg_cp(
                        "exec_takeaway.audit_block",
                        block_id=_bid,
                        audit_path=_audit_path,
                        final_src=_cached_row.get("final_src"),
                        contract_mode=_cached_row.get("contract_mode"),
                        used_repair=_cached_row.get("used_repair"),
                        hard_reasons=_cached_row.get("hard_reasons"),
                        soft_reasons=_cached_row.get("soft_reasons"),
                        selected_by=_cached_row.get("selected_by"),
                    )
                continue

            _payload = (_exec_results or {}).get(_bid) or {}
            _candidate_text = _normalize_exec_takeaway_text(_payload.get("text"))
            _candidate_src = str(_payload.get("src") or "")
            _candidate_meta = _payload.get("meta") if isinstance(_payload.get("meta"), dict) else {}
            _candidate_forced_by_block_validator = bool(
                _candidate_meta.get("forced_by_block_validator")
                or _candidate_src == "deterministic_forced_by_block_validator"
            )
            _candidate_forced_gate_reasons = list(_candidate_meta.get("forced_gate_reasons") or [])

            _block_stats = _stats_by_label.get(_bid) or {}
            _contract_profile = _build_exec_contract_profile(_bid, _block_stats)
            _contract_mode = str(_contract_profile.get("contract_mode") or "full")
            _llm_block_audit = _validate_exec_takeaway_by_block_detail(
                _candidate_text,
                _bid,
                _block_stats,
            )
            _passed_block_gate = bool(_llm_block_audit.get("hard_ok"))
            _block_gate_reasons = list(_llm_block_audit.get("reasons") or [])
            _block_hard_reasons = list(_llm_block_audit.get("hard_reasons") or [])
            _block_soft_reasons = list(_llm_block_audit.get("soft_reasons") or [])
            if _candidate_forced_by_block_validator:
                _passed_block_gate = False

            _det_text = _normalize_exec_takeaway_text(_fallback_from_stats(_b))
            _det_block_audit = _validate_exec_takeaway_by_block_detail(
                _det_text,
                _bid,
                _block_stats,
            )
            _det_passed_gate = bool(_det_block_audit.get("hard_ok"))
            _det_gate_reasons = list(_det_block_audit.get("reasons") or [])
            _det_hard_reasons = list(_det_block_audit.get("hard_reasons") or [])
            _det_soft_reasons = list(_det_block_audit.get("soft_reasons") or [])

            _llm_eval = _quick_narrative_score(
                _candidate_text,
                block_id=_bid,
                mode="exec_takeaway",
                stats=_block_stats,
            )
            _det_eval = _quick_narrative_score(
                _det_text,
                block_id=_bid,
                mode="exec_takeaway",
                stats=_block_stats,
            )

            _llm_score = int(_llm_eval.get("score") or 0)
            _det_score = int(_det_eval.get("score") or 0)
            _llm_penalties = {str(x) for x in (_llm_eval.get("penalties") or [])}
            _det_penalties = {str(x) for x in (_det_eval.get("penalties") or [])}
            _llm_bonus_obj = _exec_selector_quality_bonus(_candidate_text, _bid, _block_stats)
            _det_bonus_obj = _exec_selector_quality_bonus(_det_text, _bid, _block_stats)
            _llm_score += int(_llm_bonus_obj.get("bonus") or 0)
            _det_score += int(_det_bonus_obj.get("bonus") or 0)
            _llm_has_placeholder = _has_exec_numeric_placeholder(_candidate_text)
            _det_has_placeholder = _has_exec_numeric_placeholder(_det_text)

            if _llm_has_placeholder:
                _llm_penalties.add("placeholder_numeric")
                _llm_score = max(0, _llm_score - 2)
            if _det_has_placeholder:
                _det_penalties.add("placeholder_numeric")
                _det_score = max(0, _det_score - 2)

            if not _passed_block_gate and _candidate_text:
                _llm_score = max(0, _llm_score - 2)
            elif _block_soft_reasons and _candidate_text:
                _llm_score = max(0, _llm_score - 1)
            if not _det_passed_gate and _det_text:
                _det_score = max(0, _det_score - 2)
            elif _det_soft_reasons and _det_text:
                _det_score = max(0, _det_score - 1)

            if _contract_mode in {"reduced", "sparse"} and _candidate_text and _passed_block_gate:
                _llm_score += 1
            if _contract_mode in {"reduced", "sparse"} and _det_text and _det_passed_gate:
                _det_score = max(0, _det_score)

            _critical_penalties = {
                "placeholder_numeric",
                "generic_action",
                "descriptive_with_no_why_now",
            }
            _llm_critical_penalties = sorted(_llm_penalties.intersection(_critical_penalties))
            _det_critical_penalties = sorted(_det_penalties.intersection(_critical_penalties))
            _llm_has_why_now = bool(_detect_exec_why_now_tokens(_candidate_text))
            _det_has_why_now = bool(_detect_exec_why_now_tokens(_det_text))
            _llm_tie_prefer_blocks = {
                "cot__seasonality",
                "cot__start_end",
                "cot__concentration",
                "cot__winners_losers",
            }

            _select_llm = False
            _selection_reason = ""
            if _candidate_text and _passed_block_gate and not _llm_critical_penalties:
                if _llm_score > _det_score:
                    _select_llm = True
                    _selection_reason = "llm_score_gt_det_score"
                elif (
                    _contract_mode in {"reduced", "sparse"}
                    and _llm_score >= max(4, _det_score - 1)
                    and int(_llm_bonus_obj.get("bonus") or 0) >= int(_det_bonus_obj.get("bonus") or 0)
                    and not _llm_has_placeholder
                ):
                    _select_llm = True
                    _selection_reason = "llm_preferred_for_sparse_contract"
                elif (
                    _bid == "cot__concentration"
                    and _contract_mode in {"reduced", "sparse"}
                    and _llm_score >= max(4, _det_score - 1)
                    and not _llm_has_placeholder
                    and not _block_hard_reasons
                    and {
                        "named_mechanism",
                        "explicit_business_lever",
                    }.issubset(set(_llm_bonus_obj.get("reasons") or []))
                    and (
                        "explicit_why_now" in set(_llm_bonus_obj.get("reasons") or [])
                        or "horizon_or_kpi" in set(_llm_bonus_obj.get("reasons") or [])
                    )
                    and _exec_has_concentration_exec_signal(_candidate_text)
                ):
                    _select_llm = True
                    _selection_reason = "llm_concentration_sparse_ok"
                elif (
                    _bid == "cot__winners_losers"
                    and _llm_score >= max(4, _det_score - 1)
                    and not _llm_has_placeholder
                    and not _block_hard_reasons
                    and len(_block_soft_reasons or []) <= 1
                    and {
                        "named_mechanism",
                        "explicit_why_now",
                        "wl_dual_lever",
                    }.issubset(set(_llm_bonus_obj.get("reasons") or []))
                    and _exec_has_winners_losers_exec_signal(_candidate_text)
                ):
                    _select_llm = True
                    _selection_reason = "llm_wl_dual_lever_ok"
                elif (
                    _bid == "cot__start_end"
                    and _llm_score >= max(4, _det_score - 1)
                    and not _llm_has_placeholder
                    and not _block_hard_reasons
                    and len(_block_soft_reasons or []) <= 1
                    and {
                        "named_mechanism",
                        "start_end_structural_signal",
                    }.issubset(set(_llm_bonus_obj.get("reasons") or []))
                    and _exec_has_start_end_exec_signal(_candidate_text)
                ):
                    _select_llm = True
                    _selection_reason = "llm_start_end_structural_ok"
                elif (
                    _bid == "cot__start_end"
                    and _llm_score >= max(4, _det_score - 1)
                    and not _llm_has_placeholder
                    and not _block_hard_reasons
                    and len(_block_soft_reasons or []) <= 1
                    and (
                        _exec_has_start_end_exec_signal(_candidate_text)
                        or (
                            "named_mechanism" in set(_llm_bonus_obj.get("reasons") or [])
                            and (
                                "explicit_business_lever" in set(_llm_bonus_obj.get("reasons") or [])
                                or "horizon_or_kpi" in set(_llm_bonus_obj.get("reasons") or [])
                                or _llm_has_why_now
                            )
                        )
                    )
                ):
                    _select_llm = True
                    _selection_reason = "llm_start_end_safe_margin_ok"
                elif (
                    _bid == "cot__mix_share_topN"
                    and _llm_score >= max(5, _det_score - 1)
                    and not _llm_has_placeholder
                    and not _block_hard_reasons
                    and len(_block_soft_reasons or []) <= 1
                    and {
                        "explicit_why_now",
                        "explicit_business_lever",
                        "horizon_or_kpi",
                        "mix_dual_anchor",
                    }.issubset(set(_llm_bonus_obj.get("reasons") or []))
                ):
                    _select_llm = True
                    _selection_reason = "llm_mix_soft_ok"
                elif (
                    _llm_score >= max(4, _det_score - 1)
                    and not _llm_has_placeholder
                    and not _block_hard_reasons
                    and (
                        int(_llm_bonus_obj.get("bonus") or 0) > int(_det_bonus_obj.get("bonus") or 0)
                        or (_llm_has_why_now and not _det_has_why_now)
                        or (_det_critical_penalties and not _llm_critical_penalties)
                    )
                ):
                    _select_llm = True
                    _selection_reason = "llm_safe_margin_preference"
                elif _llm_score == _det_score:
                    if not _det_passed_gate:
                        _select_llm = True
                        _selection_reason = "llm_tie_det_failed_gate"
                    elif _bid in _llm_tie_prefer_blocks and not _det_critical_penalties:
                        _select_llm = True
                        _selection_reason = "llm_tie_preferred_over_deterministic"
                    elif _det_critical_penalties and not _llm_critical_penalties:
                        _select_llm = True
                        _selection_reason = "llm_tie_det_has_critical_penalty"
                    elif int(_llm_bonus_obj.get("bonus") or 0) > int(_det_bonus_obj.get("bonus") or 0):
                        _select_llm = True
                        _selection_reason = "llm_tie_higher_exec_bonus"
                    elif _llm_has_why_now and not _det_has_why_now:
                        _select_llm = True
                        _selection_reason = "llm_tie_has_why_now"
                    elif (not _block_soft_reasons) and _det_soft_reasons:
                        _select_llm = True
                        _selection_reason = "llm_tie_det_soft_fail"

            if _all_exec_results_empty:
                _final_text = _det_text
                _final_src = "deterministic_fallback" if _final_text else "deterministic_fallback_empty_guard"
                _gate_reason = "global_empty_exec_results"
                _final_meta_extra = {
                    "origin_stage": "global_empty_exec_results",
                    "contract_mode": _contract_mode,
                    "used_repair": False,
                    "hard_reasons": list(_block_hard_reasons or []),
                    "soft_reasons": list(_block_soft_reasons or []),
                    "selected_by": "global_empty_exec_results",
                    "selection_reason": "global_empty_exec_results",
                    "score_llm": _llm_score,
                    "score_det": _det_score,
                    "llm_bonus": int(_llm_bonus_obj.get("bonus") or 0),
                    "det_bonus": int(_det_bonus_obj.get("bonus") or 0),
                }
            elif _select_llm:
                _final_text = _candidate_text
                _final_src = _candidate_src or "llm_selected"
                _gate_reason = ""
                _final_meta_extra = {
                    "origin_stage": _candidate_src or "llm_selected",
                    "contract_mode": _contract_mode,
                    "used_repair": str(_candidate_src or "").startswith("llm_repair"),
                    "hard_reasons": list(_block_hard_reasons or []),
                    "soft_reasons": list(_block_soft_reasons or []),
                    "selected_by": _selection_reason or "llm_selected",
                    "selection_reason": _selection_reason or "llm_selected",
                    "score_llm": _llm_score,
                    "score_det": _det_score,
                    "llm_bonus": int(_llm_bonus_obj.get("bonus") or 0),
                    "det_bonus": int(_det_bonus_obj.get("bonus") or 0),
                }
                dbg_cp(
                    "exec_takeaway.selector_decision",
                    block_id=_bid,
                    selected_src=_final_src,
                    llm_score=_llm_score,
                    det_score=_det_score,
                    contract_mode=_contract_mode,
                    contract_profile=_contract_profile,
                    llm_eval=_llm_eval,
                    det_eval=_det_eval,
                    llm_block_hard_reasons=_block_hard_reasons,
                    llm_block_soft_reasons=_block_soft_reasons,
                    det_block_hard_reasons=_det_hard_reasons,
                    det_block_soft_reasons=_det_soft_reasons,
                    llm_bonus=_llm_bonus_obj,
                    det_bonus=_det_bonus_obj,
                    selection_reason=_selection_reason or "llm_selected",
                    preview=_candidate_text[:160],
                )
            else:
                _forced_det_candidate = bool(_candidate_forced_by_block_validator and _candidate_text)
                _final_text = _candidate_text if _forced_det_candidate else _det_text
                _final_src = (
                    _candidate_src
                    if _forced_det_candidate
                    else ("deterministic_fallback" if _final_text else "deterministic_fallback_empty_guard")
                )
                _gate_reasons = list(_candidate_forced_gate_reasons or _block_hard_reasons or [])
                if not _gate_reasons:
                    _gate_reasons.extend(_block_soft_reasons or [])
                _gate_reasons.extend(_llm_critical_penalties)
                if _forced_det_candidate and "block_validator_forced_fallback" not in _gate_reasons:
                    _gate_reasons.insert(0, "block_validator_forced_fallback")
                if _candidate_text and not _selection_reason and not _gate_reasons:
                    _gate_reasons.append("deterministic_selected_over_llm")
                _gate_reason = ";".join(dict.fromkeys(_gate_reasons)) or "deterministic_selected_over_llm"
                _final_meta_extra = {
                    "origin_stage": "block_validator_forced_fallback" if _forced_det_candidate else "deterministic_selector",
                    "contract_mode": _contract_mode,
                    "used_repair": False,
                    "hard_reasons": list(_candidate_forced_gate_reasons or _block_hard_reasons or []),
                    "soft_reasons": list(_block_soft_reasons or []),
                    "selected_by": (
                        "block_validator_forced_fallback"
                        if _forced_det_candidate
                        else (_gate_reason or "deterministic_selected_over_llm")
                    ),
                    "selection_reason": (
                        "block_validator_forced_fallback"
                        if _forced_det_candidate
                        else (_selection_reason or "deterministic_selected_over_llm")
                    ),
                    "score_llm": _llm_score,
                    "score_det": _det_score,
                    "llm_bonus": int(_llm_bonus_obj.get("bonus") or 0),
                    "det_bonus": int(_det_bonus_obj.get("bonus") or 0),
                    "validator_force_reasons": list(_candidate_forced_gate_reasons or []),
                }

                dbg_cp(
                    "exec_takeaway.block_gate_fallback",
                    block_id=_bid,
                    candidate_src=_candidate_src,
                    gate_reasons=_gate_reasons,
                    fallback_used=bool(_final_text),
                    llm_score=_llm_score,
                    det_score=_det_score,
                    contract_mode=_contract_mode,
                    contract_profile=_contract_profile,
                    llm_eval=_llm_eval,
                    det_eval=_det_eval,
                    llm_block_hard_reasons=_block_hard_reasons,
                    llm_block_soft_reasons=_block_soft_reasons,
                    det_block_hard_reasons=_det_hard_reasons,
                    det_block_soft_reasons=_det_soft_reasons,
                    llm_bonus=_llm_bonus_obj,
                    det_bonus=_det_bonus_obj,
                    preview=_candidate_text[:160],
                )

            if not str(_final_text or "").strip():
                _forced_text = _fallback_from_stats(_b)
                if str(_forced_text or "").strip():
                    _final_text = _forced_text
                    _final_src = "deterministic_forced_guard"
                    _gate_reason = _gate_reason or "forced_guard_after_empty_candidate"

            _set_exec_final(
                _b,
                text=_final_text,
                src=_final_src,
                gate_reason=_gate_reason,
                meta_extra=_final_meta_extra,
            )

            if not str((_b.get("_exec_final") or {}).get("text") or "").strip():
                _hard_fallback_text = _fallback_from_stats(_b)
                if _hard_fallback_text:
                    _set_exec_final(
                        _b,
                        text=_hard_fallback_text,
                        src="deterministic_fallback_hard",
                        gate_reason=_gate_reason or "post_set_exec_final_empty",
                        meta_extra={
                            "origin_stage": "deterministic_fallback_hard",
                            "contract_mode": _contract_mode,
                            "used_repair": False,
                            "hard_reasons": list(_block_hard_reasons or []),
                            "soft_reasons": list(_block_soft_reasons or []),
                            "selected_by": "deterministic_fallback_hard",
                            "selection_reason": "deterministic_fallback_hard",
                            "score_llm": _llm_score,
                            "score_det": _det_score,
                            "llm_bonus": int(_llm_bonus_obj.get("bonus") or 0),
                            "det_bonus": int(_det_bonus_obj.get("bonus") or 0),
                        },
                    )
                else:
                    dbg_cp(
                        "exec_takeaway.ERROR_empty_final_render",
                        block_id=_bid,
                        final_src=_final_src,
                    )

            _final_payload_now = (_b.get("_exec_final") or {}) if isinstance(_b, dict) else {}
            _final_src_now = str((_final_payload_now.get("meta") or {}).get("src") or _final_payload_now.get("src") or _final_src or "")
            _used_repair = _final_src_now.startswith("llm_repair")
            _row = _exec_audit_row_from_final_payload(
                _bid,
                _final_payload_now,
                _block_stats,
                audit_path=_audit_path,
            )
            _exec_llm_share_rows.append(_row)

            if _force_cold_audit_run or bool(st.session_state.get("__cot_exec_dbg_on")):
                dbg_cp(
                    "exec_takeaway.audit_block",
                    block_id=_bid,
                    audit_path=_audit_path,
                    final_src=_final_src_now,
                    contract_mode=_row.get("contract_mode"),
                    used_repair=_row.get("used_repair"),
                    hard_reasons=_row.get("hard_reasons"),
                    soft_reasons=_row.get("soft_reasons"),
                    selected_by=_row.get("selected_by"),
                )

        if _exec_llm_share_rows:
            _llm_count = sum(1 for r in _exec_llm_share_rows if r.get("is_llm"))
            _det_count = sum(1 for r in _exec_llm_share_rows if r.get("is_det"))
            _total_count = len(_exec_llm_share_rows)
            _llm_share_pct = (float(_llm_count) / float(_total_count)) if _total_count else 0.0
            dbg_cp(
                "exec_takeaway.llm_share_summary",
                cache_key=_exec_cache_key if "_exec_cache_key" in locals() else None,
                audit_path=_audit_path,
                blocks_count=_total_count,
                llm_count=_llm_count,
                det_count=_det_count,
                llm_share_pct=_llm_share_pct,
                final_src_by_block={str(r.get("block_id")): str(r.get("final_src")) for r in _exec_llm_share_rows},
                contract_mode_by_block={str(r.get("block_id")): str(r.get("contract_mode")) for r in _exec_llm_share_rows},
            )
            _global_target_pct = 0.80
            _below_target_blocks = []
            for _row in _exec_llm_share_rows:
                _row_target = _exec_llm_target_for_block(
                    str(_row.get("block_id") or ""),
                    str(_row.get("contract_mode") or "full"),
                )
                if _row.get("is_det"):
                    _below_target_blocks.append({
                        "block_id": str(_row.get("block_id") or ""),
                        "contract_mode": str(_row.get("contract_mode") or "full"),
                        "target_pct": _row_target,
                        "final_src": str(_row.get("final_src") or ""),
                    })
            dbg_cp(
                "exec_takeaway.llm_share_target_check",
                cache_key=_exec_cache_key if "_exec_cache_key" in locals() else None,
                audit_path=_audit_path,
                llm_share_pct=_llm_share_pct,
                target_pct=_global_target_pct,
                meets_target=bool(_llm_share_pct >= _global_target_pct),
                below_target_blocks=_below_target_blocks,
                final_src_by_block={str(r.get("block_id")): str(r.get("final_src")) for r in _exec_llm_share_rows},
            )

            if _force_cold_audit_run:
                _cold_stats_store = st.session_state.setdefault("__cot_exec_cold_audit_block_stats_v1", {})
                for _row in _exec_llm_share_rows:
                    _block_id = str(_row.get("block_id") or "")
                    _contract_mode_row = str(_row.get("contract_mode") or "full")
                    if not _block_id:
                        continue
                    _agg_key = f"{_block_id}::{_contract_mode_row}"
                    _agg = _cold_stats_store.setdefault(
                        _agg_key,
                        {
                            "block_id": _block_id,
                            "contract_mode": _contract_mode_row,
                            "llm": 0,
                            "det": 0,
                            "total": 0,
                        },
                    )
                    _agg["total"] = int(_agg.get("total") or 0) + 1
                    if _row.get("is_llm"):
                        _agg["llm"] = int(_agg.get("llm") or 0) + 1
                    elif _row.get("is_det"):
                        _agg["det"] = int(_agg.get("det") or 0) + 1

                    _block_total = int(_agg.get("total") or 0)
                    _block_llm_share = (float(_agg.get("llm") or 0) / float(_block_total)) if _block_total else 0.0
                    _block_target = _exec_llm_target_for_block(_block_id, _contract_mode_row)
                    if _block_llm_share >= _block_target:
                        _recommendation = "graduate_from_forced_fallback"
                    elif _block_llm_share >= max(0.60, _block_target - 0.20):
                        _recommendation = "repair_only"
                    else:
                        _recommendation = "keep_hard_guard"

                    dbg_cp(
                        "exec_takeaway.graduation_policy_block",
                        block_id=_block_id,
                        contract_mode=_contract_mode_row,
                        total_runs=_block_total,
                        llm_share_pct=_block_llm_share,
                        target_pct=_block_target,
                        recommendation=_recommendation,
                    )
                _consume_force_cold_exec_run_flag()

        if (not _final_cache_hit) and (not _force_cold_audit_run):
            _final_cache_payload = {}
            for _blk in (_exec_blocks or []):
                _blk_id = _exec_block_id(_blk)
                if not _blk_id:
                    continue
                _target_block = next((b for b in (blocks or []) if str(b.get("id") or "") == _blk_id), None)
                _final_payload = (_target_block or {}).get("_exec_final") if isinstance(_target_block, dict) else None
                if isinstance(_final_payload, dict) and str(_final_payload.get("text") or "").strip():
                    _final_cache_payload[_blk_id] = dict(_final_payload)

            _missing_final_ids = _get_missing_exec_block_ids(
                _exec_blocks,
                {str(k): str((v or {}).get("text") or "") for k, v in _final_cache_payload.items()},
            )
            _final_cache_audit = _inspect_exec_final_cache_payload_map(_final_cache_payload)
            if _all_exec_results_empty:
                _exec_final_cache.pop(_exec_cache_key, None)
                dbg_cp(
                    "exec_takeaway.final_cache_not_stored_global_empty",
                    cache_key=_exec_cache_key,
                    blocks_count=len(_exec_blocks),
                )
            elif bool(_final_cache_audit.get("poisoned")):
                _exec_final_cache.pop(_exec_cache_key, None)
                dbg_cp(
                    "exec_takeaway.final_cache_not_stored_poisoned",
                    cache_key=_exec_cache_key,
                    blocks_count=len(_exec_blocks),
                    reasons=_final_cache_audit.get("reasons"),
                    final_src_by_block=_final_cache_audit.get("final_src_by_block"),
                )
            elif not _missing_final_ids:
                _exec_final_cache[_exec_cache_key] = dict(_final_cache_payload)
                dbg_cp(
                    "exec_takeaway.final_cache_store",
                    cache_key=_exec_cache_key,
                    blocks_count=len(_exec_blocks),
                    cache_engine_version=_EXEC_FINAL_CACHE_ENGINE_VERSION,
                )
            else:
                _exec_final_cache.pop(_exec_cache_key, None)
                dbg_cp(
                    "exec_takeaway.final_cache_not_stored_incomplete",
                    cache_key=_exec_cache_key,
                    missing_blocks=_missing_final_ids,
                    blocks_count=len(_exec_blocks),
                )

        if any(not str(v.get("text") or "").strip() for v in _exec_results.values()):
            dbg_cp(
                "exec_takeaway.ERROR_empty_blocks_after_build_exec_results",
                empty_labels=[
                    k for k, v in _exec_results.items()
                    if not str(v.get("text") or "").strip()
                ],
            )

        # Sanitize LLM outputs (including those that passed through repair) before rendering
        _stats_by_id = {b.get("id"): (b.get("stats") or {}) for b in (blocks or [])}

        def _has_bad_et_anchor(_bid: str, _txt: str, _stats: dict) -> bool:
            _t = str(_txt or "")
            _tl = _t.lower()
            if _bid in {"cot__winners_losers", "cot__start_end"}:
                if ('other' in _tl) or ('najsłabszej kategorii' in _tl) or ('„kategorii”' in _t) or ('"kategorii"' in _t):
                    return True
                if re.search(r"[+-]?0[\.,]0\s*pp", _t):
                    _w = _first_not_none(_stats.get("winner_delta_pp_non_other"), _stats.get("winner_delta_pp"), _stats.get("biggest_gainer_delta_pp"))
                    _l = _first_not_none(_stats.get("loser_delta_pp_non_other"), _stats.get("loser_delta_pp"), _stats.get("biggest_loser_delta_pp"))
                    try:
                        if (_w is not None and abs(float(_w)) > 1e-9) or (_l is not None and abs(float(_l)) > 1e-9):
                            return True
                    except Exception:
                        return True
            if _bid == "cot__seasonality":
                if ('„kategorii”' in _t) or ('"kategorii"' in _t) or ('jako „”' in _t) or ('jako ""' in _t):
                    return True
                if re.search(r"waga sezonowości\s+0([\.,]0+)?", _tl) and re.search(r"amplituda\s+0([\.,]0+)?", _tl):
                    _w = _first_not_none(_stats.get("seasonality_focus_weight"), _stats.get("seasonality_weight_max"))
                    _a = _first_not_none(_stats.get("seasonality_focus_amplitude"), _stats.get("seasonality_amplitude_mean"))
                    try:
                        if (_w is not None and float(_w) > 0) or (_a is not None and abs(float(_a)) > 1e-9):
                            return True
                    except Exception:
                        return True
            if _bid == "cot__mix_share_topN":
                if re.search(r"z\s+0[\.,]0%\s+do\s+0[\.,]0%", _tl):
                    _d = _first_not_none(_stats.get("topN_delta_pp"), _stats.get("delta_pp"))
                    try:
                        if _d is not None and abs(float(_d)) > 1e-9:
                            return True
                    except Exception:
                        return True
            return False

        # Final ET source-of-truth for UI: build -> set -> render.
        _exec_source_map = locals().get("_exec_source_by_id")
        if not isinstance(_exec_source_map, dict):
            _exec_source_map = {}

        for b in blocks:
            st.markdown(f"### {b['title']}")
            desc = b["desc"]
            if b.get("id") == "cot__mix_share_topN":
                desc = (
                    "Udział TopN względem całości (vs TOTAL). "
                    + ("Z włączonym 'Other' wykres sumuje się do 100%." if include_other else "Po wyłączeniu 'Other' oś Y auto-skaluję się do sumy udziałów TopN (większa czytelność).")
                )
            elif b.get("id") == "cot__winners_losers":
                desc = (
                    "Linie trendu udziałów (%) względem całości (vs TOTAL) dla TopN kategorii. "
                    + ("Z włączonym 'Other' pokazujemy również linię 'Other' (reszta kategorii)." if include_other else "Po wyłączeniu 'Other' oś Y auto-skaluję się do zakresu udziałów TopN (większa czytelność).")
                )
            if b.get("id") == "cot__winners_losers":
                desc = (
                    f"Linie trendu udziałów (%) względem całości (vs TOTAL) dla TopN elementów wymiaru {dimension_label}. "
                    + ("Z włączonym 'Other' pokazujemy również linię 'Other' (reszta elementów wymiaru)." if include_other else "Po wyłączeniu 'Other' oś Y auto-skaluje się do zakresu udziałów TopN (większa czytelność).")
                )
            st.caption(desc)

            if b.get("id") == "cot__seasonality":
                pass

            # Streamlit multipage: avoid importing via "app.*" ("app" isn't a package at runtime)
            from core.ui_safe import altair_chart_stretch

            if b.get("id") == "cot__seasonality":
                st.markdown(f"#### 🧩 Heatmapa czystej sezonowości per {dimension_entity} + waga sezonowości")
                st.caption("Odchylenie sezonowe (po usunięciu trendu) — skala symetryczna: ujemne ↔ dodatnie.")

            ch = b.get("chart")
            _raw_ch = ch
            ch = _unwrap_altair_chart(ch)
            if ch is None and b.get("id") == "cot__seasonality":
                _raw_ch = _fallback_seasonality_heatmap(df_time, cat_col, top_n=top_n)
                ch = _unwrap_altair_chart(_raw_ch)
            if ch is None:
                if debug_perf:
                    st.error(
                        "❌ Nieprawidłowy obiekt wykresu — pomijam render, aby uniknąć crasha."
                        f"\n\n• block: `{b.get('id')}`"
                        f"\n• type: `{type(_raw_ch)}`"
                        f"\n• repr: `{_short_repr(_raw_ch)}`"
                    )
                else:
                    st.warning(
                        "⚠️ Wykres chwilowo niedostępny dla aktualnego zakresu dat. "
                        "Spróbuj odświeżyć lub zmienić zakres."
                    )
            else:
                altair_chart_stretch(st, ch)

            if b.get("id") == "cot__concentration":
                st_stats = b.get("stats") or {}
                badge = str(st_stats.get("badge") or "—")
                badge_desc = str(st_stats.get("badge_desc") or "")
                badge_icon = str(st_stats.get("badge_icon") or "🟢")
                badge_display = str(st_stats.get("badge_display") or badge)
                evidence = str(st_stats.get("evidence") or "—")
                conf = str(st_stats.get("confidence") or "—")

                # 3) Szybki werdykt (headline insight)
                # Badge: kolor i kwalifikator skali (umiarkowana / silna) zależnie od strength_score
                # --- Board-ready typography for meta lines (Badge / Evidence / Pewność) ---
                st.markdown(
                    """
                    <style>
                      .cot-cotmeta { font-size: 0.92rem; line-height: 1.28; }
                      .cot-cotmeta .line { margin: 0.10rem 0 0.18rem 0; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="cot-cotmeta">
                      <div class="line">{badge_icon} <b>Badge:</b> {badge_display} — {badge_desc}</div>
                      <div class="line">🔎 <b>Evidence:</b> {evidence}</div>
                      <div class="line">📄 <b>Pewność:</b> {conf}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # 6) Executive Takeaway + 7) Guidance — single final resolver path only
                _render_exec_takeaway_final(b)
                b["_skip_exec_guidance"] = True

                # Guidance: wersja rozszerzona o „Ograniczenia” i „Kiedy uważać”
                g = _guidance_for(b["id"], dimension_label=dimension_label, dimension_entity=dimension_entity) or {}
                with st.expander("Guidance"):
                    # Board-ready: mniejsza typografia + reguły decyzyjne na górze (bez "Sens")
                    rule1 = "Jeśli HHI rośnie szybciej niż Top-5 → rośnie dominacja 1–2 kategorii (ryzyko koncentracji)."
                    rule2 = "Jeśli Top-5 rośnie, a HHI jest stabilne → konsolidacja szeroka (liderzy rosną równomiernie)."
                    rule3 = "Jeśli Top-5 i HHI spadają → postępuje dywersyfikacja portfela (niższa koncentracja)."

                    case_line = ""
                    try:
                        if isinstance(badge, str) and badge.strip():
                            case_line = f"📌 <b>W tym przypadku:</b> {badge_desc}"
                    except Exception:
                        case_line = ""

                    st.markdown(
                        f"""
                        <style>
                          .cot-small p, .cot-small li {{ font-size: 0.92rem; line-height: 1.28; }}
                          .cot-small p {{ margin: 0.10rem 0 0.22rem 0; }}
                          .cot-small li {{ margin: 0.08rem 0; }}
                          .cot-small ul {{ margin: 0.10rem 0 0.22rem 1.2rem; }}
                        </style>
                        <div class="cot-small">
                          <p>🧭 <b>Reguły decyzyjne (Top-5 vs HHI):</b></p>
                          <ul>
                            <li>{rule1}</li>
                            <li>{rule2}</li>
                            <li>{rule3}</li>
                          </ul>
                          {f"<p>{case_line}</p>" if case_line else ""}
                          <p>💡 <b>Jak użyć tego zarządczo:</b> dopasuj działania do typu zmiany (obrona/skalowanie liderów vs inwestycje w challengers vs porządkowanie ogona).</p>
                          <p>⚠️ <b>Ograniczenia:</b> miary oparte o Top-3/Top-5 i trend liniowy mogą nie wychwycić zmian w ogonie oraz bywają wrażliwe na krótkie okna lub jednorazowe skoki.</p>
                          <p>🧩 <b>Kiedy uważać:</b> przy zmianach strukturalnych (np. nowa kategoria, rebranding, zmiana przypisania produktów) interpretuj trend ostrożnie i porównaj równolegle HHI.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander("Jak liczymy badge i HHI?"):

                    st.markdown(
                        """
                        <style>
                          .cot-small p, .cot-small li { font-size: 0.92rem; line-height: 1.28; }
                          .cot-small p { margin: 0.10rem 0 0.22rem 0; }
                          .cot-small li { margin: 0.08rem 0; }
                          .cot-small ul { margin: 0.10rem 0 0.22rem 1.2rem; }
                        </style>
                        <div class="cot-small">
                          <p><b>1) Udziały i Top-3 / Top-5</b><br/>
                          W każdym miesiącu liczymy udział każdej kategorii w sumie (TOTAL). Następnie sortujemy kategorie po udziale i sumujemy udziały dla Top-3 oraz Top-5.</p>

                          <p><b>2) Przełącznik „Dodaj 'Other'”</b><br/>
                          To jest kontrola <b>granularności</b>: gdy jest włączona, ogon kategorii (poza Top-N) jest łączony do jednej kategorii „Other”. To może istotnie zmienić ranking (np. gdy „Other” staje się największą kategorią), więc wpływa na Top-3/Top-5.</p>

                          <p><b>3) Δ i trend (pp/rok) + R²</b><br/>
                          Δ = (ostatni miesiąc – pierwszy miesiąc) w punktach procentowych. Trend estymujemy prostą regresją liniową po czasie (oś czasu w latach), a R² opisuje dopasowanie trendu.</p>

                          <p><b>3a) HHI (Herfindahl–Hirschman Index)</b><br/>
                          HHI to miara koncentracji liczona jako suma kwadratów udziałów kategorii: <b>HHI = Σ(udział²)</b>.</p>
                          <ul>
                            <li>HHI bliżej 0 → struktura rozproszona (wiele podobnych udziałów).</li>
                            <li>HHI bliżej 1 → dominacja nielicznych kategorii (wysoka koncentracja).</li>
                          </ul>
                          <p>W tym bloku liczymy HHI miesięcznie na tej samej granularności co wykres (zależnie od „Dodaj 'Other'”).</p>

                          <p><b>4) Badge i pewność</b></p>
                          <ul>
                            <li><i>Konsolidacja</i> — gdy koncentracja rośnie (Δ / trend dodatnie).</li>
                            <li><i>Dywersyfikacja</i> — gdy koncentracja spada (Δ / trend ujemne).</li>
                            <li><i>Stabilna struktura</i> — gdy zmiany są małe.</li>
                          </ul>
                          <p>Pewność to heurystyka oparta o długość szeregu (liczbę miesięcy) i R².</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
# dalsze globalne renderowanie ET/Guidance pomijamy (żeby nie dublować i utrzymać kolejność narracji)
                b["_skip_exec_guidance"] = True

            # --- DEBUG: audit prawdy (TopN + Top-3 udziały: 2009-12 .. 2010-09) ---
            if debug_perf and b.get("id") == "cot__mix_share_topN":
                st.markdown("#### 🧪 Audit prawdy — TopN + udziały Top-3 (Dec 2009 – Sep 2010)")
                stats_dbg = b.get("stats") or {}
                top_dbg = list(stats_dbg.get("top_categories") or [])
                st.write("TopN (wg sumy w całym okresie; wykres liczy udziały *w obrębie TopN*):", top_dbg)

                # Window: prefer Dec 2009..Sep 2010, but clamp to available data for universality
                min_m = pd.to_datetime(df_time["__month"]).min() if (df_time is not None and not df_time.empty and "__month" in df_time.columns) else None
                max_m = pd.to_datetime(df_time["__month"]).max() if (df_time is not None and not df_time.empty and "__month" in df_time.columns) else None
                target_start = pd.Timestamp("2009-12-01")
                target_end = pd.Timestamp("2010-09-01")
                start_m = max([d for d in [min_m, target_start] if d is not None])
                end_m = min([d for d in [max_m, target_end] if d is not None])
                st.caption(f"Okno auditowe użyte w debug: {start_m.date()} → {end_m.date()} (clamp do dostępnych miesięcy).")

                audit_df = _audit_share_table_compare_top3(
                    df_time=df_time,
                    cat_col=cat_col,
                    top=[str(x) for x in top_dbg],
                    start=pd.Timestamp("2009-12-01"),
                    end=pd.Timestamp("2010-09-01"),
                )
                if audit_df.empty:
                    st.info("Brak danych w oknie audytowym dla bieżących filtrów.")
                else:
                    st.caption("Porównanie udziałów Top-3: **vs Total (z 'Other')** oraz **vs TopN** (historyczna definicja).")
                    st.dataframe(audit_df, width='stretch')
            # Special: seasonality scorecard under heatmap
            if b["id"] == "cot__seasonality":
                # --- Scorecard (B1)
                sdf = b.get("score_df")
                if isinstance(sdf, pd.DataFrame) and not sdf.empty:
                    st.markdown(f"#### 🧾 Scorecard sezonowości per {dimension_entity}")

                    sc = sdf.copy()
                    sc["Category"] = sc["Category"].astype(str)

                    # kolejność = TOP10 wg wartości sprzedaży w CAŁYM okresie (ustalona w _block3_seasonality)
                    _ord = (b.get("stats") or {}).get("sort_order") or (b.get("stats") or {}).get("selected")
                    if _ord:
                        sc["_ord"] = pd.Categorical(sc["Category"], categories=list(_ord), ordered=True)
                        sc = sc.sort_values("_ord").drop(columns=["_ord"])

                    # przygotuj kolumny widoku
                    rename_map = {
                        "seasonality_strength": "Strength (0–1)",
                        "seasonality_amplitude": "Amplitude",
                        "seasonality_share": "Seasonality Share (0–1)",
                        "stability_slope": "Stability slope",
                        "peak_drift": "Peak drift (mies.)",
                        "noise_cover_ratio": "Noise cover ratio",
                    }
                    sc_show = sc.rename(columns=rename_map)

                    if "Stability slope" in sc_show.columns:
                        sc_show["Stability slope (abs)"] = pd.to_numeric(sc_show["Stability slope"], errors="coerce").abs()

                                        # --- Werdykt 2.0 (gating + weighted sorting; stałe wagi; 3 klasy)
                    # Seasonality Share = var(seasonal) / var(original) -> używane w scoringu
                    # Amplitude NIE jest używane w scoringu (pozostaje kolumną informacyjną)
                    # Klasyfikacja:
                    #   1) Niestabilność jeśli instability (P75 w TOP10)
                    #   2) Eventowa jeśli noise lub peak_drift (P75 w TOP10)
                    #   3) W pozostałych przypadkach: Kalendarzowa
                    # Score służy TYLKO do sortowania w obrębie klasy (debug-only kolumny score).

                    def _safe_ratio(num, den, default=0.0):
                        try:
                            num = float(num)
                            den = float(den)
                            if den <= 0 or not np.isfinite(num) or not np.isfinite(den):
                                return float(default)
                            return float(num / den)
                        except Exception:
                            return float(default)

                    def _robust_norm_series(v: pd.Series) -> pd.Series:
                        """Robust normalization to [0,1] using 5–95 pct range (per series)."""
                        vv = pd.to_numeric(v, errors="coerce")
                        lo = float(vv.quantile(0.05)) if vv.notna().any() else 0.0
                        hi = float(vv.quantile(0.95)) if vv.notna().any() else 1.0
                        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                            return pd.Series([0.0] * len(vv), index=vv.index, dtype=float)
                        out = (vv - lo) / (hi - lo)
                        return out.clip(0.0, 1.0).fillna(0.0).astype(float)

                    def _q(v: pd.Series, q: float, default: float = 0.0) -> float:
                        try:
                            vv = pd.to_numeric(v, errors="coerce").dropna()
                            if vv.empty:
                                return float(default)
                            return float(vv.quantile(q))
                        except Exception:
                            return float(default)

                    # predeclare locals for static analyzers / editor robustness
                    share01 = pd.Series(dtype=float)
                    strength_n = pd.Series(dtype=float)
                    noise_n = pd.Series(dtype=float)
                    peak_n = pd.Series(dtype=float)
                    instab_n = pd.Series(dtype=float)
                    t_noise = 0.0
                    t_peak = 0.0
                    t_instab = 0.0
                    t_instab_hi = 0.0
                    is_event = pd.Series(dtype=bool)
                    is_instab_hard = pd.Series(dtype=bool)
                    is_instab = pd.Series(dtype=bool)

                    # base metrics
                    strength = pd.to_numeric(sc.get("seasonality_strength"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
                    slope_abs = pd.to_numeric(sc.get("stability_slope"), errors="coerce").abs().fillna(0.0)
                    peak_drift = pd.to_numeric(sc.get("peak_drift_months", sc.get("peak_drift")), errors="coerce").fillna(0.0).abs()
                    noise = pd.to_numeric(sc.get("noise_cover_ratio"), errors="coerce").fillna(0.0)

                    # seasonality share (prefer precomputed ratio if present)
                    if "seasonality_share" in sc.columns:
                        share_raw = pd.to_numeric(sc.get("seasonality_share"), errors="coerce").fillna(0.0)
                    elif "ratio_var_seasonal/original" in sc.columns:
                        share_raw = pd.to_numeric(sc.get("ratio_var_seasonal/original"), errors="coerce").fillna(0.0)
                    else:
                        v_orig = pd.to_numeric(sc.get("var_original"), errors="coerce").fillna(np.nan)
                        v_seas = pd.to_numeric(sc.get("var_seasonal"), errors="coerce").fillna(np.nan)
                        share_raw = pd.Series([_safe_ratio(n, d, default=0.0) for n, d in zip(v_seas, v_orig)], index=sc.index, dtype=float)

                    # share używane do scoringu: absolutne 0..1 (bez robust_norm, żeby zachować interpretowalność)
                    share01 = pd.to_numeric(share_raw, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)

                    # normalize only "risk-like" signals (dla gatingu i sortowania)
                    strength_n = strength.astype(float)  # already [0,1]
                    noise_n = _robust_norm_series(noise.clip(lower=0.0))
                    peak_n = _robust_norm_series(peak_drift.clip(lower=0.0))
                    instab_n = _robust_norm_series(slope_abs.clip(lower=0.0))

                    # gating thresholds (stałe produkcyjnie w TOP10)
                    # UWAGA metodologiczna (fix):
                    # Sam wysoki slope_abs może oznaczać zmianę poziomu/siły sezonowości (trend), a nie "niestabilność" w sensie
                    # eventowości / dryfu piku / szumu. Dlatego "Niestabilność" wymaga potwierdzenia (peak lub noise) albo
                    # ekstremalnie wysokiego instab.
                    t_noise = _q(noise_n, 0.75, default=0.0)
                    t_peak = _q(peak_n, 0.75, default=0.0)
                    t_instab = _q(instab_n, 0.75, default=0.0)
                    t_instab_hi = _q(instab_n, 0.90, default=float(t_instab))

                    # bramki klas (decision layer)
                    is_event = (noise_n >= t_noise) | (peak_n >= t_peak)
                    is_instab_hard = instab_n >= t_instab_hi
                    is_instab_soft = (instab_n >= t_instab) & ((peak_n >= t_peak) | (noise_n >= t_noise))
                    is_instab = is_instab_hard | is_instab_soft

                    verdict = pd.Series("Sezonowość kalendarzowa", index=sc.index, dtype=str)
                    verdict = verdict.mask(is_event, "Sezonowość eventowa")
                    verdict = verdict.mask(is_instab, "Niestabilność sezonowości")

                    # fixed production weights (agreed) — scoring only for within-class sorting
                    w_cal = dict(strength=0.40, share=0.40, inv_peak=0.10, inv_noise=0.07, inv_instab=0.03)
                    w_evt = dict(noise=0.45, peak=0.25, share=0.20, inv_strength=0.10)
                    w_ins = dict(instab=0.60, peak=0.20, noise=0.20)

                    calendar_score = (
                        w_cal["strength"] * strength_n
                        + w_cal["share"] * share01
                        + w_cal["inv_peak"] * (1.0 - peak_n)
                        + w_cal["inv_noise"] * (1.0 - noise_n)
                        + w_cal["inv_instab"] * (1.0 - instab_n)
                    )

                    event_score = (
                        w_evt["noise"] * noise_n
                        + w_evt["peak"] * peak_n
                        + w_evt["share"] * share01
                        + w_evt["inv_strength"] * (1.0 - strength_n)
                    )

                    instability_score = (
                        w_ins["instab"] * instab_n
                        + w_ins["peak"] * peak_n
                        + w_ins["noise"] * noise_n
                    )

                    scores_df = pd.DataFrame(
                        {
                            "_calendar_score": calendar_score.astype(float),
                            "_event_score": event_score.astype(float),
                            "_instability_score": instability_score.astype(float),
                        },
                        index=sc.index,
                    )

                    # persist core columns
                    sc["seasonality_share"] = pd.to_numeric(share_raw, errors="coerce").fillna(0.0).astype(float)
                    sc["verdict"] = verdict

                    # debug-score columns (always computed & stored in data model; UI may hide when debug_perf=False)
                    sc["_calendar_score"] = scores_df["_calendar_score"]
                    sc["_event_score"] = scores_df["_event_score"]
                    sc["_instability_score"] = scores_df["_instability_score"]

                    # additional debug-only explainability columns
                    if bool(debug_perf):
                        sc["_seasonality_share_used"] = share01
                        sc["_thr_noise_p75"] = float(t_noise)
                        sc["_thr_peak_p75"] = float(t_peak)
                        sc["_thr_instab_p75"] = float(t_instab)
                        sc["_thr_instab_p90"] = float(t_instab_hi)
                        sc["_is_event_gate"] = is_event.astype(bool)
                        sc["_is_instab_gate"] = is_instab.astype(bool)
                        sc["_is_instab_hard"] = is_instab_hard.astype(bool)

                    # sorting: kalendarzowa -> eventowa -> niestabilność; within-class by relevant score desc
                    verdict_rank = {
                        "Sezonowość kalendarzowa": 0,
                        "Sezonowość eventowa": 1,
                        "Niestabilność sezonowości": 2,
                    }
                    sc["_verdict_rank"] = sc["verdict"].map(verdict_rank).fillna(1).astype(int)

                    within_score = (
                        scores_df["_calendar_score"].where(sc["_verdict_rank"] == 0, np.nan)
                        .fillna(scores_df["_event_score"].where(sc["_verdict_rank"] == 1, np.nan))
                        .fillna(scores_df["_instability_score"])
                    )
                    sc["_within_score"] = within_score.astype(float)

                    sc = sc.sort_values(["_verdict_rank", "_within_score"], ascending=[True, False]).copy()

# --- Render scorecard jako HTML (światowy UX: pill verdict + conditional bars)
                    # (kolor progressów: grafit domyślnie; czerwony tylko dla wysokiego Noise / Stability(abs) / Peak drift)

                    # ustandaryzuj typy
                    sc_num = sc.copy()
                    for _c in ["seasonality_strength","seasonality_amplitude","stability_slope","peak_drift","noise_cover_ratio"]:
                        if _c in sc_num.columns:
                            sc_num[_c] = pd.to_numeric(sc_num[_c], errors="coerce")

                    # dodaj Stability(abs) do sc_num i sc_show
                    sc_num["stability_slope_abs"] = sc_num.get("stability_slope").abs()
                    if "Stability slope (abs)" not in sc_show.columns:
                        sc_show["Stability slope (abs)"] = sc_num["stability_slope_abs"]

                    
                    def _q(series: Any, q: float, default: float = 0.0) -> float:
                        """Bezpieczny kwantyl dla serii (również gdy series=None)."""
                        try:
                            if series is None:
                                return float(default)
                            v = pd.to_numeric(series, errors="coerce")
                            if hasattr(v, "dropna"):
                                v = v.dropna()
                            if v is None or len(v) == 0:
                                return float(default)
                            return float(v.quantile(q))
                        except Exception:
                            return float(default)

                    # progi alertów (kwantyle)
                    q_noise_hi = _q(sc_num.get("noise_cover_ratio"), 0.75, default=0.0)
                    q_peak_hi = _q(sc_num.get("peak_drift"), 0.75, default=0.0)
                    q_slope_abs_hi = _q(sc_num.get("stability_slope_abs"), 0.75, default=0.0)

                    def _safe_minmax(series: pd.Series, default_min: float, default_max: float) -> tuple[float,float]:
                        try:
                            v = pd.to_numeric(series, errors="coerce").dropna()
                            if v.empty:
                                return default_min, default_max
                            mn = float(v.min()); mx = float(v.max())
                            if mn == mx:
                                return mn, mx + 1e-9
                            return mn, mx
                        except Exception:
                            return default_min, default_max

                    r_amp = _safe_minmax(sc_num.get("seasonality_amplitude"), 0.0, 1.0)
                    r_slope = _safe_minmax(sc_num.get("stability_slope"), 0.0, 1.0)
                    r_slope_abs = _safe_minmax(sc_num.get("stability_slope_abs"), 0.0, 1.0)
                    r_peak = _safe_minmax(sc_num.get("peak_drift"), 0.0, 12.0)
                    r_noise = _safe_minmax(sc_num.get("noise_cover_ratio"), 0.0, 1.0)

                    def _pct(v: float, mn: float, mx: float) -> float:
                        try:
                            if v is None or (pd.isna(v)):
                                return 0.0
                            v = float(v)
                            if mx <= mn:
                                return 0.0
                            x = (v - mn) / (mx - mn)
                            return float(max(0.0, min(1.0, x))) * 100.0
                        except Exception:
                            return 0.0

                    def _pill(label: str) -> str:
                        # pill style + tekst (bez osobnej kolumny z kropkami)
                        bg = "#eef2f7"; fg = "#334155"; dot = "🟣"
                        if label == "Sezonowość kalendarzowa":
                            bg = "#e6f4ea"; fg = "#137333"; dot = "🟢"
                        elif label == "Sezonowość eventowa":
                            bg = "#fef7e0"; fg = "#b06000"; dot = "🟠"
                        elif label == "Niestabilność sezonowości":
                            bg = "#fce8e6"; fg = "#c5221f"; dot = "🔴"
                        elif label == "Umiarkowana sezonowość":
                            bg = "#eef2f7"; fg = "#334155"; dot = "🟤"
                        else:
                            bg = "#f3f4f6"; fg = "#374151"; dot = "⚪"
                        return f'<span class="pill" style="background:{bg};color:{fg};"><span class="pill-dot">{dot}</span>{label}</span>'

                    def _bar(value: float, mn: float, mx: float, alert: bool, fmt: str) -> str:
                        p = _pct(value, mn, mx)
                        fill = "#6b7280"  # grafit
                        if alert:
                            fill = "#ef4444"  # czerwony (alert)
                        try:
                            label = "" if value is None or pd.isna(value) else (fmt % float(value))
                        except Exception:
                            label = ""
                        return (
                            '<div class="barwrap">'
                            f'<div class="bartrack"><div class="barfill" style="width:{p:.1f}%;background:{fill};"></div></div>'
                            f'<div class="barval">{label}</div>'
                            '</div>'
                        )

                    # CSS (minimal, scoped)
                    st.markdown(
                        """
                        <style>
                        .season-scorecard table {width: 100%; border-collapse: collapse; font-size: 12px;}
                        .season-scorecard th, .season-scorecard td {padding: 8px 10px; border-top: 1px solid #eef2f7; vertical-align: middle;}
                        .season-scorecard thead th {font-weight: 600; color: #475569; background: #fafafa; border-top: 0;}
                        .pill {display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; font-weight:600; white-space:nowrap;}
                        .pill-dot {font-size:12px; line-height:1;}
                        .barwrap{display:flex; align-items:center; gap:10px; min-width:160px;}
                        .bartrack{flex:1; height:6px; background:#e5e7eb; border-radius:999px; overflow:hidden;}
                        .barfill{height:100%; border-radius:999px;}
                        .barval{width:72px; text-align:right; font-variant-numeric: tabular-nums; color:#111827;}
                        .cat{font-weight:600; color:#111827;}
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )


                    # Buduj HTML tabeli
                    # Jeśli stability_slope nie wnosi informacji (brak wartości ujemnych → signed==abs),
                    # ukrywamy kolumnę "Stability slope" i zostawiamy tylko "Stability slope (abs)".
                    _slope_series = pd.to_numeric(sc_num.get("stability_slope"), errors="coerce")
                    include_signed_slope = bool((_slope_series.dropna() < 0).any())

                    rows_html = []
                    for _, r in sc.iterrows():
                        cat = str(r.get("Category", ""))
                        vlabel = str(r.get("Werdykt", r.get("Werdykt sezonowości", r.get("verdict", ""))))

                        strength = pd.to_numeric(r.get("seasonality_strength"), errors="coerce")
                        share = pd.to_numeric(r.get("seasonality_share"), errors="coerce")
                        share_ui = (min(float(share), 1.0) if pd.notna(share) else np.nan)
                        amp = pd.to_numeric(r.get("seasonality_amplitude"), errors="coerce")

                        slope = pd.to_numeric(r.get("stability_slope"), errors="coerce")
                        slope_abs = abs(float(slope)) if pd.notna(slope) else np.nan

                        peak = pd.to_numeric(r.get("peak_drift"), errors="coerce")
                        noise = pd.to_numeric(r.get("noise_cover_ratio"), errors="coerce")

                        # wiersz (kolumny dynamiczne)
                        tds = [
                            f'<td class="cat">{cat}</td>',
                            f"<td>{_pill(vlabel)}</td>",
                            f"<td>{_bar(strength, 0.0, 1.0, False, '%.4f')}</td>",
                            f"<td>{_bar(share_ui, 0.0, 1.0, False, '%.2f')}</td>",
                            f"<td>{_bar(amp, r_amp[0], r_amp[1], False, '%.4f')}</td>",
                        ]

                        if include_signed_slope:
                            tds.append(f"<td>{_bar(slope, r_slope[0], r_slope[1], False, '%.4f')}</td>")

                        tds.extend(
                            [
                                f"<td>{_bar(slope_abs, r_slope_abs[0], r_slope_abs[1], (pd.notna(slope_abs) and slope_abs >= q_slope_abs_hi and q_slope_abs_hi > 0), '%.4f')}</td>",
                                f"<td>{_bar(peak, r_peak[0], r_peak[1], (pd.notna(peak) and peak >= q_peak_hi and q_peak_hi > 0), '%.2f')}</td>",
                                f"<td>{_bar(noise, r_noise[0], r_noise[1], (pd.notna(noise) and noise >= q_noise_hi and q_noise_hi > 0), '%.4f')}</td>",
                            ]
                        )

                        rows_html.append("<tr>" + "".join(tds) + "</tr>")

                    # nagłówki dynamiczne
                    ths = [
                        "<th>Category</th>",
                        "<th>Werdykt sezonowości</th>",
                        "<th>Strength (0–1)</th>",
                        "<th>Seasonality Share (0–1)</th>",
                        "<th>Amplitude</th>",
                    ]
                    if include_signed_slope:
                        ths.append("<th>Stability slope</th>")
                    ths.extend(
                        [
                            "<th>Stability slope (abs)</th>",
                            "<th>Peak drift (mies.)</th>",
                            "<th>Noise cover ratio</th>",
                        ]
                    )

                    table_html = (
                        '<div class="season-scorecard"><table>'
                        "<thead><tr>"
                        + "".join(ths)
                        + "</tr></thead>"
                        "<tbody>"
                        + "".join(rows_html)
                        + "</tbody></table></div>"
                    )
                    st.markdown(table_html, unsafe_allow_html=True)

                    # --- wzbogacenie statystyk dla Executive Takeaway (A1+B1)
                    try:
                        tmp = sc.copy()
                        tmp["verdict"] = verdict
                        tmp["strength"] = pd.to_numeric(tmp.get("seasonality_strength"), errors="coerce")
                        tmp["amp"] = pd.to_numeric(tmp.get("seasonality_amplitude"), errors="coerce")
                        tmp["score"] = (tmp["strength"].fillna(0.0) * tmp["amp"].fillna(0.0))

                        cal = tmp[tmp["verdict"] == "Sezonowość kalendarzowa"].sort_values("score", ascending=False)["Category"].astype(str).head(3).tolist()
                        evt = tmp[tmp["verdict"] == "Sezonowość eventowa"].sort_values(["noise_cover_ratio","peak_drift"], ascending=False)["Category"].astype(str).head(1).tolist()
                        unst = tmp[tmp["verdict"] == "Niestabilność sezonowości"].assign(_abs=tmp["stability_slope"].abs()).sort_values("_abs", ascending=False)["Category"].astype(str).head(1).tolist()

                        mean_strength = float(pd.to_numeric(tmp.get("seasonality_strength"), errors="coerce").mean())

                        b.setdefault("stats", {})
                        b["stats"].setdefault("seasonality_exec", {})
                        b["stats"]["seasonality_exec"].update({
                            "top3_calendar": cal,
                            "top1_event": evt[0] if evt else None,
                            "top1_unstable": unst[0] if unst else None,
                            "mean_strength": mean_strength,
                        })
                    except Exception:
                        pass

# --- Executive Takeaway + Guidance bezpośrednio po B1 (single final resolver path)
                    _render_exec_takeaway_final(b)
                    g = _guidance_for(b["id"], dimension_label=dimension_label, dimension_entity=dimension_entity)
                    render_guidance(g.get("sens"), g.get("interpretacja"), g.get("best_practice"))

                    # globalny renderer ET/Guidance pomijamy tylko dlatego,
                    # że ten sam finalny renderer został już wywołany tutaj.
                    b["_skip_exec_guidance"] = True

                    # --- Expandery (w ustalonej kolejności)
                    with st.expander("🔍 Jak czytać i interpretować sezonowość?"):
                        st.markdown("""
                        <style>
                          .dc-compact p { margin: 0.15rem 0; font-size: 0.9rem; line-height: 1.25; }
                          .dc-compact li { margin: 0.15rem 0; font-size: 0.9rem; line-height: 1.25; }
                          .dc-compact ul { margin: 0.10rem 0 0.70rem 1.2rem; padding: 0; }
                          .dc-compact .hdr { margin-top: 1.75rem; }
                        </style>
                        """, unsafe_allow_html=True)
                        st.markdown("""<div class="dc-compact">
<p><strong>✅ Jak powstaje werdykt sezonowości</strong></p>
<p>Werdykt jest ustalany na podstawie progów (P75/P90) liczonych w obrębie TOP10.<br/>Scoring nie wybiera klasy — służy jedynie do uporządkowania elementów wymiaru w ramach tej samej klasy.</p>

<p class="hdr"><strong>🟢 Sezonowość kalendarzowa</strong></p>
<ul>
  <li>wysoki <strong>Seasonality Share (0–1)</strong> (sezon wyjaśnia istotną część zmienności),</li>
  <li>niski <strong>Peak drift (mies.)</strong> (miesiąc piku jest stały),</li>
  <li>niski <strong>Noise cover ratio</strong> (mało zakłóceń).</li>
</ul>

<p class="hdr"><strong>🟠 Sezonowość eventowa</strong></p>
<ul>
  <li>sezon jest obecny (wysoki share), ale pojawiają się zakłócenia: <strong>Noise</strong> i/lub <strong>Peak drift</strong> są relatywnie wysokie (bramka P75),</li>
  <li>typowe źródła: promocje, kampanie, zdarzenia jednorazowe, przesunięcia popytu.</li>
</ul>

<p class="hdr"><strong>🔴 Niestabilność sezonowości</strong></p>
<ul>
  <li><strong>Stability slope (abs)</strong> jest bardzo wysoki (trendowo zmienia się siła sezonu). Oznacza to, że siła sezonu istotnie zmienia się między latami.</li>
  <li>oraz występuje potwierdzenie niestabilności w <strong>Noise</strong> lub <strong>Peak drift</strong> (P75) <strong>lub</strong> wyjątkowo wysoki slope (P90).</li>
</ul>

<p class="hdr"><strong>✅ Jak czytać heatmapę + scorecard razem</strong></p>
<ul>
  <li>Heatmapa pokazuje <strong>kiedy</strong> (miesiące) pojawiają się piki/dołki i czy są powtarzalne.</li>
  <li>Ramki oznaczają <strong>najsilniejsze odchylenia sezonowe</strong> (P90 per element wymiaru).</li>
  <li>Scorecard pokazuje, <strong>jak silny, jak przewidywalny i jak stabilny</strong> jest wzorzec sezonowy między elementami wymiaru.</li>
</ul>
</div>""", unsafe_allow_html=True)


                    with st.expander("📚 Co oznaczają metryki w scorecard?"):
                        st.markdown("""<div class="dc-compact">
<ul>
  <li><strong>Strength (0–1)</strong> – miara „czystości” sezonowości: im bliżej 1, tym większa część zmienności jest wyjaśniana przez komponent sezonowy (po odjęciu trendu).</li>
  <li><strong>Seasonality Share (0–1)</strong> – udział zmienności wyjaśnianej przez sezonowość: <strong>var(seasonal) / var(original)</strong>. Im bliżej 1, tym sezon dominuje wahań, a nie trend/szum.</li>
  <li><strong>Amplitude</strong> – amplituda sezonowości: różnica między maksymalnym i minimalnym odchyleniem komponentu sezonowego (większa = mocniejsze piki/dołki).</li>
  <li><strong>Stability slope (abs)</strong> – skala zmian siły sezonowości w czasie (bezwzględnie). Im wyżej, tym mocniej zmienia się intensywność sezonu między okresami (ryzyko „dryfu” sezonu).</li>
  <li><strong>Peak drift (mies.)</strong> – dryf miesiąca szczytu: jak bardzo zmienia się miesiąc maksimum sezonu między latami (niżej = bardziej kalendarzowo).</li>
  <li><strong>Noise cover ratio</strong> – ile „szumu” względem sezonowości: wyżej = bardziej eventowo/losowo, niżej = bardziej regularnie.</li>
</ul>
</div>""", unsafe_allow_html=True)


                    with st.expander("🔎 Podstawa werdyktu (share + bramki + score)"):

                        st.caption("Progi bramek liczone w obrębie TOP10 (P75 oraz P90 dla instab).")
                        try:
                            st.json({
                                "noise_p75": float(t_noise),
                                "peak_p75": float(t_peak),
                                "instab_p75": float(t_instab),
                                "instab_p90": float(t_instab_hi),
                            })
                        except Exception:
                            # Fallback (np. gdy brak zmiennych w scope) – pokaż tylko to, co jest w tabeli debug.
                            pass

                        # Tabela audytowa: metryki + flagi bramek + score (score służy wyłącznie do sortowania w obrębie klasy)
                        audit_cols = [
                            "Category",
                            "verdict",
                            "seasonality_strength",
                            "seasonality_share",
                            "stability_slope_abs",
                            "peak_drift",
                            "noise_cover_ratio",
                            "_is_event_gate",
                            "_is_instab_gate",
                            "_is_instab_hard",
                            "_calendar_score",
                            "_event_score",
                            "_instability_score",
                        ]
                        audit_cols = [c for c in audit_cols if c in sc.columns]
                        if audit_cols:
                            st.dataframe(sc[audit_cols], width='stretch', hide_index=True)
                        st.caption("Werdykt wynika z bramek (flagi). Score służy tylko do sortowania w obrębie klasy.")

                    with st.expander("🧪 Demo pełnej dekompozycji"):
                        # automatyczny wybór + dropdown override (TOP10) — wybieramy najlepszy przykład do interpretacji
                        sc_demo = sc.copy()
                        sc_demo['Category'] = sc_demo['Category'].astype(str)

                        # Impact (demo auto-select) = Amplitude × Strength / (1 + Noise cover ratio)
                        # Uwaga: to tylko logika wyboru przykładu w demo — nie zmienia metryk ani STL.
                        demo_strength = pd.to_numeric(sc_demo.get('seasonality_strength', sc_demo.get('Strength')), errors='coerce').fillna(0.0).clip(0, 1)
                        demo_amp = pd.to_numeric(sc_demo.get('seasonality_amplitude', sc_demo.get('Amplitude')), errors='coerce').fillna(0.0)
                        demo_noise = pd.to_numeric(sc_demo.get('noise_cover_ratio', sc_demo.get('Noise cover ratio')), errors='coerce').fillna(0.0).clip(lower=0.0)
                        demo_impact = (demo_amp * demo_strength) / (1.0 + demo_noise)

                        # Auto-wybór najlepszej kategorii (max impact) + możliwość ręcznego override
                        default_cat = sc_demo.loc[demo_impact.idxmax(), 'Category'] if len(sc_demo) else (_ord[0] if _ord else '')
                        options = list(_ord) if _ord else sc_demo['Category'].tolist()
                        options = [str(x) for x in options if str(x) != '']
                        pick = ''  # ustawiane przez selectbox; pusty string => demo pokaże komunikat
                        if not options:
                            st.info(f'Brak pozycji TOP10 dla wymiaru {dimension_label} do pokazania demo.')
                        else:
                            auto_row = sc_demo[sc_demo['Category'] == str(default_cat)].head(1)
                            if not auto_row.empty:
                                _s = float(pd.to_numeric(auto_row.get('seasonality_strength'), errors='coerce').iloc[0]) if 'seasonality_strength' in auto_row.columns else float(demo_strength.loc[auto_row.index[0]])
                                _a = float(pd.to_numeric(auto_row.get('seasonality_amplitude'), errors='coerce').iloc[0]) if 'seasonality_amplitude' in auto_row.columns else float(demo_amp.loc[auto_row.index[0]])
                                _n = float(pd.to_numeric(auto_row.get('noise_cover_ratio'), errors='coerce').iloc[0]) if 'noise_cover_ratio' in auto_row.columns else float(demo_noise.loc[auto_row.index[0]])
                                _imp = float(demo_impact.loc[auto_row.index[0]])
                                st.caption(f"Auto-wybrano {dimension_entity} **{default_cat}** (najwyższy **Impact** = {_imp:,.0f}). Strength={_s:.3f}, Amplitude={_a:,.0f}, Noise={_n:.3f}.")
                            pick = st.selectbox(f'{dimension_label} (TOP10):', options=options, index=(options.index(str(default_cat)) if str(default_cat) in options else 0))
                            st.caption('Jak czytać wykresy poniżej: **Original** (poziom sprzedaży), **Trend** (bazowy kierunek), **Seasonal** (czysty sezon), **Noise** (reszta/zdarzenia).')

                        try:
                            # odtwórz STL dla wybranej kategorii (wartość sprzedaży)
                            demo_df = df_time.copy()
                            demo_df = demo_df.rename(columns={cat_col: "Category"}).copy()
                            demo_df["Category"] = demo_df["Category"].astype(str)
                            demo_df = demo_df[demo_df["Category"] == str(pick)]
                            demo_df = demo_df.groupby("__month", as_index=False)["__value"].sum().sort_values("__month")
                            demo_df["__month"] = pd.to_datetime(demo_df["__month"], errors="coerce")
                            demo_df = demo_df.dropna(subset=["__month"])
                            if demo_df.empty:
                                st.info("Brak danych dla wybranej kategorii.")
                            else:
                                from statsmodels.tsa.seasonal import STL as _STL
                                idx_full = pd.date_range(demo_df["__month"].min(), demo_df["__month"].max(), freq="MS")
                                y = demo_df.set_index("__month")["__value"].reindex(idx_full)
                                y = pd.to_numeric(y, errors="coerce").fillna(0.0).astype(float)

                                if len(y) < 18 or float(np.nanvar(y.values)) == 0.0:
                                    st.info("Za krótki lub stały szereg do dekompozycji STL.")
                                else:
                                    res = _STL(y, period=12, robust=True).fit()
                                    dplot = pd.DataFrame(
                                        {
                                            "period": idx_full,
                                            "original": y.values,
                                            "trend": res.trend,
                                            "seasonal": res.seasonal,
                                            "noise": res.resid,
                                        }
                                    )

                                    st.caption(f'Kategoria: **{pick}** (wartość sprzedaży), dekompozycja STL: original / trend / seasonal / noise')
                                    # mini-narrative: liczbowy skrót (bez ET) dla wybranej kategorii
                                    try:
                                        _idx = int(dplot['seasonal'].abs().idxmax())
                                        _peak_dt = pd.to_datetime(dplot.loc[_idx, 'period'])
                                        _peak_val = float(abs(dplot.loc[_idx, 'seasonal']))
                                        _peak_lab = _peak_dt.strftime('%b %Y')
                                        st.markdown(f"- **Max |Seasonal|**: {_peak_val:,.0f}  ")
                                        st.markdown(f"- **Miesiąc najsilniejszego odchylenia**: {_peak_lab}")
                                    except Exception:
                                        pass


                                    ch_orig = alt.Chart(dplot).mark_line().encode(x=alt.X("period:T", title="Miesiąc"), y=alt.Y("original:Q", title="Original (wartość)")).properties(height=120)
                                    ch_trend = alt.Chart(dplot).mark_line().encode(x=alt.X("period:T", title="Miesiąc"), y=alt.Y("trend:Q", title="Trend")).properties(height=120)
                                    ch_seas = alt.Chart(dplot).mark_line().encode(x=alt.X("period:T", title="Miesiąc"), y=alt.Y("seasonal:Q", title="Seasonal")).properties(height=120)
                                    ch_noise = alt.Chart(dplot).mark_line().encode(x=alt.X("period:T", title="Miesiąc"), y=alt.Y("noise:Q", title="Noise")).properties(height=120)

                                    st.altair_chart(alt.vconcat(ch_orig, ch_trend, ch_seas, ch_noise).resolve_scale(y="independent"), width='stretch')
                        except Exception as e:
                            st.info(f"Nie udało się wyrenderować dekompozycji demo: {e}")
# Executive Takeaway — single final resolver path
            if not b.get("_skip_exec_guidance"):
                _render_exec_takeaway_final(b)

                g = _guidance_for(b["id"], dimension_label=dimension_label, dimension_entity=dimension_entity)
                render_guidance(g.get("sens"), g.get("interpretacja"), g.get("best_practice"))

            st.markdown('<div class="after-guidance"></div>', unsafe_allow_html=True)

        _timings["tab2_insights_total"] = (perf_counter() - _t_tab2) * 1000.0

    _timings["TOTAL_render"] = (perf_counter() - _t0_total) * 1000.0
    if debug_perf and timing_slot is not None:
        _render_timing_ui(timing_slot, _timings)

    # --- DEBUG: Executive Takeaway + checkpoints (global) ---
    try:
        dbg = bool(ctx.get("debug_exec_takeaway") or ctx.get("debug") or ctx.get("debug_mode"))
    except Exception:
        dbg = False

    try:
        if "cot__debug_interp_checkpoints" not in st.session_state:
            st.session_state["cot__debug_interp_checkpoints"] = bool(
                ctx.get("debug_interp_checkpoints") or ctx.get("debug_exec_checkpoints") or False
            )
        cp_dbg = bool(st.session_state.get("cot__debug_interp_checkpoints"))
    except Exception:
        cp_dbg = bool(
            st.session_state.get("cot__debug_interp_checkpoints")
            or ctx.get("debug_interp_checkpoints")
            or ctx.get("debug_exec_checkpoints")
            or False
        )

    try:
        _sticky_cp = bool(st.session_state.get("cot__debug_interp_checkpoints"))
        st.session_state["__cot_exec_dbg_on"] = bool(dbg or cp_dbg or _sticky_cp or st.session_state.get("__cot_exec_dbg_on"))
        st.session_state["cot__debug_interp_checkpoints"] = bool(cp_dbg or _sticky_cp)
        ctx["debug_interp_checkpoints"] = bool(cp_dbg or _sticky_cp)
        ctx["debug_exec_checkpoints"] = bool(cp_dbg or _sticky_cp)
    except Exception:
        pass

    if dbg:
        with st.expander("🧪 DEBUG · Executive Takeaway (global)", expanded=False):
            st.caption("Debug pokazuje stan globalny (session_state) niezależnie od gałęzi.")
            try:
                if st.session_state.pop("__cot_exec_cache_cleared_notice", False):
                    st.success("Wyczyszczono runtime cache ET i wymuszono cold rerun.", icon="✅")
                if st.session_state.pop("__cot_overview_cache_cleared_notice", False):
                    st.success("Wyczyszczono runtime cache Interpretacji i wymuszono cold rerun.", icon="✅")
            except Exception:
                pass
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.button(
                    "🧹 Wyczyść cache Executive Takeaway",
                    key="cot__clear_exec_cache",
                    on_click=_request_exec_cold_run,
                )
            with c2:
                st.button(
                    "🧹 Wyczyść cache Interpretacji",
                    key="cot__clear_overview_interp_cache",
                    on_click=_request_overview_cold_run,
                )
            with c3:
                pass

            st.markdown("### 🧱 Errors (collected)")
            st.write(st.session_state.get(_EXEC_ERR_KEY, []))

            st.markdown("### 🧠 LLM status")
            _status_obj = st.session_state.get(_STATUS_KEY, {})
            st.caption(f"Liczba wpisów status: {len(_status_obj) if isinstance(_status_obj, dict) else 0}")
            with st.expander("Pokaż pełny LLM status", expanded=False):
                st.json(_status_obj)

            st.markdown("### 🗃️ Exec takeaway cache (session_state)")
            _cache_obj = st.session_state.get(_EXEC_CACHE_KEY, {})
            st.caption(f"Liczba wpisów cache: {len(_cache_obj) if isinstance(_cache_obj, dict) else 0}")
            with st.expander("Pokaż pełny cache ET", expanded=False):
                st.json(_cache_obj)

            st.markdown("### 📊 Per-block table (block_id / source / stats_hash / text_hash / gate / preview)")
            rows = st.session_state.get(_EXEC_META_KEY, [])
            if rows:
                try:
                    import pandas as _pd
                    st.dataframe(_debug_df_safe(rows), width='stretch')
                except Exception:
                    st.write(rows)
            else:
                st.info("Brak danych meta — uruchom analizę pytania, aby wypełnić cache/meta.", icon="ℹ️")

    if cp_dbg:
        with st.expander("🧪 DEBUG · Checkpoints Interpretacji / ET (global)", expanded=False):
            st.caption("Checkpointy dbg_cp(...) zapisane w session_state. To jest osobny strumień debugowy od końcowego statusu walidacji.")
            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("🧹 Wyczyść checkpoints", key="cot__clear_exec_checkpoints"):
                    try:
                        st.session_state[_EXEC_CP_KEY] = []
                    except Exception:
                        pass
                    st.success("Wyczyszczono checkpoints dbg_cp(...).", icon="✅")
            with c2:
                cps = st.session_state.get(_EXEC_CP_KEY, [])
                total_cps = len(cps) if isinstance(cps, list) else 0
                st.caption(f"Liczba checkpointów: {total_cps}")

            cps = st.session_state.get(_EXEC_CP_KEY, [])
            cps = cps if isinstance(cps, list) else []

            def _cp_bucket(_where: str) -> str:
                _w = str(_where or "")
                if _w.startswith("overview.interp.cache_lookup") or _w.startswith("overview.interp.cache_hit"):
                    return "interp_cache"
                if _w.startswith("overview.interp.render_repeat") or _w.startswith("overview.interp.render"):
                    return "interp_cache"
                if (
                    _w.startswith("overview.interp.cold_start")
                    or _w.startswith("llm.wrapper.overview.")
                    or _w.startswith("interp.llm.")
                    or _w.startswith("overview.validator_")
                    or _w.startswith("overview.interp.selector_decision")
                    or _w.startswith("overview.interp.final_source")
                    or _w.startswith("overview.interp.before_fallback_decision")
                    or _w.startswith("overview.interp.exec_quality_audit")
                    or _w.startswith("overview.interp.candidate_postprocess")
                    or _w.startswith("overview.interp.runtime_quality_summary")
                ):
                    return "interp_cold"
                if _w.startswith("overview.interp."):
                    return "interp_other"
                if (
                    _w.startswith("exec_takeaway.final_cache_hit")
                    or _w.startswith("exec_takeaway.batch_cache_hit")
                    or _w.startswith("exec_takeaway.batch_cache_complete")
                    or _w.startswith("exec_takeaway.final_cache_partial_miss")
                    or _w.startswith("exec_takeaway.final_cache_poison_detected")
                    or _w.startswith("exec_takeaway.batch_cache_partial_miss")
                    or _w.startswith("exec_takeaway.batch_cache_invalidate_empty_after_normalize")
                    or _w.startswith("exec_takeaway.batch_cache_not_stored_incomplete")
                ):
                    return "exec_cache"
                if (
                    _w.startswith("exec_takeaway.cold_")
                    or _w.startswith("exec_takeaway.batch_first_request")
                    or _w.startswith("exec_takeaway.batch_first_response")
                    or _w.startswith("exec_takeaway.batch_first_exception")
                ):
                    return "exec_cold"
                if _w.startswith("exec_takeaway."):
                    return "exec_core"
                if _w.startswith("seasonality."):
                    return "exec_aux"
                return "other"

            bucket_items = {
                "interp_cache": [],
                "interp_cold": [],
                "interp_other": [],
                "exec_cache": [],
                "exec_cold": [],
                "exec_core": [],
                "exec_aux": [],
                "other": [],
            }
            for cp in cps:
                cpd = cp if isinstance(cp, dict) else {"where": str(cp), "value": cp}
                where = str(cpd.get("where", "") or "")
                bucket_items[_cp_bucket(where)].append(cpd)

            interp_cache_cps = bucket_items["interp_cache"]
            interp_cold_cps = bucket_items["interp_cold"]
            interp_other_cps = bucket_items["interp_other"]
            exec_cache_cps = bucket_items["exec_cache"]
            exec_cold_cps = bucket_items["exec_cold"]
            exec_core_cps = bucket_items["exec_core"]
            exec_aux_cps = bucket_items["exec_aux"]
            other_cps = bucket_items["other"]

            interp_cps = interp_cache_cps + interp_cold_cps + interp_other_cps
            exec_cps = exec_cache_cps + exec_cold_cps + exec_core_cps + exec_aux_cps

            _bucket_sum = (
                len(interp_cache_cps)
                + len(interp_cold_cps)
                + len(interp_other_cps)
                + len(exec_cache_cps)
                + len(exec_cold_cps)
                + len(exec_core_cps)
                + len(exec_aux_cps)
                + len(other_cps)
            )

            csum1, csum2, csum3, csum4, csum5, csum6, csum7, csum8 = st.columns(8)
            csum1.metric("Wszystkie", len(cps))
            csum2.metric("interp cache", len(interp_cache_cps))
            csum3.metric("interp cold", len(interp_cold_cps))
            csum4.metric("interp other", len(interp_other_cps))
            csum5.metric("exec cache", len(exec_cache_cps))
            csum6.metric("exec cold", len(exec_cold_cps))
            csum7.metric("exec", len(exec_core_cps) + len(exec_aux_cps))
            csum8.metric("inne", len(other_cps))

            st.caption(
                "Kontrola sumy: "
                f"{len(interp_cache_cps)} + {len(interp_cold_cps)} + {len(interp_other_cps)} + "
                f"{len(exec_cache_cps)} + {len(exec_cold_cps)} + {len(exec_core_cps)} + "
                f"{len(exec_aux_cps)} + {len(other_cps)} = {_bucket_sum}"
            )
            _et_force_cold_pending = bool(
                st.session_state.get("__cot_force_cold_exec_run")
                or st.session_state.get("__cot_force_cold_audit_run")
                or _exec_force_cold_pending()
            )
            _et_last_mode_obj = st.session_state.get(_EXEC_LAST_RUN_MODE_KEY, {})
            _et_last_mode = str((_et_last_mode_obj or {}).get("mode") or "")
            _et_pending_req = _get_exec_force_cold_request_id()
            _et_consumed_req = _get_exec_force_cold_consumed_id()
            st.caption(
                "Status ET MODE pokazuje ostatni run ET; tabele checkpointów niżej mogą zawierać historię z poprzednich rerunów."
            )
            if _et_force_cold_pending:
                st.caption(
                    f"ET cold request id: {_et_pending_req} | consumed: {_et_consumed_req}"
                )
            if _et_last_mode == "cold_confirmed" or exec_cold_cps:
                st.success("ET MODE: COLD CONFIRMED", icon="✅")
            elif _et_last_mode in {"pending_force_cold", "cold_requested"} or _et_force_cold_pending:
                st.info("ET MODE: PENDING FORCE COLD", icon="ℹ️")
            elif _et_last_mode in {"warm_cache", "warm_batch_cache"} or exec_cache_cps:
                st.warning("ET MODE: WARM CACHE", icon="⚠️")
            if _bucket_sum != len(cps):
                st.warning(
                    f"Niespójność bucketów checkpointów: suma bucketów = {_bucket_sum}, a Wszystkie = {len(cps)}.",
                    icon="⚠️",
                )

            def _cp_rows(items):
                rows = []
                for i, cp in enumerate(items):
                    cp = cp if isinstance(cp, dict) else {"value": cp}
                    rows.append({
                        "idx": i,
                        "where": cp.get("where", ""),
                        "block_id": cp.get("block_id", ""),
                        "from_path": cp.get("from_path", ""),
                        "hash": cp.get("hash", ""),
                        "mode": cp.get("mode", ""),
                        "topic_hint": cp.get("topic_hint", cp.get("topic", "")),
                        "src": cp.get("src", ""),
                        "fallback_used": cp.get("fallback_used", ""),
                        "repair_used": cp.get("repair_used", ""),
                        "text_len": cp.get("text_len", cp.get("read_len", cp.get("written_len", ""))),
                        "cache_key": cp.get("cache_key", ""),
                    })
                return rows

            st.markdown("### 🧭 Checkpoints (raw JSON)")
            st.json(cps)

            if cps:
                try:
                    import pandas as _pd
                    rows_all = _cp_rows(cps)
                    rows_interp = _cp_rows(interp_cps)
                    rows_interp_cache = _cp_rows(interp_cache_cps)
                    rows_interp_cold = _cp_rows(interp_cold_cps)
                    rows_interp_other = _cp_rows(interp_other_cps)
                    rows_exec_cache = _cp_rows(exec_cache_cps)
                    rows_exec_cold = _cp_rows(exec_cold_cps)
                    rows_exec = _cp_rows(exec_cps)
                    rows_exec_core = _cp_rows(exec_core_cps)
                    rows_exec_aux = _cp_rows(exec_aux_cps)
                    rows_other = _cp_rows(other_cps)

                    st.markdown("### 📊 Checkpoints table · wszystkie")
                    st.dataframe(_debug_df_safe(rows_all), width='stretch', hide_index=True)

                    st.markdown(
                        "### 🧩 Sekcja Interpretacja · "
                        f"total: {len(rows_interp)} = cache {len(rows_interp_cache)} + cold {len(rows_interp_cold)} + other {len(rows_interp_other)}"
                    )
                    if rows_interp:
                        st.dataframe(_debug_df_safe(rows_interp), width='stretch', hide_index=True)
                    else:
                        st.info("Brak checkpointów Interpretacja w __cot_exec_cp_v1.", icon="ℹ️")

                    st.markdown(f"### 🗂️ Interpretacja · cache-path · liczba: {len(rows_interp_cache)}")
                    if rows_interp_cache:
                        st.dataframe(_debug_df_safe(rows_interp_cache), width='stretch', hide_index=True)
                    else:
                        st.info("Brak checkpointów cache-path dla Interpretacja.", icon="ℹ️")

                    st.markdown(f"### ❄️ Interpretacja · cold-path · liczba: {len(rows_interp_cold)}")
                    if rows_interp_cold:
                        st.dataframe(_debug_df_safe(rows_interp_cold), width='stretch', hide_index=True)
                    else:
                        st.info("Brak checkpointów cold-path dla Interpretacja.", icon="ℹ️")

                    st.markdown(f"### 🧾 Interpretacja · other · liczba: {len(rows_interp_other)}")
                    if rows_interp_other:
                        st.dataframe(_debug_df_safe(rows_interp_other), width='stretch', hide_index=True)
                    else:
                        st.info("Brak checkpointów other dla Interpretacja.", icon="ℹ️")

                    st.markdown(
                        "### 🧱 Sekcja ET · "
                        f"total: {len(rows_exec)} = cache {len(rows_exec_cache)} + cold {len(rows_exec_cold)} + "
                        f"core {len(rows_exec_core)} + aux {len(rows_exec_aux)}"
                    )
                    if rows_exec:
                        st.dataframe(_debug_df_safe(rows_exec), width='stretch', hide_index=True)
                    else:
                        st.info("Brak checkpointów ET w __cot_exec_cp_v1.", icon="ℹ️")

                    st.markdown(f"### ❄️ ET · cold-path · liczba: {len(rows_exec_cold)}")
                    if rows_exec_cold:
                        st.dataframe(_debug_df_safe(rows_exec_cold), width='stretch', hide_index=True)
                    else:
                        st.info("Brak checkpointów cold-path dla ET.", icon="ℹ️")

                    st.markdown(f"### 🗂️ ET · cache-path · liczba: {len(rows_exec_cache)}")
                    if rows_exec_cache:
                        st.dataframe(_debug_df_safe(rows_exec_cache), width='stretch', hide_index=True)
                    else:
                        st.info("Brak checkpointów cache-path dla ET.", icon="ℹ️")

                    st.markdown(f"### 🧱 ET · core (exec_takeaway.*) · liczba: {len(rows_exec_core)}")
                    if rows_exec_core:
                        st.dataframe(_debug_df_safe(rows_exec_core), width='stretch', hide_index=True)
                    else:
                        st.info("Brak checkpointów exec_takeaway.* w __cot_exec_cp_v1.", icon="ℹ️")

                    st.markdown(f"### 🪄 ET · aux (seasonality.*) · liczba: {len(rows_exec_aux)}")
                    if rows_exec_aux:
                        st.dataframe(_debug_df_safe(rows_exec_aux), width='stretch', hide_index=True)
                    else:
                        st.info("Brak checkpointów pomocniczych ET w __cot_exec_cp_v1.", icon="ℹ️")

                    if rows_other:
                        st.markdown(f"### 🪵 Sekcja inne prefixy · liczba: {len(rows_other)}")
                        st.dataframe(_debug_df_safe(rows_other), width='stretch', hide_index=True)
                except Exception:
                    st.write(cps)
            else:
                st.info("Brak checkpointów — włącz checkbox w sidebarze i uruchom pytanie ponownie.", icon="ℹ️")

    return {"chart_meta": {"kind": "cot", "mode": mode}, "chart_context": {"cot": True}}

# Backward compatibility (older imports)
def data_chat_render(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
    return render(df, ctx)

def _upgrade_share_interp_to_executive_legacy(out: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(out or {})
    driver = _first_not_none(
        stats.get("driver_category_name"),
        stats.get("leader_category"),
        stats.get("top1_category"),
        "pozycja",
    )
    start_pct = _first_not_none(
        stats.get("share_start_pct"),
        stats.get("top1_start_pct"),
        stats.get("driver_start_pct"),
    )
    end_pct = _first_not_none(
        stats.get("share_end_pct"),
        stats.get("top1_end_pct"),
        stats.get("driver_end_pct"),
    )
    delta_pp = _first_not_none(
        stats.get("delta_pp"),
        stats.get("top1_delta_pp"),
        stats.get("driver_delta_pp"),
    )

    if driver and start_pct is not None and end_pct is not None and delta_pp is not None:
        try:
            _delta_num = float(delta_pp)
        except Exception:
            _delta_num = 0.0

        if _delta_num > 0:
            _mech = "wzmacnianie pozycji i przejęcie dodatkowego udziału w miksie"
            _decision = (
                f'Dlatego: utrzymaj wsparcie i ekspozycję dla „{driver}”, '
                f'a dodatkową alokację kieruj tam, gdzie wzrost udziału dalej się materializuje.'
            )
        elif _delta_num < 0:
            _mech = "oddawanie udziału i osłabienie pozycji w miksie"
            _decision = (
                f'Dlatego: przesuń część wsparcia do rosnących pozycji, '
                f'a dla „{driver}” sprawdź cenę, ekspozycję i rolę asortymentową zanim zwiększysz budżet.'
            )
        else:
            _mech = "utrzymanie pozycji przy ograniczonym przesunięciu miksu"
            _decision = (
                f'Dlatego: utrzymaj bazową ekspozycję dla „{driver}” i obserwuj, '
                f'czy zmiana udziału utrzyma się w kolejnych okresach.'
            )

        s = (
            f'KPI: udział (%) — „{driver}” zmienił udział z {_cot_fmt_pct(start_pct)} do {_cot_fmt_pct(end_pct)} '
            f'({_cot_fmt_pp(delta_pp)}), co oznacza {_mech}. '
            f'{_decision}'
        )

        txt = str(out.get("one_sentence") or "")
        tl = str(s or txt or "").lower()

        _has_mech = any(w in tl for w in [
            "oddawanie udziału",
            "osłabienie pozycji",
            "transfer udziału",
            "trwałe przesunięcie",
            "erozj",
        ])
        _has_decision = any(w in tl for w in [
            "cena",
            "alokacja wsparcia",
            "decyzja:",
            "dlatego:",
            "przesuń",
        ])

        _mech = None
        try:
            _delta = float(delta_pp) if delta_pp is not None else None
        except Exception:
            _delta = None

        if not _has_mech:
            if _delta is not None and _delta < 0:
                _mech = "oddawanie udziału i osłabienie pozycji"
            elif _delta is not None and _delta > 0:
                _mech = "wzmacnianie pozycji i przejęcie dodatkowego udziału"
            elif "leader" in str(stats).lower() or "topn" in str(stats).lower() or "rdzeń" in tl:
                _mech = "erozję rdzenia"
            else:
                _mech = "trwałe przesunięcie struktury"

            if ";" in s:
                left, right = s.split(";", 1)
                s = left.rstrip(" .") + f", co oznacza {_mech};" + right.lstrip()
            elif "." in s:
                left, right = s.split(".", 1)
                s = left.rstrip(" .") + f", co oznacza {_mech}. " + right.lstrip()
            else:
                s = s.rstrip(" .") + f", co oznacza {_mech}."

        tl = str(s or "").lower()
        _has_decision = any(w in tl for w in [
            "cena",
            "alokacja wsparcia",
            "decyzja:",
            "dlatego:",
            "przesuń",
        ])

        if not _has_decision:
            _decision = "Dlatego: przesuń część alokacji wsparcia i ekspozycji do pozycji o lepszej trajektorii udziału."
            if "." in s:
                s = s.rstrip()
                if not s.endswith("."):
                    s += "."
                s += " " + _decision
            else:
                s = s.rstrip(" .") + ". " + _decision

        out["one_sentence"] = (s or "").strip()

    return out


def _upgrade_value_interp_to_executive(out: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(out or {})
    total_value = _first_not_none(stats.get("total_value"), stats.get("sum_value"), stats.get("total_sales"))
    peak_value = _first_not_none(stats.get("peak_value"), stats.get("max_value"))
    peak_month_raw = _first_not_none(stats.get("peak_month"), stats.get("peak_month_label"))
    peak_month = _format_month_pl(str(peak_month_raw)) if peak_month_raw else None

    if total_value is not None and peak_value is not None and peak_month:
        out["one_sentence"] = (
            f"Całkowita wartość sprzedaży wyniosła {_cot_fmt_pln(total_value)}, a szczyt osiągnęła {_cot_fmt_pln(peak_value)} "
            f"w {peak_month}, co wskazuje na silną koncentrację przychodu w okresach szczytu. "
            f"Dlatego: zabezpiecz dostępność i ekspozycję przed pikiem oraz traktuj miesiące poza szczytem jako obszar selektywnej aktywacji."
        )
    try:
        _s = str(out.get("one_sentence") or "")
        _s = re.sub(r"\bw\s+w\b", "w", _s, flags=re.IGNORECASE)
        _s = re.sub(r"\s{2,}", " ", _s).strip()
        out["one_sentence"] = _s
    except Exception:
        pass

    return out


_upgrade_value_interp_to_executive_base = _upgrade_value_interp_to_executive


def _upgrade_value_interp_to_executive_v2(out: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    out = _upgrade_value_interp_to_executive_base(out, stats)
    out = dict(out or {})
    stats = stats if isinstance(stats, dict) else {}

    peak_month_raw = _first_not_none(stats.get("peak_month"), stats.get("peak_month_label"))
    peak_month = _format_month_pl(str(peak_month_raw)) if peak_month_raw else None
    peak_window = f"przed szczytem w {peak_month}" if peak_month else "przed szczytem"

    recs = [str(x or "").strip() for x in list(out.get("recommendations") or []) if str(x or "").strip()]

    def _is_soft_value_recommendation(row: str) -> bool:
        rl = str(row or "").lower()
        return not any(
            token in rl
            for token in [
                "zabezpiecz",
                "ogranicz",
                "przenieś",
                "przenies",
                "ekspozycj",
                "dostępno",
                "dostepno",
                "fill-rate",
                "fill rate",
                "aktywacj",
                "m/m",
                "rotacj",
            ]
        )

    if (len(recs) < 2) or all(_is_soft_value_recommendation(r) for r in recs):
        recs = [
            f"Zabezpiecz dostępność i ekspozycję {peak_window}, monitorując odchylenie m/m oraz fill-rate w tygodniach poprzedzających pik.",
            "Poza oknem szczytu ogranicz szeroką aktywację i przenieś wsparcie do miesięcy z najwyższą rotacją, aby nie rozpraszać budżetu.",
        ]

    out["recommendations"] = recs[:3]
    if not str(out.get("recommendation") or "").strip():
        out["recommendation"] = out["recommendations"][0]

    return out


_upgrade_value_interp_to_executive = _upgrade_value_interp_to_executive_v2


def _upgrade_share_interp_to_executive(out: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(out or {})
    stats = stats if isinstance(stats, dict) else {}

    driver = _first_not_none(
        stats.get("driver_dimension_value"),
        stats.get("driver_category_name"),
        stats.get("leader_dimension_value"),
        stats.get("leader_category"),
        stats.get("top1_dimension_value"),
        stats.get("top1_category"),
        stats.get("primary_dimension_value"),
        stats.get("primary_category"),
        "pozycja",
    )
    start_pct = _first_not_none(
        stats.get("share_start_pct"),
        stats.get("top1_start_pct"),
        stats.get("driver_start_pct"),
        stats.get("leader_start_pct"),
    )
    end_pct = _first_not_none(
        stats.get("share_end_pct"),
        stats.get("top1_end_pct"),
        stats.get("driver_end_pct"),
        stats.get("leader_end_pct"),
    )
    delta_pp = _first_not_none(
        stats.get("delta_pp"),
        stats.get("top1_delta_pp"),
        stats.get("driver_delta_pp"),
        stats.get("leader_delta_pp"),
    )

    if driver and start_pct is not None and end_pct is not None and delta_pp is not None:
        try:
            delta_num = float(delta_pp)
        except Exception:
            delta_num = 0.0

        start_pct_fmt = _cot_fmt_pct(start_pct)
        end_pct_fmt = _cot_fmt_pct(end_pct)
        delta_pp_fmt = _cot_fmt_pp(delta_pp)

        if delta_num > 0:
            mechanism = "wzmacnianie pozycji i przejęcie dodatkowego udziału w miksie"
            decision = (
                f'Dlatego: utrzymaj wsparcie i ekspozycję dla "{driver}", dopóki udział '
                f'pozostaje w pobliżu {end_pct_fmt}, a dodatkową alokację kieruj tam, gdzie '
                f'wzrost udziału nadal wynosi {delta_pp_fmt}.'
            )
            recommendations = [
                f'Utrzymaj priorytet dla "{driver}", dopóki udział pozostaje w pobliżu {end_pct_fmt} '
                f'i przewaga względem startu okresu wynosi {delta_pp_fmt}.',
                f'Dodatkową alokację kieruj do pozycji, które utrzymują wzrost udziału o co najmniej '
                f'{delta_pp_fmt} lub kończą okres powyżej {end_pct_fmt}.',
            ]
        elif delta_num < 0:
            mechanism = "oddawanie udziału i osłabienie pozycji w miksie"
            decision = (
                f'Dlatego: przesuń 1,0 pp wsparcia do pozycji rosnących, bo "{driver}" spadł '
                f'z {start_pct_fmt} do {end_pct_fmt} ({delta_pp_fmt}), a dla tej pozycji sprawdź '
                f'cenę i ekspozycję przed zwiększeniem budżetu.'
            )
            recommendations = [
                f'Przesuń 1,0 pp wsparcia do pozycji rosnących, bo "{driver}" spadł z {start_pct_fmt} '
                f'do {end_pct_fmt} ({delta_pp_fmt}).',
                f'Dla "{driver}" sprawdź cenę i ekspozycję, jeśli udział utrzymuje się poniżej '
                f'{end_pct_fmt} po spadku o {delta_pp_fmt}.',
            ]
        else:
            mechanism = "utrzymanie pozycji przy ograniczonym przesunięciu miksu"
            decision = (
                f'Dlatego: utrzymaj bazową ekspozycję dla "{driver}", jeśli udział pozostaje blisko '
                f'{end_pct_fmt}, a odchylenie nie przekracza {delta_pp_fmt}.'
            )
            recommendations = [
                f'Utrzymaj bazową ekspozycję dla "{driver}", jeśli udział pozostaje blisko {end_pct_fmt}, '
                f'a zmiana nie przekracza {delta_pp_fmt}.',
                f'Przegląd alokacji wykonaj ponownie, jeśli udział odchyli się od {end_pct_fmt} '
                f'o więcej niż 0.5 pp.',
            ]

        out["one_sentence"] = (
            f'KPI: udział (%) — "{driver}" zmienił udział z {start_pct_fmt} do {end_pct_fmt} '
            f'({delta_pp_fmt}), co oznacza {mechanism}. {decision}'
        ).strip()
        out["recommendations"] = recommendations[:3]
        out["recommendation"] = out["recommendations"][0]

    return out
