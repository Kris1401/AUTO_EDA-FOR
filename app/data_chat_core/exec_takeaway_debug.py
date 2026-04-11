# -*- coding: utf-8 -*-
"""Executive Takeaway debug renderer (global).

Minimal helper to visualize:
- dc_errors
- dc_llm_status_v1
- exec_takeaway_cache_v3
- exec_takeaway_meta_v1

This is intentionally decoupled from branches: call it from 03_Data_Chat.py once per run.
"""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
import streamlit as st


ERR_KEY = "dc_errors"
STATUS_KEY = "dc_llm_status_v1"
CACHE_KEY = "exec_takeaway_cache_v3"
META_KEY = "exec_takeaway_meta_v1"
CP_KEY = "__cot_exec_cp_v1"
EXEC_LAST_RUN_MODE_KEY = "__cot_exec_last_run_mode_v1"
EXEC_FORCE_COLD_REQ_KEY = "__cot_exec_force_cold_request_id_v1"
EXEC_FORCE_COLD_DONE_KEY = "__cot_exec_force_cold_consumed_id_v1"


def _get_int(ss: Dict[str, Any], key: str) -> int:
    try:
        return int(ss.get(key) or 0)
    except Exception:
        return 0


def _exec_force_cold_pending(ss: Dict[str, Any]) -> bool:
    try:
        return _get_int(ss, EXEC_FORCE_COLD_REQ_KEY) > _get_int(ss, EXEC_FORCE_COLD_DONE_KEY)
    except Exception:
        return False


def _set_exec_last_run_mode(ss: Dict[str, Any], mode: str, **meta: Any) -> None:
    try:
        ss[EXEC_LAST_RUN_MODE_KEY] = {
            "mode": str(mode or ""),
            **dict(meta or {}),
        }
    except Exception:
        return


def _request_exec_cold_run(session_state: Dict[str, Any] | None = None) -> None:
    ss = session_state if session_state is not None else st.session_state
    try:
        for key in [
            CACHE_KEY,
            META_KEY,
            ERR_KEY,
            "exec_takeaway_cache",
            "exec_takeaway_meta_v1",
            "exec_takeaway_errors_v1",
            "__cot_exec_batch_cache_v1",
            "__cot_exec_final_cache_v1",
            "__cot_exec_cold_audit_block_stats_v1",
        ]:
            ss.pop(key, None)
        ss[CP_KEY] = []
        ss["__cot_exec_cache_cleared_notice"] = True
        ss["__cot_force_cold_exec_run"] = True
        _req_id = _get_int(ss, EXEC_FORCE_COLD_REQ_KEY) + 1
        ss[EXEC_FORCE_COLD_REQ_KEY] = _req_id
        _set_exec_last_run_mode(ss, "pending_force_cold", request_id=_req_id)
    except Exception:
        return


def render_exec_takeaway_debug(session_state: Dict[str, Any] | None = None) -> None:
    ss = session_state if session_state is not None else st.session_state

    with st.expander("🧪 DEBUG · Executive Takeaway (global)", expanded=False):
        try:
            if ss.pop("__cot_exec_cache_cleared_notice", False):
                st.success("Wyczyszczono runtime cache ET i wymuszono cold rerun.", icon="✅")
        except Exception:
            pass
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.button(
                "🧹 Wyczyść cache Executive Takeaway",
                key="dbg_exec_takeaway_clear",
                on_click=_request_exec_cold_run,
                args=(ss,),
            )
        with col_r:
            st.caption("Debug pokazuje stan globalny (session_state) niezależnie od gałęzi.")

        _last_mode_obj = ss.get(EXEC_LAST_RUN_MODE_KEY, {})
        _last_mode = str((_last_mode_obj or {}).get("mode") or "")
        _pending = bool(
            ss.get("__cot_force_cold_exec_run")
            or ss.get("__cot_force_cold_audit_run")
            or _exec_force_cold_pending(ss)
        )
        if _pending:
            st.info(
                f"ET MODE (shared): PENDING FORCE COLD | request={_get_int(ss, EXEC_FORCE_COLD_REQ_KEY)} | "
                f"consumed={_get_int(ss, EXEC_FORCE_COLD_DONE_KEY)}",
                icon="ℹ️",
            )
        elif _last_mode == "cold_confirmed":
            st.success("ET MODE (shared): COLD CONFIRMED", icon="✅")
        elif _last_mode in {"warm_cache", "warm_batch_cache"}:
            st.warning("ET MODE (shared): WARM CACHE", icon="⚠️")

        st.markdown("### 🧱 Errors (collected)")
        st.json(ss.get(ERR_KEY, []) or [])

        st.markdown("### 🧠 LLM status")
        st.json(ss.get(STATUS_KEY, {}) or {})

        st.markdown("### 🧠 Exec takeaway cache (session_state)")
        st.json(ss.get(CACHE_KEY, {}) or {})

        st.markdown("### 📊 Per-block table (block_id / source / stats_hash / text_hash / gate / preview)")
        meta = ss.get(META_KEY, None)

        rows: List[Dict[str, Any]] = []
        if isinstance(meta, dict):
            # stary format dict -> list rows
            for cache_key, m in meta.items():
                if not isinstance(m, dict):
                    rows.append({"cache_key": cache_key, "value": str(m)})
                    continue
                rows.append({
                    "cache_key": cache_key,
                    "block_id": m.get("block_id"),
                    "intent": m.get("intent"),
                    "source": m.get("source"),
                    "stats_hash": m.get("stats_hash"),
                    "text_hash": m.get("text_hash"),
                    "gate": m.get("gate"),
                    "preview": (m.get("text") or "")[:120],
                })

        elif isinstance(meta, list):
            # nowy format list -> użyj wprost (ale obroń się, gdy elementy nie są dict)
            for item in meta:
                if isinstance(item, dict):
                    vv = dict(item)
                    vv.setdefault("preview", (vv.get("text") or vv.get("preview") or "")[:120])
                    rows.append(vv)
                else:
                    rows.append({"value": str(item)})
        else:
            rows = []

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, width='stretch', hide_index=True)
        else:
            st.info("Brak danych meta — uruchom analizę pytania, aby wypełnić cache/meta.", icon="ℹ️")
