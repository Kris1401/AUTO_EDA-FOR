from __future__ import annotations

import re
from typing import Any, Dict, Optional

import pandas as pd

def cp0_detect_has_time(df: pd.DataFrame, roles: Dict[str, Any] | None = None) -> tuple[bool, str | None]:
    """
    1) Czy dane mają czas? (TYLKO dane, nie pytanie)
    - najpierw roles['time_col'] jeśli poprawny
    - potem datetime dtype
    - potem lekko parsowalne kolumny (bez ciężkich prób na całej kolumnie)
    """
    roles = roles or {}
    time_col = roles.get("time_col")
    if isinstance(time_col, str) and time_col in df.columns:
        return True, time_col

    # 1) dtype datetime
    dt_cols = _infer_datetime_cols(df)
    if dt_cols:
        return True, dt_cols[0]

    # 2) lekka heurystyka nazw (bez parsowania całej kolumny)
    name_hints = ("date", "time", "datetime", "data", "timestamp", "ts")
    for c in df.columns:
        cl = str(c).lower()
        if any(h in cl for h in name_hints):
            return True, str(c)

    return False, None


def cp0_time_intent_from_question(question: str) -> str:
    """
    2) Co pytanie mówi o czasie? (INTENCJA CZASU)
    Zwraca: OVER_TIME | POINT_IN_TIME | UNSPECIFIED
    """
    q = (question or "").strip().lower()

    # A) OVER_TIME
    over_time_kw = [
        "w czasie", "jak zmienia", "zmienia się", "trend", "trendy",
        "miesiąc po miesiącu", "miesiac po miesiacu", "na przestrzeni",
        "yoy", "mom", "rok do roku", "m/m", "r/r", "qoq", "kwartał po kwartale",
        "dzien po dniu", "tydzień po tygodniu", "tydzien po tygodniu",
        "over time", "time series", "sezon", "sezonowo", "sezonowość", "seasonality"
    ]
    if any(k in q for k in over_time_kw):
        return "OVER_TIME"

    # B) POINT_IN_TIME (snapshot)
    snapshot_kw = [
        "w czerwcu", "w maju", "w lipcu", "w sierpniu", "we wrześniu", "we wrzesniu",
        "w październiku", "w pazdzierniku", "w listopadzie", "w grudniu",
        "za ostatni okres", "ostatni miesiąc", "ostatni miesiac", "ostatni kwartał", "ostatni kwartal",
        "q1", "q2", "q3", "q4",
        "w 20", "w 19",  # proste „w 2021” itd. (bez regexów – wystarcza jako sygnał)
        "snapshot", "point in time"
    ]
    if any(k in q for k in snapshot_kw):
        return "POINT_IN_TIME"

    # C) UNSPECIFIED
    return "UNSPECIFIED"


def cp0_time_mode(has_time: bool, time_intent: str) -> str:
    """
    3) Ustal time_mode (JEDNA DECYZJA)
    """
    if not has_time:
        return "STATIC"
    if time_intent == "POINT_IN_TIME":
        return "STATIC_SNAPSHOT"
    if time_intent == "OVER_TIME":
        return "OVER_TIME"
    return "STATIC"


def cp0_branch_from_intent_and_time(intent: str, time_mode: str, chart_spec: Dict[str, Any] | None) -> str:
    """
    4) Wybór gałęzi Abeli (obowiązkowy) — zapis do analysis_struct.branch
    """
    intent = (intent or "").strip().lower()
    time_mode = (time_mode or "").strip().upper()

    if intent == "distribution":
        return "DISTRIBUTION"

    if intent == "composition":
        if time_mode in {"OVER_TIME", "FULL_PERIOD"}:
            return "COMPOSITION_CHANGING_OVER_TIME"
        return "COMPOSITION_STATIC"

    if intent == "comparison":
        if time_mode in {"OVER_TIME", "FULL_PERIOD"}:
            return "COMPARISON_OVER_TIME"
        return "COMPARISON_AMONG_ITEMS"

    if intent == "relationship":
        # minimalny podział: 2 vs 3 zmienne (heurystyka z chart_spec)
        ps = (chart_spec or {}).get("primary_chart") if isinstance(chart_spec, dict) else None
        ps = ps if isinstance(ps, dict) else {}
        x = ps.get("x") or (chart_spec or {}).get("x")
        y = ps.get("y") or (chart_spec or {}).get("y")
        color = ps.get("color") or (chart_spec or {}).get("color")
        z_like = color  # w v1 to wystarczy jako „trzecia zmienna”
        if x and y and z_like:
            return "RELATIONSHIP_THREE_VARIABLES"
        return "RELATIONSHIP_TWO_VARIABLES"

    # fallback bezpieczny
    return "DISTRIBUTION"


def cp0_compute_checkpoint0_analysis_struct(
    df: pd.DataFrame,
    question: str,
    chart_spec: Dict[str, Any] | None,
    schema_ctx: Dict[str, Any] | None,
    roles: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    CHECKPOINT 0 — single source of truth:
    has_time -> time_intent -> time_mode -> branch
    """
    roles = roles or {}
    schema_ctx = schema_ctx or {}

    has_time, time_col = cp0_detect_has_time(df, roles=roles)

    # jeśli schema_ctx ma date_cols, traktujemy je jako dowód has_time (stabilne)
    date_cols = schema_ctx.get("date_cols") or []
    if date_cols and not has_time:
        has_time = True
        if not time_col:
            time_col = str(date_cols[0])

    time_intent = cp0_time_intent_from_question(question)
    time_mode = cp0_time_mode(has_time, time_intent)

    intent = ""
    if isinstance(chart_spec, dict):
        intent = str(chart_spec.get("intent") or "").strip().lower()
        if not intent and isinstance(chart_spec.get("primary_chart"), dict):
            intent = str(chart_spec["primary_chart"].get("intent") or "").strip().lower()

    branch = cp0_branch_from_intent_and_time(intent=intent, time_mode=time_mode, chart_spec=chart_spec)

    return {
        "checkpoint": 0,
        "has_time": bool(has_time),
        "time_col": time_col,
        "time_intent": time_intent,   # OVER_TIME | POINT_IN_TIME | UNSPECIFIED
        "time_mode": time_mode,       # STATIC | STATIC_SNAPSHOT | OVER_TIME | FULL_PERIOD
        "analysis_intent": intent,     # distribution | composition | comparison | relationship
        "branch": branch,             # Abeli branch (z time_mode)
        "__question": question,       # ✅ CP1: twardy dostęp do intencji pytania (kategoria vs kraj)
    }


def _infer_datetime_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                cols.append(str(c))
        except Exception:
            continue
    return cols
