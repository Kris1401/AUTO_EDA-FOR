# app/data_chat_core/exec_takeaway.py
from __future__ import annotations

import hashlib
import time
import json
import re
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

# ─────────────────────────────────────────────────────────────
# Executive Takeaway Engine (minimal patch, hard contract ready)
# Goals:
# 1) One cache namespace (v3) + auto-clear old v2 keys
# 2) LLM ON when available (via session_state["dc_llm_text"]) + explicit source
# 3) Meta logging per block (block_id/source/stats_hash/text_hash/preview)
# 4) Two hashes: stats_hash + text_hash (dedupe by text_hash)
# 5) Never crash the app – errors collected to session_state["dc_errors"]
# ─────────────────────────────────────────────────────────────

CACHE_KEY_V3 = "exec_takeaway_cache_v3"
META_KEY = "exec_takeaway_meta_v1"
ERRORS_KEY = "dc_errors"
STATUS_KEY = "dc_llm_status_v1"

# Backward-compat (old) cache key names you might already have
_OLD_CACHE_KEYS = ("exec_takeaway_cache_v2", "exec_takeaway_cache", "exec_takeaway_cache_v1")


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def _sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _hash_obj(obj: Any) -> str:
    return _sha1_short(_safe_json_dumps(obj))


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _ensure_dict(session_state: Optional[dict], key: str) -> dict:
    if session_state is None:
        return {}
    if key not in session_state or not isinstance(session_state.get(key), dict):
        session_state[key] = {}
    return session_state[key]


def _ensure_list(session_state: Optional[dict], key: str) -> list:
    if session_state is None:
        return []
    if key not in session_state or not isinstance(session_state.get(key), list):
        session_state[key] = []
    return session_state[key]


def _append_error(session_state: Optional[dict], where: str, err: Exception, extra: Optional[dict] = None) -> None:
    if session_state is None:
        return
    errs = _ensure_list(session_state, ERRORS_KEY)
    rec = {"where": where, "error": f"{type(err).__name__}: {err}"}
    if extra:
        rec["extra"] = extra
    errs.append(rec)


def ensure_exec_cache(session_state: Optional[dict]) -> dict:
    """
    Ensure v3 cache exists. If old caches exist, keep them but do not read from them
    unless v3 is empty (migration is intentionally conservative).
    """
    cache_v3 = _ensure_dict(session_state, CACHE_KEY_V3)

    # If v3 empty and old cache exists, we can optionally import – but for stability we clear old keys.
    # The v2 cache carries "bad" generic strings and will mask improvements.
    if session_state is not None:
        # hard clear old cache dicts to prevent stale reuse
        for k in _OLD_CACHE_KEYS:
            if k in session_state and isinstance(session_state.get(k), dict) and session_state.get(k):
                session_state[k] = {}
    return cache_v3


def _build_cache_key_v4(intent: str, block_id: str, stats_hash: str, question: str) -> str:
    # v3 key includes explicit block_id + stats hash; question shortened to avoid infinite variety
    q_sig = _sha1_short((question or "")[:200])
    return f"exec:v4:{intent}:{block_id}:{stats_hash}:{q_sig}"


def _quality_gate_hard(
    text: str,
    *,
    require_numbers: bool = True,
    require_exact_two_sentences: bool = False,
    require_no_ellipsis: bool = True,
    require_terminal_punct: bool = True,
    require_decision_in_sentence2: bool = False,
) -> Tuple[bool, str]:
    """
    Minimal hard gate for Option B:
    - 1–2 sentences (split by .!?)
    - has at least one number token (optional)
    - contains a decision verb hint in sentence 2 if two sentences (optional soft)
    """
    t = (text or "").strip()
    if not t:
        return False, "empty"
    # 1–2 sentences (rough)
    sents = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", t) if s.strip()]
    if require_exact_two_sentences:
        if len(sents) != 2:
            return False, f"sentences={len(sents)}"
    else:
        if len(sents) < 1 or len(sents) > 2:
            return False, f"sentences={len(sents)}"

    if require_no_ellipsis and ("..." in t or "…" in t or t.rstrip().endswith("..")):
        return False, "ellipsis"

    if require_terminal_punct and not re.search(r"[\.\!\?]$", t):
        return False, "no_terminal_punct"
    if require_numbers and not re.search(r"\d", t):
        return False, "no_number"

    if require_decision_in_sentence2 and len(sents) == 2:
        s2 = sents[1].lower()
        # very simple Polish decision/action cues (avoid false positives)
        decision_kw = (
            "rekomend", "zalec", "decyz", "ustaw", "skup", "skup się", "priorytet",
            "wdr", "ogranicz", "zwi", "zmniej", "podnie", "obniż", "monitor",
            "segment", "prog", "kpi",
            "recommend", "recommended", "should", "focus", "prioritize", "it is recommended", "we recommend", "to optimize", "to improve", "to reduce", "to increase", "to monitor", "consider",
        )
        if not any(k in s2 for k in decision_kw):
            return False, "no_decision_sentence2"
    return True, "ok"

# ─────────────────────────────────────────────────────────────
# ET v1.0 — minimal validator (McKinsey-grade, realistic)
# HARD: PL-only, exactly 2 sentences, >=2 numbers in sentence1, decision verb in sentence2
# SOFT warnings: ellipsis, missing terminal punct, too long
# ─────────────────────────────────────────────────────────────

_DECISION_VERBS_PL = (
    "rekomend", "zalec", "wdroż", "wdrozyc", "wdrożyć",
    "zwiększ", "zwieksz", "zmniejsz", "ogranicz", "skup", "skup się",
    "priorytet", "przetest", "monitor", "zweryfik",
    "usuń", "usun", "rozdziel", "segment", "standaryz",
    "zabezpiecz", "zautomatyz", "porówn", "porown",
    "wyklucz", "ustaw", "ustal", "określ", "wyznacz", "zmień", "zmien", "podnieś", "obniż", "zalecam",
)

_NUM_RE = re.compile(r"\d+(?:[\,\.]\d+)?")

def _looks_non_polish(text: str) -> bool:
    """Heuristic EN detector: no Polish diacritics + many English stopwords/cues."""
    t = (text or "").strip().lower()
    if not t:
        return False
    has_pl = bool(re.search(r"[ąćęłńóśźż]", t))
    en_cues = (" the ", " and ", " to ", " of ", " in ", " we ", " recommend", "should", "increase", "decrease")
    en_score = sum(1 for c in en_cues if c in t)
    if has_pl:
        return False
    return en_score >= 2

def _split_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    return [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", t) if s.strip()]

def _gate_et_v1(text: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Returns (ok, hard_reason, diagnostics) where diagnostics may contain warnings."""
    diag: Dict[str, Any] = {"warnings": []}
    t = (text or "").strip()
    if not t:
        return False, "empty", diag

    if _looks_non_polish(t):
        return False, "language_not_pl", diag

    sents = _split_sentences(t)
    diag["sentences"] = len(sents)
    if len(sents) != 2:
        return False, f"sentences={len(sents)}", diag

    s1, s2 = sents[0], sents[1]

    nums1 = _NUM_RE.findall(s1)
    diag["numbers_s1"] = len(nums1)
    if len(nums1) < 2:
        return False, "numbers_s1<2", diag

    s2l = s2.lower().strip()

    # HARD: unikamy deklaracji zamiast akcji ("Ustal, że ...")
    # (akcje typu "Ustal priorytety / Ustal progi / Ustal węższe przedziały" są OK)
    if re.match(r"^ustal\s*,?\s*(że|ze|iż|iz)\b", s2l):
        return False, "sentence2_not_action", diag

    if not any(v in s2l for v in _DECISION_VERBS_PL):
        return False, "no_decision_sentence2", diag

    # SOFT warnings (nie mogą zrzucać do fallback!)
    if ("..." in t) or ("…" in t) or t.rstrip().endswith(".."):
        diag["warnings"].append("ellipsis")
    if not re.search(r"[\.\!\?]$", t):
        diag["warnings"].append("no_terminal_punct")
    if len(t) > 320:
        diag["warnings"].append("too_long>320")

    return True, "ok", diag

def _fallback_takeaway(block: Dict[str, Any]) -> str:
    # Priority: explicit fallback, then injected exec, then empty
    for k in ("exec_fallback", "exec", "takeaway"):
        v = block.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _call_llm_text(session_state: Optional[dict], prompt: str) -> Optional[str]:
    """
    Contract:
    - używa session_state["dc_llm_text"] (callable) jeśli dostępny
    - ZAWSZE aktualizuje dc_llm_status_v1 (nie może zostać {})
    - przy braku callable: zapisuje dc_errors(where=exec_takeaway.llm, error=llm_fn_missing) + extra
    - przy pustym output: zapisuje dc_errors(where=exec_takeaway.llm, error=llm_empty_output)
    """
    if session_state is None:
        return None

    # status MUST exist and be dict
    status = session_state.get(STATUS_KEY)
    if not isinstance(status, dict):
        status = {}
    status.setdefault("calls", 0)
    status.setdefault("last_ok", False)
    status.setdefault("last_error", None)
    session_state[STATUS_KEY] = status

    fn = session_state.get("dc_llm_text")

    # 1) missing callable
    if not callable(fn):
        status["calls"] = int(status.get("calls", 0) or 0)
        status["last_ok"] = False
        status["last_error"] = "llm_fn_missing"
        session_state[STATUS_KEY] = status

        # extra (as requested): co jest w state i czy klucz istnieje
        keys_preview = sorted([k for k in session_state.keys()])[:80]
        _append_error(
            session_state,
            "exec_takeaway.llm",
            RuntimeError("llm_fn_missing"),
            extra={
                "dc_llm_text_present": "dc_llm_text" in session_state,
                "dc_llm_text_callable": False,
                "session_state_keys_preview": keys_preview,
            },
        )
        return None

    # 2) callable exists
    try:
        status["calls"] = int(status.get("calls", 0) or 0) + 1
        session_state[STATUS_KEY] = status

        out = str(fn(prompt) or "")
        out = out.strip()

        if not out:
            status["last_ok"] = False
            status["last_error"] = "llm_empty_output"
            session_state[STATUS_KEY] = status

            _append_error(
                session_state,
                "exec_takeaway.llm",
                RuntimeError("llm_empty_output"),
                extra={"prompt_preview": prompt[:160]},
            )
            return None

        status["last_ok"] = True
        status["last_error"] = None
        session_state[STATUS_KEY] = status
        return out

    except Exception as e:
        status["last_ok"] = False
        status["last_error"] = f"exception:{type(e).__name__}"
        session_state[STATUS_KEY] = status

        _append_error(session_state, "exec_takeaway.llm", e, extra={"prompt_preview": prompt[:160]})
        return None

# -------------------------------------------------------------------------
# Compatibility wrapper (v2 callers)
# Some branches still call `get_exec_takeaway_v2(...)` and may pass extra
# keyword args like `chart_context` / `injected`. We accept and ignore them
# to avoid regressions while keeping a single implementation.
def get_exec_takeaway_v2(*args, **kwargs):
    """Backward/forward compatible wrapper for Executive Takeaway.

    Accepts extra kwargs used by older patches (e.g. `chart_context`, `injected`)
    and avoids 'multiple values for spec_id' when both positional and keyword are passed.
    """
    # drop legacy/experimental kwargs
    kwargs.pop("chart_context", None)
    kwargs.pop("injected", None)
    # avoid: got multiple values for argument 'spec_id'
    if len(args) > 0 and "spec_id" in kwargs:
        kwargs.pop("spec_id", None)
    return get_exec_takeaway(*args, **kwargs)

def get_exec_takeaway(
    *,
    intent: str,
    block: Dict[str, Any],
    stats: Dict[str, Any],
    question: Optional[str] = None,
    session_state: Optional[dict] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    force_refresh: bool = False,

    **_ignored
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (takeaway_text, meta).
    Minimal patch keeps signature stable for branches.

    Sources:
      - "cache"    : from v3 cache
      - "llm"      : generated by LLM via llm_fn or session_state["dc_llm_text"]
      - "fallback" : explicit fallback text in block
      - "heuristic": fallback default when nothing else
    """
    # Ensure cache + meta structures
    cache = ensure_exec_cache(session_state)
    meta_list = _ensure_list(session_state, META_KEY)

    q = (question or "").strip()
    block_id = str(block.get("key") or block.get("id") or block.get("label") or block.get("title") or "block").strip()
    stats_hash = _hash_obj(stats or {})
    cache_key = _build_cache_key_v4(intent=intent, block_id=block_id, stats_hash=stats_hash, question=q)

    # One-shot global override (set from debug panel) — minimal / reversible.
    if (not force_refresh) and isinstance(session_state, dict) and session_state.get("exec_takeaway_force_refresh_v1"):
        force_refresh = True
        try:
            session_state.pop("exec_takeaway_force_refresh_v1", None)
        except Exception:
            pass

    # Determine injected exec (if branch already computed per-block)
    injected_exec = ""
    if isinstance(block.get("exec"), str) and block.get("exec").strip():
        # if branch marks as LLM injected, treat as preferred
        if str(block.get("_exec_source") or "").strip().lower() == "llm":
            injected_exec = block.get("exec", "").strip()

    # ─────────────────────────────────────────────
    # One-time cache migration: normalize legacy cache entries
    # (removes JSON wrappers, ellipsis, ensures terminal punct)
    # ─────────────────────────────────────────────
    try:
        cache_migrated = bool(session_state.get("exec_takeaway_cache_migrated_v1", False))
        if isinstance(cache, dict) and not cache_migrated:
            new_cache = {}
            changed = 0
            for k, v in cache.items():
                vv = _normalize_llm_text_out(str(v or ""))
                if vv != (v or ""):
                    changed += 1
                new_cache[k] = vv
            session_state[CACHE_KEY_V3] = new_cache
            session_state["exec_takeaway_cache_migrated_v1"] = True
            st_status = session_state.get("dc_llm_status_v1")
            if isinstance(st_status, dict):
                st_status["cache_migrated_v1"] = changed
                session_state["dc_llm_status_v1"] = st_status
            cache = new_cache
    except Exception:
        pass

    # 1) cache (but validate: old cache entries may be bad, e.g. truncated with "...")
    if (not force_refresh) and cache_key in cache and isinstance(cache.get(cache_key), str) and cache.get(cache_key).strip():
        cached = cache[cache_key].strip()
        # ET v1.0: normalize cached output (removes JSON wrappers, ellipsis, adds terminal punct)
        cached_norm = _normalize_llm_text_out(cached)

        ok, why, diag = _gate_et_v1(cached_norm)
        if ok:
            txt = _final_sanitize_et(cached_norm)
            source = "cache"
            text_hash = _sha1_short(_norm_text(txt))
            meta = {
                "cache_key": cache_key,
                "intent": intent,
                "block_id": block_id,
                "source": source,
                "stats_hash": stats_hash,
                "text_hash": text_hash,
                "preview": txt[:80],
                # ET v1.0 observability (cache hit)
                "attempts_used": 0,
                "repair_used": False,
                "coercion_used": False,
                "warnings": list(diag.get("warnings") or []),
                "gate_reason": "ok",
            }
            txt = _final_sanitize_et(txt)
            meta_list.append(meta)
            return txt, meta

        # invalidate and continue (LLM / fallback)
        _append_error(
            session_state,
            "exec_takeaway.cache_gate",
            Exception(why),
            extra={
                "block_id": block_id,
                "cache_key": cache_key,
                "stats_hash": stats_hash,
                "warnings": list(diag.get("warnings") or []),
            },
        )
        try:
            del cache[cache_key]
        except Exception:
            pass

    # 2) injected exec (already LLM) – prefer to avoid extra calls
    if injected_exec:
        txt = injected_exec
        source = "llm"
        ok, why = _quality_gate_hard(
            txt,
            require_numbers=False,  # injected might not have numbers yet
            require_exact_two_sentences=False,
            require_decision_in_sentence2=False,
        )
        if not ok:
            _append_error(session_state, "exec_takeaway.injected_gate", Exception(why), extra={"block_id": block_id})
        txt = _normalize_llm_text_out(txt)
        txt = _apply_sentence2_threshold_sanity(txt, stats or {})
        txt = _final_sanitize_et(txt)
        cache[cache_key] = txt
        text_hash = _sha1_short(_norm_text(txt))
        meta = {
            "cache_key": cache_key,
            "intent": intent,
            "block_id": block_id,
            "source": source,
            "stats_hash": stats_hash,
            "text_hash": text_hash,
            "preview": txt[:80],
        }
        meta["preview"] = txt[:80]
        meta_list.append(meta)
        return txt, meta

    # 3) LLM call (Option B) — ET v1.0 (McKinsey-grade, realistic).
    # Goal: 99% LLM. Fallback ONLY when LLM is unavailable (missing/exception).
    txt = ""
    source = "llm"
    llm_available = True

    # observability for meta
    attempts_used = 0
    repair_used = False
    coercion_used = False
    warnings: list[str] = []
    final_gate_reason = ""

    try:
        base_prompt = _build_prompt(intent=intent, block=block, stats=stats, question=q)
        last_raw = ""
        last_why = ""
        last_diag: Dict[str, Any] = {"warnings": []}

        for attempt in range(3):
            attempts_used = attempt + 1
            if attempt == 0:
                prompt = base_prompt
            elif attempt == 1:
                repair_used = True
                prompt = _build_repair_prompt(
                    base_prompt=base_prompt,
                    why=last_why,
                    raw_output=last_raw,
                )
            else:
                repair_used = True
                prompt = _build_repair_prompt_hard_pl(
                    base_prompt=base_prompt,
                    why=last_why,
                    raw_output=last_raw,
                    stats=stats,
                )

            if llm_fn is not None:
                last_raw = str(llm_fn(prompt) or "").strip()
            else:
                last_raw = str(_call_llm_text(session_state, prompt) or "").strip()

            candidate = _normalize_llm_text_out(last_raw)

            ok, why, diag = _gate_et_v1(candidate)
            last_diag = diag
            final_gate_reason = why

            if ok:
                txt = candidate
                warnings = list(diag.get("warnings") or [])
                break

            last_why = why

        # If still not ok after 3 attempts, keep LLM output but coerce formatting minimally (still LLM-based).
        if not txt and last_raw:
            coercion_used = True
            coerced = _coerce_two_sentences_no_ellipsis(last_raw)
            coerced = _hard_trim_no_ellipsis(coerced, 320)
            candidate = _normalize_llm_text_out(coerced.strip() if coerced.strip() else last_raw.strip())

            ok, why, diag = _gate_et_v1(candidate)
            final_gate_reason = why
            warnings = list(diag.get("warnings") or [])
            txt = candidate

            # HARD error only if it STILL fails minimal v1.0 after coercion
            if not ok:
                _append_error(
                    session_state,
                    "exec_takeaway.llm_gate_final",
                    Exception(why),
                    extra={
                        "block_id": block_id,
                        "intent": intent,
                        "attempts_used": attempts_used,
                        "repair_used": repair_used,
                        "coercion_used": coercion_used,
                        "warnings": warnings,
                    },
                )

        # If we succeeded but had SOFT warnings, keep them in meta (no dc_errors noise).
        if txt and warnings:
            pass

    except Exception as e:
        llm_available = False
        _append_error(session_state, "exec_takeaway.llm", e, extra={"block_id": block_id, "intent": intent})
        txt = ""

    # 4) fallback / heuristic — ONLY if LLM is not available.
    if not txt and not llm_available:
        txt = _fallback_takeaway(block).strip()
        source = "fallback" if txt else "heuristic"
        if not txt:
            txt = "Brak wystarczających podstaw w danych, aby sformułować wiarygodny takeaway."
            source = "heuristic"

    # If LLM is available but output is empty, keep it explicit (no silent heuristic).
    if not txt and llm_available:
        txt = "[LLM_EMPTY] Model zwrócił pustą odpowiedź — sprawdź prompt/limity i spróbuj ponownie."
        source = "llm"

    # persist
    txt = _normalize_llm_text_out(txt)
    txt = _apply_sentence2_threshold_sanity(txt, stats or {})
    cache[cache_key] = txt
    text_hash = _sha1_short(_norm_text(txt))
    meta = {
        "cache_key": cache_key,
        "intent": intent,
        "block_id": block_id,
        "source": source,
        "stats_hash": stats_hash,
        "text_hash": text_hash,
        "preview": txt[:80],
        # ET v1.0 observability (only meaningful for source == "llm")
        "attempts_used": attempts_used if source == "llm" else 0,
        "repair_used": repair_used if source == "llm" else False,
        "coercion_used": coercion_used if source == "llm" else False,
        "warnings": warnings if source == "llm" else [],
        "gate_reason": final_gate_reason if source == "llm" else "",
    }
    txt = _final_sanitize_et(txt)
    meta_list.append(meta)
    return txt, meta

def _build_prompt(*, intent: str, block: Dict[str, Any], stats: Dict[str, Any], question: str) -> str:
    """
    Hard contract prompt: 1–2 sentences, sentence1 fact+numbers, sentence2 decision.
    Strictly stats-driven.
    """
    # Keep prompt compact and deterministic
    block_label = str(block.get("label") or block.get("title") or block.get("key") or "").strip()
    stats_str = _safe_json_dumps(stats or {})
    q = (question or "").strip()

    # NOTE: We request strict JSON to prevent truncation/ellipsis and make validation deterministic.
    # The caller will parse it and join to 2 sentences.
    return (
        "ROLA: Lead Analytics Engineer (McKinsey/Bain).\n"
        "ZADANIE: Napisz Executive Takeaway dla JEDNEGO bloku wykresu.\n"
        "TWARDY KONTRAKT (MUST):\n"
        "1) Język polski.\n"
        "2) Dokładnie 2 zdania: s1 (fakt + liczby), s2 (implikacja + decyzja).\n"
        "3) W s1 użyj MIN. 2 kotwic liczbowych z STATS_JSON (np. %/PLN/median/IQR/top).\n"
        "4) W s2 musi paść jednoznaczna decyzja (np. 'Ustal...', 'Skup się...', 'Zmień...', 'Zweryfikuj...').\n"
        "5) Używaj TYLKO danych ze STATS_JSON. Bez zgadywania.\n"
        "6) Bez ogólników. Bez marketingu.\n"
        "7) Unikaj wielokropków (...) i urwanych fraz (jeśli musisz — tylko jako ostateczność).\n"
        "8) Łącznie (s1+s2) <= 320 znaków. Każde zdanie zakończ kropką.\n\n"
        f"INTENT: {intent}\n"
        f"BLOCK: {block_label}\n"
        f"QUESTION: {q}\n"
        f"STATS_JSON: {stats_str}\n\n"
        "ZWRÓĆ TYLKO 2 ZDANIA (bez markdown, bez numeracji, bez JSON).\n"
    )


def _parse_llm_json_takeaway(raw: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Parse JSON {s1, s2} returned by the model.

    Returns:
      combined_text: "<s1> <s2>" or None if parsing fails
      meta: parsing diagnostics
    """
    meta: Dict[str, Any] = {"parsed_json": False}
    txt = (raw or "").strip()
    if not txt:
        meta["parse_error"] = "empty"
        return None, meta

    # Sometimes models wrap JSON with extra text — try to extract the first JSON object.
    m = re.search(r"\{[\s\S]*\}", txt)
    candidate = m.group(0) if m else txt

    try:
        obj = json.loads(candidate)
    except Exception as e:
        meta["parse_error"] = f"json_loads_failed: {e}"
        return None, meta

    if not isinstance(obj, dict):
        meta["parse_error"] = "not_a_dict"
        return None, meta

    s1 = (obj.get("s1") or "").strip()
    s2 = (obj.get("s2") or "").strip()
    if not s1 or not s2:
        meta["parse_error"] = "missing_s1_s2"
        return None, meta

    meta["parsed_json"] = True
    meta["s1_len"] = len(s1)
    meta["s2_len"] = len(s2)
    combined = f"{s1} {s2}".strip()
    return combined, meta


def _coerce_two_sentences_no_ellipsis(text: str) -> str:
    """Coerce to exactly 2 sentences, remove ellipsis, keep it short.

    Used ONLY as a last-resort formatting fix when LLM output is close-but-not-valid.
    """
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text.replace("…", ".").strip())
    # Remove literal ellipsis
    t = t.replace("...", ".")
    # Split by sentence end. Keep punctuation.
    parts = re.split(r"(?<=[\.!\?])\s+", t)
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        return ""

    s1 = parts[0]
    rest = " ".join(parts[1:]).strip()

    if not rest:
        # Create a minimal decision sentence (generic but still compatible with PL contract)
        s2 = "Rekomendacja: zweryfikuj progi/KPI i wykonaj szybki drill-down po segmentach."
    else:
        # If rest contains multiple sentences, keep only the first one as sentence2.
        s2 = re.split(r"(?<=[\.!\?])\s+", rest)[0].strip()
        if not re.search(r"[\.!\?]$", s2):
            s2 += "."

    # Ensure sentence1 ends with punctuation.
    if not re.search(r"[\.!\?]$", s1):
        s1 += "."

    out = f"{s1} {s2}"
    # Final: no ellipsis at end
    out = re.sub(r"(\.|\?|\!)\s*\.+\s*$", r"\1", out).strip()
    out = out.replace("...", ".").replace("…", ".")
    return out

def _hard_trim_no_ellipsis(text: str, max_chars: int) -> str:
    """Trim text to max_chars without adding '...'. Keep a proper sentence ending."""
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars].rstrip()
    # Prefer trimming at a whitespace boundary.
    if " " in cut[-20:]:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    cut = cut.rstrip(",;:")
    if not re.search(r"[.!?]$", cut):
        cut = cut + "."

    cut = cut.replace("…", ".").replace("...", ".")
    cut = re.sub(r"\(\s*\.\s*\.\s*\.\s*\)", ".", cut)
    cut = re.sub(r"\.{2,}", ".", cut).strip()
    if cut and not re.search(r"[.!?]$", cut):
        cut += "."
    return cut

def _normalize_llm_text_out(raw: str) -> str:
    """
    Normalize LLM output to plain 2 sentences text:
    - unwrap JSON like {"s1":"...","s2":"..."} or {"s1":"..."}
    - remove ellipsis (...) / ... / …
    - ensure terminal punctuation
    """
    t = (raw or "").strip()
    if not t:
        return ""

    # 1) Unwrap JSON if present
    if t.startswith("{") and ("\"s1\"" in t or "'s1'" in t):
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                s1 = str(obj.get("s1") or "").strip()
                s2 = str(obj.get("s2") or "").strip()
                if s1 and s2:
                    t = f"{s1} {s2}".strip()
                elif s1:
                    t = s1
        except Exception:
            # keep original t if json parsing fails
            pass

    # also handle the case where JSON is wrapped in quotes (your cache shows "\"{...}\"")
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        tt = t[1:-1].strip()
        if tt.startswith("{") and ("\"s1\"" in tt or "'s1'" in tt):
            try:
                obj = json.loads(tt)
                if isinstance(obj, dict):
                    s1 = str(obj.get("s1") or "").strip()
                    s2 = str(obj.get("s2") or "").strip()
                    if s1 and s2:
                        t = f"{s1} {s2}".strip()
                    elif s1:
                        t = s1
            except Exception:
                pass

    # 2) Remove ellipsis variants
    t = t.replace("…", ".")
    t = t.replace("...", ".")
    t = re.sub(r"\(\s*\.\s*\.\s*\.\s*\)", ".", t)
    t = re.sub(r"\.{2,}", ".", t).strip()
    if t and not re.search(r"[.!?]$", t):
        t += "."

    # collapse repeated dots/spaces
    t = re.sub(r"\.{2,}", ".", t)
    t = re.sub(r"\s{2,}", " ", t).strip()

    # 3) Ensure terminal punctuation
    if t and not re.search(r"[.!?]$", t):
        t = t + "."

    return t

def _final_sanitize_et(txt: str) -> str:
    """
    Final invariant for Executive Takeaway:
    - no JSON wrappers
    - no ellipsis (… / ... / (...))
    - always terminal punctuation
    """
    t = _normalize_llm_text_out(txt)

    # hard stop for typographic ellipsis
    t = t.replace("…", ".")

    # remove leftover "(...)" patterns
    t = re.sub(r"\(\s*\.\s*\.\s*\.\s*\)", ".", t)

    # collapse multiple dots
    t = re.sub(r"\.{2,}", ".", t)

    # fix trailing comma + dot cases: ",."
    t = re.sub(r",\s*\.$", ".", t)

    t = t.strip()
    if t and not re.search(r"[.!?]$", t):
        t += "."

    return t

def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(" ", "").replace(",", ".")
        return float(s)
    except Exception:
        return None


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "?"
    # prosta estetyka: bez .0 jeśli całkowite
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(round(x, 2)).replace(".", ",")


def _apply_sentence2_threshold_sanity(et: str, stats: Dict[str, Any]) -> str:
    """If sentence2 contains contradictory thresholds vs stats[min,max], rewrite sentence2 to a safe CTA."""
    sents = _split_sentences(et or "")
    if len(sents) != 2:
        return et

    s1, s2 = sents[0].strip(), sents[1].strip()
    s2l = s2.lower()

    # --- PATCH2A: akcja zamiast "Ustal, że" (minimalnie, bez rozdmuchiwania) ---
    # Jeśli LLM zaczyna zdanie 2 od "Ustal, że ...", zamień to na CTA.
    # Dzięki temu:
    # - sentence2 przechodzi gate (decyzja = działanie),
    # - nie zmieniasz reszty treści bardziej niż trzeba.
    if re.match(r"^\s*ustal,\s*że\b", s2l):
        # Dobór CTA: jeśli już pada "powyżej/poniżej/od-do" zostaw logikę progu,
        # tylko zmień start; w przeciwnym razie ustaw bezpieczne CTA ogólne.
        s2 = re.sub(r"^\s*Ustal,\s*że\b", "Skup się na tym, że", s2, flags=re.IGNORECASE)
        s2l = s2.lower()

    mn = _as_float(stats.get("min"))
    mx = _as_float(stats.get("max"))
    q1 = _as_float(stats.get("q1"))
    q3 = _as_float(stats.get("q3"))
    p95 = _as_float(stats.get("p95"))
    med = _as_float(stats.get("median"))

    if mn is None or mx is None:
        return et

    eps = max(1e-9, 0.001 * (mx - mn if mx > mn else 1.0))

    def safe_upper_tail() -> str:
        # prefer p95 if available, else q3
        if p95 is not None:
            return f"Skup się na górnym ogonie (p95={_fmt_num(p95)}) i sprawdź czynniki stojące za wartościami blisko maksimum {_fmt_num(mx)}."
        if q3 is not None:
            return f"Skup się na wartościach powyżej Q3={_fmt_num(q3)} i sprawdź czynniki stojące za wartościami blisko maksimum {_fmt_num(mx)}."
        return f"Skup się na najwyższych obserwacjach i sprawdź czynniki stojące za wartościami blisko maksimum {_fmt_num(mx)}."

    def safe_lower_tail() -> str:
        if q1 is not None:
            return f"Skup się na dolnym ogonie (Q1={_fmt_num(q1)}) i zdiagnozuj przyczyny wartości blisko minimum {_fmt_num(mn)}."
        return f"Skup się na najniższych obserwacjach i zdiagnozuj przyczyny wartości blisko minimum {_fmt_num(mn)}."

    def safe_main_range() -> str:
        if q1 is not None and q3 is not None:
            tail = f" oraz osobno ogony (p95={_fmt_num(p95)})" if p95 is not None else ""
            return f"Skup się na głównym zakresie (Q1–Q3: {_fmt_num(q1)}–{_fmt_num(q3)}){tail}, aby ustalić priorytety dalszej analizy."
        if med is not None:
            return f"Skup się na odchyleniach względem mediany {_fmt_num(med)} i sprawdź, co napędza skrajne wartości."
        return "Skup się na głównym zakresie danych i osobno przeanalizuj skrajne obserwacje."

    # pattern: "powyżej X"
    m = re.search(r"\bpowyżej\s+(\d+(?:[\,\.]\d+)?)", s2l)
    if m:
        thr = _as_float(m.group(1))
        if thr is not None and (thr >= mx - eps):
            s2 = safe_upper_tail()

    # pattern: "poniżej X"
    m = re.search(r"\bponiżej\s+(\d+(?:[\,\.]\d+)?)", s2l)
    if m:
        thr = _as_float(m.group(1))
        if thr is not None and (thr <= mn + eps):
            s2 = safe_lower_tail()

    # --- PATCH2B: sanity-check dla "górnej/dolnej granicy (X)" ---
    # np. "w pobliżu górnej granicy (2249,75)" przy max=1453
    m = re.search(r"(g[óo]rn\w*\s+granic\w*)\s*\(\s*(\d+(?:[\.,]\d+)?)\s*\)", s2l)
    if m:
        thr = _as_float(m.group(2))
        if thr is not None and thr > mx + eps:
            s2 = safe_upper_tail()

    m = re.search(r"(doln\w*\s+granic\w*)\s*\(\s*(\d+(?:[\.,]\d+)?)\s*\)", s2l)
    if m:
        thr = _as_float(m.group(2))
        if thr is not None and thr < mn - eps:
            s2 = safe_lower_tail()

    # pattern: "od A do B" / "między A a B"
    m = re.search(r"\bod\s+(\d+(?:[\,\.]\d+)?)\s+do\s+(\d+(?:[\,\.]\d+)?)", s2l)
    if not m:
        m = re.search(r"\bmi[eę]dzy\s+(\d+(?:[\,\.]\d+)?)\s+a\s+(\d+(?:[\,\.]\d+)?)", s2l)

    if m:
        a = _as_float(m.group(1))
        b = _as_float(m.group(2))
        if a is None or b is None or a >= b or a < mn - eps or b > mx + eps:
            s2 = safe_main_range()

    out = f"{s1} {s2}".strip()
    return _final_sanitize_et(out)

def _build_repair_prompt(
    *,
    base_prompt: str,
    why: str,
    raw_output: Optional[str] = None,
    last_output: Optional[str] = None,
    **_kw,
) -> str:
    """Build a repair prompt that is backward-compatible (raw_output vs last_output)."""
    prev = (raw_output or last_output or "").strip()
    return (
        base_prompt
        + "\n\nNIE SPEŁNIASZ KONTRAKTU. POWTÓRZ OD ZERA I POPRAW.\n"
        + f"BŁĄD: {why}\n"
        + "Twoja poprzednia odpowiedź (do poprawy):\n"
        + prev
        + "\n\nZWRÓĆ TYLKO 2 ZDANIA (bez markdown, bez numeracji, bez JSON)."
    )

def _extract_numeric_anchors(stats: Dict[str, Any], limit: int = 8) -> list[str]:
    """Extract a few numeric anchors from stats for hard repair prompts."""
    try:
        s = _safe_json_dumps(stats or {})
    except Exception:
        s = str(stats or "")
    nums = _NUM_RE.findall(s)
    seen = set()
    out = []
    for n in nums:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= limit:
            break
    return out

def _build_repair_prompt_hard_pl(
    *,
    base_prompt: str,
    why: str,
    raw_output: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Hard repair: force 2 sentences, PL-only, and force using numeric anchors."""
    prev = (raw_output or "").strip()
    anchors = _extract_numeric_anchors(stats or {}, limit=8)
    anchors_txt = ", ".join(anchors[:8]) if anchors else "(brak wyekstrahowanych liczb — użyj liczb ze STATS_JSON)"
    template = (
        "SZABLON (OBOWIĄZKOWY):\n"
        "Zdanie 1: <fakt + MIN.2 liczby z listy>.\n"
        "Zdanie 2: <AKCJA (czasownik + obiekt + kryterium) + liczba/prog>.\n"
        "DOZWOLONE starty zdania 2 (przykłady): 'Skup się...', 'Zalecam...', 'Rekomenduję...', 'Ustal priorytety...', 'Ustal progi...'.\n"
        "ZAKAZ: nie zaczynaj od 'Ustal, że/ze/iż...' (to jest deklaracja, nie akcja).\n"
    )
    return (
        base_prompt
        + "\n\nNIE SPEŁNIASZ KONTRAKTU ET v1.0 — POPRAW OD ZERA.\n"
        + f"BŁĄD (HARD): {why}\n"
        + "WYMUSZONE KOTWICE LICZBOWE (użyj MIN.2 w zdaniu 1): "
        + anchors_txt
        + "\n"
        + template
        + "Twoja poprzednia odpowiedź (do poprawy):\n"
        + prev
        + "\n\nZWRÓĆ TYLKO 2 ZDANIA (bez markdown, bez numeracji, bez JSON)."
    )
