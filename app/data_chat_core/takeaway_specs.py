# app/data_chat_core/takeaway_specs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import re

_NUM_RE = re.compile(r"\d+(?:[\,\.]\d+)?")

@dataclass(frozen=True)
class TakeawaySpec:
    """Specyfikacja Executive Takeaway per wykres/blok."""
    spec_id: str
    required_fields: Tuple[str, ...]
    # deterministic fallback (MUST be chart-specific)
    deterministic_fn: Callable[[Dict[str, Any]], str]
    # prompt builder for LLM
    prompt_fn: Callable[[Dict[str, Any], Dict[str, Any]], str]
    # validator (hard gate) for a final text
    validate_fn: Callable[[str, Dict[str, Any]], Tuple[bool, str, Dict[str, Any]]]

def _split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    return [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", t) if s.strip()]

_DECISION_VERBS_PL = (
    "rekomend", "zalec", "wdroż", "wdrożyć", "zwiększ", "zwieksz", "zmniejsz", "ogranicz",
    "skup", "priorytet", "monitor", "zweryfik", "ustaw", "zmień", "zmien", "podnieś", "obniż",
    "standaryz", "segment", "przetest", "wyklucz",
)

def _has_decision_verb(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(v in s for v in _DECISION_VERBS_PL)

def _count_numbers(text: str) -> int:
    return len(_NUM_RE.findall(text or ""))

def _fmt_num(x: Any, *, digits: int = 1) -> str:
    try:
        if x is None:
            return "—"
        if isinstance(x, bool):
            return "—"
        fx = float(x)
        if abs(fx) >= 1000 and float(int(fx)) == fx:
            return f"{int(fx):,}".replace(",", " ")
        if float(int(fx)) == fx:
            return str(int(fx))
        return f"{fx:.{digits}f}".replace(".", ",")
    except Exception:
        return "—"


def _digits_only(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

def _has_number_token_in_text(val: Any, text: str, *, digits: int = 1) -> bool:
    """Robust numeric token match:
    - supports thousand separators/spaces in text
    - supports decimal comma/dot
    """
    t = (text or "").lower()
    t_digits = _digits_only(t)
    try:
        fv = float(val)
    except Exception:
        return False
    # integer-like token
    token_i = str(int(round(fv)))
    if _digits_only(token_i) and _digits_only(token_i) in t_digits:
        return True
    # decimal token (one decimal)
    token_d = f"{fv:.{digits}f}".replace(".", ",")
    if _digits_only(token_d) and _digits_only(token_d) in t_digits:
        return True
    # raw repr fallback
    raw = str(val)
    if raw and _digits_only(raw) and _digits_only(raw) in t_digits:
        return True
    return False

def _validate_two_sentences_pl(text: str) -> Tuple[bool, str, Dict[str, Any]]:
    diag: Dict[str, Any] = {"warnings": []}
    t = (text or "").strip()
    if not t:
        return False, "empty", diag
    sents = _split_sentences(t)
    diag["sentences_n"] = len(sents)
    if len(sents) != 2:
        return False, f"sentences={len(sents)}", diag
    if _count_numbers(sents[0]) < 2:
        return False, "s1_numbers<2", diag
    if not _has_decision_verb(sents[1]):
        return False, "no_decision_s2", diag
    if "..." in t or "…" in t:
        return False, "ellipsis", diag
    if not re.search(r"[\.\!\?]$", t):
        return False, "no_terminal_punct", diag
    return True, "ok", diag

# ─────────────────────────────────────────────────────────────
# SPEC: cs_pareto
# Required anchors (per prompt): cutoff, n_80, share_top1/top_segments, tail_share, metric_name, total_value
# ─────────────────────────────────────────────────────────────

def _det_pareto(a: Dict[str, Any]) -> str:
    metric = str(a.get("metric_name") or a.get("metric") or "wartość")
    total = a.get("total_value")
    top1 = a.get("share_top1_pct")
    n80 = a.get("n_80")
    cutoff = a.get("cutoff") or 80
    tail = a.get("tail_share_pct")

    # Sentence 1: cutoff + topN + tail (+ optional top1)
    parts = []
    if top1 is not None:
        parts.append(f"Największy segment ma {_fmt_num(top1)}% udziału")
    if n80 is not None:
        parts.append(f"a Top {int(n80)} segmentów generuje {int(cutoff)}% {metric}")
    if tail is not None:
        parts.append(f"podczas gdy ogon odpowiada za {_fmt_num(tail)}%")
    if total is not None:
        parts.append(f"({_fmt_num(total)} łącznie)")
    s1 = ", ".join(parts).rstrip(".") + "."

    # Sentence 2: dual strategy
    # Top-N: availability/price/promo focus; Tail: rationalize + targeted experiments
    s2 = (
        f"Zabezpiecz Top {int(n80)} (dostępność/miks/egzekucja cenowo-promocyjna), "
        f"a dla ogona wdroż playbook: konsolidacja słabych segmentów + szybkie testy 2–3 hipotez, "
        f"żeby ograniczyć złożoność bez utraty {_fmt_num(tail)}% {metric}."
    )

    return f"{s1} {s2}"

def _prompt_pareto(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    cutoff = float(a.get("cutoff", 0.80) or 0.80)
    cutoff_pct = int(round(cutoff * 100))
    metric = str(a.get("metric_name") or a.get("metric") or "wartość")
    top_segments = a.get("top_segments") or []
    # Provide both numeric anchors + segment names to force specificity
    anchors = {
        "cutoff_pct": cutoff_pct,
        "n_80": a.get("n_80"),
        "share_top1_pct": a.get("share_top1_pct"),
        "tail_share_pct": a.get("tail_share_pct"),
        "metric_name": metric,
        "total_value": a.get("total_value"),
        "top_segments": top_segments[:5],
    }
    return f"""
Jesteś analitykiem (McKinsey/Bain). Napisz Executive Takeaway DOKŁADNIE w 2 zdaniach po polsku dla wykresu Pareto (koncentracja wartości).

Wymagania twarde (MUST):
- Zdanie 1: fakt + MIN 3 liczby i MUSI zawierać:
  (1) próg Pareto: {cutoff_pct}%,
  (2) Top-N do progu (n_80),
  (3) udział ogona (tail_share_pct).
  Dodatkowo możesz podać udział największego segmentu (share_top1_pct) i/lub łączną wartość (total_value), jeśli są w ANCHORS.
- Zdanie 2: implikacja + JEDNOZNACZNA decyzja (czasownik decyzyjny) w dwóch torach:
  (A) co robimy z Top-N (np. zabezpieczyć dostępność/miks/egzekucję cenowo-promocyjną),
  (B) co robimy z ogonem (np. konsolidacja/standaryzacja + szybkie testy/segmentacja).
  Musi paść słowo/zwrot wskazujący ogon: "ogon" lub "pozostałe segmenty".

Zakazy (NIE WOLNO):
- "ten wykres pokazuje", meta-komentarze, wielokropki, trzecie zdanie/linia.
- decyzja typu "inwestuj w największy segment" jako jedyna (Top-1-only).
- ogólniki typu "warto przeanalizować" / "zaleca się analizę".

Dane:
- Użyj WYŁĄCZNIE danych z ANCHORS (nie zgaduj).
- Użyj co najmniej 1 nazwy segmentu z top_segments, jeśli dostępne.

ANCHORS (Pareto / koncentracja):
{anchors}

Zwróć JSON w formacie:
{{
  "candidates": [
    {{"text": "<2-zdaniowy takeaway>"}},
    {{"text": "<2-zdaniowy takeaway>"}},
    {{"text": "<2-zdaniowy takeaway>"}}
  ]
}}
""".strip()

import re

def _validate_pareto(text: str, a: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    ok, why, diag = _validate_two_sentences_pl(text)
    if not ok:
        return ok, why, diag

    t_raw = (text or "").strip()
    t = t_raw.lower()

    # --- HARD GATE: no contradictions when anchors are present ---
    # If we do have Pareto anchors, the text must not claim "brak/nieokreślone/0%".
    has_any_anchor = any(a.get(k) is not None for k in ("cutoff", "n_80", "tail_share_pct"))
    if has_any_anchor:
        CONTRA_PATTERNS = [
            r"\bbrak\b",
            r"\bnie (zosta[łl]a|jest) (okre[śs]lon[ay]|ustalon[ay])\b",
            r"\bnieznan[ay]\b",
            r"\bbrakuje\b",
            r"\b0%\b",
            r"\bzerow[ay]\b",
            r"\bnie ma\b",
        ]
        if any(re.search(p, t) for p in CONTRA_PATTERNS):
            # allow missing-data statement only if explicitly about NaN / braki danych (rare for Pareto)
            if not re.search(r"\bnan\b|\bbraki danych\b|\bbrak danych\b", t):
                return False, "contradiction_with_anchors", diag

    # --- HARD GATE: ban generic recommendations ---
    BANNED_GENERIC = [
        "zaleca się analizę",
        "warto przeanalizować",
        "należy przeanalizować",
        "konieczne jest podjęcie działań w celu zwiększenia dokładności danych",
    ]
    if any(bg in t for bg in BANNED_GENERIC):
        return False, "generic_decision", diag

    cutoff = a.get("cutoff")          # e.g. 80
    n80 = a.get("n_80")               # e.g. 13
    top1 = a.get("share_top1_pct")    # e.g. 12.42
    tail = a.get("tail_share_pct")    # e.g. 20.0
    metric = (a.get("metric_name") or "").lower()

    # Helpers
    def _has_number_token(val: Any) -> bool:
        try:
            fv = float(val)
        except Exception:
            return False
        token_i = str(int(round(fv)))
        token_d = f"{fv:.1f}".replace(".", ",")
        return (token_i in t) or (token_d in t)

    # 1) Must include Pareto threshold (cutoff) and Top-N (n_80) if provided
    if cutoff is not None and not _has_number_token(cutoff):
        return False, "missing_cutoff", diag
    if n80 is not None:
        # accept either the raw number OR explicit "top {n80}"
        if (str(int(n80)) not in t) and (f"top {int(n80)}" not in t):
            return False, "missing_top_n", diag

    # 2) Must include tail share (ogon) if provided
    if tail is not None and not _has_number_token(tail):
        return False, "missing_tail_share", diag
    if tail is not None and ("ogon" not in t and "pozosta" not in t and "long tail" not in t):
        diag["warnings"].append("tail_without_context")

    # 3) Should include at least one of: top1 share OR total value
    # (top1 is optional, but if present it's good)
    if top1 is not None and not _has_number_token(top1):
        diag["warnings"].append("missing_top1_share_token")

    # 4) Decision must be explicit and cover BOTH: Top-N AND tail
    DECISION_VERBS = ["ustal", "wdroż", "zabezpiecz", "skoncentruj", "zwiększ", "ogranicz", "przenieś", "skaluj", "wytnij", "upraszczaj"]
    if not any(v in t for v in DECISION_VERBS):
        return False, "weak_or_missing_decision", diag

    # Must mention both parts of the strategy
    has_topn_ref = ("top" in t) or (n80 is not None and str(int(n80)) in t)
    has_tail_ref = ("ogon" in t) or ("pozosta" in t) or ("long tail" in t)

    if not (has_topn_ref and has_tail_ref):
        return False, "decision_not_covering_topn_and_tail", diag

    # 5) Prevent Top-1-only recommendations
    # If text focuses on "największy segment" but does not mention Top-N and tail, reject (already covered),
    # additionally block phrasing that implies ONLY top1 action.
    TOP1_ONLY_PATTERNS = [
        r"\btylko\b.*\bnajwi[eę]ksz",
        r"\bskup\s*si[eę]\b.*\bnajwi[eę]ksz",
        r"\binwest\w+\b.*\bnajwi[eę]ksz",
    ]
    if any(re.search(p, t) for p in TOP1_ONLY_PATTERNS) and not has_tail_ref:
        return False, "top1_only_bias", diag

    return True, "ok", diag

# ─────────────────────────────────────────────────────────────
# SPEC: cs_price_corridor
# Required anchors: p20, p80, corridor_share, p80_price, bin_at_p80, nan_bin_share(optional), metric_name, total_value
# ─────────────────────────────────────────────────────────────

def _det_price_corridor(a: Dict[str, Any]) -> str:
    metric = str(a.get("metric_name") or a.get("metric") or "wartość")
    total = a.get("total_value")
    p20 = a.get("p20_price")
    p80 = a.get("p80_price")
    corridor_share = a.get("corridor_share_pct")
    bin80 = a.get("bin_at_p80")
    nan_share = a.get("nan_price_share_pct")

    s1 = (
        f"Korytarz P20–P80 dla {metric} to {_fmt_num(p20)}–{_fmt_num(p80)} "
        f"i obejmuje {_fmt_num(corridor_share)}% {metric} ({_fmt_num(total)} łącznie)."
    )
    s2_bits = [f"Ustal priorytet cenowy i ekspozycję oferty w korytarzu do „{bin80}” (P80) oraz zoptymalizuj miks produktów w tym zakresie"]
    if nan_share is not None:
        s2_bits.append(f"i usuń braki cen (NaN: {_fmt_num(nan_share)}%), bo zaniżają wiarygodność wniosków")
    s2_bits.append("— to najszybsza dźwignia wzrostu przy kontrolowanym ryzyku marży.")
    s2 = ", ".join(s2_bits).replace(" ,", ",") + "."
    return f"{s1} {s2}"

def _prompt_price_corridor(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    metric = str(a.get("metric_name") or a.get("metric") or "wartość")
    anchors = {
        "p20_price": a.get("p20_price"),
        "p80_price": a.get("p80_price"),
        "corridor_share_pct": a.get("corridor_share_pct"),
        "bin_at_p80": a.get("bin_at_p80"),
        "nan_price_share_pct": a.get("nan_price_share_pct"),
        "metric_name": metric,
        "total_value": a.get("total_value"),
    }
    return f"""
Jesteś analitykiem (McKinsey/Bain). Napisz Executive Takeaway DOKŁADNIE w 2 zdaniach po polsku dla wykresu "korytarz cenowy".
Wymagania twarde:
- Zdanie 1: fakt + MIN 2 liczby, MUSI zawierać P20, P80 i udział korytarza.
- Zdanie 2: implikacja + decyzja (czasownik), odniesienie do progu P80 (bin_at_p80) i/lub jakości danych (NaN) jeśli podane.
- Zakaz: ogólniki, "ten wykres pokazuje", wielokropki, trzecie zdanie.
- Zakaz sprzeczności: JEŚLI anchors zawierają P20/P80/udział, NIE WOLNO pisać "brak", "0% udziału", "nieokreślone".
- Decyzja musi dotyczyć działań cenowo-ofertowych (pricing / dostępność / promka / miks), nie "analizuj dane".
- Użyj WYŁĄCZNIE danych z ANCHORS.

ANCHORS (Price corridor):
{anchors}

Zwróć JSON:
{{
  "candidates": [
    {{"text": "<2-zdaniowy takeaway>"}},
    {{"text": "<2-zdaniowy takeaway>"}},
    {{"text": "<2-zdaniowy takeaway>"}}
  ]
}}
""".strip()

import re

def _validate_price_corridor(text: str, a: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    ok, why, diag = _validate_two_sentences_pl(text)
    if not ok:
        return ok, why, diag

    t_raw = (text or "").strip()
    t = t_raw.lower()

    # --- Sentence split (alignment gate działa na 2. zdaniu = decyzji) ---
    parts = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", t_raw) if s.strip()]
    s2 = (parts[1].lower() if len(parts) >= 2 else t)

    # --- 0) No contradiction when anchors are present ---
    # If we DO have numeric anchors, the text must not claim they are missing/undefined/0%.
    p20 = a.get("p20_price")
    p80 = a.get("p80_price")
    share = a.get("corridor_share_pct")
    nan_share = a.get("nan_price_share_pct")

    has_p20 = p20 is not None
    has_p80 = p80 is not None
    has_share = share is not None

    CONTRA_PATTERNS = [
        r"\bbrak\b", r"\bnie (zosta[łl]a|jest) (okre[śs]lon[ay]|ustalon[ay])\b",
        r"\bnieznan[ay]\b", r"\bbrakuje\b", r"\b0% (udzia[łl]u|udzia[łl])\b",
        r"\bzerow[ay]\b", r"\bnie ma (udzia[łl]u|sensu)\b",
    ]
    if (has_p20 or has_p80 or has_share) and any(re.search(p, t) for p in CONTRA_PATTERNS):
        # allow "brak cen (NaN)" if that's what we mean
        if not re.search(r"\bnan\b|\bbrak cen\b|\bbraki danych\b", t):
            return False, "contradiction_with_anchors", diag

    # --- 1) Must include P20 and P80 tokens if provided ---
    def _has_number_token(val: Any) -> bool:
        try:
            fv = float(val)
        except Exception:
            return False
        token_i = str(int(round(fv)))                 # 120000
        token_d = f"{fv:.1f}".replace(".", ",")       # 120000,0
        return (token_i in t) or (token_d in t)

    if has_p20 and not _has_number_token(p20):
        return False, "missing_p20", diag
    if has_p80 and not _has_number_token(p80):
        return False, "missing_p80", diag

    # --- 2) Must include corridor share % if provided ---
    if has_share:
        if not _has_number_token(share):
            return False, "missing_corridor_share", diag
        # Also enforce that it's described as "udział" / "korytarz"
        if ("udział" not in t) and ("korytarz" not in t):
            diag["warnings"].append("share_without_context")

    # --- 3) Must use bin_at_p80 label if present (force operational threshold) ---
    b = a.get("bin_at_p80")
    if isinstance(b, str) and b.strip():
        if b.strip().lower() not in t:
            return False, "missing_bin_at_p80", diag

    # --- 4) If NaN share is provided and non-trivial, must be mentioned explicitly ---
    try:
        ns = float(nan_share) if nan_share is not None else None
    except Exception:
        ns = None
    if (ns is not None) and (ns >= 1.0):  # >=1% is worth mentioning
        if ("nan" not in t) and ("brak cen" not in t) and ("braki danych" not in t):
            return False, "missing_nan_quality_note", diag

    # --- 5) Decision must be explicit (imperative verb) and price-action oriented ---
    # We want a real "so what": adjust pricing/offer/availability, not vague "analizuj".
    DECISION_VERBS = ["ustal", "wdroż", "skoncentruj", "zwiększ", "ogranicz", "podnieś", "obniż", "przesuń", "zablokuj", "testuj", "skaluj"]
    if not any(v in t for v in DECISION_VERBS):
        return False, "weak_or_missing_decision", diag

    # Avoid generic second sentence
    BANNED_GENERIC = ["zaleca się analizę", "warto przeanalizować", "należy przeanalizować dane"]
    if any(bg in t for bg in BANNED_GENERIC):
        return False, "generic_decision", diag

    # --- 6) Alignment gate: jeśli korytarz ma wysoki udział, decyzja nie może być WYŁĄCZNIE "powyżej P80" ---
    # Wymóg: gdy corridor_share_pct >= 50%, w 2. zdaniu musi paść działanie dot. P20–P80/korytarza,
    # a nie tylko "powyżej/ponad P80".
    corridor_share = None
    try:
        corridor_share = float(share) if share is not None else None
    except Exception:
        corridor_share = None

    if corridor_share is not None and corridor_share >= 50.0:
        mentions_above_p80 = (
            (("powyżej" in s2) or ("ponad" in s2) or ("above" in s2) or (">" in s2))
            and ("p80" in s2 or "progu p80" in s2)
        )
        corridor_terms = [
            "korytarz",
            "p20",
            "20–80", "20-80",
            "p20–p80", "p20-p80",
            "do p80",
            "w przedziale",
            "w zakresie",
            "wewnątrz",
        ]
        mentions_corridor_action = any(term in s2 for term in corridor_terms)

        if mentions_above_p80 and not mentions_corridor_action:
            return False, "decision_only_above_p80_high_corridor_share", diag

    return True, "ok", diag


# ─────────────────────────────────────────────────────────────
# Additional SPECS (v2 scaling) — Composition Static + Distribution
# NOTE: deterministic_fn MUST pass validate_fn.
# ─────────────────────────────────────────────────────────────

# ===== cs_ranking_topn =====
def _det_cs_ranking_topn(a: Dict[str, Any]) -> str:
    metric = str(a.get("metric_name") or a.get("metric") or "wartość")
    cat1 = str(a.get("cat1") or "kategoria")
    top_seg = str(a.get("top_segment") or "Top-1")
    top_val = a.get("top_value")
    top_share = a.get("top_share_pct")
    nseg = a.get("n_segments")
    hhi = a.get("hhi")
    s1_bits = [
        f"Największy segment {top_seg} ma {_fmt_num(top_share)}% udziału i {_fmt_num(top_val)} {metric} w {cat1}",
        f"przy {int(nseg)} segmentach"
    ]
    if hhi is not None:
        s1_bits.append(f"(HHI: {_fmt_num(hhi, digits=0)})")
    s1 = ", ".join(s1_bits).rstrip(".") + "."
    s2 = (
        f"Skup zasoby na Top‑N (utrzymaj dostępność i egzekucję cenowo‑promocyjną dla {top_seg}), "
        f"a równolegle uruchom 2–3 testy wzrostu dla kolejnych segmentów, aby zmniejszyć koncentrację i podnieść udział poza Top‑1."
    )
    return f"{s1} {s2}"

def _prompt_cs_ranking_topn(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    anchors = {
        "metric": a.get("metric_name") or a.get("metric"),
        "cat1": a.get("cat1"),
        "top_segment": a.get("top_segment"),
        "top_value": a.get("top_value"),
        "top_share_pct": a.get("top_share_pct"),
        "n_segments": a.get("n_segments"),
        "hhi": a.get("hhi"),
    }
    return f"""
Jesteś analitykiem (McKinsey/Bain). Napisz Executive Takeaway DOKŁADNIE w 2 zdaniach po polsku dla bloku Ranking (Top‑N).
MUST:
- Zdanie 1: fakt + MIN 3 liczby (top_value, top_share_pct, n_segments) i nazwa top_segment.
- Zdanie 2: decyzja (czasownik) dot. alokacji zasobów: co robimy z Top‑N oraz co robimy z resztą (testy/opt).
Zakazy: ogólniki, "ten wykres pokazuje", wielokropki, trzecie zdanie, sprzeczności ("brak"/"0%") jeśli anchors są.
Użyj WYŁĄCZNIE ANCHORS.
ANCHORS:
{anchors}
Zwróć JSON:
{{"candidates":[{{"text":"..."}},{{"text":"..."}},{{"text":"..."}}]}}
""".strip()

def _validate_cs_ranking_topn(text: str, a: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    ok, why, diag = _validate_two_sentences_pl(text)
    if not ok:
        return ok, why, diag
    t_raw = (text or "").strip()
    t = t_raw.lower()

    # hard ban generic
    if any(bg in t for bg in ("warto przeanalizować", "zaleca się analizę", "należy przeanalizować")):
        return False, "generic", diag

    # must include numeric anchors
    for k in ("top_value", "top_share_pct", "n_segments"):
        v = a.get(k)
        if v is not None and not _has_number_token_in_text(v, t_raw):
            return False, f"missing_{k}", diag

    # must include top segment name
    top_seg = str(a.get("top_segment") or "").strip().lower()
    if top_seg and top_seg not in t:
        diag["warnings"].append("missing_top_segment_name")

    return True, "ok", diag

# ===== cs_waterfall_contrib =====
def _det_cs_waterfall_contrib(a: Dict[str, Any]) -> str:
    metric = str(a.get("metric_name") or a.get("metric") or "wartość")
    cat1 = str(a.get("cat1") or "kategoria")
    top = str(a.get("top_item") or "Top-1")
    top_val = a.get("top_item_value")
    top_share = a.get("top_item_share_pct")
    n_items = a.get("n_items")
    s1 = (
        f"Największy wkład ma {top}: {_fmt_num(top_val)} {metric}, czyli {_fmt_num(top_share)}% całości, "
        f"przy {int(n_items)} pozycjach na waterfall."
    )
    s2 = (
        f"Zabezpiecz utrzymanie wkładu Top‑1 ({top}) i ustaw KPI dla Top‑N, "
        f"a ogon skonsoliduj (Other/małe pozycje) oraz testuj 2–3 dźwignie, aby zwiększyć sumę bez wzrostu złożoności."
    )
    return f"{s1} {s2}"

def _prompt_cs_waterfall_contrib(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    anchors = {
        "metric": a.get("metric_name") or a.get("metric"),
        "cat1": a.get("cat1"),
        "top_item": a.get("top_item"),
        "top_item_value": a.get("top_item_value"),
        "top_item_share_pct": a.get("top_item_share_pct"),
        "n_items": a.get("n_items"),
    }
    return f"""
Jesteś analitykiem (McKinsey/Bain). Napisz Executive Takeaway DOKŁADNIE w 2 zdaniach po polsku dla bloku Waterfall (wkłady).
MUST:
- Zdanie 1: fakt + MIN 3 liczby (top_item_value, top_item_share_pct, n_items) i nazwa top_item.
- Zdanie 2: decyzja (czasownik) – co zabezpieczyć w Top oraz co uprościć/skalować w ogonie.
Zakazy: ogólniki, "ten wykres pokazuje", wielokropki, trzecie zdanie.
ANCHORS:
{anchors}
Zwróć JSON:
{{"candidates":[{{"text":"..."}},{{"text":"..."}},{{"text":"..."}}]}}
""".strip()

def _validate_cs_waterfall_contrib(text: str, a: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    ok, why, diag = _validate_two_sentences_pl(text)
    if not ok:
        return ok, why, diag
    t_raw = (text or "").strip()
    t = t_raw.lower()
    if any(bg in t for bg in ("warto przeanalizować", "zaleca się analizę", "należy przeanalizować")):
        return False, "generic", diag
    for k in ("top_item_value", "top_item_share_pct", "n_items"):
        v = a.get(k)
        if v is not None and not _has_number_token_in_text(v, t_raw):
            return False, f"missing_{k}", diag
    top = str(a.get("top_item") or "").strip().lower()
    if top and top not in t:
        diag["warnings"].append("missing_top_item_name")
    return True, "ok", diag

# ===== cs_mix_share / cs_marimekko =====
def _det_cs_mix_share(a: Dict[str, Any]) -> str:
    cat1 = str(a.get("cat1") or "grupa")
    cat2 = str(a.get("cat2") or "składnik")
    top_n = a.get("top_n")
    dom_sub = str(a.get("dominant_sub") or "Top składnik")
    dom_sub_share = a.get("dominant_sub_share_pct")
    max_group = str(a.get("max_group") or "Top grupa")
    max_group_share = a.get("max_group_share_pct")
    n_groups = a.get("n_groups")
    n_subs = a.get("n_subs")
    s1 = (
        f"Największy składnik {dom_sub} ma {_fmt_num(dom_sub_share)}% w mix, "
        f"a największa grupa {max_group} ma {_fmt_num(max_group_share)}% udziału; analizujemy {int(n_groups)} grup i {int(n_subs)} składników (Top‑{int(top_n)})."
    )
    s2 = (
        f"Ustal docelowy mix w {cat1} (zabezpiecz {dom_sub} tam, gdzie jest krytyczny) i ogranicz proliferację w {cat2} "
        f"przez standaryzację słabych składników + 2–3 testy zamienników, aby poprawić marżę/efektywność bez utraty udziału."
    )
    return f"{s1} {s2}"

def _prompt_cs_mix_share(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    return f"""
Jesteś analitykiem biznesowym (McKinsey/Bain grade).

CEL:
Napisz Executive Takeaway dla wykresu MIX (100% stacked) pokazującego strukturę składników W RAMACH GRUP.

FORMAT (OBOWIĄZKOWE):
- Dokładnie 2 zdania.
- Zdanie 1 = FAKT (liczby + opis mixu w grupach).
- Zdanie 2 = DECYZJA OPERACYJNA (czasownik decyzyjny + 2 działania).

ZASADY TWARDЕ:
- Użyj WYŁĄCZNIE danych z ANCHORS.
- Minimum 2 liczby w całym tekście.
- Wspomnij nazwę dominującego składnika i nazwę największej grupy.
- Zakaz ogólników, zakaz meta-opisu wykresu, zakaz 3. zdania.

ZDANIE 1 MUSI ZAWIERAĆ:
- top_component + jego udział w grupie (%)
- top_group + jej udział (%)
- liczbę grup (n_groups)
- kontekst: „mix w ramach grup”

ZDANIE 2 MUSI ZAWIERAĆ (OBA):
1) decyzję o standaryzacji / ograniczeniu wariantów w dominującym mixie,
2) decyzję o testach miksu (np. 2–3 warianty) w słabszych grupach.

ANCHORS:
{a}

ZWRÓĆ WYŁĄCZNIE TEKST (bez JSON, bez list, bez nagłówków).
""".strip()


def _validate_cs_mix_share(text: str, a: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    diag = {"warnings": []}

    ok, why, _ = _validate_two_sentences_pl(text)
    if not ok:
        return False, why, diag

    t = text.lower()

    if _count_numbers(text) < 2:
        return False, "too_few_numbers", diag

    if not any(w in t for w in ["mix", "w ramach grup"]):
        return False, "missing_mix_context", diag

    if str(a.get("top_component", "")).lower() not in t:
        return False, "missing_top_component", diag

    if not any(v in t for v in ["standaryzuj", "ogranicz"]):
        return False, "missing_standardization_decision", diag

    if not any(v in t for v in ["testuj", "przetestuj"]):
        return False, "missing_testing_decision", diag

    return True, "ok", diag

def _det_cs_marimekko(a: Dict[str, Any]) -> str:
    cat1 = str(a.get("cat1") or "grupa")
    cat2 = str(a.get("cat2") or "składnik")
    top_n = a.get("top_n")
    dom_sub = str(a.get("dominant_sub") or "Top składnik")
    dom_sub_share = a.get("dominant_sub_share_pct")
    max_group = str(a.get("max_group") or "Top grupa")
    max_group_share = a.get("max_group_share_pct")
    n_groups = a.get("n_groups")
    n_subs = a.get("n_subs")

    s1 = (
        f"Wykres Marimekko łączy skalę (szerokość {cat1}) i strukturę (wysokość {cat2}): "
        f"największy składnik {dom_sub} ma {_fmt_num(dom_sub_share)}% w swojej grupie, "
        f"a największa grupa {max_group} odpowiada za {_fmt_num(max_group_share)}% wartości; analizujemy {int(n_groups)} grup i {int(n_subs)} składników (Top‑{int(top_n)})."
    )
    s2 = (
        f"Decyzja: zabezpiecz {max_group} (największa szerokość) i standaryzuj {cat2} w jej obrębie (Top‑{int(top_n)} + Other), "
        f"priorytetyzując uproszczenia tam, gdzie {dom_sub} dominuje, aby ograniczyć złożoność bez utraty udziału."
    )
    return f"{s1} {s2}"

def _prompt_cs_marimekko(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    return f"""
Jesteś analitykiem biznesowym (McKinsey/Bain grade).

CEL:
Napisz Executive Takeaway dla wykresu MARIMEKKO pokazującego skalę × strukturę (wkład komórek do totalu).

FORMAT (OBOWIĄZKOWE):
- Dokładnie 2 zdania.
- Zdanie 1 = FAKT (top komórka + liczby).
- Zdanie 2 = DECYZJA OPERACYJNA (priorytety TOP vs reszta).

ZASADY TWARDЕ:
- Użyj WYŁĄCZNIE danych z ANCHORS.
- Minimum 2 liczby.
- Użyj pojęcia „komórka” lub zapisu A×B.
- Zakaz ogólników, zakaz meta-opisu, zakaz 3. zdania.

ZDANIE 1 MUSI ZAWIERAĆ:
- top_cell_group × top_cell_component
- top_cell_share_pct (%)
- top_cell_value ORAZ total_value
- liczbę komórek (n_cells)
- kontekst: wkład do totalu (skala × struktura)

ZDANIE 2 MUSI ZAWIERAĆ (OBA):
1) decyzję o zabezpieczeniu / skalowaniu TOP komórki,
2) decyzję o racjonalizacji / konsolidacji pozostałych komórek (ogon).

ANCHORS:
{a}

ZWRÓĆ WYŁĄCZNIE TEKST (bez JSON, bez list, bez nagłówków).
""".strip()

def _validate_cs_marimekko(text: str, a: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    diag = {"warnings": []}

    ok, why, _ = _validate_two_sentences_pl(text)
    if not ok:
        return False, why, diag

    t = text.lower()

    if _count_numbers(text) < 2:
        return False, "too_few_numbers", diag

    if not ("×" in text or "komórk" in t):
        return False, "missing_cell_reference", diag

    if not any(w in t for w in ["wkład", "total"]):
        return False, "missing_total_context", diag

    if not any(v in t for v in ["zabezpiecz", "skaluj"]):
        return False, "missing_top_cell_action", diag

    if not any(v in t for v in ["konsoliduj", "ogranicz", "priorytetyzuj", "uproszcz"]):
        return False, "missing_tail_action", diag

    return True, "ok", diag

# ===== Distribution specs (dist_*) =====
def _det_dist_boxplot(a: Dict[str, Any]) -> str:
    col = str(a.get("col") or "zmienna")
    n = a.get("n")
    med = a.get("median")
    iqr = a.get("iqr")
    outp = a.get("outliers_pct")
    s1 = f"Mediana {col} wynosi {_fmt_num(med)}, a IQR {_fmt_num(iqr)} przy N={int(n)}; outliery to {_fmt_num(outp)}% obserwacji."
    s2 = "Ustal regułę obsługi outlierów (winsoryzacja lub cap na fence) i monitoruj {_fmt_num(outp)}% odchyleń, aby stabilizować wnioski i modele."
    return f"{s1} {s2}"

def _prompt_dist_boxplot(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    anchors = {k: a.get(k) for k in ("col","n","median","iqr","outliers_pct","lower_fence","upper_fence")}
    return f"""
Napisz Executive Takeaway (PL) w 2 zdaniach dla Boxplot/outliers.
MUST: zdanie1 ma median, IQR, outliers_pct, N. Zdanie2: decyzja jak traktować outliery (winsoryzacja/cap) + monitoruj.
Zakazy: ogólniki.
ANCHORS:{anchors}
Zwróć JSON candidates.
""".strip()

def _validate_dist_boxplot(text: str, a: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    ok, why, diag = _validate_two_sentences_pl(text)
    if not ok: return ok, why, diag
    t_raw = (text or "").strip()
    for k in ("median","iqr","outliers_pct","n"):
        v=a.get(k)
        if v is not None and not _has_number_token_in_text(v, t_raw): return False, f"missing_{k}", diag
    if any(bg in (text or "").lower() for bg in ("warto przeanalizować","zaleca się analizę")): return False,"generic",diag
    return True,"ok",diag

def _det_dist_violin(a: Dict[str, Any]) -> str:
    col=str(a.get("col") or "zmienna"); n=a.get("n"); med=a.get("median"); p95=a.get("p95") or a.get("p90"); skew=a.get("skewness")
    s1=f"Mediana {col} to {_fmt_num(med)}, a p95 {_fmt_num(p95)} przy N={int(n)} (skośność: {_fmt_num(skew)})."
    s2="Dostosuj segmentację/próg KPI do ogona (p95) i przetestuj transformację (log/Box-Cox), aby ograniczyć wpływ skośności na decyzje."
    return f"{s1} {s2}"

def _prompt_dist_violin(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    anchors={k:a.get(k) for k in ("col","n","median","p95","skewness")}
    return f"Napisz 2-zdaniowy ET dla violin. MUST: median, p95, N, skewness. Zdanie2: decyzja dot. ogona/transformacji. ANCHORS:{anchors}"

def _validate_dist_violin(text: str, a: Dict[str, Any]) -> Tuple[bool,str,Dict[str,Any]]:
    ok,why,diag=_validate_two_sentences_pl(text)
    if not ok: return ok,why,diag
    t_raw=(text or "").strip()
    for k in ("median","p95","n"):
        v=a.get(k) or a.get("p90") if k=="p95" else a.get(k)
        if v is not None and not _has_number_token_in_text(v,t_raw): return False,f"missing_{k}",diag
    return True,"ok",diag

def _det_dist_rug(a: Dict[str, Any]) -> str:
    col=str(a.get("col") or "zmienna"); n=a.get("n"); p99=a.get("p99") or a.get("p95"); outp=a.get("outliers_pct")
    s1=f"Dla {col} p99 wynosi {_fmt_num(p99)} przy N={int(n)}, a outliery stanowią {_fmt_num(outp)}%."
    s2="Ustaw alerty na skrajne wartości (p99) i ogranicz wpływ odstających obserwacji przez cap/winsoryzację, aby uniknąć błędnych decyzji."
    return f"{s1} {s2}"

def _prompt_dist_rug(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    anchors={k:a.get(k) for k in ("col","n","p99","outliers_pct")}
    return f"Napisz 2-zdaniowy ET dla rug. MUST: p99, outliers_pct, N. Zdanie2: decyzja alerty/cap. ANCHORS:{anchors}"

def _validate_dist_rug(text: str, a: Dict[str, Any]) -> Tuple[bool,str,Dict[str,Any]]:
    ok,why,diag=_validate_two_sentences_pl(text)
    if not ok: return ok,why,diag
    t_raw=(text or "").strip()
    for k in ("p99","outliers_pct","n"):
        v=a.get(k)
        if v is not None and not _has_number_token_in_text(v,t_raw): return False,f"missing_{k}",diag
    return True,"ok",diag

def _det_dist_ecdf(a: Dict[str, Any]) -> str:
    col=str(a.get("col") or "zmienna"); n=a.get("n"); p95=a.get("p95"); p99=a.get("p99")
    s1=f"Dystrybuanta pokazuje, że p95 dla {col} to {_fmt_num(p95)}, a p99 {_fmt_num(p99)} przy N={int(n)}."
    s2="Ustal progi operacyjne na p95/p99 (SLA/KPI) i monitoruj przekroczenia, aby sterować ryzykiem ogona."
    return f"{s1} {s2}"

def _prompt_dist_ecdf(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    anchors={k:a.get(k) for k in ("col","n","p95","p99")}
    return f"Napisz 2-zdaniowy ET dla ECDF. MUST: p95, p99, N. Zdanie2: decyzja progi/monitoring. ANCHORS:{anchors}"

def _validate_dist_ecdf(text: str, a: Dict[str, Any]) -> Tuple[bool,str,Dict[str,Any]]:
    ok,why,diag=_validate_two_sentences_pl(text)
    if not ok: return ok,why,diag
    t_raw=(text or "").strip()
    for k in ("p95","p99","n"):
        v=a.get(k)
        if v is not None and not _has_number_token_in_text(v,t_raw): return False,f"missing_{k}",diag
    return True,"ok",diag

def _det_dist_hist_log(a: Dict[str, Any]) -> str:
    col=str(a.get("col") or "zmienna"); n=a.get("n"); med=a.get("median"); p99=a.get("p99"); skew=a.get("skewness")
    s1=f"Rozkład {col} jest skośny (skośność: {_fmt_num(skew)}): mediana {_fmt_num(med)} vs p99 {_fmt_num(p99)} przy N={int(n)}."
    s2="Stosuj skalę log/transformację przy analizie i progach, a decyzje opieraj o percentyle (p90/p95), żeby ograniczyć wpływ ogona."
    return f"{s1} {s2}"

def _prompt_dist_hist_log(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    anchors={k:a.get(k) for k in ("col","n","median","p99","skewness")}
    return f"Napisz 2-zdaniowy ET dla histogram log. MUST: skewness, median, p99, N. Zdanie2: decyzja transformacja/percentyle. ANCHORS:{anchors}"

def _validate_dist_hist_log(text: str, a: Dict[str, Any]) -> Tuple[bool,str,Dict[str,Any]]:
    ok,why,diag=_validate_two_sentences_pl(text)
    if not ok: return ok,why,diag
    t_raw=(text or "").strip()
    for k in ("median","p99","n"):
        v=a.get(k)
        if v is not None and not _has_number_token_in_text(v,t_raw): return False,f"missing_{k}",diag
    return True,"ok",diag

def _det_dist_kde(a: Dict[str, Any]) -> str:
    col=str(a.get("col") or "zmienna"); n=a.get("n"); med=a.get("median"); iqr=a.get("iqr"); skew=a.get("skewness")
    s1=f"Gęstość (KDE) dla {col} ma medianę {_fmt_num(med)} i IQR {_fmt_num(iqr)} przy N={int(n)} (skośność: {_fmt_num(skew)})."
    s2="Ustal segmenty/progi wokół mediany i IQR oraz monitoruj odchylenia, aby szybciej wykrywać przesunięcia rozkładu."
    return f"{s1} {s2}"

def _prompt_dist_kde(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    anchors={k:a.get(k) for k in ("col","n","median","iqr","skewness")}
    return f"Napisz 2-zdaniowy ET dla KDE. MUST: median, IQR, N, skewness. Zdanie2: decyzja progi/monitoring. ANCHORS:{anchors}"

def _validate_dist_kde(text: str, a: Dict[str, Any]) -> Tuple[bool,str,Dict[str,Any]]:
    ok,why,diag=_validate_two_sentences_pl(text)
    if not ok: return ok,why,diag
    t_raw=(text or "").strip()
    for k in ("median","iqr","n"):
        v=a.get(k)
        if v is not None and not _has_number_token_in_text(v,t_raw): return False,f"missing_{k}",diag
    return True,"ok",diag

def _det_dist_hist_narrow(a: Dict[str, Any]) -> str:
    col=str(a.get("col") or "zmienna"); n=a.get("n"); mn=a.get("min"); mx=a.get("max"); med=a.get("median")
    s1=f"W węższych binach dla {col} zakres to {_fmt_num(mn)}–{_fmt_num(mx)}, a mediana {_fmt_num(med)} przy N={int(n)}."
    s2="Zwiększ rozdzielczość binów tylko dla kluczowego zakresu (okolice mediany) i ustaw progi operacyjne, aby lepiej sterować decyzjami."
    return f"{s1} {s2}"

def _prompt_dist_hist_narrow(ctx: Dict[str, Any], a: Dict[str, Any]) -> str:
    anchors={k:a.get(k) for k in ("col","n","min","max","median")}
    return f"Napisz 2-zdaniowy ET dla histogram wąskie biny. MUST: min, max, median, N. Zdanie2: decyzja o zakresie/progach. ANCHORS:{anchors}"

def _validate_dist_hist_narrow(text: str, a: Dict[str, Any]) -> Tuple[bool,str,Dict[str,Any]]:
    ok,why,diag=_validate_two_sentences_pl(text)
    if not ok: return ok,why,diag
    t_raw=(text or "").strip()
    for k in ("min","max","median","n"):
        v=a.get(k)
        if v is not None and not _has_number_token_in_text(v,t_raw): return False,f"missing_{k}",diag
    return True,"ok",diag

TAKEAWAY_SPECS: Dict[str, TakeawaySpec] = {
    "cs_pareto": TakeawaySpec(
        spec_id="cs_pareto",
        required_fields=("cutoff", "n_80", "share_top1_pct", "tail_share_pct", "metric_name", "total_value"),
        deterministic_fn=_det_pareto,
        prompt_fn=_prompt_pareto,
        validate_fn=_validate_pareto,
    ),
    "cs_price_corridor": TakeawaySpec(
        spec_id="cs_price_corridor",
        required_fields=("p20_price", "p80_price", "corridor_share_pct", "bin_at_p80", "metric_name", "total_value"),
        deterministic_fn=_det_price_corridor,
        prompt_fn=_prompt_price_corridor,
        validate_fn=_validate_price_corridor,
    ),

    # Composition Static — scaled
    "cs_ranking_topn": TakeawaySpec(
        spec_id="cs_ranking_topn",
        required_fields=("metric", "cat1", "top_segment", "top_value", "top_share_pct", "n_segments"),
        deterministic_fn=_det_cs_ranking_topn,
        prompt_fn=_prompt_cs_ranking_topn,
        validate_fn=_validate_cs_ranking_topn,
    ),
    "cs_waterfall_contrib": TakeawaySpec(
        spec_id="cs_waterfall_contrib",
        required_fields=("metric", "cat1", "top_item", "top_item_value", "top_item_share_pct", "n_items"),
        deterministic_fn=_det_cs_waterfall_contrib,
        prompt_fn=_prompt_cs_waterfall_contrib,
        validate_fn=_validate_cs_waterfall_contrib,
    ),
    "cs_mix_share": TakeawaySpec(
        spec_id="cs_mix_share",
        required_fields=("cat1", "cat2", "top_n", "dominant_sub", "dominant_sub_share_pct", "max_group", "max_group_share_pct", "n_groups", "n_subs"),
        deterministic_fn=_det_cs_mix_share,
        prompt_fn=_prompt_cs_mix_share,
        validate_fn=_validate_cs_mix_share,
    ),
    "cs_marimekko": TakeawaySpec(
        spec_id="cs_marimekko",
        required_fields=("cat1", "cat2", "top_n", "dominant_sub", "dominant_sub_share_pct", "max_group", "max_group_share_pct", "n_groups", "n_subs"),
        deterministic_fn=_det_cs_marimekko,
        prompt_fn=_prompt_cs_marimekko,
        validate_fn=_validate_cs_marimekko,
    ),

    # Distribution — scaled
    "dist_boxplot_outliers": TakeawaySpec(
        spec_id="dist_boxplot_outliers",
        required_fields=("col", "n", "median", "iqr", "outliers_pct"),
        deterministic_fn=_det_dist_boxplot,
        prompt_fn=_prompt_dist_boxplot,
        validate_fn=_validate_dist_boxplot,
    ),
    "dist_violin_density": TakeawaySpec(
        spec_id="dist_violin_density",
        required_fields=("col", "n", "median", "p95", "skewness"),
        deterministic_fn=_det_dist_violin,
        prompt_fn=_prompt_dist_violin,
        validate_fn=_validate_dist_violin,
    ),
    "dist_rug_points": TakeawaySpec(
        spec_id="dist_rug_points",
        required_fields=("col", "n", "p99", "outliers_pct"),
        deterministic_fn=_det_dist_rug,
        prompt_fn=_prompt_dist_rug,
        validate_fn=_validate_dist_rug,
    ),
    "dist_ecdf_thresholds": TakeawaySpec(
        spec_id="dist_ecdf_thresholds",
        required_fields=("col", "n", "p95", "p99"),
        deterministic_fn=_det_dist_ecdf,
        prompt_fn=_prompt_dist_ecdf,
        validate_fn=_validate_dist_ecdf,
    ),
    "dist_hist_log": TakeawaySpec(
        spec_id="dist_hist_log",
        required_fields=("col", "n", "median", "p99", "skewness"),
        deterministic_fn=_det_dist_hist_log,
        prompt_fn=_prompt_dist_hist_log,
        validate_fn=_validate_dist_hist_log,
    ),
    "dist_kde_density": TakeawaySpec(
        spec_id="dist_kde_density",
        required_fields=("col", "n", "median", "iqr", "skewness"),
        deterministic_fn=_det_dist_kde,
        prompt_fn=_prompt_dist_kde,
        validate_fn=_validate_dist_kde,
    ),
    "dist_hist_narrow_bins": TakeawaySpec(
        spec_id="dist_hist_narrow_bins",
        required_fields=("col", "n", "min", "max", "median"),
        deterministic_fn=_det_dist_hist_narrow,
        prompt_fn=_prompt_dist_hist_narrow,
        validate_fn=_validate_dist_hist_narrow,
    ),
}
