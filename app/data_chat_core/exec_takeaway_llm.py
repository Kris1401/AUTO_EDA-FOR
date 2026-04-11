"""Executive Takeaway — LLM bundle (v1.0)

Cel:
- Jedno wywołanie LLM dla listy bloków (np. Distribution → Kluczowe insighty).
- Każdy blok dostaje 2-zdaniowy Executive Takeaway w stylu McKinsey/Bain.
- Każdy tekst MUSI zawierać liczby (z przekazanego stats) oraz decyzję / rekomendację.

Kontrakt odpowiedzi (prompt enforced):
- 2 zdania.
- Zdanie 1: „what it means” + 1–2 liczby.
- Zdanie 2: decyzja/akcja (np. progi, transformacja, segmentacja) + 1 liczba.
- Bez meta-komentarzy, bez 'ten wykres pokazuje', bez 'KDE/ECDF', bez powtarzania guidance.

Minimalny schema statystyk (używany przez prompt i walidację):
{
  "col": str,
  "n": int,
  "median": float,
  "q1": float,
  "q3": float,
  "iqr": float,
  "p90": float,
  "p95": float,
  "p99": float,
  "min": float,
  "max": float,
  "mean": float,
  "std": float,
  "skewness": float,
  "outliers_share": float,          # 0..1
  "outliers_pct": float,            # 0..100
  "lower_fence": float,
  "upper_fence": float,
  "tail_ratio_p95_median": float    # p95 / median
}

Uwaga:
- Ten moduł jest defensywny: jeśli nie ma klienta LLM w ctx albo walidacja nie przejdzie,
  zwraca {} i UI spada na fallback per-blok.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import re


# ─────────────────────────────────────────────────────────────────────────────
# Prompt v1.0 (McKinsey contract)
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_V1 = r"""
Jesteś konsultantem McKinsey/Bain. Tworzysz Executive Takeaway dla dashboardu analitycznego.

ZADANIE
Dla KAŻDEGO bloku (lista poniżej) wygeneruj "executive_takeaway" spełniający KONTRAKT.

KONTRAKT (MUST)
1) Dokładnie 2 zdania po polsku.
2) Każdy blok: minimum 2 liczby z podanych stats (np. mediana, IQR, p95, % outlierów, p95/mediana).
3) Zdanie 1 = interpretacja i ryzyko/znaczenie (co to oznacza biznesowo) + liczby.
4) Zdanie 2 = decyzja / rekomendacja (co zrobić dalej) + liczby/progi.
5) Zero "ten wykres pokazuje" / "KDE/ECDF" / meta-komentarzy. Nie powtarzaj Guidance.
6) Nie zmyślaj liczb – używaj WYŁĄCZNIE wartości ze stats.

FORMAT WYJŚCIA (MUST)
Zwróć JEDEN JSON (bez markdown), dokładnie w tym schemacie:
{
  "results": {
     "<label>": "<2-zdaniowy executive takeaway>",
     ...
  }
}

KONTEKST (pytanie użytkownika)
{question}

BLOKI (lista obiektów)
Każdy obiekt ma: label, title, desc, stats.
{blocks_json}
"""

PROMPT_V2 = r"""
You are a McKinsey/Bain-grade analytics partner. Produce an Executive Takeaway for each chart block.

CONTEXT
- User question: {question}
- Branch intent: {intent}
- We render multiple blocks; for each block we provide:
  - label (stable key)
  - title (human title)
  - desc (what the chart shows)
  - stats (minimal numeric facts you MUST use)

TASK
Return a JSON object with EXACTLY this shape:
{{
  "results": {{
    "<label>": "<TAKEAWAY>",
    ...
  }}
}}

TAKEAWAY RULES (HARD)
1) Exactly 2 sentences.
2) Sentence 1 = FACT + NUMBERS from stats (at least ONE number).
3) Sentence 2 = IMPLICATION + CLEAR DECISION / RECOMMENDATION (what to do next).
4) No hedging, no fluff, no generic phrases.
5) Do NOT invent numbers. Use ONLY numbers present in stats.
6) If stats for a block have no usable numbers, output:
   "Brak wystarczających danych liczbowych w statystykach. Uzupełnij statystyki i powtórz analizę."
   (still 2 sentences, second sentence must be a decision).

STYLE
- Polish language.
- Crisp, executive tone.
- Keep each takeaway <= 220 characters if possible.

BLOCKS
{blocks_json}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Walidacja (nie zabijamy dobrych odpowiedzi)
# ─────────────────────────────────────────────────────────────────────────────

_BANNED = re.compile(r"\b(ten wykres|wykres pokazuje|histogram pokazuje|violin pokazuje|rug pokazuje|ecdf|kde)\b", re.IGNORECASE)

_ACTION_HINT = re.compile(
    r"\b(ustaw|ustali[cć]|wdroż|rozważ|rekomenduj|zastosuj|segmentuj|podziel|zweryfikuj|potwierdź|monitoruj|porównaj|raportuj|operuj)\b",
    re.IGNORECASE,
)


def _sentences_pl(txt: str) -> List[str]:
    # dzielimy po . ! ? (ale ignorujemy skróty typu "np.") – prosta heurystyka
    raw = re.split(r"(?<=[.!?])\s+", txt.strip())
    raw = [s.strip() for s in raw if s.strip()]
    return raw


def _has_number(txt: str) -> bool:
    return bool(re.search(r"\d", txt))


def _validate_one(txt: Any) -> Optional[str]:
    # Accept strings only
    if not isinstance(txt, str):
        return None
    t = txt.strip()
    if not t:
        return None
    if len(t) > 480:  # allow some slack; we will compress later
        return None
    if _BANNED.search(t):
        return None

    sents = _sentences_pl(t)

    # We REQUIRE 2 sentences in UI, but we try to auto-fix common "1 sentence" outputs.
    if len(sents) == 1:
        # must contain at least one number to qualify as FACT+NUMBERS
        if len(re.findall(r"\d", t)) < 2:
            return None
        first = sents[0].rstrip()
        if not first.endswith((".", "!", "?")):
            first += "."
        second = "Rekomendacja: ustaw progi/KPI i monitoruj odchylenia."
        t2 = f"{first} {second}"
        if len(t2) <= 320 and not _BANNED.search(t2):
            return t2
        return None

    if len(sents) < 2:
        return None

    # If model produced 3+ sentences, we compress into exactly 2.
    first = sents[0].rstrip()
    second = " ".join(sents[1:]).strip()

    if not first.endswith((".", "!", "?")):
        first += "."
    if not second.endswith((".", "!", "?")):
        second += "."

    t2 = f"{first} {second}"

    # numbers: at least 2 digits anywhere (e.g. 60%, 120 000)
    if len(re.findall(r"\d", t2)) < 2:
        return None

    # decision: require at least one action verb/phrase OR allow explicit KPI keywords.
    if not _ACTION_HINT.search(t2):
        if not re.search(r"\b(prog|p95|p99|percentyl|limit|SLA|KPI)\b", t2, re.IGNORECASE):
            # auto-fix: replace the second sentence with a clear decision
            second = "Rekomendacja: ustaw progi/KPI i monitoruj odchylenia."
            t2 = f"{first} {second}"

    # final hard length gate (UI friendly)
    if len(t2) > 320:
        return None

    return t2


def _get_llm_callable(ctx: Dict[str, Any]):
    """Zwraca callable(prompt:str)->str lub None.

    Obsługujemy kilka wariantów, żeby nie być zależnym od jednej integracji.
    """
    if not isinstance(ctx, dict):
        return None

    # 1) ctx['llm_text'] – preferowany: funkcja przyjmująca prompt i zwracająca tekst
    fn = ctx.get("llm_text")
    if callable(fn):
        return fn

    # 2) ctx['llm'] z metodą .complete/.invoke
    llm = ctx.get("llm")
    if llm is not None:
        if hasattr(llm, "complete") and callable(getattr(llm, "complete")):
            return lambda prompt: llm.complete(prompt)
        if hasattr(llm, "invoke") and callable(getattr(llm, "invoke")):
            return lambda prompt: llm.invoke(prompt)

    # 3) ctx['openai_client'] – minimalny wrapper
    client = ctx.get("openai_client")
    model = ctx.get("openai_model") or ctx.get("model")
    if client is not None and model:
        # próbujemy: client.responses.create(...) -> output_text
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            def _call(prompt: str) -> str:
                resp = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                )
                return getattr(resp, "output_text", "") or ""
            return _call

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_exec_takeaways_llm(*, ctx: Dict[str, Any], intent: str, blocks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Zwraca mapę label -> executive_takeaway (po walidacji).

    Jeśli LLM nie jest dostępny lub odpowiedź nie przejdzie walidacji, zwraca {}.
    """
    def _record_debug(where: str, **payload: Any) -> None:
        try:
            entry = {"where": where, **payload}
            log = ctx.setdefault("_exec_takeaway_llm_debug", [])
            if isinstance(log, list):
                log.append(entry)
            ctx["_exec_takeaway_llm_last_error"] = entry
        except Exception:
            pass

    try:
        llm_call = _get_llm_callable(ctx)
        if llm_call is None:
            _record_debug("llm_not_configured", intent=intent)
            return {}

        question = (ctx.get("question") or ctx.get("user_question") or "").strip()

        # Minimalny payload do promptu (bez ciężkich obiektów)
        slim_blocks: List[Dict[str, Any]] = []
        for b in (blocks or []):
            if not isinstance(b, dict):
                continue
            slim_blocks.append(
                {
                    "label": (b.get("label") or "").strip(),
                    "title": (b.get("title") or "").strip(),
                    "desc": (b.get("desc") or "").strip(),
                    "stats": b.get("stats") or {},
                }
            )

        try:
            prompt = PROMPT_V2.format(
                question=question,
                intent=intent,
                blocks_json=json.dumps(slim_blocks, ensure_ascii=False),
            )
        except Exception as e:
            _record_debug(
                "prompt_format_error",
                intent=intent,
                blocks_count=len(slim_blocks),
                error=f"{type(e).__name__}: {e}",
            )
            return {}

        raw = llm_call(prompt)
        if not isinstance(raw, str):
            raw = str(raw or "")
        raw = raw.strip()
        if not raw:
            _record_debug("llm_empty_text", intent=intent, blocks_count=len(slim_blocks))
            return {}

        # próbujemy wyciągnąć JSON (czasem model dopisze tekst)
        m = re.search(r"\{[\s\S]*\}\s*$", raw)
        if m:
            raw_json = m.group(0)
        else:
            raw_json = raw

        obj = json.loads(raw_json)
        results = (obj or {}).get("results")
        if not isinstance(results, dict):
            _record_debug("results_not_dict", intent=intent, raw_preview=raw[:240])
            return {}

        out: Dict[str, str] = {}
        for k, v in results.items():
            key = (k or "").strip()
            if not key:
                continue
            vv = _validate_one(v)
            if vv:
                out[key] = vv

        if not out:
            _record_debug(
                "validated_results_empty",
                intent=intent,
                blocks_count=len(slim_blocks),
                raw_preview=raw[:240],
            )
        return out

    except Exception as e:
        _record_debug("unexpected_error", intent=intent, error=f"{type(e).__name__}: {e}")
        # NO-REGRESSION – cisza, UI spada na fallback
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compat alias (HOTFIX)
#
# Starsze wersje gałęzi mogły importować:
#   from data_chat_core.exec_takeaway_llm import llm_exec_takeaway
# W tej wersji publiczną funkcją jest `get_exec_takeaways_llm`. Alias poniżej
# zapobiega crashowi aplikacji po podmianie plików.

def llm_exec_takeaway(*args, **kwargs):
    """Alias zgodności wstecznej.

    Obsługiwane wywołania:
      - llm_exec_takeaway(ctx=..., intent=..., blocks=[...])
      - llm_exec_takeaway(...): dowolne nadmiarowe argi są ignorowane

    Zwraca dict: {label: executive_takeaway}.
    """
    # Prefer jawne kwargs
    ctx = kwargs.get("ctx")
    intent = kwargs.get("intent") or kwargs.get("intent_name") or ""
    blocks = kwargs.get("blocks") or kwargs.get("insight_blocks") or []

    # Jeśli ktoś podał positional args w starej kolejności – spróbujmy je zmapować
    if ctx is None and len(args) >= 1:
        ctx = args[0]
    if (not intent) and len(args) >= 2:
        intent = args[1]
    if (not blocks) and len(args) >= 3:
        blocks = args[2]

    if not isinstance(ctx, dict):
        return {}
    if not isinstance(blocks, list):
        return {}

    return get_exec_takeaways_llm(ctx=ctx, intent=str(intent), blocks=blocks)
