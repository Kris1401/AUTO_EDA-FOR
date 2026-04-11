# app/data_chat_core/llm.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI


def llm_fn(
    *,
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    # --- compatibility params (ignored unless used for overrides) ---
    ctx: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """
    Minimalny, wspólny wrapper LLM (OpenAI SDK 1.x/2.x style).
    Zwraca dict z JSON (parsowane z message.content).

    Kompatybilność:
    - Przyjmujemy (ctx, payload, **_) bo różne moduły CORE/branches mogą to przekazywać.
    - Jeśli ctx zawiera override'y (openai_model/openai_temperature), używamy ich.
    """
    _ctx = ctx or {}

    # allow runtime overrides from ctx (bez wpływu na obecny UI/kontrakt)
    model = str(_ctx.get("openai_model") or model)
    try:
        temperature = float(_ctx.get("openai_temperature") or temperature)
    except Exception:
        pass

    client = OpenAI()

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if response_format is not None:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()

    # LLM ma zwrócić JSON — parsujemy defensywnie
    try:
        return json.loads(text) if text else {}
    except Exception:
        return {}
