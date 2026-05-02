# app/data_chat_core/llm.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

try:
    import streamlit as st
except Exception:
    st = None  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


def _get_env_or_secret(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return str(value).strip()
    if st is not None:
        try:
            value = st.secrets.get(name, "")  # type: ignore[attr-defined]
        except Exception:
            value = ""
        return str(value).strip() if value is not None else ""
    return ""


def _chat_completion_rest(
    *,
    api_key: str,
    model: str,
    temperature: float,
    messages: List[Dict[str, Any]],
    response_format: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> str:
    body: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    if response_format is not None:
        body["response_format"] = response_format

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout or 90,
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


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

    api_key = _get_env_or_secret("OPENAI_API_KEY")
    if not api_key:
        return {}

    text = ""
    if OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
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
        except Exception:
            text = ""

    if not text:
        try:
            text = _chat_completion_rest(
                api_key=api_key,
                model=model,
                temperature=temperature,
                messages=messages,
                response_format=response_format,
                timeout=timeout,
            )
        except Exception:
            return {}

    # LLM ma zwrócić JSON — parsujemy defensywnie
    try:
        return json.loads(text) if text else {}
    except Exception:
        return {}
