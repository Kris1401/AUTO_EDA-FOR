# app/core/pii.py
from __future__ import annotations
import re
import pandas as pd
from typing import Dict, Tuple, List

# proste reguły PII (rozszerzymy później)
RE_EMAIL = re.compile(r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+\.[A-Z]{{2,}})", re.IGNORECASE)
RE_PHONE = re.compile(r"\+?\d[\d\-\s]{{7,}}\d")
RE_PESEL = re.compile(r"(?<!\d)(\d{{11}})(?!\d)")
RE_NIP = re.compile(r"(?<!\d)(\d{{3}}[\-\s]?\d{{3}}[\-\s]?\d{{2}}[\-\s]?\d{{2}})(?!\d)")

PII_COL_GUESSES = {"email","e-mail","mail","telefon","phone","pesel","nip","adres","address"}

def _mask_text(s) -> str:
    if pd.isna(s):
        return s
    if not isinstance(s, str):
        s = str(s)
    s = RE_EMAIL.sub(lambda m: m.group(1)[0]+"***@***."+m.group(2).split(".")[-1], s)
    s = RE_PHONE.sub(lambda m: "***"+m.group(0)[-3:], s)
    s = RE_PESEL.sub(lambda m: "*******"+m.group(1)[-4:], s)
    s = RE_NIP.sub(lambda m: "***-***-**"+m.group(1)[-2:], s)
    return s

def _mask_cell(val):
    if pd.isna(val):
        return val
    if isinstance(val, (int,float)):
        s = str(val)
        if len(s) >= 6:
            return "***" + s[-2:]
        return val
    return _mask_text(str(val))

def mask_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """Zwraca (df_masked, raport_liczników)."""
    masked = df.copy()
    report: Dict[str,int] = {}
    # 1) maskowanie po nazwach kolumn
    for c in masked.columns:
        if str(c).strip().lower() in PII_COL_GUESSES:
            report[c] = masked[c].notna().sum()
            masked[c] = masked[c].map(_mask_cell)
    # 2) skan tekstowy reszty (delikatnie – tylko object/string)
    for c in masked.select_dtypes(include=["object","string"]).columns:
        before = masked[c].copy()
        after = before.map(_mask_text)
        n = (before.astype(str).values != after.astype(str).values).sum()
        if n:
            report[c] = report.get(c,0) + n
            masked[c] = after
    return masked, report
