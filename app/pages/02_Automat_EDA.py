# app/pages/02_Automat_EDA.py
from __future__ import annotations

import os
import io
import json
import math
import base64
from datetime import datetime
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
# --- wspólna paleta barw ---
PALETTE_MAIN = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# --- Stylizacja przycisków Streamlit (tylko secondary) ---
st.markdown("""
<style>
/* 🎨 Styl TYLKO dla przycisków typu 'secondary'
   (np. '🔎 Uruchom/odśwież weryfikację winsoryzacji') */

/* obsługa obu wariantów data-testid z różnych wersji Streamlita */
button[data-testid="baseButton-secondary"],
button[data-testid="stBaseButton-secondary"] {
  background: #fff4e5 !important;      /* blady pomarańcz, nawiązanie do koloru 'po' (#ff7f0e) */
  color: #5c3b00 !important;           /* ciemny brąz – dobry kontrast */
  border: 1px solid #ffb74d !important;
  box-shadow: none !important;
}

button[data-testid="baseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
  background: #ffedd5 !important;      /* lekko ciemniej na hover */
}
</style>
""", unsafe_allow_html=True)

FACET_CHART_WIDTH = 980  # szerokość jednego panelu w układzie facet

def _scale_for_categories(cats: List[str]) -> alt.Scale:
    """Zwraca Scale z domeną = kolejność kategorii i rangą z naszej palety (powiela gdy potrzeba)."""
    if not cats:
        return alt.Scale(range=PALETTE_MAIN)
    if len(cats) <= len(PALETTE_MAIN):
        rng = PALETTE_MAIN[: len(cats)]
    else:
        k = int(np.ceil(len(cats) / len(PALETTE_MAIN)))
        rng = (PALETTE_MAIN * k)[: len(cats)]
    return alt.Scale(domain=cats, range=rng)

import requests
import hashlib, re
import openai as _openai
from uuid import uuid4
from langfuse import Langfuse
from langfuse.decorators import observe  # opcjonalny dekorator

from dotenv import load_dotenv

# Wczytaj .env (OPENAI_API_KEY, ELEVENLABS_API_KEY, VOICE_* itp.)
load_dotenv(override=True)

# Altair: nie tniemy >5k wierszy
alt.data_transformers.disable_max_rows()

try:
    from langfuse.openai import OpenAI as LFOpenAI  # wrapper kompatybilny z openai.OpenAI
except Exception:
    LFOpenAI = None

@st.cache_resource(show_spinner=False)
def get_langfuse():
    try:
        # Zwraca obiekt Langfuse lub None, jeśli brak kluczy/połączenia
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            return None
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            release=os.getenv("LANGFUSE_RELEASE", "app-eda@dev"),
            sdk_integration="streamlit"
        )
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_lf_openai_client():
    # Zwraca klienta OpenAI z automatycznym tracingiem Langfuse (dla Chat)
    try:
        if LFOpenAI is None:
            return None
        return LFOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    except Exception:
        return None



# --- OpenAI TTS: listy wyboru ---
# --- OpenAI: głosy podzielone wg płci (praktyczny podział) ---
OPENAI_VOICES = {
    "female": ["nova", "shimmer", "coral", "ballad", "sage", "marin"],
    "male":   ["alloy", "echo", "fable", "onyx", "verse", "ash", "cedar"],
}
DEFAULT_FEMALE_VOICE = "shimmer"
DEFAULT_MALE_VOICE   = "verse"

# jeśli kiedyś dodasz inne modele TTS, dopisz je tutaj
OPENAI_TTS_MODELS = [os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")]


# ==== AUDIO AUTOPLAY (Streamlit nie autoodtwarza st.audio) ====
def _autoplay_audio(audio_bytes: bytes, mime: str = "audio/mpeg"):
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    html = f"""
    <audio autoplay>
      <source src="data:{mime};base64,{b64}" type="{mime}">
    </audio>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==== TL;DR fallback (deterministyczny) ====
def _make_eda_summary_text(
    source_name: str,
    readiness_score: int,
    duplicates_count: int,
    global_missing_pct: float,
    auto_drop_candidates: list[str],
    prep_report: dict | None,
) -> str:
    drops = ", ".join(auto_drop_candidates) if auto_drop_candidates else "brak"
    wins = []
    if prep_report:
        if prep_report.get("duplicates_removed", 0) > 0:
            wins.append(f"usunięto duplikaty: {prep_report['duplicates_removed']}")
        if prep_report.get("outlier_flags"):
            wins.append(f"zaflagowano outliery: {len(prep_report['outlier_flags'])} kolumn")
    wins_txt = ("; " + "; ".join(wins)) if wins else ""
    return (
        f"Jakość danych w zbiorze „{source_name}” oceniamy na {readiness_score}/100. "
        f"Braki globalne wynoszą ok. {global_missing_pct:.1f}%, a liczba pełnych duplikatów to {duplicates_count}. "
        f"Kandydaci do wyłączenia: {drops}. Dane po czyszczeniu są gotowe do trenowania{wins_txt}."
    )

# =====================  UTILITIES  =======================

def _load_latest_dataset() -> Tuple[pd.DataFrame | None, dict | None, str | None]:
    """Wczytuje surowy zbiór z poprzedniego kroku."""
    latest_info = st.session_state.get("latest_artifacts")
    if latest_info is None:
        return None, None, (
            "Brak gotowych danych w pamięci aplikacji. "
            "Przejdź do zakładki 'Analiza Danych' i kliknij "
            "'Przelicz teraz (pełny zbiór)'."
        )
    csv_path = latest_info.get("csv_path")
    if not csv_path or not os.path.exists(csv_path):
        return None, latest_info, (
            f"Nie mogę znaleźć pliku z danymi: {csv_path!r}. "
            "Najpierw przejdź do 'Analiza Danych' i kliknij "
            "'Przelicz teraz (pełny zbiór)'."
        )
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, latest_info, f"Nie udało się wczytać CSV: {e}"
    return df, latest_info, None


def _calc_global_missing_pct(df: pd.DataFrame) -> float:
    """Procent braków w całym dataframe, licząc NaNy + puste stringi."""
    if df.shape[0] == 0 or df.shape[1] == 0:
        return 0.0
    tmp = df.copy()
    obj_cols = tmp.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        tmp[c] = tmp[c].replace("", np.nan)
    total_cells = tmp.shape[0] * tmp.shape[1]
    missing_cells = tmp.isna().sum().sum()
    return round(100.0 * (missing_cells / max(total_cells, 1)), 2)


def _infer_logical_type(series: pd.Series) -> str:
    """
    Heurystyka 'logicznego' typu:
      numeric / datetime / id_like / text_long / categorical
    """
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    sample = series.dropna().astype(str).head(200)
    if len(sample) > 0:
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        if parsed.notna().mean() > 0.9:
            return "datetime"

    nunique = series.nunique(dropna=True)
    ratio_unique = nunique / max(len(series), 1)
    if ratio_unique > 0.9:
        return "id_like"

    avg_len = (series.dropna().astype(str).str.len().mean()
               if len(series.dropna()) > 0 else 0)
    if avg_len and avg_len > 50:
        return "text_long"

    return "categorical"


def _try_detect_numeric_from_object(series: pd.Series) -> bool:
    """Czy obiektowa kolumna wygląda na liczbową po naszych regułach specjalnych?"""
    if series.dtype != object:
        return False
    s = series.dropna().astype(str)
    if s.empty:
        return False

    # spróbuj parsery specjalne
    parsed = _coerce_to_numeric_special(s)
    if parsed.notna().mean() > 0.9:
        return True

    return False

def _augment_numeric_derivatives_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dla potrzeb podglądu/analizy (Sekcja 2–6) dorzuć kolumny liczbowe pochodne:
    - object: HH:MM:SS/MM:SS -> <col>_sec, min/km -> <col>_sec_per_km, % -> <col>_pct, km/h -> <col>_kmh, waluty/liczby -> <col>_num
    - datetime: sekundy od północy -> <col>_sec
    Nie dotyka oryginałów; zwraca kopię df z dodatkowymi kolumnami.
    """
    out = df.copy()
    for col in list(df.columns):
        s = df[col]

        # --- object: użyj Twoich parserów specjalnych ---
        if s.dtype == object:
            ss = s.astype(str)
            sec = ss.map(_parse_duration_to_seconds)
            if pd.notna(sec).sum() >= max(5, int(0.6 * len(ss.dropna()))):
                out[f"{col}_sec"] = pd.to_numeric(sec, errors="coerce")
                continue

            pace = ss.map(_parse_pace_min_per_km_to_sec_per_km)
            if pd.notna(pace).sum() >= max(5, int(0.6 * len(ss.dropna()))):
                out[f"{col}_sec_per_km"] = pd.to_numeric(pace, errors="coerce")
                continue

            perc = ss.map(_parse_percentage)
            if pd.notna(perc).sum() >= max(5, int(0.6 * len(ss.dropna()))):
                out[f"{col}_pct"] = pd.to_numeric(perc, errors="coerce")
                continue

            kmh = ss.map(_parse_kmh)
            if pd.notna(kmh).sum() >= max(5, int(0.6 * len(ss.dropna()))):
                out[f"{col}_kmh"] = pd.to_numeric(kmh, errors="coerce")
                continue

            cur = ss.map(_parse_currency_or_plain_number)
            if pd.notna(cur).sum() >= max(5, int(0.6 * len(ss.dropna()))):
                out[f"{col}_num"] = pd.to_numeric(cur, errors="coerce")
                continue

        # --- datetime: zrób sekundę od północy ---
        if pd.api.types.is_datetime64_any_dtype(s):
            try:
                sec = (s.dt.hour * 3600 + s.dt.minute * 60 + s.dt.second).astype("float")
                # jeśli to faktycznie wygląda na czas (ma zmienność), dodaj kolumnę
                if pd.notna(sec).sum() >= max(5, int(0.6 * len(s.dropna()))):
                    out[f"{col}_sec"] = sec
            except Exception:
                pass

    return out

def _try_detect_datetime_from_object(series: pd.Series) -> bool:
    if series.dtype == object:
        sample = series.dropna().astype(str).head(200)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        return parsed.notna().mean() > 0.9
    return False

# ---------- Parsowanie specjalnych formatów: H:M:S, min/km, %, waluty, jednostki ----------

_DURATION_RE = re.compile(r"^\s*(\d{1,2}):([0-5]?\d):([0-5]?\d)\s*$")   # HH:MM:SS
_MMSS_RE     = re.compile(r"^\s*([0-5]?\d):([0-5]?\d)\s*$")             # MM:SS (np. 4:35)
_PERCENT_RE  = re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*%\s*$")
CURR_SIGNS   = "€$£złPLN"  # można rozszerzyć
_UNIT_KMH_RE = re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*(?:km/?h|kmh)\s*$", re.I)
_UNIT_MINKM  = re.compile(r"^\s*([0-5]?\d):([0-5]?\d)\s*(?:min/?km|min/km)\s*$", re.I)

def _parse_duration_to_seconds(val: str) -> float | None:
    """ 'HH:MM:SS' -> sekundy, 'MM:SS' -> sekundy """
    if not isinstance(val, str):
        return None
    m = _DURATION_RE.match(val)
    if m:
        h, mi, s = map(int, m.groups())
        return float(h*3600 + mi*60 + s)
    m = _MMSS_RE.match(val)
    if m:
        mi, s = map(int, m.groups())
        return float(mi*60 + s)
    return None

def _parse_percentage(val: str) -> float | None:
    if not isinstance(val, str):
        return None
    m = _PERCENT_RE.match(val)
    if m:
        v = m.group(1).replace(",", ".")
        try:
            return float(v) / 100.0
        except Exception:
            return None
    return None

def _parse_currency_or_plain_number(val: str) -> float | None:
    """ '1 234,56 zł' / '$1,234.56' / '1 234' -> float """
    if not isinstance(val, str):
        return None
    s = val.strip()
    # usuń separatory tysięcy (spacje, niełamliwe, kropki) — ale zostaw kropkę dziesiętną
    s = s.replace("\u202f", " ").replace("\xa0", " ")
    # usuń waluty/słowa
    for token in CURR_SIGNS.split():
        s = s.replace(token, "")
    s = re.sub(r"[€$£]|zł|PLN", "", s, flags=re.I)
    s = s.strip()

    # polski przecinek na kropkę
    s = s.replace(" ", "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    # jeśli wciąż są literki → nie parsujemy
    if re.search(r"[^\d\.\-+eE]", s):
        return None
    try:
        return float(s)
    except Exception:
        return None

def _parse_kmh(val: str) -> float | None:
    if not isinstance(val, str):
        return None
    m = _UNIT_KMH_RE.match(val)
    if not m:
        return None
    v = m.group(1).replace(",", ".")
    try:
        return float(v)
    except Exception:
        return None

def _parse_pace_min_per_km_to_sec_per_km(val: str) -> float | None:
    """ '4:35 min/km' -> sekundy na km (275.0) """
    if not isinstance(val, str):
        return None
    m = _UNIT_MINKM.match(val) or _MMSS_RE.match(val)
    if not m:
        return None
    mi, s = map(int, m.groups())
    return float(mi*60 + s)

def _coerce_to_numeric_special(series: pd.Series) -> pd.Series:
    """Najpierw spróbuj duration, % , min/km, km/h, waluty/liczby. Zwraca numeric z NaN dla nieparsowalnych."""
    s = series.astype(str)

    # 1) HH:MM:SS / MM:SS
    parsed = s.map(_parse_duration_to_seconds)
    if pd.notna(parsed).sum() >= max(5, int(0.6*len(s.dropna()))):
        return pd.to_numeric(parsed, errors="coerce")

    # 2) pace 'min/km' lub MM:SS (jeśli w kolumnie są takie stringi)
    pace = s.map(_parse_pace_min_per_km_to_sec_per_km)
    if pd.notna(pace).sum() >= max(5, int(0.6*len(s.dropna()))):
        return pd.to_numeric(pace, errors="coerce")

    # 3) procenty
    perc = s.map(_parse_percentage)
    if pd.notna(perc).sum() >= max(5, int(0.6*len(s.dropna()))):
        return pd.to_numeric(perc, errors="coerce")

    # 4) km/h
    kmh = s.map(_parse_kmh)
    if pd.notna(kmh).sum() >= max(5, int(0.6*len(s.dropna()))):
        return pd.to_numeric(kmh, errors="coerce")

    # 5) waluty/„gołe” liczby z tysiącami itp.
    cur = s.map(_parse_currency_or_plain_number)
    if pd.notna(cur).sum() >= max(5, int(0.6*len(s.dropna()))):
        return pd.to_numeric(cur, errors="coerce")

    # fallback: oryginalne zachowanie (przecinek→kropka, spacje)
    s_norm = (series.dropna().astype(str)
              .str.replace(",", ".", regex=False)
              .str.replace(" ", "", regex=False))
    return pd.to_numeric(s_norm, errors="coerce")

def _analyze_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Raport jakości kolumn i heurystyki."""
    rows = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        dtype_raw = str(s.dtype)
        logical = _infer_logical_type(s)

        s_na_test = s.copy()
        if s_na_test.dtype == object:
            s_na_test = s_na_test.replace("", np.nan)

        null_cnt = int(s_na_test.isna().sum())
        null_pct = (null_cnt / max(n_rows, 1)) * 100.0 if n_rows else 0.0
        nunique = int(s.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(s):
            col_min = s.min(skipna=True)
            col_mean = s.mean(skipna=True)
            col_max = s.max(skipna=True)
            col_std = s.std(skipna=True)
        else:
            col_min = col_mean = col_max = col_std = None

        drop_candidate = False
        convert_numeric = False
        convert_datetime = False
        long_text_flag = False

        notes = []
        flag_icons = []

        if null_pct > 30:
            notes.append("Dużo braków (>30%).")
        if null_pct > 80 and logical != "numeric":
            notes.append("Prawie cała pusta (>80%).")
            drop_candidate = True
            flag_icons.append("🗑")
        if nunique <= 1:
            notes.append("Kolumna stała / bez zmienności.")
            drop_candidate = True
            if "🗑" not in flag_icons:
                flag_icons.append("🗑")
        if logical == "id_like":
            notes.append("Wygląda jak identyfikator (prawie każda wartość inna).")
            drop_candidate = True
            if "🗑" not in flag_icons:
                flag_icons.append("🗑")
        if _try_detect_numeric_from_object(s):
            notes.append("Wygląda na liczbę, ale jest stringiem → konwersja do float.")
            convert_numeric = True
            flag_icons.append("🔢")
        if _try_detect_datetime_from_object(s):
            notes.append("Wygląda na datę → konwersja do datetime i rozbicie (rok/miesiąc/dzień_tyg).")
            convert_datetime = True
            flag_icons.append("📅")
        if logical == "text_long":
            notes.append("Długi opis tekstowy / wolny tekst → raczej NLP niż klasyczny ML.")
            long_text_flag = True
            flag_icons.append("📝")

        rows.append(
            {
                "kolumna": col,
                "dtype_raw": dtype_raw,
                "logical_type": logical,
                "null_cnt": int(null_cnt),
                "null_pct": round(null_pct, 2),
                "n_unique": nunique,
                "min": col_min,
                "mean": col_mean,
                "max": col_max,
                "std": col_std,
                "drop_candidate": drop_candidate,
                "convert_numeric": convert_numeric,
                "convert_datetime": convert_datetime,
                "text_long_flag": long_text_flag,
                "flags": " ".join(flag_icons),
                "comment": " ".join(notes),
            }
        )

    info_df = pd.DataFrame(rows)
    high_null_cols = info_df.loc[info_df["null_pct"] > 30, "kolumna"].tolist()
    return info_df, high_null_cols


def _build_correlation_report(
    df: pd.DataFrame,
    info_df: pd.DataFrame,
    threshold: float = 0.9,
) -> Tuple[alt.Chart | None, List[Dict[str, Any]], List[str]]:
    """Heatmapa i lista par korelacji dla kolumn numerycznych."""
    # Guard: tylko sensowne kolumny numeryczne
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].notna().sum() > 1]
    if len(numeric_cols) < 2:
        return None, [], []

    corr_mat = df[numeric_cols].corr(method="pearson").fillna(0.0)
    if corr_mat.empty:
        return None, [], []

    corr_melt = (
        corr_mat.reset_index()
        .melt("index", var_name="col2", value_name="corr")
        .rename(columns={"index": "col1"})
    )

    heatmap = (
        alt.Chart(corr_melt)
        .mark_rect()
        .encode(
            x=alt.X("col1:N", title="", sort=numeric_cols),
            y=alt.Y("col2:N", title="", sort=numeric_cols),
            color=alt.Color(
                "corr:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1], domainMid=0),
                title="Pearson r",
            ),
            tooltip=[
                alt.Tooltip("col1:N", title="kolumna 1"),
                alt.Tooltip("col2:N", title="kolumna 2"),
                alt.Tooltip("corr:Q", title="r", format=".3f"),
            ],
        )
        .properties(width=500, height=500)
    )

    pairs = []
    high_corr_drop_candidates = []
    null_pct_map = info_df.set_index("kolumna")["null_pct"].to_dict()

    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            r_val = float(corr_mat.loc[c1, c2])
            n1 = null_pct_map.get(c1, 0.0)
            n2 = null_pct_map.get(c2, 0.0)
            suggest_drop = c1 if n1 > n2 else c2
            item = {"col1": c1, "col2": c2, "corr": r_val, "suggest_drop": suggest_drop}
            pairs.append(item)
            if abs(r_val) >= threshold:
                high_corr_drop_candidates.append(suggest_drop)

    high_corr_drop_candidates = sorted(list(set(high_corr_drop_candidates)))
    pairs_sorted = sorted(pairs, key=lambda d: abs(d["corr"]), reverse=True)
    return heatmap, pairs_sorted, high_corr_drop_candidates


def _detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    dups_mask = df.duplicated(keep="first")
    duplicates_count = int(dups_mask.sum())
    duplicates_pct = ((100.0 * duplicates_count / len(df)) if len(df) else 0.0)
    return {"duplicates_count": duplicates_count, "duplicates_pct": round(duplicates_pct, 2)}

# ---------- Pomocnicze formaty/liczby ----------

def _fmt_int(x: Any) -> str:
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return str(x)

def _num(x: Any) -> Any:
    try:
        xf = float(x)
        return round(xf, 4) if np.isfinite(xf) else None
    except Exception:
        return None

def slice_df(df: pd.DataFrame, where: dict[str, Any]) -> pd.DataFrame:
    """Prosty builder: where={'Survived': 1, 'Sex': 'female'} → zwróci pocięty df."""
    mask = pd.Series(True, index=df.index)
    for col, val in (where or {}).items():
        if col not in df.columns:
            continue
        if isinstance(val, (list, tuple, set)):
            mask &= df[col].isin(list(val))
        else:
            mask &= (df[col] == val)
    return df[mask].copy()

def hist_kde(df: pd.DataFrame, value_col: str, maxbins: int = 30) -> alt.Chart:
    """Histogram (liczebność) + KDE na jednym układzie."""
    use = pd.DataFrame({value_col: pd.to_numeric(df[value_col], errors="coerce")}).dropna()
    if use.empty:
        return alt.Chart(pd.DataFrame({"msg":["Brak danych"]})).mark_text(size=14).encode(text="msg")
    hist = (
        alt.Chart(use).mark_bar(opacity=0.85)
        .encode(
            x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=maxbins), title=value_col),
            y=alt.Y("count():Q", title="Liczebność"),
            tooltip=[alt.Tooltip("count():Q", title="liczebność")]
        ).properties(height=220)
    )
    kde = (
        alt.Chart(use).transform_density(value_col, as_=[value_col, "density"])
        .mark_line(size=2, opacity=0.9)
        .encode(x=f"{value_col}:Q", y=alt.Y("density:Q", title="Gęstość"))
        .properties(height=220)
    )
    return alt.layer(hist, kde).resolve_scale(y="independent")

def violin_or_box(df: pd.DataFrame, value_col: str, cat_label: str, show: str = "violin") -> alt.Chart:
    """
    „Violin” dla jednej kategorii (cat_label) lub klasyczny boxplot.
    cat_label to etykieta na osi (np. 'female • survived').
    """
    use = pd.DataFrame({value_col: pd.to_numeric(df[value_col], errors="coerce")}).dropna()
    if use.empty:
        return alt.Chart(pd.DataFrame({"msg":["Brak danych"]})).mark_text(size=14).encode(text="msg")

    use["label"] = cat_label

    if show == "box":
        return (
            alt.Chart(use).mark_boxplot()
            .encode(x=alt.X("label:N", title=""), y=alt.Y(f"{value_col}:Q", title=value_col))
            .properties(height=180)
        )

    # violin (lustrzane density po osi X)
    dens = (
        alt.Chart(use)
        .transform_density(value_col, groupby=["label"], as_=[value_col, "density"])
        .properties(height=180)
    )
    left = dens.mark_area(opacity=0.85).encode(
        x=alt.X("density:Q", title=None, axis=None),
        y=alt.Y(f"{value_col}:Q", title=value_col),
        row=None,
        color=alt.Color("label:N", legend=None),
    )
    right = dens.mark_area(opacity=0.85).encode(
        x=alt.X("calculate(-datum.density):Q", title=None, axis=None),
        y=alt.Y(f"{value_col}:Q", title=value_col),
        color=alt.Color("label:N", legend=None),
    )
    # oś kategorii jako kolumna z jedną etykietą
    base = alt.hconcat(left, right)
    return base

# ---------- Czynniki eliminujące (Sekcja 5) ----------

def _col_factors(df: pd.DataFrame, info_df: pd.DataFrame, col: str) -> Dict[str, float]:
    """Wkład czynników: missing, id_like, engineered, variance_inv, outlier_ratio, cardinality, skew_abs."""
    # missing %
    try:
        missing_pct = float(info_df.loc[info_df["kolumna"] == col, "null_pct"].fillna(0).iloc[0]) / 100.0
    except Exception:
        missing_pct = 0.0

    row = info_df.set_index("kolumna").to_dict(orient="index").get(col, {})
    id_like = 1.0 if row.get("logical_type") == "id_like" else 0.0
    engineered = 1.0 if (col.endswith("_year") or col.endswith("_month") or col.endswith("_dow") or col.startswith("is_outlier_")) else 0.0

    if pd.api.types.is_numeric_dtype(df[col]):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        var = float(np.var(s)) if len(s) > 1 else 0.0
        # outlier ratio
        if len(s) > 3:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                outlier_ratio = 0.0
            else:
                lo, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_ratio = float(((s < lo) | (s > up)).mean())
        else:
            outlier_ratio = 0.0
        skew = float(pd.Series(s).skew()) if len(s) > 2 else 0.0
        try:
            s_num = pd.to_numeric(df[col], errors="coerce")
            rng = float(s_num.max() - s_num.min())
            variance_inv = 1.0 - min(1.0, (var / (rng**2 + 1e-9)))
            variance_inv = max(0.0, variance_inv)
        except Exception:
            variance_inv = 0.0
    else:
        variance_inv = 0.0
        outlier_ratio = 0.0
        skew = 0.0

    card = df[col].nunique(dropna=True) / max(len(df[col]), 1)

    return {
        "missing_pct": missing_pct,
        "id_like": id_like,
        "engineered": engineered,
        "variance_inv": variance_inv,
        "outlier_ratio": outlier_ratio,
        "cardinality": float(card),
        "skew_abs": float(abs(skew)),
    }

def _elimination_score(f: Dict[str, float]) -> float:
    """Łączny score (priorytety: braki, ID-like, engineered...)."""
    w = {
        "missing_pct": 0.40,
        "id_like": 0.15,
        "engineered": 0.10,
        "variance_inv": 0.10,
        "outlier_ratio": 0.10,
        "cardinality": 0.05,
        "skew_abs": 0.05,
    }
    return sum(w[k] * f.get(k, 0.0) for k in w)

# ---------- Rozkład kolumn ----------

def _numeric_distribution_details(s: pd.Series, col_name: str) -> Tuple[alt.VConcatChart, pd.DataFrame, str, Dict[str, float]]:
    """Boxplot + histogram + metryki + komentarz
    - Tekst pokazuje REALNE whiskers (jak na wykresie), nie teoretyczne płotki.
    - Tabela metryk zawiera dodatkowo 'liczebność'.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    s_clean = s_num.dropna()

    if len(s_clean) == 0:
        empty_df = pd.DataFrame({"msg": ["brak danych"]})
        empty_chart = (
            alt.Chart(empty_df)
            .mark_text(text="Brak danych do wizualizacji", size=14)
            .encode()
            .properties(height=200)
        )
        stats_table = pd.DataFrame(
            {"metryka": ["liczebność", "min", "Q1 (25%)", "mediana (50%)", "Q3 (75%)", "max", "średnia", "odch.std."],
             "wartość": [0] + [None]*7}
        )
        return empty_chart, stats_table, "Brak danych liczbowych do analizy.", {}

    plot_df = pd.DataFrame({"value": s_clean})
    plot_df["label"] = col_name

    # Podstawowe statystyki
    desc = s_clean.describe(percentiles=[0.25, 0.5, 0.75])
    q1       = float(desc.get("25%",  np.nan))
    q3       = float(desc.get("75%",  np.nan))
    median_v = float(desc.get("50%",  np.nan))
    mean_v   = float(desc.get("mean", np.nan))
    min_v    = float(desc.get("min",  np.nan))
    max_v    = float(desc.get("max",  np.nan))
    std_v    = float(desc.get("std",  np.nan))
    cnt_v    = int(len(s_clean))

    # IQR + płotki Tukeya
    iqr = q3 - q1 if (np.isfinite(q1) and np.isfinite(q3)) else np.nan
    if np.isfinite(iqr) and iqr != 0:
        fence_lo = q1 - 1.5 * iqr
        fence_hi = q3 + 1.5 * iqr
    else:
        fence_lo, fence_hi = q1, q3

    # >>> REALNE WĄSY (whiskers) – to co rysuje boxplot:
    # dolny wąs = najmniejsza wartość >= fence_lo,
    # górny wąs = największa wartość <= fence_hi
    if np.isfinite(fence_lo):
        whisker_lo = float(s_clean[s_clean >= fence_lo].min()) if not s_clean[s_clean >= fence_lo].empty else float(s_clean.min())
    else:
        whisker_lo = float(s_clean.min())

    if np.isfinite(fence_hi):
        whisker_hi = float(s_clean[s_clean <= fence_hi].max()) if not s_clean[s_clean <= fence_hi].empty else float(s_clean.max())
    else:
        whisker_hi = float(s_clean.max())

    # Outliery: wszystko poniżej/ponad płotkami
    plot_df["is_outlier"] = False
    if np.isfinite(fence_lo) and np.isfinite(fence_hi):
        plot_df["is_outlier"] = (plot_df["value"] < fence_lo) | (plot_df["value"] > fence_hi)
        outlier_ratio = (plot_df["is_outlier"].mean() * 100.0) if len(plot_df) else 0.0
    else:
        outlier_ratio = 0.0
    outliers_df = plot_df[plot_df["is_outlier"]]

    # Wykresy: boxplot (z outlierami) + histogram
    box_base = (
        alt.Chart(plot_df)
        .mark_boxplot(color="#1f77b4", outliers={"color": "#dc3545"})
        .encode(
            x=alt.X("value:Q", title=col_name),
            y=alt.Y("label:N", title=""),
        )
    )
    outlier_layer = (
        alt.Chart(outliers_df)
        .mark_point(filled=True, size=50, color="#dc3545")
        .encode(x=alt.X("value:Q", title=col_name), y=alt.Y("label:N", title=""))
    )
    box_layer = (box_base + outlier_layer).properties(height=70)

    hist_chart = (
        alt.Chart(plot_df)
        .mark_bar(color="#1f77b4")
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=col_name),
            y=alt.Y("count():Q", title="Liczebność"),
            tooltip=[alt.Tooltip("count():Q", title="liczebność"), alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=160)
    )
    combo_chart = alt.vconcat(box_layer, hist_chart).resolve_scale(x="shared")

    # Komentarz – pokazujemy REALNE WHISKERS (identyczne z wykresem), bez teoretycznych płotków
    phys_hint = ""
    if s_clean.min() >= 0 and np.isfinite(whisker_lo) and whisker_lo < 0:
        phys_hint = " (wartości ujemne mogą być niefizyczne dla tej cechy)"

    comment_text = (
        f"Szacowany bezpieczny zakres (wg IQR, realne wąsy boxplota): "
        f"[{whisker_lo:.2f} ; {whisker_hi:.2f}]{phys_hint}. "
        f"Około {outlier_ratio:.2f}% rekordów poza zakresem.\n\n"
        "Domyślnie **nie usuwamy** odstających; dodamy flagę `is_outlier_<kolumna>`. "
        "Na życzenie możesz przyciąć ekstremy (winsoryzacja) w Sekcji 7."
    )

    # Tabela metryk — dodajemy 'liczebność'
    stats_table = pd.DataFrame(
        {
            "metryka": ["liczebność", "min", "Q1 (25%)", "mediana (50%)", "Q3 (75%)", "max", "średnia", "odch.std."],
            "wartość": [cnt_v, _num(min_v), _num(q1), _num(median_v), _num(q3), _num(max_v), _num(mean_v), _num(std_v)],
        }
    )

    details = {
        "lower": whisker_lo,
        "upper": whisker_hi,
        "outlier_ratio_pct": outlier_ratio,
    }
    return combo_chart, stats_table, comment_text, details


def _numeric_by_category_charts(
    s_num: pd.Series,
    s_cat: pd.Series,
    col_num_name: str,
    col_cat_name: str,
    categories_keep: list[str] | None = None,
    maxbins: int = 30,
    mode: str = "facet",           # "facet" | "overlay"
    scale: str = "count",          # "count" | "share" | "kde"
    opacity: float = 0.55,
) -> tuple[alt.Chart, pd.DataFrame]:
    """
    Rozkład liczb (hist/KDE) wg kategorii z opcją facet/overlay.
    Zmiany:
    • 1A: delikatne odstępy między słupkami histogramu (binSpacing),
          wyrzucenie 'bounds' z properties (by nie wywoływać błędu 'Spec has no parameter named "bounds"').
    • 1B: dla 'share' (Udział w grupie) wymuszamy overlay i używamy stack='normalize' (100%).
    """
    # --- przygotowanie danych / statystyk ---
    v = pd.to_numeric(s_num, errors="coerce")
    g = s_cat.astype(str)
    dfp = pd.DataFrame({col_num_name: v, col_cat_name: g}).dropna()
    if categories_keep:
        dfp = dfp[dfp[col_cat_name].isin(categories_keep)]
    if dfp.empty:
        empty = (
            alt.Chart(pd.DataFrame({"msg": ["Brak danych po filtrze"]}))
            .mark_text(size=14)
            .encode(text="msg:N")
            .properties(height=120, width="container")
        )
        cols = [col_cat_name, "n", "min", "Q1", "median", "Q3", "max", "mean", "std"]
        return empty, pd.DataFrame(columns=cols)

    def _stats(x: pd.Series) -> dict:
        d = x.describe(percentiles=[.25, .5, .75])
        return {"min": float(d.get("min", np.nan)), "Q1": float(d.get("25%", np.nan)),
                "median": float(d.get("50%", np.nan)), "Q3": float(d.get("75%", np.nan)),
                "max": float(d.get("max", np.nan)), "mean": float(d.get("mean", np.nan)),
                "std": float(d.get("std", np.nan)), "n": int(x.notna().count())}

    stats_rows = []
    for k, sub in dfp.groupby(col_cat_name, dropna=False):
        stt = _stats(sub[col_num_name]); stt[col_cat_name] = k; stats_rows.append(stt)
    cols = [col_cat_name, "n", "min", "Q1", "median", "Q3", "max", "mean", "std"]
    stats_df = pd.DataFrame(stats_rows).reindex(columns=cols)
    cat_order = stats_df[col_cat_name].astype(str).tolist()
    if not stats_df.empty and col_cat_name in stats_df.columns:
        stats_df = stats_df.sort_values(col_cat_name, na_position="last")

    base = alt.Chart(dfp)

    # --- inteligentne binowanie / extent ---
    vals = pd.to_numeric(dfp[col_num_name], errors="coerce").dropna()
    unique_vals = np.sort(vals.unique())
    lo, hi = float(vals.min()), float(vals.max())
    bin_param = alt.Bin(maxbins=maxbins, extent=[lo, hi])

    # dyskretne zbiory (0/1 lub małe całkowite) → step=1 z buforem
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
        bin_param = alt.Bin(extent=[-0.5, 1.5], step=1)
    elif (len(unique_vals) <= 12
          and np.all(np.isfinite(unique_vals))
          and np.all(np.mod(unique_vals, 1) == 0)):
        loi, hii = float(unique_vals.min()), float(unique_vals.max())
        bin_param = alt.Bin(extent=[loi - 0.5, hii + 0.5], step=1)

    # --- 1B: „Udział w grupie” wymusza overlay (ma sens tylko jako 100% stacked na jednej osi) ---
    if scale == "share":
        mode = "overlay"

    # --- KDE ---
    if scale == "kde":
        density = base.transform_density(col_num_name, groupby=[col_cat_name],
                                         as_=[col_num_name, "density"])
        tips = [
            alt.Tooltip(f"{col_cat_name}:N", title="kategoria"),
            alt.Tooltip(f"{col_num_name}:Q", title=col_num_name, format=".2f"),
            alt.Tooltip("density:Q", title="gęstość", format=".3f"),
        ]

        if mode == "facet":
            chart = (
                density.mark_line(size=2, opacity=0.95)
                .encode(
                    x=alt.X(f"{col_num_name}:Q", title=col_num_name),
                    y=alt.Y("density:Q", title="Gęstość"),
                    color=alt.Color(f"{col_cat_name}:N", legend=None,
                                    scale=_scale_for_categories(cat_order)),
                    tooltip=tips,
                )
                .properties(height=160, width=FACET_CHART_WIDTH)
                .facet(row=alt.Row(f"{col_cat_name}:N",
                                   header=alt.Header(labelFontWeight="bold")))
            )
        else:
            chart = (
                density.mark_line(size=2, opacity=0.95)
                .encode(
                    x=alt.X(f"{col_num_name}:Q", title=col_num_name),
                    y=alt.Y("density:Q", title="Gęstość"),
                    color=alt.Color(f"{col_cat_name}:N", title=col_cat_name,
                                    scale=_scale_for_categories(cat_order)),
                    tooltip=tips,
                )
                .properties(height=260, width="container")
            )

        return chart, stats_df

    # --- COUNT / SHARE (histogram) — 100% stacked z poprawnie wycentrowanymi etykietami ---
    if scale == "share":
        # 1) policzmy count, total, share oraz środki binów
        share_data = (
            base
            .transform_bin(as_=["bin_start", "bin_end"], field=col_num_name, bin=bin_param)
            .transform_aggregate(
                count="count()", groupby=[col_cat_name, "bin_start", "bin_end"]
            )
            .transform_joinaggregate(
                total="sum(count)", groupby=["bin_start", "bin_end"]
            )
            .transform_calculate(
                share="datum.total == 0 ? 0 : datum.count / datum.total",
                bin_mid="(datum.bin_start + datum.bin_end) / 2"
            )
        )

        # 2) porządek kategorii zgodny z paletą (kluczowe dla poprawnej pozycji etykiet)
        #    – Vega-Lite będzie stackować wg 'order', a my kumulujemy wg tej samej kolejności
        _cat_domain_json = json.dumps(list(cat_order))
        _rank_expr = f"indexof({_cat_domain_json}, datum['{col_cat_name}'])"
        share_data = (
            share_data
            .transform_calculate(cat_rank=_rank_expr)
            .transform_window(
                cum_share="sum(share)",
                groupby=["bin_start", "bin_end"],
                sort=[alt.SortField("cat_rank", order="ascending")]
            )
            .transform_calculate(
                y_mid="datum.cum_share - datum.share/2"
            )
            .transform_filter("isFinite(datum.y_mid)")
        )

        # 3) słupki 100% stacked (z tooltipem także o udziale %)
        bars = (
            share_data.mark_bar(opacity=opacity)
            .encode(
                x=alt.X("bin_start:Q", title=col_num_name, bin="binned"),
                x2=alt.X2("bin_end:Q"),
                y=alt.Y(
                    "count:Q",
                    stack="normalize",
                    title="Udział w binie (100%)",
                    axis=alt.Axis(format=".0%")
                ),
                color=alt.Color(
                    f"{col_cat_name}:N",
                    legend=(None if mode == "facet" else alt.Legend(title=col_cat_name)),
                    scale=_scale_for_categories(cat_order)
                ),
                # ⬇️ wymuszamy identyczny porządek stacku jak w kumulacji
                order=alt.Order("cat_rank:Q", sort="ascending"),
                tooltip=[
                    alt.Tooltip(f"{col_cat_name}:N", title="kategoria"),
                    alt.Tooltip("count:Q", title="liczebność w binie"),
                    alt.Tooltip("share:Q", title="udział %", format=".1%")
                ],
            )
        )

        # 4) etykiety – czarne, wyśrodkowane pionowo w segmencie (y_mid)
        labels = (
            share_data
            .transform_filter("datum.share >= 0.04")  # próg widoczności, dopasuj wg potrzeb
            .mark_text(
                baseline="middle",
                align="center",
                fontSize=11,
                color="black"
            )
            .encode(
                x=alt.X("bin_mid:Q", title=col_num_name),
                y=alt.Y("y_mid:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                # ⬇️ ten sam porządek co stack (gdy wiele etykiet na tej samej x-pozycji)
                order=alt.Order("cat_rank:Q", sort="ascending"),
                text=alt.Text("share:Q", format=".0%")
            )
        )

        # 5) warstwa: słupki + etykiety (reszta – facet/overlay – bez zmian poniżej)
        mark_base = alt.layer(bars, labels)

    else:
        # ── Liczebność (klasyczny histogram) „obok siebie” ───────────────────────────
        # 1) pre-binning + środki i szerokości binów (wspólne dla wszystkich kategorii)
        binned = (
            base
            .transform_bin(
                as_=["bin_start", "bin_end"],
                field=col_num_name,
                bin=bin_param
            )
            .transform_calculate(
                bin_mid="(datum.bin_start + datum.bin_end) / 2",
                bin_w="(datum.bin_end - datum.bin_start)"
            )
        )

        # 2) stały porządek kategorii (rangi) zgodny z domeną palety
        _cat_domain_json = json.dumps(list(cat_order))
        _rank_expr = f"indexof({_cat_domain_json}, datum['{col_cat_name}'])"
        # ile kategorii łącznie (stała liczba; łatwiej niż liczyć w oknach)
        _k_total = len(cat_order)

        binned = (
            binned
            .transform_calculate(
                cat_rank=_rank_expr,
                k_total=str(_k_total),
                # szerokość „paska” zajmowanego przez wszystkie kategorie w jednym binie (0..1 * szerokość binu)
                band_fraction="0.85",
                # szerokość jednej kategorii wewnątrz binu
                cat_w="(toNumber(datum.band_fraction) * datum.bin_w) / toNumber(datum.k_total)",
                # środek paska danej kategorii (lewo/prawo od środka binu)
                x_mid="datum.bin_mid + ( (datum.cat_rank - (toNumber(datum.k_total)-1)/2 ) * datum.cat_w )",
                # krawędzie słupka dla kategorii
                x_left="datum.x_mid - datum.cat_w/2",
                x_right="datum.x_mid + datum.cat_w/2"
            )
        )

        # 3) rysujemy słupki: x/x2 = wyliczone krawędzie, y = count, bez stackowania
        mark_base = (
            binned.mark_bar(opacity=opacity)
            .encode(
                x=alt.X("x_left:Q", title=col_num_name),
                x2=alt.X2("x_right:Q"),
                y=alt.Y("count():Q", title="Liczebność", stack=None),
                color=alt.Color(
                    f"{col_cat_name}:N",
                    legend=(None if mode == "facet" else alt.Legend(title=col_cat_name)),
                    scale=_scale_for_categories(cat_order)
                ),
                order=alt.Order("cat_rank:Q", sort="ascending"),
                tooltip=[
                    alt.Tooltip(f"{col_cat_name}:N", title="kategoria"),
                    alt.Tooltip("count():Q", title="liczebność"),
                    alt.Tooltip("bin_start:Q", title="bin od", format=".2f"),
                    alt.Tooltip("bin_end:Q",   title="bin do", format=".2f"),
                ],
            )
        )

    if mode == "facet":
        chart = (
            mark_base
            .properties(height=160, width=FACET_CHART_WIDTH)
            .facet(row=alt.Row(f"{col_cat_name}:N",
                            header=alt.Header(labelFontWeight="bold")))
        )
    else:
        chart = mark_base.properties(height=260, width="container")

    # 👇 Teraz odstępy między „binami” działają
    chart = chart.configure_bar(binSpacing=4)
    return chart, stats_df



def _categorical_distribution_details(
    s: pd.Series, col_name: str, top_n: int = 20
) -> Tuple[alt.Chart, alt.Chart, pd.DataFrame, str]:
    """Top kategorie, coverage, tabela i komentarz."""
    ser = s.astype(str)
    ser = ser.replace("nan", np.nan).dropna()

    vc = ser.value_counts()
    total = vc.sum() if len(vc) > 0 else 0
    tmp_df = vc.head(top_n).reset_index()

    if tmp_df.shape[1] == 2:
        top_df = tmp_df.rename(columns={tmp_df.columns[0]: "kategoria", tmp_df.columns[1]: "liczebność"})
    else:
        top_df = pd.DataFrame(columns=["kategoria", "liczebność"])

    top_df["udział_%"] = np.where(total > 0, (top_df["liczebność"] / total) * 100.0, 0.0)
    cov_df = top_df.copy()
    cov_df["cum_count"] = cov_df["liczebność"].cumsum()
    cov_df["cum_pct"] = np.where(total > 0, (cov_df["cum_count"] / total) * 100.0, 0.0)
    cov_df["rank"] = np.arange(1, len(cov_df) + 1)

    bar_chart_top = (
        alt.Chart(top_df).mark_bar(color="#1f77b4")
        .encode(
            x=alt.X("liczebność:Q", title="Liczebność"),
            y=alt.Y("kategoria:N", sort="-x", title=col_name),
            tooltip=[alt.Tooltip("kategoria:N", title="kategoria"),
                     alt.Tooltip("liczebność:Q", title="liczebność"),
                     alt.Tooltip("udział_%:Q", title="udział %", format=".1f")],
        )
        .properties(height=250, title="TOP kategorie")
    )
    coverage_chart = (
        alt.Chart(cov_df).mark_line(point=True, color="#1f77b4")
        .encode(
            x=alt.X("rank:Q", title="Top N kategorii"),
            y=alt.Y("cum_pct:Q", title="Skumulowany udział %"),
            tooltip=[alt.Tooltip("rank:Q", title="Top N"),
                     alt.Tooltip("cum_pct:Q", title="Pokrycie [%]", format=".1f")],
        )
        .properties(height=250, title="Jak szybko TOP kategorie pokrywają zbiór?")
    )

    unique_ratio = (ser.nunique(dropna=True) / max(len(ser), 1) if len(ser) > 0 else 0.0)
    if len(top_df) > 0:
        top_cat = top_df.iloc[0]["kategoria"]
        top_share = (top_df.iloc[0]["liczebność"] / total) if total else 0.0
    else:
        top_cat = "(brak danych)"
        top_share = 0.0

    if unique_ratio > 0.9:
        comment_text = (
            "Ta kolumna ma prawie same unikalne wartości. Wygląda jak ID lub wolny opis. "
            "W klasycznym ML raczej niewiele wnosi bez przetwarzania (embeddingi NLP)."
        )
    elif top_share > 0.5:
        comment_text = (f"Wartość '{top_cat}' dominuje (>50%). To prawdopodobnie główny segment. "
                        "Model łatwo się tego nauczy, ale cecha ma ograniczoną różnorodność.")
    else:
        comment_text = "Rozkład kategorii jest dość zrównoważony — dobra, stabilna cecha kategoryczna."

    freq_table = top_df.copy()
    return bar_chart_top, coverage_chart, freq_table, comment_text


# ---------- Automatyczne przygotowanie danych ----------

def _auto_prepare_for_training(
    df_raw: pd.DataFrame,
    info_df: pd.DataFrame,
    decisions: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Czyszczenie i transformacje zgodnie z zasadami + wyborami usera."""
    df_work = df_raw.copy()

    drop_cols = decisions.get("drop_cols", [])
    winsorize_cols = decisions.get("winsorize_cols", [])
    remove_duplicates = decisions.get("remove_duplicates", True)

    df_work = df_work.drop(columns=[c for c in drop_cols if c in df_work.columns], errors="ignore")

    conversions: Dict[str, str] = {}
    imputations: Dict[str, str] = {}
    outlier_flags: List[str] = []
    winsorized: Dict[str, Dict[str, float]] = {}
    new_time_cols_added: List[str] = []

    info_map = info_df.set_index("kolumna").to_dict(orient="index")

    # Konwersje typów
    for col in list(df_work.columns):
        col_info = info_map.get(col, {})
        if col not in df_work.columns:
            continue

        # --- PRÓBY KONWERSJI SPECJALNYCH DLA OBIEKTÓW ---
        if df_work[col].dtype == object:
            s = df_work[col].astype(str)

            # 1) HH:MM:SS / MM:SS  -> <col>_sec
            sec = s.map(_parse_duration_to_seconds)
            if pd.notna(sec).sum() >= max(5, int(0.6*len(s.dropna()))):
                newc = f"{col}_sec"
                df_work[newc] = pd.to_numeric(sec, errors="coerce")
                conversions[newc] = "derived_seconds_from_time_string"
                continue  # nie ruszamy oryginału

            # 2) pace 'min/km' lub 'MM:SS' -> <col>_sec_per_km
            pace = s.map(_parse_pace_min_per_km_to_sec_per_km)
            if pd.notna(pace).sum() >= max(5, int(0.6*len(s.dropna()))):
                newc = f"{col}_sec_per_km"
                df_work[newc] = pd.to_numeric(pace, errors="coerce")
                conversions[newc] = "derived_pace_sec_per_km"
                continue

            # 3) procenty -> <col>_pct (0..1)
            perc = s.map(_parse_percentage)
            if pd.notna(perc).sum() >= max(5, int(0.6*len(s.dropna()))):
                newc = f"{col}_pct"
                df_work[newc] = pd.to_numeric(perc, errors="coerce")
                conversions[newc] = "derived_ratio_from_percent"
                continue

            # 4) km/h -> <col>_kmh
            kmh = s.map(_parse_kmh)
            if pd.notna(kmh).sum() >= max(5, int(0.6*len(s.dropna()))):
                newc = f"{col}_kmh"
                df_work[newc] = pd.to_numeric(kmh, errors="coerce")
                conversions[newc] = "derived_speed_kmh"
                continue

            # 5) waluty/liczby z tysiącami -> <col>_num
            cur = s.map(_parse_currency_or_plain_number)
            if pd.notna(cur).sum() >= max(5, int(0.6*len(s.dropna()))):
                newc = f"{col}_num"
                df_work[newc] = pd.to_numeric(cur, errors="coerce")
                conversions[newc] = "derived_numeric_from_currency/str"
                continue

        # --- ISTNIEJĄCE REGUŁY ---
        if col_info.get("convert_numeric", False):
            df_work[col] = (df_work[col].astype(str)
                            .str.replace(",", ".", regex=False)
                            .str.replace(" ", "", regex=False))
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
            conversions[col] = "text->float"

        if col_info.get("convert_datetime", False):
            parsed = pd.to_datetime(df_work[col], errors="coerce", utc=False)
            df_work[col] = parsed
            conversions[col] = "text->datetime(+year/month/dow)"
            df_work[f"{col}_year"] = parsed.dt.year
            df_work[f"{col}_month"] = parsed.dt.month
            df_work[f"{col}_dow"] = parsed.dt.dayofweek
            new_time_cols_added.extend([f"{col}_year", f"{col}_month", f"{col}_dow"])
                # + liczbowy wariant czasu: sekundy od północy
            try:
                sec = (parsed.dt.hour * 3600 + parsed.dt.minute * 60 + parsed.dt.second).astype("float")
                df_work[f"{col}_sec"] = sec
                conversions[f"{col}_sec"] = "derived_seconds_from_datetime"
            except Exception:
                pass


    # Imputacja braków
    for col in list(df_work.columns):
        s = df_work[col]
        if s.dtype == object:
            s = s.replace("", np.nan)
            df_work[col] = s
        if s.isna().any():
            if pd.api.types.is_numeric_dtype(s):
                med = s.median(skipna=True)
                df_work[col] = s.fillna(med)
                imputations[col] = f"median={med}"
            elif pd.api.types.is_datetime64_any_dtype(s):
                mode_val = s.dropna().mode()
                if not mode_val.empty:
                    fill_val = mode_val.iloc[0]
                elif s.dropna().size > 0:
                    fill_val = s.dropna().min()
                else:
                    fill_val = pd.Timestamp("1970-01-01")
                df_work[col] = s.fillna(fill_val)
                imputations[col] = f"datetime_fill={fill_val}"
            else:
                mode_val = s.dropna().mode()
                fill_val = (mode_val.iloc[0] if not mode_val.empty else "__MISSING__")
                df_work[col] = s.fillna(fill_val)
                imputations[col] = f"mode='{fill_val}'"

    # Outliery + winsoryzacja
    for col in df_work.select_dtypes(include=[np.number]).columns:
        s = df_work[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        if pd.isna(q1) or pd.isna(q3):
            continue
        iqr = q3 - q1
        if iqr == 0:
            lower, upper = q1, q3
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

        flag_col = f"is_outlier_{col}"
        df_work[flag_col] = ((s < lower) | (s > upper)).astype(int)
        outlier_flags.append(flag_col)

        if col in winsorize_cols:
            df_work[col] = s.clip(lower, upper)
            winsorized[col] = {"lower": float(lower if pd.notna(lower) else 0.0),
                               "upper": float(upper if pd.notna(upper) else 0.0)}

    # Duplikaty
    duplicates_removed = 0
    if remove_duplicates:
        before = len(df_work)
        df_work = df_work.drop_duplicates(keep="first")
        duplicates_removed = before - len(df_work)

    prep_report = {
        "dropped_columns": drop_cols,
        "type_conversions": conversions,
        "new_time_cols_added": new_time_cols_added,
        "imputations": imputations,
        "outlier_flags": outlier_flags,
        "winsorized": winsorized,
        "duplicates_removed": int(duplicates_removed),
        "n_rows_final": int(df_work.shape[0]),
        "n_cols_final": int(df_work.shape[1]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return df_work, prep_report

def _save_summary_to_artifacts(text_markdown: str, latest_info: Dict[str, Any]) -> str:
    """Zapisuje summary_ai.md obok artefaktów bieżącego runu i zwraca pełną ścieżkę."""
    run_dir = latest_info.get("run_dir")
    if not run_dir:
        csv_path = latest_info.get("csv_path", ".")
        run_dir = os.path.dirname(csv_path)
    os.makedirs(run_dir, exist_ok=True)

    path = os.path.join(run_dir, "summary_ai.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write((text_markdown or "").strip() + "\n")

    st.session_state["summary_ai_path"] = path
    return path

def _persist_artifacts(
    df_ready: pd.DataFrame,
    prep_report: Dict[str, Any],
    latest_info: Dict[str, Any],
) -> Tuple[str, str]:
    """Zapisuje gotowy zbiór + raport i aktualizuje session_state."""
    run_dir = latest_info.get("run_dir")
    if not run_dir:
        csv_path = latest_info.get("csv_path", ".")
        run_dir = os.path.dirname(csv_path)
    os.makedirs(run_dir, exist_ok=True)

    ready_path = os.path.join(run_dir, "ready_for_training.csv")
    report_path = os.path.join(run_dir, "prep_report.json")

    df_ready.to_csv(ready_path, index=False, encoding="utf-8")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(prep_report, f, ensure_ascii=False, indent=2)

    st.session_state["latest_artifacts"].update(
        {
            "ready_csv_path": ready_path,
            "prep_report_path": report_path,
            "n_rows_ready": prep_report["n_rows_final"],
            "n_cols_ready": prep_report["n_cols_final"],
            "status_ready": "ok",
        }
    )
    st.session_state["prep_report"] = prep_report
    return ready_path, report_path


def _compute_readiness_score(
    high_null_cols: List[str],
    duplicates_pct: float,
    auto_drop_candidates: List[str],
    pairs_sorted: List[Dict[str, Any]],
) -> int:
    """Indeks gotowości 0–100."""
    score = 100
    if len(high_null_cols) > 0:
        score -= 10
    if duplicates_pct > 1.0:
        score -= 10
    if len(auto_drop_candidates) > 0:
        score -= 10
    if len(pairs_sorted) > 0:
        score -= 10
    return max(0, min(score, 100))


def _estimate_hours_saved(
    n_rows_raw: int,
    n_cols_raw: int,
    high_null_cols: list[str],
    duplicates_count: int,
    auto_drop_candidates: list[str],
    pairs_sorted: list[dict[str, Any]],
) -> float:
    base_hours = 4.0
    size_hours = (math.log10(max(n_rows_raw, 10)) * 2.0) + (n_cols_raw * 0.3)
    issues = 0
    if len(high_null_cols) > 0:
        issues += 1
    if duplicates_count > 0:
        issues += 1
    if len(auto_drop_candidates) > 0:
        issues += 1
    if len(pairs_sorted) > 0:
        issues += 1
    quality_hours = issues * 2.0
    est = base_hours + size_hours + quality_hours
    est = max(est, 4.0)
    est = min(est, 40.0)
    return round(est, 1)

def _estimate_cost_saved_pln(hours_saved: float, hourly_rate_pln: float = 450.0) -> float:
    return round(hours_saved * hourly_rate_pln, 2)

# ---------- Sekcja 7: kontrola widoczności (reset po zmianie danych) ----------
def _sec7_signature(csv_path: str, df: pd.DataFrame) -> str:
    """Lekka sygnatura danych do resetu widoczności sekcji 7 po rerunie/zmianie pliku."""
    try:
        mtime = os.path.getmtime(csv_path) if csv_path and os.path.exists(csv_path) else 0
    except Exception:
        mtime = 0
    cols = ",".join(list(df.columns)[:50])
    return f"{csv_path}|{mtime}|{df.shape[0]}x{df.shape[1]}|{hash(cols)}"

def _reset_sec7_state():
    st.session_state["sec7_revealed"] = False
    st.session_state["latest_summary_text"] = ""
    st.session_state["prep_done_and_summarized"] = False
    st.session_state["play_tts_now"] = False

    # 🔄 czyścimy cache wyników winsoryzacji (dla bezpieczeństwa przy zmianie zbioru)
    st.session_state.pop("winsor_cache", None)

# ---------- UI helpers ----------

def _metric_card_html(title: str, value: str, subtitle: str, bg: str = "#f8f9fa", fg: str = "#111", border: str = "#ddd") -> str:
    return f"""
    <div style="border:1px solid {border}; border-radius:0.5rem; padding:0.75rem 1rem; background:{bg};">
      <div style="font-size:0.75rem; color:#666; line-height:1.2; font-weight:500;">{title}</div>
      <div style="font-size:1.5rem; font-weight:600; color:{fg}; line-height:1.4; margin-top:0.25rem;">{value}</div>
      <div style="font-size:0.8rem; color:#666; line-height:1.2; margin-top:0.25rem;">{subtitle}</div>
    </div>
    """

def _status_banner_html(level: str, text_md: str) -> str:
    """
    level: 'ok' | 'warn' | 'err'
    text_md: prosty markdown w jednej linii (obsługujemy **bold** i `code`)
    """
    import re
    # mini-renderer: **...** -> <strong>...</strong>, `...` -> <code>...</code>
    def _md_to_inline_html(s: str) -> str:
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    styles = {
        "ok":   {"bg":"#f0fff4","bd":"#c6f6d5","fg":"#22543d","icon":"✅"},
        "warn": {"bg":"#fffaf0","bd":"#feebc8","fg":"#7b341e","icon":"⚠️"},
        "err":  {"bg":"#fff5f5","bd":"#fed7d7","fg":"#742a2a","icon":"⛔"},
    }[level]

    text_html = _md_to_inline_html(text_md)

    return f"""
    <div style="border:1px solid {styles['bd']};background:{styles['bg']};
                border-radius:0.6rem;padding:0.9rem 1rem;margin-top:0.6rem;
                color:{styles['fg']}; line-height:1.55;">
      <span style="font-weight:700;margin-right:.35rem;">{styles['icon']}</span>{text_html}
    </div>
    """


def _tech_details_html(csv_path: str, run_dir: str, mask_pii: bool, global_missing_pct: float,
                       duplicates_pct: float, auto_drop_candidates: List[str],
                       predicted_cols_after_drop: int) -> str:
    return f"""
    <ul style="list-style-type:none; padding-left:0; font-size:0.9rem; line-height:1.5;">
      <li>• <b>Ścieżka pliku źródłowego:</b> <code>{csv_path}</code></li>
      <li>• <b>Folder roboczy runu:</b> <code>{run_dir}</code></li>
      <li>• <b>Maskowanie wrażliwych danych (PII):</b> {"tak" if mask_pii else "nie"}</li>
      <li>• <b>Globalne braki:</b> ~{global_missing_pct:.1f}% wszystkich komórek</li>
      <li>• <b>Potencjalne duplikaty:</b> {duplicates_pct:.1f}% rekordów</li>
      <li>• <b>Kolumny do potencjalnego usunięcia / scalania:</b> {", ".join(auto_drop_candidates) if auto_drop_candidates else "(brak)"}</li>
      <li>• <b>Szacowana liczba kolumn po czyszczeniu:</b> {predicted_cols_after_drop} (+ cechy z dat / outlierów)</li>
    </ul>
    """

def _success_hero_box(hours_saved: float, cost_saved: float) -> str:
    return f"""
    <div style="border:1px solid #d4edda;background:#f6fff8;border-radius:0.5rem;padding:1rem 1.25rem;font-size:1rem;line-height:1.5;color:#155724;">
      <div style="font-size:1.25rem; font-weight:600; margin-bottom:0.5rem;">
        ✅ Dane gotowe do trenowania modelu
      </div>
      <div>
        🎉 Szacunkowo zaoszczędzono <b>{hours_saved} godzin</b> pracy analityka (~<b>{cost_saved} PLN</b>),
        bez ręcznego czyszczenia Excela i bez ryzyka pomyłki.
      </div>
      <div style="margin-top:0.75rem;">
        Gotowy zbiór: <code>ready_for_training.csv</code> · Raport: <code>prep_report.json</code>.
      </div>
    </div>
    """

def _before_after_cards_html(n_cols_raw: int, n_cols_final: int, missing_before_pct: float,
                             duplicates_before: int, duplicates_removed: int, outlier_flags_count: int) -> str:
    return f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:0.75rem; margin-top:1rem;">
      <div style="border:1px solid #ddd;border-radius:0.5rem;padding:0.75rem;">
        <div style="font-size:0.8rem;color:#666;">Kolumny (przed ➜ po)</div>
        <div style="font-size:1.1rem;font-weight:600;">{n_cols_raw} ➜ {n_cols_final}</div>
        <div style="font-size:0.8rem;color:#666;margin-top:0.25rem;">Usunęliśmy śmieciowe/zdublowane kolumny i dodaliśmy cechy z dat oraz flagi outlierów.</div>
      </div>
      <div style="border:1px solid #ddd;border-radius:0.5rem;padding:0.75rem;">
        <div style="font-size:0.8rem;color:#666;">Braki danych</div>
        <div style="font-size:1.1rem;font-weight:600;">~{missing_before_pct:.1f}% ➜ 0.0%</div>
        <div style="font-size:0.8rem;color:#666;margin-top:0.25rem;">Uzupełniliśmy luki medianą/trybem/sensowną datą.</div>
      </div>
      <div style="border:1px solid #ddd;border-radius:0.5rem;padding:0.75rem;">
        <div style="font-size:0.8rem;color:#666;">Duplikaty</div>
        <div style="font-size:1.1rem;font-weight:600;">{duplicates_before} ➜ {duplicates_before - duplicates_removed}</div>
        <div style="font-size:0.8rem;color:#666;margin-top:0.25rem;">Usunęliśmy zdublowane wiersze.</div>
      </div>
      <div style="border:1px solid #ddd;border-radius:0.5rem;padding:0.75rem;">
        <div style="font-size:0.8rem;color:#666;">Wartości odstające</div>
        <div style="font-size:1.1rem;font-weight:600;">Zflagowano {outlier_flags_count} kolumn</div>
        <div style="font-size:0.8rem;color:#666;margin-top:0.25rem;">Nie kasujemy ekstremów — dajemy flagę, model widzi nietypowe przypadki.</div>
      </div>
    </div>
    """

# ---------- Scatter z trendem ----------

def _scatter_with_trend(df: pd.DataFrame, x: str, y: str, height: int = 200, title: str | None = None) -> alt.Chart:
    use_df = df[[x, y]].dropna()
    base = (
        alt.Chart(use_df)
        .mark_point(filled=True, size=40, opacity=0.6, color="#007bff")
        .encode(x=alt.X(f"{x}:Q", title=x), y=alt.Y(f"{y}:Q", title=y), tooltip=[x, y])
        .properties(height=height)
    )
    trend = (
        alt.Chart(use_df)
        .transform_regression(x, y)
        .mark_line(color="#dc3545")
        .encode(x=alt.X(f"{x}:Q", title=x), y=alt.Y(f"{y}:Q", title=y))
    )
    return (base + trend).properties(title=title) if title else (base + trend)

def _pairs_dataframe(pairs_sorted: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{"col1": p["col1"], "col2": p["col2"], "corr": float(p["corr"]), "suggest_drop": p["suggest_drop"]} for p in pairs_sorted])

# ---------- TTS + AI TL;DR ----------

# ───────────────────── Helpery / cache TTS i ElevenLabs ─────────────────────
ELEVEN_VOICE_ID_RE = re.compile(r"^[A-Za-z0-9]{20,40}$")

@st.cache_data(show_spinner=False, ttl=1800, max_entries=64)
def _eleven_list_voices_cached(api_key: str):
    """Zwraca listę głosów z konta ElevenLabs lub [] przy błędzie/braku uprawnień."""
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": api_key, "accept": "application/json"},
            timeout=20,
        )
        if r.ok:
            j = r.json() or {}
            voices = j.get("voices") or []
            # Normalizujemy pola, które wykorzystujesz niżej
            out = []
            for v in voices:
                if isinstance(v, dict):
                    out.append({
                        "voice_id": v.get("voice_id") or v.get("voiceID") or v.get("id"),
                        "name": v.get("name") or "voice",
                        "labels": v.get("labels") or {},
                    })
            return out
    except Exception:
        pass
    return []


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
    return h.hexdigest()

@st.cache_data(show_spinner=False, ttl=1800, max_entries=64)
def _tts_eleven_cached(text: str, voice_id: str, model: str, api_key: str):
    """
    Generuje audio przez ElevenLabs.
    Zwraca krotkę: (audio_bytes, err_msg) – gdy OK, err_msg == "".
    """
    vid = (voice_id or "").strip()
    if not vid:
        return b"", "Brak voice_id (uzupełnij w .env lub w sidebarze)."

    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
        payload = {
            "text": text,
            "model_id": model,
            "voice_settings": {"stability": 0.55, "similarity_boost": 0.65},
        }
        r = requests.post(
            url,
            json=payload,
            headers={
                "xi-api-key": api_key,
                "accept": "audio/mpeg",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        if r.ok and r.content:
            return r.content, ""
        # spróbuj odczytać komunikat z API
        try:
            j = r.json()
            msg = j.get("detail") or j.get("message") or str(j)
        except Exception:
            msg = r.text[:400]
        return b"", f"HTTP {r.status_code}: {msg}"
    except requests.RequestException as e:
        return b"", f"Network error: {e}"
    except Exception as e:
        return b"", f"Unexpected error: {e}"


def _validate_eleven_voice_id(voice_id: str, api_key: str) -> tuple[bool, str]:
    vid = (voice_id or "").strip()
    if not vid:
        return False, "Brak voice_id."
    if not ELEVEN_VOICE_ID_RE.match(vid):
        return False, "Podejrzany format voice_id (sprawdź literówki/ucięcie)."
    voices = _eleven_list_voices_cached(api_key)
    if not voices:
        return False, "Brak dostępu do listy głosów (sprawdź ELEVENLABS_API_KEY / plan)."
    if any(v.get("voice_id") == vid for v in voices):
        return True, ""
    return False, "Nie znaleziono takiego voice_id na Twoim koncie."

def _run_tts_for_summary(
    text_for_tts: str,
    provider: str,
    openai_tts_model_selected: str,
    openai_voice_selected: str,
    eleven_tts_model_selected: str,
    eleven_voice_id_selected: str,
    cur_tts_hash: str | None = None,
):
    """
    Generuje audio do podsumowania (AI) i odpala autoplay.
    Używana zarówno przy pierwszym generowaniu TL;DR (w jednym spinnerze),
    jak i przy ręcznym odświeżeniu TTS niżej.
    """
    text_for_tts = (text_for_tts or "").strip()
    if not text_for_tts:
        return

    # jeśli nie podano hash-a, policz tutaj (musi być spójny z blokiem niżej)
    if cur_tts_hash is None:
        cur_tts_hash = _hash_key(
            text_for_tts,
            provider or "",
            openai_tts_model_selected or "",
            openai_voice_selected or "",
            eleven_tts_model_selected or "",
            eleven_voice_id_selected or "",
        )

    # Bezpieczniki na klucze/API
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    eleven_key = os.getenv("ELEVENLABS_API_KEY", "").strip()

    # Trace wspólny dla obu dostawców (opcjonalny – tylko jeśli Langfuse jest dostępny)
    lf_client = None
    if "get_langfuse" in globals():
        try:
            lf_client = get_langfuse()
        except Exception:
            lf_client = None

    tts_trace = (
        lf_client.trace(
            name="eda_tts",
            user_id=st.session_state.get("wf_session_id", "anon"),
            input=text_for_tts[:2000],
            metadata={
                "provider": provider,
                "openai_model": openai_tts_model_selected if provider == "OpenAI" else None,
                "eleven_model": eleven_tts_model_selected if provider == "ElevenLabs" else None,
            },
        )
        if lf_client
        else None
    )

    audio_bytes, err = b"", ""

    try:
        if provider == "OpenAI":
            if not openai_key:
                err = "Brak OPENAI_API_KEY w .env"
            else:
                voice = openai_voice_selected or DEFAULT_FEMALE_VOICE
                audio_bytes, err = _tts_openai_cached(
                    text_for_tts,
                    voice=voice,
                    model=openai_tts_model_selected,
                    api_key=openai_key,
                )
        else:
            if not eleven_key:
                err = "Brak ELEVENLABS_API_KEY w .env"
            else:
                ok, msg = _validate_eleven_voice_id(eleven_voice_id_selected or "", eleven_key)
                if not ok:
                    err = f"ElevenLabs voice_id niezweryfikowany: {msg}"
                else:
                    audio_bytes, err = _tts_eleven_cached(
                        text_for_tts,
                        voice_id=eleven_voice_id_selected,
                        model=eleven_tts_model_selected,
                        api_key=eleven_key,
                    )

        if audio_bytes:
            _autoplay_audio(audio_bytes, mime="audio/mpeg")
            st.session_state["tts_last_hash"] = cur_tts_hash
            if tts_trace:
                tts_trace.update(status="success", metadata={"bytes": len(audio_bytes)})
        else:
            st.warning(f"Nie udało się wygenerować TTS ({provider}): {err}")
            if tts_trace:
                tts_trace.update(status="error", metadata={"error": err})
    finally:
        # resetujemy jednorazowy trigger
        st.session_state["play_tts_now"] = False

@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _eleven_list_models_cached(api_key: str):
    """Zwraca listę modeli ElevenLabs lub [] przy błędzie/ braku uprawnień."""
    try:
        r = requests.get(
            "https://api.elevenlabs.io/v1/models",
            headers={"xi-api-key": api_key, "accept": "application/json"},
            timeout=20,
        )
        if r.ok:
            j = r.json() or []
            # Ujednolicamy strukturę: każdy element ma 'model_id' i label 'name'
            out = []
            for m in j:
                if isinstance(m, dict):
                    out.append({
                        "model_id": m.get("model_id") or m.get("id") or m.get("name"),
                        "name": m.get("name") or m.get("model_id") or m.get("id"),
                    })
            return out
    except Exception:
        pass
    return []


@st.cache_data(show_spinner=False, ttl=1800, max_entries=64)
def _tts_openai_cached(text: str, voice: str, model: str, api_key: str):
    # zwraca (audio_bytes, err)
    try:
        import openai as _openai
        client = _openai.OpenAI(api_key=api_key)
        # Speech API (nowy SDK)
        res = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        return res.read(), ""
    except Exception as e:
        return b"", f"OpenAI TTS error: {e}"

@st.cache_data(show_spinner=False, ttl=1800, max_entries=64)
def _prep_anomaly_for_column(series: pd.Series, max_points_param: int = 5000):
    """Szybkie przygotowanie danych i zakresów Y do sekcji 6 (bez użycia zmiennych globalnych)."""
    s = pd.to_numeric(series, errors="coerce")
    temp_df = pd.DataFrame({"idx": np.arange(len(s)), "value": s}).dropna()

    if temp_df.empty:
        return {"empty": True}

    # sampling wyłącznie do WIZUALIZACJI (metryki licz na pełnym temp_df)
    if len(temp_df) > max_points_param:
        temp_df_vis = temp_df.sample(n=max_points_param, random_state=0).sort_values("idx")
    else:
        temp_df_vis = temp_df

    q1 = temp_df["value"].quantile(0.25)
    q3 = temp_df["value"].quantile(0.75)
    iqr = q3 - q1
    lower_o, upper_o = (q1, q3) if iqr == 0 else (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    temp_df["is_outlier"] = ((temp_df["value"] < lower_o) | (temp_df["value"] > upper_o)).astype(int)
    median_v = float(temp_df["value"].median())

    eps = max(1e-6, 0.01 * max(1.0, float(temp_df["value"].max()) - float(temp_df["value"].min())))
    y_min = float(min(0.0, temp_df["value"].min() - eps))
    y_max = float(max(0.0, temp_df["value"].max() + eps))

    return {
        "empty": False,
        "temp_df": temp_df,
        "temp_df_vis": temp_df_vis,  # jeśli zechcesz użyć
        "median": median_v,
        "y_min": y_min,
        "y_max": y_max,
    }

# ====================   MAIN VIEW   ======================
def _is_categorical_like(s: pd.Series, max_unique_abs: int = 50, max_unique_ratio: float = 0.3) -> bool:
    ser = s.astype("object")
    n = ser.nunique(dropna=True)
    return (n <= max_unique_abs) and (n / max(len(ser), 1) <= max_unique_ratio)

def _coerce_numeric_safe(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s): 
        return s
    return pd.to_numeric(_coerce_to_numeric_special(s), errors="coerce")

def _guess_outcome_group_value_cols(df: pd.DataFrame) -> tuple[str|None, str|None, str|None]:
    cols = list(df.columns)
    # kandydaci
    cat_cols = [c for c in cols if _is_categorical_like(df[c])]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) or _coerce_numeric_safe(df[c]).notna().any()]
    # outcome: binarne / niska krotność
    binary = [c for c in cat_cols if df[c].nunique(dropna=True) == 2]
    prio_names = ["target","label","y","survived","churn","default","outcome"]
    def pick_by_name(cands: list[str]) -> str|None:
        lower = {c.lower(): c for c in cands}
        for p in prio_names:
            if p in lower: return lower[p]
        return cands[0] if cands else None

    outcome = pick_by_name(binary) or pick_by_name(cat_cols) or (cols[0] if cols else None)
    group   = pick_by_name([c for c in cat_cols if c != outcome]) or (cols[1] if len(cols) > 1 else outcome)
    value   = pick_by_name([c for c in num_cols if c not in {outcome, group}]) or (num_cols[0] if num_cols else None)
    return outcome, group, value

def _save_datachat_handoff(
    df_ready: pd.DataFrame,
    latest_info: Dict[str, Any],
    summary_text: str,
    pairs_sorted: List[Dict[str, Any]],
    prep_report_path: str,
) -> str:
    """
    Tworzy pakiet startowy dla etapu Data Chat i zapisuje go do JSON.
    Zawiera: ścieżki, metadane, wybrane kolumny (outcome/group/value), wartość docelową,
    oraz przykładową parę skorelowaną używaną w insightach.
    """
    # gdzie zapisać
    run_dir = latest_info.get("run_dir") or os.path.dirname(latest_info.get("csv_path", ".")) or "."
    os.makedirs(run_dir, exist_ok=True)
    ready_csv_path = os.path.join(run_dir, "ready_for_training.csv")
    handoff_path   = os.path.join(run_dir, "datachat_handoff.json")

    # spróbuj domyślnie wytypować outcome/group/value z już przygotowanego zbioru
    out_col, grp_col, val_col = _guess_outcome_group_value_cols(df_ready)

    # bezpieczna wartość docelowa: '1' jeżeli jest w danych, inaczej tryb
    import statistics
    outcome_series = df_ready[out_col].astype(str) if out_col in df_ready.columns else pd.Series([], dtype=str)
    unique_vals = sorted(outcome_series.dropna().unique().tolist())
    if "1" in unique_vals:
        target_val = "1"
    else:
        try:
            target_val = str(statistics.mode(outcome_series.dropna().tolist())) if not outcome_series.empty else ""
        except statistics.StatisticsError:
            target_val = unique_vals[0] if unique_vals else ""

    # przykład silnej korelacji używany w podsumowaniu
    example_pair = next((p for p in pairs_sorted if abs(float(p.get("corr", 0))) >= 0.9), None)
    example_pair = {
        "col1": example_pair["col1"],
        "col2": example_pair["col2"],
        "r": float(example_pair["corr"])
    } if example_pair else None

    payload = {
        "version": 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ready_csv_path": ready_csv_path,
        "prep_report_path": prep_report_path,
        "n_rows": int(df_ready.shape[0]),
        "n_cols": int(df_ready.shape[1]),
        "columns": list(map(str, df_ready.columns)),
        "selected": {
            "outcome_col": out_col,
            "group_col": grp_col,
            "value_col": val_col,
            "target_val": target_val,
        },
        # Surowy tekst podsumowania — Data Chat może na jego podstawie dobrać pierwsze wykresy
        "summary_text": (summary_text or "").strip(),
        # Dodatkowe wskazówki do wstępnych wizualizacji
        "hints": {
            "example_high_corr_pair": example_pair,
            "prefer_share_chart_for_groups": True,   # pierwszy wykres: udział outcome==target per grupa
            "prefer_box_and_hist_for_value": True,   # drugi: box + histogram/KDE dla value
        },
    }

    with open(handoff_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # dostęp z innych zakładek
    st.session_state["datachat_handoff"] = payload
    st.session_state["datachat_handoff_path"] = handoff_path
    return handoff_path


def main():
    st.title("Automat EDA — szybka diagnostyka danych (Etap 2)")

    # Langfuse: identyfikator sesji użytkownika
    if "wf_session_id" not in st.session_state:
        st.session_state["wf_session_id"] = str(uuid4())

    lf = get_langfuse()

    # --- FLASH MESSAGE po rerunie ---
    _flash = st.session_state.pop("_flash_msg", None)
    if _flash:
        # Możesz wybrać formę: toast (zanika po kilku sekundach) albo klasyczny alert
        if _flash.get("use_toast", True):
            st.toast(_flash.get("text", ""), icon="✅" if _flash.get("type") == "success" else "⚠️")
        else:
            level = _flash.get("type", "info")  # "success" | "warning" | "error" | "info"
            getattr(st, level)(_flash.get("text", ""))


    st.markdown(
        """
        Ten moduł automatycznie podsumowuje dane z kroku
        **„Przelicz na całości i zapisz artefakty”** (zakładka **Analiza Danych**).

        **Co dostajesz:**\n
        • diagnozę jakości danych zanim zaczniesz trenować model,  
        • automatyczne czyszczenie (usunięcie śmieci, uzupełnienie braków, rozbicie dat, flagi outlierów),  
        • gotowy zestaw ready_for_training.csv,  
        • możliwość przejścia do **Trenowanie Modelu** praktycznie bez ręcznej roboty.
        """
    )


    # ───────────────────────── SIDEBAR: Lektor & TL;DR ─────────────────────────
    with st.sidebar.expander("🎙️ Lektor & TL;DR", expanded=False):
        fast_sidebar = st.checkbox("🧠 Generuj TL;DR (OpenAI)", value=True,
                                  help="4–6 zwięzłych zdań na bazie metryk i wniosków")
        openai_tldr_model = st.selectbox("OpenAI TL;DR model", ["gpt-4o-mini", "gpt-4o"], index=0)
        
        # defensywne domyślne wartości; dzięki temu zmienne zawsze istnieją
        openai_voice_selected = None
        openai_tts_model_selected = OPENAI_TTS_MODELS[0]
        eleven_voice_id_selected = None
        eleven_tts_model_selected = os.getenv("ELEVEN_TTS_MODEL", "eleven_multilingual_v2")

        st.markdown("---")
        st.checkbox("✅ Włącz lektora (TTS)", value=True, key="tts_enabled")

        provider = st.radio("Dostawca TTS", ["OpenAI", "ElevenLabs"], index=0)


        gender = st.radio("Głos", ["Kobieta", "Mężczyzna"], horizontal=True, index=0)

        # Gdy użytkownik zmieni dostawcę lub głos – skasuj hash, by wymusić ponowny odczyt
        _sidebar_sig = _hash_key(provider or "", str(openai_voice_selected or ""), str(eleven_voice_id_selected or ""))
        if st.session_state.get("_sidebar_sig") != _sidebar_sig:
            st.session_state["_sidebar_sig"] = _sidebar_sig
            st.session_state.pop("tts_last_hash", None)

        if provider == "OpenAI":
            gender_key = "female" if gender == "Kobieta" else "male"
            voice_pool = OPENAI_VOICES[gender_key]
            default_voice = DEFAULT_FEMALE_VOICE if gender == "Kobieta" else DEFAULT_MALE_VOICE
            if default_voice not in voice_pool:
                default_voice = voice_pool[0]
            openai_voice_selected = st.selectbox("OpenAI voice", options=voice_pool,
                                                 index=voice_pool.index(default_voice))
            if len(OPENAI_TTS_MODELS) > 1:
                openai_tts_model_selected = st.selectbox("OpenAI TTS model", options=OPENAI_TTS_MODELS, index=0)
            else:
                openai_tts_model_selected = OPENAI_TTS_MODELS[0]
                st.caption(f"OpenAI TTS model: **{openai_tts_model_selected}**")
        else:
            VOICE_FEMALE_ID = os.getenv("VOICE_FEMALE_ID", "").strip()
            VOICE_MALE_ID   = os.getenv("VOICE_MALE_ID", "").strip()
            api_key_el      = os.getenv("ELEVENLABS_API_KEY", "").strip()
            voices = _eleven_list_voices_cached(api_key_el)
            voice_map, female_keys, male_keys = {}, [], []
            if voices:
                for v in voices:
                    vid = v.get("voice_id") or ""
                    name = v.get("name") or "voice"
                    lab = v.get("labels", {}) or {}
                    g = (lab.get("gender") or "").lower()
                    label = f"{name} · {vid[:6]}"
                    voice_map[label] = vid
                    (female_keys if g == "female" else male_keys if g == "male" else []).append(label)
            if voice_map:
                options = (female_keys or list(voice_map.keys())) if gender == "Kobieta" else (male_keys or list(voice_map.keys()))
                default_vid = VOICE_FEMALE_ID if gender == "Kobieta" else VOICE_MALE_ID
                try:
                    default_label = next(k for k, v in voice_map.items() if v == default_vid)
                except StopIteration:
                    default_label = options[0]
                idx = options.index(default_label) if default_label in options else 0
                selected_label = st.selectbox("ElevenLabs voice", options=options, index=idx)
                eleven_voice_id_selected = voice_map[selected_label]
            else:
                eleven_voice_id_selected = st.text_input(
                    "ElevenLabs voice_id",
                    value=(VOICE_FEMALE_ID if gender == "Kobieta" else VOICE_MALE_ID),
                    help="Wklej voice_id z panelu ElevenLabs"
                )
                valid_voice, msg = _validate_eleven_voice_id(eleven_voice_id_selected, api_key_el)
                if valid_voice:
                    st.caption("✅ Głos zweryfikowany na Twoim koncie ElevenLabs.")
                else:
                    st.warning(f"⚠️ ElevenLabs voice_id niezweryfikowany: {msg}")
            
            models = _eleven_list_models_cached(api_key_el)
            if models:
                model_map = { (m.get("name") or m.get("model_id")): m.get("model_id") for m in models }
                names = list(model_map.keys())
                pref  = os.getenv("ELEVEN_TTS_MODEL", "eleven_multilingual_v2")
                idx   = names.index(pref) if pref in names else 0
                chosen = st.selectbox("ElevenLabs TTS model", options=names, index=idx)
                eleven_tts_model_selected = model_map[chosen]
            else:
                eleven_tts_model_selected = st.text_input(
                    "ElevenLabs TTS model",
                    value=os.getenv("ELEVEN_TTS_MODEL", "eleven_multilingual_v2")
                )
                valid_voice, msg = _validate_eleven_voice_id(eleven_voice_id_selected, api_key_el)
                if valid_voice:
                    st.caption("✅ Głos zweryfikowany na Twoim koncie ElevenLabs.")
                else:
                    st.warning(f"⚠️ ElevenLabs voice_id niezweryfikowany: {msg}")

    # 0) Wczytanie danych
    df, latest_info, err = _load_latest_dataset()
    if err:
        st.error(err); st.stop()

    # ➜ na potrzeby Sekcji 2–6 dorzuć warianty liczbowe (np. Czas_sec)
    df = _augment_numeric_derivatives_for_ui(df)

    # Reset widoczności sekcji 7, gdy zmienił się zbiór
    csv_path = latest_info.get("csv_path", "?")
    sig_now = _sec7_signature(csv_path, df)
    if st.session_state.get("sec7_signature") != sig_now:
        _reset_sec7_state()
        st.session_state["sec7_signature"] = sig_now

    # ───────────────────────── Sekcje 1–6 (wracają do main) ─────────────────────────
    run_dir    = latest_info.get("run_dir", "(brak)")
    source_name = latest_info.get("source_name", os.path.basename(csv_path) or "(brak nazwy źródła)")
    n_rows_raw = latest_info.get("n_rows", df.shape[0])
    n_cols_raw = latest_info.get("n_cols", df.shape[1])
    mask_pii   = latest_info.get("pii_masked", True)
    timestamp  = latest_info.get("timestamp", "(brak)")

    global_missing_pct = _calc_global_missing_pct(df)
    info_df, high_null_cols = _analyze_columns(df)
    corr_chart, pairs_sorted, corr_drop_suggestions = _build_correlation_report(df, info_df, threshold=0.9)
    pairs_df_full = _pairs_dataframe(pairs_sorted)
    dups_info = _detect_duplicates(df)
    duplicates_count = dups_info["duplicates_count"]
    duplicates_pct   = dups_info["duplicates_pct"]

    auto_drop_candidates = set(info_df.loc[info_df["drop_candidate"], "kolumna"].tolist())
    auto_drop_candidates.update(corr_drop_suggestions)
    auto_drop_candidates = sorted(list(auto_drop_candidates))
    predicted_cols_after_drop = max(0, n_cols_raw - len(auto_drop_candidates))

    readiness_score = _compute_readiness_score(
        high_null_cols=high_null_cols,
        duplicates_pct=duplicates_pct,
        auto_drop_candidates=auto_drop_candidates,
        pairs_sorted=pairs_sorted,
    )

    quality_flags = []
    if len(high_null_cols) > 0:
        quality_flags.append("⚠ Występują kolumny z dużą liczbą braków (>30%).")
    if duplicates_count > 0:
        quality_flags.append(f"⚠ Znaleziono {duplicates_count} potencjalnych duplikatów ({duplicates_pct}%).")
    if len(auto_drop_candidates) > 0:
        quality_flags.append("⚠ Część kolumn wygląda na zbędne / prawie puste / duplikujące sygnał.")
    if not quality_flags:
        quality_flags.append("✅ Dane wyglądają stabilnie. Możesz praktycznie od razu przejść do trenowania modelu.")

    # 1) Status zbioru
    st.header("1. Status zbioru")

    # — 1A. Rząd metryk
    colA, colB, colC, colD, colE = st.columns(5)
    with colA:
        st.markdown(_metric_card_html(
            "Wiersze × Kolumny",
            f"{_fmt_int(n_rows_raw)} × {_fmt_int(n_cols_raw)}",
            "Rozmiar surowego zbioru"
        ), unsafe_allow_html=True)
    with colB:
        color_missing = "#28a745" if global_missing_pct < 10 else ("#ffc107" if global_missing_pct < 30 else "#dc3545")
        st.markdown(_metric_card_html(
            "Braki danych (globalnie)",
            f"{global_missing_pct:.1f}%",
            "Imputacja medianą / trybem / sensowną datą",
            fg=color_missing
        ), unsafe_allow_html=True)
    with colC:
        color_dup = "#28a745" if duplicates_pct < 1 else ("#ffc107" if duplicates_pct < 5 else "#dc3545")
        st.markdown(_metric_card_html(
            "Potencjalne duplikaty",
            f"{_fmt_int(duplicates_count)} ({duplicates_pct:.1f}%)",
            "Możemy je automatycznie usunąć",
            fg=color_dup
        ), unsafe_allow_html=True)
    with colD:
        risk_cols = len(auto_drop_candidates)
        st.markdown(_metric_card_html(
            "Kolumny ryzykowne",
            _fmt_int(risk_cols),
            f"Po czyszczeniu przewidujemy ok. {_fmt_int(predicted_cols_after_drop)} kolumn",
            fg=("#dc3545" if risk_cols > 0 else "#28a745")
        ), unsafe_allow_html=True)
    with colE:
        color_ready = "#28a745" if readiness_score >= 85 else ("#ffc107" if readiness_score >= 60 else "#dc3545")
        st.markdown(_metric_card_html(
            "Data Readiness Index",
            f"{readiness_score} / 100",
            "Czy można już trenować model bez wstydu",
            fg=color_ready
        ), unsafe_allow_html=True)

    # — 1B. Jeden baner pod metrykami (CTA, bez dodatkowych boxów nad/poniżej)
    if readiness_score >= 85:
        banner = _status_banner_html(
            "ok",
            "Dane wyglądają stabilnie. Przejdź do dalszych sekcji i zobacz jak **automatycznie przygotujemy** dane dla modelu predykcyjnego."
        )
    elif readiness_score >= 60:
        banner = _status_banner_html(
            "warn",
            "Widzimy podwyższone braki/duplikaty/ryzykowne kolumny – **naprawimy to automatycznie** dla modelu predykcyjnego w sekcji **7. Przygotowanie danych**."
        )
    else:
        banner = _status_banner_html(
            "err",
            "Dane nie wyglądają dobrze. **Nie martw się** – w sekcji **7. Przygotowanie danych** oczyścimy je automatycznie i przygotujemy bezpieczny zbiór dla modelu predykcyjnego."
        )
    st.markdown(banner, unsafe_allow_html=True)

    # — 1C. Krótka stopka źródła (zostawiamy)
    st.caption(
        f"Źródło danych: **{source_name}** · Plik: {os.path.basename(csv_path)} · "
        f"Maskowanie PII: {'tak' if mask_pii else 'nie'} · Ostatnia aktualizacja: {timestamp}"
    )

    with st.expander("Szczegóły techniczne (dla zespołu data science)"):
        st.markdown(
            _tech_details_html(
                csv_path=csv_path, run_dir=run_dir, mask_pii=mask_pii,
                global_missing_pct=global_missing_pct, duplicates_pct=duplicates_pct,
                auto_drop_candidates=auto_drop_candidates,
                predicted_cols_after_drop=predicted_cols_after_drop
            ),
            unsafe_allow_html=True,
        )

    # 2) Podgląd danych
    st.header("2. Podgląd danych")
    st.caption("Szybki rzut oka na zawartość zbioru. To są dane już po anonimizacji PII (jeśli była włączona).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Pierwsze wiersze (head)", divider="gray")
        st.dataframe(df.head(10), width="stretch", hide_index=True)
    with c2:
        st.subheader("Ostatnie wiersze (tail)", divider="gray")
        st.dataframe(df.tail(10), width="stretch", hide_index=True)

    # 3) Jakość kolumn
    st.header("3. Jakość kolumn")

    # — przygotuj treść banera (żółty/g zielony), ale NIE wyświetlaj go jeszcze —
    if len(high_null_cols) > 0:
        _null_banner_kind = "warn"
        _null_banner_text = (
            "Kolumny z dużą liczbą braków (>30%): "
            + ", ".join(high_null_cols)
            + ". Przy czyszczeniu uzupełnimy je lub usuniemy, jeśli są praktycznie puste/ID."
        )
    else:
        _null_banner_kind = "ok"
        _null_banner_text = "Brak kolumn z krytycznie dużą liczbą braków (>30%)."

    left3, right3 = st.columns([1, 1])

    # --------- LEWA: tabela z flagami (bez zmian) ---------
    with left3:
        info_view_cols = ["kolumna", "logical_type", "null_pct", "n_unique", "comment"]
        df_show = info_df[info_view_cols].copy()

        icon_map = info_df.set_index("kolumna")["flags"].to_dict()
        df_show["comment"] = [
            ((icon_map.get(r.kolumna, "") + " " if icon_map.get(r.kolumna) else "") + (r.comment or ""))
            for r in df_show.itertuples()
        ]

        ui_lang = (st.session_state.get("ui_lang", "PL") or "PL").upper()
        if ui_lang == "PL":
            header_map = {
                "kolumna": "Kolumna",
                "logical_type": "Typ logiczny",
                "null_pct": "% braków",
                "n_unique": "Liczba unikalnych",
                "comment": "Komentarz",
            }
        else:
            header_map = {
                "kolumna": "column",
                "logical_type": "logical_type",
                "null_pct": "% missing",
                "n_unique": "unique values",
                "comment": "comment",
            }

        df_show_sorted = df_show.sort_values("kolumna")
        df_vis = df_show_sorted.rename(columns=header_map)

        st.dataframe(
            df_vis.style.format({header_map["null_pct"]: "{:.1f}"}),
            width="stretch",
            hide_index=True,
        )

        st.markdown("**Legenda flag:**  \n"
                    "🗑 kandydat do usunięcia (ID/stała/prawie pusta)  \n"
                    "🔢 / 📅 automatyczna konwersja typu (tekst→liczba / tekst→data+cechy)  \n"
                    "📝 długie opisy tekstowe (raczej NLP niż klasyczny ML)")

    # --------- PRAWA: wykres braków + legenda kolorów (bez zmian) ---------
    with right3:
        null_plot_df = info_df[["kolumna", "null_pct"]].copy()
        null_plot_df["null_pct"] = null_plot_df["null_pct"].astype(float)
        bar_missing = (
            alt.Chart(null_plot_df).mark_bar()
            .encode(
                x=alt.X("null_pct:Q", title="% braków"),
                y=alt.Y("kolumna:N", sort="-x", title="kolumna"),
                tooltip=[alt.Tooltip("kolumna:N", title="kolumna"),
                        alt.Tooltip("null_pct:Q", title="% braków", format=".1f")],
                color=alt.condition(alt.datum.null_pct > 30, alt.value("#dc3545"), alt.value("#007bff")),
            )
            .properties(height=320, title="Braki danych w kolumnach (% braków)")
        )
        st.altair_chart(bar_missing, use_container_width=True)

        legend_html = """
        <div style="margin-top:.5rem;">
        <div style="font-weight:600; margin-bottom:.25rem;">Objaśnienie kolorów:</div>
        <ul style="list-style:none; padding-left:0; margin:0;">
            <li style="display:flex; align-items:center; gap:.5rem; margin:.15rem 0;">
            <span style="display:inline-block; width:18px; height:10px; border-radius:999px; background:#007bff;"></span>
            <span>kolumna w normie</span>
            </li>
            <li style="display:flex; align-items:center; gap:.5rem; margin:.15rem 0;">
            <span style="display:inline-block; width:18px; height:10px; border-radius:999px; background:#dc3545;"></span>
            <span>kolumna z podwyższonym ryzykiem (&gt;30%)</span>
            </li>
        </ul>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

    # --------- TERAZ wyświetlamy baner z brakami POD legendą ---------
    if _null_banner_kind == "warn":
        st.warning(_null_banner_text)
    else:  # "ok"
        st.success(_null_banner_text)


    st.info("Kolumny prawie puste, stałe lub wyglądające jak ID oznaczyliśmy jako kandydatów do usunięcia 🗑. "
            "Zajmiemy się nimi automatycznie w kroku 7, żeby model nie uczył się śmieci.")

    # 4) Analiza wybranej kolumny
    st.header("4. Analiza wybranej kolumny")
    st.caption("Zbadaj konkretną kolumnę: rozkład, odstające wartości, dominujące kategorie, pokrycie TOP segmentów.")
    available_cols = list(df.columns)
    col_to_plot = st.selectbox("Wybierz kolumnę do analizy:", options=available_cols, index=0 if available_cols else None)

    if col_to_plot:
        s_col = df[col_to_plot]

        # jeśli nie jest numeryczna, spróbuj z naszych parserów specjalnych
        if not pd.api.types.is_numeric_dtype(s_col):
            try_coerced = _coerce_to_numeric_special(s_col)
            if pd.notna(try_coerced).mean() > 0.8:  # wystarczająco dużo parsuje się na liczbę
                s_col = try_coerced

        # po tej próbie decydujemy o ścieżce
        if pd.api.types.is_numeric_dtype(s_col):
            st.caption("Czy to wygląda zdrowo?")
            combo_chart, stats_table, comment_text, details = _numeric_distribution_details(s_col, col_to_plot)
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                st.altair_chart(combo_chart, use_container_width=True)
            with cc2:
                st.subheader("Metryki kolumny", divider="gray")
                st.dataframe(stats_table, hide_index=True, use_container_width=True)
            outlier_ratio = details.get("outlier_ratio_pct", 0.0)
            if outlier_ratio > 5.0:
                st.warning("Widzimy wartości odstające (czerwone). Zostaną oznaczone flagą `is_outlier_*`. Nie skasujemy ich bez pytania.")
            else:
                st.success("Ta kolumna wygląda stabilnie — model dostanie czysty sygnał.")
            st.info(comment_text)
        else:
            # (oryginalna ścieżka kategoryczna)
            unique_vals = sorted(list(pd.Series(s_col.astype(str)).replace("nan", np.nan).dropna().unique()))

            with st.expander("Filtruj kategorie (opcjonalnie)"):
                selected = st.multiselect("Wybierz kategorie", options=unique_vals, default=unique_vals[:min(10, len(unique_vals))])
            ser_for_plot = s_col[s_col.astype(str).isin(selected)] if selected else s_col
            bar_chart_top, coverage_chart, freq_table, comment_text = _categorical_distribution_details(ser_for_plot, col_to_plot, top_n=20)
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                st.altair_chart(bar_chart_top, use_container_width=True)
            with cc2:
                st.altair_chart(coverage_chart, use_container_width=True)
                with st.expander("Tabela TOP kategorii (dla analityków)"):
                    st.dataframe(freq_table, hide_index=True, use_container_width=True)
            st.info(comment_text)

        # ───────────────────── Porównanie rozkładu wg kategorii ─────────────────────

        # ── 4X) Cross-Filter (All × All × All) — stabilnie ─────────────────────────────
        ui_lang = (st.session_state.get("ui_lang", "PL") or "PL").upper()
        _ico = "🔎"
        expander_label = (
            f"{_ico} Analiza wielowymiarowa (3D) — filtr → grupa → miara"
            if ui_lang == "PL"
            else f"{_ico} Multidimensional analysis (3D) — filter → group → measure"
        )

        # 1) Niewielki badge/CTA nad expanderem (w 100% stabilny)
        st.markdown("""
        <style>
        .xfl-cta {
        display:inline-block;
        background:#fff4e5;
        border:1px solid #ffb74d;
        color:#5c3b00;
        font-weight:700;
        padding:.35rem .6rem;
        border-radius:.6rem;
        margin: .25rem 0 .25rem 0;
        font-size:0.95rem;
        }
        .xfl-cta:hover { background:#ffedd5; }
        </style>
        <span class="xfl-cta">👇 Kliknij poniżej, aby otworzyć analizę 3D ⤵️</span>
        """, unsafe_allow_html=True)

        # 2) Sam expander — bez modyfikacji (stabilny)
        with st.expander(expander_label, expanded=False):
            all_cols = list(df.columns)
            d_out, d_grp, d_val = _guess_outcome_group_value_cols(df)

            c1, c2, c3 = st.columns(3)
            with c1:
                outcome_col = st.selectbox(
                    "Wynik / filtr (dowolna kolumna)", options=all_cols,
                    index=(all_cols.index(d_out) if d_out in all_cols else 0),
                    key="xfl_outcome"
                )
            with c2:
                group_col = st.selectbox(
                    "Grupa (dowolna kolumna kategoryczna / tekstowa)", options=all_cols,
                    index=(all_cols.index(d_grp) if d_grp in all_cols else min(1, len(all_cols)-1)),
                    key="xfl_group"
                )
            with c3:
                value_col = st.selectbox(
                    "Miara liczbowa (z automatycznym rzutowaniem na float)", options=all_cols,
                    index=(all_cols.index(d_val) if d_val in all_cols else min(2, len(all_cols)-1)),
                    key="xfl_value"
                )

            # ── Normalizacje typów ────────────────────────────────────────────────────
            # grupa zawsze „tekstowo”
            g_series = df[group_col].astype(str)

            # miara: spróbuj zrzucić na numeric (użyj naszych parserów specjalnych -> fallback na to_numeric)
            v_raw = df[value_col]
            if not pd.api.types.is_numeric_dtype(v_raw):
                v_try = _coerce_to_numeric_special(v_raw)
                v_num = pd.to_numeric(v_try, errors="coerce")
            else:
                v_num = pd.to_numeric(v_raw, errors="coerce")

            # outcome: jako tekst do wyboru docelowej wartości (toleruje 0/1, True/False, 'yes' itd.)
            o_series = df[outcome_col].astype(str)
            o_values = sorted(pd.Series(o_series).dropna().unique().tolist())
            # sensowny default: '1' jeśli istnieje; inaczej pierwsza wartość
            default_outcome_val = "1" if "1" in o_values else (o_values[0] if o_values else "")
            # nowy rząd o tych samych proporcjach co rząd powyżej (3 kolumny po 1/3)
            cO1, cO2, cO3 = st.columns(3)
            with cO1:
                target_val = st.selectbox(
                    f"Wartość wyniku (= filtr docelowy) dla {outcome_col}",
                    options=o_values if len(o_values) > 0 else [""],
                    index=(o_values.index(default_outcome_val) if default_outcome_val in o_values else 0)
                )
            # wyrównanie wysokości wiersza (puste wypełniacze)
            with cO2:
                st.write("")
            with cO3:
                st.write("")

            st.markdown("---")


            # ── (A) Udział outcome==target w grupach (NA GÓRZE) ──────────────────────
            st.subheader(f"A) Udział {outcome_col} = {target_val} w grupach", divider="gray")
            tmp = pd.DataFrame({
                group_col: g_series, 
                "__hit__": (o_series == str(target_val)).astype(int)
            }).dropna()
            if tmp.empty:
                st.info("Brak danych do policzenia udziałów. Zmień selekcję.")
            else:
                rate = tmp.groupby(group_col)["__hit__"].mean().reset_index(name="rate")
                cat_order = rate[group_col].astype(str).tolist()

                bar_rate = (
                    alt.Chart(rate)
                    .mark_bar()
                    .encode(
                    x=alt.X("rate:Q", title="Udział", axis=alt.Axis(format=".0%")),
                    y=alt.Y(f"{group_col}:N", sort="-x", title=group_col),
                    color=alt.Color(f"{group_col}:N", legend=None, scale=_scale_for_categories(cat_order)),
                    tooltip=[alt.Tooltip(f"{group_col}:N"), alt.Tooltip("rate:Q", format=".1%", title="Udział")]
                    )
                    .properties(height=280)
                )
                st.altair_chart(bar_rate, use_container_width=True)
            st.markdown("---")


            # ── (B) Rozkład miary w wybranym wycinku + violin/box ────────────────────
            st.subheader("B) Rozkład miary w wycięciu (Outcome==target & wybór jednej grupy)", divider="gray")
            cA1, cA2 = st.columns([4, 1])  # mniej miejsca na presety, więcej na wykresy

            with cA2:
                pick_groups = sorted(pd.Series(g_series).dropna().unique().tolist())
                pick_group = st.selectbox(
                    f"Wybierz wartość {group_col} do A)",
                    options=pick_groups,
                    index=(pick_groups.index("female") if "female" in pick_groups else 0)
                )

                # ZOSTAWIAMY przełącznik skali (w tym Overlay), usuwamy „Kształt 2”
                dist_mode_A = st.radio("Skala rozkładu", ["Liczebność", "KDE", "Overlay"], index=0)

                # nowa etykieta
                bins_A = st.slider("Biny (histogram)", 5, 80, 30, 1, key="binsA")

            maskA = (o_series == str(target_val)) & (g_series == str(pick_group))
            dfA = pd.DataFrame({group_col: g_series[maskA], value_col: v_num[maskA]}).dropna()

            with cA1:
                if dfA.empty or dfA[value_col].dropna().empty:
                    st.info("Brak danych po filtrze A). Zmień selekcję.")
                else:
                    # ── 1) BOX (u góry, poziomy) ────────────────────────────────────
                    dfA2 = dfA.copy()
                    dfA2["__lbl__"] = f"{pick_group} • {outcome_col}={target_val}"

                    box_top = (
                        alt.Chart(dfA2)
                        .mark_boxplot(color=PALETTE_MAIN[0])
                        .encode(
                            x=alt.X(f"{value_col}:Q", title=value_col),
                            y=alt.Y("__lbl__:N", title="")   # poziomy box
                        )
                        .properties(height=120)
                    )
                    st.altair_chart(box_top, use_container_width=True)

                    # ── 2) Histogram / KDE / Overlay (na dole) ──────────────────────
                    hist = (
                        alt.Chart(dfA)
                        .mark_bar(opacity=0.85, color=PALETTE_MAIN[0])
                        .encode(
                            x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=bins_A), title=value_col),
                            y=alt.Y("count():Q", title="Liczebność"),
                            tooltip=[alt.Tooltip("count():Q", title="liczebność")]
                        )
                        .properties(height=240)
                    )

                    kde = (
                        alt.Chart(dfA)
                        .transform_density(value_col, as_=[value_col, "density"])
                        .mark_line(size=2, opacity=0.95, color=PALETTE_MAIN[1])
                        .encode(
                            x=alt.X(f"{value_col}:Q", title=value_col),
                            y=alt.Y("density:Q", title="Gęstość")
                        )
                        .properties(height=240)
                    )

                    if dist_mode_A == "Liczebność":
                        chart_bottom = hist
                    elif dist_mode_A == "KDE":
                        chart_bottom = kde
                    else:  # Overlay
                        chart_bottom = alt.layer(hist, kde).resolve_scale(y="independent")

                    st.altair_chart(chart_bottom, use_container_width=True)

            # ── (C) Rozkład miary wg grup (wszyscy) ───────────────────────────────────
            st.subheader("C) Rozkład miary wg grup (wszyscy)", divider="gray")

            # lokalne presety (tu – nie na górze)
            cLay, cSca = st.columns([1, 1])
            with cLay:
                layout = st.radio("Układ porównań", ["Osobne panele", "Overlay"], horizontal=True, index=0, key="secC_layout")
            with cSca:
                scale_choice = st.radio("Skala", ["Liczebność", "Udział w grupie", "KDE"], index=0, key="secC_scale")
                if scale_choice == "Udział w grupie":
                    st.caption("W trybie udziałów wykres jest zawsze rysowany jako 100% stacked (Overlay).")

            bins_B = st.slider("Biny (histogram)", 5, 80, 30, 1, key="binsB")

            # 🧲 opcjonalne zastosowanie filtra wyniku również w C)
            apply_outcome_in_C = st.checkbox("Zastosuj filtr wyniku (Outcome==target) także w C", value=True)
            maskC = (o_series == str(target_val)) if apply_outcome_in_C else pd.Series(True, index=df.index)

            mode_key  = "facet" if layout == "Osobne panele" else "overlay"
            scale_key = {"Liczebność": "count", "Udział w grupie": "share", "KDE": "kde"}[scale_choice]

            # 👇 przekaż wycięte serie (reagują na filtr → grupa → miara)
            chart_B, _ = _numeric_by_category_charts(
                s_num=v_num[maskC],
                s_cat=g_series[maskC],
                col_num_name=value_col,
                col_cat_name=group_col,
                categories_keep=None,
                maxbins=bins_B,
                mode=mode_key,
                scale=scale_key,
                opacity=0.85,
            )
            st.altair_chart(chart_B, use_container_width=True)

            nA = len(dfA)
            if nA < 30:
                st.warning(f"Mała próba w wycinku (n={nA}). Traktuj rozkład orientacyjnie.")

    # 5) Korelacje i redundancje
    st.header("5. Korelacje i redundancje")
    st.caption("Heatmapa ogólna, podgląd pary i rekomendacja eliminacji redundantnej kolumny.")

    top_left, top_right = st.columns([1, 1])
    with top_left:
        st.subheader("Mapa korelacji (numeryczne ↔ numeryczne)", divider="gray")
        if corr_chart is not None:
            st.altair_chart(corr_chart, use_container_width=True)
            st.caption("Każdy kwadrat to siła związku. Czerwone / ciemnoniebieskie pola = bardzo mocny związek.")
        else:
            st.info("Za mało kolumn numerycznych, żeby policzyć korelacje.")
    with top_right:
        st.subheader("Podgląd wybranej pary", divider="gray")
        if not pairs_df_full.empty:
            option_labels = [f"{r['col1']} ↔ {r['col2']} (r={r['corr']:.3f}, usuń: {r['suggest_drop']})" for _, r in pairs_df_full.iterrows()]
            picked_idx = st.selectbox(
                "Wybierz parę do podglądu:", 
                options=list(range(len(option_labels))) or [0], 
                format_func=lambda idx: option_labels[idx] if option_labels else "(brak par)"
            )

            sel_row = pairs_df_full.iloc[picked_idx]
            pair_chart = _scatter_with_trend(df, sel_row["col1"], sel_row["col2"], height=220)
            st.altair_chart(pair_chart, use_container_width=True)
            st.markdown(f"**Co to znaczy?**  \n• **{sel_row['col1']}** i **{sel_row['col2']}** są mocno powiązane (r={sel_row['corr']:.2f}).  \n"
                        f"• Najbezpieczniej wyłączyć: **{sel_row['suggest_drop']}** (więcej braków / wtórna).")
        else:
            st.success("Brak par z istotną korelacją — kolumny nie dublują sygnału.")

    bottom_left, bottom_right = st.columns([1, 1])

    with bottom_left:
        # JEDEN nagłówek (z dividerem) – bez duplikatu
        st.subheader("Pary o bardzo wysokiej korelacji (|r| ≥ 0.9)", divider="gray")
        
        # --- od tego miejsca treść LEWEGO dołu bez zmian logicznych ---
        pairs_df_local = pairs_df_full.copy()
        pairs_df_local["abs_r"] = pairs_df_local["corr"].abs()
        high_corr_df = (
            pairs_df_local.loc[pairs_df_local["abs_r"] >= 0.9]
            .sort_values("abs_r", ascending=False)
        )

        ui_lang = (st.session_state.get("ui_lang", "PL") or "PL").upper()
        if ui_lang == "PL":
            col_map = {
                "col1": "Kolumna 1",
                "col2": "Kolumna 2",
                "abs_r": "Wartość korelacji (|r|, Pearson)",
                "suggest_drop": "Propozycja wyłączenia",
            }
            empty_msg = "Brak par z |r| ≥ 0.9. Super – nie duplikujemy informacji."
            why_html = """
    <div style="border:1px solid #fff3cd;background:#fffef4;border-radius:0.5rem;
    padding:0.75rem 1rem;font-size:0.9rem;line-height:1.4;color:#856404;margin-top:0.75rem;">
    <div style="font-weight:600;margin-bottom:0.4rem;">Dlaczego to ważne i co z tym zrobić?</div>
    • Redundancja zaciemnia interpretację i może destabilizować model.<br>
    • Zostaw jedną z mocno skorelowanych kolumn – zwykle tę z mniejszą liczbą braków lub bardziej zrozumiałą biznesowo.<br>
    • Nasza sugestia jest konserwatywna (patrzymy m.in. na braki danych).
    </div>
    """
        else:
            col_map = {
                "col1": "Column 1",
                "col2": "Column 2",
                "abs_r": "|r| (Pearson)",
                "suggest_drop": "Suggested column to drop",
            }
            empty_msg = "No pairs with |r| ≥ 0.9. Great — no duplicated signal."
            why_html = """
    <div style="border:1px solid #fff3cd;background:#fffef4;border-radius:0.5rem;
    padding:0.75rem 1rem;font-size:0.9rem;line-height:1.4;color:#856404;margin-top:0.75rem;">
    <div style="font-weight:600;margin-bottom:0.4rem;">Why it matters & what to do</div>
    • Redundancy hides interpretation and may destabilize the model.<br>
    • Keep only one of the highly correlated columns — usually the one with fewer missings or clearer business meaning.<br>
    • Our suggestion is conservative (we look at missingness, etc.).
    </div>
    """

        if high_corr_df.empty:
            st.caption(empty_msg)
        else:
            # limit wysokości tabeli (żółty box ma być zaraz pod nią)
            PANE_LIMIT_PX = 700
            WHYBOX_EST_PX = 120
            TABLE_MAX_PX  = max(180, PANE_LIMIT_PX - WHYBOX_EST_PX)
            ROW_H, HDR_H, PAD_H = 36, 38, 10
            nrows = len(high_corr_df)
            needed_px = min(TABLE_MAX_PX, HDR_H + nrows * ROW_H + PAD_H)

            st.markdown('<div class="hc-scope hc-tight">', unsafe_allow_html=True)  # ⟵ KLUCZ: usuwa szczelinę

            tbl = high_corr_df[["col1", "col2", "abs_r", "suggest_drop"]].rename(columns=col_map)
            st.dataframe(
                tbl.style.format({col_map["abs_r"]: "{:.4f}"}),
                use_container_width=True, hide_index=True, height=needed_px
            )
        st.markdown(why_html, unsafe_allow_html=True)

    with bottom_right:
        st.subheader("Którą kolumnę wyłączyć? — wynik i uzasadnienie", divider="gray")
        if not pairs_df_full.empty:
            c1, c2 = sel_row["col1"], sel_row["col2"]
            # 1) Czynniki i łączny score z utili
            f1 = _col_factors(df, info_df, c1)
            f2 = _col_factors(df, info_df, c2)
            total1 = _elimination_score(f1)
            total2 = _elimination_score(f2)
            # spójna paleta dla obu wykresów w całej sekcji
            color_scale = alt.Scale(domain=[c1, c2], range=["#0b5ed7", "#6ea8fe"])

            # 2) OGÓŁ — dwa horyzontalne słupki (więcej miejsca na wartości)
            import pandas as _pd
            totals_df = _pd.DataFrame({"kolumna": [c1, c2], "łączny_score": [total1, total2]})
            order_cols = totals_df.sort_values("łączny_score", ascending=False)["kolumna"].tolist()
            max_x = float(totals_df["łączny_score"].max())
            x_scale = alt.Scale(domain=[0, max_x * 1.18], nice=False)  # +18% zapasu

            # --- Łączny score (paski horyzontalne) ---
            totals_df = pd.DataFrame({"kolumna": [c1, c2], "łączny_score": [total1, total2]})
            order_cols = totals_df.sort_values("łączny_score", ascending=False)["kolumna"].tolist()

            max_x = float(totals_df["łączny_score"].max())
            # 15% zapasu, żeby etykiety NIGDY się nie ucinały
            x_scale = alt.Scale(domain=[0, max_x * 1.15], nice=False)

            # więcej „powietrza” między paskami
            totals_bars = (
                alt.Chart(totals_df)
                .mark_bar(size=26, cornerRadiusEnd=5)
                .encode(
                    y=alt.Y("kolumna:N", sort=order_cols, title=None,
                            axis=alt.Axis(labelFontWeight=600)),
                    x=alt.X("łączny_score:Q", title="Łączny score (wyższy = gorszy)", scale=x_scale),
                    color=alt.Color("kolumna:N", legend=None,
                                    scale=alt.Scale(domain=[c1, c2], range=["#1f77b4", "#ff7f0e"])),
                    tooltip=[alt.Tooltip("kolumna:N"),
                            alt.Tooltip("łączny_score:Q", format=".3f")]
                )
                .properties(height=120)
            )

            # etykiety: normalna czcionka, trochę większa, leciutkie odsunięcie
            totals_labels = (
                alt.Chart(totals_df)
                .mark_text(align="left", dx=10, baseline="middle", fontSize=13, color="black")
                .encode(
                    y=alt.Y("kolumna:N", sort=order_cols, title=None),
                    x=alt.X("łączny_score:Q", scale=x_scale),
                    text=alt.Text("łączny_score:Q", format=".3f")
                )
            )

            totals_layer = alt.layer(totals_bars, totals_labels).properties(padding={"right": 46})
            st.altair_chart(totals_layer, use_container_width=True)

            # „Chip” Δ tuż pod wykresem, po prawej stronie
            delta = abs(total1 - total2)
            worse = totals_df.sort_values("łączny_score", ascending=False).iloc[0]["kolumna"]
            cL, cR = st.columns([1, 1.0])
            with cR:
                st.markdown(
                    f"<div style='display:inline-block;padding:.28rem .6rem;border-radius:.6rem;"
                    f"background:#e8f2ff;color:#0b57d0;font-weight:600;float:right;'>Δ = {delta:.3f} · gorsza: {worse}</div>",
                    unsafe_allow_html=True
                )
            st.write("")  # drobny odstęp pod chipem

            # 3) SZCZEGÓŁ — kontrybucje per czynnik (grupowane słupki OBOK SIEBIE)
            def factor_contribs(factors: dict[str, float], colname: str):
                keys = list(factors.keys())
                rows = []
                for k in keys:
                    only = {kk: 0.0 for kk in keys}
                    only[k] = float(factors.get(k, 0.0))
                    rows.append({"kolumna": colname, "czynnik": k, "kontrybucja": _elimination_score(only)})
                return rows

            expl = _pd.DataFrame(factor_contribs(f1, c1) + factor_contribs(f2, c2))
            # ⛔ NIE filtrujemy zer — chcemy widzieć wszystkie 7 czynników zawsze
            # keep_mask = expl.groupby("czynnik")["kontrybucja"].transform("sum") > 0  # ← usuń

            wide = expl.pivot(index="czynnik", columns="kolumna", values="kontrybucja").fillna(0.0)
            if c1 in wide.columns and c2 in wide.columns and not wide.empty:
                wide["abs_delta"] = (wide[c1] - wide[c2]).abs()
                factor_order_all = wide.sort_values("abs_delta", ascending=False).index.tolist()
                top_diff = wide["abs_delta"].idxmax()
            else:
                factor_order_all = list(f1.keys())
                top_diff = None

            # ── Kontrolki w jednej linii: lewo suwak, prawo checkbox ──
            col_left, col_right = st.columns([5, 2], vertical_alignment="center")
            with col_left:
                max_k = max(1, len(factor_order_all))
                # domyślnie pokazuj wszystkie 7 (lub ile jest)
                default_k = max_k
                top_k = st.slider("Pokaż Top-K czynników (wg różnicy) — posortowano wg |Δ|",
                                min_value=1, max_value=max_k, value=default_k)
            with col_right:
                show_pct = st.checkbox("Pokaż czynniki w %", value=False, help="Normalizacja per czynnik")

            factor_keep = factor_order_all[:top_k]

            plot_df = expl[expl["czynnik"].isin(factor_keep)].copy()
            if show_pct:
                plot_df["wartość"] = plot_df.groupby("czynnik")["kontrybucja"].transform(
                    lambda s: (s / s.sum()) if s.sum() else 0.0
                )
                y_title, y_fmt = "Udział w obrębie czynnika", ".0%"
                ref50 = alt.Chart(_pd.DataFrame({"ref": [0.5]})).mark_rule(
                    color="#6c757d", strokeDash=[6, 3], opacity=0.7
                ).encode(y=alt.Y("ref:Q", title=None))
            else:
                plot_df["wartość"] = plot_df["kontrybucja"]
                y_title, y_fmt = "Kontrybucja czynnika do łącznego score’u", ".3f"
                ref50 = None

            # --- Grouped bars: słupki obok siebie, więcej miejsca i czytelne opisy ---
            plot_df = expl[expl["czynnik"].isin(factor_keep)].copy()
            if show_pct:
                plot_df["wartość"] = plot_df.groupby("czynnik")["kontrybucja"].transform(
                    lambda s: (s / s.sum()) if s.sum() else 0.0
                )
                y_title, y_fmt = "Udział w obrębie czynnika", ".0%"
            else:
                plot_df["wartość"] = plot_df["kontrybucja"]
                y_title, y_fmt = "Kontrybucja czynnika do łącznego score’u", ".3f"

            # sam wykres
            grouped_bars = (
                alt.Chart(plot_df)
                .mark_bar(size=30)
                .encode(
                    x=alt.X("czynnik:N", title=None,
                            axis=alt.Axis(labelAngle=0, labelLimit=320, labelOverlap=False)),
                    xOffset=alt.XOffset("kolumna:N"),
                    y=alt.Y("wartość:Q", title=y_title, axis=alt.Axis(format=y_fmt)),
                    color=alt.Color("kolumna:N", legend=None,
                                    scale=alt.Scale(domain=[c1, c2], range=["#1f77b4", "#ff7f0e"])),
                    tooltip=[
                        alt.Tooltip("czynnik:N", title="czynnik"),
                        alt.Tooltip("kolumna:N", title="kolumna"),
                        alt.Tooltip("wartość:Q", title="wartość", format=y_fmt),
                    ],
                )
                .properties(height=360)   # + więcej wysokości
            )

            # etykiety na szczytach: bez bolda, większa czcionka, czarne
            grouped_labels = (
                alt.Chart(plot_df)
                .mark_text(dy=-6, fontSize=13, color="black")
                .encode(
                    x=alt.X("czynnik:N", sort=factor_keep, title=None),
                    xOffset=alt.XOffset("kolumna:N"),
                    y=alt.Y("wartość:Q"),
                    detail="kolumna:N",
                    text=alt.Text("wartość:Q", format=y_fmt),
                )
            )

            layers = [grouped_bars, grouped_labels]
            if show_pct:
                ref50 = (
                    alt.Chart(pd.DataFrame({"ref": [0.5]}))
                    .mark_rule(color="#6c757d", strokeDash=[6, 3], opacity=0.7)
                    .encode(y=alt.Y("ref:Q", title=None))
                )
                layers = [ref50] + layers

            final_chart = (
                alt.layer(*layers)
                .configure_axis(grid=True, gridOpacity=0.2, labelFontSize=12, titleFontSize=12)
                .configure_scale(bandPaddingInner=0.25, bandPaddingOuter=0.2)  # trochę „powietrza” w osi X
                .properties(padding={"left": 0, "right": 0, "top": 12, "bottom": 26})
            )
            st.altair_chart(final_chart, use_container_width=True)

            # 4) Werdykt
            suggest_row = totals_df.sort_values("łączny_score", ascending=False).iloc[0]
            suggest = suggest_row["kolumna"]
            delta = abs(total1 - total2)

            if delta < 1e-3:
                st.info("**Rekomendacja:** remis — oba profile są niemal identyczne. Sprawdź kontekst biznesowy lub inne pary.")
            else:
                if top_diff:
                    st.info(
                        f"**Rekomendacja:** wyłącz **`{suggest}`** — największa różnica na czynniku **{top_diff}** "
                        f"(i wyższy łączny score). **Wyłączenie uprości model** (mniej korelacji, mniejsza wariancja cech)."
                    )
                else:
                    st.info(
                        f"**Rekomendacja:** wyłącz raczej **`{suggest}`** — ma wyższy łączny score (gorszy profil czynników). "
                        f"**Wyłączenie uprości model** (mniej korelacji, mniejsza wariancja cech)."
                    )

            # if st.button(f"➕ Dodaj `{suggest}` do listy kolumn do wyłączenia (Sekcja 7)"):
            #     # ...tu Twój kod, który dopisuje kolumnę do listy...
            #     st.toast(f"Dodano `{suggest}` do czyszczenia w Sekcji 7 (możesz cofnąć w formularzu).", icon="✅")

        else:
            st.caption("Brak par do analizy.")

    # 6) Anomalie globalne
    st.header("6. Anomalie globalne: duplikaty i odstające wartości")

    # --- Przygotuj treść banerów, ale na razie nic nie wyświetlaj ---
    _outliers_info_text = (
        "Czerwone punkty = wartości odstające wg IQR. Nie usuwamy ich w ciemno: "
        "dodajemy flagi `is_outlier_*`. Możesz opcjonalnie przyciąć ekstremy (winsoryzacja) w sekcji 7."
    )
    if duplicates_count > 0:
        _dup_banner_kind = "warn"
        _dup_banner_text = (
            f"Wykryto potencjalne duplikaty: {_fmt_int(duplicates_count)} "
            f"({duplicates_pct:.1f}% próbek). Domyślnie usuniemy je przy czyszczeniu."
        )
    else:
        _dup_banner_kind = "ok"
        _dup_banner_text = "Nie wykryto zduplikowanych pełnych wierszy."

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.subheader("Podgląd wartości odstających — wybierz kolumnę numeryczną:", divider="gray")
        col_for_anomaly = st.selectbox("", options=numeric_cols, index=0)
        temp_series = pd.to_numeric(df[col_for_anomaly], errors="coerce")
        temp_df = pd.DataFrame({"idx": np.arange(len(temp_series)), "value": temp_series}).dropna()

        if not temp_df.empty:
            q1 = temp_df["value"].quantile(0.25); q3 = temp_df["value"].quantile(0.75)
            iqr = q3 - q1
            lower_o, upper_o = (q1, q3) if iqr == 0 else (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            temp_df["is_outlier"] = ((temp_df["value"] < lower_o) | (temp_df["value"] > upper_o)).astype(int)
            med_val = float(temp_df["value"].median())

            ui_col1, ui_col2 = st.columns(2)
            with ui_col1:
                max_points = st.slider(
                    "Maks. punkty w scatterze",
                    min_value=1000, max_value=20000, value=5000, step=500,
                    help="Sampling dotyczy TYLKO wizualizacji; metryki liczone są na pełnym zbiorze."
                )
            with ui_col2:
                show_tips = st.checkbox(
                    "Pokaż podpowiedzi (tooltips)", value=True,
                    help="Wyłącz, aby przyspieszyć rysowanie."
                )

            temp_df_vis = (
                temp_df.sample(n=max_points, random_state=42).sort_values("idx")
                if len(temp_df) > max_points else temp_df
            )
            eps   = max(1e-6, 0.01 * max(1.0, float(temp_df["value"].max()) - float(temp_df["value"].min())))
            y_min = float(min(0.0, temp_df["value"].min() - eps))
            y_max = float(max(0.0, temp_df["value"].max() + eps))
            CHART_HEIGHT = 280

            y_enc = alt.Y("value:Q", title=col_for_anomaly,
                        scale=alt.Scale(domain=[y_min, y_max], nice=False))

            median_y = float(temp_df["value"].median())
            median_rule = (
                alt.Chart(pd.DataFrame({"median_value": [median_y]}))
                .mark_rule(color="#1f77b4", strokeDash=[6, 3], strokeWidth=1.5, opacity=0.95)
                .encode(y=alt.Y("median_value:Q",
                                scale=alt.Scale(domain=[y_min, y_max], nice=False)))
            )

            base_scatter = (
                alt.Chart(temp_df_vis)
                .mark_point(filled=True, size=32, opacity=0.6)
                .encode(
                    x=alt.X("idx:Q", title="ID rekordu (kolejność w pliku)",
                            axis=alt.Axis(labelOverlap=True, labelLimit=1000)),
                    y=y_enc,
                    color=alt.condition(alt.datum.is_outlier == 1,
                                        alt.value("#dc3545"), alt.value("#007bff"))
                )
            )
            if show_tips:
                base_scatter = base_scatter.encode(
                    tooltip=[
                        alt.Tooltip("idx:Q",    title="rekord"),
                        alt.Tooltip("value:Q",  format=".2f", title="wartość"),
                        alt.Tooltip("is_outlier:N", title="outlier"),
                    ]
                )

            left_panel_final = alt.layer(base_scatter, median_rule).properties(width=900, height=CHART_HEIGHT)
            right_panel_final = alt.layer(
                alt.Chart(temp_df)
                .mark_boxplot(color="#1f77b4", outliers={"color": "#dc3545"})
                .encode(
                    y=alt.Y("value:Q",
                            scale=alt.Scale(domain=[y_min, y_max], nice=False),
                            axis=alt.Axis(title=None, labels=False, ticks=False, domain=False, grid=True))
                ),
                median_rule
            ).properties(width=100, height=CHART_HEIGHT)

            combined = (
                alt.hconcat(left_panel_final, right_panel_final, spacing=28)
                .resolve_scale(y="shared")
                .properties(spacing=10, bounds="flush",
                            padding={"left": 0, "right": 10, "top": 0, "bottom": 0})
            )
            colL, colM, colR = st.columns([0.3, 10, 3.0])
            with colM:
                st.altair_chart(combined, use_container_width=False)

    # --- Po wykresach: najpierw informacja o duplikatach, potem info o outlierach ---
    if _dup_banner_kind == "ok":
        st.success(_dup_banner_text)
    else:
        st.warning(_dup_banner_text)

    st.info(_outliers_info_text)

    # 7) Przygotowanie danych do trenowania (z tabami TL;DR)
    st.header("7. Przygotowanie danych do trenowania")

    # Delikatne zbicie odstępów tylko dla tej sekcji (bez hacków na DOM)
    st.markdown("""
    <style>
    .section-7-tight { margin-top: 0.25rem !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-7-tight">', unsafe_allow_html=True)

    # ── Stan sekcji 7 ───────────────────────────────────────────────────────────
    if "sec7_revealed" not in st.session_state:
        st.session_state["sec7_revealed"] = False
    if "latest_summary_text" not in st.session_state:
        st.session_state["latest_summary_text"] = ""

    # ── CTA do odsłonięcia sekcji 7 (PRIMARY) ──────────────────────────────────
    if not st.session_state["sec7_revealed"]:
        if st.button("✨ Zrób podsumowanie (AI)", type="primary", 
                    help="Wygeneruj narrację – odsłoni to całą sekcję 7."):
            st.session_state["sec7_revealed"] = True
            st.session_state["play_tts_now"] = True

    run_prep = False
    if st.session_state["sec7_revealed"]:
        # ── Przygotowanie promptu do TL;DR (raz) ────────────────────────────────
        if not st.session_state.get("latest_summary_text"):
            base_facts = [
                f"Liczba wierszy × kolumn: {n_rows_raw} × {n_cols_raw}",
                f"Braki danych (globalnie): ~{global_missing_pct:.1f}%",
                f"Duplikaty: {duplicates_count} ({duplicates_pct:.1f}%)",
                f"Kandydaci do wyłączenia (heurystyka): {', '.join(auto_drop_candidates) if auto_drop_candidates else 'brak'}",
                f"Pary silnie skorelowane (|r|≥0.9): {sum(1 for _ in [p for p in pairs_sorted if abs(p['corr'])>=0.9])}",
            ]
            example_pair = next((p for p in pairs_sorted if abs(p["corr"]) >= 0.9), None)
            example_txt = f"{example_pair['col1']} ↔ {example_pair['col2']} (r={example_pair['corr']:.2f})" if example_pair else "brak"
            tl_dr_prompt = (
                "Stwórz spójne podsumowanie w języku polskim, w trzech częściach, jako czysty markdown:\n"
                "### Podsumowanie danych\n- Rozmiar, poziom braków, duplikaty.\n- Wspomnij o liczbie kolumn do potencjalnego wyłączenia i mocnych korelacjach.\n"
                "### Insight z analizy\n- 2–4 kluczowe spostrzeżenia...\n"
                f"- Jeden przykład pary skorelowanej: {example_txt}.\n"
                "### Co dalej\n- Zachęta do kliknięcia „Przygotuj dane do trenowania”…\n\n"
                f"Fakty wejściowe: {base_facts}"
            )

        # ── Generowanie TL;DR tylko gdy pusto ───────────────────────────────────
        if not st.session_state.get("latest_summary_text"):
            with st.spinner("⏳ Generuję podsumowanie (AI) i narrację lektora…"):
                summary_text = ""
                trace = lf.trace(
                    name="eda_tldr",
                    user_id=st.session_state.get("wf_session_id", "anon"),
                    input=tl_dr_prompt,
                    metadata={
                        "source_name": source_name,
                        "csv_path": csv_path,
                        "n_rows": n_rows_raw,
                        "n_cols": n_cols_raw,
                        "model": openai_tldr_model,
                        "module": "02_Automat_EDA",
                    },
                ) if lf else None

                try:
                    # --- LLM: generacja podsumowania ---
                    lf_openai = get_lf_openai_client()
                    if lf_openai:
                        tl = lf_openai.chat.completions.create(
                            model=openai_tldr_model,
                            messages=[{"role": "user", "content": tl_dr_prompt}],
                            temperature=0.3,
                        )
                    else:
                        import openai as _openai
                        client = _openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
                        tl = client.chat.completions.create(
                            model=openai_tldr_model,
                            messages=[{"role": "user", "content": tl_dr_prompt}],
                            temperature=0.3,
                        )
                    summary_text = (tl.choices[0].message.content or "").strip()

                    # Fallback, gdyby LLM zwrócił pusty tekst
                    if not summary_text:
                        summary_text = _make_eda_summary_text(
                            source_name=source_name,
                            readiness_score=readiness_score,
                            duplicates_count=duplicates_count,
                            global_missing_pct=global_missing_pct,
                            auto_drop_candidates=auto_drop_candidates,
                            prep_report=None,           # <-- ważne: żadnego prep_report tutaj
                        )

                    st.session_state["latest_summary_text"] = summary_text

                    # --- TTS w tym samym spinnerze ---
                    if st.session_state.get("tts_enabled", True):
                        _run_tts_for_summary(
                            summary_text,
                            provider,
                            openai_tts_model_selected,
                            openai_voice_selected,
                            eleven_tts_model_selected,
                            eleven_voice_id_selected,
                        )

                    if trace:
                        trace.update(status="success", output=summary_text[:2000])

                except Exception as e:
                    if trace:
                        trace.update(status="error", metadata={"error": str(e)})

                    st.warning(
                        "Nie udało się wygenerować podsumowania przez AI – używam prostego fallbacku."
                    )
                    summary_text = _make_eda_summary_text(
                        source_name=source_name,
                        readiness_score=readiness_score,
                        duplicates_count=duplicates_count,
                        global_missing_pct=global_missing_pct,
                        auto_drop_candidates=auto_drop_candidates,
                        prep_report=None,       # <-- znowu: None, nieprep_report
                    )
                    st.session_state["latest_summary_text"] = summary_text
        else:
            summary_text = st.session_state["latest_summary_text"]

        # ── TL;DR: podgląd i edycja ─────────────────────────────────────────────
        st.subheader("Podsumowanie (AI)", divider="gray")
        tab_view, tab_edit = st.tabs(["📄 Podsumowanie", "✍️ Edytuj tekst i zapisz"])

        with tab_view:
            st.markdown(st.session_state.get("latest_summary_text", ""))

        with tab_edit:
            left, right = st.columns([1, 1], gap="large")
            with left:
                edited = st.text_area(
                    "Tekst narracji (markdown)",
                    value=st.session_state.get("latest_summary_text", ""),
                    key="summary_text_area",
                    height=380,
                    help="Po kliknięciu „Zapisz…” tekst trafi do summary_ai.md i natychmiast zaktualizuje Podsumowanie.",
                )
                if st.button("💾 Zapisz podsumowanie do artefaktów", type="primary"):
                    st.session_state["latest_summary_text"] = edited
                    saved_path = _save_summary_to_artifacts(edited, latest_info)
                    st.toast(f"Zapisano: {saved_path}", icon="✅")
                    st.rerun()

            with right:
                st.caption("Podgląd (markdown):")
                st.markdown(
                    st.session_state.get(
                        "summary_text_area",
                        st.session_state.get("latest_summary_text", ""),
                    )
                )
                st.caption("Podgląd aktualizuje się na bieżąco podczas pisania.")

        # --- Autoodtwarzanie TTS przy późniejszym odświeżaniu --- 
        _text_for_tts = (st.session_state.get("latest_summary_text", "") or "").strip()
        tts_enabled   = st.session_state.get("tts_enabled", True)

        cur_tts_hash = _hash_key(
            _text_for_tts,
            provider or "",
            openai_tts_model_selected or "",
            openai_voice_selected or "",
            eleven_tts_model_selected or "",
            eleven_voice_id_selected or "",
        )
        last_tts_hash    = st.session_state.get("tts_last_hash")
        explicit_trigger = st.session_state.get("play_tts_now", False)

        should_play = (
            tts_enabled
            and _text_for_tts
            and (explicit_trigger or (last_tts_hash != cur_tts_hash))
        )

        # Uwaga: gdy TTS został już wygenerowany w spinnerze wyżej,
        # tts_last_hash == cur_tts_hash → tutaj nic się nie odpala.
        if should_play:
            with st.spinner("🔊 Generuję narrację audio…"):
                _run_tts_for_summary(
                    _text_for_tts,
                    provider,
                    openai_tts_model_selected,
                    openai_voice_selected,
                    eleven_tts_model_selected,
                    eleven_voice_id_selected,
                    cur_tts_hash,
                )


        # ── Opis akcji czyszczenia ──────────────────────────────────────────────
        st.subheader("Ten krok zbuduje gotowy zbiór treningowy:", divider="gray")
        st.markdown(
            "🧹 usuniemy zbędne kolumny  \n"
            "📅 przekonwertujemy liczby i daty  \n"
            "🧱 uzupełnimy braki  \n"
            "🏷️ dodamy flagi `is_outlier_*` / winsoryzacja (opcjonalnie)  \n"
            "🧹 usuniemy duplikaty  \n"
            "💾 zapiszemy `ready_for_training.csv` + `prep_report.json`  \n"
            "⚙️ zaktualizujemy artefakty"
        )
        
        with st.form("eda_cleaning_form"):
            st.write("### Ustawienia szybkiego czyszczenia")
            remove_duplicates_user = st.checkbox("Usuń zduplikowane rekordy (zalecane)", value=True)
            basic_col_drop_default = sorted(
                set(auto_drop_candidates) | set(st.session_state.get("sec7_preselected_drop", set()))
            )

            # --- STATE: winsor verification ---
            if "winsor_verify_ready" not in st.session_state:
                st.session_state.winsor_verify_ready = False
            if "winsor_verify_params" not in st.session_state:
                st.session_state.winsor_verify_params = {}

            # 🔓 Czy expander ma startować rozwinięty?
            if "sec7_adv_expanded" not in st.session_state:
                st.session_state.sec7_adv_expanded = False  # na wejściu do sekcji jest zamknięty

            st.write("### Zaawansowane ustawienia (opcjonalnie)")
            with st.expander(
                "Pokaż / ukryj ustawienia zaawansowane",
                expanded=st.session_state.get("sec7_adv_expanded", False),
            ):

                drop_cols_user = st.multiselect(
                    "Kolumny do usunięcia:",
                    options=list(df.columns),
                    default=basic_col_drop_default,
                )
                numeric_cols_for_winsor = df.select_dtypes(include=[np.number]).columns.tolist()
                winsorize_cols_user = st.multiselect(
                    "Kolumny do przycięcia (winsoryzacja):",
                    options=numeric_cols_for_winsor,
                    default=[],
                    key="adv_wins_cols",           # <<— TEN KEY jest ważny
                    help="Podaj kolumny, które chcesz przyciąć (winsoryzacja).",
                )

                # ✅ JEDYNY przycisk weryfikacji winsoryzacji – widoczny od razu
                go = st.form_submit_button(
                    "🔎 Uruchom / odśwież weryfikację winsoryzacji",
                    type="secondary",                # szary (bezpieczny); chcesz „blade czerwone”? patrz sekcja 3 (opcjonalny CSS)
                    use_container_width=False,
                    key="wins_verify_btn",
                    help="Policz metryki i pokaż wizualizacje ‘przed vs po’ dla wybranej kolumny."
                )
                # jeśli użytkownik kliknął przycisk – od tej pory expander ma być zawsze rozwinięty
                if go:
                    st.session_state.sec7_adv_expanded = True

                # === WERYFIKACJA WINSORYZACJI — KONTYNUACJA SEKCJI „Ustawienia szybkiego czyszczenia” ===
                DF = locals().get("df_clean", locals().get("df", None))
                if DF is None:
                    st.warning("Nie znaleziono ramki danych DF (df / df_clean). Upewnij się, że zmienna z danymi istnieje.")
                else:
                    # pierwszy wybrany element z multiselecta „adv_wins_cols”
                    # próbujemy kilku możliwych nazw key, żeby nie polegać na jednej konkretnej
                    _possible_keys = [
                        "adv_wins_cols",          # moja propozycja
                        "wins_cols_to_clip",      # częsty wariant
                        "wins_cols",              # inny wariant
                        "winsorize_cols",         # inny wariant
                        "wins_cols_preview"       # bywa i tak
                    ]
                    _selected_list = []
                    for _k in _possible_keys:
                        if _k in st.session_state and st.session_state[_k]:
                            _selected_list = st.session_state[_k]
                            break

                    _selected = None
                    if isinstance(_selected_list, (list, tuple)) and len(_selected_list) > 0:
                        _selected = _selected_list[0]
                    elif isinstance(_selected_list, str) and _selected_list:
                        _selected = _selected_list

                    # ——— Helpers (jedna, zwięzła wersja) ———
                    def _to_num(s: pd.Series) -> pd.Series:
                        return pd.to_numeric(s, errors="coerce")

                    def compute_iqr_fences(s: pd.Series, k: float = 1.5):
                        s = _to_num(s).dropna()
                        if s.empty:
                            return np.nan, np.nan, s
                        q1, q3 = s.quantile(0.25), s.quantile(0.75)
                        iqr = q3 - q1
                        return float(q1 - k * iqr), float(q3 + k * iqr), s

                    def winsorize_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
                        return _to_num(s).clip(lower=lower, upper=upper)

                    def mini_stats(s: pd.Series) -> dict:
                        s = _to_num(s).dropna()
                        if s.empty:
                            return {"liczebność (n)": 0, "min": np.nan, "Q1": np.nan, "mediana": np.nan,
                                    "Q3": np.nan, "max": np.nan, "średnia": np.nan, "odch.std.": np.nan, "IQR": np.nan}
                        d = s.describe(percentiles=[.25, .5, .75])
                        return {
                            "liczebność (n)": int(s.size),
                            "min": float(d["min"]),
                            "Q1": float(d["25%"]),
                            "mediana": float(d["50%"]),
                            "Q3": float(d["75%"]),
                            "max": float(d["max"]),
                            "średnia": float(d["mean"]),
                            "odch.std.": float(d["std"]),
                            "IQR": float(d["75%"] - d["25%"]),
                        }

                    # ——— UI: POKAZUJEMY dopiero, gdy wybrano kolumnę ———
                    if _selected:
                        # Sterowanie bez żadnych KPI nad nim
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            viz_mode = st.radio(
                                "Tryb wizualizacji",
                                ["Histogram", "ECDF"],
                                index=0,
                                horizontal=True,
                                key="wins_viz_mode_qc",
                                help="ECDF = Empiryczna Dystrybuanta Skumulowana. Oś Y w [0,1]; dobrze pokazuje ‘ściśnięcie’ ogonów."
                            )
                        with c2:
                            k_iqr = st.slider(
                                "K (Tukey/IQR)", 1.0, 3.0,
                                float(st.session_state.get("wins_k_iqr_qc", 1.5)),
                                0.1, key="wins_k_iqr_qc",
                                help="K=1.5 (klasyczny Tukey). Większe K = łagodniejsze cięcie. Użyj większego K przy ciężkoogonowych rozkładach."
                            )

                        # Cała logika i rysunki – TYLKO po kliknięciu (nie ma "luźnych" KPI nad sterowaniem)
                        if go:
                            col = _selected

                            # 1) brak kolumny -> tylko komunikat
                            if not col:
                                st.warning("Najpierw wybierz kolumnę do winsoryzacji.")
                            else:
                                # 2) inicjalizacja cache'a, jeśli jeszcze go nie ma
                                if "winsor_cache" not in st.session_state:
                                    st.session_state["winsor_cache"] = {}

                                cache = st.session_state["winsor_cache"]

                                # 3) klucz cache: (nazwa kolumny, K zaokrąglone do 3 miejsc)
                                k_key = round(float(k_iqr), 3)
                                cache_key = (col, k_key)

                                # 4) jeśli mamy w cache – używamy, inaczej liczymy i zapisujemy
                                if cache_key in cache:
                                    data = cache[cache_key]
                                    lower = data["lower"]
                                    upper = data["upper"]
                                    before = data["before"]
                                    after = data["after"]
                                else:
                                    lower, upper, before = compute_iqr_fences(DF[col], k=k_iqr)
                                    after = winsorize_series(DF[col], lower, upper)

                                    cache[cache_key] = {
                                        "lower": lower,
                                        "upper": upper,
                                        "before": before,
                                        "after":  after,
                                    }

                                # 5) unikalna baza kluczy – zależy od kolumny i K (Tukey/IQR)
                                chart_key_base = f"wins_{col}_{float(k_iqr):.2f}"

                                # 6) liczebność = liczba NIE-NaN (spójnie z mini_stats / tabelą)
                                before_num = before                    # 'before' ma już dropna()
                                n_before = int(before_num.size)

                                after_num = _to_num(after).dropna()
                                n_after  = int(after_num.size)

                                # 7) outliery PRZED – poniżej L lub powyżej U
                                below = int((before_num < lower).sum())
                                above = int((before_num > upper).sum())
                                out_before = below + above

                                # 8) outliery PO – teoretycznie powinny być ≈ 0, ale liczmy „na wszelki wypadek”
                                out_after = int(((after_num < lower) | (after_num > upper)).sum())

                                pct_below = 100.0 * below / n_before if n_before else 0.0
                                pct_above = 100.0 * above / n_before if n_before else 0.0
                                pct_total = pct_below + pct_above

                                # 9) KPI w „kartach” (cyferki na górze)
                                kcol = st.columns(6)
                                with kcol[0]:
                                    st.markdown(
                                        f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px'>"
                                        f"<div style='font-size:12px;color:#666'>n przed &rarr; po</div>"
                                        f"<div style='font-size:18px;font-weight:600'>{n_before:,} &rarr; {n_after:,}</div>"
                                        f"<div style='font-size:11px;color:#888'>liczba obserwacji</div></div>",
                                        unsafe_allow_html=True,
                                    )
                                with kcol[1]:
                                    st.markdown(
                                        f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px'>"
                                        f"<div style='font-size:12px;color:#666'>n odstające przed &rarr; po</div>"
                                        f"<div style='font-size:18px;font-weight:600'>{out_before:,} &rarr; {out_after:,}</div>"
                                        f"<div style='font-size:11px;color:#888'>liczba wartości odstających ogółem</div></div>",
                                        unsafe_allow_html=True,
                                    )
                                with kcol[2]:
                                    st.markdown(
                                        f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px'>"
                                        f"<div style='font-size:12px;color:#666'>% &lt; lower</div>"
                                        f"<div style='font-size:18px;font-weight:600'>{pct_below:.2f}%</div>"
                                        f"<div style='font-size:11px;color:#888'>odsetek lewy ogon</div></div>",
                                        unsafe_allow_html=True,
                                    )
                                with kcol[3]:
                                    st.markdown(
                                        f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px'>"
                                        f"<div style='font-size:12px;color:#666'>% &gt; upper</div>"
                                        f"<div style='font-size:18px;font-weight:600'>{pct_above:.2f}%</div>"
                                        f"<div style='font-size:11px;color:#888'>odsetek prawy ogon</div></div>",
                                        unsafe_allow_html=True,
                                    )
                                with kcol[4]:
                                    st.markdown(
                                        f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px'>"
                                        f"<div style='font-size:12px;color:#666'>% łącznie</div>"
                                        f"<div style='font-size:18px;font-weight:600'>{pct_total:.2f}%</div>"
                                        f"<div style='font-size:11px;color:#888'>lewy+prawy</div></div>",
                                        unsafe_allow_html=True,
                                    )
                                with kcol[5]:
                                    st.markdown(
                                        f"<div style='border:1px solid #ddd;border-radius:6px;padding:8px' "
                                        f"title='L = Q1 - K·IQR, U = Q3 + K·IQR; IQR = Q3 - Q1; K ustawiasz suwakiem „K (Tukey/IQR)”.'>"
                                        f"<div style='font-size:12px;color:#666'>Progi [L, U]</div>"
                                        f"<div style='font-size:18px;font-weight:600'>[{lower:.2f}, {upper:.2f}]</div>"
                                        f"<div style='font-size:11px;color:#888'>granice winsoryzacji</div></div>",
                                        unsafe_allow_html=True,
                                    )

                                # mały odstęp pod KPI
                                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                                
                            # =========================
                            #  HISTOGRAM: słupki OBOK SIEBIE (Altair v5)
                            # =========================
                            base_vals = before.dropna()

                            # 🔍 Pusta / degenerowana kolumna:
                            # - same NaN po konwersji
                            # - albo jedna unikalna wartość (brak wariancji)
                            if base_vals.empty or base_vals.nunique() <= 1:
                                st.info(
                                    "Kolumna nie ma wariancji / brak danych liczbowych "
                                    "(same NaN lub jedna unikalna wartość) – nie można narysować wykresów."
                                )
                            else:
                                vmin, vmax = float(base_vals.min()), float(base_vals.max())

                                # JEDEN DataFrame z kolumną 'zestaw' — to KLUCZOWE dla xOffset
                                plot_df = pd.concat(
                                    [
                                        pd.DataFrame({"value": before, "zestaw": "przed"}),
                                        pd.DataFrame({"value": after,  "zestaw": "po"}),
                                    ],
                                    ignore_index=True
                                ).dropna(subset=["value"])

                                dom = ["przed", "po"]
                                rng = ["#1f77b4", "#ff7f0e"]  # stałe kolory

                            # =========================
                            #  HISTOGRAM / ECDF
                            # =========================
                            base_vals = before.dropna()
                            if base_vals.empty:
                                st.info("Brak danych liczbowych po konwersji (NaN) – nie można narysować wykresów.")
                            else:
                                # 1) jeden DF „przed/po”
                                plot_df = pd.concat(
                                    [
                                        pd.DataFrame({"value": before, "zestaw": "przed"}),
                                        pd.DataFrame({"value": after,  "zestaw": "po"}),
                                    ],
                                    ignore_index=True,
                                ).dropna(subset=["value"])

                                vals = pd.to_numeric(plot_df["value"], errors="coerce").dropna()
                                if vals.empty:
                                    st.info("Brak danych liczbowych po konwersji (NaN) – nie można narysować wykresów.")
                                else:
                                    vmin, vmax = float(vals.min()), float(vals.max())
                                    unique_vals = np.sort(vals.unique())

                                    dom = ["przed", "po"]
                                    rng = ["#1f77b4", "#ff7f0e"]

                                    # ---------- A) HISTOGRAM (Altair – jak w _numeric_by_category_charts) ----------
                                    if viz_mode == "Histogram":
                                        # inteligentne binowanie – dokładnie ta sama logika co w 4C
                                        bin_param = alt.Bin(maxbins=40, extent=[vmin, vmax])
                                        if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
                                            bin_param = alt.Bin(extent=[-0.5, 1.5], step=1)
                                        elif (
                                            len(unique_vals) <= 12
                                            and np.all(np.isfinite(unique_vals))
                                            and np.all(np.mod(unique_vals, 1) == 0)
                                        ):
                                            loi, hii = float(unique_vals.min()), float(unique_vals.max())
                                            bin_param = alt.Bin(extent=[loi - 0.5, hii + 0.5], step=1)

                                        base_hist = alt.Chart(plot_df)

                                        # wspólne biny + środek i szerokość binu
                                        binned = (
                                            base_hist
                                            .transform_bin(
                                                as_=["bin_start", "bin_end"],
                                                field="value",
                                                bin=bin_param,
                                            )
                                            .transform_calculate(
                                                bin_mid="(datum.bin_start + datum.bin_end) / 2",
                                                bin_w="(datum.bin_end - datum.bin_start)",
                                            )
                                        )

                                        # rozstawiamy słupki kategorii wewnątrz binu – jak w _numeric_by_category_charts
                                        _cat_domain_json = json.dumps(dom)   # ["przed", "po"]
                                        _rank_expr = f"indexof({_cat_domain_json}, datum['zestaw'])"
                                        _k_total = len(dom)

                                        binned = (
                                            binned
                                            .transform_calculate(
                                                cat_rank=_rank_expr,
                                                k_total=str(_k_total),
                                                band_fraction="0.85",
                                                cat_w="(toNumber(datum.band_fraction) * datum.bin_w) / toNumber(datum.k_total)",
                                                x_mid="datum.bin_mid + ( (datum.cat_rank - (toNumber(datum.k_total)-1)/2 ) * datum.cat_w )",
                                                x_left="datum.x_mid - datum.cat_w/2",
                                                x_right="datum.x_mid + datum.cat_w/2",
                                            )
                                        )

                                        hist = (
                                            binned
                                            .mark_bar(opacity=0.85)
                                            .encode(
                                                x=alt.X("x_left:Q", title=col),
                                                x2=alt.X2("x_right:Q"),
                                                y=alt.Y("count():Q", title="Liczebność", stack=None),
                                                color=alt.Color(
                                                    "zestaw:N",
                                                    sort=dom,
                                                    scale=alt.Scale(domain=dom, range=rng),
                                                    title="Zbiór",
                                                ),
                                                order=alt.Order("cat_rank:Q", sort="ascending"),
                                                tooltip=[
                                                    alt.Tooltip("zestaw:N", title="Zbiór"),
                                                    alt.Tooltip("count():Q", title="liczebność"),
                                                    alt.Tooltip("bin_start:Q", title="bin od", format=".2f"),
                                                    alt.Tooltip("bin_end:Q",   title="bin do", format=".2f"),
                                                ],
                                            )
                                            .properties(height=280)
                                        )

                                        # cieniowanie poza [L, U]
                                        shade_df = pd.DataFrame(
                                            {"x": [vmin, upper], "x2": [lower, vmax], "side": ["left", "right"]}
                                        )
                                        shade = (
                                            alt.Chart(shade_df)
                                            .mark_rect(opacity=0.10)
                                            .encode(x="x:Q", x2="x2:Q")
                                        )

                                        # linie progów [L, U]
                                        rules = (
                                            alt.Chart(pd.DataFrame({"b": [lower, upper]}))
                                            .mark_rule(strokeDash=[6, 3], opacity=0.9, color="#dc3545")
                                            .encode(x="b:Q")
                                        )

                                        chart_layer = alt.layer(shade, hist, rules).resolve_scale(y="independent")

                                        st.altair_chart(
                                            chart_layer,
                                            use_container_width=True,
                                            key=f"{chart_key_base}_hist",
                                        )

                                    # ---------- B) ECDF (lekki, zostaje jak był) ----------
                                    else:
                                        ecdf_df = plot_df.sort_values("value").assign(
                                            rank=lambda d: d.groupby("zestaw").cumcount() + 1
                                        )
                                        sizes = (
                                            ecdf_df.groupby("zestaw")["rank"]
                                            .transform("max")
                                            .replace(0, np.nan)
                                        )
                                        ecdf_df["frac"] = ecdf_df["rank"] / sizes

                                        ecdf = (
                                            alt.Chart(ecdf_df)
                                            .mark_line()
                                            .encode(
                                                x=alt.X(
                                                    "value:Q",
                                                    title=col,
                                                    scale=alt.Scale(domain=[vmin, vmax]),
                                                ),
                                                y=alt.Y(
                                                    "frac:Q",
                                                    title="ECDF",
                                                    scale=alt.Scale(domain=[0, 1]),
                                                ),
                                                color=alt.Color(
                                                    "zestaw:N",
                                                    sort=dom,
                                                    scale=alt.Scale(domain=dom, range=rng),
                                                    title="Zbiór",
                                                ),
                                                tooltip=[
                                                    alt.Tooltip("zestaw:N", title="Zbiór"),
                                                    alt.Tooltip("value:Q", title="Wartość", format=".2f"),
                                                    alt.Tooltip("frac:Q",  title="Frakcja", format=".2f"),
                                                ],
                                            )
                                            .properties(height=280)
                                        )
                                        rules = (
                                            alt.Chart(pd.DataFrame({"b": [lower, upper]}))
                                            .mark_rule(strokeDash=[6, 3], opacity=0.9, color="#dc3545")
                                            .encode(x="b:Q")
                                        )

                                        st.altair_chart(
                                            alt.layer(ecdf, rules),
                                            use_container_width=True,
                                            key=f"{chart_key_base}_ecdf",
                                        )

                                # 🔹 Boxplot (wąsy 1.5·IQR zgodnie z Tukey)
                                bL, bR = st.columns([1.3, 1.0])
                                with bL:
                                    bp = (
                                        alt.Chart(plot_df)
                                        .mark_boxplot(extent=1.5, outliers=True)  # ← tu zmiana: 1.5·IQR (zamiast "min-max")
                                        .encode(
                                            y=alt.Y("zestaw:N", title=None, sort=dom),
                                            x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[vmin, vmax])),
                                            color=alt.Color("zestaw:N", sort=dom, scale=alt.Scale(domain=dom, range=rng), legend=None),
                                        )
                                        .properties(height=150)
                                    )
                                    # Linie median ‘przed’ i ‘po’
                                    med_df = plot_df.groupby("zestaw")["value"].median().reset_index()
                                    med_rule = alt.Chart(med_df).mark_rule(strokeDash=[4,4], color="#6c757d").encode(
                                        x="value:Q", detail="zestaw:N"
                                    )

                                    # Warstwa „kropek-outlierów” w kolorze czerwonym (liczone wg progów [L,U])
                                    out_df = plot_df[(plot_df["value"] < lower) | (plot_df["value"] > upper)]
                                    out_layer = (
                                        alt.Chart(out_df)
                                        .mark_point(size=30, opacity=0.9, color="#dc3545")
                                        .encode(
                                            y=alt.Y("zestaw:N", sort=dom, title=None),
                                            x=alt.X("value:Q")
                                        )
                                    )

                                    st.altair_chart(alt.layer(bp, med_rule, out_layer), use_container_width=True, key=f"{chart_key_base}_box",)


                                with bR:
                                    before_stats = mini_stats(before)
                                    after_stats  = mini_stats(after)
                                    rows = ["liczebność (n)", "min", "Q1", "mediana", "Q3", "max", "średnia", "odch.std.", "IQR"]
                                    tbl = []
                                    for r in rows:
                                        b = before_stats.get(r, np.nan)
                                        a = after_stats.get(r, np.nan)
                                        d  = (a - b) if (isinstance(a,(int,float)) and isinstance(b,(int,float))) else np.nan
                                        dp = (d / b * 100.0) if (isinstance(b,(int,float)) and b not in (0, np.nan)) else np.nan
                                        tbl.append([r,
                                                    f"{b:.4g}" if isinstance(b,(int,float)) else b,
                                                    f"{a:.4g}" if isinstance(a,(int,float)) else a,
                                                    f"{d:+.4g}" if isinstance(d,(int,float)) else "",
                                                    f"{dp:+.2f}%" if isinstance(dp,(int,float)) else ""])
                                    stats_df = pd.DataFrame(tbl, columns=["metryka","przed","po","Δ","Δ%"])
                                    st.dataframe(stats_df, use_container_width=True, hide_index=True)


                                with bL:
                                    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
                                    st.caption(
                                        "ℹ️ Liczebność liczona jest po konwersji do typów numerycznych i odrzuceniu NaN. "
                                        "Winsoryzacja ścina ogony wg progów [L,U] (Tukey/IQR). \n\n"
                                        f"▶️ Jeśli % łącznie ({pct_total:.2f}%) jest większe niż ok. 5% i mediana w tabeli po prawej "
                                        "nie przesunęła się istotnie, zwykle możesz zaakceptować winsoryzację. "
                                        "W przeciwnym razie rozważ większe K lub rezygnację z winsoryzacji dla tej kolumny."
                                    )

            run_prep = st.form_submit_button("⚙️ Przygotuj dane do trenowania (automatycznie)", type="primary")

    # ── Submit handler ──────────────────────────────────────────────────────────
    if run_prep and st.session_state.get("sec7_revealed", False):
        if "drop_cols_user" not in locals():      drop_cols_user = basic_col_drop_default
        if "winsorize_cols_user" not in locals(): winsorize_cols_user = []
        decisions = {
            "drop_cols": drop_cols_user,
            "winsorize_cols": winsorize_cols_user,
            "remove_duplicates": remove_duplicates_user
        }

        with st.spinner("🛠️ Przygotowuję dane do trenowania…"):
            df_ready, prep_report = _auto_prepare_for_training(df, info_df, decisions)
            ready_path, report_path = _persist_artifacts(df_ready, prep_report, latest_info)
            hours_saved = _estimate_hours_saved(n_rows_raw, n_cols_raw, high_null_cols, duplicates_count, auto_drop_candidates, pairs_sorted)
            cost_saved  = _estimate_cost_saved_pln(hours_saved)
            # ➜ Handoff do etapu Data Chat (pakiet startowy)
            handoff_path = _save_datachat_handoff(
                df_ready=df_ready,
                latest_info=latest_info,
                summary_text=st.session_state.get("latest_summary_text", ""),
                pairs_sorted=pairs_sorted,
                prep_report_path=report_path,
            )
            st.toast(f"Pakiet Data Chat zapisany: {handoff_path}", icon="✅")

        st.markdown(_success_hero_box(hours_saved, cost_saved), unsafe_allow_html=True)
        outlier_flags_count = len(prep_report["outlier_flags"])
        duplicates_removed  = prep_report["duplicates_removed"]
        n_cols_final        = prep_report["n_cols_final"]

        st.markdown(
            _before_after_cards_html(n_cols_raw, n_cols_final, global_missing_pct,
                                    duplicates_count, duplicates_removed, outlier_flags_count),
            unsafe_allow_html=True,
        )
        st.subheader("Co dokładnie zrobiliśmy:", divider="gray")
        dropped_cols_txt  = ", ".join(prep_report["dropped_columns"])     if prep_report["dropped_columns"]     else "(nic)"
        new_time_cols_txt = ", ".join(prep_report["new_time_cols_added"]) if prep_report["new_time_cols_added"] else "(brak)"
        outlier_flags_txt = ", ".join(prep_report["outlier_flags"])        if prep_report["outlier_flags"]        else "(brak)"
        imputations_lines = [f"✅ {col}: {how}" for col, how in prep_report["imputations"].items()]
        winsor_lines = [f"✅ {col}: [{b['lower']:.2f}, {b['upper']:.2f}]" for col, b in prep_report["winsorized"].items()]
        st.markdown(
            f"**Pliki wynikowe:**  \n✅ `{ready_path}`  \n✅ `{report_path}`  \n\n"
            f"**Zmiany w danych:**  \n✅ Usunięte kolumny: {dropped_cols_txt}  \n"
            f"✅ Dodane cechy z dat: {new_time_cols_txt}  \n✅ Flagi outlierów: {outlier_flags_txt}  \n"
            f"✅ Usuniętych duplikatów: {prep_report['duplicates_removed']}  \n\n"
            f"**Uzupełnianie braków:**  \n" + ("\n".join(imputations_lines) if imputations_lines else "✅ (brak braków)") + "\n\n"
            f"**Winsoryzacja:**  \n" + ("\n".join(winsor_lines) if winsor_lines else "✅ (brak kolumn do przycięcia)")
        )
        with st.expander("Raport techniczny przygotowania (pełne szczegóły)"):
            st.json(prep_report)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- dla pewności żeby wszystkie zdarzenia trafiły do Langfuse ---
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()  # blokująco czeka aż kolejka wyśle wszystko
        except Exception:
            pass

# Streamlit entrypoint
if __name__ == "__main__":
    main()

