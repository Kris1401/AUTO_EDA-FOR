import streamlit as st
_DEFAULT = "PL"
_STRINGS = {
 "PL": {
  "app.title": "AUTO EDA FOR",
  "app.subtitle": "Szkielet aplikacji (Etap 0): nawigacja, i18n, motyw, walidacja konfiguracji.",
  "home.config_status": "Status konfiguracji",
  "home.config_ok": "Konfiguracja wygląda dobrze. Możesz przejść dalej.",
  "home.config_warn": "Brakuje niektórych zmiennych. Uzupełnij je w .env / secrets.",
  "home.next_steps": "Kolejny krok: **Etap 1 — Ingest (CSV/XLSX/PDF) + PII + Preview**.",
  "home.nav_info": "Skorzystaj z zakładek (po lewej), aby zobaczyć szkielety stron.",
  "home.footer_note": "Wersja demonstracyjna szkieletu. W następnych etapach dodamy logikę.",
  "page.ingest.title": "Analiza Danych — Wczytywanie (Etap 1)",
  "page.chat.title": "Data Chat (Etap 3)",
  "page.ml.title": "Trenowanie modelu (Etap 4)",
  "page.predict.title": "Predykcja (Etap 6)",
  "page.settings.title": "Ustawienia i konfiguracja",
  "settings.desc": "Podgląd aktywnej konfiguracji z .env i secrets.",
  "settings.env_vars": "Wymagane zmienne",
  "settings.values": "Bieżące (maskowane) wartości",
 },
 "EN": {
  "app.title": "AUTO EDA FOR",
  "app.subtitle": "Application skeleton (Stage 0): navigation, i18n, theme, config validation.",
  "home.config_status": "Configuration status",
  "home.config_ok": "Configuration looks good. You can proceed.",
  "home.config_warn": "Some variables are missing. Fill them in .env / secrets.",
  "home.next_steps": "Next step: **Stage 1 — Ingest (CSV/XLSX/PDF) + PII + Preview**.",
  "home.nav_info": "Use the left tabs to see page skeletons.",
  "home.footer_note": "Demo skeleton. Logic will be added in further stages.",
  "page.ingest.title": "Data Analysis — Ingest (Stage 1)",
  "page.chat.title": "Data Chat (Stage 3)",
  "page.ml.title": "Model Training (Stage 4)",
  "page.predict.title": "Prediction (Stage 6)",
  "page.settings.title": "Settings & configuration",
  "settings.desc": "Preview of active configuration read from .env and secrets.",
  "settings.env_vars": "Required variables",
  "settings.values": "Current (masked) values",
 }
}
def get_locale():
    return st.session_state.get("_locale", _DEFAULT)
def set_locale(loc:str):
    st.session_state["_locale"] = loc
def t(key: str) -> str:
    loc = get_locale()
    return _STRINGS.get(loc, {}).get(key, _STRINGS[_DEFAULT].get(key, key))
