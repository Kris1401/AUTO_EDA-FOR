# AUTO EDA FOR

Asystent analizy danych i przygotowania modelu ML w Streamlit:
- Wgrywasz plik (CSV / XLSX / PDF-tabela),
- dostajesz natychmiastowy podgląd danych + maskowanie PII,
- możesz pobrać dane jako CSV/XLSX lub ZIP z metadanymi,
- możesz przetworzyć **cały zbiór** (nie tylko podgląd), zapisać go lokalnie i oznaczyć jako gotowy do trenowania,
- aplikacja przekazuje ścieżki do tych artefaktów dalej (`latest_artifacts`), tak żeby następny krok („Trenowanie Modelu”) działał bez ponownego uploadu.

Architektura nastawiona jest na:
- prosty onboarding analityka / marketera (zero kodu),
- zgodność z prywatnością danych (maskowanie PII),
- dalszą automatyzację: EDA, trenowanie modelu, scoring.

---

## Aktualny flow aplikacji

### 1. Analiza Danych (pierwsza zakładka)
- Wczytanie pliku źródłowego (CSV, XLSX, PDF-tabela).
- Podgląd danych w dwóch trybach:
  - szybkie scrollowanie (`st.dataframe`, sticky header),
  - paginacja + wyszukiwarka (`itables` / DataTables).
- Maskowanie PII (opcjonalne, ale domyślnie włączone).
- Podsumowanie: typy kolumn, braki danych, rozmiar.
- Eksport:
  - `⬇ CSV (podgląd)`,
  - `⬇ XLSX (podgląd)`,
  - ZIP z metadanymi (`preview_masked.csv` + `meta.json`).

### 2. "Przelicz na całości i zapisz artefakty"
- Kliknięcie przycisku uruchamia pełne przetworzenie całego pliku (nie tylko podglądu).
- Dane po maskowaniu PII i metadane są zapisywane lokalnie (np. `C:\AUTO_EDA_FOR\ingest\...`).
- W `st.session_state["latest_artifacts"]` zapisujemy ścieżki do:
  - pełnego zbioru (po maskowaniu),
  - pliku meta,
  - katalogu runu,
  - liczby wierszy / kolumn,
  - znacznika czasu itp.
- Dzięki temu kolejna zakładka ("Trenowanie Modelu") wie z automatu, jakiego zbioru ma użyć — bez ponownego uploadu pliku przez użytkownika.

---

## Planowane kolejne kroki (Etap 2+)
- Zakładka "Trenowanie Modelu":
  - automatyczne wykrywanie gotowych danych przez `st.session_state["latest_artifacts"]`,
  - podgląd i sanity-check zestawu treningowego,
  - raport EDA (dystrybucje, korelacje, brakujące wartości, outliery, itp.),
  - krótkie podsumowania tekstowe.
- Później: półautomatyczne trenowanie modelu + scoring.

---

## Uruchomienie lokalne

Projekt działa jako zwykły skrypt Streamlit.  
W tej chwili **nie wymagamy osobnego wirtualnego środowiska**, ale zalecamy mieć zainstalowane wymagane biblioteki z `requirements.txt`.

1. Sklonuj repozytorium:
```bash
git clone https://github.com/Kris1401/AUTO_EDA-FOR.git
cd AUTO_EDA-FOR
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

Jeśli używasz conda i chcesz mieć powtarzalne środowisko, możesz też zrobić:
```bash
conda env create -f environment.yml
conda activate auto-eda-for
```

Ale nie jest to wymagane do prostego uruchomienia aplikacji.

3. Uruchom aplikację Streamlit:
```bash
streamlit run app/app.py
```

4. Aplikacja otworzy się w przeglądarce (domyślnie http://localhost:8501)
Przejdź do zakładki **„Analiza Danych”**, wgraj plik i zobacz podgląd.
