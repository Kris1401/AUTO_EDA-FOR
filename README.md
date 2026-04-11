# AUTO EDA FOR

AUTO EDA FOR to aplikacja Streamlit, która prowadzi użytkownika przez pełny potok pracy z danymi: od importu i sanity checku, przez automatyczną diagnostykę oraz opcjonalny Data Chat, aż do budowy modelu i predykcji.

## Co dostajesz

### 1. Analiza Danych
- import plików CSV, XLSX, PDF (tabele) i Parquet,
- podgląd danych z maskowaniem PII,
- szybki sanity check kolumn, braków i typów,
- zapis artefaktów pełnego zbioru do kolejnych etapów.

### 2. Automat EDA
- szybka diagnoza jakości danych,
- automatyczne wykrywanie braków, duplikatów, outlierów i ryzyk,
- rekomendacje cleaningu i przygotowania danych,
- TL;DR i checklistę kolejnych kroków,
- nazwy i opisy segmentów / klastrów wspierane przez LLM.

### 3. Data Chat (opcjonalny deep dive)
- zadawanie pytań o dane w języku naturalnym,
- inteligentny dobór właściwej gałęzi odpowiedzi i zestawu wykresów,
- executive takeaways, interpretacje i deep dive w wizualizacje,
- możliwość potraktowania Etapu 3 jak osobnego centrum insightów.

### 4. Trenowanie modelu
- przejście do modelowania na danych przygotowanych w poprzednich etapach,
- krótsza droga od zbioru wejściowego do gotowego pipeline'u,
- miejsce na AutoML, tuning, wybór zwycięskiego modelu i walidację.

### 5. Predykcja
- scoring jednego rekordu lub całej paczki,
- walidacja wejścia i eksport wyników,
- finalny etap gotowy do użycia biznesowego.

## Jak płyną dane przez aplikację

Najkrótsza ścieżka to:

1 -> 2 -> 4 -> 5

Jeśli chcesz wejść głębiej w analizę i zrozumieć dane przed modelowaniem, możesz skorzystać z dodatkowego przystanku:

1 -> 2 -> 3 -> 4 -> 5

Kluczowe założenie produktu jest proste: aplikacja zapisuje i przekazuje artefakty między etapami, więc użytkownik nie zaczyna od zera na każdej stronie.

## Jak działa Data Chat

Data Chat nie dobiera wykresów przypadkowo. Najpierw rozpoznaje cel pytania, a dopiero potem uruchamia rodzinę odpowiedzi najlepiej dopasowaną do tego celu.

Przykładowe rodziny odpowiedzi:
- Distribution - rozkład, percentyle, outliery, histogramy i boxploty,
- Composition Static - struktura i udziały bez osi czasu,
- Composition Over Time - zmiana udziałów i wartości w czasie,
- Comparison - liderzy, rankingi i odstępstwa,
- Relationship - zależności między zmiennymi,
- Quality / Sanity - braki, duplikaty, anomalie i kolumny ryzyka,
- Segmentation / Clusters - wielkość segmentów, profile i interpretacje klastrów.

Na stronie startowej znajduje się dodatkowa sekcja z uproszczonym schematem inspirowanym Andrew Abelą, która pokazuje, jak różne pytania uruchamiają różne rodziny odpowiedzi.

## Uruchomienie lokalne

### 1. Sklonuj repozytorium

    git clone https://github.com/Kris1401/AUTO_EDA-FOR.git
    cd AUTO_EDA-FOR

### 2. Zainstaluj zależności

    pip install -r requirements.txt

Opcjonalnie możesz skorzystać z Condy:

    conda env create -f environment.yml
    conda activate auto-eda-for

### 3. Uruchom aplikację

    streamlit run app/app.py

## Konfiguracja

W repo znajdują się pliki konfiguracyjne Streamlit i ustawienia środowiska używane przez aplikację. Jeśli pracujesz lokalnie, upewnij się, że wymagane klucze i parametry środowiskowe są dostępne zgodnie z bieżącą konfiguracją projektu.

## Dla kogo jest ten projekt

AUTO EDA FOR jest nastawione na:
- analityka lub marketera, który chce dojść od danych do insightów i modelu bez kodowania całego procesu ręcznie,
- szybki onboarding do eksploracji danych,
- możliwie mały ładunek kognitywny przy przechodzeniu przez kolejne etapy,
- spójny produktowy przepływ: dane -> diagnoza -> insighty -> model -> predykcja.
