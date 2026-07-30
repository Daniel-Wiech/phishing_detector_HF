# Phishing Detector HF

Lokalny klasyfikator phishingu dla wiadomości w języku polskim (SMS/e-mail), oparty na **wyszukiwaniu semantycznym**. Wiadomość jest zamieniana na wektor (embedding) i porównywana z bazą znanych wzorców ataków przechowywaną w indeksie **FAISS** — jeśli jest do nich wystarczająco podobna, zostaje oznaczona jako phishing.

Model embeddingowy (`intfloat/multilingual-e5-large`) jest pobierany automatycznie z Hugging Face Hub przy pierwszym uruchomieniu.

## Jak to działa

1. **Preprocessing** — tekst wiadomości (lub pełny obiekt e-maila z nadawcą, tematem, linkami i załącznikami) jest normalizowany do jednej reprezentacji tekstowej.
2. **Embedding** — tekst zamieniany jest na 1024-wymiarowy wektor przez model `multilingual-e5-large` i normalizowany (L2).
3. **Wyszukiwanie w FAISS** — pobierane jest `TOP_K` (domyślnie 5) najbardziej podobnych przykładów z lokalnej bazy wzorców (indeks `IndexFlatIP`, tj. podobieństwo kosinusowe na znormalizowanych wektorach).
4. **Scoring** — uśredniane są 2 najlepsze trafienia osobno dla klasy `phishing` i `legitimate`, co daje `phishing_score` i `legitimate_score`.
5. **Decyzja** — wiadomość zostaje sklasyfikowana jako:
   - `phishing`, jeśli `phishing_score ≥ SCORE_THRESHOLD` **i** `margin ≥ MARGIN_THRESHOLD`,
   - `legitimate`, jeśli `legitimate_score ≥ SCORE_THRESHOLD` **i** `-margin ≥ MARGIN_THRESHOLD`,
   - `uncertain` w pozostałych przypadkach,

   gdzie `margin = phishing_score - legitimate_score`.

## Funkcje

- Klasyfikacja pojedynczej wiadomości tekstowej (SMS/e-mail) z linii poleceń.
- Obsługa pełnych wiadomości e-mail (nadawca, temat, treść, URL-e, załączniki) jako ustrukturyzowane wejście.
- Przebudowa bazy wektorowej z pliku `scenarios.json`, z opcjonalną **deduplikacją semantyczną per klasa** (próg podobieństwa 0.98).
- Moduł ewaluacji modelu: podział 80/20 lub pełna walidacja **Leave-One-Out Cross-Validation**, z raportem accuracy/precision/recall/F1 i macierzą pomyłek.

## Wymagania

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)

## Instalacja

```bash
git clone https://github.com/Daniel-Wiech/phishing_detector_HF.git
cd phishing_detector_HF
poetry install
```

Przy pierwszym uruchomieniu program pobierze model embeddingowy z Hugging Face Hub (kilkaset MB–ok. 2 GB) i zbuduje lokalny indeks FAISS na podstawie `src/data/scenarios.json`.

## Użycie

### Klasyfikacja wiadomości

```bash
poetry run python src/main.py classify "Na państwa koncie znajduje się nadpłata w wysokości 800zl."
```

Przykładowy wynik:

```
=== WYNIK ANALIZY ===
Classification:   PHISHING
Category:         Podszywanie się pod dostawców energii (PGE/Tauron/PGNiG)
Confidence:       96.5%
------------------------------
Phishing Score:   0.9650 (Próg: 0.63)
Legitimate Score: 0.4120 (Próg: 0.63)
Margin:           0.5530 (Próg: 0.004)
------------------------------
Reason:           Wysokie podobieństwo do phishingu (0.97) oraz margines (0.55) powyżej progu.

Najbliższe przykłady w bazie:
  1. [PHISHING] (sim: 0.9650) -> PGE: Wykryliśmy nadpłatę 120 zł na Twoim koncie. Zaloguj się, aby odebrać zwrot.
  2. [PHISHING] (sim: 0.9510) -> Masz niewykorzystaną nadpłatę za prąd w PGE w wysokości 145 PLN. Kliknij link i potwierdź dane.
  3. [LEGITIMATE] (sim: 0.4120) -> Twoja faktura za energię elektryczną jest dostępna w e-BOK.
=====================
```

### Przebudowa bazy wektorowej

Po edycji `src/data/scenarios.json` indeks trzeba przebudować:

```bash
# standardowa przebudowa
poetry run python src/main.py rebuild

# przebudowa z deduplikacją semantyczną (usuwa niemal identyczne przykłady w obrębie tej samej klasy)
poetry run python src/main.py rebuild --dedup
```

### Ewaluacja modelu

```bash
cd src
poetry run python run_evaluation.py
```

Skrypt zapyta o metodę weryfikacji:

- **[1]** szybki podział 80/20 (trening/test, ze stratyfikacją klas),
- **[2]** pełna walidacja Leave-One-Out (każda próbka testowana na bazie zbudowanej ze wszystkich pozostałych).

Po zakończeniu testu pełna baza FAISS jest automatycznie odbudowywana na dysku.

## Konfiguracja

Wszystkie parametry znajdują się w `src/config.py`:

| Parametr           | Wartość domyślna                  | Opis                                                        |
| ------------------ | ---------------------------------- | ------------------------------------------------------------ |
| `MODEL_NAME`        | `intfloat/multilingual-e5-large`   | Model embeddingowy z Hugging Face Hub                        |
| `EMBEDDING_DIM`     | `1024`                             | Wymiar wektora embeddingu                                     |
| `SCORE_THRESHOLD`   | `0.63`                             | Minimalne podobieństwo, by uznać klasę phishing/legitimate    |
| `MARGIN_THRESHOLD`  | `0.004`                            | Minimalna różnica między `phishing_score` a `legitimate_score`|
| `TOP_K`             | `5`                                 | Liczba najbliższych sąsiadów pobieranych z FAISS              |

## Struktura projektu

```
src/
├── main.py              # CLI: classify / rebuild
├── config.py             # Konfiguracja (model, progi, ścieżki)
├── evaluator.py           # Klasa Evaluator — metryki na dowolnym zbiorze testowym
├── run_evaluation.py       # Interaktywny skrypt ewaluacyjny (80/20 lub LOO-CV)
├── classifier/
│   └── detector.py         # PhishingClassifier — logika klasyfikacji i budowy indeksu
├── embeddings/
│   └── embedder.py          # LocalEmbedder — generowanie embeddingów (SentenceTransformers)
├── vector_db/
│   ├── faiss_store.py        # FaissVectorStore — indeks FAISS + zapis/odczyt metadanych
│   ├── faiss.index            # Zbudowany indeks FAISS (generowany, wersjonowany w repo)
│   └── metadata.json           # Metadane wektorów (generowane, wersjonowane w repo)
└── data/
    ├── preprocessing.py        # EmailPreprocessor / EmailInput — normalizacja wejścia
    └── scenarios.json           # Baza wzorców phishingowych i wiadomości legalnych
```

## Baza wzorców (`scenarios.json`)

Baza zawiera obecnie **100 scenariuszy** (kategorii ataków lub wiadomości legalnych), po **3 przykładowe zdania** każdy — łącznie ok. **300 wiadomości** z etykietami `phishing` lub `legitimate`. Format pojedynczego wpisu:

```json
{
  "id": 1,
  "label": "phishing",
  "category": "PGE - Rzekoma nadpłata i zwrot środków",
  "description": "Próba wyłudzenia danych logowania pod pretekstem zwrotu nadpłaty za energię.",
  "risk": "high",
  "examples": [
    "Wykryliśmy nadpłatę [KWOTA] zł na Twoim koncie. Zaloguj się, aby odebrać zwrot.",
    "Masz niewykorzystaną nadpłatę za prąd w PGE w wysokości [KWOTA] PLN. Kliknij link i potwierdź dane."
  ]
}
```

Aby dodać nowe wzorce, wystarczy dopisać kolejny obiekt do listy w `scenarios.json` i uruchomić `rebuild`.
