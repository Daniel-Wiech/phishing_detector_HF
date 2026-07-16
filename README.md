# Phishing Detector HF

Lokalny klasyfikator phishingu dla wiadomości w języku polskim (SMS/e-mail), oparty na wyszukiwaniu semantycznym — wiadomość jest porównywana z bazą znanych wzorców ataków (FAISS) i jeśli jest do nich wystarczająco podobna, zostaje oznaczona jako phishing.

Model embeddingowy (`intfloat/multilingual-e5-large`) jest pobierany automatycznie z Hugging Face Hub przy pierwszym uruchomieniu.

## Wymagania

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)

## Instalacja

```bash
git clone https://github.com/Daniel-Wiech/phishing_detector_HF.git
cd phishing_detector_HF
poetry install
```

## Użycie

**Przebudowa bazy wektorowej** (po edycji `src/data/scenarios.json`):

```bash
poetry run python src/main.py rebuild
```

**Klasyfikacja wiadomości:**

```bash
poetry run python src/main.py classify "Na państwa koncie znajduje się nadpłata w wysokości 800zl."
```

Przykładowy wynik:

```
=== WYNIK ANALIZY ===
Classification: Phishing
Category:       Podszywanie się pod dostawców energii (PGE/Tauron/PGNiG)
Confidence:     96.5%
Similarity:     0.9650 (Próg: 0.88)
Explanation:    Wiadomość jest wysoce podobna do znanych kampanii phishingowych...
=====================
```

## Struktura

```
src/
├── main.py        # CLI: classify / rebuild
├── config.py       # Konfiguracja (model, próg podobieństwa)
├── classifier/      # Logika klasyfikacji
├── embeddings/      # Generowanie embeddingów (Hugging Face)
├── vector_db/       # Indeks FAISS + metadane
└── data/            # Baza wzorców phishingowych (scenarios.json)
```

## Autor

Daniel-Wiech
