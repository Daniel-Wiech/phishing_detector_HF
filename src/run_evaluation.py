import json
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from classifier.detector import PhishingClassifier
from config import settings


def evaluate_model(classifier: PhishingClassifier, test_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pomocnicza funkcja klasyfikująca próbki testowe."""
    results = []
    for item in test_dataset:
        res = classifier.classify(item["text"])
        results.append({
            "text": item["text"],
            "expected": item["label"],
            "predicted": res["classification"],
            "phishing_score": res["phishing_score"],
            "legitimate_score": res["legitimate_score"],
            "margin": res["margin"]
        })
    return results


def print_report(results: List[Dict[str, Any]]):
    """Wyświetla podsumowanie metryk oraz błędy."""
    y_true = [r["expected"] for r in results]
    y_pred = [r["predicted"] for r in results]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    labels = ["phishing", "legitimate", "uncertain"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n" + "=" * 50)
    print("          RAPORT DOKŁADNOŚCI KLASYFIKACJI          ")
    print("=" * 50)
    print(f"Dokładność (Accuracy): {acc * 100:.2f}%")
    print(f"Precyzja (Precision):  {precision * 100:.2f}%")
    print(f"Czułość (Recall):       {recall * 100:.2f}%")
    print(f"F1-Score:               {f1 * 100:.2f}%\n")
    
    print("Macierz Pomyłek (Confusion Matrix):")
    print(f"Etykiety: {labels}")
    print(cm)
    print("\nSzczegółowy raport sklearn:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    errors = [r for r in results if r["expected"] != r["predicted"]]
    if errors:
        print(f"\n=== PRZYPADKI BŁĘDNIE ZAKLASYFIKOWANE / UNCERTAIN ({len(errors)}) ===")
        for err in errors:
            print(f"Tekst: '{err['text']}'")
            print(f" Oczekiwano: {err['expected'].upper()} | Otrzymano: {err['predicted'].upper()}")
            print(f" P-Score: {err['phishing_score']:.4f} | L-Score: {err['legitimate_score']:.4f} | Margin: {err['margin']:.4f}\n")
    else:
        print("\nWszystkie przykłady zostały prawidłowo sklasyfikowane!")


def run_evaluation():
    print("=== SYSTEM EWALUACJI DETEKTORA PHISHINGU ===")
    
    # 1. Wybór metody w konsoli
    print("\nWybierz metodę weryfikacji:")
    print("[1] Szybki podział 80/20 (80% trening, 20% test)")
    print("[2] Leave-One-Out Cross-Validation (LOO-CV - pełna weryfikacja)")
    
    choice = input("\nTwój wybór (1/2): ").strip()
    while choice not in ["1", "2"]:
        choice = input("Nieprawidłowy wybór. Wpisz 1 lub 2: ").strip()

    # 2. Wczytanie danych
    with open(settings.DATA_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
        
    dataset = []
    for scenario in scenarios:
        label = scenario.get("label", "phishing")
        category = scenario.get("category", "General")
        for example in scenario.get("examples", []):
            dataset.append({
                "text": example,
                "label": label,
                "category": category
            })
            
    print(f"\nZaładowano {len(dataset)} przykładów ze scenarios.json.")
    classifier = PhishingClassifier()

    # ------------------------------------------------------------------
    # OPCJA 1: PODZIAŁ 80 / 20
    # ------------------------------------------------------------------
    if choice == "1":
        print("\n--- URUCHAMIANIE TESTU 80/20 ---")
        
        # Rozdzielamy dane z zachowaniem proporcji klas (stratify)
        train_data, test_data = train_test_split(
            dataset,
            test_size=0.20,
            random_state=42,
            stratify=[item["label"] for item in dataset]
        )
        
        print(f"Zbiór treningowy: {len(train_data)} próbki | Zbiór testowy: {len(test_data)} próbki.")
        print("Generowanie wektorów dla bazy treningowej...")

        # Wyznaczenie wektorów tylko dla zbioru treningowego 80%
        train_vectors = []
        train_metadata = []
        for item in tqdm(train_data, desc="Indeksowanie 80%"):
            vec = classifier.embedder.get_embedding(item["text"], is_query=False)
            train_vectors.append(vec)
            train_metadata.append({
                "label": item["label"],
                "category": item["category"],
                "risk": "medium",
                "example": item["text"]
            })

        classifier.db.clear()
        classifier.db.add_vectors(np.array(train_vectors).astype("float32"), train_metadata)

        print("\nKlasyfikacja 20% próbki testowej...")
        results = evaluate_model(classifier, test_data)
        print_report(results)

    # ------------------------------------------------------------------
    # OPCJA 2: LEAVE-ONE-OUT CROSS-VALIDATION (LOO-CV) Z CACHE WEKTORÓW
    # ------------------------------------------------------------------
    else:
        print("\n--- URUCHAMIANIE TESTU LEAVE-ONE-OUT (LOO-CV) ---")
        print("Krok 1/2: Generowanie/przygotowanie cache'u wektorów dla wszystkich zdań...")

        # Kluczowa optymalizacja: Obliczamy embeddingi TYLKO RAZ przed pętlą!
        cached_vectors = []
        for item in tqdm(dataset, desc="Tworzenie embeddingów"):
            vec = classifier.embedder.get_embedding(item["text"], is_query=False)
            cached_vectors.append(vec)

        print("\nKrok 2/2: Testowanie LOO-CV (iteracyjna weryfikacja)...")
        results = []

        # Pętla LOO z paskiem postępu tqdm
        for idx, test_item in enumerate(tqdm(dataset, desc="Ewaluacja LOO-CV")):
            classifier.db.clear()
            
            # Składanie bazy treningowej z wyłączeniem idx-owego elementu
            train_vectors = [cached_vectors[j] for j in range(len(dataset)) if j != idx]
            train_metadata = [{
                "label": dataset[j]["label"],
                "category": dataset[j]["category"],
                "risk": "medium",
                "example": dataset[j]["text"]
            } for j in range(len(dataset)) if j != idx]

            vectors_np = np.array(train_vectors).astype("float32")
            classifier.db.add_vectors(vectors_np, train_metadata)

            # Klasyfikacja
            res = classifier.classify(test_item["text"])
            results.append({
                "text": test_item["text"],
                "expected": test_item["label"],
                "predicted": res["classification"],
                "phishing_score": res["phishing_score"],
                "legitimate_score": res["legitimate_score"],
                "margin": res["margin"]
            })

        print_report(results)

    # Odbudowa pełnej bazy po zakończeniu testów
    print("\nOdbudowywanie pełnej bazy FAISS na dysku do standardowej pracy...")
    classifier.build_index_from_scenarios(deduplicate=True)
    print("Gotowe!")


if __name__ == "__main__":
    run_evaluation()