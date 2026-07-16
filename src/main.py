import argparse
import json
import sys
from classifier.detector import PhishingClassifier
from config import settings

def main():
    parser = argparse.ArgumentParser(
        description="Lokalny klasyfikator Phishingu oparty na wyszukiwaniu semantycznym i FAISS."
    )
    subparsers = parser.add_subparsers(dest="command", help="Dostępne komendy")

    # Komenda klasyfikacji
    classify_parser = subparsers.add_parser("classify", help="Klasyfikuj wiadomość")
    classify_parser.add_argument("text", type=str, help="Treść wiadomości do przeanalizowania")

    # Komenda aktualizacji indeksu
    subparsers.add_parser("rebuild", help="Przebuduj i zaktualizuj lokalną bazę wektorową")

    args = parser.parse_args()

    if args.command == "classify":
        classifier = PhishingClassifier()
        result = classifier.classify(args.text)
        
        # Wyjście CLI sformatowane zgodnie z wymaganiami
        print("\n=== WYNIK ANALIZY ===")
        print(f"Classification: {result['classification'].capitalize()}")
        print(f"Category:       {result['category']}")
        print(f"Confidence:     {result['confidence'] * 100:.1f}%")
        print(f"Similarity:     {result['similarity']:.4f} (Próg: {settings.SIMILARITY_THRESHOLD})")
        print(f"Explanation:    {result['explanation']}")
        print("=====================\n")
        
    elif args.command == "rebuild":
        print("Trwa aktualizacja i indeksowanie bazy wiedzy...")
        classifier = PhishingClassifier()
        
        # JAWNE CZYSZCZENIE: Jeśli klasyfikator posiada instancję bazy pod self.db,
        # czyścimy ją w pamięci RAM przed rozpoczęciem wczytywania scenariuszy.
        if hasattr(classifier, "db") and hasattr(classifier.db, "clear"):
            classifier.db.clear()
            
        classifier.build_index_from_scenarios()
        print("Baza wektorowa FAISS została pomyślnie zaktualizowana i zapisana na dysku!")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()