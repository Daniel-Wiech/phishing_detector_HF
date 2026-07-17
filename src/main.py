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

    # Komenda aktualizacji indeksu (teraz z opcjonalną flagą --dedup)
    rebuild_parser = subparsers.add_parser("rebuild", help="Przebuduj i zaktualizuj lokalną bazę wektorową")
    rebuild_parser.add_argument(
        "--dedup", 
        action="store_true", 
        help="Włącza automatyczne usuwanie duplikatów semantycznych podczas odbudowy bazy"
    )

    args = parser.parse_args()

    if args.command == "classify":
        classifier = PhishingClassifier()
        result = classifier.classify(args.text)
        
        print("\n=== WYNIK ANALIZY ===")
        print(f"Classification: {result['classification'].capitalize()}")
        print(f"Category:       {result['category']}")
        print(f"Confidence:     {result['confidence'] * 100:.1f}%")
        print(f"Similarity:     {result['similarity']:.4f} (Próg: {settings.SIMILARITY_THRESHOLD})")
        print(f"Explanation:    {result['explanation']}")
        print("=====================\n")
        
    elif args.command == "rebuild":
        if args.dedup:
            print("Trwa aktualizacja bazy wiedzy Z usuwaniem duplikatów semantycznych...")
        else:
            print("Trwa standardowa aktualizacja i indeksowanie bazy wiedzy...")
            
        classifier = PhishingClassifier()
        
        if hasattr(classifier, "db") and hasattr(classifier.db, "clear"):
            classifier.db.clear()
            
        # Przekazujemy flagę True/False do metody klasyfikatora
        classifier.build_index_from_scenarios(deduplicate=args.dedup)
        print("Baza wektorowa FAISS została pomyślnie zaktualizowana i zapisana na dysku!")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()