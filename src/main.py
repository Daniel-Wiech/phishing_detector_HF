import argparse
import json
import sys
from classifier.detector import PhishingClassifier
from config import settings

def main():
    parser = argparse.ArgumentParser(
        description="Lokalny klasyfikator Phishingu oparty na wyszukiwaniu semantycznym, marginesach i FAISS."
    )
    subparsers = parser.add_subparsers(dest="command", help="Dostępne komendy")

    # Komenda klasyfikacji
    classify_parser = subparsers.add_parser("classify", help="Klasyfikuj treść wiadomości")
    classify_parser.add_argument("text", type=str, help="Treść wiadomości do przeanalizowania")

    # Komenda aktualizacji indeksu
    rebuild_parser = subparsers.add_parser("rebuild", help="Przebuduj i zaktualizuj lokalną bazę wektorową")
    rebuild_parser.add_argument(
        "--dedup", 
        action="store_true", 
        help="Włącza automatyczne usuwanie duplikatów semantycznych (per-klasa) podczas odbudowy bazy"
    )

    args = parser.parse_args()

    if args.command == "classify":
        classifier = PhishingClassifier()
        result = classifier.classify(args.text)
        
        print("\n=== WYNIK ANALIZY ===")
        print(f"Classification:   {result['classification'].upper()}")
        print(f"Category:         {result['category']}")
        print(f"Confidence:       {result['confidence'] * 100:.1f}%")
        print("-" * 30)
        print(f"Phishing Score:   {result['phishing_score']:.4f} (Próg: {settings.SCORE_THRESHOLD})")
        print(f"Legitimate Score: {result['legitimate_score']:.4f}")
        print(f"Margin:           {result['margin']:.4f} (Próg: {settings.MARGIN_THRESHOLD})")
        print("-" * 30)
        print(f"Reason:           {result['reason']}")
        
        if result['matched_examples']:
            print("\nNajbliższe przykłady w bazie:")
            for idx, match in enumerate(result['matched_examples'][:3], 1):
                print(f"  {idx}. [{match['label'].upper()}] (sim: {match['similarity']:.4f}) -> {match['example']}")
                
        print("=====================\n")
        
    elif args.command == "rebuild":
        if args.dedup:
            print("Trwa aktualizacja bazy wiedzy Z usuwaniem duplikatów semantycznych per-klasa...")
        else:
            print("Trwa standardowa aktualizacja i indeksowanie bazy wiedzy...")
            
        classifier = PhishingClassifier()
        
        if hasattr(classifier, "db") and hasattr(classifier.db, "clear"):
            classifier.db.clear()
            
        classifier.build_index_from_scenarios(deduplicate=args.dedup)
        print("Baza wektorowa FAISS została pomyślnie zaktualizowana i zapisana na dysku!")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()