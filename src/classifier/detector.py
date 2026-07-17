import json
import os
import numpy as np
from typing import Dict, Any, List
from config import settings
from embeddings.embedder import LocalEmbedder
from vector_db.faiss_store import FaissVectorStore

class PhishingClassifier:
    """Główny moduł orkiestracji klasyfikacji semantycznej."""
    
    def __init__(self, threshold: float = settings.SIMILARITY_THRESHOLD):
        self.embedder = LocalEmbedder()
        # multilingual-e5-large posiada wymiarowość (dimension) równą 1024
        self.db = FaissVectorStore(dimension=1024)
        self.threshold = threshold
        
        # Spróbuj załadować istniejący indeks, w przeciwnym razie zbuduj go automatycznie
        if not self.db.load():
            self.build_index_from_scenarios()

    def build_index_from_scenarios(self, deduplicate: bool = False):
        # 1. Czyszczenie bazy na start
        self.db.clear()
        
        # 2. Wczytanie scenariuszy z pliku json
        with open(settings.DATA_PATH, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
            
        accepted_vectors = []
        accepted_metadata = []
        
        # Próg deduplikacji (95% podobieństwa)
        DEDUPLICATION_THRESHOLD = 0.95 
        removed_duplicates_count = 0
        for scenario in scenarios:
            for example in scenario["examples"]:
                current_vector = self.embedder.get_embedding(example) 
                is_duplicate = False
                
                # Filtrowanie uruchomi się TYLKO, gdy flaga deduplicate == True
                if deduplicate and accepted_vectors:
                    matrix = np.array(accepted_vectors)
                    similarities = np.dot(matrix, current_vector)
                    
                    if np.any(similarities >= DEDUPLICATION_THRESHOLD):
                        is_duplicate = True
                        removed_duplicates_count += 1  # Inkrementacja licznika
                        print(f" -> Pominięto duplikat semantyczny: '{example}'")
                
                        
                # Zapisujemy element jeśli nie jest duplikatem LUB gdy deduplikacja jest wyłączona
                if not is_duplicate:
                    accepted_vectors.append(current_vector)
                    accepted_metadata.append({
                        "category": scenario["category"],
                        "risk": scenario["risk"],
                        "matched_example": example,
                        "description": scenario["description"]
                    })
                    
        if deduplicate:
            print(f"\n=== PODSUMOWANIE DEDUPLIKACJI ===")
            print(f"Usunięto zduplikowanych wektorów: {removed_duplicates_count}")
            print(f"Pozostało unikalnych wektorów w bazie: {len(accepted_vectors)}")
            print("=================================\n")
        # 3. Zapis do bazy
        if accepted_vectors:
            vectors_np = np.array(accepted_vectors).astype('float32')
            self.db.add_vectors(vectors_np, accepted_metadata)
            self.db.save()

    def classify(self, message: str) -> Dict[str, Any]:
        """Klasyfikuje przesłaną wiadomość na podstawie odległości semantycznej."""
        query_vector = self.embedder.get_embedding(message, is_query=True)
        results = self.db.search(query_vector, k=1)
        
        if not results:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "category": "Brak",
                "similarity": 0.0,
                "explanation": "Baza wiedzy jest pusta."
            }
            
        best_match, similarity = results[0]
        
        # Klasyfikacja na podstawie zdefiniowanego progu
        classification = "phishing" if similarity >= self.threshold else "legitimate/unknown"
        
        explanation = (
            f"Wiadomość jest wysoce podobna do znanych kampanii phishingowych związanych z kategorią: '{best_match['category']}' "
            f"(Najbliższy wzorzec: '{best_match['matched_example']}')."
            if classification == "phishing" else
            "Nie dopasowano wiadomości do żadnego ze znanych schematów ataków phishingowych."
        )
        
        return {
            "classification": classification,
            "confidence": round(similarity, 4),
            "category": best_match["category"] if classification == "phishing" else "unknown",
            "similarity": round(similarity, 4),
            "explanation": explanation
        }