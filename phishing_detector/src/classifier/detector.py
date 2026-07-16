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

    def build_index_from_scenarios(self, data_path: str = settings.DATA_PATH) -> None:
        """Wczytuje scenariusze z pliku JSON, tworzy embeddingi i buduje indeks wektorowy."""
        self.db.clear()
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Brak pliku bazy wiedzy o scenariuszach pod ścieżką: {data_path}")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            scenarios: List[Dict[str, Any]] = json.load(f)
            
        texts_to_embed = []
        metadata_list = []
        
        for scenario in scenarios:
            category = scenario["category"]
            risk = scenario["risk"]
            description = scenario["description"]
            
            # Dodajemy przykłady jako punkty odniesienia
            for example in scenario["examples"]:
                texts_to_embed.append(example)
                metadata_list.append({
                    "category": category,
                    "risk": risk,
                    "matched_example": example,
                    "description": description
                })
                
        if texts_to_embed:
            embeddings = self.embedder.get_embeddings(texts_to_embed, is_query=False)
            self.db.add_vectors(embeddings, metadata_list)
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