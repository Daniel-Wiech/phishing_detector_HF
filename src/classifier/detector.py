import os
import json
import numpy as np
from typing import Dict, Any, List, Union
from config import settings
from embeddings.embedder import LocalEmbedder
from vector_db.faiss_store import FaissVectorStore
from data.preprocessing import EmailPreprocessor, EmailInput

class PhishingClassifier:
    def __init__(
        self, 
        score_threshold: float = settings.SCORE_THRESHOLD,
        margin_threshold: float = settings.MARGIN_THRESHOLD,
        top_k: int = settings.TOP_K
    ):
        self.embedder = LocalEmbedder()
        self.db = FaissVectorStore()
        self.score_threshold = score_threshold
        self.margin_threshold = margin_threshold
        self.top_k = top_k
        
        if not self.db.load():
            self.build_index_from_scenarios()

    def build_index_from_scenarios(self, deduplicate: bool = True) -> None:
        self.db.clear()
        
        if not os.path.exists(settings.DATA_PATH):
            raise FileNotFoundError(f"Nie znaleziono pliku scenariuszy: {settings.DATA_PATH}")

        with open(settings.DATA_PATH, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        accepted_vectors: List[np.ndarray] = []
        accepted_metadata: List[Dict[str, Any]] = []
        
        DEDUPLICATION_THRESHOLD = 0.98

        for scenario in scenarios:
            label = scenario.get("label", "phishing")
            category = scenario.get("category", "General")
            
            for example in scenario.get("examples", []):
                curr_vec = self.embedder.get_embedding(example, is_query=False)
                is_duplicate = False

                if deduplicate and accepted_vectors:
                    # Deduplikacja TYLKO wewnątrz tej samej klasy!
                    same_class_indices = [
                        idx for idx, meta in enumerate(accepted_metadata) if meta["label"] == label
                    ]
                    
                    if same_class_indices:
                        class_vectors = np.array([accepted_vectors[i] for i in same_class_indices])
                        similarities = np.dot(class_vectors, curr_vec)
                        if np.any(similarities >= DEDUPLICATION_THRESHOLD):
                            is_duplicate = True

                if not is_duplicate:
                    accepted_vectors.append(curr_vec)
                    accepted_metadata.append({
                        "label": label,
                        "category": category,
                        "example": example
                    })

        if accepted_vectors:
            vectors_np = np.array(accepted_vectors).astype("float32")
            self.db.add_vectors(vectors_np, accepted_metadata)
            self.db.save()

    def classify(self, message: Union[str, EmailInput]) -> Dict[str, Any]:
        formatted_text = EmailPreprocessor.process(message)
        query_vector = self.embedder.get_embedding(formatted_text, is_query=True)
        
        search_results = self.db.search(query_vector, k=self.top_k)
        
        if not search_results:
            return {
                "classification": "uncertain",
                "confidence": 0.0,
                "phishing_score": 0.0,
                "legitimate_score": 0.0,
                "margin": 0.0,
                "matched_examples": [],
                "category": "Unknown",
                "reason": "Baza wiedzy jest pusta."
            }

        phishing_sims = [sim for meta, sim in search_results if meta["label"] == "phishing"]
        legitimate_sims = [sim for meta, sim in search_results if meta["label"] == "legitimate"]

        # Uśredniamy do 2 najlepszych trafień w danej kategorii z pobranej piątki
        phishing_score = float(np.mean(phishing_sims[:2])) if phishing_sims else 0.0
        legitimate_score = float(np.mean(legitimate_sims[:2])) if legitimate_sims else 0.0

        margin = phishing_score - legitimate_score
        
        matched_examples = [
            {"example": meta["example"], "label": meta["label"], "similarity": round(sim, 4)}
            for meta, sim in search_results[:5]
        ]
        
        top_category = search_results[0][0]["category"] if search_results else "Unknown"

        # Logika decyzyjna z wykorzystaniem SCORE + MARGIN
        classification = "uncertain"
        if phishing_score >= self.score_threshold and margin >= self.margin_threshold:
            classification = "phishing"
            confidence = phishing_score
            reason = f"Wysokie podobieństwo do phishingu ({phishing_score:.2f}) oraz margines ({margin:.2f}) powyżej progu."
        elif legitimate_score >= self.score_threshold and (-margin) >= self.margin_threshold:
            classification = "legitimate"
            confidence = legitimate_score
            reason = f"Wysokie podobieństwo do wiadomości bezpiecznych ({legitimate_score:.2f})."
        else:
            confidence = max(phishing_score, legitimate_score)
            reason = (f"Wynik niepewny (uncertain). Margines ({margin:.2f}) lub Wynik "
                      f"(P: {phishing_score:.2f}, L: {legitimate_score:.2f}) nie spełniają wymogów.")

        return {
            "classification": classification,
            "confidence": round(confidence, 4),
            "phishing_score": round(phishing_score, 4),
            "legitimate_score": round(legitimate_score, 4),
            "margin": round(margin, 4),
            "matched_examples": matched_examples,
            "category": top_category,
            "reason": reason
        }