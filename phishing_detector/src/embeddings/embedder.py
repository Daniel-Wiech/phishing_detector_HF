from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from config import settings

class LocalEmbedder:
    """Klasa odpowiedzialna za lokalne generowanie embeddingów przy użyciu sentence-transformers."""
    
    def __init__(self, model_name: str = settings.MODEL_NAME):
        # Automatycznie pobiera i cache'uje model lokalnie przy pierwszym uruchomieniu
        self.model = SentenceTransformer(model_name)
        
    def _prepare_text(self, text: str, is_query: bool = False) -> str:
        """
        Modele z rodziny E5 wymagają specjalnego prefiksu w zależności od zadania:
        - 'query: ' dla wyszukiwanego tekstu (zapytania)
        - 'passage: ' dla dokumentów/wzorców w bazie wiedzy
        """
        prefix = "query: " if is_query else "passage: "
        if not text.startswith(prefix):
            return f"{prefix}{text}"
        return text

    def get_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
        """Generuje pojedynczy wektor embeddingu."""
        prepared_text = self._prepare_text(text, is_query)
        embedding = self.model.encode(prepared_text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding

    def get_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Generuje wektory embeddingów dla listy tekstów (batch processing)."""
        prepared_texts = [self._prepare_text(t, is_query) for t in texts]
        embeddings = self.model.encode(prepared_texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings