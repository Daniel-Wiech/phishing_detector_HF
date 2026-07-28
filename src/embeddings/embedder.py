import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config import settings

class LocalEmbedder:
    """Generowanie embeddingów z L2-normalizacją i sprawdzaniem wymiaru (1024)."""
    
    def __init__(self, model_name: str = settings.MODEL_NAME, expected_dim: int = settings.EMBEDDING_DIM):
        self.model = SentenceTransformer(model_name)
        self.expected_dim = expected_dim

    def _prepare_text(self, text: str, is_query: bool = False) -> str:
        prefix = "query: " if is_query else "passage: "
        return text if text.startswith(prefix) else f"{prefix}{text}"

    def _normalize_and_validate(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = embeddings.astype("float32")
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
            
        if embeddings.shape[1] != self.expected_dim:
            raise ValueError(f"Oczekiwano wymiaru {self.expected_dim}, otrzymano {embeddings.shape[1]}")
            
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return embeddings / norms

    def get_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
        prepared = self._prepare_text(text, is_query)
        raw_emb = self.model.encode(prepared, convert_to_numpy=True, normalize_embeddings=True)
        return self._normalize_and_validate(raw_emb)[0]

    def get_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        prepared = [self._prepare_text(t, is_query) for t in texts]
        raw_embs = self.model.encode(prepared, convert_to_numpy=True, normalize_embeddings=True)
        return self._normalize_and_validate(raw_embs)