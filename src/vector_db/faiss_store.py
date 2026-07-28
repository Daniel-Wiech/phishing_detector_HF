import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from config import settings

class FaissVectorStore:
    def __init__(self, dimension: int = settings.EMBEDDING_DIM):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        if len(vectors) != len(metadata_list):
            raise ValueError("Liczba wektorów i metadanych musi być zgodna!")
        
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadata_list)

    def search(self, query_vector: np.ndarray, k: int = 50) -> List[Tuple[Dict[str, Any], float]]:
        if self.index.ntotal == 0:
            return []
            
        query_vector = np.atleast_2d(query_vector).astype("float32")
        faiss.normalize_L2(query_vector)
        
        actual_k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(query_vector, actual_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(sim)))
        return results

    def save(self, index_path: str = settings.INDEX_PATH, metadata_path: str = settings.METADATA_PATH) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str = settings.INDEX_PATH, metadata_path: str = settings.METADATA_PATH) -> bool:
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            return True
        return False