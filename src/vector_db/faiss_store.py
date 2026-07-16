import faiss
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from config import settings

class FaissVectorStore:
    """Moduł integracji z biblioteką FAISS służącą jako lokalna baza wektorowa."""
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        # Używamy IndexFlatIP (Inner Product) na znormalizowanych wektorach, co daje Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        # Przechowywanie metadanych powiązanych z indeksami FAISS
        self.metadata: List[Dict[str, Any]] = []

    def clear(self) -> None:
        """Całkowicie czyści indeks FAISS oraz metadane w pamięci RAM."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        """Dodaje wektory wraz z powiązanymi metadanymi do indeksu."""
        assert len(vectors) == len(metadata_list), "Rozmiar wektorów i metadanych musi być identyczny!"
        self.index.add(vectors.astype('float32'))
        self.metadata.extend(metadata_list)

    def search(self, query_vector: np.ndarray, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """Wyszukuje 'k' najbliższych sąsiadów dla podanego wektora zapytania."""
        if self.index.ntotal == 0:
            return []
        
        # Formatowanie wektora wejściowego do formatu 2D float32
        query_vector = np.atleast_2d(query_vector).astype('float32')
        similarities, indices = self.index.search(query_vector, k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(sim)))
        return results

    def save(self, index_path: str = settings.INDEX_PATH, metadata_path: str = settings.METADATA_PATH) -> None:
        """Zapisuje stan bazy FAISS na dysku."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str = settings.INDEX_PATH, metadata_path: str = settings.METADATA_PATH) -> bool:
        """Wczytuje bazę FAISS z dysku."""
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            return True
        return False