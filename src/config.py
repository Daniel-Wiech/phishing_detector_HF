import os
from pydantic import BaseModel, Field

class Config(BaseModel):
    # Model: Multilingual-E5-Large najlepiej radzi sobie z j. polskim lokalnie
    # Prefiks "query: " i "passage: " jest wymagany dla modeli E5 v2
    MODEL_NAME: str = "intfloat/multilingual-e5-large"
    
    # Próg decyzyjny podobieństwa cosinusowego
    SIMILARITY_THRESHOLD: float = 0.88
    
    # Ścieżki do plików
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH: str = os.path.join(BASE_DIR, "data", "scenarios.json")
    INDEX_PATH: str = os.path.join(BASE_DIR, "vector_db", "faiss.index")
    METADATA_PATH: str = os.path.join(BASE_DIR, "vector_db", "metadata.json")

settings = Config()