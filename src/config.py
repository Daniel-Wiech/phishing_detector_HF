import os
from pydantic import BaseModel

class Config(BaseModel):
    MODEL_NAME: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DIM: int = 1024
    
    # Progi klasyfikacji
    SCORE_THRESHOLD: float = 0.63
    MARGIN_THRESHOLD: float = 0.004
    TOP_K: int = 5
    
    # Ścieżki
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH: str = os.path.join(BASE_DIR, "data", "scenarios.json")
    INDEX_PATH: str = os.path.join(BASE_DIR, "vector_db", "faiss.index")
    METADATA_PATH: str = os.path.join(BASE_DIR, "vector_db", "metadata.json")

settings = Config()