from pydantic import BaseModel

class Settings(BaseModel):
    # App Settings
    PROJECT_NAME: str = "High-Throughput Local RAG"
    
    # 8GB Mac RAM Constraints
    MAX_QUEUE_SIZE: int = 50           # Strict backpressure limit: max files waiting in RAM
    CHUNK_SIZE_TOKENS: int = 200       # How large each text chunk should be
    CHUNK_OVERLAP_TOKENS: int = 40     # Sliding window overlap
    
    # Local Storage Paths
    DB_DIR: str = "./local_rag_storage"
    
    # AI Models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Instantiate the settings so we can import them anywhere
settings = Settings()