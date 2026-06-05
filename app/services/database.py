import chromadb
from app.core.config import settings

class LocalVectorDB:
    def __init__(self):
        print(f"Booting up Local Vector DB at {settings.DB_DIR}...")
        
        # 1. Initialize the persistent client bypassing RAM-only storage
        self.client = chromadb.PersistentClient(path=settings.DB_DIR)
        
         # 2. Establish the primary table (collection) idempotently 
        self.collection = self.client.get_or_create_collection(name="rag_documents")

    def insert_vectors(self, chunks: list[str], vectors: list[list[float]], chunk_ids: list[str], metadatas: list[dict]):
        """
        Takes processed text, vectors, string IDs, AND metadata, saving them to the SSD.
        """
        self.collection.add(
            documents=chunks, 
            embeddings=vectors, 
            ids=chunk_ids,
            metadatas=metadatas 
        )
    
    def search_vectors(self, query_vector: list[float], top_k: int = 3):
        """
        Takes a single mathematical vector and asks ChromaDB to find the 
        closest matching chunks using the HNSW index.
        """
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        return results


# Instantiate the single, shared connection
db = LocalVectorDB()