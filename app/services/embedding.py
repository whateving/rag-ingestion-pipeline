from sentence_transformers import SentenceTransformer
from app.core.config import settings

class EmbeddingEngine:
    def __init__(self):

        """
        Initializes the AI model in memory upon server boot.
        This operation blocks momentarily but guarantees zero-latency inference later.
        """
        
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL} (This may take a second...)")
        
        # 1. Load the localized model into RAM
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """
        Takes a list of text chunks and returns a list of high-dimensional math vectors.
        """

        # 2. Safeguard against empty payloads to prevent model crashes
        if not chunks:
            return []
            
        #3. Execute the mathematical transformation and format for ChromaDB
        vectors = self.model.encode(chunks).tolist()
        
        return vectors

# Create our single, shared instance of the AI engine
embedder = EmbeddingEngine()