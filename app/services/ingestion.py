import asyncio
import uuid
from app.core.config import settings
from app.services.embedding import embedder
from app.services.database import db

class IngestionService:
    def __init__(self):
        """
        Initializes the service. The queue starts as None because asyncio primitives 
        MUST be instantiated inside an active, running event loop (not during import).
        """
        self.queue: asyncio.Queue | None = None

    def initialize_queue(self):
        """
        Instantiates the bounded queue inside the active event loop upon server startup.
        Locks the maximum capacity using our hardware-defined constraint settings.
        """
        self.queue = asyncio.Queue(maxsize=settings.MAX_QUEUE_SIZE)

    def _chunk_text(self, text: str):
        """
        Implements a Sliding Window Chunking text splitter using a Python generator.
        Instead of loading an array of hundreds of chunks into memory all at once, 
        it lazily yields one chunk at a time, protecting RAM allocations.
        """
        # 1. Break down the string into a list of individual words by spaces
        word_list = text.split()
        length_of_the_word_list = len(word_list)
        
        # 2. Calculate the moving step size (e.g., 200 chunk size - 40 overlap = 160 new words per step)
        step = settings.CHUNK_SIZE_TOKENS - settings.CHUNK_OVERLAP_TOKENS
        index = 0

        # 3. Slide through the word list until we reach the end of the text
        while index < length_of_the_word_list:
            # Determine the boundary end for the current window slice
            end_index = index + settings.CHUNK_SIZE_TOKENS
            
            # Slice the specific subset of words for this window
            chunk_words = word_list[index:end_index]

            # Yield the words joined back by a space. Using 'yield' pauses the function 
            # and returns the string to the consumer, discarding it when done.
            yield " ".join(chunk_words)

            # Move the pointer forward by our calculated step size to keep the overlap steady
            index += step

    async def enqueue_document(self, filename: str, raw_text: str):
        """
        THE PRODUCER: Receives raw payloads from API routes and places them into the queue.
        Utilizes non-blocking backpressure to halt ingestion if the queue fills up.
        """
        # 1. Package incoming data into a predictable, standardized dictionary structure
        payload = {
            "filename": filename, 
            "text": raw_text
        }
        
        # 2. Asynchronously push to the queue. If the queue already contains 50 items, 
        # this line will automatically suspend this execution path until space opens up.
        await self.queue.put(payload)

    async def worker_loop(self):
        """
        THE CONSUMER: Runs continuously in an infinite background loop. It polls the queue, 
        orchestrates the chunking sequence, invokes the AI brain, and triggers database writes.
        """
        while True:
            # 1. Await an item from the queue. This is non-blocking.
            item = await self.queue.get()
            
            # 2. Extract BOTH the text AND the filename from our dictionary
            filename, raw_text = item["filename"], item["text"]
            
            # 3. Instantiate our generator and cast it to a native list. 
            chunk_generator = self._chunk_text(raw_text)
            chunk_list = list(chunk_generator)
            
            # 4. Generate unique IDs
            chunk_id_list = [str(uuid.uuid4()) for _ in chunk_list]
            
            # 4.5 NEW METADATA FIX: Create a source dictionary for every chunk
            metadata_list = [{"source": filename} for _ in chunk_list]
            
            # 5. Route chunks to the AI Brain (Phase 3)
            vectors = embedder.generate_embeddings(chunk_list)
            
            # 6. Save everything to Phase 2 (Storage Layer).
            # NOTICE: We are now passing the metadata_list as the fourth argument!
            db.insert_vectors(chunk_list, vectors, chunk_id_list, metadata_list)
            
            # 7. CRITICAL: Signal back to the queue that this item is done.
            self.queue.task_done()

# Create our single, shared instance to be imported globally across services
ingestion_service = IngestionService()