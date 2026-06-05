from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.services.ingestion import ingestion_service
from app.services.embedding import embedder
from app.services.database import db

# This acts like a mini-FastAPI app that we will plug into main.py later
router = APIRouter()

# We use Pydantic to strictly define what a search request should look like
class SearchQuery(BaseModel):
    text: str
    top_k: int = 3  # Return the top 3 closest matches by default

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Receives user file uploads and queues them for background processing.
    """
    try:
        # 1. Read the file bytes asynchronously
        raw_binary_data = await file.read()

        # 2. Decode the bytes into a UTF-8 string
        text_content = raw_binary_data.decode("utf-8")

        # 3. Extract the filename for source tracking
        file_identifier = file.filename

        # 4. Push the payload to the ingestion queue (handles backpressure)
        await ingestion_service.enqueue_document(file_identifier, text_content)

        # 5. Return immediate success response without waiting for processing
        return {
            "status": "success", 
            "message": f"'{file_identifier}' has been queued for processing."
        }

    except UnicodeDecodeError:
        # 6. Guard against non-text file uploads crashing the decoder
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a valid text file."
        )

@router.post("/search")
async def search_documents(query: SearchQuery):
    """
    Takes a user query, converts it to a vector, and searches the database.
    """
    # 1. Convert the user's text into mathematical vectors (Wrapped in a list!)
    vectors = embedder.generate_embeddings([query.text])
    
    # 2. Extract the exact vector array from the returned list (Index 0)
    query_vector = vectors[0]
    
    # 3. Query the Chroma database using our new encapsulated method
    db_results = db.search_vectors(query_vector, top_k=query.top_k)
    
    # 4. Return the raw database results to the user
    return {
        "query": query.text,
        "matches": db_results
    }
