# app/main.py
import os
import asyncio
from fastapi import FastAPI, Depends, HTTPException, status
from app.core.security import verify_token
from app.schemas.api_request import APIRequest
from app.schemas.api_response import APIResponse
from app.services import document_processor, vector_store, llm_service

# Define the base path for the pre-built knowledge store
BASE_KNOWLEDGE_STORE_PATH = "app/data/vector_store/base_knowledge"

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System",
    description="Process natural language queries against large documents.",
    version="1.0.0"
)

# Load the base knowledge FAISS index on startup
try:
    if os.path.exists(f"{BASE_KNOWLEDGE_STORE_PATH}.faiss"):
        base_index, base_chunks = vector_store.load_faiss_index(BASE_KNOWLEDGE_STORE_PATH)
        print("✅ Base knowledge vector store loaded successfully.")
    else:
        base_index, base_chunks = None, None
        print("⚠️ Warning: Base knowledge vector store not found. System will only work with provided documents.")
except Exception as e:
    base_index, base_chunks = None, None
    print(f"❌ Error loading base knowledge vector store: {e}")


@app.post(
    "/api/v1/hackrx/run",
    response_model=APIResponse,
    dependencies=[Depends(verify_token)],
    tags=["Query & Retrieval"]
)
async def run_submission(request: APIRequest):
    """
    Processes questions against a specified document or the base knowledge.
    """
    answers = []
    
    if request.documents:
        # Scenario 1: Process a specific document provided by URL
        try:
            doc_chunks = document_processor.process_pdf_from_url(str(request.documents))
            if not doc_chunks:
                raise HTTPException(status_code=400, detail="Could not extract text from the provided document.")
            
            # Create an in-memory FAISS index for this specific document
            index, chunks = vector_store.create_faiss_index_from_chunks(doc_chunks)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    else:
        # Scenario 2: Use the pre-loaded base knowledge
        if base_index is None or base_chunks is None:
            raise HTTPException(status_code=503, detail="Base knowledge base is not loaded or available.")
        index, chunks = base_index, base_chunks

    # Asynchronously process all questions
    async def process_question(question):
        relevant_chunks = vector_store.search_faiss_index(index, chunks, question, k=7)
        return await llm_service.get_answer_from_llm(question, relevant_chunks)

    tasks = [process_question(q) for q in request.questions]
    answers = await asyncio.gather(*tasks)

    return APIResponse(answers=answers)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Intelligent Query–Retrieval System is running."}