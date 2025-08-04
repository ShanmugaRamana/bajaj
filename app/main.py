# app/main.py
import os
import asyncio
import pickle
import hashlib
from fastapi import FastAPI, Depends, HTTPException, status
from app.core.security import verify_token
from app.schemas.api_request import APIRequest
from app.schemas.api_response import APIResponse
from app.services import document_processor, vector_store, llm_service

# --- KNOWLEDGE BASE AND CACHE SETUP ---
BASE_KNOWLEDGE_STORE_PATH = "app/data/vector_store/base_knowledge"
HASH_STORE_PATH = "app/data/vector_store/base_knowledge_hashes.pkl"
base_index, base_chunks, text_hashes = None, None, set()

app = FastAPI(
    title="LLM-Powered Intelligent Query Retrieval System",
    description="Process natural language queries against large documents.",
    version="1.0.0"
)

@app.on_event("startup")
def load_base_knowledge():
    """Load pre-built knowledge base and hashes on startup."""
    global base_index, base_chunks, text_hashes
    try:
        if os.path.exists(f"{BASE_KNOWLEDGE_STORE_PATH}.faiss"):
            base_index, base_chunks = vector_store.load_faiss_index(BASE_KNOWLEDGE_STORE_PATH)
            print("✅ Base knowledge vector store loaded successfully.")

        if os.path.exists(HASH_STORE_PATH):
            with open(HASH_STORE_PATH, "rb") as f:
                text_hashes = pickle.load(f)
            print(f"✅ Loaded {len(text_hashes)} base document text hashes for caching.")
    except Exception as e:
        print(f"❌ Error during startup loading: {e}")

# --- API ENDPOINTS ---
@app.post(
    "/api/v1/hackrx/run",
    response_model=APIResponse,
    dependencies=[Depends(verify_token)],
    tags=["Query & Retrieval"]
)
async def run_submission(request: APIRequest):
    index, chunks = None, None

    if request.documents:
        pdf_bytes = document_processor.download_pdf_from_url(str(request.documents))
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Could not download document from URL.")

        # --- TEXT-BASED CACHING LOGIC ---
        extracted_text = document_processor.extract_text_from_bytes(pdf_bytes)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the provided document.")

        content_hash = hashlib.sha256(extracted_text.encode('utf-8')).hexdigest()

        if content_hash in text_hashes:
            print("✅ Cache hit. Using pre-built base knowledge index.")
            index, chunks = base_index, base_chunks
        else:
            print("⚠️ Cache miss. Processing new document in real-time.")
            doc_chunks = document_processor._chunk_text(extracted_text)
            index, chunks = vector_store.create_faiss_index_from_chunks(doc_chunks)
    else:
        print("ℹ️ No document URL provided. Using base knowledge index.")
        index, chunks = base_index, base_chunks

    if index is None:
        raise HTTPException(status_code=503, detail="Knowledge base is not available.")

    # Asynchronously process all questions
    async def process_question(question):
        relevant_chunks = vector_store.search_faiss_index(index, chunks, question, k=12)
        return await llm_service.get_answer_from_llm(question, relevant_chunks)

    tasks = [process_question(q) for q in request.questions]
    answers = await asyncio.gather(*tasks)

    return APIResponse(answers=answers)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Intelligent Query Retrieval System is running."}