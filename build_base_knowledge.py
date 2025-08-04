# build_base_knowledge.py
import os
import hashlib
import pickle
from app.services.document_processor import extract_text_from_bytes
from app.services.document_processor import _chunk_text
from app.services.vector_store import create_and_save_faiss_index

print("Starting to build the base knowledge vector store...")

BASE_KNOWLEDGE_PATH = "app/data/base_knowledge"
FAISS_STORE_PATH = "app/data/vector_store/base_knowledge"
HASH_STORE_PATH = "app/data/vector_store/base_knowledge_hashes.pkl"

def main():
    pdf_files = [f for f in os.listdir(BASE_KNOWLEDGE_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {BASE_KNOWLEDGE_PATH}. Aborting.")
        return

    all_chunks = []
    text_hashes = set()

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        file_path = os.path.join(BASE_KNOWLEDGE_PATH, pdf_file)

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        text = extract_text_from_bytes(pdf_bytes)
        if text:
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            text_hashes.add(text_hash)
            chunks = _chunk_text(text)
            all_chunks.extend(chunks)

    if not all_chunks:
        print("No text could be extracted. Aborting.")
        return

    print("Creating and saving FAISS index...")
    os.makedirs(os.path.dirname(FAISS_STORE_PATH), exist_ok=True)
    create_and_save_faiss_index(all_chunks, FAISS_STORE_PATH)

    print(f"Saving {len(text_hashes)} text content hashes...")
    with open(HASH_STORE_PATH, "wb") as f:
        pickle.dump(text_hashes, f)

    print("✅ Successfully built knowledge base and text hashes.")

if __name__ == "__main__":
    main()