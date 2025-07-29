# build_base_knowledge.py
import os
from app.services.document_processor import process_pdf_from_path
from app.services.vector_store import create_and_save_faiss_index

print("Starting to build the base knowledge vector store...")

BASE_KNOWLEDGE_PATH = "app/data/base_knowledge"
FAISS_STORE_PATH = "app/data/vector_store/base_knowledge"

def main():
    pdf_files = [f for f in os.listdir(BASE_KNOWLEDGE_PATH) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {BASE_KNOWLEDGE_PATH}. Aborting.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")
    all_chunks = []
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        file_path = os.path.join(BASE_KNOWLEDGE_PATH, pdf_file)
        chunks = process_pdf_from_path(file_path)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No text could be extracted from the PDF files. Aborting.")
        return

    print(f"Total chunks created: {len(all_chunks)}")
    print("Creating and saving FAISS index...")
    
    os.makedirs(os.path.dirname(FAISS_STORE_PATH), exist_ok=True)
    create_and_save_faiss_index(all_chunks, FAISS_STORE_PATH)
    
    print(f"Successfully built and saved the knowledge base to {FAISS_STORE_PATH}.faiss")

if __name__ == "__main__":
    main()