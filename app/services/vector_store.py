# app/services/vector_store.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Use a pre-trained model for generating embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_and_save_faiss_index(chunks: List[str], path_prefix: str):
    """Creates a FAISS index and saves it to disk."""
    embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, f"{path_prefix}.faiss")
    with open(f"{path_prefix}.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_faiss_index(path_prefix: str) -> Tuple[faiss.Index, List[str]]:
    """Loads a FAISS index and its corresponding chunks from disk."""
    index = faiss.read_index(f"{path_prefix}.faiss")
    with open(f"{path_prefix}.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def create_faiss_index_from_chunks(chunks: List[str]) -> Tuple[faiss.Index, List[str]]:
    """Creates an in-memory FAISS index without saving."""
    embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def search_faiss_index(index: faiss.Index, chunks: List[str], query: str, k: int = 5) -> List[str]:
    """Searches the FAISS index for the most relevant chunks."""
    query_embedding = embedding_model.encode([query])
    _, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0] if i < len(chunks)]