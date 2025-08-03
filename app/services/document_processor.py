# app/services/document_processor.py
import httpx
import hashlib
from io import BytesIO
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def _chunk_text(text: str) -> List[str]:
    """Splits a long text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

def process_and_hash_pdf_from_path(path: str) -> tuple[list[str], str | None]:
    """Processes a local PDF, returning chunks and a content hash."""
    try:
        with open(path, "rb") as f:
            pdf_bytes = f.read()
            content_hash = hashlib.sha256(pdf_bytes).hexdigest()
        
        reader = PdfReader(BytesIO(pdf_bytes))
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        chunks = _chunk_text(text)
        return chunks, content_hash
    except Exception as e:
        print(f"Failed to process or hash PDF from path {path}: {e}")
        return [], None

def get_content_hash_from_url(url: str) -> tuple[str | None, bytes | None]:
    """Downloads a document and returns its content hash and bytes."""
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        pdf_bytes = response.content
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()
        return content_hash, pdf_bytes
    except Exception as e:
        print(f"Failed to download or hash document from URL {url}: {e}")
        return None, None