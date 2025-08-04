# app/services/document_processor.py
import httpx
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

def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extracts all text from a PDF provided as bytes."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        print(f"Failed to extract text from bytes: {e}")
        return ""

def download_pdf_from_url(url: str) -> bytes | None:
    """Downloads a PDF from a URL and returns its content as bytes."""
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Failed to download document from URL {url}: {e}")
        return None