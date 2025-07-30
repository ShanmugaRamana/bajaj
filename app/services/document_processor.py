# app/services/document_processor.py
import httpx
from pypdf import PdfReader
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def process_pdf_from_url(url: str) -> List[str]:
    """Downloads and processes a PDF from a URL."""
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return _chunk_text(text)
    except httpx.RequestError as e:
        raise Exception(f"Failed to download document: {e}")
    except Exception as e:
        raise Exception(f"Failed to process PDF from URL: {e}")

def process_pdf_from_path(path: str) -> List[str]:
    """Processes a PDF from a local file path."""
    try:
        reader = PdfReader(path)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return _chunk_text(text)
    except Exception as e:
        raise Exception(f"Failed to process PDF from path {path}: {e}")

def _chunk_text(text: str) -> List[str]:
    """Splits a long text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)