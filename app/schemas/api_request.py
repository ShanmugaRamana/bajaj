# app/schemas/api_request.py
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class APIRequest(BaseModel):
    documents: Optional[HttpUrl] = Field(None, description="URL to a single PDF, DOCX, or email document.")
    questions: List[str] = Field(..., min_items=1, description="A list of natural language questions to ask.")