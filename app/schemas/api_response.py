# app/schemas/api_response.py
from pydantic import BaseModel
from typing import List

class APIResponse(BaseModel):
    answers: List[str]