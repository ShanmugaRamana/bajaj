# app/services/llm_service.py
import httpx
from typing import List
from app.core.config import settings

async def get_answer_from_llm(question: str, context: List[str]) -> str:
    """Gets a concise answer from the LLM based on the provided context."""
    
    context_str = "\n---\n".join(context)
    
    prompt = f"""
    You are a highly intelligent assistant. Your task is to answer the user's question based *only* on the provided context.
    - Be concise and directly answer the question.
    - Do not use any information outside of the provided context.
    - If the context does not contain the answer, state that the information is not available in the provided document.

    CONTEXT:
    ---
    {context_str}
    ---

    QUESTION: {question}

    ANSWER:
    """
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                },
                json={
                    "model": settings.LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()
            answer = data['choices'][0]['message']['content'].strip()
            return answer

    except httpx.HTTPStatusError as e:
        # Log the error for debugging
        print(f"LLM API Error: {e.response.status_code} - {e.response.text}")
        return "Error: Could not get a response from the language model."
    except Exception as e:
        print(f"An unexpected error occurred in LLM service: {e}")
        return "Error: An unexpected error occurred while processing the request."