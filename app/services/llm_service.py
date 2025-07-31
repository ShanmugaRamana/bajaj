# app/services/llm_service.py

import httpx
from typing import List
from app.core.config import settings

async def get_answer_from_llm(question: str, context: List[str]) -> str:
    """
    Gets a humanized, accurate answer from the LLM based on the provided context.
    """
    
    context_str = "\n---\n".join(context)
    
    prompt = f"""
    You are an expert assistant specializing in explaining complex policy documents in simple terms. Your goal is to answer the user's question accurately, using only the provided text, but in a clear, natural, and human-friendly way.

    **Core Task:**
    Answer the user's question based *only* on the context snippets below.

    **Style Guidelines:**
    1.  **Humanize the Response:** Write in a conversational and helpful tone. Synthesize the information into smooth, easy-to-read paragraphs.
    2.  **Avoid Raw References:** Do not include raw section numbers like 'Section 3.1.7'. Instead, refer to the rules conversationally.
    3.  **Response Type:** if the question is yes/no, provide a simple "Yes" or "No" along with a brief explanation.
    4.  **Write Out Words:** Do not use slashes ('/'). Instead, write "and" or "or" for a more natural flow (e.g., use "deliveries or terminations" instead of "deliveries/terminations").
    5.  **Be Comprehensive but Clear:** Accurately include all critical details like waiting periods, monetary limits, and specific conditions.
    6.  **More Direct, accurate and shorter:** provide a direct answer to the question, avoiding unnecessary verbosity. Use simple language and short sentences where possible.
    7.  **No additional information:** just respond to the question exactly as asked, without adding any extra information.
    8.  **Handle Missing Information:** If the answer is genuinely not in the provided text, simply state that the information isn't available in the document.
    
    **CONTEXT:**
    ---
    {context_str}
    ---

    **QUESTION:** {question}

    **ANSWER:**
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
            answer = data['choices'][0]['message']['content'].strip().replace('\n', ' ')
            return answer

    except httpx.HTTPStatusError as e:
        print(f"LLM API Error: {e.response.status_code} - {e.response.text}")
        return "Error: Could not get a response from the language model."
    except Exception as e:
        print(f"An unexpected error occurred in LLM service: {e}")
        return "Error: An unexpected error occurred while processing the request."