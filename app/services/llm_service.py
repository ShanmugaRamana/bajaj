# app/services/llm_service.py
import google.generativeai as genai
from typing import List
from app.core.config import settings

# Configure the Google AI client
genai.configure(api_key=settings.GOOGLE_API_KEY)

async def get_answer_from_llm(question: str, context: List[str]) -> str:
    """
    Gets a humanized, accurate answer from the Google Gemini API.
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
    4.  **Write Out Words:** Do not use slashes ('/'). Instead, write "and" or "or" for a more natural flow.
    5.  **Be Comprehensive but Clear:** Accurately include all critical details like waiting periods, monetary limits, and specific conditions.
    6.  **More Direct, accurate and shorter:** provide a direct answer to the question, avoiding unnecessary verbosity.
    7.  **No additional information:** just respond to the question exactly as asked.
    8.  **Handle Missing Information:** If the answer is genuinely not in the provided text, simply state that the information isn't available in the document.
    
    **CONTEXT:**
    ---
    {context_str}
    ---

    **QUESTION:** {question}

    **ANSWER:**
    """
    
    try:
        # Instantiate the Gemini model
        model = genai.GenerativeModel(settings.LLM_MODEL)
        
        # Generate the content asynchronously
        response = await model.generate_content_async(prompt)
        
        # The Google library uses response.text to get the answer
        return response.text.strip().replace('\n', ' ')


    except Exception as e:
        print(f"An unexpected error occurred in LLM service: {e}")
        return "Error: Could not get a response from the language model."