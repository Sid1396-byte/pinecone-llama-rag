from google import genai
from config import settings

client = genai.Client()

def generate_answer(query: str, context: str) -> str:
    """Passes the retrieved context and user query to Gemini to generate an answer."""
    prompt = f"""
    You are an intelligent AI assistant. Use the provided context to answer the user's question.
    If the answer is not contained within the context, state that you do not have enough information.

    Context:
    {context}

    Question:
    {query}
    """
    
    response = client.models.generate_content(
        model=settings.LLM_MODEL,
        contents=prompt
    )
    return response.text