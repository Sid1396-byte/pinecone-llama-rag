from google import genai
from google.genai import types
from config import settings

def generate_answer(question: str, context: str) -> str:
    """Uses Gemini 2.5 Flash to synthesize an accurate answer from retrieved vectors."""
    if not settings.GOOGLE_API_KEY:
        return "System Error: Google API Key missing from environment configurations."
        
    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    
    system_instruction = (
        "You are an expert production AI assistant. Answer the user's question using ONLY the provided text "
        "snippets. If the context does not contain the answer, explicitly state that you cannot find the answer "
        "in the uploaded records. Keep responses structured, professional, and formatted in clean Markdown."
    )
    
    prompt = f"Context Material:\n{context}\n\nUser Question: {question}"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2
        )
    )
    return response.text