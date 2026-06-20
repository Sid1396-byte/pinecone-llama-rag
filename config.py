import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pinecone-llama-rag")
    
    # LLM Settings
    LLM_MODEL = "gemini-3.1-flash-lite"
    
    # Pinecone Inference Settings
    PINECONE_EMBEDDING_MODEL = "llama-text-embed-v2"
    EMBEDDING_DIMENSION = 1024 
    
    # Chunking strategy
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

settings = Settings()