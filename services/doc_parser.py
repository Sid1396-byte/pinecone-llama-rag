import io
import re
from pypdf import PdfReader
from config import settings

def clean_extracted_text(text: str) -> str:
    """Removes non-printable characters, weird spacing, and ligatures from PDFs."""
    # 1. Remove null/weird bytes FIRST
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    # 2. THEN collapse the spaces and tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # 3. Finally, format the newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Safely extracts full text context from a raw binary PDF stream."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return clean_extracted_text(text)

def chunk_text(text: str) -> list[str]:
    """Splits text documents into dense chunks matching global size strategies."""
    chunks = []
    start = 0
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    
    while start < len(text):
        chunks.append(text[start:start + size])
        start += (size - overlap)
        
    return chunks