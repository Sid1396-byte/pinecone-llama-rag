import io
import re
from pypdf import PdfReader
from config import settings

def clean_extracted_text(text: str) -> str:
    """Cleans up raw extracted text from PDFs to remove weird symbols and spacing issues."""
    # Replace multiple consecutive spaces or tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove weird control characters/symbols but keep normal punctuation and alphanumerics
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    # Normalize multiple newlines down to a double newline maximum
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Reads bytes from a PDF and extracts the text payload safely."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return clean_extracted_text(text)

def chunk_text(text: str) -> list[str]:
    """Splits a large block of text into smaller, overlapping chunks."""
    chunks = []
    start = 0
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    
    while start < len(text):
        chunks.append(text[start:start + size])
        start += (size - overlap)
        
    return chunks