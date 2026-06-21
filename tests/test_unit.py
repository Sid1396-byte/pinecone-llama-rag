import pytest
from unittest.mock import patch
from services.doc_parser import chunk_text, clean_extracted_text

def test_clean_extracted_text():
    """Tests if the regex cleaner removes weird spacing and null characters."""
    dirty_text = "This   has \t weird \n\n\n spacing and \x00 null bytes."
    cleaned = clean_extracted_text(dirty_text)
    
    assert "  " not in cleaned # Multiple spaces should be gone
    assert "\x00" not in cleaned # Null bytes should be removed
    assert "\n\n\n" not in cleaned # Triple newlines reduced to double
    assert cleaned == "This has weird \n\n spacing and null bytes."

@patch("services.doc_parser.settings")
def test_chunk_text(mock_settings):
    """Tests if text is split into the correct sizes with the correct overlap."""
    # Force small chunk sizes specifically for this test
    mock_settings.CHUNK_SIZE = 10
    mock_settings.CHUNK_OVERLAP = 2
    
    text = "ABCDEFGHIJKLMNO" # Length 15
    
    # Expected behavior with chunk=10, overlap=2 (stride=8):
    # Chunk 1: indices 0-10 ("ABCDEFGHIJ")
    # Chunk 2: indices 8-15 ("IJKLMNO")
    
    chunks = chunk_text(text)
    
    assert len(chunks) == 2
    assert chunks[0] == "ABCDEFGHIJ"
    assert chunks[1] == "IJKLMNO"