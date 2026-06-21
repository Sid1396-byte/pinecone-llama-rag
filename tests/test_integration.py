import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Import the FastAPI app from your main.py file
from main import app

client = TestClient(app)

# ==========================================
# UPLOAD ENDPOINT TESTS
# ==========================================

def test_upload_invalid_file_type():
    """Ensures the API rejects non-PDF/TXT files."""
    response = client.post(
        "/upload", 
        files={"file": ("image.png", b"fake image data", "image/png")}, 
        data={"session_id": "test_session_123"}
    )
    
    assert response.status_code == 400
    assert "Only .pdf and .txt files are supported" in response.json()["detail"]

@patch("main.upsert_chunks") # Intercepts the Pinecone call
@patch("main.upload_to_s3")  # Intercepts the AWS S3 call
def test_upload_valid_txt_file(mock_s3, mock_upsert):
    """Tests a successful text file upload by mocking external APIs."""
    
    # Tell the mock functions to return None (simulate success without actually running)
    mock_s3.return_value = None
    mock_upsert.return_value = None

    file_data = b"This is a test document with some RAG text."
    
    response = client.post(
        "/upload",
        files={"file": ("test.txt", file_data, "text/plain")},
        data={"session_id": "test_session_123"}
    )
    
    # Verify the API responds with a success message
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verify that the mocked external functions were actually triggered
    mock_s3.assert_called_once()
    mock_upsert.assert_called_once()

# ==========================================
# QUERY ENDPOINT TESTS
# ==========================================

@patch("main.generate_answer")       # Intercepts the Gemini call
@patch("main.query_similar_chunks")  # Intercepts the Pinecone call
def test_query_endpoint_success(mock_query, mock_generate):
    """Tests a successful Q&A interaction."""
    
    # Fake the data that Pinecone and Gemini WOULD have returned
    mock_query.return_value = ["This is a fake chunk of context from the database."]
    mock_generate.return_value = "This is the fake AI generated answer."

    response = client.post(
        "/query",
        json={
            "query": "What is the meaning of life?", 
            "session_id": "test_session_123"
        }
    )

    assert response.status_code == 200
    
    data = response.json()
    assert data["answer"] == "This is the fake AI generated answer."
    assert "This is a fake chunk of context from the database." in data["context_used"]

@patch("main.query_similar_chunks")
def test_query_endpoint_no_context(mock_query):
    """Tests behavior when Pinecone finds zero matching chunks."""
    
    # Simulate Pinecone returning an empty list
    mock_query.return_value = []

    response = client.post(
        "/query",
        json={
            "query": "Who is the president?", 
            "session_id": "test_session_123"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "No relevant information found" in data["answer"]
    assert len(data["context_used"]) == 0