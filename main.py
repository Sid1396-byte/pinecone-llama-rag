import os
import boto3
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List

# --- IMPORTING YOUR MODULAR SERVICES ---
from config import settings
from services.doc_parser import extract_text_from_pdf, chunk_text
from services.pinecone_db import upsert_chunks, query_similar_chunks
from services.google_ai import generate_answer

# ==========================================
# 1. INITIALIZATION & SETTINGS
# ==========================================
app = FastAPI(title="Pinecone Llama Embedding RAG API")

# S3 Backup Configuration
s3_client = boto3.client('s3')
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

# Expected query payload schema
class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    answer: str
    context_used: List[str]

# ==========================================
# 2. HELPER LOGIC
# ==========================================
def upload_to_s3(file_bytes: bytes, filename: str, session_id: str):
    """Backs up the raw uploaded file to AWS S3, organized by session."""
    if not S3_BUCKET_NAME:
        return
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=f"uploaded-documents/{session_id}/{filename}",
        Body=file_bytes
    )

# ==========================================
# 3. API ENDPOINTS
# ==========================================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = Form(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")
        
    contents = await file.read()
    
    # 1. Archive to S3
    try:
        upload_to_s3(contents, file.filename, session_id)
    except Exception as e:
        print(f"S3 Upload failed: {str(e)}") 
    
    # 2. Extract Text (Using your doc_parser service)
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(contents) 
    else:
        text = contents.decode('utf-8', errors='ignore')
        
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded document contains no text.")
        
    # 3. Chunk Text (Using your doc_parser service)
    chunks = chunk_text(text)
    
    # 4. Generate Embeddings & Upsert (Using your pinecone_db service)
    try:
        upsert_chunks(chunks=chunks, namespace=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone Processing error: {str(e)}")
        
    return {
        "status": "success", 
        "message": f"Processed '{file.filename}' into {len(chunks)} chunks & secured in your private session."
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # 1. Fetch relevant chunks strictly within user's namespace (Using your pinecone_db service)
        relevant_chunks = query_similar_chunks(
            query=request.query, 
            namespace=request.session_id, 
            top_k=3
        )
        
        if not relevant_chunks:
            return QueryResponse(answer="No relevant information found in your uploaded documents.", context_used=[])
            
        # 2. Format context block
        context_block = "\n---\n".join(relevant_chunks)
        
        # 3. Generate answer using Gemini (Using your google_ai service)
        answer = generate_answer(question=request.query, context=context_block)
        
        return QueryResponse(answer=answer, context_used=relevant_chunks)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")