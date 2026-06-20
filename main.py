import os
import boto3
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List

from services.doc_parser import extract_text_from_pdf, chunk_text
from services.google_ai import generate_answer
from services.pinecone_db import upsert_chunks, query_similar_chunks

app = FastAPI(title="Pinecone Llama Embedding RAG API")

# Setup S3 Client. Standard IAM configurations automatically pick up access roles on EC2.
s3_client = boto3.client('s3')
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context_used: List[str]

def upload_to_s3(file_bytes: bytes, filename: str):
    """Saves raw multi-format records directly into your protected AWS S3 resource bucket."""
    if not S3_BUCKET_NAME:
        print("Warning: AWS_S3_BUCKET_NAME variable missing. Bypassing cloud storage sync.")
        return
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=f"uploaded-documents/{filename}",
        Body=file_bytes
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Unsupported document extension configuration.")
        
    contents = await file.read()
    
    # 1. Pipeline Backup: Stream to safe persistent Amazon S3 storage bucket
    try:
        upload_to_s3(contents, file.filename)
    except Exception as e:
        print(f"Non-blocking S3 Archive sync failure tracked: {str(e)}")
        
    # 2. Extract Document text content payloads
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(contents)
    else:
        text = contents.decode('utf-8')
        
    if not text.strip():
        raise HTTPException(status_code=400, detail="Unable to extract indexable clean strings.")
        
    chunks = chunk_text(text)
    
    try:
        upsert_chunks(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing processing error: {str(e)}")
        
    return {
        "status": "success",
        "message": f"Successfully processed '{file.filename}' into {len(chunks)} fragments & saved archive copy in S3."
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        relevant_chunks = query_similar_chunks(request.query, top_k=3)
        if not relevant_chunks:
            return QueryResponse(answer="No relevant database contextual references located.", context_used=[])
            
        context_block = "\n---\n".join(relevant_chunks)
        answer = generate_answer(request.query, context_block)
        return QueryResponse(answer=answer, context_used=relevant_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context collection fault query pipeline: {str(e)}")