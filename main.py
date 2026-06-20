import os
import io
import boto3
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List
import pypdf
from google import genai
from pinecone import Pinecone

# ==========================================
# 1. INITIALIZATION & SETTINGS
# ==========================================
app = FastAPI(title="Pinecone Llama Embedding RAG API")

# S3 Backup Configuration
s3_client = boto3.client('s3')
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

# API Keys and Constants
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pinecone-llama-rag")

# Initialize Vector DB & AI Clients
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
ai_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

# Expected query payload schema
class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    answer: str
    context_used: List[str]

# ==========================================
# 2. CORE PROCESSING LOGIC
# ==========================================

def extract_text_from_pdf(contents: bytes) -> str:
    text = ""
    try:
        pdf_file = io.BytesIO(contents)
        reader = pypdf.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"PDF extraction error: {str(e)}")
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    chunks = []
    stride = chunk_size - chunk_overlap
    for i in range(0, len(text), stride if stride > 0 else chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def upload_to_s3(file_bytes: bytes, filename: str, session_id: str):
    if not S3_BUCKET_NAME:
        return
    # Save the file with the session ID in the path to keep S3 organized
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
    
    if not pc or not ai_client:
        raise HTTPException(status_code=500, detail="API credentials missing.")
        
    contents = await file.read()
    
    # 1. Archive to S3 (isolated by session_id)
    try:
        upload_to_s3(contents, file.filename, session_id)
    except Exception as e:
        print(f"S3 Upload failed: {str(e)}") 
    
    # 2. Extract Text
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(contents)
    else:
        text = contents.decode('utf-8', errors='ignore')
        
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded document contains no text.")
        
    # 3. Chunk Text
    chunks = chunk_text(text)
    
    # 4. Generate Embeddings and Upsert with Namespace
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        
        emb_response = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=chunks,
            parameters={"input_type": "passage"}
        )
        
        vectors = []
        for idx, (chunk, emb) in enumerate(zip(chunks, emb_response)):
            vectors.append({
                "id": f"{file.filename}-chunk-{idx}",
                "values": emb['values'],
                "metadata": {"text": chunk}
            })
            
        # THE FIX: Isolate this data into the user's specific namespace
        index.upsert(vectors=vectors, namespace=session_id) 
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone Processing error: {str(e)}")
        
    return {
        "status": "success", 
        "message": f"Processed '{file.filename}' into {len(chunks)} chunks & secured in your private session."
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not pc or not ai_client:
        raise HTTPException(status_code=500, detail="API credentials missing.")
        
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        
        query_emb = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[request.query],
            parameters={"input_type": "query"}
        )
        
        # THE FIX: Only search inside this specific user's namespace
        search_results = index.query(
            vector=query_emb[0]['values'],
            top_k=3,
            include_metadata=True,
            namespace=request.session_id 
        )
        
        relevant_chunks = [match['metadata']['text'] for match in search_results['matches'] if 'metadata' in match]
        
        if not relevant_chunks:
            return QueryResponse(answer="No relevant information found in your uploaded documents.", context_used=[])
            
        context_block = "\n---\n".join(relevant_chunks)
        prompt = f"Answer the query based strictly on the provided context.\nQuery: {request.query}\nContext:\n{context_block}"
        
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        return QueryResponse(answer=response.text, context_used=relevant_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")