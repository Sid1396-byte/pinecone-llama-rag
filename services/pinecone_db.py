import uuid
from pinecone import Pinecone
from config import settings

# Initialize Pinecone Client
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# Verify index presence; if missing, build it utilizing Llama Integrated Inference definitions
if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    from pinecone import ServerlessSpec
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=1024, # Llama text-embed-v2 uses 1024 dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(settings.PINECONE_INDEX_NAME)

def upsert_chunks(chunks: list[str]):
    """Embeds text chunks via Llama Inference and registers vectors to Pinecone."""
    if not chunks:
        return
        
    # Generate integrated Llama embeddings directly through Pinecone
    embeddings_response = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=chunks,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    
    records = []
    for idx, chunk in enumerate(chunks):
        records.append({
            "id": str(uuid.uuid4()),
            "values": embeddings_response.data[idx].values,
            "metadata": {"text": chunk}
        })
        
    # Batch upsert elements into vector database index
    index.upsert(vectors=records)

def query_similar_chunks(query: str, top_k: int = 3) -> list[str]:
    """Converts user queries to embeddings and fetches the top related text records."""
    query_embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"}
    )
    
    results = index.query(
        vector=query_embedding.data[0].values,
        top_k=top_k,
        include_metadata=True
    )
    
    return [match.metadata["text"] for match in results.matches if match.metadata and "text" in match.metadata]