import uuid
from pinecone import Pinecone, ServerlessSpec
from config import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# Initialize the index with 1024 dimensions
if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=settings.EMBEDDING_DIMENSION, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(settings.PINECONE_INDEX_NAME)

def get_pinecone_embeddings(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Generates embeddings using Pinecone's Integrated Inference API."""
    response = pc.inference.embed(
        model=settings.PINECONE_EMBEDDING_MODEL,
        inputs=texts,
        # 'passage' is used for documents, 'query' is used for user questions
        parameters={"input_type": input_type, "truncate": "END"} 
    )
    return [data['values'] for data in response]

def upsert_chunks(chunks: list[str]):
    """Embeds text chunks via Pinecone and uploads them."""
    batch_size = 50 
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # 1. Generate Llama embeddings remotely on Pinecone servers
        embeddings = get_pinecone_embeddings(batch_chunks, input_type="passage")
        
        # 2. Prepare the payload
        vectors = []
        for chunk, emb in zip(batch_chunks, embeddings):
            vector_id = str(uuid.uuid4())
            vectors.append({
                "id": vector_id,
                "values": emb,
                "metadata": {"text": chunk}
            })
        
        # 3. Save to database
        index.upsert(vectors=vectors)

def query_similar_chunks(query: str, top_k: int = 5) -> list[str]:
    """Embeds the query via Pinecone and searches the database."""
    # 1. Turn query into Llama embeddings using 'query' input_type
    query_embedding = get_pinecone_embeddings([query], input_type="query")[0]
    
    # 2. Search database
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match["metadata"]["text"] for match in results["matches"]]