import uuid
from typing import Dict, Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from .config import settings
from .confluence import fetch_pages_in_space, page_to_doc_parts, table_to_text
from .qdrant_store import ensure_collection, upsert
 
 
def _embedder() -> BedrockEmbeddings:
    return BedrockEmbeddings(model_id=settings.BEDROCK_EMBED_MODEL_ID, region_name=settings.AWS_REGION)
 
 
def sync_confluence(limit: int = 25) -> Dict[str, int]:
    pages = fetch_pages_in_space(limit=limit)
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
 
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        meta, text, tables = page_to_doc_parts(page)
 
        for i, ch in enumerate(splitter.split_text(text)):
            chunks.append({"chunk_id": str(uuid.uuid4()), "text": ch, "payload": meta | {"chunk_type": "text", "chunk_index": i}})
 
        for t in tables:
            chunks.append({"chunk_id": str(uuid.uuid4()), "text": table_to_text(t), "payload": meta | {"chunk_type": "table", "table_index": t["table_index"]}})
 
    if not chunks:
        return {"pages": len(pages), "chunks_ingested": 0}
 
    emb = _embedder()
    vectors = emb.embed_documents([c["text"] for c in chunks])
 
    ensure_collection(vector_size=len(vectors[0]))
 
    points = []
    for c, vec in zip(chunks, vectors):
        points.append({"id": c["chunk_id"], "vector": vec, "payload": c["payload"] | {"chunk_id": c["chunk_id"], "text": c["text"]}})
 
    upsert(points)
    return {"pages": len(pages), "chunks_ingested": len(points)}