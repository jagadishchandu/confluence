from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from .config import settings
 
 
def _client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL)
 
 
def ensure_collection(vector_size: int):
    c = _client()
    names = [col.name for col in c.get_collections().collections]
    if settings.QDRANT_COLLECTION in names:
        return
    c.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )
 
 
def upsert(points: List[Dict[str, Any]]):
    c = _client()
    c.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[qm.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points],
    )
 
 
def search(query_vector: List[float], k: int = 6, filters: Optional[Dict[str, Any]] = None):
    c = _client()
    qfilter = None
    if filters:
        must = [qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in filters.items()]
        qfilter = qm.Filter(must=must)
 
    return c.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=k,
        query_filter=qfilter,
        with_payload=True,
    )