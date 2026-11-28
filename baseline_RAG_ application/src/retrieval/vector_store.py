from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.shared.models import DocumentChunk, RetrievedChunk


class QdrantVectorStore:
    def __init__(self, collection_name: str, dimension: int, path: str = ":memory:") -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(location=path)
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(id=index, vector=vector, payload=chunk.model_dump())
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points
        chunks: list[RetrievedChunk] = []
        for rank, point in enumerate(results, start=1):
            payload = dict(point.payload or {})
            chunks.append(RetrievedChunk(**payload, score=float(point.score), rank=rank))
        return chunks
