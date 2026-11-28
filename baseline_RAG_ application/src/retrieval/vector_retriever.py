from __future__ import annotations

from src.shared.models import RetrievedChunk
from src.shared.protocols import EmbeddingProvider, VectorStore


class VectorRetriever:
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> tuple[list[RetrievedChunk], float, float]:
        from src.shared.timing import measure_ms

        query_vector, embedding_ms = measure_ms(lambda: self.embedder.embed_query(query))
        chunks, retrieval_ms = measure_ms(lambda: self.vector_store.search(query_vector, top_k))
        return chunks, embedding_ms, retrieval_ms
