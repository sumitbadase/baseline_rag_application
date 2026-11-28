from __future__ import annotations

from typing import Protocol

from src.shared.models import DocumentChunk, RetrievedChunk


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]: ...


class GenerationProvider(Protocol):
    def generate(self, prompt: str) -> str: ...
