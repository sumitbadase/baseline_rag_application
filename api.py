from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.chunking.fixed_chunker import FixedSizeChunker
from src.generation.fixture_provider import FixtureProvider
from src.generation.ollama_provider import OllamaProvider
from src.generation.rag_pipeline import RagPipeline
from src.ingestion.baseline_loader import BaselineLoader
from src.retrieval.embeddings import HashEmbeddingProvider, SentenceTransformerEmbeddingProvider
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.vector_store import QdrantVectorStore
from src.shared.models import RagResponse
from src.shared.settings import settings


class QueryRequest(BaseModel):
    question: str


def build_services(
    generation_backend: str | None = None,
    embedding_backend: str | None = None,
):
    selected_generation = generation_backend or settings.generation_backend
    selected_embedding = embedding_backend or settings.embedding_backend

    if selected_generation not in {"fixture", "ollama"}:
        raise ValueError(f"Unsupported generation backend: {selected_generation}")
    if selected_embedding not in {"hash", "sentence_transformers"}:
        raise ValueError(f"Unsupported embedding backend: {selected_embedding}")

    embedder = (
        SentenceTransformerEmbeddingProvider(settings.embedding_model)
        if selected_embedding == "sentence_transformers"
        else HashEmbeddingProvider()
    )
    store = QdrantVectorStore(settings.collection_name, embedder.dimension)
    loader = BaselineLoader(FixedSizeChunker(settings.chunk_size, settings.chunk_overlap))
    chunks, ingestion_result = loader.load_directory(settings.data_dir)
    store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    generator = (
        OllamaProvider(settings.ollama_base_url, settings.ollama_model)
        if selected_generation == "ollama"
        else FixtureProvider()
    )
    return RagPipeline(VectorRetriever(embedder, store), generator, settings.top_k), ingestion_result


app = FastAPI(title="Baseline RAG API")
pipeline, ingestion_result = build_services()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": ingestion_result.document_count,
        "chunks": ingestion_result.chunk_count,
        "generation_backend": settings.generation_backend,
        "embedding_backend": settings.embedding_backend,
    }


@app.post("/query", response_model=RagResponse)
def query(request: QueryRequest) -> RagResponse:
    return pipeline.ask(request.question)
