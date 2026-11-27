from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data" / "sample_docs"
    artifacts_dir: Path = project_root / "artifacts"
    collection_name: str = "baseline_rag"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "hash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    generation_backend: str = "fixture"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 3


settings = Settings()
