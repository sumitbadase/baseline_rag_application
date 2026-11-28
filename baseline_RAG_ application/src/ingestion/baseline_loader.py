from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from src.chunking.fixed_chunker import FixedSizeChunker
from src.shared.models import DocumentChunk, IngestionResult


class BaselineLoader:
    def __init__(self, chunker: FixedSizeChunker) -> None:
        self.chunker = chunker

    def load_file(self, path: Path) -> list[DocumentChunk]:
        document_id = path.stem.replace(" ", "_").lower()
        if path.suffix.lower() == ".pdf":
            chunks: list[DocumentChunk] = []
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                chunks.extend(self.chunker.chunk(document_id, path.name, page.extract_text() or "", page_number))
            return chunks
        if path.suffix.lower() in {".txt", ".md"}:
            return self.chunker.chunk(document_id, path.name, path.read_text(encoding="utf-8"))
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def load_directory(self, directory: Path) -> tuple[list[DocumentChunk], IngestionResult]:
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".txt", ".md", ".pdf"})
        chunks = [chunk for path in files for chunk in self.load_file(path)]
        return chunks, IngestionResult(document_count=len(files), chunk_count=len(chunks), sources=[p.name for p in files])
