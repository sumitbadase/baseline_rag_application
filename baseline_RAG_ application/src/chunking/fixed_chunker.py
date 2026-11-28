from __future__ import annotations

from src.shared.models import DocumentChunk


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 80) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document_id: str, source_name: str, text: str, page_number: int | None = None) -> list[DocumentChunk]:
        normalized = " ".join(text.split())
        if not normalized:
            return []
        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            chunk_text = normalized[start:end]
            chunks.append(DocumentChunk(
                chunk_id=f"{document_id}-{page_number or 0}-{index:03d}",
                document_id=document_id,
                source_name=source_name,
                page_number=page_number,
                text=chunk_text,
                metadata={"start_char": start, "end_char": end},
            ))
            if end == len(normalized):
                break
            start = end - self.overlap
            index += 1
        return chunks
