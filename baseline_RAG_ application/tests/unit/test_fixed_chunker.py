from src.chunking.fixed_chunker import FixedSizeChunker


def test_fixed_chunker_creates_overlapping_chunks():
    chunker = FixedSizeChunker(chunk_size=20, overlap=5)
    chunks = chunker.chunk("doc", "doc.txt", "abcdefghijklmnopqrstuvwxyz0123456789")
    assert len(chunks) >= 2
    assert chunks[0].metadata["end_char"] - chunks[1].metadata["start_char"] == 5
    assert all(chunk.text for chunk in chunks)
