def test_retrieval_trace_and_latency_are_exposed(pipeline):
    response = pipeline.ask("How many annual leave days do employees receive?")
    assert response.answer
    assert response.citations
    assert response.retrieval.chunks
    first = response.retrieval.chunks[0]
    assert first.text
    assert first.source_name
    assert first.rank == 1
    assert 0 <= first.score <= 1
    assert response.latency.total_ms > 0
    assert response.latency.embedding_ms >= 0
    assert response.latency.retrieval_ms >= 0
    assert response.latency.prompt_construction_ms >= 0
    assert response.latency.generation_ms >= 0
