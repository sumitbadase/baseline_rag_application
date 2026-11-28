from src.evaluation.comparison import comparison_table, summarize_results


def sample_result(case_id: str, passed: bool, latency_ms: float) -> dict:
    return {
        "test_case_id": case_id,
        "category": "successful",
        "question": "Example?",
        "actual_answer": "Example answer",
        "expected_source_retrieved": True,
        "answer_contains_expected_text": passed,
        "citations_present": True,
        "correct_source_cited": True,
        "abstention_expected": False,
        "abstention_observed": False,
        "retrieved_chunk_ids": ["chunk-1"],
        "latency_ms": latency_ms,
        "passed": passed,
        "failure_reason": None if passed else "failed",
    }


def test_summarize_results_keeps_only_key_metrics():
    summary = summarize_results(
        [sample_result("one", True, 10.0), sample_result("two", False, 30.0)]
    )

    assert summary["case_count"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["average_latency_ms"] == 20.0
    assert summary["source_retrieval_rate"] == 1.0
    assert summary["abstention_accuracy"] == 1.0


def test_comparison_table_handles_missing_ollama_run():
    table = comparison_table([sample_result("one", True, 10.0)], [])

    assert list(table.columns) == [
        "Case",
        "Category",
        "Deterministic",
        "Ollama",
        "Deterministic latency (ms)",
        "Ollama latency (ms)",
    ]
    assert table.iloc[0]["Deterministic"] == "Pass"
    assert table.iloc[0]["Ollama"] == "Not run"
