from pathlib import Path

from src.evaluation.benchmark import BenchmarkRunner


def test_baseline_acceptance(pipeline, tmp_path: Path):
    response = pipeline.ask("How many annual leave days do employees receive?")
    assert response.citations
    assert response.retrieval.chunks
    assert response.latency.total_ms > 0

    runner = BenchmarkRunner(pipeline)
    cases = Path(__file__).parents[1] / "fixtures" / "benchmark_cases.json"
    results = runner.run(cases)
    runner.export(
        results,
        tmp_path,
        metadata={"provider": "fixture", "embedding_backend": "hash"},
    )

    assert len(results) == 5
    assert len({result.category for result in results}) >= 4
    assert (tmp_path / "benchmark_results.json").exists()
    assert (tmp_path / "benchmark_results.csv").exists()
    assert (tmp_path / "failure_analysis.md").exists()
    assert (tmp_path / "run_metadata.json").exists()
