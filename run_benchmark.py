from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from api import build_services
from src.evaluation.benchmark import BenchmarkRunner
from src.shared.settings import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Baseline RAG benchmark without overwriting results from another provider."
    )
    parser.add_argument(
        "--provider",
        choices=("fixture", "ollama"),
        default="fixture",
        help="Generation provider. Fixture is deterministic; Ollama evaluates the live local model.",
    )
    parser.add_argument(
        "--embedding",
        choices=("hash", "sentence_transformers"),
        default=None,
        help="Embedding backend. Defaults to EMBEDDING_BACKEND from .env, or hash.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=settings.project_root / "tests" / "fixtures" / "benchmark_cases.json",
        help="Path to the benchmark case JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output directory. By default, fixture and Ollama use separate directories.",
    )
    return parser.parse_args()


def default_output_dir(provider: str) -> Path:
    run_name = "deterministic" if provider == "fixture" else "ollama"
    return settings.artifacts_dir / run_name


def main() -> None:
    args = parse_args()
    embedding_backend = args.embedding or settings.embedding_backend
    output_dir = args.output_dir or default_output_dir(args.provider)

    pipeline, ingestion_result = build_services(
        generation_backend=args.provider,
        embedding_backend=embedding_backend,
    )
    runner = BenchmarkRunner(pipeline)
    results = runner.run(args.cases)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "generation_model": settings.ollama_model if args.provider == "ollama" else "fixture-provider",
        "embedding_backend": embedding_backend,
        "embedding_model": settings.embedding_model if embedding_backend == "sentence_transformers" else "deterministic-hash",
        "ollama_base_url": settings.ollama_base_url if args.provider == "ollama" else None,
        "top_k": settings.top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "document_count": ingestion_result.document_count,
        "chunk_count": ingestion_result.chunk_count,
        "cases_path": str(args.cases.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    runner.export(results, output_dir, metadata=metadata)

    for result in results:
        print(f"{result.test_case_id}: {'PASS' if result.passed else 'FAIL'}")
    print(f"Artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
