from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.generation.rag_pipeline import RagPipeline
from src.shared.models import BenchmarkCase, BenchmarkResult

ABSTENTION = "I could not find this information in the provided documents."


def _normalized(text: str) -> str:
    return " ".join(text.strip().lower().split())


class BenchmarkRunner:
    def __init__(self, pipeline: RagPipeline) -> None:
        self.pipeline = pipeline

    def run(self, cases_path: Path) -> list[BenchmarkResult]:
        cases = [BenchmarkCase.model_validate(item) for item in json.loads(cases_path.read_text())]
        results: list[BenchmarkResult] = []
        for case in cases:
            response = self.pipeline.ask(case.question)
            retrieved_sources = {chunk.source_name for chunk in response.retrieval.chunks}
            expected_sources = set(case.expected_sources or ([case.expected_source] if case.expected_source else []))
            source_retrieved = expected_sources.issubset(retrieved_sources) if expected_sources else None
            answer_match = (
                all(term.lower() in response.answer.lower() for term in case.expected_answer_contains)
                if case.expected_answer_contains
                else None
            )
            abstained = _normalized(response.answer) == _normalized(ABSTENTION)
            contains_abstention_phrase = _normalized(ABSTENTION) in _normalized(response.answer)
            citation_correct = expected_sources.issubset(set(response.citations)) if expected_sources else None

            checks = [value for value in (source_retrieved, answer_match, citation_correct) if value is not None]
            if case.must_abstain:
                checks.append(abstained)
            else:
                checks.append(not contains_abstention_phrase)

            passed = all(checks)
            failure_parts: list[str] = []
            if source_retrieved is False:
                failure_parts.append("expected source was not retrieved")
            if answer_match is False:
                failure_parts.append("expected fact was missing from the answer")
            if citation_correct is False:
                failure_parts.append("expected source was not cited")
            if case.must_abstain and not abstained:
                failure_parts.append("unsupported question did not return the exact abstention")
            if not case.must_abstain and abstained:
                failure_parts.append("model abstained despite an expected answer")
            elif not case.must_abstain and contains_abstention_phrase:
                failure_parts.append("answer mixed abstention language with a claimed answer")
            failure_reason = None if passed else "; ".join(failure_parts)

            results.append(
                BenchmarkResult(
                    test_case_id=case.id,
                    category=case.category,
                    question=case.question,
                    actual_answer=response.answer,
                    expected_source_retrieved=source_retrieved,
                    answer_contains_expected_text=answer_match,
                    citations_present=bool(response.citations),
                    correct_source_cited=citation_correct,
                    abstention_expected=case.must_abstain,
                    abstention_observed=abstained,
                    retrieved_chunk_ids=[chunk.chunk_id for chunk in response.retrieval.chunks],
                    latency_ms=response.latency.total_ms,
                    passed=passed,
                    failure_reason=failure_reason,
                )
            )
        return results

    @staticmethod
    def export(
        results: list[BenchmarkResult],
        artifacts_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not results:
            raise ValueError("Cannot export an empty benchmark result set.")

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        records = [result.model_dump() for result in results]
        (artifacts_dir / "benchmark_results.json").write_text(json.dumps(records, indent=2))

        with (artifacts_dir / "benchmark_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        failures = [result for result in results if not result.passed]
        lines = ["# Baseline RAG Failure Analysis", ""]
        for result in failures:
            lines.extend(
                [
                    f"## {result.category}: {result.test_case_id}",
                    "",
                    f"**Question:** {result.question}",
                    "",
                    f"**Observed answer:** {result.actual_answer}",
                    "",
                    f"**Failure reason:** {result.failure_reason}",
                    "",
                    f"**Retrieved chunks:** {', '.join(result.retrieved_chunk_ids)}",
                    "",
                    "**Likely cause:** Baseline fixed chunking or dense-only retrieval did not provide sufficient evidence, or generation did not follow the grounding contract.",
                    "",
                    "**Proposed improvement:** Add query expansion, hybrid retrieval, reranking, metadata filters, multi-step retrieval, or stricter grounded-generation controls.",
                    "",
                ]
            )
        (artifacts_dir / "failure_analysis.md").write_text("\n".join(lines))

        if metadata is not None:
            (artifacts_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
