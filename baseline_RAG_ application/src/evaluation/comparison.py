from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_FIELDS = (
    "Pass rate",
    "Average latency",
    "Source retrieval",
    "Abstention accuracy",
)


def load_benchmark_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_run_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not results:
        return {
            "case_count": 0,
            "pass_rate": None,
            "average_latency_ms": None,
            "source_retrieval_rate": None,
            "abstention_accuracy": None,
        }

    source_checks = [
        bool(row["expected_source_retrieved"])
        for row in results
        if row.get("expected_source_retrieved") is not None
    ]
    abstention_checks = [
        bool(row.get("abstention_observed")) == bool(row.get("abstention_expected"))
        for row in results
    ]
    return {
        "case_count": len(results),
        "pass_rate": _rate([bool(row.get("passed")) for row in results]),
        "average_latency_ms": sum(float(row.get("latency_ms", 0.0)) for row in results) / len(results),
        "source_retrieval_rate": _rate(source_checks),
        "abstention_accuracy": _rate(abstention_checks),
    }


def comparison_table(
    deterministic: list[dict[str, Any]],
    ollama: list[dict[str, Any]],
) -> pd.DataFrame:
    deterministic_by_id = {row["test_case_id"]: row for row in deterministic}
    ollama_by_id = {row["test_case_id"]: row for row in ollama}
    case_ids = list(dict.fromkeys([*deterministic_by_id, *ollama_by_id]))

    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        det = deterministic_by_id.get(case_id, {})
        live = ollama_by_id.get(case_id, {})
        rows.append(
            {
                "Case": case_id,
                "Category": det.get("category") or live.get("category"),
                "Deterministic": _status(det),
                "Ollama": _status(live),
                "Deterministic latency (ms)": _rounded_latency(det),
                "Ollama latency (ms)": _rounded_latency(live),
            }
        )
    return pd.DataFrame(rows)


def answer_comparison_table(
    deterministic: list[dict[str, Any]],
    ollama: list[dict[str, Any]],
) -> pd.DataFrame:
    deterministic_by_id = {row["test_case_id"]: row for row in deterministic}
    ollama_by_id = {row["test_case_id"]: row for row in ollama}
    case_ids = list(dict.fromkeys([*deterministic_by_id, *ollama_by_id]))

    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        det = deterministic_by_id.get(case_id, {})
        live = ollama_by_id.get(case_id, {})
        rows.append(
            {
                "Case": case_id,
                "Question": det.get("question") or live.get("question"),
                "Deterministic answer": det.get("actual_answer", "Not run"),
                "Ollama answer": live.get("actual_answer", "Not run"),
            }
        )
    return pd.DataFrame(rows)


def _status(row: dict[str, Any]) -> str:
    if not row:
        return "Not run"
    return "Pass" if row.get("passed") else "Fail"


def _rounded_latency(row: dict[str, Any]) -> float | None:
    if not row:
        return None
    return round(float(row.get("latency_ms", 0.0)), 1)
