from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from api import build_services
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.comparison import (
    answer_comparison_table,
    comparison_table,
    load_benchmark_results,
    load_run_metadata,
    summarize_results,
)
from src.shared.settings import settings

st.set_page_config(page_title="Baseline RAG Lab", layout="wide")
st.title("Baseline RAG Lab")
st.caption("Build, inspect, and compare a deterministic baseline with a live Ollama run.")

CASES_PATH = settings.project_root / "tests" / "fixtures" / "benchmark_cases.json"
DETERMINISTIC_DIR = settings.artifacts_dir / "deterministic"
OLLAMA_DIR = settings.artifacts_dir / "ollama"


@st.cache_resource
def default_services():
    return build_services()


def run_benchmark(provider: str, embedding_backend: str, output_dir: Path) -> None:
    pipeline, ingestion = build_services(
        generation_backend=provider,
        embedding_backend=embedding_backend,
    )
    runner = BenchmarkRunner(pipeline)
    results = runner.run(CASES_PATH)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "generation_model": settings.ollama_model if provider == "ollama" else "fixture-provider",
        "embedding_backend": embedding_backend,
        "embedding_model": (
            settings.embedding_model
            if embedding_backend == "sentence_transformers"
            else "deterministic-hash"
        ),
        "top_k": settings.top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "document_count": ingestion.document_count,
        "chunk_count": ingestion.chunk_count,
        "cases_path": str(CASES_PATH.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    runner.export(results, output_dir, metadata=metadata)


def percent(value: float | None) -> str:
    return "Not run" if value is None else f"{value:.0%}"


def latency(value: float | None) -> str:
    return "Not run" if value is None else f"{value:,.1f} ms"


def summary_frame(
    deterministic_summary: dict[str, float | int | None],
    ollama_summary: dict[str, float | int | None],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": "Pass rate",
                "Deterministic": percent(deterministic_summary["pass_rate"]),
                "Ollama": percent(ollama_summary["pass_rate"]),
            },
            {
                "Metric": "Average latency",
                "Deterministic": latency(deterministic_summary["average_latency_ms"]),
                "Ollama": latency(ollama_summary["average_latency_ms"]),
            },
            {
                "Metric": "Expected source retrieved",
                "Deterministic": percent(deterministic_summary["source_retrieval_rate"]),
                "Ollama": percent(ollama_summary["source_retrieval_rate"]),
            },
            {
                "Metric": "Abstention behavior",
                "Deterministic": percent(deterministic_summary["abstention_accuracy"]),
                "Ollama": percent(ollama_summary["abstention_accuracy"]),
            },
        ]
    )


pipeline, ingestion_result = default_services()
st.sidebar.success(
    f"Corpus: {ingestion_result.document_count} documents · {ingestion_result.chunk_count} chunks"
)
st.sidebar.caption(
    f"Query mode: {settings.generation_backend} generator · {settings.embedding_backend} embeddings"
)

query_tab, comparison_tab = st.tabs(["Ask a Question", "Benchmark Comparison"])

with query_tab:
    question = st.text_input(
        "Question",
        "How many annual leave days do employees receive?",
    )

    if st.button("Run query", type="primary"):
        response = pipeline.ask(question)
        st.subheader("Answer")
        st.write(response.answer)
        st.caption("Citations: " + (", ".join(response.citations) or "None"))

        metric_columns = st.columns(3)
        metric_columns[0].metric("Total latency", f"{response.latency.total_ms:,.1f} ms")
        metric_columns[1].metric("Retrieved chunks", len(response.retrieval.chunks))
        metric_columns[2].metric("Citations", len(response.citations))

        with st.expander("Stage latency"):
            latency_rows = pd.DataFrame(
                {
                    "Stage": ["Embedding", "Retrieval", "Prompt", "Generation"],
                    "Latency (ms)": [
                        round(response.latency.embedding_ms, 2),
                        round(response.latency.retrieval_ms, 2),
                        round(response.latency.prompt_construction_ms, 2),
                        round(response.latency.generation_ms, 2),
                    ],
                }
            )
            st.dataframe(latency_rows, hide_index=True, use_container_width=True)

        with st.expander("Retrieved evidence"):
            for chunk in response.retrieval.chunks:
                st.markdown(
                    f"**#{chunk.rank} · {chunk.source_name} · score {chunk.score:.3f}**"
                )
                st.write(chunk.text)

with comparison_tab:
    st.subheader("Deterministic vs. Ollama")
    st.caption(
        "Both runs use the same five questions. Use the same embedding backend for a fair generator comparison."
    )

    embedding_backend = st.selectbox(
        "Embedding backend for new runs",
        options=["hash", "sentence_transformers"],
        index=0,
    )

    run_columns = st.columns(2)
    with run_columns[0]:
        if st.button("Run deterministic benchmark", use_container_width=True):
            with st.spinner("Running deterministic benchmark..."):
                try:
                    run_benchmark("fixture", embedding_backend, DETERMINISTIC_DIR)
                    st.success("Deterministic results saved.")
                except Exception as error:
                    st.error(f"Deterministic benchmark failed: {error}")

    with run_columns[1]:
        if st.button("Run Ollama benchmark", use_container_width=True, type="primary"):
            with st.spinner("Running Ollama benchmark..."):
                try:
                    run_benchmark("ollama", embedding_backend, OLLAMA_DIR)
                    st.success("Ollama results saved.")
                except Exception as error:
                    st.error(
                        "Ollama benchmark failed. Confirm `ollama serve` is running and "
                        f"the `{settings.ollama_model}` model is available. Details: {error}"
                    )

    deterministic_results = load_benchmark_results(
        DETERMINISTIC_DIR / "benchmark_results.json"
    )
    ollama_results = load_benchmark_results(OLLAMA_DIR / "benchmark_results.json")
    deterministic_metadata = load_run_metadata(DETERMINISTIC_DIR / "run_metadata.json")
    ollama_metadata = load_run_metadata(OLLAMA_DIR / "run_metadata.json")

    if not deterministic_results and not ollama_results:
        st.info("Run at least one benchmark to display the comparison.")
    else:
        det_summary = summarize_results(deterministic_results)
        ollama_summary = summarize_results(ollama_results)

        det_embedding = deterministic_metadata.get("embedding_backend")
        ollama_embedding = ollama_metadata.get("embedding_backend")
        if det_embedding and ollama_embedding and det_embedding != ollama_embedding:
            st.warning(
                f"These runs use different embeddings: deterministic={det_embedding}, "
                f"Ollama={ollama_embedding}. Rerun both with the same backend for a fair comparison."
            )

        st.markdown("#### Overall")
        st.dataframe(
            summary_frame(det_summary, ollama_summary),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("#### By test case")
        st.dataframe(
            comparison_table(deterministic_results, ollama_results),
            hide_index=True,
            use_container_width=True,
        )

        with st.expander("Compare generated answers"):
            st.dataframe(
                answer_comparison_table(deterministic_results, ollama_results),
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("Run details"):
            details = pd.DataFrame(
                [
                    {
                        "Run": "Deterministic",
                        "Model": deterministic_metadata.get("generation_model", "Not run"),
                        "Embeddings": deterministic_metadata.get("embedding_backend", "Not run"),
                        "Timestamp": deterministic_metadata.get("timestamp_utc", "Not run"),
                    },
                    {
                        "Run": "Ollama",
                        "Model": ollama_metadata.get("generation_model", "Not run"),
                        "Embeddings": ollama_metadata.get("embedding_backend", "Not run"),
                        "Timestamp": ollama_metadata.get("timestamp_utc", "Not run"),
                    },
                ]
            )
            st.dataframe(details, hide_index=True, use_container_width=True)
