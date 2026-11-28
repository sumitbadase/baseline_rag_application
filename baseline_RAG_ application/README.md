# Hands-on Lab: Build and Benchmark the Baseline RAG Application

This lab builds the first working version of an Enterprise Knowledge Intelligence Platform with FastAPI, Streamlit, Qdrant local mode, local embeddings, Ollama support, Pydantic, and Pytest.

## Acceptance evidence

- Text and PDF ingestion through `BaselineLoader`
- Answers with source citations
- Expandable retrieval trace containing chunk text, source, rank, score, page, and metadata
- Embedding, retrieval, prompt, generation, and total latency in milliseconds
- Five-case benchmark with JSON, CSV, Markdown, and run-metadata artifacts
- Failure categories for vocabulary mismatch, exact identifiers, multi-hop retrieval, and unsupported questions

## Setup

```bash
cd "/Users/sumitbadase/Desktop/sucess_plan/advanced_rag_engineering/baseline_RAG_ application"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest -q
```

## Run Streamlit

```bash
streamlit run app.py
```

## Run FastAPI

```bash
uvicorn api:app --reload
```

Open `/docs` for the API contract. `POST /query` returns the answer, citations, retrieval trace, and stage-level latency.

## Benchmark modes

The benchmark runner has two independent generation modes. Each writes to a separate directory, so an Ollama run never overwrites the deterministic evidence.

### Deterministic acceptance benchmark

```bash
python run_benchmark.py
# equivalent:
python run_benchmark.py --provider fixture
```

Outputs:

```text
artifacts/deterministic/
├── benchmark_results.json
├── benchmark_results.csv
├── failure_analysis.md
└── run_metadata.json
```

Use this mode for automated tests, course recording, and acceptance validation. It uses the fixture generator and defaults to deterministic hash embeddings.

### Live Ollama benchmark

Start Ollama and make sure the configured model is available:

```bash
ollama serve
ollama pull llama3.2:3b
python run_benchmark.py --provider ollama
```

Outputs:

```text
artifacts/ollama/
├── benchmark_results.json
├── benchmark_results.csv
├── failure_analysis.md
└── run_metadata.json
```

The live run records the Ollama model, embedding backend, chunk settings, Top-K value, timestamp, and corpus size. Its generated answers and latency can vary across runs and machines.

### Optional Sentence Transformers retrieval

Sentence Transformers is intentionally optional because it is a larger dependency. Install it in a compatible Python environment, then run:

```bash
pip install sentence-transformers
python run_benchmark.py --provider ollama --embedding sentence_transformers
```

The default hash backend remains useful for deterministic acceptance testing and for making baseline retrieval failures reproducible.

### Custom benchmark destination

```bash
python run_benchmark.py --provider ollama --output-dir artifacts/experiments/llama32-run-01
```

## Configuration

Copy `.env.example` to `.env` and adjust these values as needed:

```text
EMBEDDING_BACKEND=hash
GENERATION_BACKEND=fixture
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
TOP_K=3
CHUNK_SIZE=500
CHUNK_OVERLAP=80
```

`GENERATION_BACKEND` controls the Streamlit and FastAPI applications. The benchmark CLI's explicit `--provider` argument takes precedence for that benchmark run.

## Verification

```bash
pytest -q
python run_benchmark.py --provider fixture
python run_benchmark.py --help
```

The deterministic benchmark is the acceptance gate. The Ollama benchmark is the live behavioral evaluation; it should be compared with, not substituted for, the deterministic benchmark.
