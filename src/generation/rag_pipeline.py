from __future__ import annotations

from time import perf_counter

from src.retrieval.vector_retriever import VectorRetriever
from src.shared.models import LatencyMetrics, RagResponse, RetrievalTrace
from src.shared.protocols import GenerationProvider
from src.shared.timing import measure_ms


class RagPipeline:
    def __init__(self, retriever: VectorRetriever, generator: GenerationProvider, top_k: int = 3) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    @staticmethod
    def _build_prompt(question: str, chunks: list) -> str:
        context = "\n\n".join(
            f"SOURCE: {chunk.source_name}\nCHUNK_ID: {chunk.chunk_id}\nCONTENT: {chunk.text}"
            for chunk in chunks
        )
        return (
            "You are a grounded question-answering system. Use only the supplied context.\n"
            "Follow this output contract exactly:\n"
            "1. If the context contains enough evidence, answer the question directly in one or two sentences.\n"
            "2. Include supporting source filenames in square brackets.\n"
            "3. Do not say information is missing when the context states the requested fact.\n"
            "4. If the context is insufficient, output exactly this sentence and nothing else:\n"
            "I could not find this information in the provided documents.\n"
            "5. Never combine the abstention sentence with an answer or explanation.\n\n"
            f"QUESTION: {question}\n\nCONTEXT:\n{context}"
        )

    def ask(self, question: str) -> RagResponse:
        started = perf_counter()
        chunks, embedding_ms, retrieval_ms = self.retriever.retrieve(question, self.top_k)
        prompt, prompt_ms = measure_ms(lambda: self._build_prompt(question, chunks))
        answer, generation_ms = measure_ms(lambda: self.generator.generate(prompt))
        citations = sorted({chunk.source_name for chunk in chunks if chunk.source_name in answer})
        total_ms = (perf_counter() - started) * 1000
        return RagResponse(
            answer=answer,
            citations=citations,
            retrieval=RetrievalTrace(query=question, top_k=self.top_k, chunks=chunks),
            latency=LatencyMetrics(
                embedding_ms=embedding_ms,
                retrieval_ms=retrieval_ms,
                prompt_construction_ms=prompt_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
        )
