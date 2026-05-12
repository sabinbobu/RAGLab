# retrieval/hybrid.py
from raglab.retrieval.base import RetrievedChunk, Retriever


class HybridRetriever:
    def __init__(self, retrievers: list[Retriever], k: int = 60) -> None:
        self.retrievers = retrievers
        self.k = k  # RRF constant

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for retriever in self.retrievers:
            results = retriever.retrieve(query, top_k=top_k * 2)
            for rank, chunk in enumerate(results):
                key = f"{chunk.source}::{chunk.page}::{chunk.text[:50]}"
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (self.k + rank + 1)
                chunk_map[key] = chunk

        top_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                text=chunk_map[k].text,
                source=chunk_map[k].source,
                page=chunk_map[k].page,
                score=rrf_scores[k],
            )
            for k in top_keys
        ]
