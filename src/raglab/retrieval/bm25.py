# retrieval/bm25.py
from rank_bm25 import BM25Okapi

from raglab.retrieval.base import RetrievedChunk


class BM25Retriever:
    def __init__(self, chunks: list[str], metadata: list[dict]) -> None:
        tokenized = [doc.split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.chunks = chunks
        self.metadata = metadata

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [
            RetrievedChunk(
                text=self.chunks[i],
                source=self.metadata[i].get("source", ""),
                page=self.metadata[i].get("page", 0),
                score=float(scores[i]),
            )
            for i in top_indices
        ]
