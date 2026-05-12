from typing import Any

import chromadb

from raglab.retrieval.base import Retriever
from raglab.retrieval.bm25 import BM25Retriever
from raglab.retrieval.chroma import ChromaRetriever
from raglab.retrieval.hybrid import HybridRetriever


def get_retriever(name: str, collection: chromadb.Collection) -> Retriever:
    all_docs = collection.get(include=["documents", "metadatas"])
    chunks: list[str] = all_docs["documents"] or []
    metadata: list[dict[str, Any]] = [dict(m) for m in (all_docs["metadatas"] or [])]

    if name == "chroma":
        return ChromaRetriever(collection.name)
    elif name == "bm25":
        return BM25Retriever(chunks, metadata)
    elif name == "hybrid":
        chroma = ChromaRetriever(collection.name)
        bm25 = BM25Retriever(chunks, metadata)
        return HybridRetriever([chroma, bm25])
    else:
        raise ValueError(f"Unknown retriever: {name}")
