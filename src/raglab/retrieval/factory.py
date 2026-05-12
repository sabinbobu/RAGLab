import chromadb

from raglab.retrieval.base import Retriever
from raglab.retrieval.bm25 import BM25Retriever
from raglab.retrieval.chroma import ChromaRetriever
from raglab.retrieval.hybrid import HybridRetriever


def get_retriever(name: str, collection: chromadb.Collection) -> Retriever:
    all_docs = collection.get(include=["documents", "metadatas"])
    chunks = all_docs["documents"]
    metadata = all_docs["metadatas"]

    if name == "chroma":
        return ChromaRetriever(collection)
    elif name == "bm25":
        return BM25Retriever(chunks, metadata)
    elif name == "hybrid":
        chroma = ChromaRetriever(collection)
        bm25 = BM25Retriever(chunks, metadata)
        return HybridRetriever([chroma, bm25])
    else:
        raise ValueError(f"Unknown retriever: {name}")
