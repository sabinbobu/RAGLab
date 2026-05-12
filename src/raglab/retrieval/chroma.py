import chromadb
import logfire

from raglab.ingestion.embedder import embed_batch
from raglab.retrieval.base import RetrievedChunk


class ChromaRetriever:
    def __init__(self, collection_name: str) -> None:
        # persistent client reads from the same .chroma folder the CLI wrote to
        client = chromadb.PersistentClient(path=".chroma")
        self.collection = client.get_collection(name=collection_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """
        Embed the query, search ChromaDB for top_k similar chunks,
        return results with text and metadata.

        Args:
            query: user question as plain text
            top_k: number of chunks to retrieve

        Returns:
            list of RetrievedChunk ordered by similarity score
        """
        # embed_batch expects a list, returns a list — we only need the first vector
        query_vector = embed_batch([query])[0]

        with logfire.span("chroma.query", collection=self.collection.name, top_k=top_k):
            results = self.collection.query(
                query_embeddings=[query_vector],  # type: ignore[arg-type]
                n_results=top_k,
                # tell ChromaDB to return documents, metadatas and distances
                include=["documents", "metadatas", "distances"],
            )

        chunks = []

        # results are nested in lists because ChromaDB supports batch queries
        # [0] unwraps the first (and only) query in our batch
        # the fields are always present given our include= parameter, asserts confirm it
        assert results["documents"] is not None
        assert results["metadatas"] is not None
        assert results["distances"] is not None
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=str(metadata["source"]),
                    page=int(metadata["page"]),  # type: ignore[arg-type]
                    # ChromaDB returns distance (lower = more similar)
                    # convert to score (higher = more similar) for readability
                    score=round(1 - distance, 4),
                )
            )

        return chunks
