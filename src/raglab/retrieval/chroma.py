from raglab.retrieval.base import RetrievedChunk


class ChromaRetriever:
    def __init__(self, collection_name: str) -> None:
        # initialize ChromaDB persistent client and get collection
        raise NotImplementedError

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
        raise NotImplementedError
