from unittest.mock import MagicMock, patch

from raglab.retrieval.chroma import ChromaRetriever


def test_chroma_retriever_returns_retrieved_chunks():
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["chunk text 1", "chunk text 2"]],
        "metadatas": [
            [
                {"source": "test.pdf", "page": 1, "chunk_index": 0},
                {"source": "test.pdf", "page": 1, "chunk_index": 1},
            ]
        ],
        "distances": [[0.1, 0.3]],
    }

    with patch("raglab.retrieval.chroma.chromadb.PersistentClient") as mock_client:
        with patch("raglab.retrieval.chroma.embed_batch", return_value=[[0.1] * 1536]):
            mock_client.return_value.get_collection.return_value = mock_collection

            retriever = ChromaRetriever(collection_name="test")
            results = retriever.retrieve("what is RAG?", top_k=2)

    assert len(results) == 2
    assert results[0].text == "chunk text 1"
    assert results[0].source == "test.pdf"
    assert results[0].page == 1
    # higher score = more similar (1 - distance)
    assert results[0].score > results[1].score


def test_chroma_retriever_score_is_inverted_distance():
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["chunk"]],
        "metadatas": [[{"source": "f.pdf", "page": 1, "chunk_index": 0}]],
        "distances": [[0.2]],
    }

    with patch("raglab.retrieval.chroma.chromadb.PersistentClient") as mock_client:
        with patch("raglab.retrieval.chroma.embed_batch", return_value=[[0.1] * 1536]):
            mock_client.return_value.get_collection.return_value = mock_collection

            retriever = ChromaRetriever(collection_name="test")
            results = retriever.retrieve("query")

    # score should be 1 - 0.2 = 0.8
    assert results[0].score == round(1 - 0.2, 4)
