from openai import OpenAI

from raglab.config import settings


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of strings into embedding vectors using OpenAI.

    Args:
        texts: list of text chunks to embed

    Returns:
        list of vectors, one per input text — each vector is 1536 floats
    """
    if not texts:
        return []

    client = OpenAI(api_key=settings.openai_api_key)

    response = client.embeddings.create(
        input=texts,
        # small model is 1536 dimensions, cheap, strong retrieval performance
        model="text-embedding-3-small",
    )

    # response.data is a list of Embedding objects, one per input text
    # .embedding is the actual vector — a list of 1536 floats
    return [item.embedding for item in response.data]
