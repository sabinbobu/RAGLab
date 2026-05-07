def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks of approximately chunk_size characters.

    Args:
        text: raw text to split
        chunk_size: maximum characters per chunk
        overlap: number of characters to repeat between consecutive chunks

    Returns:
        list of text chunks
    """
    if not text.strip():
        return []

    chunks = []
    # step is how far we move forward after each chunk
    # smaller than chunk_size by `overlap` so consecutive chunks share context
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # only keep chunks that have actual content
        if chunk:
            chunks.append(chunk)

        # move forward by step, not chunk_size
        # this is what creates the overlap
        start += step

    return chunks
