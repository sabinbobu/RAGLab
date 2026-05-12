from raglab.ingestion.chunkers import recursive_chunk


def test_chunk_splits_text_into_correct_sizes():
    text = "a" * 1000
    chunks = recursive_chunk(text, chunk_size=100, overlap=10)

    assert len(chunks) > 1
    # no chunk should exceed chunk_size
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_overlap_exists_between_consecutive_chunks():
    text = "abcdefghij" * 100
    chunks = recursive_chunk(text, chunk_size=50, overlap=10)

    # the end of chunk N should appear at the start of chunk N+1
    assert chunks[0][-10:] == chunks[1][:10]


def test_chunk_empty_text_returns_empty_list():
    assert recursive_chunk("") == []
    assert recursive_chunk("   ") == []


def test_chunk_short_text_returns_single_chunk():
    text = "short text"
    chunks = recursive_chunk(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text
