import pytest
from app.ingest import clean_text, chunk_text

def test_clean_text():
    assert clean_text("a   b\\n\\n\\nc") == "a b\\n\\nc"

def test_chunking():
    chunks = chunk_text(" ".join(f"word{i}" for i in range(100)), 100, 20)
    assert len(chunks) > 1 and all(chunks)

def test_invalid_overlap():
    with pytest.raises(ValueError): chunk_text("hello", 10, 10)
