from app.retrieval import tokenize

def test_tokenize():
    assert tokenize("Audit-Logs, RETENTION!") == ["audit","logs","retention"]
