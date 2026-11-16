import json, logging, re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)
def tokenize(text): return re.findall(r"[a-zA-Z0-9_]+", text.lower())

class HybridRetriever:
    def __init__(self, embedding_model, enable_reranker=True, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss
        self.faiss = faiss
        self.encoder = SentenceTransformer(embedding_model)
        self.reranker = None
        if enable_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(reranker_model)
            except Exception as exc: log.warning("Reranker disabled: %s", exc)
        self.records, self.index, self.bm25 = [], None, None

    def build(self, records):
        if not records: raise ValueError("No chunks to index")
        self.records = records
        vectors = self.encoder.encode([r["text"] for r in records], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=True).astype("float32")
        self.index = self.faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.bm25 = BM25Okapi([tokenize(r["text"]) for r in records])

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(directory / "vectors.faiss"))
        (directory / "records.json").write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def load(self, directory):
        directory = Path(directory)
        self.index = self.faiss.read_index(str(directory / "vectors.faiss"))
        self.records = json.loads((directory / "records.json").read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([tokenize(r["text"]) for r in self.records])

    def search(self, query, top_k=5, candidate_k=20):
        if self.index is None: raise RuntimeError("Index is not loaded")
        q = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        _, dense_ids = self.index.search(q, min(candidate_k, len(self.records)))
        lexical = self.bm25.get_scores(tokenize(query))
        lexical_ids = np.argsort(lexical)[::-1][:candidate_k]
        fused = {}
        for rank, idx in enumerate(dense_ids[0], 1):
            if idx >= 0: fused[int(idx)] = fused.get(int(idx), 0) + 1/(60+rank)
        for rank, idx in enumerate(lexical_ids, 1):
            fused[int(idx)] = fused.get(int(idx), 0) + 1/(60+rank)
        hits = []
        for idx in sorted(fused, key=fused.get, reverse=True):
            item = dict(self.records[idx]); item["score"] = float(fused[idx]); hits.append(item)
        if self.reranker and hits:
            scores = self.reranker.predict([(query, h["text"]) for h in hits])
            for h, score in zip(hits, scores): h["rerank_score"] = float(score)
            hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        return hits[:top_k]
