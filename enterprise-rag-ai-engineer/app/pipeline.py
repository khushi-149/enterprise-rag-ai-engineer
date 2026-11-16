import time
from pathlib import Path
from app.config import get_settings
from app.llm import generate_answer
from app.retrieval import HybridRetriever

class RAGPipeline:
    def __init__(self):
        self.cfg = get_settings()
        c = self.cfg
        self.retriever = HybridRetriever(c.embedding_model, c.enable_reranker, c.reranker_model)
        folder = Path(c.storage_dir)
        if (folder/"vectors.faiss").exists(): self.retriever.load(folder)

    def query(self, question, top_k=None):
        start = time.perf_counter()
        hits = self.retriever.search(question, top_k or self.cfg.top_k, self.cfg.candidate_k)
        abstained = not hits or hits[0]["score"] < self.cfg.min_relevance_score
        answer = ("I don't have enough information in the provided knowledge base to answer that."
                  if abstained else generate_answer(question, hits))
        citations = [] if abstained else [
            {"citation_id":i,"source":h["source"],"chunk_id":h["chunk_id"],
             "score":round(float(h["score"]),5),"excerpt":h["text"][:400]}
            for i,h in enumerate(hits,1)]
        return {"answer":answer,"citations":citations,"abstained":abstained,
                "latency_ms":round((time.perf_counter()-start)*1000,2)}
