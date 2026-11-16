import json
from pathlib import Path
from app.config import get_settings
from app.retrieval import HybridRetriever

def main():
    c = get_settings()
    r = HybridRetriever(c.embedding_model, False, c.reranker_model); r.load(c.storage_dir)
    cases = json.loads(Path("data/eval_questions.json").read_text())
    recalls, rr = [], []
    for case in cases:
        hits = r.search(case["question"], 10, 30)
        sources = [h["source"] for h in hits]; expected = set(case["relevant_sources"])
        recalls.append(float(bool(expected.intersection(sources))))
        rank = next((i for i,s in enumerate(sources,1) if s in expected), None)
        rr.append(1/rank if rank else 0)
    print(f"Queries: {len(cases)} | Recall@10: {sum(recalls)/len(recalls):.3f} | MRR@10: {sum(rr)/len(rr):.3f}")

if __name__ == "__main__": main()
