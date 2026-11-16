from app.config import get_settings
from app.ingest import load_chunks
from app.retrieval import HybridRetriever

def main():
    c = get_settings()
    records = load_chunks(c.knowledge_dir, c.chunk_size, c.chunk_overlap)
    if not records: raise SystemExit("No supported documents found in knowledge directory")
    retriever = HybridRetriever(c.embedding_model, c.enable_reranker, c.reranker_model)
    retriever.build(records); retriever.save(c.storage_dir)
    print(f"Indexed {len(records)} chunks from {c.knowledge_dir}")

if __name__ == "__main__": main()
