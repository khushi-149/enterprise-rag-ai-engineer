# AtlasRAG — Enterprise Knowledge Assistant

Advanced portfolio-grade Retrieval-Augmented Generation (RAG) application for AI Engineer roles.

## Highlights
- Ingests PDF, DOCX, Markdown and TXT documents.
- Hybrid retrieval: Sentence Transformers embeddings + FAISS dense search + BM25 lexical search.
- Reciprocal Rank Fusion combines retrieval rankings.
- Optional CrossEncoder reranking.
- Grounded LLM responses with source citations and abstention when evidence is weak.
- FastAPI backend, Streamlit interface, retrieval evaluation, tests and Docker.
- Synthetic example knowledge base included: no dataset download required to try it.

## Architecture
Documents → parsing/chunking → embeddings + BM25 → hybrid retrieval → reranking → evidence-grounded LLM → answer + citations.

## Setup
Python 3.10–3.12 recommended.

```bash
git clone https://github.com/<your-username>/enterprise-rag-ai-engineer.git
cd enterprise-rag-ai-engineer
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Configure generation
Default backend is Ollama. Install Ollama and run `ollama pull llama3.1:8b`, then keep `LLM_PROVIDER=ollama`.

Alternatively use OpenAI: set `LLM_PROVIDER=openai` and add `OPENAI_API_KEY` in `.env`.
For retrieval-only mode set `LLM_PROVIDER=none`.

### Index documents and launch
```bash
python -m scripts.ingest
uvicorn app.api:app --reload --port 8000
```
In another terminal:
```bash
streamlit run streamlit_app.py
```
API docs: http://localhost:8000/docs

### Example API request
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"How long are audit logs retained?","top_k":5}'
```

## Dataset / knowledge base
`data/knowledge/` contains fictional sample policy, product, support and privacy documents. They are synthetic demo data and are safe to publish. Add your own permitted `.pdf`, `.docx`, `.md` or `.txt` documents there, then rerun ingestion. Never commit confidential data or secrets.

`data/eval_questions.json` is a small editable retrieval benchmark with expected source documents. Run:
```bash
python -m scripts.evaluate
```
It reports Recall@10 and MRR@10. This is a retrieval benchmark, not proof of answer faithfulness. Expand it with human-reviewed question/source/answer examples before making performance claims.

## Tests and Docker
```bash
pip install -r requirements-dev.txt
pytest -q
docker compose up --build
```
Docker starts the API. The Streamlit UI can run locally. Configure Ollama networking appropriately if using Docker.

## Configuration
See `.env.example`. Key options: `LLM_PROVIDER`, `OPENAI_MODEL`, `OLLAMA_MODEL`, `EMBEDDING_MODEL`, `ENABLE_RERANKER`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_RELEVANCE_SCORE`, `STORAGE_DIR`.

## Production hardening
Before real deployment add authentication, tenant isolation and ACL-aware retrieval, rate limiting, secrets management, document deletion/versioning, prompt-injection defenses, PII controls, managed vector storage, async ingestion, and monitoring for p95 latency, cost, retrieval quality and abstention rate.

## Resume bullet
Built AtlasRAG, a hybrid RAG platform using FastAPI, Sentence Transformers, FAISS and BM25, with reciprocal-rank fusion, optional cross-encoder reranking, citation-grounded generation, abstention, evaluation (Recall@K/MRR), Docker and automated tests. Add measured metrics only after running your own benchmark.

## Interview topics
Explain hybrid vs dense retrieval, chunk-size tradeoffs, RRF, reranking, faithfulness evaluation, prompt injection, ACL filtering, index freshness and latency/cost optimization.

Choose and review a software license and all model/dependency licenses before public release.
