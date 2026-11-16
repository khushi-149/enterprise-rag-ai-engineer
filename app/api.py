from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.pipeline import RAGPipeline

app = FastAPI(title="AtlasRAG API", version="1.0.0")
pipeline = None

@app.on_event("startup")
def startup():
    global pipeline
    try: pipeline = RAGPipeline()
    except Exception as exc: app.state.startup_error = str(exc)

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

@app.get("/health")
def health():
    ready = pipeline is not None and pipeline.retriever.index is not None
    return {"status":"ready" if ready else "not_ready",
            "indexed_chunks":len(pipeline.retriever.records) if ready else 0,
            "startup_error":getattr(app.state,"startup_error",None)}

@app.post("/query")
def query(req: QueryRequest):
    if pipeline is None or pipeline.retriever.index is None:
        raise HTTPException(503, "Index unavailable. Run python -m scripts.ingest first.")
    try: return pipeline.query(req.question, req.top_k)
    except Exception as exc: raise HTTPException(500, f"Query failed: {exc}") from exc
