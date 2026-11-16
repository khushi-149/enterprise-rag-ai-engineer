from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_provider: str = "ollama"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranker: bool = True
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    candidate_k: int = 20
    min_relevance_score: float = 0.12
    storage_dir: str = "storage"
    knowledge_dir: str = "data/knowledge"

@lru_cache
def get_settings():
    return Settings()
