
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_api_key: str = "EMPTY"
    use_llm_fallback: bool = False
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    embedding_model: str = "intfloat/multilingual-e5-base"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    chroma_path: str = "data/index/chroma"
    collection_name: str = "egyptian_civil_code"
    top_k: int = 8
    chunk_size: int = 1200
    chunk_overlap: int = 150

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "legal-rag"
    mlflow_model_name: str = "LegalRAGConfig"

    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    pii_block: bool = True
    max_question_length: int = 2000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
