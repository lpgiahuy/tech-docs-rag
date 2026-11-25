from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@127.0.0.1:5434/postgres"
    llm_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    # Comma-separated models tried in order when llm_model is out of quota, overloaded or gone.
    llm_fallback_models: str = "gemini-3.5-flash-lite,gemini-3.5-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    min_score: float = 0.25
    max_upload_mb: int = 20


settings = Settings()
