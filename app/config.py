from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    project_name: str = "Rihla AI Service"
    environment: str = "local"

    port: int = 3003
    log_level: str = "INFO"

    gemini_api_keys: List[str] = []
    jwt_access_secret: str = "change-me-in-production"
    internal_api_key: str = "change-me-in-production"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    gis_service_url: str = "http://geocontext:8000"
    risk_service_url: str = "http://risk-intelligence:3001"
    core_server_url: str = "http://core-server:3000"

    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333

    rag_data_dir: str = "data/rag"

    rate_limit_per_user: int = 30
    max_tool_calls_per_turn: int = 5
    max_tool_timeout: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def gemini_key_list(self) -> List[str]:
        if isinstance(self.gemini_api_keys, str):
            return [k.strip() for k in self.gemini_api_keys.split(",") if k.strip()]
        return self.gemini_api_keys


settings = Settings()
