from typing import List

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Rihla AI Service"
    environment: str = "local"

    port: int = 3003
    log_level: str = "INFO"

    # Read as plain string from .env
    gemini_api_keys: str = ""
    gemini_model: str = "gemini-3.6-flash"
    jina_api_key: str = ""
    tts_voice: str = "Zephyr"

    jwt_access_secret: str = "change-me-in-production"
    internal_api_key: str = "change-me-in-production"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    gis_service_url: str = "http://geocontext:8000"
    risk_service_url: str = "http://risk-intelligence:3000"
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

    @computed_field
    @property
    def gemini_key_list(self) -> List[str]:
        if not self.gemini_api_keys:
            return []

        return [
            key.strip()
            for key in self.gemini_api_keys.split(",")
            if key.strip()
        ]


settings = Settings()