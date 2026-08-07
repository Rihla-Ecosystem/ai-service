from typing import List

from pydantic import computed_field, model_validator
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
    rate_limit_internal: int = 600
    max_tool_calls_per_turn: int = 5
    max_tool_timeout: int = 10

    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    # Comma-separated list of allowed origins. Use "*" only for local development.
    cors_origins: str = "http://localhost:3001,http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @computed_field
    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()] or ["*"]

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

    @model_validator(mode="after")
    def fail_fast_on_weak_secrets(self) -> "Settings":
        if self.environment == "production":
            weak = ("change-me-in-production", "secret", "changeme", "")
            if self.jwt_access_secret in weak or self.internal_api_key in weak:
                raise ValueError(
                    "Production requires strong JWT_ACCESS_SECRET and "
                    "INTERNAL_API_KEY (not placeholder values)."
                )
        return self


settings = Settings()