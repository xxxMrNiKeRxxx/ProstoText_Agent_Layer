from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8000
    transport_transfer_url: str = "http://127.0.0.1:8080/simplify/transfer"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    max_segment_bytes: int = 1000
    sentences_per_part: int = 2
    transport_timeout_sec: float = 15.0
    ollama_timeout_sec: float = 120.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()