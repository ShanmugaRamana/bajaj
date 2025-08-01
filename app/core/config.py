# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    BEARER_TOKEN: str
    OPENROUTER_API_KEY: str
    LLM_MODEL: str = "openrouter/horizon-beta"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()