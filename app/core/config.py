# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    BEARER_TOKEN: str
    GOOGLE_API_KEY: str  # Changed to Google's key
    LLM_MODEL: str = "gemini-1.5-flash" # Changed to a Gemini model name

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()