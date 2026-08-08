from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # --- AI image generation ---
    ai_image_provider: str = "gemini"

    gemini_api_key: str | None = None
    gemini_image_model: str = "gemini-3.1-flash-image"

    replicate_api_token: str | None = None
    replicate_model: str = "google/nano-banana-2"
    replicate_image_input_field: str = "image_input"
    replicate_image_input_is_array: bool = True
    replicate_aspect_ratio: str = "match_input_image"
    replicate_output_format: str = "png"

    # --- Product matching (Pinecone + Supabase) ---
    supabase_url: str | None = None
    supabase_key: str | None = None

    pinecone_api_key: str | None = None
    pinecone_index_name: str = "ikea-products"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    clip_model_name: str = "clip-ViT-B-32"
    embedding_dimension: int = 512


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
