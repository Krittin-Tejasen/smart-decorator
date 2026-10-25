from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
# Same two files main.py loads. Read here as well because `settings` is built
# at import time, which can happen before main.py has called load_dotenv.
ENV_FILES = (BACKEND_DIR / ".env", BACKEND_DIR / "backend.env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILES, env_file_encoding="utf-8", extra="ignore")

    # --- Product matching (Pinecone + Supabase) ---
    supabase_url: str | None = None
    # The backend reads the catalog with the project's secret key; accept the
    # names it has had in different .env files.
    supabase_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_KEY", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY"),
    )

    pinecone_api_key: str | None = None
    pinecone_index_name: str = "ikea-products"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    clip_model_name: str = "clip-ViT-B-32"
    embedding_dimension: int = 512

    # Matches whose image similarity (cosine, 0..1) is below this are not shown.
    # Provisional until it is calibrated on real generated rooms.
    match_min_score: float = 0.75


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
