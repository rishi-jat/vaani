from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=ROOT / "data", alias="VAANI_DATA_DIR")
    index_name: str = Field(default="shipped", alias="VAANI_INDEX_NAME")
    languages: str = Field(default="hi,mr", alias="VAANI_LANGUAGES")
    host: str = Field(default="0.0.0.0", alias="VAANI_HOST")
    port: int = Field(default=8080, alias="VAANI_PORT")

    model_name: str = "intfloat/multilingual-e5-small"
    # If this directory contains a full snapshot, Encoder loads from disk
    # and never hits the Hub at query time.
    local_model_dir: str = Field(default="", alias="VAANI_LOCAL_MODEL")
    # Skip FAISS + the e5 encoder; BM25-only. Needed to boot on Railway Trial 1GB.
    low_mem: bool = Field(default=False, alias="VAANI_LOW_MEM")
    embed_dim: int = 384
    max_query_chars: int = 512
    top_k: int = 8
    rrf_k: int = 60
    hnsw_m: int = 32
    hnsw_ef_construction: int = 80
    hnsw_ef_search: int = 64

    support_threshold: float = 0.42
    retrieve_threshold: float = 0.22
    budget_ms: float = 200.0
    retrieve_timeout_ms: float = 80.0
    extract_timeout_ms: float = 80.0
    generate_timeout_s: float = 4.0
    stt_timeout_s: float = 20.0
    stt_retries: int = 2

    stt_provider: str = Field(default="sarvam", alias="STT_PROVIDER")
    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    xai_api_key: str = Field(default="", alias="XAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    llm_model: str = "grok-4.5"
    llm_base_url: str = "https://api.x.ai/v1"

    @property
    def active_llm_api_key(self) -> str:
        return self.groq_api_key or self.xai_api_key or self.openai_api_key

    @property
    def active_llm_base_url(self) -> str:
        if self.groq_api_key:
            return "https://api.groq.com/openai/v1"
        if self.xai_api_key:
            return self.llm_base_url
        if self.openai_api_key:
            return "https://api.openai.com/v1"
        return self.llm_base_url

    @property
    def active_llm_model(self) -> str:
        if self.groq_api_key:
            return "llama-3.1-8b-instant"
        return self.llm_model

    @property
    def lang_list(self) -> list[str]:
        return [p.strip() for p in self.languages.split(",") if p.strip()]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes" / self.index_name

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def hf_cache(self) -> Path:
        return self.data_dir / "hf_cache"

    @property
    def model_cache(self) -> Path:
        return self.data_dir / "models"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    s.hf_cache.mkdir(parents=True, exist_ok=True)
    s.model_cache.mkdir(parents=True, exist_ok=True)
    return s
