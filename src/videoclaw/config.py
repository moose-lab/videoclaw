"""Global configuration loaded from environment variables and .env files."""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VideoClawConfig(BaseSettings):
    """VideoClaw application configuration.

    Values are read from environment variables prefixed with ``VIDEOCLAW_``,
    falling back to a ``.env`` file in the current working directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="VIDEOCLAW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Directories ---
    projects_dir: Path = Field(default=Path("./projects"))
    models_dir: Path = Field(default=Path("./models_cache"))

    # --- Model defaults ---
    default_llm: str = "gpt-4o"
    default_video_model: str = "mock"

    # --- Logging ---
    log_level: str = "info"

    # --- API keys (read without the VIDEOCLAW_ prefix too) ---
    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)

    # --- Budget & resilience ---
    budget_default_usd: float = 10.0
    max_retries: int = 3

    def ensure_dirs(self) -> None:
        """Create project and model cache directories if they don't exist."""
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_config() -> VideoClawConfig:
    """Return the singleton :class:`VideoClawConfig` instance.

    The config is created on first call and cached for the process lifetime.
    To reload, clear the cache via ``get_config.cache_clear()``.
    """
    return VideoClawConfig()
