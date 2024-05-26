from __future__ import annotations as _annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource


class Configuration(BaseSettings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Change source priority order"""
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    training_config_path: Path = "/app/config"
    data_path: Path = "/tmp/dataset"
    training_weight_path: Path = "/tmp/weights"
    model_weights_path: Path = "/app/weights"
    master_url: str = "192.168.30.183:6000"
    env_mode_auto: bool = True
    worker_name: str
    source_server: str
    requests_scheme: str = "http"
    capabilities_version: float = 0.8

    logfile: Optional[Path] = None
    pending_jobs_check_interval: float = 5.0
    executing_for_customer: Optional[str] = None
