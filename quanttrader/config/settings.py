"""
config/settings.py  – Centralised configuration with optional Pydantic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_instrument_config(symbol: str) -> dict:
    """Return cost/metadata config for *symbol*, falling back to defaults."""
    cfg_path = ROOT_DIR / "config" / "instruments.yaml"
    raw = _load_yaml(cfg_path)
    defaults = raw.get("default", {
        "spread": 0.0, "commission_pct": 0.001, "commission_fixed": 0.0,
        "slippage_pct": 0.0005, "margin_rate": 1.0, "lot_size": 1.0, "min_lot": 1.0,
    })
    instruments = raw.get("instruments", {})
    inst = instruments.get(symbol, {})
    return {**defaults, **inst}


class _DataSettings:
    def __init__(self):
        self.cache_dir = ROOT_DIR / "data" / "cache"
        self.default_source = os.environ.get("DATA__DEFAULT_SOURCE", "yfinance")
        self.alpha_vantage_key = os.environ.get("DATA__ALPHA_VANTAGE_KEY")
        self.csv_dir = None


class _Settings:
    def __init__(self):
        self.data = _DataSettings()
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.log_dir = ROOT_DIR / "logs"
        self.models_dir = ROOT_DIR / "models"
        self.reports_dir = ROOT_DIR / "reports"
        self.default_initial_capital = 100_000.0
        self.default_timeframe = "1d"

    def ensure_dirs(self):
        for d in [self.log_dir, self.models_dir, self.reports_dir,
                  self.data.cache_dir, ROOT_DIR / "data"]:
            Path(d).mkdir(parents=True, exist_ok=True)


# Try pydantic first, fall back to plain class
try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from python_dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env",
                                           env_nested_delimiter="__", extra="ignore")
        log_level: str = "INFO"
        log_dir: Path = ROOT_DIR / "logs"
        models_dir: Path = ROOT_DIR / "models"
        reports_dir: Path = ROOT_DIR / "reports"
        default_initial_capital: float = 100_000.0

        def ensure_dirs(self):
            for d in [self.log_dir, self.models_dir, self.reports_dir,
                      ROOT_DIR / "data", ROOT_DIR / "data" / "cache"]:
                Path(d).mkdir(parents=True, exist_ok=True)

    settings = Settings()

except Exception:
    settings = _Settings()

