"""
tests/conftest.py
=================
Shared pytest fixtures and configuration.
"""
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use a temp directory for test databases
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(autouse=True)
def tmp_data_dirs(tmp_path, monkeypatch):
    """Redirect data directories to a temporary location for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    yield
