"""
live/state.py
──────────────
Persistent state management for live trading.
Saves strategy state to JSON so restarts don't lose context.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class StateManager:
    """Persist and reload strategy state to/from disk."""

    def __init__(self, state_dir: Path, strategy_name: str):
        self.path = Path(state_dir) / f"{strategy_name}_state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: Dict[str, Any]) -> None:
        state["_saved_at"] = datetime.utcnow().isoformat()
        with open(self.path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.debug("State saved: %s", self.path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            logger.info("No state file found at %s – starting fresh.", self.path)
            return {}
        with open(self.path) as f:
            state = json.load(f)
        logger.info("State loaded from %s (saved %s)", self.path, state.get("_saved_at"))
        return state

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
            logger.info("State cleared: %s", self.path)
