"""
experiments/database.py
────────────────────────
Experiment storage. Uses SQLAlchemy ORM when available, falls back to
raw sqlite3 so the module works without heavy dependencies during testing.
"""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from config.settings import ROOT_DIR

logger = logging.getLogger(__name__)

DB_PATH = ROOT_DIR / "data" / "experiments.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Schema DDL ─────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT DEFAULT (datetime('now')),
    strategy_name    TEXT,
    strategy_version TEXT DEFAULT '1.0.0',
    symbols          TEXT,
    timeframe        TEXT,
    params           TEXT,
    date_range_start TEXT,
    date_range_end   TEXT,
    metrics          TEXT,
    equity_curve     BLOB,
    trades           TEXT,
    notes            TEXT DEFAULT '',
    tags             TEXT DEFAULT ''
)
"""


class ExperimentDB:
    """SQLite-backed experiment store."""

    def __init__(self, db_url: Optional[str] = None):
        # Accept both "sqlite:///path" and plain paths
        if db_url and db_url.startswith("sqlite:///"):
            path = db_url[len("sqlite:///"):]
        elif db_url:
            path = db_url
        else:
            path = str(DB_PATH)

        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(_DDL)

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_experiment(self, result: Any, notes: str = "", tags: str = "") -> int:
        symbols = (result.symbol if isinstance(result.symbol, str)
                   else ",".join(result.symbol))
        trades_json = json.dumps([
            {k: str(v) for k, v in t.__dict__.items()} for t in result.trades
        ])
        equity_blob = pickle.dumps(result.equity)

        sql = """INSERT INTO experiments
                 (strategy_name, symbols, timeframe, params,
                  date_range_start, date_range_end, metrics,
                  equity_curve, trades, notes, tags)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?)"""
        vals = (
            result.strategy_name, symbols, result.timeframe,
            json.dumps(result.params), result.start, result.end,
            json.dumps(result.metrics), equity_blob, trades_json,
            notes, tags,
        )
        with self._conn() as conn:
            cur = conn.execute(sql, vals)
            eid = cur.lastrowid
        logger.info("Experiment #%d logged (%s, %s)", eid, result.strategy_name, symbols)
        return eid

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_experiments(
        self,
        strategy_name: Optional[str] = None,
        symbols: Optional[str] = None,
        tags: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        where, params = [], []
        if strategy_name:
            where.append("strategy_name LIKE ?")
            params.append(f"%{strategy_name}%")
        if symbols:
            where.append("symbols LIKE ?")
            params.append(f"%{symbols}%")
        if tags:
            where.append("tags LIKE ?")
            params.append(f"%{tags}%")

        sql = ("SELECT id,timestamp,strategy_name,strategy_version,symbols,timeframe,"
               "params,date_range_start,date_range_end,metrics,notes,tags "
               "FROM experiments")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY timestamp DESC LIMIT {limit}"

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            cols = [d[0] for d in conn.execute(sql, params).description] if rows else []

        if not rows:
            return pd.DataFrame()

        # Re-fetch with description
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()

        df = pd.DataFrame(rows, columns=cols)
        df["params"]  = df["params"].apply(lambda x: json.loads(x or "{}"))
        df["metrics"] = df["metrics"].apply(lambda x: json.loads(x or "{}"))

        # Flatten metrics into columns
        metrics_df = pd.json_normalize(df["metrics"])
        df = pd.concat([df.drop(columns=["metrics", "params"]), metrics_df], axis=1)
        return df

    def get_equity_curve(self, experiment_id: int) -> Optional[pd.Series]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT equity_curve FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
        if row and row[0]:
            return pickle.loads(row[0])
        return None

    def delete_experiment(self, experiment_id: int) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
        return True

    # ── Import / Export ───────────────────────────────────────────────────────

    def export_csv(self, path: Union[str, Path]) -> None:
        df = self.get_experiments(limit=10_000)
        df.to_csv(path, index=False)

    def export_json(self, path: Union[str, Path]) -> None:
        df = self.get_experiments(limit=10_000)
        df.to_json(path, orient="records", indent=2)


# ── Module-level helpers ──────────────────────────────────────────────────────

_default_db: Optional[ExperimentDB] = None


def _get_db() -> ExperimentDB:
    global _default_db
    if _default_db is None:
        _default_db = ExperimentDB()
    return _default_db


def log_experiment(result: Any, notes: str = "", tags: str = "") -> int:
    return _get_db().log_experiment(result, notes, tags)


def get_experiments(**kwargs) -> pd.DataFrame:
    return _get_db().get_experiments(**kwargs)
