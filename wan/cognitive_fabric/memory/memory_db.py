"""SQLite project memory (stdlib only): shots, chunks, quality reports and
repair history, queryable across sessions."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shots (
    shot_id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    shot_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    output_path TEXT,
    spec_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS quality_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL,
    overall REAL NOT NULL,
    report_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS repairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    created_at REAL NOT NULL
);
"""


class MemoryDB:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- shots ----------------------------------------------------------------

    def upsert_shot(self, shot_id: str, scene_id: str,
                    spec: Dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO shots (shot_id, scene_id, spec_json, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(shot_id) DO UPDATE SET spec_json=excluded.spec_json",
            (shot_id, scene_id, json.dumps(spec), time.time()))
        self._conn.commit()

    # ---- chunks ----------------------------------------------------------------

    def upsert_chunk(self, chunk_id: str, shot_id: str, status: str,
                     spec: Dict[str, Any], attempt: int = 0,
                     output_path: Optional[str] = None) -> None:
        self._conn.execute(
            "INSERT INTO chunks (chunk_id, shot_id, status, attempt, "
            "output_path, spec_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET status=excluded.status, "
            "attempt=excluded.attempt, output_path=excluded.output_path, "
            "spec_json=excluded.spec_json, updated_at=excluded.updated_at",
            (chunk_id, shot_id, status, attempt, output_path,
             json.dumps(spec), time.time()))
        self._conn.commit()

    def chunk_status(self, chunk_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT status FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return row[0] if row else None

    def chunks_by_status(self, status: str) -> List[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE status=? ORDER BY chunk_id",
            (status,)).fetchall()
        return [r[0] for r in rows]

    # ---- quality / repairs ---------------------------------------------------------

    def add_quality_report(self, chunk_id: str, overall: float,
                           report: Dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO quality_reports (chunk_id, overall, report_json, "
            "created_at) VALUES (?, ?, ?, ?)",
            (chunk_id, overall, json.dumps(report), time.time()))
        self._conn.commit()

    def add_repair(self, chunk_id: str, attempt: int, action: str,
                   reason: str = "") -> None:
        self._conn.execute(
            "INSERT INTO repairs (chunk_id, attempt, action, reason, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, attempt, action, reason, time.time()))
        self._conn.commit()

    def repair_history(self, chunk_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT attempt, action, reason, created_at FROM repairs "
            "WHERE chunk_id=? ORDER BY id", (chunk_id,)).fetchall()
        return [{"attempt": a, "action": ac, "reason": re, "created_at": t}
                for a, ac, re, t in rows]
