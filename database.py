from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from config import APP_DIR, ensure_dirs


@dataclass
class BarkEvent:
    start_ts: str
    end_ts: str
    duration_sec: float
    avg_confidence: float


@dataclass
class ThunderEvent:
    start_ts: str
    end_ts: str
    duration_sec: float
    avg_confidence: float


class BarkDatabase:
    def __init__(self) -> None:
        ensure_dirs()
        self.path = APP_DIR / "database.sqlite"
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bark_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_ts TEXT NOT NULL,
                end_ts TEXT NOT NULL,
                duration_sec REAL NOT NULL,
                avg_confidence REAL NOT NULL,
                day TEXT NOT NULL,
                hour INTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thunder_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_ts TEXT NOT NULL,
                end_ts TEXT NOT NULL,
                duration_sec REAL NOT NULL,
                avg_confidence REAL NOT NULL,
                day TEXT NOT NULL,
                hour INTEGER NOT NULL
            )
            """
        )
        self.conn.commit()

    def add_event(self, event: BarkEvent) -> None:
        start_dt = datetime.fromisoformat(event.start_ts)
        self.conn.execute(
            """
            INSERT INTO bark_events (start_ts, end_ts, duration_sec, avg_confidence, day, hour)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.start_ts,
                event.end_ts,
                event.duration_sec,
                event.avg_confidence,
                start_dt.date().isoformat(),
                start_dt.hour,
            ),
        )
        self.conn.commit()

    def list_days(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT day FROM bark_events ORDER BY day DESC").fetchall()
        return [r[0] for r in rows]

    def count_for_day(self, day: str) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM bark_events WHERE day = ?", (day,)).fetchone()
        return int(row["c"]) if row else 0

    def events_for_range(self, start_day: str, end_day: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM bark_events WHERE day BETWEEN ? AND ? ORDER BY start_ts",
            (start_day, end_day),
        ).fetchall()

    def delete_day(self, day: str) -> None:
        self.conn.execute("DELETE FROM bark_events WHERE day = ?", (day,))
        self.conn.commit()

    def delete_all(self) -> None:
        self.conn.execute("DELETE FROM bark_events")
        self.conn.commit()

    def add_thunder_event(self, event: ThunderEvent) -> None:
        """Add a thunder event to the database.
        
        Args:
            event: Thunder event to store
        """
        start_dt = datetime.fromisoformat(event.start_ts)
        self.conn.execute(
            """
            INSERT INTO thunder_events (start_ts, end_ts, duration_sec, avg_confidence, day, hour)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.start_ts,
                event.end_ts,
                event.duration_sec,
                event.avg_confidence,
                start_dt.date().isoformat(),
                start_dt.hour,
            ),
        )
        self.conn.commit()

    def list_thunder_days(self) -> list[str]:
        """List all days with thunder events.
        
        Returns:
            List of day strings in ISO format, sorted descending
        """
        rows = self.conn.execute("SELECT DISTINCT day FROM thunder_events ORDER BY day DESC").fetchall()
        return [r[0] for r in rows]

    def count_thunder_for_day(self, day: str) -> int:
        """Count thunder events for a specific day.
        
        Args:
            day: Day string in ISO format
            
        Returns:
            Number of thunder events
        """
        row = self.conn.execute("SELECT COUNT(*) AS c FROM thunder_events WHERE day = ?", (day,)).fetchone()
        return int(row["c"]) if row else 0

    def events_for_thunder_range(self, start_day: str, end_day: str) -> list[sqlite3.Row]:
        """Get all thunder events within date range.
        
        Args:
            start_day: Start date in ISO format
            end_day: End date in ISO format
            
        Returns:
            List of thunder event rows
        """
        return self.conn.execute(
            "SELECT * FROM thunder_events WHERE day BETWEEN ? AND ? ORDER BY start_ts",
            (start_day, end_day),
        ).fetchall()

    def delete_thunder_day(self, day: str) -> None:
        """Delete all thunder events for a specific day.
        
        Args:
            day: Day string in ISO format
        """
        self.conn.execute("DELETE FROM thunder_events WHERE day = ?", (day,))
        self.conn.commit()

    def delete_all_thunder(self) -> None:
        """Delete all thunder events from database."""
        self.conn.execute("DELETE FROM thunder_events")
        self.conn.commit()

    def export_json(self, rows: Iterable[sqlite3.Row], target: Path) -> None:
        target.write_text(json.dumps([dict(r) for r in rows], indent=2))

    def export_csv(self, rows: Iterable[sqlite3.Row], target: Path) -> None:
        rows = list(rows)
        with target.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows([dict(r) for r in rows])

    def close(self) -> None:
        self.conn.close()
