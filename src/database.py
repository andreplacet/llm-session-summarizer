import sqlite3
import os
from datetime import datetime
from typing import Optional

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "sessions.db")


class Database:

    def __init__(self, path: Optional[str] = None):
        self.path = path or DB_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    filenames TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS summaries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'summary',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS provider_keys (
                    provider TEXT PRIMARY KEY,
                    encrypted_key TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Migration: add type column if missing (for databases created before this field)
            try:
                conn.execute("ALTER TABLE summaries ADD COLUMN type TEXT NOT NULL DEFAULT 'summary'")
            except sqlite3.OperationalError:
                pass  # column already exists

    def save_session(
        self,
        id: str,
        title: str,
        source: str,
        provider: str,
        filenames: str,
        message_count: int,
    ):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions (id, title, source, provider, filenames, message_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (id, title, source, provider, filenames, message_count, datetime.utcnow()),
            )

    def save_summary(self, id: str, session_id: str, content: str, model_used: str, type: str = "summary"):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO summaries (id, session_id, content, model_used, type, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (id, session_id, content, model_used, type, datetime.utcnow()),
            )

    def get_all_sessions(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()

    def get_summary(self, session_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM summaries WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return row["content"] if row else None

    def delete_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # --- Key storage ---

    def save_encrypted_key(self, provider: str, encrypted_key: str):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO provider_keys (provider, encrypted_key, created_at, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                (provider, encrypted_key),
            )

    def get_encrypted_key(self, provider: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT encrypted_key FROM provider_keys WHERE provider = ?",
                (provider,),
            ).fetchone()
            return row["encrypted_key"] if row else None

    def has_encrypted_key(self, provider: str) -> bool:
        return self.get_encrypted_key(provider) is not None

    def delete_encrypted_key(self, provider: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM provider_keys WHERE provider = ?", (provider,))
