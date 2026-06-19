"""
Session Manager — manages conversation sessions backed by SQLite.

Stores session data in the unified cliver.db with:
- sessions table: metadata (id, title, timestamps, options)
- turns table: conversation turns (role, content, timestamp)
- turns_fts: FTS5 virtual table for full-text search across all turns
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliver.db import SQLiteStore, get_store
from cliver.messages import CLIverMessage

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    lab_id TEXT,
    kind TEXT DEFAULT 'general',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turn_count INTEGER DEFAULT 0,
    options TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    message_json TEXT,
    timestamp TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    content,
    content=turns,
    content_rowid=id
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO turns_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class SessionManager:
    """Manages conversation sessions stored in SQLite with FTS5 search."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._store: Optional[SQLiteStore] = None

    def _get_store(self):
        if self._store is None:
            self._store = get_store(self._db_path)
            self._store.execute_schema(_SCHEMA)
        return self._store

    # -- Session lifecycle -----------------------------------------------------

    def create_session(self, title: str = "", options: dict | None = None, kind: str = "general") -> str:
        """Create a new session. Returns session_id.

        Options (agent, system_message, filter_tools, etc.) are saved as JSON
        in the options column so the session's full configuration is immediately
        available for reload/fork without re-sending from the client.
        """
        session_id = str(uuid.uuid4())[:8]
        now = _timestamp()
        with self._get_store().write() as db:
            db.execute(
                "INSERT INTO sessions (id, title, kind, options, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    title or None,
                    kind,
                    json.dumps(options) if options else None,
                    now,
                    now,
                ),
            )
        return session_id

    def create_lab_session(self, lab_id: str, title: str = "", options: dict | None = None, kind: str = "lab") -> str:
        """Create a new session linked to a lab. Returns session_id."""
        session_id = str(uuid.uuid4())[:8]
        now = _timestamp()
        with self._get_store().write() as db:
            db.execute(
                "INSERT INTO sessions (id, title, lab_id, kind, options, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, title or None, lab_id, kind, json.dumps(options) if options else None, now, now),
            )
        return session_id

    def list_lab_sessions(self, lab_id: str) -> List[Dict[str, Any]]:
        """List sessions belonging to a lab, most recent first."""
        with self._get_store().read() as db:
            rows = db.execute(
                "SELECT id, title, kind, created_at, updated_at, turn_count, options FROM sessions "
                "WHERE lab_id = ? ORDER BY updated_at DESC",
                (lab_id,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with metadata, most recent first."""
        with self._get_store().read() as db:
            rows = db.execute(
                "SELECT id, title, kind, created_at, updated_at, turn_count, options "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_general_sessions(self) -> List[Dict[str, Any]]:
        """List chat sessions (kind='chat'), most recent first."""
        with self._get_store().read() as db:
            rows = db.execute(
                "SELECT id, title, kind, created_at, updated_at, turn_count, options "
                "FROM sessions WHERE kind = 'chat' ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its turns (CASCADE)."""
        with self._get_store().write() as db:
            cursor = db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    # -- Conversation recording ------------------------------------------------

    def append_turn(self, session_id: str, role: str, content: str, *, message: Any = None) -> None:
        """Append a conversation turn to a session.

        ``message`` is an optional ``CLIverMessage`` whose full JSON is
        persisted alongside the plain-text columns.  When provided, the
        session can be fully reconstructed (tool calls, thinking blocks, etc.).
        """
        message_json = None
        if message is not None:
            try:
                message_json = message.model_dump_json()
            except Exception:
                pass

        now = _timestamp()
        with self._get_store().write() as db:
            db.execute(
                "INSERT INTO turns (session_id, role, content, message_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, message_json, now),
            )
            db.execute(
                "UPDATE sessions SET updated_at = ?, turn_count = turn_count + 1 WHERE id = ?",
                (now, session_id),
            )
            # Auto-set title from first user message
            if role == "user":
                db.execute(
                    "UPDATE sessions SET title = ? WHERE id = ? AND (title IS NULL OR title = '')",
                    (content[:80], session_id),
                )

    def load_turns(self, session_id: str) -> List[Dict[str, str]]:
        """Load all conversation turns from a session.

        Each turn dict has ``role``, ``content``, ``timestamp``, and
        optionally ``message`` (a ``CLIverMessage`` reconstructed from JSON).
        """
        with self._get_store().read() as db:
            rows = db.execute(
                "SELECT role, content, message_json, timestamp FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        result = []
        for r in rows:
            turn = {"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
            if r["message_json"]:
                try:
                    turn["message"] = CLIverMessage.model_validate_json(r["message_json"]).model_dump(exclude_none=True)
                except Exception:
                    pass
            result.append(turn)
        return result

    def trim_turns(self, session_id: str, keep_last: int = 50) -> int:
        """Delete older turns, keeping only the most recent ones.

        Returns the number of turns deleted.
        """
        with self._get_store().write() as db:
            cursor = db.execute(
                """DELETE FROM turns WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?
                )""",
                (session_id, session_id, keep_last),
            )
            deleted = cursor.rowcount
            if deleted > 0:
                db.execute(
                    "UPDATE sessions SET turn_count = (SELECT COUNT(*) FROM turns WHERE session_id = ?) WHERE id = ?",
                    (session_id, session_id),
                )
        return deleted

    def delete_oldest_sessions(self, keep: int = 300) -> int:
        """Delete oldest sessions when total count exceeds keep limit.

        Returns the number of sessions deleted.
        """
        with self._get_store().write() as db:
            total = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            if total <= keep:
                return 0
            cursor = db.execute(
                """DELETE FROM sessions WHERE id IN (
                    SELECT id FROM sessions ORDER BY updated_at ASC LIMIT ?
                )""",
                (total - keep,),
            )
            deleted = cursor.rowcount
        return deleted

    def delete_stale_sessions(self, max_age_days: int = 90) -> int:
        """Delete sessions not updated within max_age_days.

        Returns the number of sessions deleted.
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        with self._get_store().write() as db:
            cursor = db.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
            deleted = cursor.rowcount
        return deleted

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a session."""
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, title, lab_id, kind, created_at, updated_at, turn_count, options "
                "FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    # -- Options ---------------------------------------------------------------

    def save_options(self, session_id: str, options: Dict[str, Any]) -> None:
        """Persist session options as JSON — full replacement, keeps nulls."""
        clean = {k: v for k, v in options.items() if k != "statusbar"}
        with self._get_store().write() as db:
            db.execute(
                "UPDATE sessions SET options = ? WHERE id = ?",
                (json.dumps(clean, ensure_ascii=False), session_id),
            )

    def merge_options(self, session_id: str, patch: Dict[str, Any]) -> None:
        """Merge fields into existing options in one transaction.

        Reads current options, applies patch fields, writes back atomically.
        Fields set to ``null`` in the patch are removed from options.
        """
        with self._get_store().write() as db:
            row = db.execute("SELECT options FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return
            current: dict = {}
            if row["options"]:
                try:
                    current = json.loads(row["options"])
                except (json.JSONDecodeError, TypeError):
                    pass
            for k, v in patch.items():
                if k == "statusbar":
                    continue
                if v is None:
                    current.pop(k, None)
                else:
                    current[k] = v
            db.execute(
                "UPDATE sessions SET options = ? WHERE id = ?",
                (json.dumps(current, ensure_ascii=False) if current else None, session_id),
            )

    def load_options(self, session_id: str) -> Dict[str, Any]:
        """Load persisted session options."""
        with self._get_store().read() as db:
            row = db.execute("SELECT options FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None or row["options"] is None:
            return {}
        try:
            return json.loads(row["options"])
        except (json.JSONDecodeError, TypeError):
            return {}

    # -- Title -----------------------------------------------------------------

    def update_title(self, session_id: str, title: str) -> None:
        """Update the title of a session."""
        with self._get_store().write() as db:
            db.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))

    # -- Search ----------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search across all session turns.

        Returns list of dicts with session metadata and matching snippets,
        grouped by session, ordered by relevance.
        """
        with self._get_store().read() as db:
            # Use FTS5 match to find matching turns with snippets
            rows = db.execute(
                """
                SELECT
                    t.session_id,
                    t.role,
                    snippet(turns_fts, 0, '**', '**', '...', 32) AS snippet,
                    s.title,
                    s.created_at,
                    s.turn_count
                FROM turns_fts
                JOIN turns t ON t.id = turns_fts.rowid
                JOIN sessions s ON s.id = t.session_id
                WHERE turns_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit * 5),  # fetch more rows, then group by session
            ).fetchall()

        # Group by session
        sessions: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            sid = row["session_id"]
            if sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "turn_count": row["turn_count"],
                    "snippets": [],
                }
            sessions[sid]["snippets"].append(
                {
                    "role": row["role"],
                    "content": row["snippet"],
                }
            )

        result = list(sessions.values())
        return result[:limit]

    def close(self) -> None:
        """Close the database connection."""
        if self._store:
            self._store.close()
            self._store = None


def _row_to_dict(row: Any) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict, parsing options JSON."""
    d = dict(row)
    if "options" in d and d["options"]:
        try:
            d["options"] = json.loads(d["options"])
        except (json.JSONDecodeError, TypeError):
            d["options"] = {}
    return d


def _timestamp() -> str:
    from cliver.util import format_datetime

    return format_datetime()
