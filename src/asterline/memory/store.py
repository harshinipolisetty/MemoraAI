"""SQLite storage engine for conversations, messages, user memories, and audit logs.

Ensures strict user-level isolation, ACID transactional consistency,
foreign key enforcement, and parameterized SQL queries.
"""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional


class SQLiteStore:
    """Manages SQLite database operations, schema initialization, and connections."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys and Row factory enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an explicit atomic transaction."""
        conn = self.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Initialize database tables and indexes safely (idempotent, never destroys data)."""
        with self.get_connection() as conn:
            conn.executescript(
                """
                -- Conversations Table
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_user 
                    ON conversations(user_id);

                -- Messages Table
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    citations_json TEXT,
                    route_taken TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                    ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_user 
                    ON messages(user_id);

                -- User Memories Table
                CREATE TABLE IF NOT EXISTS user_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL CHECK(category IN ('preference', 'workspace_context', 'project_role', 'investigation_note', 'general')),
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source_message_id TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_user_memories_user 
                    ON user_memories(user_id, is_active);

                -- Memory Audit Log
                CREATE TABLE IF NOT EXISTS memory_audit_log (
                    audit_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('CREATE', 'UPDATE', 'DELETE', 'PURGE')),
                    previous_summary TEXT,
                    reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_memory_audit_user 
                    ON memory_audit_log(user_id, timestamp);
                """
            )

    # -------------------------------------------------------------------------
    # Conversation Operations (Scoped by user_id)
    # -------------------------------------------------------------------------

    def create_conversation(self, conversation_id: str, user_id: str, title: str) -> Dict[str, Any]:
        """Create a new conversation thread owned by user_id."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, user_id, title, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (conversation_id, user_id, title),
            )
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return dict(row)

    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a conversation by ID, enforcing user ownership."""
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM conversations 
                WHERE conversation_id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            ).fetchone()
            return dict(row) if row else None

    def list_conversations(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List conversations owned by user_id, ordered by recent update."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversations 
                WHERE user_id = ? 
                ORDER BY updated_at DESC 
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # Message Operations (Scoped by user_id)
    # -------------------------------------------------------------------------

    def add_message(
        self,
        message_id: str,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        citations_json: Optional[str] = None,
        route_taken: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a message to a conversation owned by user_id in an atomic transaction."""
        with self.transaction() as conn:
            # Verify conversation exists and is owned by user
            conv = conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if not conv:
                raise PermissionError(f"Conversation '{conversation_id}' not found or access denied.")

            conn.execute(
                """
                INSERT INTO messages (message_id, conversation_id, user_id, role, content, citations_json, route_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, user_id, role, content, citations_json, route_taken),
            )
            # Touch parent conversation updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                (conversation_id,),
            )

            row = conn.execute(
                "SELECT * FROM messages WHERE message_id = ?", (message_id,)
            ).fetchone()
            return dict(row)

    def get_messages(self, conversation_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a conversation in chronological order, verifying user ownership."""
        with self.get_connection() as conn:
            # Verify conversation exists and is owned by user
            conv = conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if not conv:
                return []

            rows = conn.execute(
                """
                SELECT * FROM messages 
                WHERE conversation_id = ? AND user_id = ? 
                ORDER BY created_at ASC
                """,
                (conversation_id, user_id),
            ).fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # Memory Operations (Scoped by user_id)
    # -------------------------------------------------------------------------

    def insert_memory_with_audit(
        self,
        memory_id: str,
        user_id: str,
        category: str,
        content: str,
        audit_id: str,
        confidence: float = 1.0,
        source_message_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert a new memory and create its audit record in a single atomic transaction."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO user_memories (
                    memory_id, user_id, category, content, confidence, source_message_id, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (memory_id, user_id, category, content, confidence, source_message_id),
            )
            conn.execute(
                """
                INSERT INTO memory_audit_log (
                    audit_id, user_id, memory_id, action, previous_summary, reason
                ) VALUES (?, ?, ?, 'CREATE', NULL, ?)
                """,
                (audit_id, user_id, memory_id, reason or "Initial memory creation"),
            )
            row = conn.execute(
                "SELECT * FROM user_memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            return dict(row)

    def update_memory_with_audit(
        self,
        memory_id: str,
        user_id: str,
        new_content: str,
        audit_id: str,
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an active memory owned by user_id with audit logging in one transaction."""
        with self.transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM user_memories 
                WHERE memory_id = ? AND user_id = ? AND is_active = 1
                """,
                (memory_id, user_id),
            ).fetchone()
            if not existing:
                return None

            # Non-sensitive summary of previous content for audit trail
            prev_summary = f"Chars: {len(existing['content'])}, Category: {existing['category']}"

            conn.execute(
                """
                UPDATE user_memories 
                SET content = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE memory_id = ? AND user_id = ?
                """,
                (new_content, memory_id, user_id),
            )
            conn.execute(
                """
                INSERT INTO memory_audit_log (
                    audit_id, user_id, memory_id, action, previous_summary, reason
                ) VALUES (?, ?, ?, 'UPDATE', ?, ?)
                """,
                (audit_id, user_id, memory_id, prev_summary, reason or "User memory updated"),
            )
            updated = conn.execute(
                "SELECT * FROM user_memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            return dict(updated)

    def delete_memory_with_audit(
        self,
        memory_id: str,
        user_id: str,
        audit_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Deactivate a memory owned by user_id and record audit event in one transaction."""
        with self.transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM user_memories 
                WHERE memory_id = ? AND user_id = ? AND is_active = 1
                """,
                (memory_id, user_id),
            ).fetchone()
            if not existing:
                return False

            prev_summary = f"Category: {existing['category']}, Length: {len(existing['content'])}"

            conn.execute(
                """
                UPDATE user_memories 
                SET is_active = 0, updated_at = CURRENT_TIMESTAMP 
                WHERE memory_id = ? AND user_id = ?
                """,
                (memory_id, user_id),
            )
            conn.execute(
                """
                INSERT INTO memory_audit_log (
                    audit_id, user_id, memory_id, action, previous_summary, reason
                ) VALUES (?, ?, ?, 'DELETE', ?, ?)
                """,
                (audit_id, user_id, memory_id, prev_summary, reason or "User requested deletion (forget)"),
            )
            return True

    def get_memory(self, memory_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an active memory by ID and user_id."""
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM user_memories 
                WHERE memory_id = ? AND user_id = ? AND is_active = 1
                """,
                (memory_id, user_id),
            ).fetchone()
            return dict(row) if row else None

    def list_active_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """List all active memories for a given user."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM user_memories 
                WHERE user_id = ? AND is_active = 1 
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_audit_events(self, user_id: str) -> List[Dict[str, Any]]:
        """List audit events for a given user."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_audit_log 
                WHERE user_id = ? 
                ORDER BY timestamp DESC
                """,
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
