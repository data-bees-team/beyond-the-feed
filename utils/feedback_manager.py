"""
Centralized User Feedback Management for Beyond the Feed.

Provides persistent, multi-user, SQLite-backed storage for user interactions
(👍 Interested / 👎 Not Interested) and recommendation impressions.
Serves as the shared data layer for adaptive personalization and future
contextual bandit / reinforcement learning.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import json
import os


# Valid feedback types
FEEDBACK_INTERESTED = "interested"
FEEDBACK_NOT_INTERESTED = "not_interested"
VALID_FEEDBACK_TYPES = {FEEDBACK_INTERESTED, FEEDBACK_NOT_INTERESTED}


def _get_utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class FeedbackManager:
    """
    Thread-safe SQLite manager for user feedback and recommendation impressions.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the feedback manager with an SQLite database.

        Args:
            db_path: Path to SQLite database file. Defaults to data/feedback.db.
                     Use ':memory:' for isolated automated testing.
        """
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "feedback.db")
        else:
            self.db_path = str(db_path)
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection with dictionary-like row factory."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                pass
        return conn

    def _init_database(self) -> None:
        """Create tables and indexes if they do not already exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                title TEXT,
                publisher TEXT,
                country TEXT,
                continent TEXT,
                topics TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, article_id)
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_impressions (
                impression_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                rank_position INTEGER NOT NULL,
                relevance_score REAL,
                timestamp TEXT NOT NULL
            );
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback(user_id);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_user_art ON user_feedback(user_id, article_id);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_impressions_user ON recommendation_impressions(user_id);
            """)
            conn.commit()
        finally:
            conn.close()

    def record_feedback(
        self,
        user_id: str,
        article_id: str,
        feedback_type: str,
        article_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record or update feedback for a user and article."""
        if feedback_type not in VALID_FEEDBACK_TYPES:
            raise ValueError(
                f"Invalid feedback_type '{feedback_type}'. Must be one of: {VALID_FEEDBACK_TYPES}"
            )

        article_metadata = article_metadata or {}
        title = str(article_metadata.get("title", ""))
        publisher = str(article_metadata.get("publisher", ""))
        country = str(article_metadata.get("country", ""))
        continent = str(article_metadata.get("continent", ""))

        raw_topics = article_metadata.get("topics") or article_metadata.get("topic_list", [])
        if isinstance(raw_topics, (list, tuple, set)):
            topics_str = "; ".join(str(t).strip() for t in raw_topics if str(t).strip())
        else:
            topics_str = str(raw_topics).strip()

        now_ts = _get_utc_timestamp()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO user_feedback (
                user_id, article_id, feedback_type, title, publisher, country,
                continent, topics, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, article_id) DO UPDATE SET
                feedback_type = excluded.feedback_type,
                title = CASE WHEN excluded.title != '' THEN excluded.title ELSE user_feedback.title END,
                publisher = CASE WHEN excluded.publisher != '' THEN excluded.publisher ELSE user_feedback.publisher END,
                country = CASE WHEN excluded.country != '' THEN excluded.country ELSE user_feedback.country END,
                continent = CASE WHEN excluded.continent != '' THEN excluded.continent ELSE user_feedback.continent END,
                topics = CASE WHEN excluded.topics != '' THEN excluded.topics ELSE user_feedback.topics END,
                updated_at = excluded.updated_at;
            """, (
                user_id, article_id, feedback_type, title, publisher,
                country, continent, topics_str, now_ts, now_ts
            ))
            conn.commit()
            return True
        finally:
            conn.close()

    def remove_feedback(self, user_id: str, article_id: str) -> bool:
        """Remove feedback for a specific user and article."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            DELETE FROM user_feedback WHERE user_id = ? AND article_id = ?;
            """, (user_id, article_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_user_feedback(self, user_id: str) -> Dict[str, str]:
        """Retrieve all current feedback for a given user as a dictionary."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT article_id, feedback_type FROM user_feedback WHERE user_id = ?;
            """, (user_id,))
            rows = cursor.fetchall()
            return {row["article_id"]: row["feedback_type"] for row in rows}
        finally:
            conn.close()

    def get_feedback_records(
        self,
        user_id: Optional[str] = None,
        feedback_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve full feedback records with metadata."""
        query = "SELECT * FROM user_feedback"
        params: List[Any] = []
        conditions: List[str] = []

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        if feedback_type is not None:
            conditions.append("feedback_type = ?")
            params.append(feedback_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY updated_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            records = []
            for row in rows:
                topics_raw = row["topics"] or ""
                topic_list = [t.strip() for t in topics_raw.split(";") if t.strip()]
                records.append({
                    "feedback_id": row["feedback_id"],
                    "user_id": row["user_id"],
                    "article_id": row["article_id"],
                    "feedback_type": row["feedback_type"],
                    "title": row["title"] or "",
                    "publisher": row["publisher"] or "",
                    "country": row["country"] or "",
                    "continent": row["continent"] or "",
                    "topics": topic_list,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                })
            return records
        finally:
            conn.close()

    def get_feedback_summary(self, user_id: str) -> Dict[str, Any]:
        """Compute an aggregate summary of user feedback interactions."""
        records = self.get_feedback_records(user_id=user_id)

        interested_count = sum(1 for r in records if r["feedback_type"] == FEEDBACK_INTERESTED)
        not_interested_count = sum(1 for r in records if r["feedback_type"] == FEEDBACK_NOT_INTERESTED)

        publishers = {r["publisher"] for r in records if r["publisher"]}
        countries = {r["country"] for r in records if r["country"]}
        topics = set()
        for r in records:
            topics.update(r["topics"])

        return {
            "user_id": user_id,
            "total_count": len(records),
            "interested_count": interested_count,
            "not_interested_count": not_interested_count,
            "unique_publishers": len(publishers),
            "unique_countries": len(countries),
            "unique_topics": len(topics)
        }

    def clear_user_feedback(self, user_id: str) -> int:
        """Clear all feedback history for a specific user (Reset Profile)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_feedback WHERE user_id = ?;", (user_id,))
            deleted_fb = cursor.rowcount
            cursor.execute("DELETE FROM recommendation_impressions WHERE user_id = ?;", (user_id,))
            conn.commit()
            return deleted_fb
        finally:
            conn.close()

    def log_impressions(
        self,
        user_id: str,
        algorithm: str,
        articles: List[Dict[str, Any]]
    ) -> None:
        """Log recommendation slate impressions for future contextual bandit analysis."""
        now_ts = _get_utc_timestamp()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            rows_to_insert = []
            for rank, art in enumerate(articles, start=1):
                article_id = str(art.get("article_id", ""))
                relevance = float(art.get("relevance_score", 0.0))
                rows_to_insert.append((user_id, article_id, algorithm, rank, relevance, now_ts))

            cursor.executemany("""
            INSERT INTO recommendation_impressions (
                user_id, article_id, algorithm, rank_position, relevance_score, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?);
            """, rows_to_insert)
            conn.commit()
        finally:
            conn.close()


# Shared singleton instance for the Streamlit application
_GLOBAL_FEEDBACK_MANAGER: Optional[FeedbackManager] = None


def get_feedback_manager() -> FeedbackManager:
    """Return or initialize the singleton FeedbackManager instance."""
    global _GLOBAL_FEEDBACK_MANAGER
    if _GLOBAL_FEEDBACK_MANAGER is None:
        _GLOBAL_FEEDBACK_MANAGER = FeedbackManager()
    return _GLOBAL_FEEDBACK_MANAGER
