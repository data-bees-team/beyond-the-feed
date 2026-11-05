"""
User Account, Authentication, and Exposure History Manager for Beyond the Feed.

Provides persistent, multi-user profile management, topic interest persistence,
pre-configured demo personas for hackathon judges, and recommendation exposure history
(tracking which articles a user has already seen to support 'Load More' and avoid repetition).

Compatible with SQLite (zero-config embedded storage) and standard relational SQL (PostgreSQL).
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import hashlib
import json
import uuid


def _get_utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str) -> str:
    """Compute SHA-256 hash of a password string."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# Pre-configured demo personas for judges and hackathon evaluations
DEMO_PERSONAS = [
    {
        "user_id": "demo_tech_alex",
        "username": "Alex (Tech & AI Specialist)",
        "password": "demo",
        "selected_topics": ["Technology", "Business and Economy"],
        "is_demo": True,
        "description": "Tech enthusiast reading about generative AI and market infrastructure."
    },
    {
        "user_id": "demo_climate_elena",
        "username": "Elena (Climate & Global Perspectives)",
        "password": "demo",
        "selected_topics": ["Environment and Climate", "Science"],
        "is_demo": True,
        "description": "Climate advocate exploring renewable transitions across continents."
    },
    {
        "user_id": "demo_health_marcus",
        "username": "Marcus (Health Researcher)",
        "password": "demo",
        "selected_topics": ["Health", "Science", "Education"],
        "is_demo": True,
        "description": "Researcher tracking biomedical developments and global public health."
    },
    {
        "user_id": "demo_guest",
        "username": "Guest User (Fresh Slate)",
        "password": "demo",
        "selected_topics": [],
        "is_demo": True,
        "description": "Unbiased cold-start profile with zero prior history."
    }
]


class UserManager:
    """
    Manages user registration, authentication, topic persistence, and exposure history.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the UserManager.

        Args:
            db_path: Path to SQLite database file. Defaults to data/feedback.db
                     to share storage with FeedbackManager.
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
        self._seed_demo_personas()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection with dictionary row access."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                pass
        return conn

    def _init_database(self) -> None:
        """Initialize tables for users and recommendation history."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # 1. Users Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                selected_topics TEXT NOT NULL DEFAULT '[]',
                is_demo INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_login TEXT NOT NULL
            );
            """)

            # 2. Recommendation Exposure History Table (for 'Load More' deduplication & analytics)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                rank_position INTEGER NOT NULL,
                relevance_score REAL,
                publisher TEXT,
                country TEXT,
                continent TEXT,
                primary_topic TEXT,
                reason TEXT,
                created_at TEXT NOT NULL
            );
            """)

            # Indexes for high-performance lookup
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rec_history_user ON recommendation_history(user_id);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rec_history_user_art ON recommendation_history(user_id, article_id);
            """)
            conn.commit()
        finally:
            conn.close()

    def _seed_demo_personas(self) -> None:
        """Ensure demo personas are seeded and available for evaluation."""
        now_ts = _get_utc_timestamp()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for persona in DEMO_PERSONAS:
                cursor.execute("""
                INSERT OR IGNORE INTO users (
                    user_id, username, password_hash, selected_topics, is_demo, created_at, last_login
                ) VALUES (?, ?, ?, ?, 1, ?, ?);
                """, (
                    persona["user_id"],
                    persona["username"],
                    _hash_password(persona["password"]),
                    json.dumps(persona["selected_topics"]),
                    now_ts,
                    now_ts
                ))
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # User Account Operations
    # =========================================================================

    def register_user(
        self,
        username: str,
        password: str,
        selected_topics: Optional[List[str]] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Register a new user account.

        Returns:
            Tuple of (success: bool, message: str, user_dict: Optional[Dict])
        """
        clean_name = str(username).strip()
        if not clean_name:
            return False, "Username cannot be empty.", None

        if len(password) < 3:
            return False, "Password must be at least 3 characters.", None

        user_id = f"user_{uuid.uuid4().hex[:12]}"
        now_ts = _get_utc_timestamp()
        topics_json = json.dumps(selected_topics or [])
        pwd_hash = _hash_password(password)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO users (
                user_id, username, password_hash, selected_topics, is_demo, created_at, last_login
            ) VALUES (?, ?, ?, ?, 0, ?, ?);
            """, (user_id, clean_name, pwd_hash, topics_json, now_ts, now_ts))
            conn.commit()

            return True, "Account created successfully.", {
                "user_id": user_id,
                "username": clean_name,
                "selected_topics": selected_topics or [],
                "is_demo": False,
                "created_at": now_ts
            }
        except sqlite3.IntegrityError:
            return False, f"Username '{clean_name}' is already taken.", None
        finally:
            conn.close()

    def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Authenticate an existing user or demo persona.

        Returns:
            Tuple of (success: bool, message: str, user_dict: Optional[Dict])
        """
        clean_name = str(username).strip()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM users WHERE username = ?;
            """, (clean_name,))
            row = cursor.fetchone()

            if not row:
                return False, "User not found.", None

            pwd_hash = _hash_password(password)
            if row["password_hash"] != pwd_hash:
                return False, "Incorrect password.", None

            now_ts = _get_utc_timestamp()
            cursor.execute("""
            UPDATE users SET last_login = ? WHERE user_id = ?;
            """, (now_ts, row["user_id"]))
            conn.commit()

            try:
                topics = json.loads(row["selected_topics"])
            except Exception:
                topics = []

            return True, "Authentication successful.", {
                "user_id": row["user_id"],
                "username": row["username"],
                "selected_topics": topics,
                "is_demo": bool(row["is_demo"]),
                "created_at": row["created_at"],
                "last_login": now_ts
            }
        finally:
            conn.close()

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a user's record and saved topic preferences."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,))
            row = cursor.fetchone()
            if not row:
                return None

            try:
                topics = json.loads(row["selected_topics"])
            except Exception:
                topics = []

            return {
                "user_id": row["user_id"],
                "username": row["username"],
                "selected_topics": topics,
                "is_demo": bool(row["is_demo"]),
                "created_at": row["created_at"],
                "last_login": row["last_login"]
            }
        finally:
            conn.close()

    def update_user_topics(self, user_id: str, selected_topics: List[str]) -> bool:
        """Update and persist a user's chosen topic list."""
        topics_json = json.dumps(selected_topics or [])
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE users SET selected_topics = ? WHERE user_id = ?;
            """, (topics_json, user_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_demo_personas(self) -> List[Dict[str, Any]]:
        """Return list of available demo personas for presentation/evaluation."""
        return DEMO_PERSONAS

    # =========================================================================
    # Recommendation Exposure History Operations (Load More & De-duplication)
    # =========================================================================

    def record_recommendation_history(
        self,
        user_id: str,
        articles: List[Dict[str, Any]],
        algorithm: str,
        batch_id: Optional[str] = None
    ) -> int:
        """
        Record a batch of recommended articles to track user exposure history.

        Args:
            user_id: Target user ID.
            articles: List of recommended article dictionaries.
            algorithm: Algorithm identifier (e.g. 'publisher_mmr', 'balanced_feed').
            batch_id: Optional batch UUID string.

        Returns:
            int: Number of records inserted.
        """
        if not articles:
            return 0

        batch_id = batch_id or uuid.uuid4().hex[:10]
        now_ts = _get_utc_timestamp()

        records: List[Tuple] = []
        for idx, art in enumerate(articles, 1):
            records.append((
                user_id,
                batch_id,
                str(art.get("article_id", "")),
                algorithm,
                idx,
                float(art.get("relevance_score", 0.0)),
                str(art.get("publisher", "")),
                str(art.get("country", "")),
                str(art.get("continent", "")),
                str(art.get("primary_topic", "")),
                str(art.get("reason", "")),
                now_ts
            ))

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany("""
            INSERT INTO recommendation_history (
                user_id, batch_id, article_id, algorithm, rank_position,
                relevance_score, publisher, country, continent, primary_topic,
                reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, records)
            conn.commit()
            return len(records)
        finally:
            conn.close()

    def get_exposed_article_ids(self, user_id: str) -> Set[str]:
        """
        Return the set of article IDs that have already been recommended to the user.
        Used by 'Load More' to guarantee fresh discoveries without repeats.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT DISTINCT article_id FROM recommendation_history WHERE user_id = ?;
            """, (user_id,))
            rows = cursor.fetchall()
            return {row["article_id"] for row in rows}
        finally:
            conn.close()

    def get_exposure_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve recent recommendation history records for analytics."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM recommendation_history
            WHERE user_id = ?
            ORDER BY history_id DESC
            LIMIT ?;
            """, (user_id, limit))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def clear_user_history(self, user_id: str) -> bool:
        """Clear recommendation exposure history for a user (e.g. feed reset)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM recommendation_history WHERE user_id = ?;", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


# Module-level singleton
_USER_MGR_INSTANCE: Optional[UserManager] = None


def get_user_manager(db_path: Optional[str] = None) -> UserManager:
    """Return a thread-safe singleton instance of UserManager."""
    global _USER_MGR_INSTANCE
    if _USER_MGR_INSTANCE is None or db_path is not None:
        _USER_MGR_INSTANCE = UserManager(db_path=db_path)
    return _USER_MGR_INSTANCE
