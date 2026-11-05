import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join("data", "app.db")

def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables for user profiles, feedback, and history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # User account details & selected interests
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                selected_topics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Interested / Not Interested feedback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                feedback_type TEXT CHECK(feedback_type IN ('interested', 'not_interested')),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            );
        """)
        
        # Exposure tracking to prevent repeated recommendations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                rank INTEGER,
                reason TEXT,
                batch_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            );
        """)
        conn.commit()

def save_user_topics(user_id: str, username: str, topics: list):
    """Create or update user preferences safely."""
    topics_str = ",".join(topics) if isinstance(topics, list) else str(topics)
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Check if user already exists by user_id OR username
        cursor.execute("SELECT user_id FROM users WHERE user_id = ? OR username = ?", (user_id, username))
        row = cursor.fetchone()
        
        if row:
            # 2. Update existing user's topics
            cursor.execute("""
                UPDATE users 
                SET selected_topics = ? 
                WHERE user_id = ? OR username = ?
            """, (topics_str, user_id, username))
        else:
            # 3. Insert new user
            cursor.execute("""
                INSERT INTO users (user_id, username, selected_topics)
                VALUES (?, ?, ?)
            """, (user_id, username, topics_str))
            
        conn.commit()

def save_feedback(user_id: str, article_id: str, feedback_type: str):
    """Store explicit user feedback on an article."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (user_id, article_id, feedback_type)
            VALUES (?, ?, ?);
        """, (user_id, article_id, feedback_type))
        conn.commit()

def get_user_feedback(user_id: str):
    """Retrieve feedback history for NLP profile vector updates."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT article_id, feedback_type FROM feedback WHERE user_id = ?", (user_id,))
        return [dict(row) for row in cursor.fetchall()]
    
def log_recommended_articles(user_id: str, article_ids: list):
    """
    Saves the list of recommended article IDs so we don't show them again.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        for rank, art_id in enumerate(article_ids, start=1):
            cursor.execute("""
                INSERT INTO recommendation_history (user_id, article_id, rank)
                VALUES (?, ?, ?);
            """, (user_id, str(art_id), rank))
        conn.commit()

def get_seen_article_ids(user_id: str) -> set:
    """
    Retrieves all article IDs previously shown to this user.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT article_id FROM recommendation_history WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        return {row["article_id"] for row in rows}