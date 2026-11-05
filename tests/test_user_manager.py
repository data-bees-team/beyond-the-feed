"""
Unit Tests for UserManager (User Accounts, Demo Personas, Topic Persistence, Exposure History).
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path

from utils.user_manager import UserManager, _hash_password, DEMO_PERSONAS


class TestUserManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_users.db")
        self.mgr = UserManager(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_demo_personas_seeded(self):
        """Verify standard demo personas are automatically seeded."""
        for persona in DEMO_PERSONAS:
            ok, msg, user = self.mgr.authenticate_user(persona["username"], persona["password"])
            self.assertTrue(ok, f"Failed to authenticate demo persona: {persona['username']}")
            self.assertEqual(user["username"], persona["username"])
            self.assertEqual(user["selected_topics"], persona["selected_topics"])
            self.assertTrue(user["is_demo"])

    def test_user_registration_and_authentication(self):
        """Test creating a new account and authenticating."""
        ok, msg, user = self.mgr.register_user("testuser", "securepass123", ["Technology"])
        self.assertTrue(ok)
        self.assertEqual(user["username"], "testuser")
        self.assertEqual(user["selected_topics"], ["Technology"])

        # Authenticate with correct password
        auth_ok, auth_msg, auth_user = self.mgr.authenticate_user("testuser", "securepass123")
        self.assertTrue(auth_ok)
        self.assertEqual(auth_user["user_id"], user["user_id"])

        # Authenticate with wrong password
        fail_ok, fail_msg, fail_user = self.mgr.authenticate_user("testuser", "wrongpassword")
        self.assertFalse(fail_ok)
        self.assertIsNone(fail_user)

    def test_duplicate_username_rejected(self):
        """Test registering with an existing username fails gracefully."""
        self.mgr.register_user("unique_name", "pass1", ["Health"])
        ok, msg, user = self.mgr.register_user("unique_name", "pass2", ["Science"])
        self.assertFalse(ok)
        self.assertIn("already taken", msg)

    def test_update_user_topics(self):
        """Test updating a user's selected interests."""
        ok, _, user = self.mgr.register_user("topic_updater", "pass123", ["Technology"])
        uid = user["user_id"]

        # Update topics
        new_topics = ["Health", "Environment and Climate"]
        update_ok = self.mgr.update_user_topics(uid, new_topics)
        self.assertTrue(update_ok)

        # Retrieve profile
        profile = self.mgr.get_user_profile(uid)
        self.assertEqual(profile["selected_topics"], new_topics)

    def test_exposure_history_and_deduplication(self):
        """Test recording recommendation exposure and retrieving exposed IDs."""
        uid = "user_exposure_test"
        sample_articles = [
            {"article_id": "art_101", "relevance_score": 0.8, "publisher": "bbc.com", "country": "UK"},
            {"article_id": "art_102", "relevance_score": 0.7, "publisher": "reuters.com", "country": "US"}
        ]

        # Record first batch
        count = self.mgr.record_recommendation_history(uid, sample_articles, algorithm="publisher_mmr")
        self.assertEqual(count, 2)

        # Check exposed article IDs
        exposed = self.mgr.get_exposed_article_ids(uid)
        self.assertEqual(exposed, {"art_101", "art_102"})

        # Record second batch
        batch_2 = [{"article_id": "art_103", "relevance_score": 0.9}]
        self.mgr.record_recommendation_history(uid, batch_2, algorithm="publisher_mmr")

        # Check union
        exposed_updated = self.mgr.get_exposed_article_ids(uid)
        self.assertEqual(exposed_updated, {"art_101", "art_102", "art_103"})

    def test_multi_user_isolation(self):
        """Verify user A's exposure history does not leak into user B."""
        self.mgr.record_recommendation_history("user_A", [{"article_id": "art_A1"}], "geo_mmr")
        self.mgr.record_recommendation_history("user_B", [{"article_id": "art_B1"}], "geo_mmr")

        self.assertEqual(self.mgr.get_exposed_article_ids("user_A"), {"art_A1"})
        self.assertEqual(self.mgr.get_exposed_article_ids("user_B"), {"art_B1"})


if __name__ == "__main__":
    unittest.main()
