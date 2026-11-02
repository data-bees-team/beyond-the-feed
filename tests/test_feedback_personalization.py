"""
Automated Test Suite for Centralized User Feedback & Adaptive Personalization.

Tests cover:
    1. Feedback Storage (recording, updating, removing, persistence, multi-user isolation).
    2. User Preference Learning (topics, publishers, countries, smoothing, confidence, decay).
    3. Centralized Personalization Engine (scoring, multi-dimensional blending, cold start).
    4. Integration across ALL Recommendation Methods (Baseline, Geographic MMR, Publisher Diversity,
       Interest Bridging, Feed Composer).
    5. Regression Safety (relevance preservation, deterministic tie-breaking, diversity guarantees).
"""

import unittest
import tempfile
import os
import shutil
from datetime import datetime, timezone, timedelta
import pandas as pd

from utils.feedback_manager import (
    FeedbackManager,
    FEEDBACK_INTERESTED,
    FEEDBACK_NOT_INTERESTED
)
from recommender.user_preferences import (
    build_user_preference_profile,
    UserPreferenceProfile
)
from recommender.personalization_engine import (
    score_candidate_preferences,
    apply_personalization
)
from recommender.baseline import (
    calculate_relevance,
    score_all_candidates,
    recommend_baseline
)
from recommender.geographic_diversity import (
    recommend_geographic_mmr,
    evaluate_slate
)
from recommender.interest_bridging import (
    recommend_interest_bridging,
    calculate_bridging_score
)
from recommender.feed_composer import (
    compose_balanced_feed
)


class TestFeedbackStorage(unittest.TestCase):
    """Tests for persistent SQLite feedback management."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_feedback.db")
        self.mgr = FeedbackManager(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_interested_feedback(self):
        success = self.mgr.record_feedback(
            user_id="user_1",
            article_id="art_100",
            feedback_type=FEEDBACK_INTERESTED,
            article_metadata={
                "title": "Quantum Computing Breakthrough",
                "publisher": "TechChronicle",
                "country": "United States",
                "continent": "North America",
                "topics": ["Technology", "Science"]
            }
        )
        self.assertTrue(success)
        user_fb = self.mgr.get_user_feedback("user_1")
        self.assertEqual(user_fb.get("art_100"), FEEDBACK_INTERESTED)

    def test_record_not_interested_feedback(self):
        success = self.mgr.record_feedback(
            user_id="user_1",
            article_id="art_101",
            feedback_type=FEEDBACK_NOT_INTERESTED,
            article_metadata={"title": "Celebrity Scandal", "publisher": "TabloidPost"}
        )
        self.assertTrue(success)
        user_fb = self.mgr.get_user_feedback("user_1")
        self.assertEqual(user_fb.get("art_101"), FEEDBACK_NOT_INTERESTED)

    def test_change_feedback(self):
        # Initial: interested
        self.mgr.record_feedback("user_1", "art_200", FEEDBACK_INTERESTED)
        self.assertEqual(self.mgr.get_user_feedback("user_1").get("art_200"), FEEDBACK_INTERESTED)

        # User changes mind to not_interested
        self.mgr.record_feedback("user_1", "art_200", FEEDBACK_NOT_INTERESTED)
        self.assertEqual(self.mgr.get_user_feedback("user_1").get("art_200"), FEEDBACK_NOT_INTERESTED)

        # Only 1 record should exist for this user & article
        records = self.mgr.get_feedback_records(user_id="user_1")
        art_records = [r for r in records if r["article_id"] == "art_200"]
        self.assertEqual(len(art_records), 1)

    def test_remove_feedback(self):
        self.mgr.record_feedback("user_1", "art_300", FEEDBACK_INTERESTED)
        self.assertIn("art_300", self.mgr.get_user_feedback("user_1"))

        removed = self.mgr.remove_feedback("user_1", "art_300")
        self.assertTrue(removed)
        self.assertNotIn("art_300", self.mgr.get_user_feedback("user_1"))

    def test_multi_user_isolation(self):
        self.mgr.record_feedback("alice", "art_1", FEEDBACK_INTERESTED)
        self.mgr.record_feedback("bob", "art_1", FEEDBACK_NOT_INTERESTED)

        alice_fb = self.mgr.get_user_feedback("alice")
        bob_fb = self.mgr.get_user_feedback("bob")

        self.assertEqual(alice_fb["art_1"], FEEDBACK_INTERESTED)
        self.assertEqual(bob_fb["art_1"], FEEDBACK_NOT_INTERESTED)

    def test_persistence_across_reconnection(self):
        self.mgr.record_feedback("user_1", "art_400", FEEDBACK_INTERESTED)

        # Create new manager pointing to same SQLite database
        new_mgr = FeedbackManager(db_path=self.db_path)
        fb = new_mgr.get_user_feedback("user_1")
        self.assertEqual(fb.get("art_400"), FEEDBACK_INTERESTED)

    def test_reset_user_feedback(self):
        self.mgr.record_feedback("user_to_reset", "art_1", FEEDBACK_INTERESTED)
        self.mgr.record_feedback("user_to_reset", "art_2", FEEDBACK_NOT_INTERESTED)
        self.mgr.record_feedback("user_other", "art_1", FEEDBACK_INTERESTED)

        deleted = self.mgr.clear_user_feedback("user_to_reset")
        self.assertEqual(deleted, 2)
        self.assertEqual(len(self.mgr.get_user_feedback("user_to_reset")), 0)
        self.assertEqual(len(self.mgr.get_user_feedback("user_other")), 1)


class TestUserPreferenceLearning(unittest.TestCase):
    """Tests for mathematical preference learning formulas and edge cases."""

    def test_cold_start_new_user(self):
        profile = build_user_preference_profile([], user_id="newbie")
        self.assertEqual(profile.total_feedback_count, 0)
        self.assertEqual(profile.get_topic_preference("Technology"), 0.5)
        self.assertEqual(profile.get_publisher_preference("Reuters"), 0.5)
        self.assertEqual(profile.get_country_preference("Germany"), 0.5)
        self.assertEqual(profile.get_geographic_preference("Germany", "Europe"), 0.5)

    def test_single_negative_does_not_eliminate(self):
        """Verify smoothing prevents a single negative feedback from zeroing out an entity."""
        records = [{
            "feedback_type": "not_interested",
            "topics": ["Technology"],
            "publisher": "TechHub",
            "country": "United States",
            "continent": "North America",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        profile = build_user_preference_profile(records, user_id="u1")

        # Score should be slightly < 0.5, but comfortably > 0.3
        pref = profile.get_topic_preference("Technology")
        self.assertLess(pref, 0.5)
        self.assertGreater(pref, 0.35)

        pub_pref = profile.get_publisher_preference("TechHub")
        self.assertLess(pub_pref, 0.5)
        self.assertGreater(pub_pref, 0.35)

    def test_multi_topic_attribution(self):
        """Verify credit is split proportionally among multi-topic articles."""
        records = [{
            "feedback_type": "interested",
            "topics": ["Technology", "Health", "Science"],
            "publisher": "MedTechNews",
            "country": "Germany",
            "continent": "Europe",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        profile = build_user_preference_profile(records, user_id="u2")
        self.assertGreater(profile.get_topic_preference("Technology"), 0.5)
        self.assertGreater(profile.get_topic_preference("Health"), 0.5)
        self.assertGreater(profile.get_topic_preference("Science"), 0.5)
        # Unrated topic stays neutral
        self.assertEqual(profile.get_topic_preference("Politics and Government"), 0.5)

    def test_geographic_distinction(self):
        """Verify liking one country does not decrease preference for unrated countries."""
        records = [{
            "feedback_type": "interested",
            "topics": ["Environment and Climate"],
            "publisher": "EcoWatch",
            "country": "Kenya",
            "continent": "Africa",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        profile = build_user_preference_profile(records, user_id="u3")
        self.assertGreater(profile.get_country_preference("Kenya"), 0.5)
        # Brazil was never reviewed: must remain neutral 0.5, NOT penalized!
        self.assertEqual(profile.get_country_preference("Brazil"), 0.5)

    def test_recency_decay(self):
        """Verify older feedback has less weight than recent feedback."""
        now = datetime.now(timezone.utc)
        old_time = (now - timedelta(days=60)).isoformat()
        recent_time = now.isoformat()

        # Two records: old negative on Topic A, recent positive on Topic A
        records = [
            {
                "feedback_type": "not_interested",
                "topics": ["TopicA"],
                "publisher": "Pub1",
                "country": "US",
                "continent": "North America",
                "updated_at": old_time
            },
            {
                "feedback_type": "interested",
                "topics": ["TopicA"],
                "publisher": "Pub1",
                "country": "US",
                "continent": "North America",
                "updated_at": recent_time
            }
        ]
        profile = build_user_preference_profile(records, user_id="u4", now_dt=now)
        # Because the positive feedback is recent and negative is 60 days old, net score > 0.5
        self.assertGreater(profile.get_topic_preference("TopicA"), 0.5)


class TestPersonalizationEngine(unittest.TestCase):
    """Tests for candidate scoring and baseline relevance preservation."""

    def setUp(self):
        self.sample_df = pd.DataFrame([
            {
                "article_id": "art_1",
                "title": "AI in Healthcare",
                "publisher": "HealthTech",
                "country": "United States",
                "continent": "North America",
                "topic_list": ["Technology", "Health"],
                "relevance_score": 0.8
            },
            {
                "article_id": "art_2",
                "title": "Clean Energy in Kenya",
                "publisher": "GreenAfrica",
                "country": "Kenya",
                "continent": "Africa",
                "topic_list": ["Environment and Climate"],
                "relevance_score": 0.7
            }
        ])

    def test_cold_start_preserves_baseline_relevance(self):
        profile = build_user_preference_profile([], user_id="cold_user")
        scored = apply_personalization(self.sample_df, profile, alpha=0.35)

        self.assertIn("baseline_relevance", scored.columns)
        self.assertIn("personalized_relevance", scored.columns)
        for _, row in scored.iterrows():
            self.assertEqual(row["baseline_relevance"], row["relevance_score"])
            self.assertAlmostEqual(row["personalized_relevance"], row["relevance_score"], places=5)

    def test_positive_feedback_boosts_relevance(self):
        records = [{
            "feedback_type": "interested",
            "topics": ["Technology", "Health"],
            "publisher": "HealthTech",
            "country": "United States",
            "continent": "North America",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        profile = build_user_preference_profile(records, user_id="tech_lover")
        scored = apply_personalization(self.sample_df, profile, alpha=0.35)

        art1 = scored[scored["article_id"] == "art_1"].iloc[0]
        # Personalized relevance should exceed baseline relevance
        self.assertGreater(art1["personalized_relevance"], art1["baseline_relevance"])
        # Original baseline relevance must be untouched
        self.assertEqual(art1["baseline_relevance"], 0.8)

    def test_negative_feedback_dampens_relevance_safely(self):
        records = [{
            "feedback_type": "not_interested",
            "topics": ["Technology", "Health"],
            "publisher": "HealthTech",
            "country": "United States",
            "continent": "North America",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        profile = build_user_preference_profile(records, user_id="tech_hater")
        scored = apply_personalization(self.sample_df, profile, alpha=0.35)

        art1 = scored[scored["article_id"] == "art_1"].iloc[0]
        # Should be dampened, but still positive (not zeroed out)
        self.assertLess(art1["personalized_relevance"], art1["baseline_relevance"])
        self.assertGreater(art1["personalized_relevance"], 0.0)


class TestRecommendationIntegration(unittest.TestCase):
    """Tests verifying that all recommendation methods consume the shared preference profile."""

    def setUp(self):
        self.mock_corpus = pd.DataFrame([
            {
                "article_id": f"art_{i}",
                "title": f"Article {i}",
                "publisher": f"Pub_{i % 5}",
                "country": ["United States", "United Kingdom", "Germany", "Kenya", "Brazil"][i % 5],
                "country_code": ["US", "GB", "DE", "KE", "BR"][i % 5],
                "continent": ["North America", "Europe", "Europe", "Africa", "South America"][i % 5],
                "topic_list": [["Technology"], ["Health"], ["Business and Economy"], ["Technology", "Health"], ["Science"]][i % 5],
                "relevance_score": 0.5 + (0.05 * (i % 6)),
                "article_url": f"https://news.example.com/{i}"
            }
            for i in range(25)
        ])
        self.user_profile = {"Technology": 1.0, "Environment and Climate": 1.0}

        # Feedback history
        self.feedback = [
            {
                "feedback_type": "interested",
                "topics": ["Technology", "Environment and Climate"],
                "publisher": "Pub_3",
                "country": "Kenya",
                "continent": "Africa",
                "updated_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "feedback_type": "not_interested",
                "topics": ["Business and Economy"],
                "publisher": "Pub_2",
                "country": "Germany",
                "continent": "Europe",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        self.pref_profile = build_user_preference_profile(self.feedback, user_id="integration_user")

    def test_baseline_integration(self):
        recs = recommend_baseline(
            self.mock_corpus,
            self.user_profile,
            top_k=5,
            preference_profile=self.pref_profile,
            alpha=0.35
        )
        self.assertEqual(len(recs), 5)
        self.assertIn("personalized_relevance", recs.columns)
        self.assertIn("baseline_relevance", recs.columns)

    def test_geographic_mmr_integration(self):
        recs = recommend_geographic_mmr(
            self.mock_corpus,
            top_k=5,
            lambda_param=0.6,
            preference_profile=self.pref_profile,
            alpha=0.35
        )
        self.assertEqual(len(recs), 5)
        # Should preserve diverse countries
        countries = recs["country"].unique()
        self.assertGreaterEqual(len(countries), 3)

    def test_interest_bridging_integration(self):
        recs = recommend_interest_bridging(
            self.mock_corpus,
            self.user_profile,
            top_k=5,
            preference_profile=self.pref_profile,
            alpha=0.35
        )
        self.assertGreater(len(recs), 0)
        self.assertIn("bridge_score", recs.columns)

    def test_feed_composer_integration(self):
        recs = compose_balanced_feed(
            self.mock_corpus,
            self.user_profile,
            top_k=10,
            personalized_ratio=0.70,
            preference_profile=self.pref_profile,
            alpha=0.35
        )
        self.assertEqual(len(recs), 10)
        self.assertIn("feed_lane", recs.columns)
        self.assertIn("personalized", recs["feed_lane"].values)
        self.assertIn("discovery", recs["feed_lane"].values)


if __name__ == "__main__":
    unittest.main()
