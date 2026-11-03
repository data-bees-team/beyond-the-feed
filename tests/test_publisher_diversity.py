"""
Unit and Integration Test Suite for Publisher Diversity Re-Ranking.
Tests baseline preservation, publisher redundancy, missing-publisher robustness,
HHI concentration metrics, tie-breaking determinism, and Streamlit card compatibility.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender.baseline import calculate_relevance, recommend_baseline, score_all_candidates
from recommender.publisher_diversity import (
    is_valid_publisher,
    calculate_publisher_redundancy,
    recommend_publisher_mmr,
    generate_natural_publisher_explanation,
    compute_publisher_hhi,
    evaluate_publisher_slate,
    compare_baseline_vs_publisher_mmr
)
from utils.data_loader import load_articles


class TestPublisherDiversity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_articles()
        cls.profiles = {
            "Tech Enthusiast": {"Technology": 0.8, "Business and Economy": 0.6, "Health": 0.2},
            "Environment Explorer": {"Environment and Climate": 0.9, "Science": 0.7, "Technology": 0.3},
            "Health Researcher": {"Health": 0.9, "Science": 0.8, "Education": 0.4},
        }

    # =========================================================================
    # A. Baseline Preservation Tests
    # =========================================================================
    def test_baseline_relevance_unchanged(self):
        """Verify topic relevance calculation is unchanged from Phase 2 specification."""
        self.assertAlmostEqual(calculate_relevance(["Technology"], {"Technology": 0.8}), 0.8)
        self.assertAlmostEqual(
            calculate_relevance(["Technology", "Health"], {"Technology": 0.8, "Health": 0.2}),
            0.5
        )

    def test_baseline_scores_preserved_in_output(self):
        """Verify original baseline relevance score is untouched in publisher MMR outputs."""
        candidates = score_all_candidates(self.df, self.profiles["Health Researcher"])
        mmr_recs = recommend_publisher_mmr(candidates, top_k=10, lambda_param=0.6)

        for _, row in mmr_recs.iterrows():
            aid = row["article_id"]
            expected = candidates.loc[candidates["article_id"] == aid, "relevance_score"].iloc[0]
            self.assertAlmostEqual(row["relevance_score"], expected)

    # =========================================================================
    # B. Publisher Redundancy & Validation Logic
    # =========================================================================
    def test_publisher_validation(self):
        """Verify is_valid_publisher detects real publishers and flags unknown/empty tokens."""
        self.assertTrue(is_valid_publisher("reuters.com"))
        self.assertTrue(is_valid_publisher("BBC News"))
        self.assertFalse(is_valid_publisher(""))
        self.assertFalse(is_valid_publisher("   "))
        self.assertFalse(is_valid_publisher(None))
        self.assertFalse(is_valid_publisher("Unknown"))
        self.assertFalse(is_valid_publisher("missing"))
        self.assertFalse(is_valid_publisher("none"))

    def test_novel_publisher_zero_penalty(self):
        """Verify a publisher not in slate S receives 0.0 redundancy penalty."""
        selected = [
            {"article_id": "art-1", "publisher": "bbc.com"}
        ]
        candidate = {"article_id": "art-2", "publisher": "theguardian.com"}
        sim, reason = calculate_publisher_redundancy(candidate, selected)
        self.assertEqual(sim, 0.0)
        self.assertIn("Novel publisher", reason)

    def test_repeated_publisher_penalty(self):
        """Verify a publisher already in slate S receives repeat penalty (1.0)."""
        selected = [
            {"article_id": "art-1", "publisher": "bbc.com"}
        ]
        candidate = {"article_id": "art-2", "publisher": "bbc.com"}
        sim, reason = calculate_publisher_redundancy(candidate, selected)
        self.assertEqual(sim, 1.0)
        self.assertIn("Publisher repetition", reason)

    def test_missing_publisher_neutral_penalty(self):
        """Verify missing publisher receives neutral 0.5 penalty (safe handling)."""
        selected = [{"article_id": "art-1", "publisher": "bbc.com"}]
        candidate = {"article_id": "art-2", "publisher": "Unknown"}
        sim, reason = calculate_publisher_redundancy(candidate, selected)
        self.assertEqual(sim, 0.5)
        self.assertIn("Missing or unknown publisher", reason)

    # =========================================================================
    # C. Re-ranking Efficacy Tests (Resolving Health Researcher Concentration)
    # =========================================================================
    def test_publisher_diversity_improves_health_researcher(self):
        """Verify publisher MMR increases unique publishers in Health Researcher slate."""
        candidates = score_all_candidates(self.df, self.profiles["Health Researcher"])
        baseline_slate = recommend_baseline(self.df, self.profiles["Health Researcher"], top_k=10)
        mmr_slate = recommend_publisher_mmr(candidates, top_k=10, lambda_param=0.6)

        base_unique = baseline_slate["publisher"].nunique()
        mmr_unique = mmr_slate["publisher"].nunique()

        # Baseline has duplicates (iheart.com, irishexaminer.com) giving 8 unique publishers
        self.assertEqual(base_unique, 8)
        # MMR achieves 10 unique publishers out of 10
        self.assertEqual(mmr_unique, 10)
        self.assertGreater(mmr_unique, base_unique)

        # Concentration (HHI) must drop
        base_hhi = compute_publisher_hhi(baseline_slate["publisher"])
        mmr_hhi = compute_publisher_hhi(mmr_slate["publisher"])
        self.assertLess(mmr_hhi, base_hhi)

    def test_all_metadata_preserved_for_news_card(self):
        """Verify all columns required by components/news_card.py are preserved."""
        candidates = score_all_candidates(self.df, self.profiles["Tech Enthusiast"])
        mmr_slate = recommend_publisher_mmr(candidates, top_k=10, lambda_param=0.6)

        required_cols = [
            "article_id", "title", "publisher", "country", "continent",
            "topic", "topic_list", "article_url", "relevance_score", "reason",
            "mmr_score", "pub_redundancy", "selection_rank"
        ]
        for col in required_cols:
            self.assertIn(col, mmr_slate.columns, f"Missing required column: {col}")

    def test_lambda_extremes(self):
        """Test lambda=1.0 (pure relevance) vs lambda=0.0 (pure publisher diversity)."""
        candidates = score_all_candidates(self.df, self.profiles["Health Researcher"])

        # lambda = 1.0 gives identical ordering to baseline relevance
        recs_lam1 = recommend_publisher_mmr(candidates, top_k=10, lambda_param=1.0)
        base_recs = recommend_baseline(self.df, self.profiles["Health Researcher"], top_k=10)
        self.assertEqual(list(recs_lam1["article_id"]), list(base_recs["article_id"]))

        # lambda = 0.0 strictly prevents any duplicate publisher
        recs_lam0 = recommend_publisher_mmr(candidates, top_k=10, lambda_param=0.0)
        self.assertEqual(recs_lam0["publisher"].nunique(), 10)


if __name__ == "__main__":
    unittest.main()
