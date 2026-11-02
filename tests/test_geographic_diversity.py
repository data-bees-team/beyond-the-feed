"""
Unit and Integration Test Suite for Geographic Diversity Re-Ranking.
Tests baseline preservation, geographic redundancy, missing-geography robustness,
candidate pool scaling, tie-breaking determinism, and Streamlit components.
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
from recommender.geographic_diversity import (
    calculate_geo_redundancy,
    recommend_geographic_mmr,
    generate_natural_explanation,
    is_valid_geo,
    evaluate_slate,
    compute_hhi,
    compare_baseline_vs_mmr,
)
from utils.data_loader import load_articles
from components.news_card import is_valid_geo_label


class TestGeographicDiversity(unittest.TestCase):
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
    def test_baseline_relevance_calculation_unchanged(self):
        """Verify calculate_relevance matches the arithmetic mean of valid topics."""
        # Single topic
        self.assertAlmostEqual(calculate_relevance(["Technology"], {"Technology": 0.8}), 0.8)
        # Multiple topics
        self.assertAlmostEqual(
            calculate_relevance(["Technology", "Health"], {"Technology": 0.8, "Health": 0.2}),
            0.5
        )
        # Topics with Unknown and Other excluded
        self.assertAlmostEqual(
            calculate_relevance(["Technology", "Unknown", "Other"], {"Technology": 0.8}),
            0.8
        )
        # Empty or only invalid
        self.assertEqual(calculate_relevance(["Unknown"], {"Technology": 0.8}), 0.0)
        self.assertEqual(calculate_relevance([], {"Technology": 0.8}), 0.0)

    def test_baseline_recommendations_work_and_unchanged(self):
        """Verify recommend_baseline returns top 10 deterministic recommendations."""
        user_prof = self.profiles["Tech Enthusiast"]
        base_recs = recommend_baseline(self.df, user_prof, top_k=10)
        self.assertEqual(len(base_recs), 10)
        self.assertTrue((base_recs["relevance_score"] > 0).all())
        # Check sort order: relevance descending
        scores = list(base_recs["relevance_score"])
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_baseline_scores_preserved_after_mmr(self):
        """Verify original baseline relevance score is untouched in MMR outputs."""
        user_prof = self.profiles["Environment Explorer"]
        candidates = score_all_candidates(self.df, user_prof)
        mmr_recs = recommend_geographic_mmr(candidates, top_k=10, lambda_param=0.6)

        # Cross check with candidate scores
        for _, row in mmr_recs.iterrows():
            aid = row["article_id"]
            expected_score = candidates.loc[candidates["article_id"] == aid, "relevance_score"].iloc[0]
            self.assertAlmostEqual(row["relevance_score"], expected_score)

    # =========================================================================
    # B. Geographic Diversity Logic Tests
    # =========================================================================
    def test_country_level_redundancy(self):
        """Verify same primary country produces maximum redundancy (1.0)."""
        selected = [
            {"article_id": "art-1", "country": "United States", "continent": "North America", "all_countries_set": {"United States"}}
        ]
        cand = {"article_id": "art-2", "country": "United States", "continent": "North America", "all_countries_set": {"United States"}}
        sim, reason = calculate_geo_redundancy(cand, selected)
        self.assertEqual(sim, 1.0)
        self.assertIn("Country redundancy", reason)

    def test_continent_level_redundancy(self):
        """Verify same continent with different country produces continent penalty (0.4)."""
        selected = [
            {"article_id": "art-1", "country": "United States", "continent": "North America", "all_countries_set": {"United States"}}
        ]
        cand = {"article_id": "art-2", "country": "Canada", "continent": "North America", "all_countries_set": {"Canada"}}
        sim, reason = calculate_geo_redundancy(cand, selected)
        self.assertEqual(sim, 0.4)
        self.assertIn("Continent redundancy", reason)

    def test_secondary_country_overlap(self):
        """Verify secondary country overlap across distinct continents applies secondary penalty (0.1)."""
        selected = [
            {"article_id": "art-1", "country": "South Africa", "continent": "Africa", "all_countries_set": {"South Africa"}}
        ]
        cand = {"article_id": "art-2", "country": "Chile", "continent": "South America", "all_countries_set": {"Chile", "South Africa"}}
        sim, reason = calculate_geo_redundancy(cand, selected)
        self.assertEqual(sim, 0.1)
        self.assertIn("Secondary country overlap", reason)

    def test_missing_geography_neutral_penalty(self):
        """Verify un-geotagged candidates receive neutral penalty (0.5), not artificial novelty (0.0)."""
        selected = [
            {"article_id": "art-1", "country": "United States", "continent": "North America", "all_countries_set": {"United States"}}
        ]
        for invalid_val in ["", "Unknown", "None", "NULL", "undetermined", None]:
            cand = {"article_id": "art-unk", "country": invalid_val, "continent": "North America", "all_countries_set": set()}
            sim, reason = calculate_geo_redundancy(cand, selected, weight_missing_geo=0.5)
            self.assertEqual(sim, 0.5, f"Failed for invalid value '{invalid_val}'")
            self.assertIn("Missing or unknown primary geography", reason)

    def test_namibia_code_preserved_as_valid(self):
        """Verify Namibia's ISO code 'NA' is recognized as a valid geographic entity."""
        self.assertTrue(is_valid_geo("Namibia"))
        self.assertTrue(is_valid_geo("NA"))
        self.assertTrue(is_valid_geo_label("NA"))
        namibia_rows = self.df[self.df["country"] == "Namibia"]
        self.assertGreater(len(namibia_rows), 0)
        self.assertEqual(namibia_rows["country_code"].isna().sum(), 0)
        self.assertTrue((namibia_rows["country_code"] == "NA").all())

    def test_missing_continent_handles_country_safely(self):
        """Verify valid country with missing continent compares cleanly without raising exceptions."""
        selected = [
            {"article_id": "art-1", "country": "Italy", "continent": "Europe", "all_countries_set": {"Italy"}}
        ]
        # Same country, missing continent
        cand_same = {"article_id": "art-2", "country": "Italy", "continent": "", "all_countries_set": {"Italy"}}
        sim, _ = calculate_geo_redundancy(cand_same, selected)
        self.assertEqual(sim, 1.0)

        # Different country, missing continent
        cand_diff = {"article_id": "art-3", "country": "Vatican City", "continent": "", "all_countries_set": {"Vatican City"}}
        sim, _ = calculate_geo_redundancy(cand_diff, selected)
        self.assertEqual(sim, 0.0)

    # =========================================================================
    # C. Recommendation Behaviour Tests
    # =========================================================================
    def test_requested_top_k_returned(self):
        """Verify MMR returns exact requested top_k count."""
        candidates = score_all_candidates(self.df, self.profiles["Tech Enthusiast"])
        for k in [5, 10, 15]:
            recs = recommend_geographic_mmr(candidates, top_k=k)
            self.assertEqual(len(recs), k)
            self.assertEqual(len(recs["article_id"].unique()), k)

    def test_candidate_pool_optimization_and_bounds(self):
        """Verify M >= K is respected and candidate pool size produces deterministic results."""
        candidates = score_all_candidates(self.df, self.profiles["Health Researcher"])
        recs_500 = recommend_geographic_mmr(candidates, top_k=10, top_m_candidates=500)
        recs_all = recommend_geographic_mmr(candidates, top_k=10, top_m_candidates=None)
        self.assertEqual(list(recs_500["article_id"]), list(recs_all["article_id"]))

    def test_deterministic_tie_breaking(self):
        """Verify results are strictly invariant under candidate order shuffling."""
        candidates = score_all_candidates(self.df, self.profiles["Environment Explorer"])
        recs_orig = recommend_geographic_mmr(candidates, top_k=10, lambda_param=0.6)
        orig_ids = list(recs_orig["article_id"])

        for seed in [42, 123, 999]:
            shuffled = candidates.sample(frac=1.0, random_state=seed)
            recs_shuf = recommend_geographic_mmr(shuffled, top_k=10, lambda_param=0.6)
            self.assertEqual(list(recs_shuf["article_id"]), orig_ids)

    # =========================================================================
    # D. Streamlit & UI Component Integration Tests
    # =========================================================================
    def test_natural_explanations_formatting(self):
        """Verify natural language explanations format properly without errors."""
        # Rank 1 explanation
        exp1 = generate_natural_explanation(1, "Japan", "Asia", ["Technology"], 0.0)
        self.assertIn("Top recommendation", exp1)
        self.assertIn("Japan", exp1)

        # Novel continent explanation
        exp_novel = generate_natural_explanation(2, "Kenya", "Africa", ["Environment"], 0.0)
        self.assertIn("fresh perspective", exp_novel)
        self.assertIn("Kenya", exp_novel)

        # Missing geography explanation
        exp_unk = generate_natural_explanation(3, "Unknown", "Unknown", ["Health"], 0.5)
        self.assertIn("Curated for relevance", exp_unk)

    def test_news_card_geo_label_validation(self):
        """Verify news card label validator catches nulls and preserves valid names."""
        self.assertTrue(is_valid_geo_label("United States"))
        self.assertTrue(is_valid_geo_label("NA"))
        self.assertFalse(is_valid_geo_label(""))
        self.assertFalse(is_valid_geo_label("Unknown"))
        self.assertFalse(is_valid_geo_label(None))

    # =========================================================================
    # E. Regression & Benchmark Comparison Tests
    # =========================================================================
    def test_reproduce_benchmark_evaluation(self):
        """Verify empirical improvements on all three benchmark personas."""
        expected_benchmarks = {
            "Tech Enthusiast": {"unique_countries": 10, "unique_continents": 6, "country_hhi": 0.10},
            "Environment Explorer": {"unique_countries": 10, "unique_continents": 6, "country_hhi": 0.10},
            "Health Researcher": {"unique_countries": 10, "unique_continents": 6, "country_hhi": 0.10},
        }

        for name, expected in expected_benchmarks.items():
            user_prof = self.profiles[name]
            candidates = score_all_candidates(self.df, user_prof)
            base_slate = recommend_baseline(self.df, user_prof, top_k=10)
            mmr_slate = recommend_geographic_mmr(candidates, top_k=10, lambda_param=0.6)

            comp = compare_baseline_vs_mmr(base_slate, mmr_slate, name, lambda_param=0.6)
            self.assertIsNotNone(comp)
            self.assertEqual(mmr_slate["country"].nunique(), expected["unique_countries"])
            self.assertEqual(mmr_slate["continent"].nunique(), expected["unique_continents"])
            self.assertAlmostEqual(compute_hhi(mmr_slate["country"]), expected["country_hhi"], places=2)


if __name__ == "__main__":
    unittest.main()
