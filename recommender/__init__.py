"""
Recommendation Engine Package for Beyond the Feed.
"""

from recommender.baseline import (
    calculate_relevance,
    recommend_baseline,
    score_all_candidates,
)
from recommender.geographic_diversity import (
    recommend_geographic_mmr,
    calculate_geo_redundancy,
    generate_natural_explanation,
    is_valid_geo,
    evaluate_slate,
    compare_baseline_vs_mmr,
)

__all__ = [
    "calculate_relevance",
    "recommend_baseline",
    "score_all_candidates",
    "recommend_geographic_mmr",
    "calculate_geo_redundancy",
    "generate_natural_explanation",
    "is_valid_geo",
    "evaluate_slate",
    "compare_baseline_vs_mmr",
]
