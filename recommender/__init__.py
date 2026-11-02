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
from recommender.user_preferences import (
    UserPreferenceProfile,
    build_user_preference_profile,
)
from recommender.personalization_engine import (
    apply_personalization,
    score_candidate_preferences,
    DEFAULT_ALPHA_PERSONALIZATION,
)
from recommender.interest_bridging import (
    recommend_interest_bridging,
    calculate_bridging_score,
)
from recommender.feed_composer import (
    compose_balanced_feed,
)

__all__ = [
    # Baseline
    "calculate_relevance",
    "recommend_baseline",
    "score_all_candidates",
    # Geographic Diversity
    "recommend_geographic_mmr",
    "calculate_geo_redundancy",
    "generate_natural_explanation",
    "is_valid_geo",
    "evaluate_slate",
    "compare_baseline_vs_mmr",
    # User Preferences
    "UserPreferenceProfile",
    "build_user_preference_profile",
    # Personalization Engine
    "apply_personalization",
    "score_candidate_preferences",
    "DEFAULT_ALPHA_PERSONALIZATION",
    # Interest Bridging
    "recommend_interest_bridging",
    "calculate_bridging_score",
    # Feed Composition
    "compose_balanced_feed",
]
