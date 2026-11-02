"""
Baseline Recommendation Engine for Beyond the Feed.

Computes topic-based relevance scores based on user interest profiles
and provides candidate generation and pure-relevance top-K recommendations.
Supports optional adaptive personalization via the centralized UserPreferenceProfile.
"""

from typing import Dict, Optional, Any
import pandas as pd

from recommender.user_preferences import UserPreferenceProfile
from recommender.personalization_engine import apply_personalization, DEFAULT_ALPHA_PERSONALIZATION


def calculate_relevance(topic_list, user_profile):
    """
    Calculate baseline relevance based on selected user interests.

    Formula:
        Rel_base(u, a) = sum_{t in ValidTopics} user_profile[t] / |ValidTopics|
    """
    valid_topics = [
        topic for topic in topic_list
        if topic not in {"Unknown", "Other"}
    ]

    if not valid_topics:
        return 0.0

    return sum(
        user_profile.get(topic, 0.0)
        for topic in valid_topics
    ) / len(valid_topics)


def score_all_candidates(
    df: pd.DataFrame,
    user_profile: Dict[str, float],
    preference_profile: Optional[UserPreferenceProfile] = None,
    alpha: float = DEFAULT_ALPHA_PERSONALIZATION
) -> pd.DataFrame:
    """
    Score all articles against user topic preferences and return all eligible
    candidates (relevance_score > 0) deterministically sorted.

    If preference_profile is provided and contains learned feedback, applies
    centralized personalization and sorts by personalized_relevance descending.
    Otherwise, sorts strictly by baseline relevance_score descending, article_id ascending.

    This preserves the full candidate pool for multi-stage re-ranking components
    (such as Geographic Diversity MMR) rather than truncating prematurely.
    """
    candidates = df.copy()

    candidates["relevance_score"] = (
        candidates["topic_list"].apply(
            lambda topics: calculate_relevance(
                topics, user_profile
            )
        )
    )

    candidates = candidates[
        candidates["relevance_score"] > 0
    ].copy()

    # Always ensure baseline_relevance is explicitly recorded
    candidates["baseline_relevance"] = candidates["relevance_score"].copy()

    if preference_profile is not None and preference_profile.total_feedback_count > 0:
        candidates = apply_personalization(candidates, preference_profile, alpha=alpha)
        return candidates.sort_values(
            by=["personalized_relevance", "article_id"],
            ascending=[False, True]
        ).copy()

    return candidates.sort_values(
        by=["relevance_score", "article_id"],
        ascending=[False, True]
    ).copy()


def recommend_baseline(
    df: pd.DataFrame,
    user_profile: Dict[str, float],
    top_k: int = 10,
    preference_profile: Optional[UserPreferenceProfile] = None,
    alpha: float = DEFAULT_ALPHA_PERSONALIZATION
) -> pd.DataFrame:
    """
    Return the top-k most relevant articles using deterministic tie-breaking.
    Optionally incorporates learned user preferences if preference_profile is provided.
    """
    candidates = score_all_candidates(
        df,
        user_profile,
        preference_profile=preference_profile,
        alpha=alpha
    )
    return candidates.head(top_k).copy()
