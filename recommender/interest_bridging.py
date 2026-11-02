"""
Interest Bridging Recommendation Engine for Beyond the Feed.

Discovers meaningful connections between a user's core interests and adjacent,
unfamiliar topic domains (e.g. Technology -> Health via AI Diagnostics, or
Business -> Environment via Clean Energy Investment).

Integrates with the centralized UserPreferenceProfile to ensure:
    1. Positive feedback on bridge topics encourages further exploration in that direction.
    2. Explicitly disliked topics are excluded from bridge pathways.
    3. The recommendation avoids echo chambers by purposefully introducing adjacent themes.
"""

from typing import Dict, List, Optional, Set, Any
import pandas as pd
import numpy as np

from recommender.user_preferences import UserPreferenceProfile
from recommender.personalization_engine import apply_personalization, DEFAULT_ALPHA_PERSONALIZATION


def calculate_bridging_score(
    topics: List[str],
    user_interests: Set[str],
    preference_profile: Optional[UserPreferenceProfile] = None,
    bridge_weight: float = 0.4
) -> float:
    """
    Calculate the interest bridging affinity of an article's topic mix.

    Args:
        topics: List of topics assigned to the article.
        user_interests: Set of user's selected core interest topics.
        preference_profile: Optional learned user preferences.
        bridge_weight: Weight given to bridge novelty bonus.

    Returns:
        float: Bridging multiplier >= 0.0. Returns 0.0 if the article contains
               a topic the user explicitly disliked.
    """
    valid_topics = [t for t in topics if t not in {"Unknown", "Other"}]
    if not valid_topics:
        return 0.0

    core_overlap = [t for t in valid_topics if t in user_interests]
    novel_topics = [t for t in valid_topics if t not in user_interests]

    # Need at least one core topic anchor and one novel topic for bridging
    if not core_overlap or not novel_topics:
        return 0.0

    # Check for negative feedback on novel topics
    if preference_profile is not None:
        for t in novel_topics:
            if preference_profile.topic_scores.get(t, 0.0) < -0.15:
                # User has negative feedback on this bridge topic: skip
                return 0.0

    # Topic bridge bonus proportional to novel topics count, boosted by positive feedback
    base_bonus = min(1.0, len(novel_topics) * 0.35)

    if preference_profile is not None:
        pos_bonus = sum(
            max(0.0, preference_profile.topic_scores.get(t, 0.0))
            for t in novel_topics
        )
        base_bonus += 0.2 * pos_bonus

    return float(base_bonus * bridge_weight)


def recommend_interest_bridging(
    df: pd.DataFrame,
    user_profile: Dict[str, float],
    top_k: int = 10,
    top_m_candidates: int = 500,
    preference_profile: Optional[UserPreferenceProfile] = None,
    alpha: float = DEFAULT_ALPHA_PERSONALIZATION,
    bridge_weight: float = 0.4
) -> pd.DataFrame:
    """
    Recommend articles that bridge user's selected interests with adjacent domains.

    Args:
        df: Full article DataFrame.
        user_profile: Dict mapping selected topics to weights.
        top_k: Number of recommendations to return (default 10).
        top_m_candidates: Candidate pool size (default 500).
        preference_profile: Optional learned UserPreferenceProfile.
        alpha: Personalization weight (default 0.35).
        bridge_weight: Weight of bridge bonus (default 0.4).

    Returns:
        pd.DataFrame of top_k interest-bridging articles.
    """
    if df.empty or not user_profile:
        return pd.DataFrame()

    user_interests = set(user_profile.keys())

    # Filter to articles that have at least one user topic and at least one novel topic
    def is_bridge_candidate(t_list):
        has_core = any(t in user_interests for t in t_list)
        has_novel = any(t not in user_interests and t not in {"Unknown", "Other"} for t in t_list)
        return has_core and has_novel

    bridge_pool = df[df["topic_list"].apply(is_bridge_candidate)].copy()

    if bridge_pool.empty:
        # Fallback to candidates with user topics if no strict multi-topic bridges exist
        from recommender.baseline import score_all_candidates
        fallback = score_all_candidates(df, user_profile, preference_profile, alpha).head(top_k).copy()
        if not fallback.empty:
            fallback["bridge_score"] = fallback.get("personalized_relevance", fallback["relevance_score"]).copy()
            fallback["selection_rank"] = range(1, len(fallback) + 1)
        return fallback

    # Compute baseline relevance
    from recommender.baseline import calculate_relevance
    bridge_pool["relevance_score"] = bridge_pool["topic_list"].apply(
        lambda topics: calculate_relevance(topics, user_profile)
    )
    bridge_pool = bridge_pool[bridge_pool["relevance_score"] > 0].copy()
    bridge_pool["baseline_relevance"] = bridge_pool["relevance_score"].copy()

    # Apply personalization if active
    if preference_profile is not None and preference_profile.total_feedback_count > 0:
        bridge_pool = apply_personalization(bridge_pool, preference_profile, alpha=alpha)
        rel_col = "personalized_relevance"
    else:
        bridge_pool["personalized_relevance"] = bridge_pool["baseline_relevance"].copy()
        rel_col = "relevance_score"

    # Compute bridging bonus
    bridge_pool["bridge_bonus"] = bridge_pool["topic_list"].apply(
        lambda topics: calculate_bridging_score(
            topics=topics,
            user_interests=user_interests,
            preference_profile=preference_profile,
            bridge_weight=bridge_weight
        )
    )

    # Combined bridge score
    bridge_pool["bridge_score"] = bridge_pool[rel_col] * (1.0 + bridge_pool["bridge_bonus"])

    # Exclude articles with bridge_bonus == 0 (e.g. disliked topics)
    valid_bridges = bridge_pool[bridge_pool["bridge_bonus"] > 0].copy()
    if valid_bridges.empty:
        valid_bridges = bridge_pool.copy()

    sorted_bridges = valid_bridges.sort_values(
        by=["bridge_score", "article_id"],
        ascending=[False, True]
    )

    actual_m = max(top_m_candidates, top_k)
    results = sorted_bridges.head(actual_m).head(top_k).copy()
    results["selection_rank"] = range(1, len(results) + 1)

    return results
