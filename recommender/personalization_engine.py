"""
Centralized Personalization Engine for Beyond the Feed.

Provides a unified, reusable adaptive personalization layer consumed by every
recommendation method in the system. Calculates multi-dimensional preference
signals for candidate articles without altering the original baseline relevance scores.

Mathematical Formulation:
    1. Multi-dimensional Preference Scoring:
       UserPref(a) = w_topic * TopicPref(a) + w_pub * PubPref(a) + w_geo * GeoPref(a)
       Where TopicPref, PubPref, GeoPref in [0.0, 1.0] (0.5 = neutral).

    2. Personalized Relevance Blending:
       DeltaPref(a) = 2.0 * UserPref(a) - 1.0   in [-1.0, +1.0]
       Rel_pers(a)  = clip(Rel_base(a) * (1.0 + alpha * DeltaPref(a)), 0.0, 1.0)

    Properties:
       - If UserPref(a) == 0.5 (cold start / neutral): Rel_pers(a) == Rel_base(a).
       - If UserPref(a) > 0.5 (positive affinity): Rel_pers(a) is boosted up to (1 + alpha).
       - If UserPref(a) < 0.5 (negative affinity): Rel_pers(a) is dampened down to (1 - alpha).
       - Original baseline relevance is strictly preserved in 'baseline_relevance'.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

from recommender.user_preferences import UserPreferenceProfile


# Default personalization weights
DEFAULT_WEIGHT_TOPIC = 0.60
DEFAULT_WEIGHT_PUBLISHER = 0.25
DEFAULT_WEIGHT_GEOGRAPHY = 0.15
DEFAULT_ALPHA_PERSONALIZATION = 0.35  # Balance between baseline intent and learned feedback


def score_candidate_preferences(
    candidate: Dict[str, Any],
    profile: Optional[UserPreferenceProfile],
    weight_topic: float = DEFAULT_WEIGHT_TOPIC,
    weight_publisher: float = DEFAULT_WEIGHT_PUBLISHER,
    weight_geography: float = DEFAULT_WEIGHT_GEOGRAPHY,
    alpha: float = DEFAULT_ALPHA_PERSONALIZATION
) -> Dict[str, float]:
    """
    Score a single candidate record against the user preference profile.

    Returns:
        Dict with keys:
            - topic_preference_score
            - publisher_preference_score
            - geographic_preference_score
            - combined_user_preference_score
            - baseline_relevance
            - personalized_relevance
    """
    base_rel = float(candidate.get("relevance_score", 0.0))

    if profile is None or profile.total_feedback_count == 0:
        return {
            "baseline_relevance": base_rel,
            "topic_preference_score": 0.5,
            "publisher_preference_score": 0.5,
            "geographic_preference_score": 0.5,
            "combined_user_preference_score": 0.5,
            "personalized_relevance": base_rel
        }

    # 1. Topic preference: average over candidate's topics
    topics = candidate.get("topic_list") or candidate.get("topics") or []
    if isinstance(topics, str):
        topics = [t.strip() for t in topics.split(";") if t.strip()]

    valid_topics = [t for t in topics if t not in {"Unknown", "Other"}]
    if valid_topics:
        t_prefs = [profile.get_topic_preference(t) for t in valid_topics]
        topic_pref = sum(t_prefs) / len(t_prefs)
    else:
        topic_pref = 0.5

    # 2. Publisher preference
    pub = str(candidate.get("publisher", "")).strip()
    pub_pref = profile.get_publisher_preference(pub)

    # 3. Geographic preference
    country = candidate.get("country")
    continent = candidate.get("continent")
    geo_pref = profile.get_geographic_preference(country, continent)

    # 4. Combined user preference score
    # Normalize weights if custom
    total_w = weight_topic + weight_publisher + weight_geography
    w_t = weight_topic / total_w
    w_p = weight_publisher / total_w
    w_g = weight_geography / total_w

    combined_pref = (w_t * topic_pref) + (w_p * pub_pref) + (w_g * geo_pref)

    # 5. Personalized relevance
    delta_pref = 2.0 * combined_pref - 1.0  # [-1.0, 1.0]
    pers_rel = float(np.clip(base_rel * (1.0 + alpha * delta_pref), 0.0, 1.0))

    return {
        "baseline_relevance": base_rel,
        "topic_preference_score": float(topic_pref),
        "publisher_preference_score": float(pub_pref),
        "geographic_preference_score": float(geo_pref),
        "combined_user_preference_score": float(combined_pref),
        "personalized_relevance": pers_rel
    }


def apply_personalization(
    candidates_df: pd.DataFrame,
    profile: Optional[UserPreferenceProfile],
    weight_topic: float = DEFAULT_WEIGHT_TOPIC,
    weight_publisher: float = DEFAULT_WEIGHT_PUBLISHER,
    weight_geography: float = DEFAULT_WEIGHT_GEOGRAPHY,
    alpha: float = DEFAULT_ALPHA_PERSONALIZATION
) -> pd.DataFrame:
    """
    Apply centralized user preference scoring across all candidates in a DataFrame.

    Appends the following columns without modifying original columns:
        - 'baseline_relevance' (preserves original relevance_score)
        - 'topic_preference_score'
        - 'publisher_preference_score'
        - 'geographic_preference_score'
        - 'combined_user_preference_score'
        - 'personalized_relevance'

    Args:
        candidates_df: DataFrame of candidate articles (must have 'relevance_score' and 'topic_list').
        profile: Learned UserPreferenceProfile.
        weight_topic: Weight for topic preference (default 0.60).
        weight_publisher: Weight for publisher preference (default 0.25).
        weight_geography: Weight for geographic preference (default 0.15).
        alpha: Personalization scaling factor (default 0.35).

    Returns:
        pd.DataFrame with added personalization scores.
    """
    if candidates_df.empty:
        df_copy = candidates_df.copy()
        for col in [
            "baseline_relevance", "topic_preference_score", "publisher_preference_score",
            "geographic_preference_score", "combined_user_preference_score", "personalized_relevance"
        ]:
            if col not in df_copy.columns:
                df_copy[col] = []
        return df_copy

    df = candidates_df.copy()

    # Ensure baseline_relevance is explicitly recorded
    if "baseline_relevance" not in df.columns:
        df["baseline_relevance"] = df["relevance_score"].copy()

    # Fast path for cold-start / empty feedback
    if profile is None or profile.total_feedback_count == 0:
        df["topic_preference_score"] = 0.5
        df["publisher_preference_score"] = 0.5
        df["geographic_preference_score"] = 0.5
        df["combined_user_preference_score"] = 0.5
        df["personalized_relevance"] = df["baseline_relevance"].copy()
        return df

    # Vectorized / mapped calculation for efficiency
    def _score_row(row):
        return score_candidate_preferences(
            candidate=row,
            profile=profile,
            weight_topic=weight_topic,
            weight_publisher=weight_publisher,
            weight_geography=weight_geography,
            alpha=alpha
        )

    scored_dicts = df.apply(_score_row, axis=1)

    df["topic_preference_score"] = [d["topic_preference_score"] for d in scored_dicts]
    df["publisher_preference_score"] = [d["publisher_preference_score"] for d in scored_dicts]
    df["geographic_preference_score"] = [d["geographic_preference_score"] for d in scored_dicts]
    df["combined_user_preference_score"] = [d["combined_user_preference_score"] for d in scored_dicts]
    df["personalized_relevance"] = [d["personalized_relevance"] for d in scored_dicts]

    return df

