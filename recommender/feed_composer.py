"""
Unified Feed Composer for Beyond the Feed.

Orchestrates multi-objective feed composition by balancing:
    1. Personalized Core (70-75%): Highly relevant, diversity-aware articles matching
       user interests and learned feedback preferences.
    2. Serendipitous Discovery (25-30%): Exploratory perspectives, interest-bridging
       topics, and under-represented global voices outside the user's echo chamber.

Integrates with the centralized UserPreferenceProfile to ensure learned feedback
improves personalization while maintaining exploration guarantees.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from recommender.user_preferences import UserPreferenceProfile
from recommender.geographic_diversity import recommend_geographic_mmr
from recommender.interest_bridging import recommend_interest_bridging
from recommender.baseline import score_all_candidates
from recommender.personalization_engine import DEFAULT_ALPHA_PERSONALIZATION


def compose_balanced_feed(
    df: pd.DataFrame,
    user_profile: Dict[str, float],
    top_k: int = 10,
    personalized_ratio: float = 0.70,
    diversity_method: str = "geographic",
    preference_profile: Optional[UserPreferenceProfile] = None,
    alpha: float = DEFAULT_ALPHA_PERSONALIZATION,
    lambda_param: float = 0.6,
    interleave: bool = True
) -> pd.DataFrame:
    """
    Generate a balanced news feed containing both personalized and discovery articles.

    Args:
        df: Full cleaned GDELT article dataset.
        user_profile: User selected topic dictionary {topic: weight}.
        top_k: Total recommendations to return (default 10).
        personalized_ratio: Fraction of feed dedicated to personalized items (default 0.70).
        diversity_method: Diversity algorithm for personalized items ('geographic' or 'publisher').
        preference_profile: Shared learned UserPreferenceProfile.
        alpha: Personalization weight (default 0.35).
        lambda_param: Diversity trade-off parameter (default 0.6).
        interleave: Whether to interleave personalized and discovery items (default True).

    Returns:
        pd.DataFrame containing top_k articles with 'feed_lane' and 'feed_badge' labels.
    """
    if df.empty or not user_profile:
        return pd.DataFrame()

    k_personalized = max(1, int(round(top_k * personalized_ratio)))
    k_discovery = max(1, top_k - k_personalized)

    # 1. Generate Personalized Core Candidates
    candidates = score_all_candidates(
        df=df,
        user_profile=user_profile,
        preference_profile=preference_profile,
        alpha=alpha
    )

    if candidates.empty:
        return pd.DataFrame()

    # Generate personalized items using Geographic Diversity MMR
    pers_df = recommend_geographic_mmr(
        candidates_df=candidates,
        top_k=k_personalized,
        lambda_param=lambda_param,
        preference_profile=preference_profile,
        alpha=alpha
    )

    pers_df["feed_lane"] = "personalized"
    pers_df["feed_badge"] = "✦ CURATED FOR YOU"

    selected_ids = set(pers_df["article_id"].tolist())

    # 2. Generate Discovery & Bridging Items (excluding already selected articles)
    remaining_df = df[~df["article_id"].isin(selected_ids)].copy()

    disc_df = recommend_interest_bridging(
        df=remaining_df,
        user_profile=user_profile,
        top_k=k_discovery,
        preference_profile=preference_profile,
        alpha=alpha
    )

    # If bridging pool is smaller than needed, fill with novel geographic/publisher articles
    if len(disc_df) < k_discovery:
        needed = k_discovery - len(disc_df)
        existing_ids = selected_ids.union(set(disc_df["article_id"].tolist())) if not disc_df.empty else selected_ids
        backfill_candidates = remaining_df[~remaining_df["article_id"].isin(existing_ids)].copy()

        # Score on baseline and pick long-tail / high-diversity items
        if not backfill_candidates.empty:
            backfill_scored = score_all_candidates(
                backfill_candidates, user_profile, preference_profile, alpha
            )
            backfill_items = backfill_scored.head(needed).copy()
            disc_df = pd.concat([disc_df, backfill_items], ignore_index=True)

    if not disc_df.empty:
        disc_df["feed_lane"] = "discovery"
        disc_df["feed_badge"] = "✦ DISCOVERY PERSPECTIVE"

    # 3. Interleaving or Concatenation
    if interleave and not disc_df.empty:
        combined_records: List[Dict[str, Any]] = []
        p_list = pers_df.to_dict("records")
        d_list = disc_df.to_dict("records")

        p_idx, d_idx = 0, 0
        total_needed = min(top_k, len(p_list) + len(d_list))

        # Pattern: e.g. 2 personalized, 1 discovery, 2 personalized, 1 discovery...
        step = 0
        while len(combined_records) < total_needed:
            step += 1
            if step % 3 == 0 and d_idx < len(d_list):
                combined_records.append(d_list[d_idx])
                d_idx += 1
            elif p_idx < len(p_list):
                combined_records.append(p_list[p_idx])
                p_idx += 1
            elif d_idx < len(d_list):
                combined_records.append(d_list[d_idx])
                d_idx += 1
            else:
                break

        result_df = pd.DataFrame(combined_records)
    else:
        result_df = pd.concat([pers_df, disc_df], ignore_index=True).head(top_k)

    result_df["selection_rank"] = range(1, len(result_df) + 1)
    return result_df

