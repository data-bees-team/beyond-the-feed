"""
Publisher Diversity Re-Ranking Engine for Beyond the Feed.

This module implements a Maximal Marginal Relevance (MMR) recommendation
algorithm that mitigates media concentration and publisher monopolies (e.g.
over-representation of dominant outlets like iheart.com, indiatimes.com, aol.com)
by balancing user topic relevance with publisher pluralism.

Mathematical Formulation:
    Score(a, S) = lambda * Rel(u, a) - (1 - lambda) * Redundancy_pub(a, S)

Where:
    - Rel(u, a) is the baseline topic relevance score (or feedback-personalized score).
    - Redundancy_pub(a, S) is the publisher redundancy relative to already selected slate S.
    - lambda in [0, 1] parameterizes the relevance vs. publisher diversity trade-off.
    - S is the slate of articles selected so far (|S| < K).
"""

from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import pandas as pd


# Canonical invalid/missing publisher value tokens (case-insensitive)
INVALID_PUB_VALUES = {"", "unknown", "none", "null", "undetermined", "n/a", "missing", "nan"}


def is_valid_publisher(value: Optional[str]) -> bool:
    """
    Check if a publisher entity string is valid, recognized, and non-empty.

    Returns:
        bool: False for empty strings, None, and unknown/null markers.
    """
    if value is None:
        return False
    s = str(value).strip().lower()
    return bool(s) and (s not in INVALID_PUB_VALUES)


def calculate_publisher_redundancy(
    candidate: Dict[str, Any],
    selected_items: List[Dict[str, Any]],
    weight_repeat_publisher: float = 1.0,
    weight_missing_publisher: float = 0.5
) -> Tuple[float, str]:
    """
    Compute publisher redundancy of a candidate relative to articles already
    selected in slate S.

    Redundancy Policy:
        1. Missing / Unknown publisher: penalty = weight_missing_publisher (default 0.5 neutral).
           Ensures articles without publisher metadata receive neither an artificial novelty
           bonus (0.0) nor an unfair disqualification (1.0).
        2. First appearance in slate S: redundancy = 0.0 (novel media voice encouraged).
        3. Already represented in slate S: redundancy = weight_repeat_publisher (default 1.0).

    Args:
        candidate: Candidate article record dictionary.
        selected_items: List of article records already selected in slate S.
        weight_repeat_publisher: Penalty applied when publisher is already present (default 1.0).
        weight_missing_publisher: Neutral penalty for un-attributed candidates (default 0.5).

    Returns:
        Tuple of (redundancy_penalty: float, explanation_reason: str)
    """
    if not selected_items:
        return 0.0, "Initial selection (empty slate)"

    c_pub = candidate.get("publisher", "")

    # Handle missing / unverified publisher metadata
    if not is_valid_publisher(c_pub):
        return (
            weight_missing_publisher,
            f"Missing or unknown publisher ('{c_pub}'; assigned neutral penalty {weight_missing_publisher})"
        )

    # Check how many times this publisher has already been selected
    prior_matches = [
        s for s in selected_items
        if is_valid_publisher(s.get("publisher", "")) and str(s.get("publisher", "")).strip().lower() == str(c_pub).strip().lower()
    ]
    count = len(prior_matches)

    if count == 0:
        return 0.0, f"Novel publisher ('{c_pub}') introducing a fresh media viewpoint"

    first_match_id = prior_matches[0].get("article_id", "prior selection")
    return (
        weight_repeat_publisher,
        f"Publisher repetition with {first_match_id} ('{c_pub}' already selected {count} time(s))"
    )


def generate_natural_publisher_explanation(
    rank: int,
    publisher: Optional[str],
    matched_topics: List[str],
    pub_redundancy: float
) -> str:
    """
    Generate an engaging, natural-language explanation communicating the
    editorial source diversity and topical relevance to the user.
    """
    topics_str = ", ".join(matched_topics) if matched_topics else "your interests"
    has_pub = is_valid_publisher(publisher)

    if rank == 1:
        if has_pub:
            return f"Top recommendation matching your interest in {topics_str}, published by {publisher}."
        return f"Top recommendation matching your interest in {topics_str}."

    if not has_pub:
        return f"Curated for relevance based on your interest in {topics_str}."

    if pub_redundancy == 0.0:
        return f"Broadening your news diet with a fresh editorial perspective from {publisher}, matching your interest in {topics_str}."

    return f"High-relevance story from {publisher} aligned with your interest in {topics_str}."


def recommend_publisher_mmr(
    candidates_df: pd.DataFrame,
    top_k: int = 10,
    lambda_param: float = 0.6,
    top_m_candidates: Optional[int] = 500,
    weight_repeat_publisher: float = 1.0,
    weight_missing_publisher: float = 0.5,
    preference_profile: Optional[Any] = None,
    alpha: float = 0.35
) -> pd.DataFrame:
    """
    Perform publisher diversity-aware re-ranking using Maximal Marginal Relevance (MMR).

    Args:
        candidates_df: DataFrame with 'relevance_score' computed and > 0.
        top_k: Number of recommendations to select (default 10).
        lambda_param: Trade-off parameter between relevance and publisher diversity in [0, 1].
                      lambda = 1.0 reduces to pure topic relevance (baseline).
                      lambda = 0.0 maximizes publisher uniqueness ignoring relevance.
                      lambda = 0.6 is the recommended balanced default.
        top_m_candidates: Pre-filtered candidate pool size sorted by relevance (default 500).
        weight_repeat_publisher: Penalty applied when publisher is already selected (default 1.0).
        weight_missing_publisher: Penalty applied to missing/unknown publishers (default 0.5).
        preference_profile: Optional UserPreferenceProfile instance for personalized scoring.
        alpha: Weight for personalization adjustment (default 0.35).

    Returns:
        pd.DataFrame: Re-ranked top_k articles with MMR scores, publisher redundancy,
                      contributions, explanations, and selection ranks.
    """
    if candidates_df.empty or top_k <= 0:
        return pd.DataFrame()

    pool = candidates_df.copy()

    # Apply personalized scoring if preference profile is provided
    rel_col = "relevance_score"
    if preference_profile is not None and getattr(preference_profile, "total_feedback_count", 0) > 0:
        try:
            from recommender.personalization_engine import score_personalized_candidate
            pool["personalized_relevance"] = pool.apply(
                lambda row: score_personalized_candidate(
                    base_relevance=float(row.get("relevance_score", 0.0)),
                    article_record=row.to_dict(),
                    profile=preference_profile,
                    alpha=alpha
                ),
                axis=1
            )
            rel_col = "personalized_relevance"
        except ImportError:
            pass

    # Ensure sorting by active relevance score and deterministic tie-breaker
    pool = pool.sort_values(
        by=[rel_col, "article_id"],
        ascending=[False, True]
    )

    if top_m_candidates is not None and top_m_candidates > 0:
        pool = pool.head(top_m_candidates)

    candidate_records: List[Dict[str, Any]] = pool.to_dict("records")
    selected_items: List[Dict[str, Any]] = []

    while len(selected_items) < top_k and candidate_records:
        if not selected_items:
            # First item: top relevance anchor
            best_item = candidate_records.pop(0)
            item_rel = float(best_item[rel_col])
            best_item["mmr_score"] = item_rel
            best_item["pub_redundancy"] = 0.0
            best_item["relevance_contribution"] = round(lambda_param * item_rel, 4)
            best_item["diversity_penalty"] = 0.0
            best_item["pub_explanation"] = "First selection (top relevance)"

            matched_topics = [t for t in best_item.get("topic_list", []) if t not in {"Unknown", "Other"}]
            best_item["reason"] = generate_natural_publisher_explanation(
                rank=1,
                publisher=best_item.get("publisher"),
                matched_topics=matched_topics,
                pub_redundancy=0.0
            )
            best_item["selection_rank"] = 1
            selected_items.append(best_item)
            continue

        best_idx = -1
        best_mmr = -float("inf")
        best_redundancy = 0.0
        best_explanation = ""

        for idx, candidate in enumerate(candidate_records):
            redundancy, reason = calculate_publisher_redundancy(
                candidate=candidate,
                selected_items=selected_items,
                weight_repeat_publisher=weight_repeat_publisher,
                weight_missing_publisher=weight_missing_publisher
            )

            rel = float(candidate[rel_col])
            mmr = (lambda_param * rel) - ((1.0 - lambda_param) * redundancy)

            # Deterministic tie-breaking:
            # 1. Higher MMR score
            # 2. Higher active relevance
            # 3. Lexicographically smaller article_id
            if (
                best_idx == -1
                or mmr > best_mmr + 1e-9
                or (
                    abs(mmr - best_mmr) <= 1e-9
                    and rel > float(candidate_records[best_idx][rel_col]) + 1e-9
                )
                or (
                    abs(mmr - best_mmr) <= 1e-9
                    and abs(rel - float(candidate_records[best_idx][rel_col])) <= 1e-9
                    and str(candidate["article_id"]) < str(candidate_records[best_idx]["article_id"])
                )
            ):
                best_mmr = mmr
                best_idx = idx
                best_redundancy = redundancy
                best_explanation = reason

        chosen = candidate_records.pop(best_idx)
        chosen_rel = float(chosen[rel_col])
        chosen["mmr_score"] = round(best_mmr, 4)
        chosen["pub_redundancy"] = round(best_redundancy, 4)
        chosen["relevance_contribution"] = round(lambda_param * chosen_rel, 4)
        chosen["diversity_penalty"] = round((1.0 - lambda_param) * best_redundancy, 4)
        chosen["pub_explanation"] = best_explanation

        matched_topics = [t for t in chosen.get("topic_list", []) if t not in {"Unknown", "Other"}]
        chosen["reason"] = generate_natural_publisher_explanation(
            rank=len(selected_items) + 1,
            publisher=chosen.get("publisher"),
            matched_topics=matched_topics,
            pub_redundancy=best_redundancy
        )
        chosen["selection_rank"] = len(selected_items) + 1
        selected_items.append(chosen)

    return pd.DataFrame(selected_items)


def compute_publisher_hhi(series: pd.Series) -> float:
    """
    Compute Herfindahl-Hirschman Index (HHI) for a publisher series.
    Formula: sum(p_i^2) where p_i is the share of occurrences.
    Scale: [1/N, 1.0], where 1/N represents perfect uniformity (maximal diversity)
    and 1.0 represents complete monopoly by a single publisher.
    """
    n = len(series)
    if n == 0:
        return 0.0
    shares = series.value_counts().to_numpy() / n
    return float(np.sum(shares ** 2))


def evaluate_publisher_slate(slate_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive diversity and relevance metrics for a recommended slate.
    """
    if slate_df.empty:
        return {
            "avg_relevance": 0.0,
            "unique_publishers": 0,
            "publisher_hhi": 0.0,
            "top_publisher_share": 0.0,
            "unique_countries": 0,
            "publisher_distribution": {}
        }

    n = len(slate_df)
    pub_counts = slate_df["publisher"].value_counts()
    top_share = float(pub_counts.iloc[0] / n) if not pub_counts.empty else 0.0

    return {
        "avg_relevance": float(round(slate_df["relevance_score"].mean(), 4)),
        "unique_publishers": int(slate_df["publisher"].nunique()),
        "publisher_hhi": float(round(compute_publisher_hhi(slate_df["publisher"]), 4)),
        "top_publisher_share": float(round(top_share * 100, 2)),
        "unique_countries": int(slate_df["country"].nunique()),
        "publisher_distribution": dict(pub_counts)
    }


def compare_baseline_vs_publisher_mmr(
    baseline_df: pd.DataFrame,
    mmr_df: pd.DataFrame,
    profile_name: str,
    lambda_param: float = 0.6
) -> pd.DataFrame:
    """
    Generate side-by-side comparative evaluation metrics table for baseline vs publisher MMR.
    """
    base_m = evaluate_publisher_slate(baseline_df)
    mmr_m = evaluate_publisher_slate(mmr_df)

    rel_deg = 0.0
    if base_m["avg_relevance"] > 0:
        rel_deg = (
            (base_m["avg_relevance"] - mmr_m["avg_relevance"])
            / base_m["avg_relevance"]
        ) * 100.0

    hhi_change = 0.0
    if base_m["publisher_hhi"] > 0:
        hhi_change = (
            (mmr_m["publisher_hhi"] - base_m["publisher_hhi"])
            / base_m["publisher_hhi"]
        ) * 100.0

    col_name = f"MMR (lambda={lambda_param})"
    return pd.DataFrame([
        {
            "Profile": profile_name,
            "Metric": "Average Relevance",
            "Baseline (Top-10)": base_m["avg_relevance"],
            col_name: mmr_m["avg_relevance"],
            "Absolute Change": round(mmr_m["avg_relevance"] - base_m["avg_relevance"], 4),
            "Relative Change (%)": f"{rel_deg:+.2f}% degradation" if rel_deg > 0 else "0.00%"
        },
        {
            "Profile": profile_name,
            "Metric": "Unique Publishers",
            "Baseline (Top-10)": base_m["unique_publishers"],
            col_name: mmr_m["unique_publishers"],
            "Absolute Change": mmr_m["unique_publishers"] - base_m["unique_publishers"],
            "Relative Change (%)": f"+{((mmr_m['unique_publishers'] - base_m['unique_publishers']) / max(base_m['unique_publishers'], 1)) * 100:.1f}%"
        },
        {
            "Profile": profile_name,
            "Metric": "Publisher Concentration (HHI)",
            "Baseline (Top-10)": base_m["publisher_hhi"],
            col_name: mmr_m["publisher_hhi"],
            "Absolute Change": round(mmr_m["publisher_hhi"] - base_m["publisher_hhi"], 4),
            "Relative Change (%)": f"{hhi_change:.1f}% concentration"
        },
        {
            "Profile": profile_name,
            "Metric": "Top Publisher Market Share",
            "Baseline (Top-10)": f"{base_m['top_publisher_share']}%",
            col_name: f"{mmr_m['top_publisher_share']}%",
            "Absolute Change": f"{round(mmr_m['top_publisher_share'] - base_m['top_publisher_share'], 2)}%",
            "Relative Change (%)": f"{((mmr_m['top_publisher_share'] - base_m['top_publisher_share']) / max(base_m['top_publisher_share'], 1)) * 100:.1f}%"
        },
        {
            "Profile": profile_name,
            "Metric": "Unique Countries (Preserved)",
            "Baseline (Top-10)": base_m["unique_countries"],
            col_name: mmr_m["unique_countries"],
            "Absolute Change": mmr_m["unique_countries"] - base_m["unique_countries"],
            "Relative Change (%)": f"{((mmr_m['unique_countries'] - base_m['unique_countries']) / max(base_m['unique_countries'], 1)) * 100:.1f}%"
        }
    ])
