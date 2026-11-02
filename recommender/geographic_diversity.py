"""
Geographic Diversity Re-Ranking Engine for Beyond the Feed.

This module implements a Maximal Marginal Relevance (MMR) recommendation
algorithm that mitigates geographic coverage disparities (e.g. over-concentration
in US/UK/India news) by balancing user topic relevance with country- and
continent-level geographic diversity.

Mathematical Formulation:
    Score(a, S) = lambda * Rel(u, a) - (1 - lambda) * Sim_geo(a, S)

Where:
    - Rel(u, a) is the baseline topic relevance score.
    - Sim_geo(a, S) is the maximum geographic redundancy relative to already selected slate S.
    - lambda in [0, 1] parameterizes the relevance vs. diversity trade-off.
    - S is the slate of articles selected so far (|S| < K).
"""

from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import pandas as pd


# Canonical invalid/missing geographic value tokens (case-insensitive)
INVALID_GEO_VALUES = {"", "unknown", "none", "null", "undetermined", "n/a", "missing"}


def is_valid_geo(value: Optional[str]) -> bool:
    """
    Check if a geographic entity string is valid, recognized, and non-empty.

    Returns:
        bool: False for empty strings, None, and unknown/null markers.
              Preserves valid country names and ISO codes, including Namibia ('NA').
    """
    if value is None:
        return False
    s = str(value).strip().lower()
    return bool(s) and (s not in INVALID_GEO_VALUES) and (s != "nan")


def calculate_geo_redundancy(
    candidate: Dict[str, Any],
    selected_items: List[Dict[str, Any]],
    weight_country: float = 1.0,
    weight_continent: float = 0.4,
    weight_secondary: float = 0.1,
    weight_missing_geo: float = 0.5
) -> Tuple[float, str]:
    """
    Compute geographic redundancy (similarity) of a candidate relative to
    articles already selected in slate S.

    Formula:
        Sim_geo(a, S) = max_{s in S} sim_geo(a, s)

    Hierarchy:
        1. Missing / Unknown primary geography: sim = weight_missing_geo (0.5 neutral penalty).
           Ensures un-geotagged articles receive neither an artificial novelty bonus (0.0)
           nor an unfair disqualification (1.0).
        2. Primary country match: sim = weight_country (1.0)
        3. Continent match: sim = weight_continent (0.4)
        4. Secondary country overlap (optional): sim = weight_secondary (0.1)
        5. Distinct continent & country: sim = 0.0 (novel geography)

    Args:
        candidate: Candidate article record dictionary.
        selected_items: List of article records already selected in slate S.
        weight_country: Penalty for primary country repetition (default 1.0).
        weight_continent: Penalty for continent repetition (default 0.4).
        weight_secondary: Penalty for secondary mentioned country overlap (default 0.1).
        weight_missing_geo: Neutral penalty for un-geotagged candidates (default 0.5).

    Returns:
        Tuple of (sim_geo: float, reason: str)
    """
    if not selected_items:
        return 0.0, "Initial selection (empty slate)"

    c_country = candidate.get("country", "")
    c_continent = candidate.get("continent", "")
    c_secondary = candidate.get("all_countries_set", set())

    # Check if candidate has valid primary geographic attribution
    if not is_valid_geo(c_country):
        return (
            weight_missing_geo,
            f"Missing or unknown primary geography ('{c_country}'; assigned neutral penalty {weight_missing_geo})"
        )

    max_sim = 0.0
    reason = "Distinct continent and country (novel geography)"

    for s in selected_items:
        s_country = s.get("country", "")
        s_continent = s.get("continent", "")
        s_secondary = s.get("all_countries_set", set())

        # Level 1: Primary country match (only between valid recognized countries)
        if is_valid_geo(s_country) and c_country == s_country:
            sim = weight_country
            if sim >= max_sim:
                max_sim = sim
                reason = f"Country redundancy with {s['article_id']} ({s_country})"
            continue

        # Level 2: Continent match (only between valid recognized continents)
        if is_valid_geo(c_continent) and is_valid_geo(s_continent) and c_continent == s_continent:
            sim = weight_continent
            if sim > max_sim:
                max_sim = sim
                reason = f"Continent redundancy with {s['article_id']} ({s_continent})"

        # Level 3: Secondary country overlap (only over valid entities)
        if weight_secondary > 0 and c_secondary and s_secondary:
            valid_c_sec = {c for c in c_secondary if is_valid_geo(c)}
            valid_s_sec = {c for c in s_secondary if is_valid_geo(c)}
            overlap = valid_c_sec.intersection(valid_s_sec)
            if overlap:
                sim = weight_secondary
                if sim > max_sim:
                    max_sim = sim
                    reason = f"Secondary country overlap with {s['article_id']} ({', '.join(sorted(overlap))})"

    return max_sim, reason


def generate_natural_explanation(
    rank: int,
    country: Optional[str],
    continent: Optional[str],
    matched_topics: List[str],
    sim_geo: float
) -> str:
    """
    Generate an engaging, natural-language explanation communicating the
    geographic and topical contribution to a user.
    """
    topics_str = ", ".join(matched_topics) if matched_topics else "your interests"
    has_country = is_valid_geo(country)
    has_continent = is_valid_geo(continent)

    if rank == 1:
        if has_country:
            return f"Top recommendation matching your interest in {topics_str}, from {country}."
        return f"Top recommendation matching your interest in {topics_str}."

    if not has_country:
        return f"Curated for relevance based on your interest in {topics_str}."

    if sim_geo == 0.0:
        if has_continent:
            return f"Expanding your feed with a fresh perspective from {country} ({continent}), matching your interest in {topics_str}."
        return f"Expanding geographical representation with coverage from {country}."

    if sim_geo <= 0.4:
        return f"Bringing an important regional angle from {country}, matching your interest in {topics_str}."

    return f"Highly relevant story from {country} aligned with your interest in {topics_str}."


def recommend_geographic_mmr(
    candidates_df: pd.DataFrame,
    top_k: int = 10,
    lambda_param: float = 0.6,
    top_m_candidates: Optional[int] = 500,
    weight_country: float = 1.0,
    weight_continent: float = 0.4,
    weight_secondary: float = 0.1,
    weight_missing_geo: float = 0.5
) -> pd.DataFrame:
    """
    Perform diversity-aware re-ranking using Maximal Marginal Relevance (MMR).

    Args:
        candidates_df: DataFrame with 'relevance_score' computed and > 0.
        top_k: Number of recommendations to select (default 10).
        lambda_param: Trade-off parameter between relevance and diversity in [0, 1].
        top_m_candidates: Candidate pool size considered before MMR (default 500).
        weight_country: Penalty for primary country redundancy (default 1.0).
        weight_continent: Penalty for continent redundancy (default 0.4).
        weight_secondary: Penalty for secondary country overlap (default 0.1).
        weight_missing_geo: Neutral penalty for un-geotagged records (default 0.5).

    Returns:
        pd.DataFrame containing top_k recommendations with MMR scores and audit traces.
    """
    if candidates_df.empty:
        return pd.DataFrame()

    pool = candidates_df.copy()

    # Ensure secondary countries set exists for fast lookup
    if "all_countries_set" not in pool.columns:
        if "all_countries" in pool.columns:
            pool["all_countries_set"] = pool["all_countries"].apply(
                lambda x: {c.strip() for c in str(x).split(";") if c.strip()}
            )
        else:
            pool["all_countries_set"] = [set() for _ in range(len(pool))]

    # Deterministic sort: relevance descending, article_id ascending
    ranked_pool = pool.sort_values(
        by=["relevance_score", "article_id"],
        ascending=[False, True]
    )

    # Candidate pool sizing: ensure M >= K
    if top_m_candidates is not None and top_m_candidates > 0:
        actual_m = max(top_m_candidates, top_k)
        ranked_pool = ranked_pool.head(actual_m)

    candidate_records = ranked_pool.to_dict("records")
    selected_items: List[Dict[str, Any]] = []

    while len(selected_items) < top_k and candidate_records:
        if not selected_items:
            # First item selected by top relevance
            best_idx = 0
            best_item = candidate_records.pop(best_idx)
            best_item["mmr_score"] = float(best_item["relevance_score"])
            best_item["geo_redundancy"] = 0.0
            best_item["relevance_contribution"] = round(lambda_param * float(best_item["relevance_score"]), 4)
            best_item["diversity_penalty"] = 0.0
            best_item["geo_explanation"] = "First selection (top relevance)"
            best_item["selection_rank"] = 1
            selected_items.append(best_item)
            continue

        best_idx = -1
        best_mmr = -float("inf")
        best_redundancy = 0.0
        best_explanation = ""

        for idx, candidate in enumerate(candidate_records):
            sim_geo, reason = calculate_geo_redundancy(
                candidate=candidate,
                selected_items=selected_items,
                weight_country=weight_country,
                weight_continent=weight_continent,
                weight_secondary=weight_secondary,
                weight_missing_geo=weight_missing_geo
            )

            rel = float(candidate["relevance_score"])
            mmr = (lambda_param * rel) - ((1.0 - lambda_param) * sim_geo)

            # Deterministic tie-breaking:
            # 1. Higher MMR score
            # 2. Higher baseline relevance
            # 3. Lexicographically smaller article_id
            if (
                best_idx == -1
                or mmr > best_mmr + 1e-9
                or (
                    abs(mmr - best_mmr) <= 1e-9
                    and rel > candidate_records[best_idx]["relevance_score"] + 1e-9
                )
                or (
                    abs(mmr - best_mmr) <= 1e-9
                    and abs(rel - candidate_records[best_idx]["relevance_score"]) <= 1e-9
                    and str(candidate["article_id"]) < str(candidate_records[best_idx]["article_id"])
                )
            ):
                best_mmr = mmr
                best_idx = idx
                best_redundancy = sim_geo
                best_explanation = reason

        chosen = candidate_records.pop(best_idx)
        chosen["mmr_score"] = round(best_mmr, 4)
        chosen["geo_redundancy"] = round(best_redundancy, 4)
        chosen["relevance_contribution"] = round(lambda_param * float(chosen["relevance_score"]), 4)
        chosen["diversity_penalty"] = round((1.0 - lambda_param) * best_redundancy, 4)
        chosen["geo_explanation"] = best_explanation
        chosen["selection_rank"] = len(selected_items) + 1
        selected_items.append(chosen)

    result_df = pd.DataFrame(selected_items)
    if "all_countries_set" in result_df.columns:
        result_df = result_df.drop(columns=["all_countries_set"])

    return result_df


def compute_hhi(series: pd.Series) -> float:
    """
    Compute Herfindahl-Hirschman Index (HHI) for a category series.
    Formula: sum(p_i^2) where p_i is the share of occurrences.
    Scale: [1/N, 1.0], where 1/N represents perfect uniformity.
    """
    n = len(series)
    if n == 0:
        return 0.0
    shares = series.value_counts().to_numpy() / n
    return float(np.sum(shares ** 2))


def evaluate_slate(slate_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary metrics for a recommended Top-K slate.
    """
    if slate_df.empty:
        return {
            "avg_relevance": 0.0,
            "unique_countries": 0,
            "unique_continents": 0,
            "unique_publishers": 0,
            "country_hhi": 0.0,
            "country_distribution": {},
            "continent_distribution": {}
        }

    return {
        "avg_relevance": float(round(slate_df["relevance_score"].mean(), 4)),
        "unique_countries": int(slate_df["country"].nunique()),
        "unique_continents": int(slate_df["continent"].nunique()),
        "unique_publishers": int(slate_df["publisher"].nunique()),
        "country_hhi": float(round(compute_hhi(slate_df["country"]), 4)),
        "country_distribution": dict(slate_df["country"].value_counts()),
        "continent_distribution": dict(slate_df["continent"].value_counts())
    }


def compare_baseline_vs_mmr(
    baseline_df: pd.DataFrame,
    mmr_df: pd.DataFrame,
    profile_name: str,
    lambda_param: float = 0.6
) -> pd.DataFrame:
    """
    Generate side-by-side comparative evaluation metrics table.
    """
    base_m = evaluate_slate(baseline_df)
    mmr_m = evaluate_slate(mmr_df)

    rel_deg = 0.0
    if base_m["avg_relevance"] > 0:
        rel_deg = (
            (base_m["avg_relevance"] - mmr_m["avg_relevance"])
            / base_m["avg_relevance"]
        ) * 100.0

    hhi_country_change = 0.0
    if base_m["country_hhi"] > 0:
        hhi_country_change = (
            (mmr_m["country_hhi"] - base_m["country_hhi"])
            / base_m["country_hhi"]
        ) * 100.0

    return pd.DataFrame([
        {
            "Profile": profile_name,
            "Metric": "Average Relevance",
            "Baseline (Top-10)": base_m["avg_relevance"],
            f"MMR (λ={lambda_param})": mmr_m["avg_relevance"],
            "Absolute Change": round(mmr_m["avg_relevance"] - base_m["avg_relevance"], 4),
            "Relative Change (%)": f"{rel_deg:+.2f}% degradation" if rel_deg > 0 else "0.00%"
        },
        {
            "Profile": profile_name,
            "Metric": "Unique Primary Countries",
            "Baseline (Top-10)": base_m["unique_countries"],
            f"MMR (λ={lambda_param})": mmr_m["unique_countries"],
            "Absolute Change": mmr_m["unique_countries"] - base_m["unique_countries"],
            "Relative Change (%)": f"+{((mmr_m['unique_countries'] - base_m['unique_countries']) / max(base_m['unique_countries'], 1)) * 100:.1f}%"
        },
        {
            "Profile": profile_name,
            "Metric": "Unique Continents",
            "Baseline (Top-10)": base_m["unique_continents"],
            f"MMR (λ={lambda_param})": mmr_m["unique_continents"],
            "Absolute Change": mmr_m["unique_continents"] - base_m["unique_continents"],
            "Relative Change (%)": f"+{((mmr_m['unique_continents'] - base_m['unique_continents']) / max(base_m['unique_continents'], 1)) * 100:.1f}%"
        },
        {
            "Profile": profile_name,
            "Metric": "Geographic Concentration (Country HHI)",
            "Baseline (Top-10)": base_m["country_hhi"],
            f"MMR (λ={lambda_param})": mmr_m["country_hhi"],
            "Absolute Change": round(mmr_m["country_hhi"] - base_m["country_hhi"], 4),
            "Relative Change (%)": f"{hhi_country_change:.1f}% concentration"
        },
        {
            "Profile": profile_name,
            "Metric": "Unique Publishers",
            "Baseline (Top-10)": base_m["unique_publishers"],
            f"MMR (λ={lambda_param})": mmr_m["unique_publishers"],
            "Absolute Change": mmr_m["unique_publishers"] - base_m["unique_publishers"],
            "Relative Change (%)": f"+{((mmr_m['unique_publishers'] - base_m['unique_publishers']) / max(base_m['unique_publishers'], 1)) * 100:.1f}%"
        }
    ])
