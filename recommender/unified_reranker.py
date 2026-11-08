
"""Unified relevance and diversity reranking for Beyond the Feed."""

from typing import Any, Dict, List

from recommender.geographic_diversity import calculate_geo_redundancy
from recommender.publisher_diversity import calculate_publisher_redundancy

import numpy as np


def calculate_semantic_redundancy(
    candidate_embedding: np.ndarray,
    selected_embeddings: np.ndarray,
) -> float:
    """Return maximum cosine similarity to already selected articles."""
    if len(selected_embeddings) == 0:
        return 0.0

    candidate = np.asarray(candidate_embedding, dtype=np.float32)
    selected = np.asarray(selected_embeddings, dtype=np.float32)

    candidate_norm = np.linalg.norm(candidate)
    selected_norms = np.linalg.norm(selected, axis=1)

    if candidate_norm == 0:
        return 0.0

    valid = selected_norms > 0
    if not np.any(valid):
        return 0.0

    similarities = (
        selected[valid] @ candidate
        / (selected_norms[valid] * candidate_norm)
    )

    return float(np.clip(np.max(similarities), 0.0, 1.0))



def calculate_combined_redundancy(
    candidate: Dict[str, Any],
    selected_items: List[Dict[str, Any]],
    candidate_embedding: np.ndarray,
    selected_embeddings: np.ndarray,
    semantic_weight: float = 0.5,
    geographic_weight: float = 0.3,
    publisher_weight: float = 0.2,
) -> Dict[str, float]:
    """Combine semantic, geographic, and publisher redundancy."""

    weights = [semantic_weight, geographic_weight, publisher_weight]

    if any(w < 0 for w in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("Diversity weights must be nonnegative and sum to 1.")

    semantic = calculate_semantic_redundancy(
        candidate_embedding, selected_embeddings
    )

    geographic, _ = calculate_geo_redundancy(
        candidate, selected_items
    )

    publisher, _ = calculate_publisher_redundancy(
        candidate, selected_items
    )

    combined = (
        semantic_weight * semantic
        + geographic_weight * geographic
        + publisher_weight * publisher
    )

    return {
        "semantic": float(semantic),
        "geographic": float(geographic),
        "publisher": float(publisher),
        "combined": float(combined),
    }
    

def recommend_unified_mmr(
    candidates_df,
    embeddings,
    top_k=10,
    lambda_param=0.7,
    candidate_pool_size=300,
):
    """Select articles using hybrid relevance and unified diversity MMR."""
    import pandas as pd

    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1.")
    if top_k < 0 or candidate_pool_size < 0:
        raise ValueError("top_k and candidate_pool_size must be nonnegative.")

    if candidates_df.empty or top_k == 0 or candidate_pool_size == 0:
        return candidates_df.iloc[0:0].copy()

    if "embedding_idx" not in candidates_df.columns:
        raise ValueError(
            "Candidates must include embedding_idx aligned with the original CSV."
        )

    if "hybrid_relevance" not in candidates_df.columns:
        raise ValueError("Candidates must include hybrid_relevance.")

    pool = (
        candidates_df.sort_values(
            ["hybrid_relevance", "article_id"],
            ascending=[False, True],
            kind="stable",
        )
        .head(candidate_pool_size)
        .copy()
    )

    embedding_indices = pool["embedding_idx"].to_numpy(dtype=int)

    if (
        np.any(embedding_indices < 0)
        or np.any(embedding_indices >= len(embeddings))
    ):
        raise ValueError("Invalid embedding_idx found.")

    pool_records = pool.to_dict("records")
    pool_embeddings = np.asarray(embeddings[embedding_indices])

    selected_positions = []
    remaining_positions = list(range(len(pool_records)))

    while remaining_positions and len(selected_positions) < top_k:
        selected_items = [pool_records[i] for i in selected_positions]

        selected_embeddings = (
            pool_embeddings[selected_positions]
            if selected_positions
            else np.empty((0, pool_embeddings.shape[1]))
        )

        best_position = None
        best_score = float("-inf")
        best_redundancy = None

        for position in remaining_positions:
            candidate = pool_records[position]

            redundancy = calculate_combined_redundancy(
                candidate,
                selected_items,
                pool_embeddings[position],
                selected_embeddings,
            )

            mmr_score = (
                lambda_param * float(candidate["hybrid_relevance"])
                - (1 - lambda_param) * redundancy["combined"]
            )

            if mmr_score > best_score:
                best_position = position
                best_score = mmr_score
                best_redundancy = redundancy

        selected = pool_records[best_position].copy()
        selected["mmr_score"] = best_score
        selected["semantic_redundancy"] = best_redundancy["semantic"]
        selected["geographic_redundancy"] = best_redundancy["geographic"]
        selected["publisher_redundancy"] = best_redundancy["publisher"]
        selected["combined_redundancy"] = best_redundancy["combined"]
        selected["selection_rank"] = len(selected_positions) + 1

        pool_records[best_position] = selected
        selected_positions.append(best_position)
        remaining_positions.remove(best_position)

    return pd.DataFrame(
        [pool_records[i] for i in selected_positions]
    ).reset_index(drop=True)

