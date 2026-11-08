import numpy as np
import time
from recommender.nlp_profile import compute_nlp_content_scores

from recommender.hybrid_retrieval import (
    score_max_topic_match,
    score_description_similarity,
    combine_hybrid_scores,
    deduplicate_candidates,
)
from recommender.unified_reranker import recommend_unified_mmr


def recommend_hybrid_feed(
    df,
    embeddings,
    user_profile,
    topic_description_embeddings,
    top_k=10,
    topic_weight=0.5,
    lambda_param=0.7,
    seen_ids = None,
    user_vector = None,
):
    """Generate a relevance-aware, diversity-balanced news feed."""

    candidates = df.copy()
    candidates["embedding_idx"] = np.arange(len(candidates))

    # 1. Topic relevance
    candidates = score_max_topic_match(
        candidates, user_profile
    )

    # 2. Semantic relevance
    selected_descriptions = [
        topic_description_embeddings[topic]
        for topic, weight in user_profile.items()
        if weight > 0 and topic in topic_description_embeddings
    ]

    if selected_descriptions:
        description_vectors = np.vstack(selected_descriptions)

        candidates = score_description_similarity(
            candidates,
            embeddings,
            description_vectors,
        )
    else:
        candidates["description_semantic_score"] = 0.0

    # 3. Hybrid relevance
    candidates = combine_hybrid_scores(
        candidates,
        topic_weight=topic_weight,
    )
    # Maintain compatibility with existing evaluation metrics
    
        # Feedback-based personalization
    if user_vector is not None and np.any(user_vector):
        feedback_scores = compute_nlp_content_scores(
            user_vector, embeddings
        )

        candidates["feedback_score"] = feedback_scores

        candidates["hybrid_relevance"] = (
            candidates["hybrid_relevance"]
            + 0.15 * candidates["feedback_score"]
        ).clip(0, 1)
    else:
        candidates["feedback_score"] = 0.0

    # Keep evaluation relevance aligned with final scores
    candidates["relevance_score"] = candidates["hybrid_relevance"]
    
    
    
    cross_topic = candidates[
        candidates["hybrid_topic_score"] == 0
    ]

    print(
        "DEBUG CROSS-TOPIC SEMANTIC DISTRIBUTION:",
        cross_topic["description_semantic_score"].quantile(
            [0.50, 0.75, 0.90, 0.95, 0.99]
        ).round(3).to_dict()
    )

    print(
        "DEBUG CROSS-TOPIC MAX:",
        round(cross_topic["description_semantic_score"].max(), 3)
    )
    
    topic_matched = candidates[
        candidates["hybrid_topic_score"] > 0
    ]

    print(
        "DEBUG TOPIC-MATCHED SEMANTIC DISTRIBUTION:",
        topic_matched["description_semantic_score"].quantile(
            [0.10, 0.25, 0.50, 0.75, 0.90]
        ).round(3).to_dict()
    )
    
    
    # 4. Keep articles matching selected topics
    semantic_threshold = 0.65

    candidates = candidates[
        (candidates["hybrid_topic_score"] > 0)
        | (candidates["description_semantic_score"] >= semantic_threshold)
    ].copy()

    # DEBUG: Check candidate types
    print(
        "DEBUG CROSS-TOPIC CANDIDATES:",
        (candidates["hybrid_topic_score"] == 0).sum()
    )

    print(
        "DEBUG TOPIC-MATCHED CANDIDATES:",
        (candidates["hybrid_topic_score"] > 0).sum()
    )

    candidates = deduplicate_candidates(candidates)

    # 5. Remove duplicate headlines
    candidates = deduplicate_candidates(candidates)
    
    # Exclude previously recommended articles before reranking
    if seen_ids:
        seen_set = {str(article_id) for article_id in seen_ids}
        candidates = candidates[
            ~candidates["article_id"].astype(str).isin(seen_set)
        ].copy()
        

    # 6. Unified diversity-aware ranking
    start = time.perf_counter()

    result = recommend_unified_mmr(
        candidates,
        embeddings,
        top_k=top_k,
        lambda_param=lambda_param,
        candidate_pool_size=300,
    )

    elapsed = time.perf_counter() - start
    print(f"Unified MMR took: {elapsed:.2f} seconds")

    return result