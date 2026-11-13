import numpy as np
import pandas as pd
from recommender.unified_reranker import recommend_unified_mmr

from evaluation.semantic_ablation import load_inputs
from recommender.hybrid_retrieval import (
    score_max_topic_match,
    score_description_similarity,
    combine_hybrid_scores,
    deduplicate_candidates,
)

POOL_SIZE = 300
EXPLORATION_SLOTS = 30
SEMANTIC_THRESHOLD = 0.55


def main():
    df, embeddings, topic_embeddings = load_inputs()

    profile = {
        "Technology": 1.0,
        "Health": 1.0,
    }
    candidates = df.copy()
    candidates["embedding_idx"] = np.arange(len(candidates))

    candidates = score_max_topic_match(candidates, profile)

    candidates = score_description_similarity(
        candidates,
        embeddings,
        np.vstack([
            topic_embeddings[topic]
            for topic in profile
        ]),
        
    )
    
    print("\n=== MULTI-TOPIC VERIFICATION ===")
    print("Selected topics:", list(profile.keys()))
    print("Topic embedding matrix shape:", np.vstack([
        topic_embeddings[topic] for topic in profile
    ]).shape)

    candidates = combine_hybrid_scores(
        candidates,
        topic_weight=0.5,
    )

    candidates = deduplicate_candidates(candidates)
    
    print("\n=== SEMANTIC QUALITY THRESHOLD TEST ===")

    matched = candidates[
        candidates["hybrid_topic_score"] > 0
    ]

    for threshold in [0.55, 0.60, 0.65, 0.70]:
        eligible = matched[
            matched["description_semantic_score"] >= threshold
        ]

        print(
            f"Threshold {threshold:.2f}: "
            f"{len(eligible)} eligible articles, "
            f"{eligible['country'].nunique()} countries, "
            f"{eligible['continent'].nunique()} continents"
        )

    topic_matched = candidates[
        candidates["hybrid_topic_score"] > 0
    ].sort_values(
        ["hybrid_relevance", "article_id"],
        ascending=[False, True],
        kind="stable",
    )

    cross_topic = candidates[
        (candidates["hybrid_topic_score"] == 0)
        & (
            candidates["description_semantic_score"]
            >= SEMANTIC_THRESHOLD
        )
    ].sort_values(
        ["description_semantic_score", "article_id"],
        ascending=[False, True],
        kind="stable",
    )

    exploration_pool = topic_matched.head(
        POOL_SIZE - EXPLORATION_SLOTS
    )

    exploration_pool = pd.concat(
    [exploration_pool, cross_topic.head(EXPLORATION_SLOTS)],
    ignore_index=True,
)

    print("Total pool:", len(exploration_pool))
    print(
        "Topic-matched:",
        (exploration_pool["hybrid_topic_score"] > 0).sum(),
    )
    print(
        "Cross-topic:",
        (exploration_pool["hybrid_topic_score"] == 0).sum(),
    )
    print(
        "Cross-topic semantic mean:",
        round(
            exploration_pool.loc[
                exploration_pool["hybrid_topic_score"] == 0,
                "description_semantic_score",
            ].mean(),
            3,
        ),
    )
    
    final = recommend_unified_mmr(
        exploration_pool,
        embeddings,
        top_k=10,
        lambda_param=0.5,
        candidate_pool_size=300,
    )
    
    print("\n=== RECOMMENDATION QUALITY REVIEW ===")

    for _, article in final.iterrows():
        print(
            f"\nTitle: {article['title']}"
            f"\nTopics: {article['topic_list']}"
            f"\nSemantic: {article['description_semantic_score']:.3f}"
            f"\nCountry: {article['country']}"
            f"\nPublisher: {article['publisher']}"
        )

    print("\n=== EXPLORATION FINAL TOP 10 ===")
    print(
        "Cross-topic articles:",
        int((final["hybrid_topic_score"] == 0).sum()),
    )

    for _, article in final.iterrows():
        print(
            f"{article['hybrid_topic_score']:.1f} | "
            f"{article['description_semantic_score']:.3f} | "
            f"{article['title']}"
        )
        
    print("\n=== ONE-SLOT EXPLORATION TEST ===")

    top_cross = cross_topic.head(1)

    experimental_final = pd.concat(
        [final.head(9), top_cross],
        ignore_index=True,
    )

    print(
        "Cross-topic articles:",
        int((experimental_final["hybrid_topic_score"] == 0).sum()),
    )

    for _, article in experimental_final.iterrows():
        print(
            f"{article['description_semantic_score']:.3f} | "
            f"{article['title']}"
        )
        
    print("\n=== EXPLORATION CANDIDATE REVIEW ===")

    for _, article in cross_topic.head(10).iterrows():
        print(
            f"\nSimilarity: {article['description_semantic_score']:.3f}"
            f"\nTitle: {article['title']}"
            f"\nTopics: {article['topic_list']}"
        )
        
    print("\n=== EXPLORATION REDUNDANCY TEST ===")

    selected_indices = final.head(9)["embedding_idx"].astype(int).to_numpy()
    selected_vectors = embeddings[selected_indices]

    selected_vectors = selected_vectors / np.maximum(
        np.linalg.norm(selected_vectors, axis=1, keepdims=True),
        1e-12,
    )

    for _, article in cross_topic.head(10).iterrows():
        idx = int(article["embedding_idx"])
        vector = embeddings[idx]
        vector = vector / max(np.linalg.norm(vector), 1e-12)

        max_similarity = float(
            np.max(selected_vectors @ vector)
        )

        print(
            f"Semantic relevance: {article['description_semantic_score']:.3f} | "
            f"Max redundancy: {max_similarity:.3f} | "
            f"{article['title']}"
        )
        
    print("\n=== EXPLORATION TRADEOFF ===")

    results = []

    for _, article in cross_topic.head(30).iterrows():
        idx = int(article["embedding_idx"])
        vector = embeddings[idx]
        vector = vector / max(np.linalg.norm(vector), 1e-12)

        redundancy = float(np.max(selected_vectors @ vector))
        relevance = float(article["description_semantic_score"])

        tradeoff = 0.7 * relevance - 0.3 * redundancy

        results.append({
            "title": article["title"],
            "relevance": relevance,
            "redundancy": redundancy,
            "tradeoff": tradeoff,
        })

    results = sorted(
        results,
        key=lambda x: x["tradeoff"],
        reverse=True,
    )

    for item in results[:10]:
        print(
            f"{item['tradeoff']:.3f} | "
            f"Rel: {item['relevance']:.3f} | "
            f"Red: {item['redundancy']:.3f} | "
            f"{item['title']}"
        )
        
        
    print("\n=== HUMAN-APPROVED EXPLORATION TEST ===")

    approved = cross_topic[
        cross_topic["title"].str.contains(
            "lithium-ion battery systems in submarines",
            case=False,
            regex=False,
            na=False,
        )
    ]

    if len(approved) != 1:
        print("Expected exactly one approved article; found:", len(approved))
        return

    comparison = pd.concat(
        [final.head(9), approved.head(1)],
        ignore_index=True,
    )

    for name, slate in [
        ("Existing hybrid", final),
        ("Exploration hybrid", comparison),
    ]:
        print(f"\n{name}")
        print("Countries:", slate["country"].nunique())
        print("Continents:", slate["continent"].nunique())
        print("Publishers:", slate["publisher"].nunique())
        print(
            "Cross-topic:",
            int((slate["hybrid_topic_score"] == 0).sum()),
        )
        
    print("\n=== SEMANTIC FILTER ABLATION ===")

    matched_pool = exploration_pool[
        exploration_pool["hybrid_topic_score"] > 0
    ].copy()

    for name, pool in [
        ("No filter", matched_pool),
        (
            "Semantic >= 0.60",
            matched_pool[
                matched_pool["description_semantic_score"] >= 0.60
            ],
        ),
    ]:
        slate = recommend_unified_mmr(
            pool,
            embeddings,
            top_k=10,
            lambda_param=0.5,
            candidate_pool_size=300,
        )

        print(f"\n{name}")
        print("Articles:", len(slate))
        print("Countries:", slate["country"].nunique())
        print("Continents:", slate["continent"].nunique())
        print("Publishers:", slate["publisher"].nunique())
        print(
            "Mean semantic:",
            round(slate["description_semantic_score"].mean(), 3),
        )

        for _, article in slate.iterrows():
            print(
                f"{article['description_semantic_score']:.3f}"
                f" | {article['country']}"
                f" | {article['title']}"
            )
            
    print("\n=== TOPIC LABEL DIAGNOSTIC ===")

    for _, article in final.iterrows():
        print(
            f"{len(article['topic_list'])} topics | "
            f"{article['description_semantic_score']:.3f} | "
            f"{article['title']}"
        )
        
    print("\n=== TOPIC LABEL PENALTY TEST ===")

    matched = candidates[
        candidates["hybrid_topic_score"] > 0
    ].copy()

    matched["topic_count"] = matched["topic_list"].apply(len)

    matched["adjusted_relevance"] = (
        matched["hybrid_relevance"]
        / np.sqrt(matched["topic_count"])
    )

    print(
        matched.sort_values(
            "adjusted_relevance", ascending=False
        )[
            ["title", "topic_count",
            "hybrid_relevance", "adjusted_relevance"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
    