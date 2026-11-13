
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.semantic_ablation import load_inputs
from recommender.hybrid_retrieval import (
    score_max_topic_match,
    score_description_similarity,
    combine_hybrid_scores,
    deduplicate_candidates,
)
from recommender.hybrid_pipeline import recommend_hybrid_feed


PROFILES = {
    "Technology": {"Technology": 1.0},
    "Health": {"Health": 1.0},
    "Politics": {"Politics and Government": 1.0},
    "Environment": {"Environment and Climate": 1.0},
    "Technology + Science": {
        "Technology": 1.0,
        "Science": 1.0,
    },
}

# THRESHOLD = 0.65
# POOL_SIZE = 300

THRESHOLDS = [0.65, 0.60, 0.55, 0.50, 0.45]
POOL_SIZE = 300



def evaluate_profile(name, profile, df, embeddings, topic_embeddings):
    candidates = df.copy()
    candidates["embedding_idx"] = np.arange(len(candidates))

    candidates = score_max_topic_match(candidates, profile)

    selected = [
        topic_embeddings[topic]
        for topic, weight in profile.items()
        if weight > 0 and topic in topic_embeddings
    ]

    description_vectors = np.vstack(selected)

    candidates = score_description_similarity(
        candidates, embeddings, description_vectors
    )

    candidates = combine_hybrid_scores(
        candidates, topic_weight=0.5
    )

    cross_topic = candidates["hybrid_topic_score"] == 0
    
    if name == "Technology":
        print("\n=== TOP 15 CROSS-TOPIC ARTICLES ===")

        top_cross = candidates[cross_topic].sort_values(
            "description_semantic_score",
            ascending=False
        ).head(15)

        for _, article in top_cross.iterrows():
            print(
                f"{article['description_semantic_score']:.3f} | "
                f"{article['title']}"
            )
    
    print(f"\n=== {name}: THRESHOLD COMPARISON ===")

    for threshold in THRESHOLDS:
        eligible_cross = candidates[
            cross_topic
            & (candidates["description_semantic_score"] >= threshold)
        ]

        print(
            f"Threshold {threshold:.2f}: "
            f"{len(eligible_cross)} cross-topic candidates"
        )

    eligible = candidates[
        (~cross_topic)
        | (candidates["description_semantic_score"] >= 0.65)
    ].copy()

    eligible = deduplicate_candidates(eligible)

    pool = (
        eligible.sort_values(
            ["hybrid_relevance", "article_id"],
            ascending=[False, True],
            kind="stable",
        )
        .head(POOL_SIZE)
    )

    final = recommend_hybrid_feed(
        df=df,
        embeddings=embeddings,
        user_profile=profile,
        topic_description_embeddings=topic_embeddings,
        top_k=10,
        topic_weight=0.5,
        lambda_param=0.7,
        seen_ids=None,
        user_vector=None,
    )

    final_cross = final[final["hybrid_topic_score"] == 0]
    
    # Measure additional topic exposure
    selected_topics = set(profile.keys())
    additional_topics = set()
    articles_with_extra_topics = 0

    for _, article in final.iterrows():
        extra = set(article["topic_list"]) - selected_topics
        extra -= {"Other", "Unknown"}

        if extra:
            articles_with_extra_topics += 1
            additional_topics.update(extra)

    print(f"\n=== {name}: MULTI-TOPIC EXPOSURE ===")
    print("Articles with additional topics:", articles_with_extra_topics)
    print("Distinct additional topics:", len(additional_topics))
    print("Additional topic names:", sorted(additional_topics))

    print(f"\n=== {name}: FINAL ARTICLE TOPICS ===")
    for _, article in final.iterrows():
        print(article["title"], "→", article["topic_list"])


    return {
        "profile": name,
        "eligible_cross_topic": int(
            (eligible["hybrid_topic_score"] == 0).sum()
        ),
        "cross_topic_in_top_300": int(
            (pool["hybrid_topic_score"] == 0).sum()
        ),
        "cross_topic_in_final_10": len(final_cross),
        "mean_cross_topic_semantic_score": (
            round(final_cross["description_semantic_score"].mean(), 3)
            if not final_cross.empty else None
        ),
    }


def main():
    df, embeddings, topic_embeddings = load_inputs()

    assert len(df) == len(embeddings)

    results = []

    for name, profile in PROFILES.items():
        print(f"\nEvaluating: {name}")

        result = evaluate_profile(
            name, profile, df, embeddings, topic_embeddings
        )
        results.append(result)

    output = pd.DataFrame(results)

    print("\n=== CROSS-TOPIC DISCOVERY EVALUATION ===")
    print(output.to_string(index=False))

    output.to_csv(
        ROOT / "evaluation/cross_topic_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
