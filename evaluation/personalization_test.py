
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.semantic_ablation import load_inputs
from recommender.hybrid_pipeline import recommend_hybrid_feed


def main():
    df, embeddings, topic_embeddings = load_inputs()

    # Verify article-to-embedding alignment assumptions.
    assert len(df) == len(embeddings)
    assert df["article_id"].is_unique

    profile = {"Technology": 1.0}

    common_args = {
        "df": df,
        "embeddings": embeddings,
        "user_profile": profile,
        "topic_description_embeddings": topic_embeddings,
        "top_k": 10,
        "topic_weight": 0.5,
        "lambda_param": 0.7,
        "seen_ids": None,
    }

    # Initial recommendations without feedback.
    baseline = recommend_hybrid_feed(
        **common_args,
        user_vector=None,
    )

    # Simulate feedback on the first recommended article.
    feedback_article = baseline.iloc[0]
    feedback_id = str(feedback_article["article_id"])

    id_to_position = {
        str(article_id): i
        for i, article_id in enumerate(df["article_id"])
    }

    feedback_embedding = embeddings[id_to_position[feedback_id]]
    
    # Exclude the feedback article from all three scenarios
    common_args["seen_ids"] = {feedback_id}

    # Regenerate the no-feedback slate using the same exclusion
    baseline = recommend_hybrid_feed(
        **common_args,
        user_vector=None,
)

    # Match the existing preference-vector implementation.
    positive_vector = 0.3 * feedback_embedding
    negative_vector = -0.2 * feedback_embedding

    liked = recommend_hybrid_feed(
        **common_args,
        user_vector=positive_vector,
    )

    disliked = recommend_hybrid_feed(
        **common_args,
        user_vector=negative_vector,
    )

    feedback_norm = np.linalg.norm(feedback_embedding)

    results = []

    for label, slate in [
        ("No Feedback", baseline),
        ("Liked", liked),
        ("Disliked", disliked),
    ]:
        positions = [
            id_to_position[str(article_id)]
            for article_id in slate["article_id"]
        ]

        vectors = embeddings[positions]

        similarities = (
            vectors @ feedback_embedding
            / (
                np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)
                * max(feedback_norm, 1e-12)
            )
        )

        results.append({
            "scenario": label,
            "mean_similarity_to_feedback": float(similarities.mean()),
            "mean_hybrid_relevance": float(
                slate["hybrid_relevance"].mean()
            ),
            "unique_countries": int(slate["country"].nunique()),
            "unique_publishers": int(slate["publisher"].nunique()),
            "article_ids": set(slate["article_id"].astype(str)),
        })

    original_ids = results[0]["article_ids"]

    for result in results:
        result["overlap_with_original"] = len(
            original_ids & result["article_ids"]
        )
        del result["article_ids"]

    output = pd.DataFrame(results)

    print("\n=== PERSONALIZATION EVALUATION ===")
    print("Selected feedback article:", feedback_article["title"])
    print(output.round(3).to_string(index=False))

    output.to_csv(
        ROOT / "evaluation/personalization_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
