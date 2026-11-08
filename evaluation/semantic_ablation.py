
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import recommender.unified_reranker as reranker
from recommender.hybrid_pipeline import recommend_hybrid_feed
from recommender.topic_descriptions import TOPIC_DESCRIPTIONS
from utils.data_loader import load_articles


def load_inputs():
    df = load_articles().reset_index(drop=True)
    embeddings = np.load(ROOT / "data/article_embeddings.npy")

    if len(df) != len(embeddings):
        raise ValueError("Article and embedding counts differ.")

    model = SentenceTransformer(
        "BAAI/bge-base-en-v1.5",
        device="cpu",
    )

    descriptions = [
        description
        for group in TOPIC_DESCRIPTIONS.values()
        for description in group
    ]

    encoded = model.encode(
        descriptions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    topic_embeddings = {}
    start = 0

    for topic, group in TOPIC_DESCRIPTIONS.items():
        topic_embeddings[topic] = encoded[
            start:start + len(group)
        ]
        start += len(group)

    return df, embeddings, topic_embeddings


def evaluate_slate(slate, embeddings, df):
    if slate.empty:
        return {
            "semantic_diversity": np.nan,
            "countries": 0,
            "publishers": 0,
            "mean_hybrid_relevance": np.nan,
        }

    id_to_position = {
        str(article_id): i
        for i, article_id in enumerate(df["article_id"])
    }

    positions = [
        id_to_position[str(article_id)]
        for article_id in slate["article_id"]
    ]

    vectors = embeddings[positions]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)

    similarity = vectors @ vectors.T
    upper = similarity[np.triu_indices(len(vectors), k=1)]

    return {
        "semantic_diversity": (
            1 - float(upper.mean())
            if len(upper)
            else np.nan
        ),
        "countries": slate["country"].nunique(),
        "publishers": slate["publisher"].nunique(),
        "mean_hybrid_relevance": float(
            slate["hybrid_relevance"].mean()
        ),
    }


def main():
    df, embeddings, topic_embeddings = load_inputs()

    profiles = [
        {"Technology": 1.0},
        {"Health": 1.0},
        {"Politics and Government": 1.0},
        {"Environment and Climate": 1.0},
        {"Technology": 1.0, "Science": 1.0},
    ]

    original_calculator = reranker.calculate_combined_redundancy

    def without_semantic(
        candidate,
        selected_items,
        candidate_embedding,
        selected_embeddings,
    ):
        return original_calculator(
            candidate,
            selected_items,
            candidate_embedding,
            selected_embeddings,
            semantic_weight=0.0,
            geographic_weight=0.6,
            publisher_weight=0.4,
        )

    results = []

    for profile in profiles:
        label = " + ".join(profile)

        common_args = {
            "df": df,
            "embeddings": embeddings,
            "user_profile": profile,
            "topic_description_embeddings": topic_embeddings,
            "top_k": 10,
            "topic_weight": 0.5,
            "lambda_param": 0.7,
            "seen_ids": None,
            "user_vector": None,
        }

        # Current production behavior
        full = recommend_hybrid_feed(**common_args)

        # Evaluation-only semantic ablation
        with patch.object(
            reranker,
            "calculate_combined_redundancy",
            side_effect=without_semantic,
        ):
            no_semantic = recommend_hybrid_feed(**common_args)

        for name, slate in [
            ("Full MMR", full),
            ("No Semantic", no_semantic),
        ]:
            results.append({
                "profile": label,
                "system": name,
                **evaluate_slate(slate, embeddings, df),
            })

    results_df = pd.DataFrame(results)

    output = ROOT / "evaluation/semantic_ablation_results.csv"
    results_df.to_csv(output, index=False)

    print("\n=== SEMANTIC ABLATION RESULTS ===")
    print(results_df.round(3).to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
