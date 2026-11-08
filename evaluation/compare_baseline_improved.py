
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_articles
from recommender.baseline import recommend_baseline
from recommender.hybrid_pipeline import recommend_hybrid_feed
from recommender.topic_descriptions import TOPIC_DESCRIPTIONS


def load_inputs():
    df = load_articles().reset_index(drop=True)
    embeddings = np.load(ROOT / "data/article_embeddings.npy")

    assert len(df) == len(embeddings), "Embedding count mismatch"
    assert np.isfinite(embeddings).all(), "Invalid embeddings"

    model = SentenceTransformer(
        "BAAI/bge-base-en-v1.5", device="cpu"
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
        topic_embeddings[topic] = encoded[start:start + len(group)]
        start += len(group)

    return df, embeddings, topic_embeddings


def evaluate_slate(slate, embeddings, topic_embeddings, profile, df):
    if slate.empty:
        return {}

    # Independent topic relevance: fraction of selected topics
    # matched by each article (binary match for one-topic profiles).
    selected_topics = {t for t, w in profile.items() if w > 0}
    topic_matches = [
        len(selected_topics.intersection(topics)) / len(selected_topics)
        for topics in slate["topic_list"]
    ]

    # Independent semantic relevance: cosine similarity with
    # selected topic descriptions, using the same BGE model.
    selected_vectors = np.vstack([
        topic_embeddings[t]
        for t in selected_topics
        if t in topic_embeddings
    ])
    article_positions = df.index.get_indexer(slate.index)
    # Use article IDs rather than slate indices for reliable alignment.
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
    semantic_scores = vectors @ selected_vectors.T

    # Semantic diversity: 1 - mean pairwise cosine similarity.
    similarity_matrix = vectors @ vectors.T
    upper = similarity_matrix[np.triu_indices(len(vectors), k=1)]
    semantic_diversity = (
        1 - float(upper.mean()) if len(upper) else 0.0
    )

    publisher_counts = slate["publisher"].value_counts(normalize=True)
    publisher_hhi = float((publisher_counts ** 2).sum())

    return {
        "topic_match": float(np.mean(topic_matches)),
        "bge_relevance": float(np.mean(semantic_scores.max(axis=1))),
        "countries": slate["country"].nunique(),
        "continents": slate["continent"].nunique(),
        "publishers": slate["publisher"].nunique(),
        "publisher_hhi": publisher_hhi,
        "semantic_diversity": semantic_diversity,
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

    results = []

    for profile in profiles:
        label = " + ".join(profile)

        baseline = recommend_baseline(
            df, profile, top_k=10,
            preference_profile=None,
        )

        improved = recommend_hybrid_feed(
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
        
        
        if label == "Technology":
            for name, slate in [
                ("BASELINE", baseline),
                ("IMPROVED", improved),
            ]:
                print(f"\n=== {name}: TECHNOLOGY ARTICLES ===")
                print(
                    slate[
                        ["title", "country", "publisher", "topic"]
                    ].to_string(index=False, max_colwidth=75)
                )

        for system, slate in [
            ("Baseline", baseline),
            ("Improved", improved),
        ]:
            metrics = evaluate_slate(
                slate, embeddings, topic_embeddings, profile, df
            )
            results.append({
                "profile": label,
                "system": system,
                **metrics,
            })

    results_df = pd.DataFrame(results)
    output = ROOT / "evaluation" / "comparison_results.csv"
    results_df.to_csv(output, index=False)

    print("\n=== BASELINE VS IMPROVED ===")
    print(results_df.round(3).to_string(index=False))
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
