
import pandas as pd
import numpy as np
from typing import Dict

from recommender.baseline import calculate_relevance
from recommender.nlp_profile import compute_nlp_content_scores


def score_topic_candidates(
    df: pd.DataFrame,
    user_profile: dict[str, float]
) -> pd.DataFrame:
    """
    Calculate topic relevance without removing zero-overlap articles.

    Preserve baseline scores for later evaluation.
    """
    candidates = df.copy()

    candidates["relevance_score"] = candidates["topic_list"].apply(
        lambda topics: calculate_relevance(topics, user_profile)
    )

    candidates["baseline_relevance"] = candidates["relevance_score"].copy()

    return candidates






def build_topic_semantic_profile(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    selected_topics: list[str],
    min_exclusive: int = 30
) -> np.ndarray:
    """
    Build a cold-start semantic profile from selected topics.

    Prefer exclusively labeled articles when enough unique titles exist.
    Otherwise use topic-weighted embeddings.

    Each normalized headline contributes at most once per topic.
    """
    if len(df) != len(embeddings):
        raise ValueError("Article and embedding counts do not match.")

    if not selected_topics:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    normalized_titles = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    topic_profiles = []

    for topic in dict.fromkeys(selected_topics):
        exclusive_mask = df["topic_list"].apply(
            lambda topics: topics == [topic]
        ).to_numpy()

        exclusive_indices = np.flatnonzero(exclusive_mask)

        exclusive_unique = (
            ~normalized_titles.iloc[exclusive_indices].duplicated()
        ).to_numpy()

        exclusive_indices = exclusive_indices[exclusive_unique]

        if len(exclusive_indices) >= min_exclusive:
            profile = embeddings[exclusive_indices].mean(axis=0)

        else:
            matching_mask = df["topic_list"].apply(
                lambda topics: topic in topics
            ).to_numpy()

            matching_indices = np.flatnonzero(matching_mask)

            if len(matching_indices) == 0:
                continue

            matching_unique = (
                ~normalized_titles.iloc[matching_indices].duplicated()
            ).to_numpy()

            matching_indices = matching_indices[matching_unique]

            matching_topics = df.iloc[matching_indices]["topic_list"]

            weights = matching_topics.apply(
                lambda topics: 1.0 / max(len(topics), 1)
            ).to_numpy(dtype=np.float32)

            profile = np.average(
                embeddings[matching_indices],
                axis=0,
                weights=weights
            )

        norm = np.linalg.norm(profile)

        if norm > 0:
            topic_profiles.append(profile / norm)

    if not topic_profiles:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    combined_profile = np.mean(topic_profiles, axis=0)
    combined_norm = np.linalg.norm(combined_profile)

    if combined_norm > 0:
        combined_profile /= combined_norm

    return combined_profile.astype(np.float32)



def score_semantic_candidates(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    user_vector: np.ndarray
) -> pd.DataFrame:
    """
    Calculate BGE semantic similarity for every article.

    Requires the full dataset in original CSV row order.
    """
    if len(df) != len(embeddings):
        raise ValueError("Article and embedding counts do not match.")

    if user_vector.shape != (embeddings.shape[1],):
        raise ValueError("User vector dimension does not match embeddings.")

    candidates = df.copy()

    candidates["semantic_score"] = compute_nlp_content_scores(
        user_vector,
        embeddings
    )

    return candidates



def compute_hybrid_relevance(
    candidates: pd.DataFrame,
    topic_weight: float = 0.7,
    semantic_threshold: float = 0.65
) -> pd.DataFrame:
    """
    Combine topic relevance and BGE semantic similarity.

    Preserve baseline relevance and allow high-similarity
    cross-topic articles into the candidate pool.
    """
    if not 0.0 <= topic_weight <= 1.0:
        raise ValueError("topic_weight must be between 0 and 1.")

    if not -1.0 <= semantic_threshold <= 1.0:
        raise ValueError("semantic_threshold must be between -1 and 1.")

    result = candidates.copy()

    result["semantic_relevance"] = result["semantic_score"].clip(
        lower=0.0,
        upper=1.0
    )

    result["hybrid_relevance"] = (
        topic_weight * result["relevance_score"]
        + (1.0 - topic_weight) * result["semantic_relevance"]
    )

    eligible = (
        (result["relevance_score"] > 0)
        | (result["semantic_score"] >= semantic_threshold)
    )

    result = result.loc[eligible].copy()

    return result.sort_values(
        by=["hybrid_relevance", "article_id"],
        ascending=[False, True]
    ).copy()



def deduplicate_candidates(
    candidates: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove duplicate article titles while preserving the
    highest-ranked candidate for each title.
    """
    result = candidates.copy()

    result["_normalized_title"] = (
        result["title"]
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    result = result.sort_values(
        by=["hybrid_relevance", "article_id"],
        ascending=[False, True]
    )

    result = result.drop_duplicates(
        subset=["_normalized_title"],
        keep="first"
    )

    return result.drop(
        columns=["_normalized_title"]
    ).copy()

def score_description_similarity(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    description_embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    Score articles against multiple topic descriptions.

    Uses the highest cosine similarity across descriptions.
    Does not modify the original DataFrame or baseline scores.
    """
    if len(df) != len(embeddings):
        raise ValueError(
            "DataFrame and article embeddings must have matching lengths."
        )

    descriptions = np.asarray(
        description_embeddings, dtype=np.float32
    )

    if descriptions.ndim != 2 or descriptions.shape[0] == 0:
        raise ValueError(
            "Description embeddings must be a non-empty 2D array."
        )

    if embeddings.shape[1] != descriptions.shape[1]:
        raise ValueError("Embedding dimensions do not match.")

    article_norms = np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    description_norms = np.linalg.norm(
        descriptions, axis=1, keepdims=True
    )

    normalized_articles = embeddings / np.maximum(
        article_norms, 1e-12
    )
    normalized_descriptions = descriptions / np.maximum(
        description_norms, 1e-12
    )

    similarities = (
        normalized_articles @ normalized_descriptions.T
    )

    result = df.copy()
    result["description_semantic_score"] = (
        similarities.max(axis=1).clip(-1, 1)
    )

    return result

def score_max_topic_match(
    df: pd.DataFrame,
    user_profile: Dict[str, float],
) -> pd.DataFrame:
    """
    Calculate the strongest matching topic preference.

    Preserves existing baseline scores and input DataFrame.
    """
    result = df.copy()

    def calculate_max_match(topics):
        if not isinstance(topics, (list, tuple, set)):
            return 0.0

        return max(
            (
                float(user_profile.get(topic, 0.0))
                for topic in topics
                if topic not in ("Unknown", "Other")
            ),
            default=0.0,
        )

    result["hybrid_topic_score"] = (
        result["topic_list"].apply(calculate_max_match)
    )

    return result


def combine_hybrid_scores(
    df: pd.DataFrame,
    topic_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Combine maximum topic matching and description-based
    semantic similarity without changing baseline scores.
    """
    if not 0.0 <= topic_weight <= 1.0:
        raise ValueError("topic_weight must be between 0 and 1.")

    required = {
        "hybrid_topic_score",
        "description_semantic_score",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()

    topic_scores = result["hybrid_topic_score"].clip(0, 1)
    semantic_scores = result["description_semantic_score"].clip(0, 1)

    result["hybrid_relevance"] = (
        topic_weight * topic_scores
        + (1.0 - topic_weight) * semantic_scores
    )

    return result