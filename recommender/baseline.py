def calculate_relevance(topic_list, user_profile):
    """
    Calculate relevance based on selected user interests.
    """

    valid_topics = [
        topic for topic in topic_list
        if topic not in {"Unknown", "Other"}
    ]

    if not valid_topics:
        return 0.0

    return sum(
        user_profile.get(topic, 0.0)
        for topic in valid_topics
    ) / len(valid_topics)


def score_all_candidates(df, user_profile):
    """
    Score all articles against user topic preferences and return all eligible
    candidates (relevance_score > 0) deterministically sorted by relevance descending,
    article_id ascending.

    This preserves the full candidate pool for multi-stage re-ranking components
    (such as Geographic Diversity MMR) rather than truncating prematurely.
    """
    candidates = df.copy()

    candidates["relevance_score"] = (
        candidates["topic_list"].apply(
            lambda topics: calculate_relevance(
                topics, user_profile
            )
        )
    )

    candidates = candidates[
        candidates["relevance_score"] > 0
    ]

    return candidates.sort_values(
        by=["relevance_score", "article_id"],
        ascending=[False, True]
    ).copy()


def recommend_baseline(df, user_profile, top_k=10):
    """
    Return the top-k most relevant articles using pure relevance ranking
    and deterministic tie-breaking.
    """
    candidates = score_all_candidates(df, user_profile)
    return candidates.head(top_k).copy()