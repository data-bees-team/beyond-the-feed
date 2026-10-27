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


def recommend_baseline(df, user_profile, top_k=10):
    """
    Return the top-k most relevant articles.
    """

    recommendations = df.copy()

    recommendations["relevance_score"] = (
        recommendations["topic_list"].apply(
            lambda topics: calculate_relevance(
                topics, user_profile
            )
        )
    )

    recommendations = recommendations[
        recommendations["relevance_score"] > 0
    ]

    recommendations = recommendations.sort_values(
        by=["relevance_score", "article_id"],
        ascending=[False, True]
    )

    return recommendations.head(top_k).copy()