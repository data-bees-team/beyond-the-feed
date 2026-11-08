import pandas as pd


def get_homepage_articles(
    df: pd.DataFrame,
    n_articles: int = 12,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Select a varied set of articles for the public homepage.

    This is not a personalized recommendation system.
    It favors different countries, publishers, and topics.
    """
    if df.empty or n_articles <= 0:
        return df.iloc[0:0].copy()

    articles = df.drop_duplicates(subset="article_id").copy()

    # Keep articles with at least one meaningful topic.
    excluded_topics = {"Unknown", "Other"}

    articles = articles[
        articles["topic_list"].apply(
            lambda topics: (
                isinstance(topics, list)
                and any(
                    topic not in excluded_topics
                    for topic in topics
                )
            )
        )
    ].copy()

    # Randomize candidates so refreshing can produce a new selection.
    articles = articles.sample(frac=1, random_state=seed)

    selected_indices = []
    used_countries = set()
    used_publishers = set()
    used_topics = set()

    while len(selected_indices) < min(n_articles, len(articles)):
        best_index = None
        best_score = -1

        for idx, article in articles.iterrows():
            if idx in selected_indices:
                continue

            country = str(article.get("country", "")).strip()
            publisher = str(article.get("publisher", "")).strip()

            topics = article.get("topic_list", [])
            if not isinstance(topics, list):
                topics = []

            topics = set(topics)

            score = 0

            if country and country not in used_countries:
                score += 3

            if publisher and publisher not in used_publishers:
                score += 3

            score += min(len(topics - used_topics), 3)

            if score > best_score:
                best_score = score
                best_index = idx

                if score == 9:
                    break

        if best_index is None:
            break

        article = articles.loc[best_index]

        selected_indices.append(best_index)

        country = str(article.get("country", "")).strip()
        publisher = str(article.get("publisher", "")).strip()

        if country:
            used_countries.add(country)

        if publisher:
            used_publishers.add(publisher)

        topics = article.get("topic_list", [])
        if isinstance(topics, list):
            used_topics.update(topics)

    return articles.loc[selected_indices].reset_index(drop=True)