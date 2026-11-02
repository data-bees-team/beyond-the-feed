import streamlit as st
from html import escape
from textwrap import dedent


def is_valid_geo_label(value):
    """Check if a geographic value is valid and populated."""
    if value is None:
        return False
    s = str(value).strip().lower()
    return bool(s) and (s not in {"", "unknown", "none", "null", "undetermined", "n/a", "missing", "nan"})


def display_news_card(article):
    """
    Display a reusable news article card.

    article: dictionary containing:
        title, publisher, country, continent, topics, url, reason
    """

    title = escape(str(article.get("title", "Untitled Article")))
    publisher = escape(str(article.get("publisher", "Unknown Publisher")))
    reason = escape(str(article.get("reason", "Recommended for you.")))

    raw_country = article.get("country")
    raw_continent = article.get("continent")

    # Format geographic badge safely
    if is_valid_geo_label(raw_country) and is_valid_geo_label(raw_continent):
        geo_text = f"🌍 {escape(str(raw_country))} &nbsp;·&nbsp; {escape(str(raw_continent))}"
    elif is_valid_geo_label(raw_country):
        geo_text = f"🌍 {escape(str(raw_country))}"
    else:
        geo_text = "🌍 Geography unavailable"

    topics = article.get("topics", [])
    url = article.get("url")

    # Create topic badges
    topic_badges = "".join(
        f'<span class="topic-tag">{escape(str(topic))}</span>'
        for topic in topics
    )

    # News card layout
    st.markdown(
    dedent(f"""
<div class="news-card">
<div class="news-card-label">✦ CURATED FOR YOU</div>

<h3 class="news-title">{title}</h3>

<div class="news-meta">
📰 {publisher} &nbsp; · &nbsp; {geo_text}
</div>

<div class="news-topics">{topic_badges}</div>

<div class="news-reason">
<strong>✦ Why this article?</strong>
<p>{reason}</p>
</div>
</div>
"""),
        unsafe_allow_html=True
    )

    # Actual clickable link
    if url and str(url).startswith(("https://", "http://")):
        st.link_button(
            "Read Article ↗",
            url,
            use_container_width=False
        )