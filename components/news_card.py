import streamlit as st
from html import escape
from textwrap import dedent


def display_news_card(article):
    """
    Display a reusable news article card.

    article: dictionary containing:
        title, publisher, country, topics, url, reason
    """

    title = escape(str(article.get("title", "Untitled Article")))
    publisher = escape(str(article.get("publisher", "Unknown Publisher")))
    country = escape(str(article.get("country", "Unknown")))
    reason = escape(str(article.get("reason", "Recommended for you.")))

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
📰 {publisher} &nbsp; · &nbsp; 🌍 {country}
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