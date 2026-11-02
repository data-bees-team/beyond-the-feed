"""
Reusable News Article Card Component for Beyond the Feed.

Renders responsive news cards with rich metadata badges (topics, country,
continent, publisher), natural explanations, direct article links, and
interactive 👍 Interested / 👎 Not Interested feedback controls.
"""

from typing import Dict, Any, Optional
import streamlit as st
from html import escape
from textwrap import dedent

from utils.feedback_manager import (
    FeedbackManager,
    get_feedback_manager,
    FEEDBACK_INTERESTED,
    FEEDBACK_NOT_INTERESTED
)


def is_valid_geo_label(value: Any) -> bool:
    """Check if a geographic value is valid, non-empty, and recognized."""
    if value is None:
        return False
    s = str(value).strip().lower()
    return bool(s) and (s not in {"", "unknown", "none", "null", "undetermined", "n/a", "missing", "nan"})


def display_news_card(
    article: Dict[str, Any],
    user_id: str = "default_user",
    feedback_manager: Optional[FeedbackManager] = None,
    current_feedback: Optional[str] = None
) -> None:
    """
    Display a responsive news article card with interactive feedback controls.

    Args:
        article: dictionary containing:
            article_id, title, publisher, country, continent, topics, url, reason,
            feed_badge (optional).
        user_id: Active user ID for tracking feedback.
        feedback_manager: Optional FeedbackManager instance. Defaults to global singleton.
        current_feedback: Optional current feedback status ('interested', 'not_interested', or None).
    """
    if feedback_manager is None:
        feedback_manager = get_feedback_manager()

    article_id = str(article.get("article_id", "")).strip()
    if not article_id:
        # Fallback to hashed title if article_id is absent
        article_id = str(hash(article.get("title", "unknown_article")))

    # Retrieve current feedback if not explicitly passed
    if current_feedback is None:
        user_fb_map = feedback_manager.get_user_feedback(user_id)
        current_feedback = user_fb_map.get(article_id)

    title = escape(str(article.get("title", "Untitled Article")))
    publisher = escape(str(article.get("publisher", "Unknown Publisher")))
    reason = escape(str(article.get("reason", "Recommended for you.")))
    badge_label = escape(str(article.get("feed_badge", "✦ CURATED FOR YOU")))

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
    if isinstance(topics, str):
        topics = [t.strip() for t in topics.split(";") if t.strip()]

    url = article.get("url") or article.get("article_url")

    # Create topic badges
    topic_badges = "".join(
        f'<span class="topic-tag">{escape(str(topic))}</span>'
        for topic in topics
    )

    # Render card body
    st.markdown(
        dedent(f"""
        <div class="news-card">
        <div class="news-card-label">{badge_label}</div>

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

    # Action bar: Link button and Feedback buttons
    col_link, col_interested, col_not_interested, col_status = st.columns([1.8, 1.2, 1.4, 2.0])

    if url and str(url).startswith(("https://", "http://")):
        col_link.link_button("Read Article ↗", url, use_container_width=True)

    # Unique keys per article and user
    btn_key_pos = f"btn_pos_{user_id}_{article_id}"
    btn_key_neg = f"btn_neg_{user_id}_{article_id}"

    # Visual style distinction for active feedback
    pos_label = "👍 Interested" if current_feedback != FEEDBACK_INTERESTED else "👍 Saved"
    neg_label = "👎 Dislike" if current_feedback != FEEDBACK_NOT_INTERESTED else "👎 Marked"

    if col_interested.button(pos_label, key=btn_key_pos, use_container_width=True):
        if current_feedback == FEEDBACK_INTERESTED:
            # Toggle off
            feedback_manager.remove_feedback(user_id=user_id, article_id=article_id)
        else:
            # Record or update to interested
            feedback_manager.record_feedback(
                user_id=user_id,
                article_id=article_id,
                feedback_type=FEEDBACK_INTERESTED,
                article_metadata=article
            )
        st.rerun()

    if col_not_interested.button(neg_label, key=btn_key_neg, use_container_width=True):
        if current_feedback == FEEDBACK_NOT_INTERESTED:
            # Toggle off
            feedback_manager.remove_feedback(user_id=user_id, article_id=article_id)
        else:
            # Record or update to not_interested
            feedback_manager.record_feedback(
                user_id=user_id,
                article_id=article_id,
                feedback_type=FEEDBACK_NOT_INTERESTED,
                article_metadata=article
            )
        st.rerun()

    # Visual status badge
    if current_feedback == FEEDBACK_INTERESTED:
        col_status.markdown(
            '<div style="padding-top:6px;"><span style="background-color:#DCFCE7; color:#166534; padding:5px 12px; border-radius:12px; font-size:12px; font-weight:700;">✓ Interested</span></div>',
            unsafe_allow_html=True
        )
    elif current_feedback == FEEDBACK_NOT_INTERESTED:
        col_status.markdown(
            '<div style="padding-top:6px;"><span style="background-color:#FEE2E2; color:#991B1B; padding:5px 12px; border-radius:12px; font-size:12px; font-weight:700;">✗ Not Interested</span></div>',
            unsafe_allow_html=True
        )
    else:
        col_status.markdown(
            '<div style="padding-top:8px; font-size:11.5px; color:#94A3B8;">Rate to improve feed</div>',
            unsafe_allow_html=True
        )

    st.write("")
