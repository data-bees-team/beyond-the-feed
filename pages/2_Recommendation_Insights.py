"""Personal, data-grounded Recommendation Insights for Beyond the Feed."""
from pathlib import Path
import sys
import json

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.auth import render_sidebar_auth
from utils.db_manager import init_db, get_connection
from utils.data_loader import load_articles
from utils.feedback_manager import get_feedback_manager
from utils.user_manager import get_user_manager

st.set_page_config(page_title="Your Recommendation Insights", layout="wide")
init_db()
render_sidebar_auth()

css_path = ROOT / "styles" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def parse_topics(value):
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        if value.strip().startswith("["):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parse_topics(parsed)
            except (ValueError, TypeError):
                pass
        return [v.strip() for v in value.split(";") if v.strip()]
    return []


def meaningful(value):
    if pd.isna(value) if not isinstance(value, (list, tuple, set, dict)) else False:
        return False
    return str(value).strip().lower() not in {"", "nan", "none", "unknown", "n/a"}


def topic_counts(frame):
    counts = {}
    for value in frame.get("topic_list", pd.Series(dtype=object)):
        for topic in set(parse_topics(value)):
            if topic.lower() not in {"unknown", "other"}:
                counts[topic] = counts.get(topic, 0) + 1
    return counts


def current_topics():
    session_profile = st.session_state.get("user_profile") or {}
    if isinstance(session_profile, dict) and session_profile:
        return list(session_profile.keys())
    if st.session_state.get("account_type") == "registered" and st.session_state.get("user_id"):
        saved = get_user_manager().get_user_profile(st.session_state["user_id"]) or {}
        return parse_topics(saved.get("selected_topics", []))
    return parse_topics(session_profile)


@st.cache_data(show_spinner=False)
def articles():
    return load_articles()


df = articles().copy()
if "article_id" not in df.columns:
    st.error("Article data does not include article_id.")
    st.stop()
df["article_id"] = df["article_id"].astype(str)
if "topic_list" not in df.columns:
    df["topic_list"] = df.get("topic", pd.Series([""] * len(df))).apply(parse_topics)

user_id = st.session_state.get("user_id") or "demo_user"
feedback_mgr = get_feedback_manager()
feedback = feedback_mgr.get_user_feedback(user_id) or {}
feedback = {str(k): v for k, v in feedback.items()}
likes = {k for k, v in feedback.items() if v == "interested"}
dislikes = {k for k, v in feedback.items() if v == "not_interested"}
selected = current_topics()

with get_connection() as conn:
    history_rows = conn.execute(
        "SELECT article_id FROM recommendation_history WHERE user_id = ?", (user_id,)
    ).fetchall()
history_ids = {str(row["article_id"]) for row in history_rows}
history_df = df[df["article_id"].isin(history_ids)].copy()

slate = st.session_state.get("current_feed_slate")
if (
    isinstance(slate, pd.DataFrame)
    and not slate.empty
    and st.session_state.get("feed_generated")
    and st.session_state.get("current_feed_owner_id") == user_id
):
    current_df = slate.copy()
else:
    current_df = pd.DataFrame(columns=df.columns)

st.title("Your Recommendation Insights")
st.caption("Your selected interests, article feedback, and the variety of news you receive.")

st.subheader("Your selected interests")
if selected:
    st.write(" · ".join(f"**{topic}**" for topic in selected))
else:
    st.info("Select interests in My Feed to personalize recommendations.")
st.caption("These are your chosen topics. Disliking an article does not remove a selected interest here.")

st.subheader("Your article feedback")
a, b, c = st.columns(3)
a.metric("Interested articles", len(likes))
b.metric("Not interested articles", len(dislikes))
c.metric("Total articles rated", len(likes) + len(dislikes))
st.caption("Likes and Dislikes refer to individual articles, not entire topics. Feedback can adjust future ranking; these counts are not topic preference scores.")

st.divider()
st.subheader("Your current feed")
if current_df.empty:
    st.info("Generate a feed in My Feed to see its topic coverage and source variety.")
else:
    current_topics_count = topic_counts(current_df)
    valid_countries = (current_df["country"].dropna().astype(str) if "country" in current_df else pd.Series(dtype=str))
    valid_countries = valid_countries[valid_countries.map(meaningful)]
    valid_publishers = (current_df["publisher"].dropna().astype(str) if "publisher" in current_df else pd.Series(dtype=str))
    valid_publishers = valid_publishers[valid_publishers.map(meaningful)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Articles", len(current_df))
    m2.metric("Topics covered", len(current_topics_count))
    m3.metric("Countries", valid_countries.nunique())
    m4.metric("Publishers", valid_publishers.nunique())

    if selected:
        chosen = set(selected)
        mix_counts = {"Selected topics only": 0, "Selected + other topics": 0, "Other topics only": 0}
        for _, article in current_df.iterrows():
            tags = set(parse_topics(article.get("topic_list", [])))
            matched = tags & chosen
            if matched and tags - chosen:
                mix_counts["Selected + other topics"] += 1
            elif matched:
                mix_counts["Selected topics only"] += 1
            else:
                mix_counts["Other topics only"] += 1
        st.markdown("**How your feed connects to your interests**")
        cols = st.columns(3)
        for col, (label, count) in zip(cols, mix_counts.items()):
            col.metric(label, count)
        st.caption("Based on article topic labels. 'Other topics only' means no exact topic-label match, not necessarily irrelevant content.")

    with st.expander("Explore topic, geographic and publisher details"):
        source = st.radio("View", ["Current feed", "Recommendation history"], horizontal=True)
        exposure = current_df if source == "Current feed" else history_df
        if exposure.empty:
            st.info("No articles available for this view.")
        else:
            left, right = st.columns(2)
            with left:
                counts = topic_counts(exposure)
                st.markdown("**Topics appearing in articles**")
                if counts:
                    topic_plot = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10], columns=["Topic", "Articles"])
                    fig = px.bar(topic_plot, x="Articles", y="Topic", orientation="h")
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No topic labels available.")
            with right:
                st.markdown("**Articles by continent**")
                continents = exposure["continent"].dropna().astype(str) if "continent" in exposure else pd.Series(dtype=str)
                continents = continents[continents.map(meaningful)]
                if not continents.empty:
                    cont_plot = continents.value_counts().rename_axis("Continent").reset_index(name="Articles")
                    fig = px.bar(cont_plot, x="Continent", y="Articles")
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No continent data available.")
            publishers = exposure["publisher"].dropna().astype(str) if "publisher" in exposure else pd.Series(dtype=str)
            publishers = publishers[publishers.map(meaningful)]
            st.markdown("**Publisher coverage**")
            if not publishers.empty:
                publisher_counts = publishers.value_counts()
                st.caption(f"{len(publisher_counts)} distinct publishers across {len(publishers)} articles with publisher data. Most frequent publisher: {publisher_counts.iloc[0]} article(s).")
                pub_plot = publisher_counts.head(10).rename_axis("Publisher").reset_index(name="Articles")
                fig = px.bar(pub_plot, x="Articles", y="Publisher", orientation="h")
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No publisher data available.")
            st.caption("Coverage counts describe article metadata, not viewpoint diversity or fairness.")

with st.expander("How feedback affects recommendations"):
    st.write("Your Likes and Dislikes are recorded for articles. When new recommendations are generated, the recommender can use those article embeddings to adjust rankings alongside selected interests and diversity-aware reranking.")
    st.caption("This page describes recorded feedback and current feed content. It does not prove that a specific click caused a particular recommendation.")
