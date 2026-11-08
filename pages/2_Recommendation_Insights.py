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
st.caption("A transparent view of your interests, feedback and news exposure — based on your own activity.")
if not selected:
    st.info("Select your interests in My Feed to start building your profile.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected interests", len(selected))
c2.metric("Articles rated", len(likes) + len(dislikes))
c3.metric("Articles recommended", len(history_ids), help="Distinct articles in your saved recommendation history")
c4.metric("Current feed", len(current_df), help="Articles currently displayed in My Feed during this session")

st.divider()
st.subheader("1. Your interests and feedback")
st.write("**Selected interests:** " + (", ".join(selected) if selected else "None yet"))

rated_df = df[df["article_id"].isin(likes | dislikes)].copy()
if rated_df.empty:
    st.info("No rated articles found in the dataset yet. Like or dislike stories in My Feed to see your feedback profile.")
else:
    rows = []
    for _, article in rated_df.iterrows():
        art_id = article["article_id"]
        for topic in set(parse_topics(article.get("topic_list", []))):
            if topic.lower() not in {"unknown", "other"}:
                rows.append({"Topic": topic, "Feedback": "Interested" if art_id in likes else "Not interested"})
    if rows:
        grouped = pd.DataFrame(rows).groupby(["Topic", "Feedback"]).size().reset_index(name="Articles")
        totals = grouped.groupby("Topic")["Articles"].sum().nlargest(12).index
        fig = px.bar(grouped[grouped["Topic"].isin(totals)], x="Topic", y="Articles", color="Feedback",
                     barmode="group", color_discrete_map={"Interested": "#0D9488", "Not interested": "#E58A70"})
        fig.update_layout(height=370, margin=dict(l=10, r=10, t=15, b=10), xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Counts reflect topic labels on rated articles. One article can contribute to multiple topics; these are not learned model weights.")
    else:
        st.info("Your rated articles do not have usable topic labels.")

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("2. Your feedback activity")
    if likes or dislikes:
        fig = px.pie(pd.DataFrame({"Feedback": ["Interested", "Not interested"], "Articles": [len(likes), len(dislikes)]}),
                     names="Feedback", values="Articles", hole=0.62,
                     color="Feedback", color_discrete_map={"Interested": "#0D9488", "Not interested": "#E58A70"})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Your likes and dislikes will appear here after you rate articles.")

with right:
    st.subheader("3. Your discovery mix")
    if current_df.empty:
        st.info("Generate a feed in My Feed to see how its articles connect to your selected interests.")
    elif not selected:
        st.info("Choose interests in My Feed to classify your discovery mix.")
    else:
        chosen = set(selected)
        counts = {"Your interests only": 0, "Your interests + other topics": 0, "No exact topic match": 0}
        for _, article in current_df.iterrows():
            tags = set(parse_topics(article.get("topic_list", [])))
            matched = tags & chosen
            if matched and tags - chosen:
                counts["Your interests + other topics"] += 1
            elif matched:
                counts["Your interests only"] += 1
            else:
                counts["No exact topic match"] += 1
        mix = pd.DataFrame({"Type": list(counts), "Articles": list(counts.values())})
        fig = px.pie(mix[mix["Articles"] > 0], names="Type", values="Articles", hole=0.62,
                     color="Type", color_discrete_map={
                         "Your interests only": "#0D9488", "Your interests + other topics": "#3B82F6",
                         "No exact topic match": "#F59E0B"})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Based on topic labels in your current feed, not a direct measurement of semantic similarity.")

st.divider()
st.subheader("4. Where your news comes from")
source = st.radio("Show exposure for", ["Current feed", "Recommendation history"], horizontal=True)
exposure = current_df if source == "Current feed" else history_df
if exposure.empty:
    st.info("No articles available for this view. Generate a feed in My Feed first.")
else:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Articles by continent**")
        if "continent" in exposure.columns:
            cont = exposure["continent"].dropna().astype(str)
            cont = cont[cont.map(meaningful)]
            if not cont.empty:
                counts = cont.value_counts().reset_index()
                counts.columns = ["Continent", "Articles"]
                fig = px.bar(counts, x="Continent", y="Articles", color_discrete_sequence=["#0D9488"])
                fig.update_layout(height=310, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No continent metadata available.")
        else:
            st.info("No continent metadata available.")
    with b:
        st.markdown("**Most frequent topics**")
        counts = topic_counts(exposure)
        if counts:
            plot = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10], columns=["Topic", "Articles"])
            fig = px.bar(plot, x="Articles", y="Topic", orientation="h", color_discrete_sequence=["#3B82F6"])
            fig.update_layout(height=310, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic metadata available.")
    publishers = exposure["publisher"].dropna().astype(str) if "publisher" in exposure else pd.Series(dtype=str)
    publishers = publishers[publishers.map(meaningful)]
    st.caption(f"{publishers.nunique()} distinct publishers in this view. Geographic and publisher counts describe metadata coverage, not viewpoint diversity or fairness.")

st.divider()
st.caption("Insights are descriptive. Feedback counts do not establish that an individual recommendation was caused by feedback. Historical exposure counts include distinct articles and may differ from the number of feed-generation events.")


st.divider()
st.subheader("5. How your recommendations adapt")
if not likes and not dislikes:
    st.info("No feedback profile yet. Like or dislike articles in My Feed to personalize future recommendations.")
else:
    st.write(
        f"Your feedback profile currently uses **{len(likes)} interested** and "
        f"**{len(dislikes)} not interested** article ratings."
    )
    st.write(
        "When you generate or load more recommendations, the system rebuilds a "
        "content-preference vector from your rated article embeddings. "
        "Similarity to that vector adjusts article relevance before diversity-aware reranking."
    )
    st.caption(
        "This confirms that feedback is available for personalization; it does not "
        "measure whether a particular recommendation improved or changed because of a click. "
        "Controlled before-and-after results belong in the evaluation dashboard."
    )
