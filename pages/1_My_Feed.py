import streamlit as st
from pathlib import Path
from datetime import datetime
import sys


# =====================================
# PROJECT PATH
# =====================================

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from components.news_card import display_news_card
from utils.data_loader import load_articles
from recommender.baseline import recommend_baseline


# =====================================
# LOAD DATA
# =====================================

df = load_articles()

st.caption(f"📰 {len(df):,} news articles available")


# =====================================
# LOAD CUSTOM CSS
# =====================================

css_path = project_root / "styles" / "style.css"

with open(css_path, encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# =====================================
# PAGE HEADER
# =====================================

hour = datetime.now().hour

if hour < 12:
    greeting = "Good Morning"
elif hour < 17:
    greeting = "Good Afternoon"
else:
    greeting = "Good Evening"

st.markdown(
    f"""
<div style="margin-bottom:35px;">
    <p style="color:#0D9488; font-size:13px; font-weight:700; letter-spacing:2px;">
        YOUR PERSONALIZED NEWS EXPERIENCE
    </p>
    <h1 style="font-family:'Playfair Display', Georgia, serif; font-size:44px; line-height:1.25; letter-spacing:-1px;">
        {greeting}! 👋<br>
        Discover <span style="color:#0D9488;">beyond your feed.</span>
    </h1>
    <p style="color:#64748B; font-size:17px; max-width:650px; line-height:1.7;">
        Explore stories that match your interests while discovering
        new topics, countries, and perspectives.
    </p>
</div>
    """,
    unsafe_allow_html=True
)


# =====================================
# TOPIC SELECTION
# =====================================

st.subheader("✨ Personalize Your Experience")

st.caption(
    "Choose your favorite topics. "
    "We'll help you discover stories beyond them."
)

topics = [
    "Technology",
    "Business and Economy",
    "Health",
    "Science",
    "Education",
    "Environment and Climate",
    "Politics and Government",
    "Society and Culture",
    "Conflict and Security"
]


# =====================================
# SESSION STATE
# =====================================

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None

if "feed_generated" not in st.session_state:
    st.session_state["feed_generated"] = False


# =====================================
# USER INTEREST SELECTION
# =====================================

selected_topics = st.multiselect(
    "What topics are you interested in?",
    options=topics,
    placeholder="Choose your interests..."
)


# =====================================
# GENERATE FEED BUTTON
# =====================================

if st.button(
    "✨ Generate My Feed",
    type="primary"
):

    if not selected_topics:
        st.warning("Please select at least one topic.")

    else:

        # Create user interest profile
        user_profile = {
            topic: 1.0
            for topic in selected_topics
        }

        # Save user profile
        st.session_state["user_profile"] = user_profile

        # Remember that a feed has been generated
        st.session_state["feed_generated"] = True

        st.success("✨ Your personalized feed is ready!")


# =====================================
# DISPLAY PERSONALIZED FEED
# =====================================

if st.session_state["feed_generated"]:

    # Retrieve saved user interests
    user_profile = st.session_state["user_profile"]

    saved_topics = list(user_profile.keys())

    st.markdown("### 📰 Your Daily Discoveries")

    st.caption(
        "🧪 Baseline preview — recommendations based on your selected topics. "
        "Interest Bridging and diversity improvements will be integrated later."
    )

    # =====================================
    # BASELINE RECOMMENDATIONS
    # =====================================

    preview_articles = recommend_baseline(
        df,
        user_profile,
        top_k=10
    )

    # =====================================
    # DISPLAY NEWS CARDS
    # =====================================

    for _, row in preview_articles.iterrows():

        # Identify which interests matched this article
        matched_topics = [
            topic
            for topic in row["topic_list"]
            if topic in saved_topics
        ]

        # Generate recommendation explanation
        if matched_topics:

            reason = (
                "Matches your interest in "
                + ", ".join(matched_topics)
                + "."
            )

        else:
            reason = "Recommended based on your interests."

        # Prepare article information
        article = {
            "title": row["title"],
            "publisher": row["publisher"],
            "country": row["country"],
            "topics": row["topic_list"],
            "url": row["article_url"],
            "reason": reason
        }

        # Display article card
        display_news_card(article)



    