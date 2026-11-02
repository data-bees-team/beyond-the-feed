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
from recommender.baseline import recommend_baseline, score_all_candidates
from recommender.geographic_diversity import (
    recommend_geographic_mmr,
    generate_natural_explanation,
    evaluate_slate,
)


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
# SIDEBAR CONFIGURATION (OPTIONAL CONTROLS)
# =====================================

with st.sidebar.expander("⚙️ Recommendation Preferences", expanded=False):
    st.caption("Configurable parameters for the multi-stage recommendation pipeline.")
    top_k_param = st.slider("Recommendations (K)", min_value=5, max_value=20, value=10, step=1)
    lambda_param = st.slider(
        "Geographic Diversity (λ)",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Controls the balance between topic relevance (1.0) and geographic diversity (0.0). Default 0.6."
    )
    top_m_candidates = st.slider(
        "Candidate Pool (M)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Number of top relevance candidates evaluated during re-ranking."
    )
    show_insights = st.checkbox("Show Slate Diversity Insights", value=False)


# =====================================
# DISPLAY PERSONALIZED FEED
# =====================================

if st.session_state["feed_generated"]:

    # Retrieve saved user interests
    user_profile = st.session_state["user_profile"]

    saved_topics = list(user_profile.keys())

    st.markdown("### 📰 Your Daily Discoveries")

    st.caption(
        "🌍 Powered by Geographic Diversity Re-ranking (MMR). "
        "Balancing your topical interests with broad global perspectives across countries and continents."
    )

    # =====================================
    # MULTI-STAGE RECOMMENDATION PIPELINE
    # =====================================

    # Stage 1: Candidate Generation & Baseline Relevance Scoring
    eligible_candidates = score_all_candidates(df, user_profile)

    if eligible_candidates.empty:
        st.info("No matching articles found for your selected topics. Please try selecting different topics.")
    else:
        # Stage 2: Geographic Diversity MMR Re-ranking
        # Enforces candidate pool >= requested top_k
        actual_m = max(top_m_candidates, top_k_param)

        recommended_articles = recommend_geographic_mmr(
            candidates_df=eligible_candidates,
            top_k=top_k_param,
            lambda_param=lambda_param,
            top_m_candidates=actual_m,
            weight_country=1.0,
            weight_continent=0.4,
            weight_secondary=0.1,
            weight_missing_geo=0.5
        )

        # Optional Slate Diversity Insights
        if show_insights and not recommended_articles.empty:
            metrics = evaluate_slate(recommended_articles)
            st.markdown(
                f"""
                <div style="background-color:#F1F5F9; border-radius:8px; padding:12px 18px; margin-bottom:20px; font-size:14px; color:#334155;">
                    <strong>📊 Slate Diversity Insights:</strong> &nbsp;
                    🌍 <strong>{metrics['unique_countries']}</strong> countries &nbsp;·&nbsp;
                    🌐 <strong>{metrics['unique_continents']}</strong> continents &nbsp;·&nbsp;
                    📰 <strong>{metrics['unique_publishers']}</strong> publishers &nbsp;·&nbsp;
                    🎯 <strong>{metrics['avg_relevance']:.2f}</strong> avg relevance
                </div>
                """,
                unsafe_allow_html=True
            )

        # =====================================
        # DISPLAY NEWS CARDS
        # =====================================

        for _, row in recommended_articles.iterrows():

            # Identify which interests matched this article
            matched_topics = [
                topic
                for topic in row["topic_list"]
                if topic in saved_topics
            ]

            # Generate natural geographic and topic explanation
            reason = generate_natural_explanation(
                rank=int(row.get("selection_rank", 1)),
                country=row.get("country"),
                continent=row.get("continent"),
                matched_topics=matched_topics,
                sim_geo=float(row.get("geo_redundancy", 0.0))
            )

            # Prepare article information
            article = {
                "title": row.get("title", "Untitled Article"),
                "publisher": row.get("publisher", "Unknown Publisher"),
                "country": row.get("country"),
                "continent": row.get("continent"),
                "topics": row.get("topic_list", []),
                "url": row.get("article_url"),
                "reason": reason
            }

            # Display article card
            display_news_card(article)



    