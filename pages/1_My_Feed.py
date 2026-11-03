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
from utils.feedback_manager import get_feedback_manager
from recommender.user_preferences import build_user_preference_profile
from recommender.baseline import recommend_baseline, score_all_candidates
from recommender.geographic_diversity import (
    recommend_geographic_mmr,
    generate_natural_explanation,
    evaluate_slate,
)
from recommender.publisher_diversity import (
    recommend_publisher_mmr,
    generate_natural_publisher_explanation,
    evaluate_publisher_slate,
)
from recommender.interest_bridging import recommend_interest_bridging
from recommender.feed_composer import compose_balanced_feed


# =====================================
# LOAD DATA & FEEDBACK MANAGER
# =====================================

df = load_articles()
feedback_mgr = get_feedback_manager()

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
        new topics, countries, and perspectives with adaptive feedback learning.
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
# SESSION STATE INITIALIZATION
# =====================================

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None

if "feed_generated" not in st.session_state:
    st.session_state["feed_generated"] = False

if "active_user_id" not in st.session_state:
    st.session_state["active_user_id"] = "demo_user"


# =====================================
# SIDEBAR: USER ACCOUNT & PREFERENCES DASHBOARD
# =====================================

with st.sidebar:
    st.markdown("### 👤 User Account")
    user_choice = st.selectbox(
        "Active Profile",
        options=["demo_user", "analyst_user", "global_reader", "custom_user"],
        index=0,
        help="Select a profile to experience independent user preferences and feedback history."
    )
    st.session_state["active_user_id"] = user_choice
    active_user_id = st.session_state["active_user_id"]

    # Retrieve feedback and build preference profile
    user_feedback_records = feedback_mgr.get_feedback_records(user_id=active_user_id)
    user_pref_profile = build_user_preference_profile(user_feedback_records, user_id=active_user_id)
    user_fb_map = feedback_mgr.get_user_feedback(active_user_id)

    # User Feedback Insights
    st.markdown("### 📊 Adaptive Learning Profile")
    st.caption("Shared preferences learned from your 👍 / 👎 interactions across all recommendation methods.")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Interactions", user_pref_profile.total_feedback_count)
    col_m2.metric("👍 Likes", user_pref_profile.interested_count)
    col_m3.metric("👎 Dislikes", user_pref_profile.not_interested_count)

    # Show top preferred and dispreferred signals
    top_pos_topics = user_pref_profile.get_top_preferred_topics(3)
    top_neg_topics = user_pref_profile.get_dispreferred_topics(3)
    top_pos_pubs = user_pref_profile.get_preferred_publishers(2)
    top_pos_countries = user_pref_profile.get_preferred_countries(2)

    if top_pos_topics:
        st.markdown("**Top Liked Topics:**")
        st.markdown(" ".join(f"`{t[0]}`" for t in top_pos_topics))

    if top_neg_topics:
        st.markdown("**Downvoted Topics:**")
        st.markdown(" ".join(f"`{t[0]}`" for t in top_neg_topics))

    if top_pos_pubs:
        st.markdown("**Preferred Publishers:**")
        st.markdown(", ".join(f"*{p[0]}*" for p in top_pos_pubs))

    if top_pos_countries:
        st.markdown("**Preferred Regions:**")
        st.markdown(", ".join(f"🌍 {c[0]}" for c in top_pos_countries))

    if user_pref_profile.total_feedback_count == 0:
        st.caption("No feedback recorded yet. Rate articles below to teach the system your preferences.")

    # Reset feedback option with confirmation
    with st.expander("🗑️ Reset User Feedback Profile", expanded=False):
        confirm_reset = st.checkbox("I understand this will erase my learned preferences", key="chk_confirm_reset")
        if st.button("Reset Profile", type="secondary", disabled=not confirm_reset):
            deleted = feedback_mgr.clear_user_feedback(active_user_id)
            st.success(f"Cleared {deleted} feedback interactions.")
            st.rerun()

    st.markdown("---")


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
# SIDEBAR CONFIGURATION (PIPELINE CONTROLS)
# =====================================

with st.sidebar.expander("⚙️ Recommendation Engine Settings", expanded=False):
    st.caption("Configurable parameters for the multi-stage recommendation pipeline.")

    algo_choice = st.selectbox(
        "Recommendation Method",
        options=[
            "🌍 Geographic Diversity (MMR)",
            "📰 Publisher Diversity (MMR)",
            "🌐 Balanced Feed (Personalized + Discovery)",
            "🌉 Interest Bridging (Topic Discovery)",
            "📊 Baseline (Standard Relevance)"
        ],
        index=0,
        help="Select which recommendation engine generates your feed."
    )

    top_k_param = st.slider("Recommendations (K)", min_value=5, max_value=20, value=10, step=1)

    lambda_param = st.slider(
        "Diversity Weight (λ)",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Controls the balance between relevance (1.0) and diversity (0.0). Default 0.6."
    )

    alpha_param = st.slider(
        "Feedback Personalization (α)",
        min_value=0.0,
        max_value=0.8,
        value=0.35,
        step=0.05,
        help="How strongly user feedback guides candidate relevance. 0.0 uses raw topic intent only; 0.35 is recommended."
    )

    top_m_candidates = st.slider(
        "Candidate Pool (M)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Number of top relevance candidates evaluated during re-ranking."
    )

    show_insights = st.checkbox("Show Slate Diversity Insights", value=True)


# =====================================
# DISPLAY PERSONALIZED FEED
# =====================================

if st.session_state["feed_generated"]:

    # Retrieve saved user interests
    user_profile = st.session_state["user_profile"]
    saved_topics = list(user_profile.keys())

    st.markdown("### 📰 Your Daily Discoveries")

    # Algorithm description banner
    if "Geographic" in algo_choice:
        st.caption(
            "🌍 **Geographic Diversity Re-ranking (MMR):** Balancing your topical interests with "
            "broad global perspectives across countries and continents, adapted to your feedback."
        )
    elif "Publisher" in algo_choice:
        st.caption(
            "📰 **Publisher Diversity Re-ranking (MMR):** Mitigating media concentration and "
            "maximizing editorial pluralism across distinct news organizations, adapted to your feedback."
        )
    elif "Balanced" in algo_choice:
        st.caption(
            "🌐 **Balanced Multi-Objective Feed:** 70% personalized core stories + 30% discovery and "
            "interest-bridging perspectives to prevent echo chambers."
        )
    elif "Bridging" in algo_choice:
        st.caption(
            "🌉 **Interest Bridging:** Discovering adjacent, serendipitous connections that link your "
            "core topics with new perspectives."
        )
    else:
        st.caption(
            "📊 **Baseline Relevance:** Standard topic-matching ranking, boosted by your learned preferences."
        )

    # =====================================
    # MULTI-STAGE RECOMMENDATION PIPELINE
    # =====================================

    actual_m = max(top_m_candidates, top_k_param)

    # Candidate Retrieval & Relevance Scoring
    eligible_candidates = score_all_candidates(
        df=df,
        user_profile=user_profile,
        preference_profile=user_pref_profile,
        alpha=alpha_param
    )

    if eligible_candidates.empty:
        st.info("No matching articles found for your selected topics. Please try selecting different topics.")
    else:
        # Route to selected recommendation algorithm
        if "Geographic" in algo_choice:
            recommended_articles = recommend_geographic_mmr(
                candidates_df=eligible_candidates,
                top_k=top_k_param,
                lambda_param=lambda_param,
                top_m_candidates=actual_m,
                weight_country=1.0,
                weight_continent=0.4,
                weight_secondary=0.1,
                weight_missing_geo=0.5,
                preference_profile=user_pref_profile,
                alpha=alpha_param
            )
            algo_key = "geographic_mmr"

        elif "Publisher" in algo_choice:
            recommended_articles = recommend_publisher_mmr(
                candidates_df=eligible_candidates,
                top_k=top_k_param,
                lambda_param=lambda_param,
                top_m_candidates=actual_m,
                weight_repeat_publisher=1.0,
                weight_missing_publisher=0.5,
                preference_profile=user_pref_profile,
                alpha=alpha_param
            )
            algo_key = "publisher_mmr"

        elif "Balanced" in algo_choice:
            recommended_articles = compose_balanced_feed(
                df=df,
                user_profile=user_profile,
                top_k=top_k_param,
                personalized_ratio=0.70,
                diversity_method="geographic",
                preference_profile=user_pref_profile,
                alpha=alpha_param,
                lambda_param=lambda_param
            )
            algo_key = "balanced_feed"

        elif "Bridging" in algo_choice:
            recommended_articles = recommend_interest_bridging(
                df=df,
                user_profile=user_profile,
                top_k=top_k_param,
                top_m_candidates=actual_m,
                preference_profile=user_pref_profile,
                alpha=alpha_param
            )
            algo_key = "interest_bridging"

        else:  # Baseline
            recommended_articles = recommend_baseline(
                df=df,
                user_profile=user_profile,
                top_k=top_k_param,
                preference_profile=user_pref_profile,
                alpha=alpha_param
            )
            algo_key = "baseline"

        # Log impressions for future contextual bandit analysis (Phase 8)
        if not recommended_articles.empty:
            feedback_mgr.log_impressions(
                user_id=active_user_id,
                algorithm=algo_key,
                articles=recommended_articles.to_dict("records")
            )

        # Slate Diversity Insights
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

            matched_topics = [
                topic
                for topic in row["topic_list"]
                if topic in saved_topics
            ]

            # Generate natural explanation
            sim_geo_val = float(row.get("geo_redundancy", 0.0))
            reason = generate_natural_explanation(
                rank=int(row.get("selection_rank", 1)),
                country=row.get("country"),
                continent=row.get("continent"),
                matched_topics=matched_topics,
                sim_geo=sim_geo_val
            )

            # Check if feed lane or badge is assigned
            feed_badge = row.get("feed_badge", "✦ CURATED FOR YOU")

            article = {
                "article_id": row.get("article_id"),
                "title": row.get("title", "Untitled Article"),
                "publisher": row.get("publisher", "Unknown Publisher"),
                "country": row.get("country"),
                "continent": row.get("continent"),
                "topics": row.get("topic_list", []),
                "url": row.get("article_url"),
                "reason": reason,
                "feed_badge": feed_badge
            }

            # Display card with interactive feedback
            display_news_card(
                article=article,
                user_id=active_user_id,
                feedback_manager=feedback_mgr,
                current_feedback=user_fb_map.get(str(article["article_id"]))
            )
