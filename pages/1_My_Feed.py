from pathlib import Path
from datetime import datetime
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from recommender.topic_descriptions import TOPIC_DESCRIPTIONS
from recommender.hybrid_pipeline import recommend_hybrid_feed
import streamlit.components.v1 as components
import time


from utils.semantic_loader import (
    load_article_embeddings,
    get_topic_description_embeddings,
)

page_start = time.perf_counter()


# =====================================
# PROJECT PATH & MODULE IMPORTS
# =====================================
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from components.auth import render_sidebar_auth
from components.news_card import display_news_card
from utils.data_loader import load_articles
from utils.feedback_manager import get_feedback_manager
from utils.user_manager import get_user_manager, DEMO_PERSONAS
from utils.db_manager import (
    init_db,
    save_user_topics,
    get_connection,
    log_recommended_articles,
    get_seen_article_ids
)
from recommender.nlp_profile import (
    get_or_create_user_vector,
    compute_nlp_content_scores
)
from recommender.user_preferences import build_user_preference_profile
from recommender.baseline import recommend_baseline, score_all_candidates
from recommender.geographic_diversity import (
    recommend_geographic_mmr,
    evaluate_slate,
)
from recommender.publisher_diversity import (
    recommend_publisher_mmr,
    evaluate_publisher_slate,
)
from recommender.interest_bridging import recommend_interest_bridging
from recommender.feed_composer import compose_balanced_feed

# Initialize Database
init_db()

# =====================================
# LOAD DATA & EMBEDDINGS
# =====================================
start = time.perf_counter()

df = load_articles()
print(f"DATA LOADING: {time.perf_counter() - start:.2f}s")

feedback_mgr = get_feedback_manager()
user_mgr = get_user_manager()


start = time.perf_counter()

embeddings, embeddings_dict = load_article_embeddings(df)

print(f"EMBEDDINGS LOADING: {time.perf_counter() - start:.2f}s")


    

# =====================================
# LOAD CUSTOM CSS
# =====================================
css_path = project_root / "styles" / "style.css"
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
# SIDEBAR: AUTHENTICATION & PERSONAS
# =====================================
render_sidebar_auth()

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None

if "feed_generated" not in st.session_state:
    st.session_state["feed_generated"] = False

if "current_feed_slate" not in st.session_state:
    st.session_state["current_feed_slate"] = None

active_user_id = st.session_state.get("user_id", "demo_user")

with st.sidebar:
    st.markdown("---")
    # Fetch latest user feedback dynamically
    user_feedback_records = feedback_mgr.get_feedback_records(user_id=active_user_id)
    user_pref_profile = build_user_preference_profile(user_feedback_records, user_id=active_user_id)
    user_fb_map = feedback_mgr.get_user_feedback(active_user_id)

    # User Feedback Insights
    st.markdown("### Adaptive Learning Profile")
    st.caption("Shared preferences learned from your interactions.")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Interactions", user_pref_profile.total_feedback_count)
    col_m2.metric("👍 Likes", user_pref_profile.interested_count)
    col_m3.metric("👎 Dislikes", user_pref_profile.not_interested_count)

    top_pos_topics = user_pref_profile.get_top_preferred_topics(3)
    top_neg_topics = user_pref_profile.get_dispreferred_topics(3)

    if top_pos_topics:
        st.markdown("**Top Liked Topics:**")
        st.markdown(" ".join(f"`{t[0]}`" for t in top_pos_topics))

    if top_neg_topics:
        st.markdown("**Downvoted Topics:**")
        st.markdown(" ".join(f"`{t[0]}`" for t in top_neg_topics))

    if user_pref_profile.total_feedback_count == 0:
        st.caption("No feedback recorded yet. Rate articles below to teach the system your preferences.")

    st.markdown("---")

algo_choice = "Hybrid"
top_k_param = st.session_state.get("top_k_param", 10)
lambda_param = st.session_state.get("lambda_param", 0.60)
alpha_param = st.session_state.get("alpha_param", 0.35)

# =====================================
# USER INTEREST SELECTION
# =====================================
st.subheader("✨ Personalize Your Experience")
st.caption("Choose your favorite topics. We'll help you discover stories beyond them.")

topics_list = [
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

selected_topics = st.multiselect(
    "What topics are you interested in?",
    options=topics_list,
    default=[
        topic
        for topic in st.session_state.get("user_profile") or {}
        if topic in topics_list
    ],
    placeholder="Choose your interests..."
)

def generate_new_slate():
    """Generates a fresh 10-article slate using deduplication and stores it in session_state."""
    user_profile = {topic: 1.0 for topic in selected_topics}
    st.session_state["user_profile"] = user_profile
    st.session_state["feed_generated"] = True
    save_user_topics(active_user_id, st.session_state.get("username", "Guest"), selected_topics)
    
    # Dynamic NLP scoring
    user_vec = get_or_create_user_vector(active_user_id, embeddings_dict)
    
    from utils.db_manager import get_user_feedback

    debug_feedback = feedback_mgr.get_user_feedback(active_user_id)

    print("DEBUG SAVED FEEDBACK:", list(debug_feedback.items())[:5])
    print("DEBUG FEEDBACK COUNT:", len(debug_feedback))
    
    df['nlp_score'] = compute_nlp_content_scores(user_vec, embeddings)

    # Relevance scoring
    scored_df = score_all_candidates(
        df=df,
        user_profile=user_profile,
        preference_profile=user_pref_profile,
        alpha=alpha_param
    )

    # Fetch seen IDs
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT article_id FROM recommendation_history WHERE user_id = ?", (active_user_id,))
        seen_ids = [str(r["article_id"]) for r in cursor.fetchall()]

    if seen_ids:
        available_articles = scored_df[~scored_df['article_id'].astype(str).isin(seen_ids)]
    else:
        available_articles = scored_df.copy()

    if selected_topics:
        eligible_candidates = available_articles[available_articles['topic'].isin(selected_topics)]
        if eligible_candidates.empty:
            eligible_candidates = available_articles.copy()
    else:
        eligible_candidates = available_articles.copy()

    if eligible_candidates.empty:
        st.session_state["current_feed_slate"] = pd.DataFrame()
        return

    
    # Route algorithm
    if "Hybrid" in algo_choice:
        description_embeddings = get_topic_description_embeddings()

        print("DEBUG USER ID:", active_user_id)
        print("DEBUG USER VECTOR NORM:", np.linalg.norm(user_vec))
        
        hybrid_candidates = recommend_hybrid_feed(
            seen_ids= seen_ids,
            df=df,
            embeddings=embeddings,
            user_profile=user_profile,
            topic_description_embeddings=description_embeddings,
            top_k=top_k_param,
            lambda_param=lambda_param,
            user_vector=user_vec,
        )

        
        
        recommended = hybrid_candidates
        print(
            "DEBUG RECOMMENDED TOPIC COUNTS:",
            recommended["topic_list"].apply(len).tolist()
        )

        print(
            "DEBUG AVG TOPICS PER ARTICLE:",
            round(recommended["topic_list"].apply(len).mean(), 2)
        )
            
        
        
        print("\n===== HYBRID RECOMMENDATION DEBUG =====")
        print("Selected topics:", selected_topics)

        debug_cols = [
            "title",
            "topic",
            "hybrid_topic_score",
            "description_semantic_score",
            "feedback_score",
            "hybrid_relevance",
        ]

        print(
            recommended[
                [col for col in debug_cols if col in recommended.columns]
            ].to_string(index=False)
        )

        print("=======================================\n")

    elif "Publisher" in algo_choice:
        recommended = recommend_publisher_mmr(
            candidates_df=eligible_candidates, top_k=top_k_param, lambda_param=lambda_param, preference_profile=user_pref_profile, alpha=alpha_param
        )
    elif "Geographic" in algo_choice:
        recommended = recommend_geographic_mmr(
            candidates_df=eligible_candidates, top_k=top_k_param, lambda_param=lambda_param, preference_profile=user_pref_profile, alpha=alpha_param
        )
    elif "Balanced" in algo_choice:
        recommended = compose_balanced_feed(
            df=available_articles, user_profile=user_profile, top_k=top_k_param, personalized_ratio=0.70, diversity_method="publisher", preference_profile=user_pref_profile, alpha=alpha_param, lambda_param=lambda_param
        )
    elif "Bridging" in algo_choice:
        recommended = recommend_interest_bridging(
            df=available_articles, user_profile=user_profile, top_k=top_k_param, preference_profile=user_pref_profile, alpha=alpha_param
        )
    else:
        recommended = recommend_baseline(
            df=available_articles, user_profile=user_profile, top_k=top_k_param, preference_profile=user_pref_profile, alpha=alpha_param
        )

    if not recommended.empty:
        log_recommended_articles(active_user_id, recommended['article_id'].tolist())

    st.session_state["current_feed_slate"] = recommended

# =====================================
# GENERATE FEED BUTTON
# =====================================
if st.button("✨ Generate My Feed", type="primary"):
    if not selected_topics:
        st.warning("Please select at least one topic.")
    else:
        generate_new_slate()
        st.success("Your personalized feed is ready!")
        st.rerun()

# =====================================
# DISPLAY PERSONALIZED FEED SLATE
# =====================================
if st.session_state.get("feed_generated"):

    st.markdown("### 📰 Your Daily Discoveries")

    if st.session_state.pop("scroll_to_feed", False):
        components.html(
            """
            <script>
                const doc = window.parent.document;
                const headings = [...doc.querySelectorAll("h3")];

                const target = headings.find(
                    h => h.textContent.includes("Your Daily Discoveries")
                );

                if (target) {
                    target.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });
                }
            </script>
            """,
            height=0,
        )

    recommended_articles = st.session_state.get("current_feed_slate")

    if recommended_articles is None or recommended_articles.empty:
        st.info("No matching unseen articles found. Click 'Reset Feed History' to explore again.")
    else:
        # Slate Diversity Insights
        metrics = evaluate_slate(recommended_articles)
        st.markdown(
            f"""
            <div style="background-color:#F1F5F9; border-radius:8px; padding:12px 18px; margin-bottom:20px; font-size:14px; color:#334155;">
                <strong> Slate Diversity Insights:</strong> &nbsp;
                 <strong>{metrics.get('unique_publishers', 0)}</strong> publishers &nbsp;·&nbsp;
                 <strong>{metrics.get('unique_countries', 0)}</strong> countries &nbsp;·&nbsp;
                 <strong>{metrics.get('unique_continents', 0)}</strong> continents
            </div>
            """,
            unsafe_allow_html=True
        )

        saved_topics = list(st.session_state.get("user_profile", {}).keys())

        # Render News Cards with Feedback
        for _, row in recommended_articles.iterrows():
            art_id = str(row.get("article_id"))
            reason = row.get("reason")
            if not reason:
                reason = f"Matches interest in {row.get('topic', 'General')} from {row.get('publisher', 'Publisher')}"

            article = {
                "article_id": art_id,
                "title": row.get("title", "Untitled Article"),
                "publisher": row.get("publisher", "Unknown Publisher"),
                "country": row.get("country"),
                "continent": row.get("continent"),
                "topics": row.get("topic_list", [row.get("topic")]),
                "url": row.get("article_url", "#"),
                "reason": reason,
                "feed_badge": row.get("feed_badge", "✦ CURATED FOR YOU")
            }

            display_news_card(
                article=article,
                user_id=active_user_id,
                feedback_manager=feedback_mgr,
                current_feedback=user_fb_map.get(art_id)
            )

        # =====================================
        # LOAD MORE DISCOVERIES & RESET
        # =====================================
        st.write("")
        st.markdown("---")
        col_load, col_reset = st.columns([2, 1])
        with col_load:
            if st.button("Load More Discoveries", type="primary", use_container_width=True):
                generate_new_slate()
                st.session_state["scroll_to_feed"] = True
                st.rerun()
        with col_reset:
            if st.button("Reset Feed History", type="secondary", use_container_width=True):
                feedback_mgr.clear_user_feedback(active_user_id)
                with get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM recommendation_history WHERE user_id = ?", (active_user_id,))
                    conn.commit()
                st.session_state["feed_generated"] = False
                st.session_state["current_feed_slate"] = None
                st.success("History reset! Starting fresh.")
                st.rerun()

    # =====================================
    # INTERACTION DRAWER: RATED ARTICLES
    # =====================================
    st.write("")
    with st.expander("Your Session Activity", expanded=False):
        all_fb = feedback_mgr.get_user_feedback(active_user_id)
        if not all_fb:
            st.caption("You haven't liked or disliked any articles in this session yet.")
        else:
            fb_ids = list(all_fb.keys())
            rated_df = df[df['article_id'].astype(str).isin(fb_ids)]
            
            for _, r_row in rated_df.iterrows():
                r_id = str(r_row['article_id'])
                fb_type = all_fb.get(r_id)
                badge_color = "#DCFCE7" if fb_type == "interested" else "#FEE2E2"
                text_color = "#166534" if fb_type == "interested" else "#991B1B"
                icon = "👍 Liked" if fb_type == "interested" else "👎 Disliked"
                
                st.markdown(
                    f"""
                    <div style="padding:10px 14px; border-radius:6px; background-color:#F8FAFC; margin-bottom:8px; border:1px solid #E2E8F0; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong>{r_row.get('title', 'Untitled')}</strong><br>
                            <span style="font-size:12px; color:#64748B;">📰 {r_row.get('publisher', 'Unknown')} · {r_row.get('topic', 'General')}</span>
                        </div>
                        <span style="background-color:{badge_color}; color:{text_color}; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:700;">
                            {icon}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
print(f"TOTAL MY FEED SCRIPT TIME: {time.perf_counter() - page_start:.2f}s")