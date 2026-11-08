import streamlit as st
from pathlib import Path

import html

from utils.data_loader import load_articles
from utils.homepage_feed import get_homepage_articles

st.set_page_config(
    page_title="Beyond the Feed",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Load CSS
css_path = Path(__file__).parent / "styles" / "style.css"

with open(css_path, encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =====================================
# HERO SECTION
# =====================================

st.markdown("""
<div class="home-hero">
<div class="hero-eyebrow">✦ A SMARTER WAY TO DISCOVER NEWS</div>
<h1 class="home-title">Your world is bigger<br>than <span>your feed.</span></h1>
<p class="home-description">
Discover stories that match your interests while exploring
new topics, regions, and perspectives. Because the best
discoveries happen beyond what you already know.
</p>
</div>
""", unsafe_allow_html=True)

if st.button("Explore My Feed →", type="primary"):
    st.switch_page("pages/1_My_Feed.py")

st.write("")
st.write("")
st.write("")


# =====================================
# HOMEPAGE ARTICLES
# =====================================

st.markdown("## Explore the stories")
st.caption(
    "A broader mix of topics, countries, and publishers."
)

if "homepage_seed" not in st.session_state:
    st.session_state.homepage_seed = 42

if st.button("↻ Refresh Articles", key="refresh_homepage"):
    st.session_state.homepage_seed += 1

df = load_articles()

homepage_articles = get_homepage_articles(
    df,
    n_articles=12,
    seed=st.session_state.homepage_seed,
)


def display_homepage_article(article, featured=False):
    title = html.escape(str(article["title"]))
    publisher = html.escape(str(article["publisher"]))
    country = html.escape(str(article["country"]))

    topics = article["topic_list"]

    if not isinstance(topics, list):
        topics = []

    visible_topics = topics[:3]
    remaining = max(0, len(topics) - 3)

    topic_html = "".join(
        f'<span class="topic-tag">{html.escape(str(topic))}</span>'
        for topic in visible_topics
    )

    if remaining:
        topic_html += (
            f'<span class="topic-tag">+{remaining} more</span>'
        )

    title_class = "homepage-featured-title" if featured else "news-title"

    st.markdown(
        f"""
        <div class="homepage-article-card">
            <div class="news-topics">{topic_html}</div>
            <div class="{title_class}">{title}</div>
            <div class="news-meta">{publisher} · {country}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    url = str(article.get("article_url", "")).strip()

    if url.startswith(("https://", "http://")):
        st.link_button(
            "Read Article ↗",
            url,
            use_container_width=False,
        )


# Featured story
display_homepage_article(
    homepage_articles.iloc[0],
    featured=True,
)

# Remaining articles
for start in range(1, len(homepage_articles), 2):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        display_homepage_article(homepage_articles.iloc[start])

    if start + 1 < len(homepage_articles):
        with col2:
            display_homepage_article(homepage_articles.iloc[start + 1])

st.write("")
st.divider()

# =====================================
# FEATURES
# =====================================

# st.markdown("## A better way to stay informed")
# st.caption("Personalized discovery without limiting your perspective.")

# col1, col2, col3 = st.columns(3, gap="large")

# with col1:
#     st.markdown("""
#     <div class="feature-card">
#         <div class="feature-icon">🎯</div>
#         <h3>Made for You</h3>
#         <p>
#             Discover relevant news based on the topics
#             that matter most to you.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="feature-card">
#         <div class="feature-icon">🌍</div>
#         <h3>Beyond Your Circle</h3>
#         <p>
#             Explore interesting stories outside your
#             usual interests and discover new connections.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="feature-card">
#         <div class="feature-icon">⚖️</div>
#         <h3>Broader Perspectives</h3>
#         <p>
#             Explore news from different regions,
#             publishers, and topics.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# st.write("")
# st.write("")

# =====================================
# HOW IT WORKS
# =====================================

st.markdown("## How it works")
st.caption("Three simple steps to broaden your news experience.")

step1, step2, step3 = st.columns(3, gap="large")

with step1:
    st.markdown("### 01 — Choose")
    st.write("Select the topics you're interested in.")

with step2:
    st.markdown("### 02 — Discover")
    st.write("Explore personalized recommendations across different topics, regions, and publishers.")

with step3:
    st.markdown("### 03 — Shape Your Feed")
    st.write("Like or dislike articles to help your recommendations evolve with your interests.")

st.divider()

st.caption("🐝 Built by Data Bees · Behind the Feed Data Challenge 2026")