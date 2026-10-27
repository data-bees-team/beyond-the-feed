import streamlit as st
from pathlib import Path

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
# FEATURES
# =====================================

st.markdown("## A better way to stay informed")
st.caption("Personalized discovery without limiting your perspective.")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <h3>Made for You</h3>
        <p>
            Discover relevant news based on the topics
            that matter most to you.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🌍</div>
        <h3>Beyond Your Circle</h3>
        <p>
            Explore interesting stories outside your
            usual interests and discover new connections.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚖️</div>
        <h3>Broader Perspectives</h3>
        <p>
            Explore news from different regions,
            publishers, and topics.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

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
    st.write("Explore personalized recommendations and new connections.")

with step3:
    st.markdown("### 03 — Explore More")
    st.write("Discover stories beyond your usual interests.")

st.divider()

st.caption("🐝 Built by Data Bees · Behind the Feed Data Challenge 2026")