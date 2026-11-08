"""Stable, current-user comparison of topic-only and personalized hybrid news feeds."""
import json
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from recommender.baseline import recommend_baseline
from recommender.hybrid_pipeline import recommend_hybrid_feed
from recommender.nlp_profile import get_or_create_user_vector
from utils.data_loader import load_articles
from utils.db_manager import get_connection, init_db
from utils.feedback_manager import get_feedback_manager
from utils.semantic_loader import load_article_embeddings, get_topic_description_embeddings
from utils.user_manager import get_user_manager

st.set_page_config(page_title="See the Difference | Beyond the Feed", layout="wide")
st.title("Same interests. A different view of the news.")
st.caption("Compare two sets of 10 recommendations for your own interests. This page does not add articles to My Feed or mark them as seen.")


def topics_of(value):
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        if value.lstrip().startswith("["):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return topics_of(parsed)
            except (ValueError, TypeError):
                pass
        return [x.strip() for x in value.split(";") if x.strip()]
    return []


def clean_label(value):
    if pd.isna(value) if not isinstance(value, (list, dict, set, tuple)) else False:
        return None
    label = str(value).strip()
    return label if label.lower() not in {"", "nan", "none", "unknown", "n/a"} else None


def describe(frame, selected):
    chosen = set(selected)
    counts = Counter({"only": 0, "connected": 0, "outside": 0})
    all_topics, countries, publishers, continents = set(), set(), set(), set()
    rows = []
    for _, item in frame.iterrows():
        labels = set(topics_of(item.get("topic_list", [])))
        if not labels:
            labels = set(topics_of(item.get("topic", "")))
        matched = labels & chosen
        extra = labels - chosen - {"Unknown", "Other"}
        group = "connected" if matched and extra else "only" if matched else "outside"
        counts[group] += 1
        all_topics.update(labels - {"Unknown", "Other"})
        for key, target in (("country", countries), ("publisher", publishers), ("continent", continents)):
            label = clean_label(item.get(key, ""))
            if label:
                target.add(label)
        rows.append({
            "group": group,
            "topics": sorted(labels),
            "extra": sorted(extra),
            "title": str(item.get("title", "Untitled")),
            "publisher": clean_label(item.get("publisher", "")) or "Unknown publisher",
            "country": clean_label(item.get("country", "")) or "Unknown country",
        })
    return {
        "counts": dict(counts), "total": len(rows), "rows": rows,
        "topics": len(all_topics), "countries": len(countries),
        "publishers": len(publishers), "continents": len(continents),
        "new_topics": sorted(all_topics - chosen),
    }


def render_mix(info):
    total = info["total"]
    for label, key, help_text in (
        ("Only your selected topics", "only", "Selected topic labels, without additional topics"),
        ("Your interests + other topics", "connected", "At least one selected topic and one additional topic"),
    ):
        n = info["counts"][key]
        st.markdown(f"**{label} — {n}/{total}**")
        st.progress(n / total if total else 0)
        st.caption(help_text)
    if info["counts"]["outside"]:
        n = info["counts"]["outside"]
        st.caption(f"{n}/{total} have no exact selected-topic label. These are counted separately, not as topic connections.")


def render_slate(info):
    """Ten compact rows, not ten full article cards."""
    names = {"only": "Interest only", "connected": "Connected", "outside": "Related discovery"}
    for idx, item in enumerate(info["rows"], start=1):
        tags = ", ".join(item["topics"][:3]) or "Unlabeled"
        if len(item["topics"]) > 3:
            tags += " …"
        st.markdown(f"**{idx:02d}. {names[item['group']]}** · {tags}")


def coverage_metric(label, base, hybrid, denominator=10):
    diff = hybrid - base
    note = "same" if diff == 0 else f"{abs(diff)} {'more' if diff > 0 else 'fewer'}"
    st.metric(label, f"{base} → {hybrid}", help=f"Baseline to Beyond the Feed; {note} distinct {label.lower()}.")
    if diff == 0 and base >= denominator:
        st.caption("Both reach the maximum possible in this 10-article sample.")
    elif diff == 0:
        st.caption("No difference in this sample.")
    else:
        st.caption(note.capitalize() + " in this sample.")


init_db()
account_type = st.session_state.get("account_type", "guest")
user_id = st.session_state.get("user_id") or "demo_user"
if account_type == "registered" and st.session_state.get("user_id"):
    saved = get_user_manager().get_user_profile(user_id) or {}
    selected = topics_of(saved.get("selected_topics") or [])
    # My Feed writes the active selection to session_state when generating a slate.
    session_profile = st.session_state.get("user_profile")
    if isinstance(session_profile, dict) and session_profile:
        selected = list(session_profile)
else:
    session_profile = st.session_state.get("user_profile") or {}
    selected = list(session_profile) if isinstance(session_profile, dict) else topics_of(session_profile)
selected = list(dict.fromkeys(selected))

st.markdown("**Your current interests:** " + (", ".join(selected) if selected else "None selected"))
if not selected:
    st.info("Choose your interests and generate My Feed first, then return here.")
    st.stop()

signature = (str(user_id), account_type, tuple(sorted(selected)), 10, "comparison-v5")
existing = st.session_state.get("comparison_snapshot")
if existing and existing.get("signature") != signature:
    st.info("Your interests or account changed. Generate a new comparison to use the latest selection.")

if st.button("Generate / refresh comparison", type="primary", use_container_width=True):
    with st.spinner("Preparing one comparison for your current profile..."):
        try:
            df = load_articles()
            profile = {topic: 1.0 for topic in selected}
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT article_id FROM recommendation_history WHERE user_id = ?",
                    (user_id,),
                )
                seen = [str(row["article_id"]) for row in cursor.fetchall()]
            unseen = df[~df["article_id"].astype(str).isin(seen)].copy() if seen else df
            traditional = recommend_baseline(unseen, profile, top_k=10)
            # The embedding matrix must remain aligned with the unfiltered dataset.
            embeddings, embeddings_dict = load_article_embeddings(df)
            user_vector = get_or_create_user_vector(user_id, embeddings_dict)
            hybrid = recommend_hybrid_feed(
                df=df,
                embeddings=embeddings,
                user_profile=profile,
                topic_description_embeddings=get_topic_description_embeddings(),
                top_k=10,
                lambda_param=st.session_state.get("lambda_param", 0.60),
                seen_ids=seen,
                user_vector=user_vector,
            )
            if traditional.empty or hybrid.empty:
                st.warning("Not enough unseen recommendations were returned for this comparison.")
            else:
                feedback = get_feedback_manager().get_user_feedback(user_id) or {}
                norm = float(np.linalg.norm(user_vector)) if user_vector is not None else 0.0
                st.session_state["comparison_snapshot"] = {
                    "signature": signature,
                    "baseline": describe(traditional, selected),
                    "hybrid": describe(hybrid, selected),
                    "seen_count": len(seen),
                    "feedback_count": len(feedback),
                    "learned_vector": norm > 1e-8,
                    "selected": list(selected),
                    "semantic_mean": (float(pd.to_numeric(hybrid["description_semantic_score"], errors="coerce").mean())
                                      if "description_semantic_score" in hybrid else None),
                    "semantic_min": (float(pd.to_numeric(hybrid["description_semantic_score"], errors="coerce").min())
                                     if "description_semantic_score" in hybrid else None),
                    "semantic_max": (float(pd.to_numeric(hybrid["description_semantic_score"], errors="coerce").max())
                                     if "description_semantic_score" in hybrid else None),
                }
        except Exception as exc:
            st.error(f"Unable to generate the comparison: {exc}")

snapshot = st.session_state.get("comparison_snapshot")
if not snapshot or snapshot["signature"] != signature:
    st.caption("Click Generate / refresh comparison to see a fixed snapshot.")
    st.stop()

base, hybrid = snapshot["baseline"], snapshot["hybrid"]
st.caption(
    f"Fixed snapshot · {base['total']} baseline recommendations and {hybrid['total']} hybrid recommendations "
    f"· {snapshot['seen_count']} previously recommended article IDs excluded where supported"
)

st.subheader("1. What changes in the recommendations?")
a, b = st.columns(2, gap="large")
with a:
    with st.container(border=True):
        st.markdown("### Topic-focused baseline")
        st.caption("Ranks by selected topic labels. Does not use your learned semantic feedback vector.")
        render_mix(base)
with b:
    with st.container(border=True):
        st.markdown("### Beyond the Feed")
        st.caption("Combines interest matching, semantic relevance, and diversity-aware selection; can use your learned feedback vector.")
        render_mix(hybrid)

st.caption("**Why multi-topic articles can rank differently:** the baseline averages your interest weights across an article's valid topic labels; the hybrid uses the strongest selected-topic match and combines it with semantic similarity. Both can recommend multi-topic articles.")

st.subheader("2. What additional ground is covered?")
a, b, c = st.columns(3)
with a:
    coverage_metric("Topics", base["topics"], hybrid["topics"])
with b:
    coverage_metric("Countries", base["countries"], hybrid["countries"])
with c:
    coverage_metric("Publishers", base["publishers"], hybrid["publishers"])

new_topics = hybrid["new_topics"]
if new_topics:
    st.markdown("**Additional topics represented in Beyond the Feed:** " + ", ".join(new_topics))
else:
    st.caption("No additional topic labels in this particular hybrid slate.")

coverage = pd.DataFrame([
    {"Measure": label, "Topic baseline": base[key], "Beyond the Feed": hybrid[key]}
    for label, key in (("Topics", "topics"), ("Countries", "countries"), ("Publishers", "publishers"))
])
fig = px.bar(
    coverage.melt(id_vars="Measure", var_name="Recommender", value_name="Distinct labels"),
    x="Measure", y="Distinct labels", color="Recommender", barmode="group",
    color_discrete_map={"Topic baseline": "#64748B", "Beyond the Feed": "#0D9488"},
)
fig.update_layout(height=290, margin=dict(l=5, r=5, t=10, b=5), legend=dict(orientation="h", y=-0.22))
st.plotly_chart(fig, use_container_width=True)
st.caption("Descriptive coverage in two 10-article slates. Publishers and countries cannot exceed 10 here; equal counts may reflect this ceiling, not identical selections or viewpoints.")

st.subheader("3. What makes Beyond the Feed different?")
a, b, c = st.columns(3)
with a:
    with st.container(border=True):
        st.markdown("**Semantic matching**")
        st.caption("Uses article and topic-description embeddings to recognize related content beyond exact labels. This can affect ranking even when every selected article has a matching topic label.")
with b:
    with st.container(border=True):
        st.markdown("**Diversity-aware selection**")
        st.caption("Balances relevance and redundancy when assembling the hybrid slate. Coverage numbers above show the observed result, not a guaranteed improvement.")
with c:
    with st.container(border=True):
        st.markdown("**Adaptive personalization**")
        st.caption(f"Saved feedback: {snapshot['feedback_count']} article ratings. " + (
            "A nonzero user vector was supplied to the hybrid recommender."
            if snapshot["learned_vector"] else
            "No nonzero learned vector was available for this snapshot."
        ))

semantic_mean = snapshot.get("semantic_mean")
if semantic_mean is not None and np.isfinite(semantic_mean):
    st.caption(
        f"**Observed semantic signal:** mean article-to-selected-topic-description cosine similarity "
        f"in this hybrid slate: {semantic_mean:.3f} "
        f"(range {snapshot['semantic_min']:.3f}–{snapshot['semantic_max']:.3f}). "
        "This is a similarity measure, not independently judged relevance or a causal contribution estimate."
    )

st.subheader("4. Does feedback change recommendations? (controlled simulation)")
st.caption("Three previously run fixed-candidate-pool tests: one simulated like on the initial #1 article and one dislike on the initial #2 article per topic. These are not the current user's results.")
feedback_results = pd.DataFrame([
    {"Topic": "Technology", "Replaced": 7, "Retained": 3, "Disliked left top 10": "Yes"},
    {"Topic": "Health", "Replaced": 4, "Retained": 6, "Disliked left top 10": "Yes"},
    {"Topic": "Politics and Government", "Replaced": 5, "Retained": 5, "Disliked left top 10": "Yes"},
])
feedback_long = feedback_results.melt(id_vars="Topic", value_vars=["Replaced", "Retained"], var_name="Outcome", value_name="Articles")
fig = px.bar(feedback_long, x="Topic", y="Articles", color="Outcome", barmode="stack",
             color_discrete_map={"Replaced": "#0D9488", "Retained": "#94A3B8"})
fig.update_layout(height=310, margin=dict(l=5, r=5, t=10, b=5), yaxis=dict(range=[0, 10]), legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, use_container_width=True)
st.caption("Replaced articles: Technology 7/10, Health 4/10, Politics 5/10. The disliked article left the top 10 in all three tests; unique publisher, country, and continent counts stayed unchanged. This demonstrates responsiveness in these scenarios, not improved user satisfaction or general performance.")

st.subheader("5. See all 10 selections without opening articles")
st.caption("Each line is one recommendation. The two columns are alternative feeds, not 20 articles added to your personal feed.")
with st.expander("Show the two 10-article lists", expanded=False):
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Topic-focused baseline · 10 slots**")
        render_slate(base)
    with b:
        st.markdown("**Beyond the Feed · 10 slots**")
        render_slate(hybrid)

connected = hybrid["counts"]["connected"]
matched = hybrid["counts"]["only"] + connected
st.info(
    f"In this comparison, {matched} of {hybrid['total']} Beyond the Feed recommendations "
    f"include at least one of your selected topic labels; {connected} also include additional topics. "
    "Semantic relevance can contribute even when there are no articles without an exact topic match."
)
st.caption(
    "Counts are based on dataset topic and publisher labels, which may be imperfect. "
    "The baseline uses topic ranking; the hybrid uses its own scoring and selection method. "
    "Both are compared using the same interests and previously recommended article history. "
    "This is a demonstration of the two strategies, not a controlled causal evaluation of each component. "
    "Refresh the comparison after new feedback or feed activity to capture those changes."
)
