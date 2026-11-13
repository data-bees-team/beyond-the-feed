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

try:
    st.set_page_config(page_title="See the Difference | Beyond the Feed", layout="wide")
except Exception:
    pass
st.title("Beyond the Feed | Evaluation")
st.caption("Your selected interests · Topic baseline vs. Beyond the Feed")


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
        "publisher_list": sorted(publishers),
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
        pub = item.get("publisher", "Unknown")
        country = item.get("country", "")
        geo = f" ({country})" if country and country != "Unknown country" else ""
        title = item.get("title", "Untitled")
        if len(title) > 60:
            title = title[:57] + "..."
        st.markdown(f"**{idx:02d}. {title}**\n📰 *{pub}*{geo} · `{names[item['group']]}` · {tags}")


def coverage_metric(label, base, hybrid, denominator=10, extra_note=None, novel_count=None):
    diff = hybrid - base
    note = "same" if diff == 0 else f"{abs(diff)} {'more' if diff > 0 else 'fewer'}"

    if novel_count is not None:
        shared_count = hybrid - novel_count
        st.metric(
            label,
            f"{base} → {hybrid}",
            delta=f"{novel_count} new / {shared_count} shared",
            delta_color="off",
            help=f"Baseline to Beyond the Feed; {novel_count} of {hybrid} {label.lower()} were not in the baseline.",
        )
    else:
        st.metric(label, f"{base} → {hybrid}", help=f"Baseline to Beyond the Feed; {note} distinct {label.lower()}.")

    if extra_note:
        st.caption(extra_note)
    elif novel_count is not None and diff == 0 and base >= denominator:
        if novel_count == 0:
            st.caption(f"Both reach {denominator}/{denominator} with the exact same {label.lower()}.")
        else:
            st.caption(f"Both hit {denominator}/{denominator}, but **{novel_count} of {hybrid}** are different {label.lower()}.")
    elif novel_count is not None and diff == 0:
        st.caption(f"Same count ({base}), with **{novel_count} new** {label.lower()} in Beyond the Feed.")
    elif diff == 0 and base >= denominator:
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

# Include saved article-level feedback in the snapshot key so stale comparisons
# are never presented as results for the user's latest ratings.
feedback_now = get_feedback_manager().get_user_feedback(user_id) or {}
feedback_signature = tuple(sorted((str(k), str(v)) for k, v in feedback_now.items()))
signature = (str(user_id), account_type, tuple(sorted(selected)), feedback_signature, 10, "comparison-v9-personal")
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
base_pubs = set(base.get("publisher_list", []))
hybrid_pubs = set(hybrid.get("publisher_list", []))
novel_pubs = hybrid_pubs - base_pubs

st.markdown("### Your feed at a glance")
st.caption("Current account · baseline → hybrid · two independently generated top-10 feeds")
a, b, c = st.columns(3)
a.metric("Topics covered", f"{base['topics']} → {hybrid['topics']}")
b.metric("Countries covered", f"{base['countries']} → {hybrid['countries']}")
with c:
    st.metric("Publishers covered", f"{base['publishers']} → {hybrid['publishers']}",
              help="Distinct publishers in each top-10 feed; this is not a measure of viewpoint diversity.")
    st.caption(f"**{len(novel_pubs)} new sources · {len(hybrid_pubs & base_pubs)} shared** with baseline")

live_tab, evidence_tab = st.tabs(["① Your comparison", "② Why your feeds differ"])

with live_tab:
    st.markdown("#### How do the two feeds differ?")
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown("**Topic-only baseline**")
            render_mix(base)
    with b:
        with st.container(border=True):
            st.markdown("**Beyond the Feed · hybrid**")
            render_mix(hybrid)
    st.caption("Baseline averages topic matches; hybrid combines topic and semantic relevance with feedback (when available) and diversity-aware reranking. This snapshot does not isolate individual component effects.")
    with st.expander("View the actual top-10 headlines for both systems"):
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**Topic-only baseline**")
            render_slate(base)
        with b:
            st.markdown("**Beyond the Feed**")
            render_slate(hybrid)

with evidence_tab:
    st.markdown("#### How your selected interests appear in both feeds")
    st.caption("Current-user results only · " + ", ".join(snapshot["selected"]))
    st.caption("Based on your two generated top-10 feeds; this is not an isolated test of individual algorithms.")

    st.markdown("**Selected-topic coverage in your current comparison**")
    comparison = pd.DataFrame([
        {"Feed": "Topic baseline", "Only selected": base["counts"]["only"], "Selected + other": base["counts"]["connected"], "No exact match": base["counts"]["outside"]},
        {"Feed": "Hybrid", "Only selected": hybrid["counts"]["only"], "Selected + other": hybrid["counts"]["connected"], "No exact match": hybrid["counts"]["outside"]},
    ])
    chart = px.bar(comparison.melt(id_vars="Feed", var_name="Topic match", value_name="Articles"),
                   x="Feed", y="Articles", color="Topic match", barmode="stack",
                   color_discrete_map={"Only selected": "#0D9488", "Selected + other": "#3B82F6", "No exact match": "#94A3B8"})
    chart.update_layout(height=300, margin=dict(l=5, r=5, t=10, b=5),
                        yaxis=dict(range=[0, 10]), legend=dict(orientation="h", y=-0.24))
    st.plotly_chart(chart, use_container_width=True)
    st.caption("A mixed-topic article still matches your interests when it includes at least one selected topic.")

    st.markdown("**Feedback in this comparison**")
    if snapshot["feedback_count"]:
        st.write(f"{snapshot['feedback_count']} rated articles; learned feedback vector "
                 + ("active." if snapshot["learned_vector"] else "inactive."))
        st.caption("Ratings influence article-level personalization; this snapshot alone cannot measure how much each rating changed the ranking. Selected interests are not replaced by dislikes.")
    else:
        st.write("No article ratings yet. This comparison uses your selected interests without learned feedback.")


st.caption("These are top-10 comparison snapshots, not your saved My Feed. Refresh after changing interests or feedback.")
