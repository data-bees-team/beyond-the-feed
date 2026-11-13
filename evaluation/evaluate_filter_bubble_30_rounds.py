"""Offline 30-round filter-bubble stress test. Run from project root:
    python evaluation/evaluate_filter_bubble_30_rounds.py
Does not change production code, user feedback, or database records.
"""
from __future__ import annotations
import contextlib
import io
from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px

from recommender.baseline import recommend_baseline
from recommender.hybrid_pipeline import recommend_hybrid_feed
from utils.data_loader import load_articles
from utils.semantic_loader import load_article_embeddings, get_topic_description_embeddings

ROUNDS = 30
TOP_K = 10
PROFILE = {"Technology": 1.0}
FEEDBACK_LIKE = 0.3
LAMBDA = 0.60  # same default as current Compare Recommenders page
OUT = Path("reports/filter_bubble")
AI_PATTERN = re.compile(r"\b(?:AI|artificial intelligence|machine learning|deep learning|generative AI|chatgpt|large language models?|LLMs?|neural networks?)\b", re.I)


def topics(value):
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [x.strip() for x in value.split(";") if x.strip()]
    return []


def slate_metrics(slate, embedding_by_id):
    if slate.empty:
        return dict(articles=0, countries=0, publishers=0, distinct_topics=0,
                    multi_topic=0, ai_headlines=0, semantic_diversity=np.nan)
    valid = [set(t for t in topics(v) if t not in {"Other", "Unknown"})
             for v in slate.get("topic_list", pd.Series([[]] * len(slate))).tolist()]
    topic_union = set().union(*valid)
    vectors = [embedding_by_id[str(a)] for a in slate["article_id"] if str(a) in embedding_by_id]
    diversity = np.nan
    if len(vectors) > 1:
        mat = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.maximum(norms, 1e-12)
        sims = mat @ mat.T
        upper = sims[np.triu_indices(len(mat), k=1)]
        diversity = float(1 - upper.mean())
    return dict(
        articles=len(slate),
        countries=slate["country"].dropna().astype(str).replace({"Unknown": np.nan}).dropna().nunique() if "country" in slate else 0,
        publishers=slate["publisher"].dropna().astype(str).replace({"Unknown": np.nan}).dropna().nunique() if "publisher" in slate else 0,
        distinct_topics=len(topic_union),
        multi_topic=sum(len(ts) > 1 for ts in valid),
        ai_headlines=sum(bool(AI_PATTERN.search(str(t))) for t in slate["title"]),
        semantic_diversity=diversity,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_articles().reset_index(drop=True)
    embeddings, _ = load_article_embeddings(df)
    if len(df) != len(embeddings):
        raise ValueError("Embeddings must align with the complete article dataframe")
    embedding_by_id = {str(a): embeddings[i] for i, a in enumerate(df["article_id"])}
    descriptions = get_topic_description_embeddings()
    user_vec = np.zeros(embeddings.shape[1], dtype=np.float64)
    seen_hybrid, seen_baseline = set(), set()
    liked_ids = set()
    records, selected_likes = [], []

    for round_number in range(1, ROUNDS + 1):
        unseen_base = df[~df["article_id"].astype(str).isin(seen_baseline)].copy()
        baseline = recommend_baseline(unseen_base, PROFILE, top_k=TOP_K)
        # Hide verbose debug output from production function; do not alter its behavior.
        with contextlib.redirect_stdout(io.StringIO()):
            hybrid = recommend_hybrid_feed(
                df=df, embeddings=embeddings, user_profile=PROFILE,
                topic_description_embeddings=descriptions, top_k=TOP_K,
                lambda_param=LAMBDA, seen_ids=list(seen_hybrid), user_vector=user_vec,
            )
        for name, slate in (("Topic baseline", baseline), ("Hybrid", hybrid)):
            metrics = slate_metrics(slate, embedding_by_id)
            records.append({"round": round_number, "system": name,
                            "cumulative_likes_before_round": len(liked_ids), **metrics})
        if hybrid.empty or baseline.empty:
            print(f"Round {round_number}: no more articles available; stopping")
            break
        seen_hybrid.update(hybrid["article_id"].astype(str))
        seen_baseline.update(baseline["article_id"].astype(str))

        # Simulate exactly one AI-related Like if an AI headline appears in the hybrid top 10.
        # No fabricated Like if no qualifying headline is present.
        ai_candidates = hybrid[hybrid["title"].astype(str).str.contains(AI_PATTERN, na=False)]
        if not ai_candidates.empty:
            chosen = ai_candidates.iloc[0]
            article_id = str(chosen["article_id"])
            if article_id not in liked_ids and article_id in embedding_by_id:
                user_vec += FEEDBACK_LIKE * embedding_by_id[article_id]
                liked_ids.add(article_id)
                selected_likes.append({"round": round_number, "article_id": article_id,
                                       "title": str(chosen["title"])})
        print(f"Round {round_number:02d}: AI Likes accumulated={len(liked_ids)}, "
              f"hybrid countries={records[-1]['countries']}, "
              f"hybrid semantic diversity={records[-1]['semantic_diversity']:.3f}")

    results = pd.DataFrame(records)
    results.to_csv(OUT / "round_metrics.csv", index=False)
    pd.DataFrame(selected_likes, columns=["round", "article_id", "title"]).to_csv(OUT / "simulated_likes.csv", index=False)
    for metric in ("countries", "distinct_topics", "multi_topic", "ai_headlines", "semantic_diversity"):
        fig = px.line(results, x="round", y=metric, color="system", markers=True,
                      title=f"30-round test: {metric.replace('_', ' ').title()}")
        fig.write_html(str(OUT / f"{metric}.html"))
    print("\nSaved:", OUT.resolve())
    print("AI-related Likes recorded:", len(liked_ids), "across", results["round"].nunique(), "rounds")
    print("Note: baseline does not learn feedback; both systems independently exclude their own prior recommendations.")
    print("AI detection is a headline keyword proxy, not a verified AI-topic label or human behavior.")


if __name__ == "__main__":
    main()
