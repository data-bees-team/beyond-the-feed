"""Controlled, offline feedback-adaptation check for Beyond the Feed.

Run from project root:
  python evaluation/evaluate_feedback_adaptation.py --topics Technology --output-dir reports/feedback_evaluation

Requires existing project modules and data/embedding assets. No database writes.
This measures *responsiveness*, not ground-truth recommendation quality.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.hybrid_pipeline import recommend_hybrid_feed
from utils.data_loader import load_articles
from utils.semantic_loader import load_article_embeddings, get_topic_description_embeddings


def rank_map(frame):
    return {str(a): i + 1 for i, a in enumerate(frame["article_id"].tolist())}


def topic_group(topics, selected):
    valid = {t for t in topics if t not in ("Other", "Unknown")}
    matching = valid & selected
    if not matching:
        return "no_exact_topic_match"
    return "selected_and_other_topics" if valid - selected else "selected_topics_only"


def summary(frame, selected):
    out = {"article_count": len(frame)}
    for col in ("publisher", "country", "continent"):
        if col in frame.columns:
            out[f"unique_{col}s"] = int(frame[col].replace("", np.nan).nunique())
    groups = frame["topic_list"].apply(lambda x: topic_group(x, selected)).value_counts()
    for group in ("selected_topics_only", "selected_and_other_topics", "no_exact_topic_match"):
        out[group] = int(groups.get(group, 0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="+", required=True, help="Exact topic names, e.g. --topics Technology Science")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lambda-param", type=float, default=0.60)
    parser.add_argument("--output-dir", default="reports/feedback_evaluation")
    args = parser.parse_args()

    df = load_articles().reset_index(drop=True)
    embeddings, _ = load_article_embeddings(df)
    descriptions = get_topic_description_embeddings()
    if len(df) != len(embeddings):
        raise ValueError("Article/embedding alignment mismatch")
    if "article_id" not in df or not df["article_id"].is_unique:
        raise ValueError("Expected unique article_id values")
    unknown = set(args.topics) - set(descriptions)
    if unknown:
        raise ValueError(f"Unknown topic descriptions: {sorted(unknown)}. Available: {sorted(descriptions)}")

    profile = {topic: 1.0 for topic in args.topics}
    selected = set(args.topics)
    def recommend(vec):
        # Full, unchanged candidate set and no seen-id filtering in both runs.
        return recommend_hybrid_feed(
            df=df, embeddings=embeddings, user_profile=profile,
            topic_description_embeddings=descriptions,
            top_k=args.top_k, lambda_param=args.lambda_param,
            seen_ids=None, user_vector=vec,
        )

    before = recommend(np.zeros(embeddings.shape[1], dtype=np.float32))
    if len(before) < 2:
        raise RuntimeError("Need at least two initial recommendations to simulate feedback")

    # Deterministic scenario: like rank 1, dislike rank 2. This is synthetic,
    # not observed human feedback. It can be replaced with controlled test IDs.
    liked_id, disliked_id = [str(x) for x in before["article_id"].iloc[:2]]
    lookup = {str(a): i for i, a in enumerate(df["article_id"])}
    user_vec = (0.3 * embeddings[lookup[liked_id]] - 0.2 * embeddings[lookup[disliked_id]])
    after = recommend(user_vec)

    before_ranks, after_ranks = rank_map(before), rank_map(after)
    ids = list(dict.fromkeys(list(before_ranks) + list(after_ranks)))
    details = []
    for article_id in ids:
        ix = lookup[article_id]
        details.append({
            "article_id": article_id,
            "title": str(df.iloc[ix]["title"]),
            "topic_list": "; ".join(df.iloc[ix]["topic_list"]),
            "topic_group": topic_group(df.iloc[ix]["topic_list"], selected),
            "before_rank": before_ranks.get(article_id),
            "after_rank": after_ranks.get(article_id),
            "synthetic_feedback": "liked" if article_id == liked_id else ("disliked" if article_id == disliked_id else "none"),
        })

    common = set(before_ranks) & set(after_ranks)
    result = {
        "experiment": "synthetic feedback responsiveness; not user-study relevance improvement",
        "selected_topics": args.topics,
        "top_k": args.top_k,
        "lambda_param": args.lambda_param,
        "feedback": {"liked_article_id": liked_id, "disliked_article_id": disliked_id},
        "same_candidate_pool": True,
        "seen_ids_excluded": False,
        "overlap_count": len(common),
        "changed_membership_count": len(set(before_ranks) ^ set(after_ranks)),
        "changed_rank_count_among_common": sum(before_ranks[x] != after_ranks[x] for x in common),
        "liked_rank_before": before_ranks.get(liked_id),
        "liked_rank_after": after_ranks.get(liked_id),
        "disliked_rank_before": before_ranks.get(disliked_id),
        "disliked_rank_after": after_ranks.get(disliked_id),
        "before": summary(before, selected),
        "after": summary(after, selected),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(details).to_csv(output / "rank_changes.csv", index=False)
    before.assign(evaluation_stage="before").to_csv(output / "before_slate.csv", index=False)
    after.assign(evaluation_stage="after").to_csv(output / "after_slate.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved evaluation files to: {output}")


if __name__ == "__main__":
    main()
