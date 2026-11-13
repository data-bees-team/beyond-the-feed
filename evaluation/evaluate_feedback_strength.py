"""Offline feedback-strength sensitivity test. Does not edit production code.

Run from the project root:
  python evaluation/evaluate_feedback_strength.py --topics Technology Health

Uses synthetic likes/dislikes of the original top-2 recommendations.
Measures ranking responsiveness, not actual user satisfaction.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import recommender.hybrid_pipeline as pipeline
from utils.data_loader import load_articles
from utils.semantic_loader import load_article_embeddings, get_topic_description_embeddings


def cosine_to_anchor(embeddings, anchor):
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(anchor)
    return np.divide(embeddings @ anchor, norms, out=np.zeros(len(embeddings)), where=norms > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics', nargs='+', required=True)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--lambda-param', type=float, default=0.6)
    parser.add_argument('--output-dir', default='reports/feedback_strength')
    args = parser.parse_args()

    df = load_articles().reset_index(drop=True)
    embeddings, _ = load_article_embeddings(df)
    descriptions = get_topic_description_embeddings()
    if len(df) != len(embeddings):
        raise ValueError('Article/embedding alignment mismatch')
    if not df['article_id'].is_unique:
        raise ValueError('Expected unique article_id values')
    unknown = set(args.topics) - set(descriptions)
    if unknown:
        raise ValueError(f'Unknown topics: {sorted(unknown)}')
    profile = {topic: 1.0 for topic in args.topics}
    original_score_fn = pipeline.compute_nlp_content_scores

    def recommend(vec, strength):
        # Production pipeline uses a fixed 0.15 multiplier. Scale only the
        # feedback-score function temporarily so effective strength is varied.
        # Restore the function even if ranking raises an exception.
        def scaled_score(user_vector, all_embeddings):
            return original_score_fn(user_vector, all_embeddings) * (strength / 0.15)
        pipeline.compute_nlp_content_scores = scaled_score
        try:
            return pipeline.recommend_hybrid_feed(
                df=df, embeddings=embeddings, user_profile=profile,
                topic_description_embeddings=descriptions,
                top_k=args.top_k, lambda_param=args.lambda_param,
                seen_ids=None, user_vector=vec,
            )
        finally:
            pipeline.compute_nlp_content_scores = original_score_fn

    zero = np.zeros(embeddings.shape[1], dtype=np.float32)
    before = recommend(zero, 0)
    if len(before) < 2:
        raise ValueError('Need at least two baseline articles')
    lookup = {str(x): i for i, x in enumerate(df['article_id'])}
    liked_id, disliked_id = [str(x) for x in before['article_id'].iloc[:2]]
    liked_vec = embeddings[lookup[liked_id]]
    disliked_vec = embeddings[lookup[disliked_id]]
    vectors = {
        'like_only': 0.3 * liked_vec,
        'dislike_only': -0.2 * disliked_vec,
        'like_and_dislike': 0.3 * liked_vec - 0.2 * disliked_vec,
    }
    before_ids = [str(x) for x in before['article_id']]
    baseline_ranks = {x: i + 1 for i, x in enumerate(before_ids)}
    sim_liked = cosine_to_anchor(embeddings, liked_vec)
    sim_disliked = cosine_to_anchor(embeddings, disliked_vec)
    records = []
    ranks = []
    for scenario, vector in vectors.items():
        for strength in (0.0, 0.05, 0.10, 0.15):
            after = recommend(vector, strength)
            after_ids = [str(x) for x in after['article_id']]
            after_ranks = {x: i + 1 for i, x in enumerate(after_ids)}
            common = set(before_ids) & set(after_ids)
            indices = [lookup[x] for x in after_ids]
            records.append({
                'scenario': scenario, 'feedback_weight': strength,
                'retained_from_baseline': len(common),
                'new_articles': len(set(after_ids) - set(before_ids)),
                'changed_ranks_among_retained': sum(baseline_ranks[x] != after_ranks[x] for x in common),
                'liked_rank': after_ranks.get(liked_id),
                'disliked_rank': after_ranks.get(disliked_id),
                'mean_similarity_to_liked': round(float(np.mean(sim_liked[indices])), 4),
                'mean_similarity_to_disliked': round(float(np.mean(sim_disliked[indices])), 4),
                'countries': after['country'].replace('', np.nan).nunique() if 'country' in after else None,
                'continents': after['continent'].replace('', np.nan).nunique() if 'continent' in after else None,
                'publishers': after['publisher'].replace('', np.nan).nunique() if 'publisher' in after else None,
            })
            for rank, article_id in enumerate(after_ids, start=1):
                ranks.append({
                    'scenario': scenario, 'feedback_weight': strength,
                    'rank': rank, 'article_id': article_id,
                    'title': str(df.iloc[lookup[article_id]]['title']),
                    'baseline_rank': baseline_ranks.get(article_id),
                })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / 'strength_summary.csv', index=False)
    pd.DataFrame(ranks).to_csv(output / 'strength_rankings.csv', index=False)
    print('Synthetic feedback anchors:', json.dumps({
        'liked': {'id': liked_id, 'title': str(df.iloc[lookup[liked_id]]['title'])},
        'disliked': {'id': disliked_id, 'title': str(df.iloc[lookup[disliked_id]]['title'])},
    }, indent=2))
    print(pd.DataFrame(records).to_string(index=False))
    print(f'Wrote {output / "strength_summary.csv"} and {output / "strength_rankings.csv"}')


if __name__ == '__main__':
    main()
