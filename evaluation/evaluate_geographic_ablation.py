"""Offline geographic MMR ablation.
Run from project root: python3 -m evaluation.evaluate_geographic_ablation
No production files are modified. Requires the same dependencies as evaluate_lambda_tradeoff.
"""
from __future__ import annotations
import contextlib
import io
from pathlib import Path
import numpy as np
import pandas as pd
import recommender.unified_reranker as reranker
from recommender.hybrid_pipeline import recommend_hybrid_feed
from utils.data_loader import load_articles
from utils.semantic_loader import load_article_embeddings, get_topic_description_embeddings

PROFILES = {
    'Technology': {'Technology': 1.0},
    'Health': {'Health': 1.0},
    'Technology + Health': {'Technology': 1.0, 'Health': 1.0},
}
LAMBDA = 0.7  # Set to 0.6 if that is the configuration you want to evaluate.
OUT = Path('reports/geographic_ablation')
ORIGINAL = reranker.calculate_combined_redundancy


def no_geography(candidate, selected_items, candidate_embedding, selected_embeddings,
                 semantic_weight=0.5, geographic_weight=0.3, publisher_weight=0.2):
    """Ablate geography; renormalize the remaining semantic/publisher weights."""
    remaining = semantic_weight + publisher_weight
    if remaining <= 0:
        raise ValueError('Remaining diversity weights must be positive.')
    return ORIGINAL(candidate, selected_items, candidate_embedding, selected_embeddings,
                    semantic_weight=semantic_weight / remaining,
                    geographic_weight=0.0,
                    publisher_weight=publisher_weight / remaining)


def unique_valid(series):
    bad = {'', 'nan', 'none', 'null', 'unknown', 'undetermined', 'n/a', 'missing'}
    return len({str(v).strip() for v in series.dropna() if str(v).strip().lower() not in bad})


def semantic_diversity(slate, embeddings):
    indices = slate['embedding_idx'].to_numpy(dtype=int)
    if len(indices) < 2:
        return float('nan')
    vectors = np.asarray(embeddings[indices], dtype=float)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    similarities = vectors @ vectors.T
    return float(1 - similarities[np.triu_indices(len(vectors), k=1)].mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_articles().reset_index(drop=True)
    embeddings, _ = load_article_embeddings(df)
    if len(df) != len(embeddings):
        raise ValueError('Article and embedding counts differ.')
    descriptions = get_topic_description_embeddings()
    metrics, articles = [], []

    for profile_name, profile in PROFILES.items():
        for variant in ('Full hybrid', 'Without geographic MMR'):
            reranker.calculate_combined_redundancy = (
                ORIGINAL if variant == 'Full hybrid' else no_geography
            )
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    slate = recommend_hybrid_feed(
                        df=df, embeddings=embeddings, user_profile=profile,
                        topic_description_embeddings=descriptions, top_k=10,
                        topic_weight=0.5, lambda_param=LAMBDA,
                        seen_ids=None, user_vector=None,
                    )
            finally:
                reranker.calculate_combined_redundancy = ORIGINAL
            if slate.empty:
                raise RuntimeError(f'Empty slate for {profile_name} / {variant}')
            row = {
                'profile': profile_name, 'variant': variant, 'lambda': LAMBDA,
                'articles': len(slate),
                'mean_hybrid_relevance': float(slate['hybrid_relevance'].mean()),
                'semantic_diversity': semantic_diversity(slate, embeddings),
                'countries': unique_valid(slate['country']),
                'publishers': unique_valid(slate['publisher']),
            }
            metrics.append(row)
            for rank, (_, article) in enumerate(slate.iterrows(), 1):
                articles.append({
                    'profile': profile_name, 'variant': variant, 'rank': rank,
                    'article_id': str(article['article_id']),
                    'title': str(article.get('title', '')),
                    'country': article.get('country', ''),
                    'publisher': article.get('publisher', ''),
                    'hybrid_relevance': float(article['hybrid_relevance']),
                })
            print(f"{profile_name:19s} {variant:23s} "
                  f"relevance={row['mean_hybrid_relevance']:.3f} "
                  f"semantic_diversity={row['semantic_diversity']:.3f} "
                  f"countries={row['countries']} publishers={row['publishers']}")

    pd.DataFrame(metrics).to_csv(OUT / 'geographic_ablation_metrics.csv', index=False)
    pd.DataFrame(articles).to_csv(OUT / 'geographic_ablation_slates.csv', index=False)
    print(f'\nSaved CSVs to: {OUT.resolve()}')
    print('Only geographic redundancy is removed; remaining weights are renormalized.')
    print('Relevance is a model score, not a human judgment. No feedback or seen history.')


if __name__ == '__main__':
    main()
