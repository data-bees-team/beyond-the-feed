"""Offline MMR lambda trade-off. Run: python3 -m evaluation.evaluate_lambda_tradeoff
Does not edit production code, feedback, or database records.
"""
from __future__ import annotations
import contextlib
import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from recommender.hybrid_pipeline import recommend_hybrid_feed
from utils.data_loader import load_articles
from utils.semantic_loader import load_article_embeddings, get_topic_description_embeddings

LAMBDAS = [0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
PROFILES = {"Technology": {"Technology": 1.0}, "Health": {"Health": 1.0}, "Technology + Health": {"Technology": 1.0, "Health": 1.0}}
OUT = Path('reports/lambda_tradeoff')

def topic_labels(value):
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(';') if v.strip()]
    return []

def distinct_nonempty(series):
    return len({str(v).strip() for v in series.dropna() if str(v).strip().lower() not in {'', 'nan', 'unknown', 'none'}})

def semantic_diversity(slate, embeddings):
    indices = slate['embedding_idx'].to_numpy(dtype=int)
    if len(indices) < 2:
        return float('nan')
    vec = np.asarray(embeddings[indices], dtype=float)
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    sims = vec @ vec.T
    return float(1 - sims[np.triu_indices(len(vec), k=1)].mean())

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_articles().reset_index(drop=True)
    embeddings, _ = load_article_embeddings(df)
    if len(df) != len(embeddings):
        raise ValueError('Embedding matrix and article dataframe must have identical row counts.')
    descriptions = get_topic_description_embeddings()
    records, slates = [], []
    for profile_name, profile in PROFILES.items():
        for lam in LAMBDAS:
            with contextlib.redirect_stdout(io.StringIO()):
                slate = recommend_hybrid_feed(
                    df=df, embeddings=embeddings, user_profile=profile,
                    topic_description_embeddings=descriptions, top_k=10,
                    topic_weight=0.5, lambda_param=lam,
                    seen_ids=None, user_vector=None,
                )
            if slate.empty:
                print(f'{profile_name} lambda={lam}: empty slate')
                continue
            topic_sets = [set(t for t in topic_labels(v) if t not in {'Other', 'Unknown'}) for v in slate['topic_list']]
            metrics = {
                'profile': profile_name, 'lambda': lam, 'articles': len(slate),
                'mean_hybrid_relevance': float(slate['hybrid_relevance'].mean()),
                'semantic_diversity': semantic_diversity(slate, embeddings),
                'countries': distinct_nonempty(slate['country']),
                'publishers': distinct_nonempty(slate['publisher']),
                'distinct_topics': len(set().union(*topic_sets)),
                'multi_topic_articles': sum(len(x) > 1 for x in topic_sets),
            }
            records.append(metrics)
            for rank, (_, article) in enumerate(slate.iterrows(), start=1):
                slates.append({'profile': profile_name, 'lambda': lam, 'rank': rank,
                               'article_id': str(article['article_id']), 'title': str(article['title']),
                               'country': article.get('country'), 'publisher': article.get('publisher'),
                               'hybrid_relevance': float(article['hybrid_relevance'])})
            print(f"{profile_name:19s} lambda={lam:.1f} relevance={metrics['mean_hybrid_relevance']:.3f} "
                  f"semantic_diversity={metrics['semantic_diversity']:.3f} "
                  f"countries={metrics['countries']} publishers={metrics['publishers']}")
    results = pd.DataFrame(records)
    if results.empty:
        raise RuntimeError('No results were produced.')
    results.to_csv(OUT / 'lambda_metrics.csv', index=False)
    pd.DataFrame(slates).to_csv(OUT / 'lambda_slates.csv', index=False)
    for measure in ['mean_hybrid_relevance', 'semantic_diversity', 'countries', 'publishers']:
        fig = px.line(results, x='lambda', y=measure, color='profile', markers=True,
                      title=f'MMR lambda trade-off: {measure.replace("_", " ").title()}')
        fig.update_layout(xaxis_title='Lambda (higher = more relevance weight)',
                          yaxis_title=measure.replace('_', ' ').title())
        fig.write_html(OUT / f'{measure}.html')
    print(f'\nSaved CSVs and charts to: {OUT.resolve()}')
    print('This isolates lambda changes with fixed interests, dataset, scoring and candidate pool. No feedback or seen-history effects.')
    print('Mean hybrid relevance is the model score, NOT human-rated relevance or satisfaction.')

if __name__ == '__main__':
    main()
