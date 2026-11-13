
import pandas as pd

from evaluation.semantic_ablation import load_inputs, evaluate_slate
from recommender.baseline import recommend_baseline
from recommender.hybrid_pipeline import recommend_hybrid_feed


def main():
    df, embeddings, topic_embeddings = load_inputs()

    profiles = [
        {"Technology": 1.0},
        {"Health": 1.0},
        {"Technology": 1.0, "Health": 1.0},
    ]

    results = []

    for profile in profiles:
        label = " + ".join(profile)

        baseline = recommend_baseline(
            df=df,
            user_profile=profile,
            top_k=10,
        )

        hybrid = recommend_hybrid_feed(
            df=df,
            embeddings=embeddings,
            user_profile=profile,
            topic_description_embeddings=topic_embeddings,
            top_k=10,
            topic_weight=0.5,
            lambda_param=0.7,
            seen_ids=None,
            user_vector=None,
        )
        
        
        topic_only_hybrid = recommend_hybrid_feed(
                    df=df,
                    embeddings=embeddings,
                    user_profile=profile,
                    topic_description_embeddings=topic_embeddings,
                    top_k=10,
                    topic_weight=1.0,
                    lambda_param=0.7,
                    seen_ids=None,
                    user_vector=None,
                )

        # for system, slate in [
        #     ("Baseline", baseline),
        #     ("Hybrid", hybrid),
        # ]:
        
        
        for system, slate in [
            ("Baseline", baseline),
            ("Hybrid", hybrid),
            ("Topic-only Hybrid", topic_only_hybrid),
        ]:
            
            if label == "Technology + Health" and system == "Hybrid":
                print("\n=== HYBRID SEMANTIC SCORE REVIEW ===")

                for i, (_, article) in enumerate(slate.iterrows(), 1):
                    print(f"\n{i}. {article['title']}")
                    print(f"   Topics: {article['topic_list']}")
                    print(f"   Topic match: {article['hybrid_topic_score']:.3f}")
                    print(f"   Semantic: {article['description_semantic_score']:.3f}")
                    print(f"   Final relevance: {article['hybrid_relevance']:.3f}")
            
            results.append({
                "profile": label,
                "system": system,
                "countries": slate["country"].nunique(),
                "continents": slate["continent"].nunique(),
                "publishers": slate["publisher"].nunique(),
                "multi_topic_articles": int(
                    (slate["topic_list"].apply(len) > 1).sum()
                ),
            })
            
    # Evaluate individual topic similarities
    import numpy as np

    print("\n=== INDIVIDUAL TOPIC SEMANTIC SCORES ===")

    profile = {"Technology": 1.0, "Health": 1.0}

    slate = recommend_hybrid_feed(
        df=df,
        embeddings=embeddings,
        user_profile=profile,
        topic_description_embeddings=topic_embeddings,
        top_k=10,
        topic_weight=0.5,
        lambda_param=0.7,
        seen_ids=None,
        user_vector=None,
    )

    tech_vector = topic_embeddings["Technology"]
    health_vector = topic_embeddings["Health"]

    def cosine_similarity(a, b):
        a = np.asarray(a)
        b = np.asarray(b)

        if b.ndim == 1:
            b = b.reshape(1, -1)

        similarities = (b @ a) / (
            np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-12
        )

        return float(np.max(similarities))

    for i, (_, article) in enumerate(slate.iterrows(), 1):
        article_vector = embeddings[int(article["embedding_idx"])]

        tech_score = cosine_similarity(article_vector, tech_vector)
        health_score = cosine_similarity(article_vector, health_vector)

        print(f"\n{i}. {article['title']}")
        print(f"   Technology: {tech_score:.3f}")
        print(f"   Health:     {health_score:.3f}")
        print(f"   Maximum:    {max(tech_score, health_score):.3f}")
        print(f"   Average:    {(tech_score + health_score) / 2:.3f}")
            
        

    
    print("\n=== SEMANTIC ABLATION: HEADLINE COMPARISON ===")

    profile = {"Technology": 1.0, "Health": 1.0}

    for name, weight in [
        ("Full Hybrid", 0.5),
        ("Topic-only Hybrid", 1.0),
    ]:
        slate = recommend_hybrid_feed(
            df=df,
            embeddings=embeddings,
            user_profile=profile,
            topic_description_embeddings=topic_embeddings,
            top_k=10,
            topic_weight=weight,
            lambda_param=0.7,
            seen_ids=None,
            user_vector=None,
        )

        print(f"\n=== {name} ===")

        for i, (_, article) in enumerate(slate.iterrows(), 1):
            print(f"{i}. {article['title']}")
    
    
    print("\n=== BASELINE VS HYBRID ===")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
