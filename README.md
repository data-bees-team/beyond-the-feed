# Beyond the Feed 🐝

**A diversity-aware, feedback-responsive news recommendation system**  
*Behind the Feed Data Challenge 2026 · Team Data Bees*

Beyond the Feed helps readers explore a broader range of news **while staying connected to their selected interests**. The current Streamlit application combines topic matching, semantic similarity, user feedback, and diversity-aware reranking. It also lets readers inspect their own recommendation exposure and compare a topic-focused baseline against the hybrid feed.

> **Scope of the claims:** Broader topic, country, or publisher coverage is measurable from article metadata; it is **not** by itself proof of greater viewpoint diversity, fairness, or improved user satisfaction.

## 1. Problem and dataset

Our Phase 1 analysis used **15,395 cleaned GDELT articles**, with coverage across **185 countries and 6 continents**. The corpus showed uneven representation:

- **54.3%** of articles came from the United States (27.5%), United Kingdom (16.1%), and India (10.7%) combined.
- Africa represented **5.34%** of inventory; South America **1.12%**.

These are **dataset-level findings**, not guarantees about what any individual user will see. They motivated our effort to make source and geographic concentration visible and to diversify the recommendation slate.

## 2. Current application

| Page | Purpose |
| --- | --- |
| **My Feed** (`pages/1_My_Feed.py`) | Select interests, receive personalized recommendations, provide 👍/👎 feedback, and load additional articles. |
| **Recommendation Insights** (`pages/2_Recommendation_Insights.py`) | Inspect selected interests, saved feedback, recommended-article history, topic mix, and geographic/publisher exposure. |
| **Compare Recommenders** (`pages/3_Compare_Recommenders.py`) | Compare 10 topic-focused baseline articles against 10 hybrid recommendations for the **current user's** interests and previously recommended history. Includes a separate, labeled controlled-feedback demonstration. |

The application also includes registration/login and user-specific profile, feedback, and recommendation-history handling. Account isolation should be validated in deployment testing.

## 3. Active recommendation pipeline

```text
Selected interests + cleaned GDELT articles
                    |
         Topic-match scoring
                    |
   Topic-description semantic similarity
                    |
        Combined hybrid relevance
                    |
     Optional semantic feedback adjustment
                    |
  Candidate eligibility + headline deduplication
                    |
   Exclude previously recommended article IDs
                    |
     Top-300 relevance candidate shortlist
                    |
   Unified diversity-aware MMR reranking
                    |
          Top-10 news recommendations
                    |
      User feedback saved for future ranking
```

### 3.1 Topic-focused baseline

`recommender/baseline.py` calculates topic relevance using the **average interest weight across an article's valid topic labels**. Articles with both selected and unselected topics can be eligible, but may receive a lower average than articles containing only selected topics. The baseline shown on the comparison page does **not** receive the learned semantic feedback vector. Other legacy baseline interfaces may optionally use the shared preference profile.

### 3.2 Hybrid topic and semantic relevance

The active implementation is in `recommender/hybrid_pipeline.py`, using helpers in `recommender/hybrid_retrieval.py`.

- **Topic match:** Uses the **maximum** selected-topic weight among an article's topic labels.
- **Semantic match:** Compares article embeddings with embeddings of the user's selected topic descriptions, using cosine similarity.
- **Hybrid score:** With the default `topic_weight=0.5`, combines topic and nonnegative semantic scores:

  `hybrid_relevance = 0.5 × topic_match + 0.5 × max(description_similarity, 0)`

- **Candidate eligibility:** Keeps articles with a positive selected-topic match **or** topic-description similarity of at least **0.65**.
- **Duplicate/seen filtering:** Deduplicates headlines and excludes previously recommended article IDs where supplied.

The semantic threshold can admit articles without exact topic matches, but it does **not** guarantee that such articles will appear in the final top 10. Multi-topic articles can also expose readers to additional topics without requiring semantic-only retrieval.

### 3.3 Feedback-responsive ranking

`recommender/nlp_profile.py` builds a semantic user vector from saved article feedback. The implementation weights liked article embeddings by **+0.3** and disliked article embeddings by **−0.2**. Candidate cosine similarity to this vector contributes to hybrid relevance:

`adjusted_relevance = clip(hybrid_relevance + 0.15 × feedback_similarity, 0, 1)`

With no nonzero feedback vector, this adjustment is inactive. The mechanism updates **ranking scores**, not the underlying embedding model; it is not reinforcement learning or online model retraining.

### 3.4 Unified diversity-aware reranking

`recommender/unified_reranker.py` applies Maximal Marginal Relevance (MMR) to a shortlist of up to **300** eligible candidates. Its redundancy measure combines:

| Redundancy signal | Weight |
| --- | ---: |
| Semantic similarity to already selected articles | 0.50 |
| Geographic redundancy | 0.30 |
| Publisher redundancy | 0.20 |

The selection objective is:

`MMR(a) = λ × adjusted_relevance(a) − (1 − λ) × combined_redundancy(a, selected)`

**My Feed uses `λ = 0.60`** in its current recommendation call. This is a relevance/diversity trade-off, **not a hard quota** for publishers, countries, or discovery articles. The final slate is normally **10 articles**, subject to candidate availability.

## 4. Feedback, persistence, and user transparency

- `utils/feedback_manager.py` handles saved 👍/👎 feedback and its supporting persistence.
- `utils/user_manager.py` and authentication components manage user accounts and saved interests.
- Recommendation history supports exclusion of previously recommended article IDs.
- **Recommendation Insights** reports descriptive information about the user's own activity and exposure. Counts of topic labels on rated articles are **not** learned model weights.
- **Compare Recommenders** generates a fixed, current-user comparison snapshot; its two slates are alternatives, not extra articles added to My Feed.

SQLite is used for application persistence. Local database files and credentials should remain excluded from Git. See `.gitignore` and the database utilities for the actual local paths and schema.

## 5. Evaluation

### 5.1 Live baseline-versus-hybrid comparison

The Compare Recommenders page uses the same selected interests and previously recommended history for both algorithms. It reports topic-label composition and distinct topics, countries, and publishers in each **10-article** slate.

In one observed demonstration with **Conflict and Security** and **Environment and Climate** selected:

| Metric (10 recommendations) | Topic baseline | Hybrid |
| --- | ---: | ---: |
| Distinct topic labels | 2 | 9 |
| Distinct countries | 4 | 10 |
| Distinct publishers | 9 | 10 |
| Articles containing selected **and** additional topics | 0 | 10 |

**Interpretation:** This snapshot illustrates broader metadata coverage while retaining selected-topic connections. It is **one example**, not an average across users, a controlled component ablation, or proof of higher recommendation quality.

### 5.2 Controlled feedback-responsiveness experiments

The existing `evaluation/evaluate_feedback_adaptation.py` experiment compares cold-start and synthetic-feedback rankings using the same article dataset, selected interest, and initial eligibility conditions. Feedback is simulated from articles in the initial slate; it does **not** modify real users' feedback.

| Selected topic | Initial top-10 articles retained | Replaced after simulated feedback |
| --- | ---: | ---: |
| Technology | 3 | 7 |
| Health | 6 | 4 |
| Politics and Government | 5 | 5 |

In all three runs, the simulated disliked article moved out of the top 10; the simulated liked article remained at rank 1. **These tests show ranking responsiveness, not that the replacement articles are objectively more relevant or preferred by users.**

### 5.3 Earlier component benchmarks (historical)

Before the current hybrid pipeline, the team evaluated separate topic-baseline, geographic-MMR, publisher-diversity, feedback-enhanced, and combined personalized/discovery configurations. The earlier benchmark used three simulated personas on the cleaned GDELT corpus.

| Earlier configuration | Mean countries (K=10) | Mean continents | Mean publishers | Mean topics |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 5.3 | 3.3 | 8.7 | 2.0 |
| Geographic MMR | 10.0 | 6.0 | 9.7 | 2.0 |
| Publisher diversity | 6.0 | 3.7 | 10.0 | 2.0 |
| Feedback-enhanced baseline | 5.7 | 3.7 | 8.7 | 2.0 |
| Feedback-enhanced geographic MMR | 10.0 | 6.0 | 9.7 | 2.0 |
| Feedback-enhanced publisher diversity | 6.7 | 4.0 | 10.0 | 2.0 |
| Combined personalized + discovery | 7.7 | 6.0 | 9.7 | 3.7 |

These figures are retained from the **earlier offline evaluation**. They are **not** results for the active `hybrid_pipeline.py` and must not be used to claim a current 70/30 split or a guaranteed 25–30% discovery allocation.

## 6. How the system evolved

The team's initial architecture developed and evaluated several complementary components:

- `recommender/geographic_diversity.py` — geographic diversity scoring and MMR.
- `recommender/publisher_diversity.py` — publisher diversity scoring.
- `recommender/interest_bridging.py` — thematic connections through multi-topic articles.
- `recommender/user_preferences.py` and `recommender/personalization_engine.py` — feedback-derived entity preferences and personalized relevance.
- `recommender/feed_composer.py` — an earlier combined personalized/discovery strategy, including a **70/30 design**.

The **current My Feed path** uses `hybrid_pipeline.py`, `nlp_profile.py`, and `unified_reranker.py`. The unified reranker reuses geographic and publisher redundancy logic. The earlier components and benchmarks remain valuable project contributions, but they should not be confused with the active feed-composition policy.

## 7. Repository guide

```text
beyond-the-feed/
├── Home.py
├── pages/
│   ├── 1_My_Feed.py
│   ├── 2_Recommendation_Insights.py
│   └── 3_Compare_Recommenders.py
├── components/
│   ├── auth.py
│   └── news_card.py
├── recommender/
│   ├── baseline.py
│   ├── hybrid_pipeline.py
│   ├── hybrid_retrieval.py
│   ├── nlp_profile.py
│   ├── unified_reranker.py
│   ├── geographic_diversity.py
│   ├── publisher_diversity.py
│   ├── interest_bridging.py
│   ├── user_preferences.py
│   ├── personalization_engine.py
│   └── feed_composer.py
├── utils/
│   ├── data_loader.py
│   ├── semantic_loader.py
│   ├── feedback_manager.py
│   ├── db_manager.py
│   └── user_manager.py
├── evaluation/
│   └── evaluate_feedback_adaptation.py
├── tests/
├── data/                       # Local dataset and supporting assets
├── reports/                    # Evaluation outputs (as available)
├── styles/
│   └── style.css
├── requirements.txt
└── README.md
```

The listing highlights the relevant modules rather than every repository file. Some data, embedding assets, and generated reports may be ignored by Git or require separate preparation.

## 8. Local setup

**Requirements:** Python 3.10+ recommended, Git, and the data/embedding assets expected by the loaders.

```bash
# Clone and enter the project
git clone https://github.com/data-bees-team/beyond-the-feed.git
cd beyond-the-feed

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Run tests
python -m pytest tests/ -q

# Start the application
streamlit run Home.py
```

Open `http://localhost:8501`. If data or embedding files are not included in the repository, supply the required assets in the locations expected by `utils/data_loader.py` and `utils/semantic_loader.py` before launching. The application should be validated again in its deployment environment, including account separation and persistent storage behavior.

For a controlled feedback test, from the project root:

```bash
python evaluation/evaluate_feedback_adaptation.py --topics Technology
```

## 9. Limitations and next steps

- Topic, country, and publisher labels are **proxies for exposure**, not direct measures of ideological viewpoint or fairness.
- A slate of 10 has an upper bound of 10 distinct countries or publishers; ceiling effects matter when comparing diversity.
- The hybrid system balances relevance and redundancy but **does not guarantee** particular regional, publisher, or cross-topic quotas.
- Synthetic feedback tests demonstrate **responsiveness**, not measured user satisfaction or causal attribution of every ranking change.
- Before final deployment, validate multi-account isolation, database persistence, required embedding assets, and the complete feedback → Load More → Insights → Comparison journey.

## 10. Team and acknowledgments

Built by **Data Bees** for the **Behind the Feed Data Challenge 2026**. The project integrates the team's original geographic-diversity, publisher-diversity, thematic-bridging, feedback-personalization, and evaluation work with the current hybrid semantic recommendation and user-facing comparison experience.
