# Beyond the Feed 🐝
### Discover More — A Diversity-Aware News Recommendation System

**Team Data Bees | Behind the Feed Data Challenge 2026 — Phase 2**

Beyond the Feed is an interactive news recommendation application designed to help readers **discover relevant stories without seeing only more of the same**. It combines personal interests and semantic similarity with feedback-driven adaptation and diversity-aware ranking across topics, countries, and publishers.

**[GitHub Repository](https://github.com/data-bees-team/beyond-the-feed)** · **[Live Demo](https://beyondthefeed-databees.streamlit.app/)**
## The problem

A news feed optimized only for relevance can repeatedly recommend similar stories or concentrate exposure among familiar sources and regions. This limits opportunities to encounter relevant news from different perspectives and places.

Our goal is to balance two needs: **recommend news a reader cares about** and **make room for relevant discovery**.

## What we built

A multi-user Streamlit application with:

- **Guest, Sign Up and Login:** explore the application or create an account with saved preferences.
- **My Feed:** personalized news recommendations with article-level **Interested / Not Interested** feedback and a **Load More** experience.
- **Discover:** a discovery-oriented feed using the same recommendation engine with stronger exploration.
- **Recommendation Insights:** visual summaries of the user's recommendations and interactions.
- **Compare Recommenders:** a side-by-side comparison of the topic-based baseline and diversity-aware hybrid feed for the current user.

## Dataset

| Property | Value |
|---|---:|
| News articles | **15,395** |
| Countries represented | **185** |
| Continents represented | **6** |
| Source | **Preprocessed GDELT news data** |

The application uses a **historical, cleaned news dataset**, not a continuously updated live news feed. News articles can have multiple topic labels.

## How recommendations work

```text
User-selected interests + past article feedback
                       │
                       ▼
         Topic relevance scoring
                       │
                       ├──────────► Baseline recommendations
                       │
                       ▼
       Semantic article matching
      (BGE title text embeddings)
                       │
                       ▼
    Hybrid candidate retrieval + scoring
                       │
                       ▼
       Unified diversity-aware reranking
    (semantic, geographic, publisher MMR)
                       │
                       ▼
       Personalized feed / Discover
                       │
                       ▼
         Interested / Not Interested
                       │
                       └──► Updated user preferences
```

### 1. Topic-based baseline

The baseline ranks articles using the reader's selected topic interests. It provides a simple, interpretable reference point for evaluating the hybrid system. Because articles may have several topics, an article that covers a selected topic **alongside other topics** can receive a lower topic score than a more narrowly focused article.

### 2. Semantic retrieval

The hybrid system uses **BAAI/bge-base-en-v1.5** embeddings of article titles to identify semantically related news, including stories that may not share an exact topic label with the reader's selected interests.

### 3. Feedback-based personalization

Readers can mark articles **Interested** or **Not Interested**. These signals adjust future recommendations while allowing exploration; a single dislike is not treated as a permanent ban on a topic.

### 4. Diversity-aware reranking

**Maximal Marginal Relevance (MMR)** balances relevance against repetition. The unified reranker considers semantic redundancy, country representation, and publisher overlap to help produce a more varied recommendation slate.

### 5. Discovery

The **Discover** experience uses the same core engine but places more emphasis on exploration. It is intended to surface relevant stories beyond the reader's most obvious matches.

## Evaluation

We evaluated the recommendation approach using baseline comparisons, ablations, feedback experiments, and diversity/relevance trade-off tests. The repository includes evaluation scripts and outputs under `evaluation/` and `reports/`.

| Experiment | What it examines |
|---|---|
| Baseline vs. hybrid | Changes in relevance and topic, geographic, and publisher coverage |
| Geographic ablation | Effect of geographic diversity reranking |
| Semantic ablation | Contribution of semantic retrieval and ranking |
| MMR lambda sweep | Trade-off between relevance and diversity |
| Feedback adaptation | How recommendations respond to simulated positive and negative feedback |
| 30-round filter-bubble simulation | Whether diversity is maintained under repeated simulated preferences |

**Selected experimental observations** (specific offline test configurations, not universal guarantees):

- In a geographic ablation for a Technology profile, a top-10 hybrid slate represented **10 countries**, compared with **6** without geographic reranking.
- In a 30-round synthetic-feedback experiment, the hybrid configuration maintained **10 countries and 9 topics** in the evaluated top-10 slates; the baseline averaged **6.13 countries and 4.87 topics**.
- The MMR lambda experiment showed the expected trade-off: greater emphasis on relevance can reduce semantic diversity.

These experiments demonstrate system behavior on a fixed dataset and simulated interactions. They **do not establish causal effects on reader satisfaction** or measure long-term outcomes with real users.

## Technology

**Python · Streamlit · pandas · NumPy · Sentence Transformers · BGE embeddings · SQLite · Maximal Marginal Relevance (MMR) · Git/GitHub**

## Repository guide

```text
beyond-the-feed/
├── Home.py                         # Application entry point
├── pages/                          # Feed, insights, and comparison pages
├── components/                     # Authentication and article UI components
├── recommender/                    # Baseline, hybrid retrieval, personalization, MMR
├── utils/                          # Data, semantic assets, accounts, and feedback
├── evaluation/                     # Offline experiments and ablation scripts
├── reports/                        # Evaluation results and visualizations
├── tests/                          # Automated tests
├── data/                           # Dataset and model assets (where available)
├── styles/                         # Application styling
├── requirements.txt
└── README.md
```

The current hybrid feed uses modules including `hybrid_pipeline.py`, `nlp_profile.py`, and `unified_reranker.py`. Other recommender modules preserve earlier components and experiments; they should not all be interpreted as separate active feed pipelines.

## Run locally

**Prerequisites:** Git and a compatible Python environment (the project has been developed with Python 3.11).

```bash
git clone https://github.com/data-bees-team/beyond-the-feed.git
cd beyond-the-feed
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
python -m pip install -r requirements.txt
streamlit run Home.py
```

Open **http://localhost:8501**. On Windows, activate the virtual environment using `.venv\Scripts\activate` instead.

**Data note:** Local execution requires the dataset and precomputed semantic embedding assets expected by the project's loaders. If those assets are not included in the repository, they must be provided separately. The live Streamlit deployment is the easiest way for judges to try the completed experience.

To run automated tests:

```bash
python -m unittest discover -s tests -v
```

## Scope and limitations

- Country and publisher diversity are **exposure indicators**, not direct measurements of viewpoint, fairness, or political balance.
- The system works with a fixed historical news corpus rather than live ingestion.
- A diverse top-10 slate does not guarantee every country, publisher, or topic will be represented.
- Simulated feedback experiments assess ranking responsiveness; real-user satisfaction would require a separate user study.

## Team

Built by **Data Bees** for the **Behind the Feed Data Challenge 2026**.

The project combines team contributions in geographic diversity, publisher diversity, semantic recommendation, user personalization, application integration, and evaluation.

---

**Our aim:** Make news recommendations personally useful **and** open the door to broader discovery.
