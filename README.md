# Beyond the Feed 🐝

A Diversity-Aware News Recommendation System built for the **Behind the Feed Data Challenge 2026**.

*Beyond the Feed* reimagines algorithmic news delivery. Rather than trapping readers inside narrow geographic and thematic echo chambers, it combines content-based topic relevance with **Geographic Diversity Re-Ranking (Maximal Marginal Relevance - MMR)** to broaden international exposure while honoring personal reading preferences.

---

## 📌 Project Overview

Empirical analysis of our global news corpus ($N = 15,395$ articles, 185 sovereign nations, 6 continents) revealed significant supply concentration:
- **54.3% of all articles** originate from just three countries: United States (27.5%), United Kingdom (16.1%), and India (10.7%).
- **Africa** represents only **5.34%** of available inventory; **South America** represents just **1.12%**.
- Conventional pure-relevance recommenders amplify this baseline skew into severe filter bubbles. For example, for a reader interested in *Health and Medicine*, the baseline recommendation slate draws **100% of its stories from North America and Europe**, with 0% representation from the Global South.

*Beyond the Feed* addresses this challenge through a multi-stage recommendation architecture that balances topical affinity with geographic dispersion across countries and continents.

---

## ⚙️ Recommendation Architecture

The recommendation pipeline operates in two sequential stages:

```
                     Cleaned GDELT Corpus (N = 15,395)
                                    ↓
                     User Selected Topic Preferences
                                    ↓
             [Stage 1: Baseline Candidate Relevance Scoring]
                                    ↓
                       Eligible Candidate Pool (M)
                     (relevance > 0, default M = 500)
                                    ↓
                 [Stage 2: Geographic MMR Re-Ranking]
                 λ · Rel(u, a) - (1 - λ) · Sim_geo(a, S)
                                    ↓
                   Final Balanced Slate (Top K = 10)
                                    ↓
                  Streamlit Personalized News Cards
```

### 1. Stage 1: Baseline Personalized Relevance
Users select their preferred topics (e.g., *Technology*, *Environment and Climate*, *Health*), with each topic receiving a weight between $0.0$ and $1.0$.

For each article $a$, its project topics $V_a$ (excluding non-actionable categories `"Unknown"` and `"Other"`) are evaluated against the user profile $u$:
$$\mathrm{Rel}(u, a) = \begin{cases}
0.0, & \text{if } |V_a| = 0 \\
\frac{1}{|V_a|} \sum_{t \in V_a} u(t), & \text{if } |V_a| > 0
\end{cases}$$

Candidate articles with $\mathrm{Rel}(u, a) > 0$ form the eligible candidate pool $A$, sorted deterministically by relevance descending and `article_id` ascending.

### 2. Stage 2: Geographic Diversity Re-Ranking (MMR)
Rather than naively truncating to the top 10 relevant items (which reproduces the US/UK/India supply monopoly), candidate articles are iteratively selected using **Maximal Marginal Relevance (MMR)**:

$$\mathrm{Score}(a, S) = \lambda \cdot \mathrm{Rel}(u, a) - (1 - \lambda) \cdot \mathrm{Sim}_{\mathrm{geo}}(a, S)$$

Where:
- $\mathrm{Rel}(u, a)$ is the baseline topic relevance score.
- $\mathrm{Sim}_{\mathrm{geo}}(a, S) = \max_{s \in S} \mathrm{sim}_{\mathrm{geo}}(a, s)$ quantifies geographic redundancy relative to articles already chosen in slate $S$.
- $\lambda \in [0, 1]$ is the trade-off parameter balancing topic relevance ($\lambda = 1.0$) and geographic novelty ($\lambda = 0.0$). Default $\lambda = 0.60$.
- $S$ is the slate of articles selected so far ($|S| < K$).

#### Hierarchical Geographic Redundancy Model $\mathrm{sim}_{\mathrm{geo}}(a, s)$
1. **Primary Country Redundancy ($w_{\mathrm{country}} = 1.0$):** High penalty when candidate $a$ shares its primary country with an article already in $S$.
2. **Continent Redundancy ($w_{\mathrm{continent}} = 0.4$):** Moderate penalty when $a$ introduces a different country from a continent already represented in $S$.
3. **Secondary Country Overlap ($w_{\mathrm{secondary}} = 0.1$):** Minor penalty when secondary locations in `all_countries` overlap with $S$.
4. **Missing/Unknown Geography Robustness ($w_{\mathrm{missing}} = 0.5$):** Neutral penalty ensuring un-geotagged articles receive neither an artificial novelty bonus ($0.0$) nor an unfair disqualification ($1.0$).
5. **Geographic Novelty ($0.0$):** Zero redundancy when $a$ introduces a continent and country not yet present in $S$.

---

## 📊 Empirical Evaluation Results

Evaluated across the project's three standardized benchmark personas ($\lambda = 0.6$, $K = 10$, $M = 500$):

| Benchmark Profile | System | Avg Relevance | Relevance Degradation | Unique Countries | Unique Continents | Country Concentration (HHI) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Tech Enthusiast** | Baseline | 0.800 | — | 6 / 10 | 3 / 6 | 0.2000 |
| | **Geographic MMR** | **0.780** | **2.50%** | **10 / 10 (+66.7%)** | **6 / 6 (+100.0%)** | **0.1000 (-50.0%)** |
| **Environment Explorer** | Baseline | 0.900 | — | 5 / 10 | 4 / 6 | 0.2800 |
| | **Geographic MMR** | **0.900** | **0.00%** | **10 / 10 (+100.0%)** | **6 / 6 (+50.0%)** | **0.1000 (-64.3%)** |
| **Health Researcher** | Baseline | 0.900 | — | 5 / 10 | 2 / 6 | 0.3200 |
| | **Geographic MMR** | **0.900** | **0.00%** | **10 / 10 (+100.0%)** | **6 / 6 (+200.0%)** | **0.1000 (-68.8%)** |

### Key Findings & Limitations
- **100% Distinct Country Representation:** Every recommended article in the MMR slate represents a distinct country (10 out of 10).
- **Global Continental Reach:** Expands representation from 2–4 continents up to all 6 inhabited continents (Africa, Asia, Europe, North America, Oceania, South America).
- **Preserved Topic Relevance:** Average relevance drops by at most $0.00$ to $0.02$ ($0.0\%$ to $2.5\%$ degradation).
- **Theoretical Minimum Concentration:** Country HHI drops to $0.1000$, the theoretical minimum for a slate of 10 items ($1/10$).
- **Limitations:** Publisher diversity and thematic interest bridging are managed by separate team modules and will be integrated into the final unified ranking stage.

---

## 📁 Repository Structure

```
beyond-the-feed/
├── Home.py                             # Streamlit application landing page & overview
├── pages/
│   └── 1_My_Feed.py                    # Personalized news discovery feed with MMR re-ranking
├── recommender/
│   ├── __init__.py                     # Package interface exporting baseline and MMR engines
│   ├── baseline.py                     # Baseline candidate scoring & relevance ranking
│   └── geographic_diversity.py         # Production geographic MMR engine & redundancy logic
├── components/
│   └── news_card.py                    # Reusable article card UI component
├── utils/
│   └── data_loader.py                  # Cached GDELT data loading utility
├── data/
│   └── gdelt_clean.csv                 # Cleaned GDELT news corpus (N = 15,395)
├── notebooks/
│   └── geographic_diversity.ipynb      # Executed research notebook with trade-off curves & audits
├── reports/
│   └── geographic_diversity/           # Empirical evaluation summaries & figures
│       ├── geographic_recommendations_summary.csv
│       ├── baseline_vs_mmr_evaluation.csv
│       ├── continent_representation_comparison.png
│       └── lambda_tradeoff_analysis.png
├── tests/
│   └── test_geographic_diversity.py    # Automated unit and regression test suite
├── styles/
│   └── style.css                       # Application CSS styles
├── requirements.txt                    # Project runtime dependencies
└── README.md                           # Project technical documentation
```

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12–3.14)
- Git

### 2. Clone Repository & Setup Environment
```bash
git clone https://github.com/data-bees-team/beyond-the-feed.git
cd beyond-the-feed

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m unittest tests/test_geographic_diversity.py
```

### 4. Launch Streamlit Web Application
```bash
streamlit run Home.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 👥 Team & Acknowledgments
Built by **Data Bees** for the *Behind the Feed Data Challenge 2026*.
