# Beyond the Feed 🐝

A Diversity-Aware News Recommendation System with Centralized User Feedback & Adaptive Personalization built for the **Behind the Feed Data Challenge 2026**.

*Beyond the Feed* reimagines algorithmic news delivery. Rather than trapping readers inside narrow geographic, publisher, and thematic echo chambers, it combines content-based topic relevance with **Geographic Diversity Re-Ranking (Maximal Marginal Relevance - MMR)**, **Publisher Diversification**, **Thematic Interest Bridging**, and **System-Wide Adaptive User Feedback Learning** to broaden international exposure while honoring personal reading preferences.

---

## 📌 Project Overview

Empirical analysis of our global news corpus ($N = 15,395$ articles, 185 sovereign nations, 6 continents) revealed significant supply concentration:
- **54.3% of all articles** originate from just three countries: United States (27.5%), United Kingdom (16.1%), and India (10.7%).
- **Africa** represents only **5.34%** of available inventory; **South America** represents just **1.12%**.
- Conventional pure-relevance recommenders amplify this baseline skew into severe filter bubbles. For example, for a reader interested in *Health and Medicine*, the baseline recommendation slate draws **100% of its stories from North America and Europe**, with 0% representation from the Global South.

*Beyond the Feed* addresses this challenge through a centralized, multi-stage recommendation architecture that balances topical affinity, source diversification, geographic coverage, and persistent user interaction learning.

---

## ⚙️ System Architecture

The recommendation and feedback pipeline operates as a unified, closed-loop architecture:

```
                            User Feedbacks (👍 / 👎)
                                       ↓
                [Centralized Persistent Storage (SQLite WAL)]
                         (utils/feedback_manager.py)
                                       ↓
                [Centralized User Preference Learning Engine]
                         (recommender/user_preferences.py)
                                       ↓
                [Centralized Shared Personalization Layer]
                     (recommender/personalization_engine.py)
                                       ↓
        ┌──────────────────────────────┼──────────────────────────────┐
        ↓                              ↓                              ↓
[Baseline Relevance]         [Geographic MMR]             [Publisher Diversity]
 (recommender/baseline.py)   (geographic_diversity.py)    (publisher_diversity.py)
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ↓
                         [Thematic Interest Bridging]
                        (recommender/interest_bridging.py)
                                       ↓
                       [Unified Feed Composition Engine]
                           (recommender/feed_composer.py)
                         (70% Personalized / 30% Discovery)
                                       ↓
                         Delivered Slate (Top K = 10)
                                       ↓
                       Streamlit Interactive News Cards
                         (components/news_card.py)
```

---

## 🗄️ Database Architecture (SQLite)

Persistent feedback storage is centralized in `data/feedback.db` (auto-excluded from Git):

### 1. `user_feedback` Table
Stores current user feedback per article with full metadata for fast profile derivation:
- `feedback_id`: Integer primary key (auto-increment).
- `user_id`: Unique user identifier (supports multi-user isolation).
- `article_id`: Canonical GDELT article identifier.
- `feedback_type`: `'interested'` (👍) or `'not_interested'` (👎).
- `title`, `publisher`, `country`, `continent`, `topics`: Denormalized article metadata.
- `created_at`, `updated_at`: ISO-8601 UTC timestamps.
- **Constraints**: `UNIQUE(user_id, article_id)` ensures atomic updates and zero duplicates.

### 2. `recommendation_impressions` Table
Prepares the platform for future **Contextual Bandits / Reinforcement Learning**:
- `impression_id`: Primary key.
- `user_id`, `article_id`, `algorithm`: Context and arm identifiers.
- `rank_position`: 1-based display rank in the slate.
- `relevance_score`: Propensity / expected reward.
- `timestamp`: UTC impression time.

---

## 📐 Mathematical Learning & Personalization Model

### 1. User Preference Learning (`recommender/user_preferences.py`)
For any entity $x$ (topic, publisher, country, continent):
1. **Recency Decay**: Interactions decay exponentially over time:
   $$w_i = \exp(-\lambda_{\mathrm{decay}} \cdot \Delta t_i)$$
   where $\lambda_{\mathrm{decay}} = 0.05$ (half-life $\approx 14$ days).
2. **Weighted Interaction Evidence**:
   $$C^+(x) = \sum_{i \in \text{Feedback}^+} w_i \cdot \text{credit}(x, a_i), \quad C^-(x) = \sum_{j \in \text{Feedback}^-} w_j \cdot \text{penalty}(x, a_j)$$
   Multi-topic articles distribute credit equally: $\text{credit} = \frac{1}{|\text{valid\_topics}|}$.
3. **Bayesian-Smoothed Polarity**:
   $$\mathrm{Polarity}(x) = \frac{C^+(x) - C^-(x)}{C^+(x) + C^-(x) + k_{\mathrm{smooth}}} \in [-1.0, +1.0]$$
   where $k_{\mathrm{smooth}} = 2.0$ prevents single negative ratings from eliminating a category.
4. **Interaction Confidence**:
   $$\mathrm{Conf}(x) = 1.0 - \exp\left(-\frac{C^+(x) + C^-(x)}{\tau}\right) \in [0.0, 1.0] \quad (\tau = 3.0)$$
5. **Net Normalized Preference**:
   $$\mathrm{Score}(x) = \mathrm{Polarity}(x) \cdot \mathrm{Conf}(x) \in [-1.0, +1.0]$$
   $$\mathrm{Normalized}(x) = \frac{1.0 + \mathrm{Score}(x)}{2.0} \in [0.0, 1.0]$$
   - $\mathrm{Normalized} = 0.5$: Neutral / cold start (zero bias).
   - $\mathrm{Normalized} > 0.5$: Positive affinity.
   - $\mathrm{Normalized} < 0.5$: Negative feedback dampening.

### 2. Multi-Dimensional Personalization Engine (`recommender/personalization_engine.py`)
For candidate article $a$:
$$\mathrm{UserPref}(a) = w_{\mathrm{topic}} \cdot \mathrm{TopicPref}(a) + w_{\mathrm{pub}} \cdot \mathrm{PubPref}(a) + w_{\mathrm{geo}} \cdot \mathrm{GeoPref}(a)$$
Defaults: $w_{\mathrm{topic}} = 0.60, w_{\mathrm{pub}} = 0.25, w_{\mathrm{geo}} = 0.15$.

$$\Delta_{\mathrm{pref}}(a) = 2.0 \cdot \mathrm{UserPref}(a) - 1.0 \in [-1.0, +1.0]$$
$$\mathrm{Rel}_{\mathrm{pers}}(a) = \mathrm{clip}\left(\mathrm{Rel}_{\mathrm{base}}(a) \cdot \left[1.0 + \alpha \cdot \Delta_{\mathrm{pref}}(a)\right], 0.0, 1.0\right)$$
- If user has no feedback ($\mathrm{UserPref} = 0.5$), $\Delta_{\mathrm{pref}} = 0 \implies \mathrm{Rel}_{\mathrm{pers}} \equiv \mathrm{Rel}_{\mathrm{base}}$.
- Original baseline relevance is **strictly preserved** in `baseline_relevance`.

---

## 🎯 Supported Recommendation Algorithms

All recommendation engines consume the shared `UserPreferenceProfile`:

1. **Baseline (`recommender/baseline.py`)**:
   Pure topic relevance, optionally boosted by learned preferences with deterministic tie-breaking.
2. **Geographic Diversity MMR (`recommender/geographic_diversity.py`)**:
   Balances personalized relevance with continental and national diversity:
   $$\mathrm{Score}(a, S) = \lambda \cdot \mathrm{Rel}_{\mathrm{pers}}(u, a) - (1 - \lambda) \cdot \mathrm{Sim}_{\mathrm{geo}}(a, S)$$
3. **Publisher Diversity MMR (`recommender/publisher_diversity.py`)**:
   Mitigates source concentration by penalizing repeated publishers and enforcing hard source caps (max 2 per slate).
4. **Thematic Interest Bridging (`recommender/interest_bridging.py`)**:
   Discovers adjacent perspectives by leveraging GDELT's 78.4% multi-topic tagging while honoring negative feedback filters.
5. **Unified Feed Composer (`recommender/feed_composer.py`)**:
   Orchestrates 70% Personalized core items + 30% Serendipitous Discovery items.

---

## 📊 Empirical Evaluation Benchmark

Benchmarked on the full cleaned GDELT dataset ($N = 15,395$) across three realistic user personas:

| Configuration | Baseline Rel | Pers. Rel | Pref. Alignment | Countries ($K=10$) | Continents | Publishers | Topics | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Original Baseline** | 1.000 | 1.000 | 0.525 | 5.3 | 3.3 | 8.7 | 2.0 | 10.0 ms |
| **2. Geographic MMR ($\lambda=0.6$)** | 1.000 | 1.000 | 0.527 | **10.0** | **6.0** | 9.7 | 2.0 | 26.1 ms |
| **3. Publisher Diversity ($\lambda=0.65$)** | 1.000 | 1.000 | 0.526 | 6.0 | 3.7 | **10.0** | 2.0 | 10.3 ms |
| **4. Feedback-Enhanced Baseline** | 1.000 | 1.000 | 0.532 | 5.7 | 3.7 | 8.7 | 2.0 | 77.8 ms |
| **5. Feedback-Enhanced Geo MMR** | 1.000 | 1.000 | 0.528 | **10.0** | **6.0** | 9.7 | 2.0 | 96.1 ms |
| **6. Feedback-Enhanced Pub Diversity**| 1.000 | 1.000 | 0.533 | 6.7 | 4.0 | **10.0** | 2.0 | 81.0 ms |
| **7. Combined Personalized + Discovery** | 0.900 | 0.908 | **0.537** | 7.7 | **6.0** | 9.7 | **3.7** | 248.7 ms |

*Simulated offline benchmark on modeled feedback interaction sessions.*

---

## 🛡️ Anti-Filter-Bubble & Exploration Guarantees

1. **No Overreaction:** Laplace smoothing ($k_{\mathrm{smooth}} = 2.0$) ensures a single negative feedback dampens an entity by at most 9–14%, never eliminating an entire topic or publisher.
2. **Cold Start Neutrality:** Unrated entities default to neutral ($0.5$), preventing unrated countries or topics from being penalized.
3. **Discovery Guarantee:** In the Balanced Feed, 25–30% of the slate is explicitly reserved for adjacent themes and long-tail global sources.

---

## 📁 Repository Structure

```
beyond-the-feed/
├── Home.py                             # Streamlit application landing page
├── pages/
│   └── 1_My_Feed.py                    # Interactive feed, feedback controls & dashboard
├── components/
│   └── news_card.py                    # Article card with 👍 / 👎 buttons & status badges
├── utils/
│   ├── data_loader.py                  # Cached GDELT loader with topic parser
│   └── feedback_manager.py             # SQLite thread-safe feedback & impression store
├── recommender/
│   ├── __init__.py                     # Package interface
│   ├── baseline.py                     # Baseline candidate scoring & recommendation
│   ├── geographic_diversity.py         # Geographic MMR re-ranking engine
│   ├── publisher_diversity.py          # Publisher diversity MMR & source capping
│   ├── interest_bridging.py            # Multi-topic thematic bridging engine
│   ├── feed_composer.py                # 70/30 personalized + discovery feed composer
│   ├── user_preferences.py             # Preference profile learning from feedback
│   └── personalization_engine.py       # Centralized adaptive relevance scoring
├── tests/
│   ├── test_geographic_diversity.py    # Geographic MMR unit & regression tests
│   └── test_feedback_personalization.py # Comprehensive feedback & personalization test suite
├── evaluation/
│   └── evaluate_feedback_system.py     # Comparative 7-configuration benchmark script
├── reports/
│   ├── geographic_diversity/           # Empirical MMR charts & reports
│   └── feedback_personalization/       # CSV benchmark outputs
├── styles/
│   └── style.css                       # Application CSS styles
├── requirements.txt                    # Runtime dependencies
└── README.md                           # Technical documentation
```

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12–3.14)
- Git

### 2. Setup Environment
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
# Run all 35 tests (feedback, preferences, MMR, publisher diversity, feed composer)
python -m unittest discover -s tests -v
```

### 4. Launch Streamlit Web Application
```bash
streamlit run Home.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 👥 Team & Acknowledgments
Built by **Data Bees** for the *Behind the Feed Data Challenge 2026*.
