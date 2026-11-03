# Algorithms Guide: Beyond the Feed 🐝

This guide explains each recommendation algorithm and personalization component in **Beyond the Feed**, what it does, and why it is essential to the system.

---

## 1. Baseline Topic Recommender
* **File:** [`recommender/baseline.py`](file:///Users/shahidabatool/Desktop/Hackathon_global_news/beyond-the-feed/recommender/baseline.py)
* **What it does:**  
  Matches the topics a user selects during onboarding (e.g., *Technology*, *Health*) against each article's topic tags to compute a normalized relevance score:
  $$\mathrm{Rel}_{\mathrm{base}}(u, a) = \frac{1}{|V_a|} \sum_{t \in V_a} u(t)$$
* **Why we need it:**  
  It serves as the foundational **intent anchor**. Every diversity and discovery algorithm needs an objective measure of raw user relevance to ensure recommendations stay meaningful.

---

## 2. Geographic Diversity Re-Ranking (MMR) — *Shahida's Core Task*
* **File:** [`recommender/geographic_diversity.py`](file:///Users/shahidabatool/Desktop/Hackathon_global_news/beyond-the-feed/recommender/geographic_diversity.py)
* **What it does:**  
  Iteratively selects articles using **Maximal Marginal Relevance (MMR)** by penalizing geographic redundancy relative to articles already chosen:
  $$\mathrm{Score}(a, S) = \lambda \cdot \mathrm{Rel}(u, a) - (1 - \lambda) \cdot \mathrm{Sim}_{\mathrm{geo}}(a, S)$$
  * Primary Country repeat penalty: $1.0$
  * Continent repeat penalty: $0.4$
  * Secondary mentioned country overlap: $0.1$
  * Un-geotagged neutral penalty: $0.5$
  * Distinct continent & country: $0.0$ (novel global perspective)
* **Why we need it:**  
  **54.3% of all global news in GDELT comes from just three countries (US, UK, India)**, while Africa has only 5.3% and South America just 1.1%. Without this algorithm, pure relevance recommenders trap users in a Western/Anglo-centric echo chamber. Geographic MMR increases country representation by **+88.7%** (10 distinct countries per 10 articles) while retaining high relevance.

---

## 2B. Publisher Diversity Re-Ranking (MMR) — *Suraiya's Core Task*
* **File:** [`recommender/publisher_diversity.py`](file:///C:/Users/Surayia%20Rahman/.gemini/antigravity/scratch/beyond-the-feed/recommender/publisher_diversity.py)
* **What it does:**  
  Iteratively selects articles using **Publisher-Aware Maximal Marginal Relevance (MMR)** by balancing topic relevance against publisher concentration:
  $$\mathrm{Score}(a, S) = \lambda \cdot \mathrm{Rel}(u, a) - (1 - \lambda) \cdot \mathrm{Redundancy}_{\mathrm{pub}}(a, S)$$
  * Novel publisher ($count = 0$ in slate $S$): $\mathrm{Redundancy} = 0.0$ (novel voice encouraged)
  * Repeated publisher ($count \ge 1$ in slate $S$): $\mathrm{Redundancy} = 1.0$ (redundancy penalty)
  * Missing/Unknown publisher: $\mathrm{Redundancy} = 0.5$ (neutral penalty; prevents treating missing data as a diversity feature)
  * Tunable parameter $\lambda \in [0, 1]$ (default $\lambda = 0.6$)
* **Why we need it:**  
  High-volume news organizations (such as `iheart.com` and `indiatimes.com`) frequently dominate greedy relevance rankings. For example, in baseline recommendations for the *Health Researcher* profile, multiple articles repeat the same publisher (`iheart.com`, `irishexaminer.com`), reducing distinct publisher exposure to 8/10. Publisher MMR eliminates monopolization by dominant media houses, maximizes distinct viewpoints, and lowers the Herfindahl-Hirschman Index (HHI) while preserving high topic relevance.

---

## 3. Centralized User Feedback System
* **File:** [`utils/feedback_manager.py`](file:///Users/shahidabatool/Desktop/Hackathon_global_news/beyond-the-feed/utils/feedback_manager.py)
* **What it does:**  
  A thread-safe, multi-user SQLite database (`data/feedback.db`) that records **👍 Interested** and **👎 Dislike** clicks per article and user, with automatic timestamps and atomic upserts.
* **Why we need it:**  
  Recommendation systems cannot improve without listening to user reactions. This provides the persistent data foundation so preferences are remembered across page reloads and sessions.

---

## 4. Adaptive User Preference Learning
* **File:** [`recommender/user_preferences.py`](file:///Users/shahidabatool/Desktop/Hackathon_global_news/beyond-the-feed/recommender/user_preferences.py)
* **What it does:**  
  Translates raw 👍 / 👎 clicks into a multi-dimensional user profile across topics, publishers, and countries:
  * **Recency decay:** Recent interactions carry more weight than older ones ($w_i = e^{-\lambda \Delta t}$).
  * **Laplace smoothing ($k = 2.0$):** Prevents a single negative rating from eliminating an entire category.
  * **Interaction confidence ($\tau = 3.0$):** Scales preference strength proportionally with the number of interactions.
* **Why we need it:**  
  Users' reading interests evolve over time. This module learns nuanced affinities (e.g., liking clean energy from Africa or disliking tabloid news) while ensuring cold-start users (0 feedback) receive completely unbiased baseline recommendations.

---

## 5. Centralized Personalization Engine
* **File:** [`recommender/personalization_engine.py`](file:///Users/shahidabatool/Desktop/Hackathon_global_news/beyond-the-feed/recommender/personalization_engine.py)
* **What it does:**  
  Combines learned topic (60%), publisher (25%), and geographic (15%) affinities to adjust candidate relevance:
  $$\mathrm{Rel}_{\mathrm{pers}}(a) = \mathrm{clip}\left(\mathrm{Rel}_{\mathrm{base}}(a) \cdot \left[1 + \alpha \cdot (2 \cdot \mathrm{UserPref}(a) - 1)\right], 0.0, 1.0\right)$$
* **Why we need it:**  
  It acts as a shared bridge between feedback and recommendation algorithms. It ensures learned preferences influence candidate selection without ever overwriting the original baseline score.

---

## 6. Thematic Interest Bridging
* **File:** [`recommender/interest_bridging.py`](file:///Users/shahidabatool/Desktop/Hackathon_global_news/beyond-the-feed/recommender/interest_bridging.py)
* **What it does:**  
  Exploits the fact that **78.4% of GDELT articles have multiple topic tags** to discover "bridge" stories connecting a user's core interests with adjacent, unselected themes (e.g., *Technology* $\to$ *AI Diagnostics in Health*). Filters out any novel topics the user explicitly disliked.
* **Why we need it:**  
  It prevents topical echo chambers. Rather than showing 10 articles on the exact same subject, it gently introduces related topics that naturally expand the reader's curiosity.

---

## 7. Unified Feed Composer (70% Personalized / 30% Discovery)
* **File:** [`recommender/feed_composer.py`](file:///Users/shahidabatool/Desktop/Hackathon_global_news/beyond-the-feed/recommender/feed_composer.py)
* **What it does:**  
  Orchestrates the final feed by combining:
  * **70% Personalized Core (7 articles):** Selected via Geographic Diversity MMR using learned preferences.
  * **30% Serendipitous Discovery (3 articles):** Selected via Interest Bridging to introduce unfamiliar global perspectives.
  * Clearly labels each card with `✦ CURATED FOR YOU` or `✦ DISCOVERY PERSPECTIVE`.
* **Why we need it:**  
  A healthy news diet requires both **exploitation** (giving users what they love) and **exploration** (introducing new perspectives). The 70/30 split provides the optimal balance to prevent boredom, complacency, and polarization.

---

## Summary Matrix

| Algorithm | Primary Purpose | How It Prevents Echo Chambers |
| :--- | :--- | :--- |
| **Baseline Recommender** | Content-based topic matching | Establishes raw relevance anchor |
| **Geographic Diversity (MMR)** | Multi-continental representation | Eliminates the 54% US/UK/India news monopoly |
| **Publisher Diversity (MMR)** | News source pluralism | Prevents dominant media outlets (e.g. iheart, indiatimes) from monopolizing recommendations |
| **User Feedback Manager** | Persistent SQLite storage | Remembers 👍/👎 clicks across sessions |
| **User Preference Learning** | Mathematical affinity modeling | Learns nuanced tastes with smoothing protection |
| **Personalization Engine** | Blended candidate scoring | Tunes relevance without erasing baseline scores |
| **Interest Bridging** | Multi-topic serendipity | Connects core interests to adjacent new themes |
| **Feed Composer** | 70% Core / 30% Discovery split | Balances exploitation with guaranteed exploration |

