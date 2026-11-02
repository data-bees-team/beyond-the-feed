"""
Centralized User Preference Learning for Beyond the Feed.

Learns multi-dimensional user preferences (topics, publishers, countries, continents)
from historical 👍 Interested and 👎 Not Interested feedback.

Mathematical Scoring Model:
    For any entity x (topic, publisher, country, continent):
    1. Recency Decay:
       w_i = exp(-lambda_decay * dt_i)
    2. Weighted Counts:
       C+(x) = sum_{i in Feedback+} w_i * credit(x, a_i)
       C-(x) = sum_{j in Feedback-} w_j * penalty(x, a_j)
       N(x)  = C+(x) + C-(x)
    3. Bayesian-Smoothed Polarity:
       Polarity(x) = (C+(x) - C-(x)) / (N(x) + k_smooth)   in [-1.0, 1.0]
    4. Interaction Confidence:
       Conf(x) = 1 - exp(-N(x) / tau)                       in [0.0, 1.0]
    5. Net Preference Score:
       Score(x) = Polarity(x) * Conf(x)                     in [-1.0, 1.0]
    6. Normalized Score:
       Normalized(x) = (1.0 + Score(x)) / 2.0               in [0.0, 1.0]
       (where 0.5 is neutral/cold-start, >0.5 preferred, <0.5 dispreferred)
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math


# Smoothing and decay hyper-parameters
DEFAULT_K_SMOOTH = 2.0        # Laplace-style shrinkage preventing overreaction
DEFAULT_TAU_CONFIDENCE = 3.0  # Number of interactions for ~63% confidence
DEFAULT_LAMBDA_DECAY = 0.05   # Exponential recency decay per day (half-life ~14 days)


@dataclass
class UserPreferenceProfile:
    """
    Structured representation of a user's learned preferences across all dimensions.
    """
    user_id: str
    total_feedback_count: int = 0
    interested_count: int = 0
    not_interested_count: int = 0

    # Entity scores in [-1.0, +1.0] (0.0 = neutral/no data)
    topic_scores: Dict[str, float] = field(default_factory=dict)
    publisher_scores: Dict[str, float] = field(default_factory=dict)
    country_scores: Dict[str, float] = field(default_factory=dict)
    continent_scores: Dict[str, float] = field(default_factory=dict)

    # Normalized scores in [0.0, 1.0] (0.5 = neutral/no data)
    topic_normalized: Dict[str, float] = field(default_factory=dict)
    publisher_normalized: Dict[str, float] = field(default_factory=dict)
    country_normalized: Dict[str, float] = field(default_factory=dict)
    continent_normalized: Dict[str, float] = field(default_factory=dict)

    # Interaction confidences in [0.0, 1.0]
    topic_confidence: Dict[str, float] = field(default_factory=dict)
    publisher_confidence: Dict[str, float] = field(default_factory=dict)
    country_confidence: Dict[str, float] = field(default_factory=dict)
    continent_confidence: Dict[str, float] = field(default_factory=dict)

    # Raw evidence counts (pos, neg)
    topic_counts: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    publisher_counts: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    country_counts: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    continent_counts: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_topic_preference(self, topic: str) -> float:
        """Return normalized preference in [0, 1] for topic (0.5 if neutral/unrated)."""
        return self.topic_normalized.get(topic, 0.5)

    def get_publisher_preference(self, publisher: str) -> float:
        """Return normalized preference in [0, 1] for publisher (0.5 if neutral/unrated)."""
        return self.publisher_normalized.get(publisher, 0.5)

    def get_country_preference(self, country: str) -> float:
        """Return normalized preference in [0, 1] for country (0.5 if neutral/unrated)."""
        return self.country_normalized.get(country, 0.5)

    def get_continent_preference(self, continent: str) -> float:
        """Return normalized preference in [0, 1] for continent (0.5 if neutral/unrated)."""
        return self.continent_normalized.get(continent, 0.5)

    def get_geographic_preference(self, country: Optional[str], continent: Optional[str]) -> float:
        """
        Compute blended geographic preference score.

        Hierarchy:
            - If country has direct feedback: 0.7 * country + 0.3 * continent
            - If country has no direct feedback but continent has feedback: continent score
            - If neither has feedback: 0.5 (neutral)
        """
        c_score = self.country_normalized.get(country) if country else None
        cont_score = self.continent_normalized.get(continent) if continent else None

        if c_score is not None and cont_score is not None:
            return 0.7 * c_score + 0.3 * cont_score
        elif c_score is not None:
            return c_score
        elif cont_score is not None:
            return cont_score
        else:
            return 0.5

    def get_top_preferred_topics(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return the user's top positively preferred topics."""
        pos = [(t, s) for t, s in self.topic_scores.items() if s > 0.05]
        return sorted(pos, key=lambda x: x[1], reverse=True)[:top_n]

    def get_dispreferred_topics(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return topics with consistent negative feedback."""
        neg = [(t, s) for t, s in self.topic_scores.items() if s < -0.05]
        return sorted(neg, key=lambda x: x[1])[:top_n]

    def get_preferred_publishers(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return the user's top positively preferred publishers."""
        pos = [(p, s) for p, s in self.publisher_scores.items() if s > 0.05]
        return sorted(pos, key=lambda x: x[1], reverse=True)[:top_n]

    def get_preferred_countries(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return the user's top positively preferred countries."""
        pos = [(c, s) for c, s in self.country_scores.items() if s > 0.05]
        return sorted(pos, key=lambda x: x[1], reverse=True)[:top_n]


def _compute_entity_scores(
    counts: Dict[str, Tuple[float, float]],
    k_smooth: float = DEFAULT_K_SMOOTH,
    tau: float = DEFAULT_TAU_CONFIDENCE
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Helper function to compute polarity scores, normalized scores, and confidences.
    """
    scores = {}
    normalized = {}
    confidence = {}

    for entity, (c_pos, c_neg) in counts.items():
        n = c_pos + c_neg
        if n <= 0.0:
            scores[entity] = 0.0
            normalized[entity] = 0.5
            confidence[entity] = 0.0
            continue

        polarity = (c_pos - c_neg) / (n + k_smooth)
        conf = 1.0 - math.exp(-n / tau)
        score = polarity * conf
        norm_score = max(0.0, min(1.0, (1.0 + score) / 2.0))

        scores[entity] = score
        normalized[entity] = norm_score
        confidence[entity] = conf

    return scores, normalized, confidence


def build_user_preference_profile(
    feedback_records: List[Dict[str, Any]],
    user_id: str,
    k_smooth: float = DEFAULT_K_SMOOTH,
    tau: float = DEFAULT_TAU_CONFIDENCE,
    lambda_decay: float = DEFAULT_LAMBDA_DECAY,
    now_dt: Optional[datetime] = None
) -> UserPreferenceProfile:
    """
    Construct a complete UserPreferenceProfile from raw feedback records.

    Args:
        feedback_records: List of feedback dicts containing:
            'feedback_type', 'topics', 'publisher', 'country', 'continent', 'updated_at'
        user_id: Identifier of the user.
        k_smooth: Smoothing pseudo-count (default 2.0).
        tau: Confidence scaling threshold (default 3.0).
        lambda_decay: Exponential decay per day (default 0.05).
        now_dt: Reference datetime for decay (defaults to current UTC).

    Returns:
        UserPreferenceProfile instance with learned preference values across all dimensions.
    """
    if not feedback_records:
        return UserPreferenceProfile(user_id=user_id)

    if now_dt is None:
        now_dt = datetime.now(timezone.utc)

    topic_counts: Dict[str, List[float]] = {}      # [pos, neg]
    publisher_counts: Dict[str, List[float]] = {}  # [pos, neg]
    country_counts: Dict[str, List[float]] = {}    # [pos, neg]
    continent_counts: Dict[str, List[float]] = {}  # [pos, neg]

    interested_count = 0
    not_interested_count = 0

    for rec in feedback_records:
        fb_type = rec.get("feedback_type")
        if fb_type not in ("interested", "not_interested"):
            continue

        is_pos = (fb_type == "interested")
        if is_pos:
            interested_count += 1
        else:
            not_interested_count += 1

        # Calculate recency weight
        updated_at_str = rec.get("updated_at") or rec.get("created_at")
        weight = 1.0
        if updated_at_str:
            try:
                rec_dt = datetime.fromisoformat(updated_at_str)
                if rec_dt.tzinfo is None:
                    rec_dt = rec_dt.replace(tzinfo=timezone.utc)
                delta_days = max(0.0, (now_dt - rec_dt).total_seconds() / 86400.0)
                weight = math.exp(-lambda_decay * delta_days)
            except (ValueError, TypeError):
                weight = 1.0

        pos_wt = weight if is_pos else 0.0
        neg_wt = 0.0 if is_pos else weight

        # A. Topics credit (distribute across multiple valid topics)
        topics = rec.get("topics") or []
        valid_topics = [
            t for t in topics
            if t and t not in {"Unknown", "Other", "none", "null"}
        ]
        if valid_topics:
            t_credit = 1.0 / len(valid_topics)
            for t in valid_topics:
                if t not in topic_counts:
                    topic_counts[t] = [0.0, 0.0]
                topic_counts[t][0] += pos_wt * t_credit
                topic_counts[t][1] += neg_wt * t_credit

        # B. Publisher credit
        pub = rec.get("publisher", "").strip()
        if pub and pub.lower() not in {"unknown", "unknown publisher", "none", ""}:
            if pub not in publisher_counts:
                publisher_counts[pub] = [0.0, 0.0]
            publisher_counts[pub][0] += pos_wt
            publisher_counts[pub][1] += neg_wt

        # C. Country credit (safe check for valid country, preserving Namibia 'NA')
        country = rec.get("country", "").strip()
        if country and country.lower() not in {"unknown", "none", "null", "n/a", "missing", ""}:
            if country not in country_counts:
                country_counts[country] = [0.0, 0.0]
            country_counts[country][0] += pos_wt
            country_counts[country][1] += neg_wt

        # D. Continent credit
        continent = rec.get("continent", "").strip()
        if continent and continent.lower() not in {"unknown", "none", "null", "undetermined", ""}:
            if continent not in continent_counts:
                continent_counts[continent] = [0.0, 0.0]
            continent_counts[continent][0] += pos_wt
            continent_counts[continent][1] += neg_wt

    # Convert to immutable tuple counts
    tc = {k: (v[0], v[1]) for k, v in topic_counts.items()}
    pc = {k: (v[0], v[1]) for k, v in publisher_counts.items()}
    cc = {k: (v[0], v[1]) for k, v in country_counts.items()}
    con_c = {k: (v[0], v[1]) for k, v in continent_counts.items()}

    # Compute scores
    t_scores, t_norm, t_conf = _compute_entity_scores(tc, k_smooth, tau)
    p_scores, p_norm, p_conf = _compute_entity_scores(pc, k_smooth, tau)
    c_scores, c_norm, c_conf = _compute_entity_scores(cc, k_smooth, tau)
    con_scores, con_norm, con_conf = _compute_entity_scores(con_c, k_smooth, tau)

    return UserPreferenceProfile(
        user_id=user_id,
        total_feedback_count=len(feedback_records),
        interested_count=interested_count,
        not_interested_count=not_interested_count,
        topic_scores=t_scores,
        publisher_scores=p_scores,
        country_scores=c_scores,
        continent_scores=con_scores,
        topic_normalized=t_norm,
        publisher_normalized=p_norm,
        country_normalized=c_norm,
        continent_normalized=con_norm,
        topic_confidence=t_conf,
        publisher_confidence=p_conf,
        country_confidence=c_conf,
        continent_confidence=con_conf,
        topic_counts=tc,
        publisher_counts=pc,
        country_counts=cc,
        continent_counts=con_c,
        last_updated=datetime.now(timezone.utc).isoformat()
    )

