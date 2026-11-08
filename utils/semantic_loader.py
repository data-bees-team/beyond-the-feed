from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from recommender.topic_descriptions import TOPIC_DESCRIPTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_article_embeddings(_df_data):
    path = PROJECT_ROOT / "data" / "article_embeddings.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Article embeddings not found: {path}"
        )

    embeddings = np.load(path)

    if len(embeddings) != len(_df_data):
        raise ValueError(
            "Embedding count does not match article count."
        )

    embeddings_dict = {
        str(row["article_id"]): embeddings[idx]
        for idx, (_, row) in enumerate(_df_data.iterrows())
    }

    return embeddings, embeddings_dict


@st.cache_resource
def load_semantic_model():
    return SentenceTransformer(
        "BAAI/bge-base-en-v1.5",
        device="cpu",
    )


@st.cache_data
def get_topic_description_embeddings():
    model = load_semantic_model()

    descriptions = []
    for topic_descriptions in TOPIC_DESCRIPTIONS.values():
        descriptions.extend(topic_descriptions)

    encoded = model.encode(
        descriptions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    result = {}
    start = 0

    for topic, topic_descriptions in TOPIC_DESCRIPTIONS.items():
        count = len(topic_descriptions)
        result[topic] = encoded[start:start + count]
        start += count

    return result
