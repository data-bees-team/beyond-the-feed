import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data
def load_articles():
    # Locate the dataset
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "gdelt_clean.csv"

    # Load CSV
    df = pd.read_csv(data_path, keep_default_na=False)

    # Convert topic strings into lists
    df["topic_list"] = df["topic"].apply(
        lambda x: [
            topic.strip()
            for topic in x.split(";")
            if topic.strip()
        ]
    )

    return df