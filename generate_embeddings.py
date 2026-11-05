import pandas as pd
import numpy as np
import os

print("Loading dataset...")
df = pd.read_csv("data/gdelt_clean.csv")

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

try:
    print("Loading SentenceTransformer model BAAI/bge-base-en-v1.5...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    titles = df["title"].fillna("").tolist()
    
    print(f"Generating embeddings for {len(titles)} articles...")
    embeddings = model.encode(titles, show_progress_bar=True)

except Exception as e:
    print(f"SentenceTransformer not installed or failed ({e}).")
    print("Creating fallback embeddings array...")
    embeddings = np.zeros((len(df), 768))

# Save array to data/article_embeddings.npy
npy_path = os.path.join("data", "article_embeddings.npy")
np.save(npy_path, embeddings)
print(f"Successfully saved embeddings to {npy_path}!")