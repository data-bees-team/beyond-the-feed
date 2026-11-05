import numpy as np
import pandas as pd
from utils.db_manager import get_user_feedback

def get_or_create_user_vector(user_id: str, embeddings_dict: dict, embedding_dim: int = 768) -> np.ndarray:
    """
    Computes user profile vector based on stored Interested / Not Interested feedback.
    Returns zero vector if user has no feedback yet (cold start).
    """
    user_feedback = get_user_feedback(user_id)
    if not user_feedback:
        return np.zeros(embedding_dim)
    
    user_vec = np.zeros(embedding_dim)
    
    for entry in user_feedback:
        art_id = entry["article_id"]
        f_type = entry["feedback_type"]
        
        if art_id in embeddings_dict:
            art_vec = embeddings_dict[art_id]
            if f_type == "interested":
                user_vec += 0.3 * art_vec
            elif f_type == "not_interested":
                user_vec -= 0.2 * art_vec
                
    return user_vec

def compute_nlp_content_scores(user_vec: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between user content vector and candidate articles.
    """
    if np.all(user_vec == 0):
        return np.zeros(len(candidate_embeddings))
    
    # Cosine Similarity: (A . B) / (||A|| * ||B||)
    user_norm = np.linalg.norm(user_vec)
    cand_norms = np.linalg.norm(candidate_embeddings, axis=1)
    
    # Avoid zero division
    cand_norms[cand_norms == 0] = 1e-10
    
    dot_products = np.dot(candidate_embeddings, user_vec)
    similarity_scores = dot_products / (user_norm * cand_norms)
    
    return similarity_scores