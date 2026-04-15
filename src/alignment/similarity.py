from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of embeddings.

    Args:
        emb1: Array shape (n1, d)
        emb2: Array shape (n2, d)

    Returns:
        Matrix shape (n1, n2) with cosine similarities in [-1, 1].
    """
    if emb1.size == 0 or emb2.size == 0:
        return np.zeros((emb1.shape[0], emb2.shape[0]))
    return cosine_similarity(emb1, emb2)


def get_aligned_pairs(chunks1: List[str], chunks2: List[str], sim_matrix: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int, float, str, str]]:
    """
    Return all chunk pairs with similarity >= threshold.

    Returns a list of tuples: (i, j, score, chunk1_text, chunk2_text)
    where i indexes `chunks1` and j indexes `chunks2`.
    """
    pairs = []
    n1, n2 = sim_matrix.shape
    for i in range(n1):
        for j in range(n2):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                pairs.append((i, j, score, chunks1[i], chunks2[j]))
    return pairs
