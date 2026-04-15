from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer


def encode_chunks(chunks: List[str], model_name: str) -> np.ndarray:
    """
    Encode a list of text chunks into dense vector embeddings using a
    multilingual sentence-transformer.

    Args:
        chunks: List of chunk strings.
        model_name: HF model name for the sentence-transformers model.

    Returns:
        A 2D numpy array of shape (len(chunks), dim).

    Note: `model_name` must be provided by the caller (e.g., read from
    `configs/config.yaml`) to avoid hardcoding models inside this module.
    """
    if not chunks:
        return np.zeros((0, 0))

    if not model_name:
        raise ValueError("model_name must be provided to encode_chunks")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return embeddings
