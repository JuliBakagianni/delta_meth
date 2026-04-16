from typing import List
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel


def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool the token embeddings taking attention mask into account."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask


def encode_chunks(chunks: List[str], model_name: str, max_length: int = 256, device: str = None) -> np.ndarray:
    """
    Encode a list of text chunks into dense vector embeddings using a
    HuggingFace transformer model with explicit padding/truncation.

    This avoids tokenizer/ batching issues by forcing `padding=True`
    and `truncation=True` when tokenizing, then applying mean-pooling
    over the last hidden states.

    Args:
        chunks: List of chunk strings.
        model_name: HF model name (e.g. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
        max_length: Tokenizer max length for truncation.
        device: Torch device string, e.g. 'cuda' or 'cpu'. If None, will pick CUDA if available.

    Returns:
        A 2D numpy array of shape (len(chunks), dim).
    """
    if not chunks:
        return np.zeros((0, 0))

    if not model_name:
        raise ValueError("model_name must be provided to encode_chunks")

    # Ensure all inputs are strings and filter empty
    proc_chunks = [str(c) for c in chunks if isinstance(c, str) and c.strip()]
    if not proc_chunks:
        return np.zeros((0, 0))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load cached tokenizer/model (loads once per model_name)
    loaded = _load_embedding_model(model_name, device=device)
    tokenizer = loaded["tokenizer"]
    model = loaded["model"]

    # Tokenize with padding and truncation to ensure batched tensors have same length
    encoded = tokenizer(proc_chunks, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Use last_hidden_state for mean pooling
        token_embeddings = outputs.last_hidden_state
        sentence_embeddings = _mean_pooling(token_embeddings, attention_mask)

    return sentence_embeddings.cpu().numpy()


# Simple module-level cache to avoid reloading embedding models/tokenizers repeatedly
_embed_models = {}


def _load_embedding_model(model_name: str, device: str = None):
    """Load and cache the tokenizer and model for embeddings.

    Returns a dict with keys: `tokenizer`, `model`, `device`.
    """
    if not model_name:
        raise ValueError("model_name must be provided to load embedding model")
    if model_name in _embed_models:
        return _embed_models[model_name]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    _embed_models[model_name] = {"tokenizer": tokenizer, "model": model, "device": device}
    return _embed_models[model_name]

