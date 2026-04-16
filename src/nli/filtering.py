from typing import List, Tuple, Dict, Optional
from .nli_model import predict_nli, predict_nli_batch


def filter_contradictions(aligned_pairs: List[Tuple[int, int, float, str, str]], threshold: float = 0.7, nli_model_name: str = None) -> Optional[Dict]:
    """
    From a list of aligned chunk tuples, keep only those where the NLI label
    is `contradiction` and the model confidence >= `threshold`.

    Args:
        aligned_pairs: List of tuples `(i, j, score, chunk1, chunk2)`.
        threshold: Minimum confidence to accept a contradiction.

    Returns:
        The highest-confidence contradiction pair as a dict with keys:
        `i, j, sim_score, nli_confidence, chunk1, chunk2`, or `None` if none found.
    """
    best = None
    if not nli_model_name:
        raise ValueError("nli_model_name must be provided to filter_contradictions")

    if not aligned_pairs:
        return None

    # Batch NLI predictions for all aligned pairs to avoid repeated model calls
    premises = [pair[3] for pair in aligned_pairs]
    hypotheses = [pair[4] for pair in aligned_pairs]
    results = predict_nli_batch(premises, hypotheses, nli_model_name)

    for (pair, (label, conf)) in zip(aligned_pairs, results):
        i, j, sim_score, chunk1, chunk2 = pair
        if label == "contradiction" and conf >= threshold:
            candidate = {
                "i": i,
                "j": j,
                "sim_score": sim_score,
                "nli_confidence": conf,
                "chunk1": chunk1,
                "chunk2": chunk2,
            }
            if best is None or conf > best["nli_confidence"]:
                best = candidate
    return best
