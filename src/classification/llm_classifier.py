from typing import List, Dict


def classify_shift(filtered_pairs: List[Dict], llm_model: str) -> List[Dict]:
    """
    Classify the type of shift for each aligned chunk pair and provide a short explanation.

    Args:
        filtered_pairs: List of pairs that passed NLI filtering.
        llm_model: Placeholder LLM model identifier (not used in dummy implementation).

    Returns:
        A list of results: {"id": str, "shift_type": str, "explanation": str, ...}

    TODO:
        - Replace heuristic logic with calls to a large language model to produce labels + rationales
    """
    results = []
    for p in filtered_pairs:
        a = p["a"]["text"]
        b = p["b"]["text"]
        # Very simple heuristic: significant wordset change => meaning shift
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        overlap = len(a_set.intersection(b_set))
        total = max(1, len(a_set.union(b_set)))
        change_ratio = 1 - (overlap / total)
        if change_ratio > 0.4:
            shift_type = "meaning_shift"
            explanation = "Substantial lexical change between chunks suggests a meaning shift."
        else:
            shift_type = "no_shift"
            explanation = "Chunks are lexically similar; no clear meaning shift detected."
        results.append({"id": p["id"], "shift_type": shift_type, "explanation": explanation, "score": p.get("score")})
    return results
