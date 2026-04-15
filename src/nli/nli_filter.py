from typing import List, Dict


def _dummy_contradiction(a: str, b: str) -> bool:
    """Simple heuristic to detect contradictions in Greek/English dummy text.

    Marks as contradiction if negation words are present in one side and positive words on the other.
    This is a placeholder — replace with a real NLI model.
    """
    negations = {"δεν", "όχι", "μη" , "not"}
    a_has_neg = any(tok in a.lower() for tok in negations)
    b_has_neg = any(tok in b.lower() for tok in negations)
    return a_has_neg ^ b_has_neg


def filter_with_nli(pairs: List[Dict], nli_model: str) -> List[Dict]:
    """
    Filter candidate pairs using an NLI model to remove contradictory alignments.

    Args:
        pairs: Candidate pairs produced by the aligner.
        nli_model: Placeholder name of NLI model (not used in dummy implementation).

    Returns:
        Filtered list of pairs where contradictions were removed.

    TODO:
        - Integrate HuggingFace/other NLI model to compute entailment/contradiction
    """
    if not pairs:
        return []
    kept = []
    for p in pairs:
        a_text = p["a"]["text"]
        b_text = p["b"]["text"]
        if _dummy_contradiction(a_text, b_text):
            # skip contradictory pairs
            continue
        kept.append(p)
    return kept
