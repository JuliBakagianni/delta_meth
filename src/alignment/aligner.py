import json
from pathlib import Path
from typing import List, Dict


def _jaccard_similarity(a: str, b: str) -> float:
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    if not a_set and not b_set:
        return 0.0
    inter = a_set.intersection(b_set)
    uni = a_set.union(b_set)
    return len(inter) / max(1, len(uni))


def align_chunks(processed_dir: str, pairs_dir: str, embedding_model: str, similarity_threshold: float) -> List[Dict]:
    """
    Create candidate aligned chunk pairs using a simple similarity function.

    Args:
        processed_dir: Directory containing `chunks.json` from preprocessing.
        pairs_dir: Directory where candidate pairs will be saved.
        embedding_model: Placeholder name of embedding model (not used in dummy implementation).
        similarity_threshold: Similarity threshold for keeping pairs.

    Returns:
        A list of pair dictionaries: {"id": str, "a": chunk, "b": chunk, "score": float}

    TODO:
        - Replace dummy Jaccard with model embeddings and cosine similarity
        - Allow cross-document alignment and configurable pairing strategies
    """
    proc = Path(processed_dir)
    pairs_p = Path(pairs_dir)
    pairs_p.mkdir(parents=True, exist_ok=True)

    chunks_file = proc / "chunks.json"
    if not chunks_file.exists():
        return []
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))

    pairs = []
    for i in range(len(chunks) - 1):
        a = chunks[i]
        b = chunks[i + 1]
        score = _jaccard_similarity(a["text"], b["text"])
        if score >= similarity_threshold:
            pairs.append({"id": f"pair-{i}", "a": a, "b": b, "score": score})

    out_file = pairs_p / "pairs.json"
    out_file.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    return pairs
