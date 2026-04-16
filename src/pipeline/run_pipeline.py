"""Run a minimal DiaShift-style pipeline: chunk -> embed -> align -> NLI filter.

This example uses multilingual models and a tiny dummy example (including
Greek) to demonstrate the pipeline stages.
"""
from pathlib import Path
from typing import Optional
import yaml

from src.preprocessing.chunking import chunk_notes
from src.alignment.embeddings import encode_chunks
from src.alignment.similarity import compute_similarity_matrix, get_aligned_pairs
from src.nli.filtering import filter_contradictions
from src.nli.nli_model import predict_nli, predict_nli_batch


def run_pipeline(note_a: Optional[str] = None, note_b: Optional[str] = None,
                 config_path: str = "configs/config.yaml", verbose: bool = True,
                 chunk_unit: str = 'words', chunk_size: Optional[int] = None):
    """
    Orchestrate chunking, embedding, alignment, and NLI-based filtering.

    Args:
        note_a: Text of the first note (premise). If None, a small dummy note
            will be used that contains English and Greek sentences.
        note_b: Text of the second note (hypothesis). If None, a small dummy
            note will be used that contradicts the first in one chunk.
        sim_threshold: Cosine similarity threshold for candidate alignment.
        nli_threshold: Minimum NLI confidence to accept a contradiction.

    Returns:
        The selected contradiction pair dict or None.
    """
    # Load config
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Defaults from config
    sim_threshold = cfg.get("similarity_threshold", 0.5)
    nli_threshold = cfg.get("nli_threshold", 0.7)
    default_chunk_size = cfg.get("chunk_size", 150)
    if chunk_size is None:
        chunk_size = default_chunk_size
    embedding_model = cfg.get("embedding_model")
    nli_model = cfg.get("nli_model")

    # Dummy examples if none provided
    if note_a is None:
        note_a = (
            "Patient reports no chest pain and is ambulatory. "
            "Ο ασθενής δεν έχει πυρετό. "  # Greek: patient has no fever
            "Blood pressure stable."
        )
    if note_b is None:
        note_b = (
            "Patient complains of severe chest pain and is non-ambulatory. "
            "Ο ασθενής έχει πυρετό. "  # Greek: patient has fever (contradiction)
            "Blood pressure slightly elevated."
        )

    # Chunking
    chunks_a = chunk_notes(note_a, chunk_size=chunk_size, chunk_unit=chunk_unit)
    chunks_b = chunk_notes(note_b, chunk_size=chunk_size, chunk_unit=chunk_unit)
    print(f"[pipeline] note A -> {len(chunks_a)} chunks")
    print(f"[pipeline] note B -> {len(chunks_b)} chunks")

    # Embeddings
    emb_a = encode_chunks(chunks_a, model_name=embedding_model)
    emb_b = encode_chunks(chunks_b, model_name=embedding_model)

    # Similarity and alignment
    sim_mat = compute_similarity_matrix(emb_a, emb_b)
    aligned = get_aligned_pairs(chunks_a, chunks_b, sim_mat, threshold=sim_threshold)
    print(f"[pipeline] aligned candidate pairs (sim>={sim_threshold}): {len(aligned)}")

    # For debugging: run batched NLI on all aligned candidates and show labels
    detailed_pairs = []
    if aligned:
        premises = [p[3] for p in aligned]
        hypotheses = [p[4] for p in aligned]
        try:
            nli_results = predict_nli_batch(premises, hypotheses, nli_model)
        except Exception as e:
            # Fallback: try single predictions to preserve behavior
            if verbose:
                print(f"[pipeline] batched NLI failed: {e}, falling back to single predictions")
            nli_results = [predict_nli(p[3], p[4], nli_model) for p in aligned]

        for (i, j, sim_score, chunk1, chunk2), (label, conf) in zip(aligned, nli_results):
            entry = {
                "i": i,
                "j": j,
                "sim_score": sim_score,
                "nli_label": label,
                "nli_confidence": conf,
                "chunk1": chunk1,
                "chunk2": chunk2,
            }
            detailed_pairs.append(entry)
            if verbose:
                print(f"[pipeline] pair ({i},{j}) sim={sim_score:.3f} nli={label} ({conf:.3f})")
                print(f"  A: {chunk1}")
                print(f"  B: {chunk2}")

    # NLI filtering (select highest-confidence contradiction)
    contradiction = filter_contradictions(aligned, threshold=nli_threshold, nli_model_name=nli_model)
    if contradiction:
        print("[pipeline] contradiction FOUND:")
        print(f"  sim_score={contradiction['sim_score']:.3f}")
        print(f"  nli_confidence={contradiction['nli_confidence']:.3f}")
        print("  chunk A:")
        print(f"    {contradiction['chunk1']}")
        print("  chunk B:")
        print(f"    {contradiction['chunk2']}")
    else:
        print("[pipeline] no contradiction found (above threshold)")

    # Return both the selected contradiction (if any) and the detailed per-pair NLI outputs
    return contradiction, detailed_pairs


if __name__ == "__main__":
    run_pipeline()
