from typing import List
import re


def chunk_notes(text: str, chunk_size: int = 150, chunk_unit: str = 'words') -> List[str]:
    """
    Split a single note text into non-overlapping chunks of approximately
    `chunk_size` words.

    Args:
        text: The full note text to chunk.
        chunk_size: Target number of words per chunk (approximate).

    Returns:
        A list of chunk strings.

    This function performs a language-agnostic whitespace tokenization so it
    works with multilingual text (including Greek). It avoids splitting
    inside words and produces contiguous, non-overlapping chunks.
    """
    if not text:
        return []

    if chunk_unit == 'words':
        words = text.split()
        chunks: List[str] = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk.strip())
        return chunks

    # sentence-level chunking
    if chunk_unit == 'sentences':
        # simple rule-based sentence splitter that works reasonably for clinical notes
        parts = [s.strip() for s in re.split(r"[.!?]+(?=\s|$)", text) if s.strip()]
        if chunk_size <= 1:
            # one sentence per chunk
            return parts
        chunks = []
        for i in range(0, len(parts), chunk_size):
            chunk = " ".join(parts[i : i + chunk_size])
            chunks.append(chunk.strip())
        return chunks

    # unknown unit: fallback to word-based
    return chunk_notes(text, chunk_size=chunk_size, chunk_unit='words')
