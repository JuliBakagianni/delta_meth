from typing import List


def chunk_notes(text: str, chunk_size: int = 150) -> List[str]:
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

    words = text.split()
    chunks: List[str] = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk.strip())
    return chunks
