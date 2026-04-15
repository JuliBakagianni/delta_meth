import json
from pathlib import Path
from typing import List, Dict


def chunk_notes(raw_dir: str, processed_dir: str, chunk_size: int = 200) -> List[Dict]:
    """
    Read raw text notes and split them into smaller textual chunks.

    Args:
        raw_dir: Path to directory containing raw text files.
        processed_dir: Path where chunked outputs will be saved.
        chunk_size: Approximate maximum characters per chunk.

    Returns:
        A list of chunk dictionaries: {"id": str, "text": str}.

    TODO:
        - Add robust sentence-splitting and language-specific logic
        - Preserve metadata and note-level IDs
    """
    raw_path = Path(raw_dir)
    out_path = Path(processed_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    chunks = []
    idx = 0
    for txt_file in sorted(raw_path.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Very simple chunking: split on periods and newlines, aggregate until size
        parts = [p.strip() for p in text.replace('\n', '. ').split('.') if p.strip()]
        current = ""
        for part in parts:
            if current and len(current) + len(part) + 1 > chunk_size:
                chunks.append({"id": f"chunk-{idx}", "text": current.strip()})
                idx += 1
                current = part
            else:
                current = (current + " " + part).strip()
        if current:
            chunks.append({"id": f"chunk-{idx}", "text": current.strip()})
            idx += 1

    # Save a basic JSON representation for downstream modules
    out_file = out_path / "chunks.json"
    out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks
