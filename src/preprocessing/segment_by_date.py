import re
from typing import List, Dict, Optional
from datetime import datetime


DATE_PATTERNS = [
    # dd/mm[/yy|yyyy] or dd-mm or dd.mm
    r"\d{1,2}[\/\.-]\d{1,2}(?:[\/\.-]\d{2,4})?",
]

# Compile a regex that finds a date or a date range near start of line
DATE_LINE_RE = re.compile(rf"^\s*(?P<header>({DATE_PATTERNS[0]}(?:\s*[-–—]\s*{DATE_PATTERNS[0]})?))(?:\s*[:\-–—]\s*)?(?P<rest>.*)$")


def _norm_date_str(s: str) -> str:
    """Normalize a single date string to YYYY-MM-DD when possible; else return original."""
    s = s.strip()
    # replace dots and dashes with slash for parsing convenience
    s2 = s.replace('.', '/').replace('-', '/').replace('–', '/').replace('—', '/')
    parts = s2.split('/')
    try:
        if len(parts) == 3:
            day, month, year = parts
            if len(year) == 2:
                year = '20' + year
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime('%Y-%m-%d')
        elif len(parts) == 2:
            day, month = parts
            # no year: return day/month with placeholder
            return f"{int(day):02d}/{int(month):02d}"
        else:
            return s
    except Exception:
        return s


def segment_note_text(text: str) -> List[Dict[str, str]]:
    """
    Segment a clinical note text into dated segments.

    Returns list of dicts: {"date": str, "text": str}
    - Leading text before any detected date is assigned date 'before_ICU'.
    - If no dates detected, returns one segment with date 'no_date_segments'.
    """
    lines = text.splitlines()
    segments: List[Dict[str, str]] = []

    current_date: Optional[str] = None
    current_lines: List[str] = []

    def flush():
        nonlocal current_date, current_lines
        if current_lines:
            seg_date = current_date if current_date is not None else 'before_ICU'
            segments.append({'date': seg_date, 'text': '\n'.join(line for line in current_lines).strip()})
            current_lines = []

    for ln in lines:
        m = DATE_LINE_RE.match(ln)
        if m:
            header = m.group('header')
            rest = m.group('rest') or ''
            # Normalize header: if it's a range, try to keep as-is but normalize endpoints
            if '-' in header or '–' in header or '—' in header:
                parts = re.split(r'[-–—]', header)
                norm_parts = [_norm_date_str(p) for p in parts]
                norm_header = '-'.join(norm_parts)
            else:
                norm_header = _norm_date_str(header)

            # flush previous
            flush()
            current_date = norm_header
            # start new segment including rest of the line (if any)
            if rest.strip():
                current_lines = [rest.strip()]
            else:
                current_lines = []
        else:
            current_lines.append(ln)

    # final flush
    flush()

    if not segments:
        # no date headers found: return entire note as single segment
        return [{'date': 'no_date_segments', 'text': text.strip()}]

    return segments


def segment_note_file(in_path: str) -> Dict:
    from pathlib import Path
    p = Path(in_path)
    text = p.read_text(encoding='utf-8')
    segments = segment_note_text(text)
    return {'note_id': p.name, 'segments': segments}
