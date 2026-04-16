#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# Ensure workspace root is on sys.path so `src` can be imported when running scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.segment_by_date import segment_note_file


def main():
    src_dir = Path('data/raw/evaggelismos_raw_txt_notes')
    out_dir = Path('data/processed/segments')
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src_dir.glob('*.txt') if p.is_file()])
    results = []
    for p in files:
        try:
            res = segment_note_file(str(p))
            out_path = out_dir / (p.stem + '.json')
            out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
            results.append({'file': p.name, 'out': str(out_path), 'segments': len(res.get('segments', []))})
        except Exception as e:
            results.append({'file': p.name, 'error': str(e)})

    summary = {'processed': len(files), 'results': results}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
