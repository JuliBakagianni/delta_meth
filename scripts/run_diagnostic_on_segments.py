#!/usr/bin/env python3
"""
Run the DiaShift pipeline sequentially across segments inside JSON files.

For each JSON in `data/processed/segments` (or a provided path), if the
file contains >1 segments, compare segment[i] vs segment[i+1] for all i.
Save per-file results to `data/results/diagnostic_shifts/`.

This script uses `src/pipeline/run_pipeline.run_pipeline` for the comparison
and captures the returned contradiction and per-pair details.
"""
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure package imports work when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.run_pipeline import run_pipeline


def process_json_file(in_path: Path, out_dir: Path, config_path: Optional[str] = None):
    data = json.loads(in_path.read_text(encoding='utf-8'))
    note_id = data.get('note_id', in_path.stem)
    segments = data.get('segments', [])

    if len(segments) <= 1:
        out = {'note_id': note_id, 'skipped': True, 'reason': 'single_segment', 'comparisons': []}
        out_path = out_dir / (in_path.stem + '.diagnostic_shifts.json')
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        return out

    comparisons = []
    for i in range(len(segments) - 1):
        a = segments[i]
        b = segments[i + 1]
        seg_meta = {
            'i': i,
            'i_date': a.get('date'),
            'j': i + 1,
            'j_date': b.get('date'),
        }
        text_a = a.get('text', '')
        text_b = b.get('text', '')
        try:
            # Use sentence-level chunking: one sentence per chunk for finer comparisons
            contradiction, detailed_pairs = run_pipeline(
                note_a=text_a,
                note_b=text_b,
                config_path=(config_path or 'configs/config.yaml'),
                verbose=False,
                chunk_unit='sentences',
                chunk_size=1,
            )
            comparisons.append({**seg_meta, 'contradiction': contradiction, 'detailed_pairs': detailed_pairs})
        except Exception as e:
            comparisons.append({**seg_meta, 'error': str(e)})

    out = {'note_id': note_id, 'skipped': False, 'comparisons': comparisons}
    out_path = out_dir / (in_path.stem + '.diagnostic_shifts.json')
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


def main(src_dir: Optional[str] = None):
    src = Path(src_dir) if src_dir else Path('data/processed/segments')
    out_dir = Path('data/results/diagnostic_shifts')
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.glob('*.json') if p.is_file()])
    summary = {'processed': 0, 'skipped': 0, 'errors': 0, 'files': []}
    for p in files:
        try:
            res = process_json_file(p, out_dir)
            summary['files'].append({'file': p.name, 'skipped': res.get('skipped', False), 'comparisons': len(res.get('comparisons', []))})
            summary['processed'] += 1
            if res.get('skipped'):
                summary['skipped'] += 1
        except Exception as e:
            summary['errors'] += 1
            summary['files'].append({'file': p.name, 'error': str(e)})

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    argv = sys.argv[1:]
    src_arg = argv[0] if argv else None
    main(src_arg)
