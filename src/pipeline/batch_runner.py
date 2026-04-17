from pathlib import Path
import json
import csv
import os
import random
from typing import Optional, List

from src.pipeline.run_pipeline import run_pipeline


def _ensure_csv_header(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['note_id', 'comp_idx', 'orig_i', 'orig_j', 'sim_score', 'nli_label', 'nli_confidence'])
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass


def _append_rows(path: Path, rows: List[List]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def _write_json_and_flush(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def run_batch(
    seg_dir: str = 'data/processed/segments',
    out_dir: str = 'data/results/diagnostic_shifts',
    local_csv: str = 'data/results/threshold_logs.csv',
    drive_root: Optional[str] = None,
    sample_size: Optional[int] = None,
    sample_seed: int = 42,
    process_limit: Optional[int] = None,
    chunk_unit: str = 'sentences',
    chunk_size: int = 1,
    config_path: Optional[str] = None,
    nli_model: Optional[str] = None,
    translate_nli: bool = False,
    translation_model: Optional[str] = None,
    auto_subdir: bool = True,
):
    """Run batch comparisons across segment JSONs.

    Returns list of processed note paths.
    """
    seg_dir = Path(seg_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally create a subdirectory encoding the configuration so different
    # runs don't overwrite each other's outputs. Only applied when `auto_subdir` is True.
    if auto_subdir:
        parts = []
        if nli_model:
            parts.append(str(nli_model).replace('/', '-'))
        else:
            parts.append('nli_default')
        if translate_nli:
            parts.append('translated')
            if translation_model:
                parts.append(str(translation_model).replace('/', '-'))
        parts.append(f'seed{sample_seed}')
        subname = '_'.join(parts)
        out_dir = out_dir / subname
        out_dir.mkdir(parents=True, exist_ok=True)

    drive_json_dir = None
    drive_csv_path = None
    use_drive = False
    if drive_root:
        drive_root_p = Path(drive_root)
        if drive_root_p.exists() and drive_root_p.is_dir():
            use_drive = True
            drive_json_dir = drive_root_p / 'diagnostic_shifts'
            drive_json_dir.mkdir(parents=True, exist_ok=True)
            drive_csv_path = drive_root_p / 'threshold_logs.csv'

    # Choose CSV location: if caller passed the default local_csv path, place it inside
    # the `out_dir` subfolder so each config run gets its own CSV; otherwise respect caller's path.
    default_local_csv = 'data/results/threshold_logs.csv'
    if auto_subdir and local_csv == default_local_csv:
        local_csv_path = out_dir / 'threshold_logs.csv'
    else:
        local_csv_path = Path(local_csv)
    local_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Gather multi-segment files
    all_files = sorted([p for p in seg_dir.glob('**/*.json') if p.is_file()])
    multi_files = []
    for p in all_files:
        try:
            note = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        if len(note.get('segments', [])) > 1:
            multi_files.append(p)

    # Sampling
    if sample_size is None or sample_size >= len(multi_files):
        files = multi_files
    else:
        rng = random.Random(sample_seed)
        files = rng.sample(multi_files, sample_size)

    if process_limit:
        files = files[:process_limit]

    _ensure_csv_header(local_csv_path)
    if use_drive and drive_csv_path:
        _ensure_csv_header(drive_csv_path)

    processed = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue

        segs = data.get('segments', [])
        # Handle single segment case
        if len(segs) <= 1:
            if len(segs) == 1 and segs[0].get('date') == 'before_ICU':
                segs[0]['date'] = 'ICU_summ'
                data['segments'] = segs
                p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            out_json = {'note_id': p.stem, 'skipped': True, 'reason': 'single_segment', 'skipped_empty_segments': [], 'comparisons': [], 'segments': segs}
            local_out = out_dir / (p.stem + '.diagnostic_shifts.json')
            _write_json_and_flush(local_out, out_json)
            if use_drive:
                _write_json_and_flush(drive_json_dir / local_out.name, out_json)
            processed.append(p)
            continue

        skipped_empty = []
        filtered = []
        for idx, seg in enumerate(segs):
            if not seg.get('text', '').strip():
                skipped_empty.append({'orig_index': idx, 'date': seg.get('date')})
            else:
                filtered.append((idx, seg))

        if len(filtered) <= 1:
            out_json = {'note_id': p.stem, 'skipped': True, 'reason': 'single_segment_after_filter', 'skipped_empty_segments': skipped_empty, 'comparisons': []}
            local_out = out_dir / (p.stem + '.diagnostic_shifts.json')
            _write_json_and_flush(local_out, out_json)
            if use_drive:
                _write_json_and_flush(drive_json_dir / local_out.name, out_json)
            processed.append(p)
            continue

        comparisons = []
        csv_rows = []
        for comp_idx in range(len(filtered) - 1):
            orig_i, a_seg = filtered[comp_idx]
            orig_j, b_seg = filtered[comp_idx + 1]
            a = a_seg.get('text', '')
            b = b_seg.get('text', '')
            try:
                contradiction, detailed = run_pipeline(
                    note_a=a,
                    note_b=b,
                        config_path=(config_path or 'configs/config.yaml'),
                    verbose=False,
                    chunk_unit=chunk_unit,
                    chunk_size=chunk_size,
                        nli_model_override=nli_model,
                        translate_nli=translate_nli,
                        translation_model=translation_model,
                )
            except Exception as e:
                comparisons.append({'comp_idx': comp_idx, 'orig_i': orig_i, 'orig_j': orig_j, 'i_date': a_seg.get('date'), 'j_date': b_seg.get('date'), 'error': str(e), 'contradiction': None, 'detailed_pairs': []})
                continue

            comparisons.append({'comp_idx': comp_idx, 'orig_i': orig_i, 'orig_j': orig_j, 'i_date': a_seg.get('date'), 'j_date': b_seg.get('date'), 'contradiction': contradiction, 'detailed_pairs': detailed})

            if detailed:
                # Read sim threshold from config file for CSV filtering
                sim_thr = 0.5
                try:
                    cfg = json.loads(Path(config_path or 'configs/config.yaml').read_text(encoding='utf-8')) if config_path and Path(config_path).exists() else None
                except Exception:
                    cfg = None
                # fallback if yaml
                if cfg is None:
                    try:
                        import yaml
                        cfg = yaml.safe_load(Path(config_path or 'configs/config.yaml').read_text(encoding='utf-8')) if Path(config_path or 'configs/config.yaml').exists() else {}
                    except Exception:
                        cfg = {}
                sim_thr = float(cfg.get('similarity_threshold', 0.5))

                for d in detailed:
                    sim = float(d.get('sim_score', 0.0))
                    label = (d.get('nli_label') or '').lower()
                    conf = d.get('nli_confidence', '')
                    if sim >= sim_thr or label == 'contradiction':
                        csv_rows.append([p.stem, comp_idx, orig_i, orig_j, sim, label, conf])

        out_json = {'note_id': p.stem, 'skipped': False, 'skipped_empty_segments': skipped_empty, 'comparisons': comparisons}
        local_out = out_dir / (p.stem + '.diagnostic_shifts.json')
        _write_json_and_flush(local_out, out_json)
        if use_drive:
            _write_json_and_flush(drive_json_dir / local_out.name, out_json)

        if csv_rows:
            _append_rows(local_csv_path, csv_rows)
            if use_drive and drive_csv_path:
                _append_rows(drive_csv_path, csv_rows)

        processed.append(p)

    return processed


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--seg-dir', default='data/processed/segments')
    ap.add_argument('--out-dir', default='data/results/diagnostic_shifts')
    ap.add_argument('--local-csv', default='data/results/threshold_logs.csv')
    ap.add_argument('--drive-root', default=None)
    ap.add_argument('--sample-size', type=int, default=None)
    ap.add_argument('--sample-seed', type=int, default=42)
    ap.add_argument('--process-limit', type=int, default=None)
    ap.add_argument('--chunk-unit', default='sentences')
    ap.add_argument('--chunk-size', type=int, default=1)
    ap.add_argument('--config-path', default=None)
    ap.add_argument('--nli-model', default=None, help='Override NLI model name')
    ap.add_argument('--translate-nli', action='store_true', help='Translate inputs before NLI')
    ap.add_argument('--translation-model', default=None, help='Seq2seq model name to use for translation')
    ap.add_argument('--auto-subdir', action='store_true', help='Create a subdirectory per configuration under out-dir')
    args = ap.parse_args()

    run_batch(
        seg_dir=args.seg_dir,
        out_dir=args.out_dir,
        local_csv=args.local_csv,
        drive_root=args.drive_root,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        process_limit=args.process_limit,
        chunk_unit=args.chunk_unit,
        chunk_size=args.chunk_size,
        config_path=args.config_path,
        nli_model=args.nli_model,
        translate_nli=args.translate_nli,
        translation_model=args.translation_model,
        auto_subdir=args.auto_subdir,
    )
