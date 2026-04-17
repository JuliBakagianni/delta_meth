from pathlib import Path
import json
import csv
import os
import random
import time
from typing import Optional, List, Dict, Any


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


def _find_result_jsons(results_dir: Path) -> List[Path]:
    """Find per-note diagnostic shift JSON outputs inside a directory tree."""
    if not results_dir.exists():
        return []
    preferred = sorted(results_dir.glob('**/*.diagnostic_shifts.json'))
    if preferred:
        return preferred
    return sorted(results_dir.glob('**/*.json'))


def _infer_setting_name(results_dir: Path, json_path: Path) -> str:
    """Infer setting name from path structure for merged multi-setting exports."""
    try:
        rel_parts = json_path.relative_to(results_dir).parts
    except Exception:
        return "unknown"

    if 'diagnostic_shifts' in rel_parts:
        idx = rel_parts.index('diagnostic_shifts')
        if idx > 0:
            return rel_parts[idx - 1]
        return "default"

    if len(rel_parts) > 1:
        return rel_parts[0]
    return "default"


def export_llm_candidates_csv(results_dir: str, out_csv: str) -> Dict[str, Any]:
    """Export contradiction candidates (one per compared segment-pair) to CSV.

    This reads per-note JSON files produced by `run_batch` and writes rows for
    comparisons where `comparison["contradiction"]` is not null.
    """
    src_root = Path(results_dir)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = _find_result_jsons(src_root)
    rows: List[List[Any]] = []
    notes_seen = 0
    bad_files = 0

    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding='utf-8'))
        except Exception:
            bad_files += 1
            continue

        note_id = str(obj.get('note_id') or fp.stem.replace('.diagnostic_shifts', ''))
        setting = _infer_setting_name(src_root, fp)
        comparisons = obj.get('comparisons') or []
        notes_seen += 1

        for comp in comparisons:
            contradiction = comp.get('contradiction')
            if not contradiction:
                continue

            rows.append([
                setting,
                note_id,
                comp.get('comp_idx'),
                comp.get('orig_i'),
                comp.get('orig_j'),
                comp.get('i_date'),
                comp.get('j_date'),
                contradiction.get('i'),
                contradiction.get('j'),
                contradiction.get('sim_score'),
                contradiction.get('nli_confidence'),
                contradiction.get('chunk1'),
                contradiction.get('chunk2'),
                str(fp),
            ])

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'setting',
            'note_id',
            'comp_idx',
            'orig_i',
            'orig_j',
            'i_date',
            'j_date',
            'sent_i',
            'sent_j',
            'sim_score',
            'nli_confidence',
            'chunk1',
            'chunk2',
            'source_json',
        ])
        writer.writerows(rows)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

    summary = {
        'results_dir': str(src_root),
        'out_csv': str(out_path),
        'json_files_found': len(files),
        'note_files_read': notes_seen,
        'bad_files': bad_files,
        'candidate_rows': len(rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _load_aws_config(aws_json_path: str) -> Dict[str, Any]:
    cfg = json.loads(Path(aws_json_path).read_text(encoding='utf-8'))
    return cfg if isinstance(cfg, dict) else {}


def _build_bedrock_client(aws_cfg: Dict[str, Any]):
    try:
        import boto3
    except Exception as e:
        raise ImportError("boto3 is required for LLM classification. Install with `pip install boto3`.") from e

    region_name = aws_cfg.get('region_name') or aws_cfg.get('region') or 'us-east-1'
    kwargs: Dict[str, Any] = {'region_name': region_name}

    access_key = aws_cfg.get('aws_access_key_id') or aws_cfg.get('access_key_id')
    secret_key = aws_cfg.get('aws_secret_access_key') or aws_cfg.get('secret_access_key')
    session_token = aws_cfg.get('aws_session_token') or aws_cfg.get('session_token')

    if access_key and secret_key:
        kwargs['aws_access_key_id'] = access_key
        kwargs['aws_secret_access_key'] = secret_key
        if session_token:
            kwargs['aws_session_token'] = session_token

    return boto3.client('bedrock-runtime', **kwargs)


def _diagnostic_shift_prompt(chunk_a: str, chunk_b: str) -> str:
    return f"""You are given two excerpts (chunks) from a patient's clinical record, written in Greek at different times.

These are partial pieces of clinical documentation. Your task is to determine whether there is a diagnostic shift between them.

A diagnostic shift refers to any meaningful change in the documented diagnosis.

IMPORTANT:

* The text is in Greek.
* Use clinical reasoning to interpret the meaning of the text.
* Your decision and justification must be supported by explicit evidence from the provided chunks.
* Do NOT introduce information that is not present in the text.

Classify the relationship using EXACTLY one of the following categories:

* "No Shift": No meaningful diagnostic change.
* "Refinement": The diagnosis becomes more specific or precise.
* "Evolution / Progression": The condition evolves or worsens over time.
* "Contradiction / Overturn": The earlier diagnosis is explicitly or implicitly contradicted.

Guidelines:

* Pay close attention to negation (e.g., "δεν υπάρχει", "χωρίς", "αρνητικό για").
* Contradictions are the strongest signal of a diagnostic shift.
* If the meaning remains consistent or unclear, choose "No Shift".

Output STRICTLY a valid JSON object with the following fields:

* "diagnostic_shift": true or false
* "shift_type": one of ["No Shift", "Refinement", "Evolution / Progression", "Contradiction / Overturn"]
* "justification": a concise explanation referencing specific evidence from both chunks

Chunk A:
\"\"\"{chunk_a}\"\"\"

Chunk B:
\"\"\"{chunk_b}\"\"\""""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or '').strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start:end + 1]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _find_segment_files(seg_dir: Path) -> Dict[str, Path]:
    files = sorted([p for p in seg_dir.glob('**/*.json') if p.is_file()])
    out: Dict[str, Path] = {}
    for p in files:
        out[p.stem] = p
    return out


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == '':
            return None
        return int(x)
    except Exception:
        return None


def classify_candidates_with_llm(
    candidates_csv: str,
    seg_dir: str,
    aws_json_path: str,
    out_csv: str,
    model_id: str = 'meta.llama3-70b-instruct-v1:0',
    temperature: float = 0.0,
    max_tokens: int = 300,
    sleep_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Call Bedrock Llama on candidate rows and write annotator-ready CSV."""
    in_path = Path(candidates_csv)
    seg_root = Path(seg_dir)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    aws_cfg = _load_aws_config(aws_json_path)
    model_id = aws_cfg.get('model_id') or model_id
    client = _build_bedrock_client(aws_cfg)

    seg_map = _find_segment_files(seg_root)
    rows_out: List[List[Any]] = []
    total = 0
    ok = 0
    llm_errors = 0
    parse_errors = 0
    missing_segments = 0

    with open(in_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            note_id = str(row.get('note_id') or '').strip()
            setting = row.get('setting', '')
            comp_idx = row.get('comp_idx', '')
            orig_i = row.get('orig_i', '')
            orig_j = row.get('orig_j', '')
            i_date = row.get('i_date', '')
            j_date = row.get('j_date', '')
            chunk1 = row.get('chunk1', '')
            chunk2 = row.get('chunk2', '')
            source_json = row.get('source_json', '')

            seg_i_date = ''
            seg_j_date = ''
            seg_i_text = ''
            seg_j_text = ''
            seg_file = seg_map.get(note_id)
            if seg_file and seg_file.exists():
                try:
                    seg_obj = json.loads(seg_file.read_text(encoding='utf-8'))
                    segments = seg_obj.get('segments') or []
                    idx_i = _safe_int(orig_i)
                    idx_j = _safe_int(orig_j)
                    if idx_i is not None and 0 <= idx_i < len(segments):
                        seg_i_date = segments[idx_i].get('date', '')
                        seg_i_text = segments[idx_i].get('text', '')
                    if idx_j is not None and 0 <= idx_j < len(segments):
                        seg_j_date = segments[idx_j].get('date', '')
                        seg_j_text = segments[idx_j].get('text', '')
                except Exception:
                    missing_segments += 1
            else:
                missing_segments += 1

            prompt = _diagnostic_shift_prompt(chunk1, chunk2)
            raw_text = ''
            parsed = None
            error = ''

            try:
                resp = client.converse(
                    modelId=model_id,
                    messages=[{
                        'role': 'user',
                        'content': [{'text': prompt}],
                    }],
                    inferenceConfig={
                        'temperature': float(temperature),
                        'maxTokens': int(max_tokens),
                    },
                )
                content = (((resp or {}).get('output') or {}).get('message') or {}).get('content') or []
                text_blocks = [c.get('text', '') for c in content if isinstance(c, dict) and 'text' in c]
                raw_text = '\n'.join([t for t in text_blocks if t]).strip()
                parsed = _extract_json_object(raw_text)
                if parsed is None:
                    parse_errors += 1
                    error = 'llm_output_not_valid_json'
                else:
                    ok += 1
            except Exception as e:
                llm_errors += 1
                error = str(e)

            shift_bool = parsed.get('diagnostic_shift') if isinstance(parsed, dict) else ''
            shift_type = parsed.get('shift_type') if isinstance(parsed, dict) else ''
            justif = parsed.get('justification') if isinstance(parsed, dict) else ''

            rows_out.append([
                setting,
                note_id,
                comp_idx,
                orig_i,
                orig_j,
                i_date,
                j_date,
                seg_i_date,
                seg_j_date,
                chunk1,
                chunk2,
                seg_i_text,
                seg_j_text,
                shift_bool,
                shift_type,
                justif,
                json.dumps(parsed, ensure_ascii=False) if parsed else '',
                raw_text,
                error,
                source_json,
            ])

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'setting',
            'note_id',
            'comp_idx',
            'orig_i',
            'orig_j',
            'i_date',
            'j_date',
            'seg_i_date',
            'seg_j_date',
            'chunk1',
            'chunk2',
            'segment_i_text',
            'segment_j_text',
            'llm_diagnostic_shift',
            'llm_shift_type',
            'llm_justification',
            'llm_raw_json',
            'llm_raw_text',
            'llm_error',
            'source_json',
        ])
        writer.writerows(rows_out)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

    summary = {
        'candidates_csv': str(in_path),
        'seg_dir': str(seg_root),
        'out_csv': str(out_path),
        'model_id': model_id,
        'temperature': temperature,
        'rows_total': total,
        'rows_with_parsed_json': ok,
        'llm_errors': llm_errors,
        'parse_errors': parse_errors,
        'missing_segment_context_rows': missing_segments,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


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
    from src.pipeline.run_pipeline import run_pipeline

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
        print(f"auto dir= {auto_subdir}, dir = {out_dir}")
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
    print(f"Running batch on {len(files)} files (sample_size={sample_size}, process_limit={process_limit})...")
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
        print(f"len filtered = {len(filtered)} ")
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
                print(f"Exception: {e}")
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
    ap.add_argument('--export-llm-csv', default=None, help='If set, export contradiction candidates from --results-dir to this CSV and exit')
    ap.add_argument('--results-dir', default=None, help='Root directory containing per-note diagnostic_shifts JSON outputs (for --export-llm-csv)')
    ap.add_argument('--classify-llm-csv', default=None, help='If set, classify candidates and write annotator CSV')
    ap.add_argument('--candidates-csv', default=None, help='Input candidates CSV (from --export-llm-csv)')
    ap.add_argument('--aws-json', default=None, help='Path to AWS credentials/config JSON for Bedrock')
    ap.add_argument('--llm-out-csv', default=None, help='Output CSV path for LLM classification results')
    ap.add_argument('--llm-model-id', default='meta.llama3-70b-instruct-v1:0', help='Bedrock modelId')
    ap.add_argument('--llm-temperature', type=float, default=0.0, help='LLM temperature')
    ap.add_argument('--llm-max-tokens', type=int, default=300, help='LLM max output tokens')
    ap.add_argument('--llm-sleep-seconds', type=float, default=0.0, help='Sleep between API calls')
    args = ap.parse_args()

    if args.export_llm_csv:
        export_src = args.results_dir or args.out_dir
        export_llm_candidates_csv(results_dir=export_src, out_csv=args.export_llm_csv)
        raise SystemExit(0)
    if args.classify_llm_csv:
        if not args.candidates_csv or not args.aws_json or not args.llm_out_csv:
            raise ValueError("--classify-llm-csv requires --candidates-csv, --aws-json, and --llm-out-csv")
        classify_candidates_with_llm(
            candidates_csv=args.candidates_csv,
            seg_dir=args.seg_dir,
            aws_json_path=args.aws_json,
            out_csv=args.llm_out_csv,
            model_id=args.llm_model_id,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            sleep_seconds=args.llm_sleep_seconds,
        )
        raise SystemExit(0)

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
