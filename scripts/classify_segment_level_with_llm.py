#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _load_aws_config(aws_json_path: Optional[str]) -> Dict[str, Any]:
    if not aws_json_path:
        return {}
    p = Path(aws_json_path)
    if not p.exists():
        raise FileNotFoundError(f"AWS config JSON not found: {p}")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    return cfg if isinstance(cfg, dict) else {}


def _build_bedrock_client(aws_cfg: Dict[str, Any]):
    try:
        import boto3
    except Exception as e:
        raise ImportError("boto3 is required. Install with `pip install boto3`.") from e

    region_name = aws_cfg.get("region_name") or aws_cfg.get("region") or "us-east-1"
    kwargs: Dict[str, Any] = {"region_name": region_name}

    access_key = aws_cfg.get("aws_access_key_id") or aws_cfg.get("access_key_id")
    secret_key = aws_cfg.get("aws_secret_access_key") or aws_cfg.get("secret_access_key")
    session_token = aws_cfg.get("aws_session_token") or aws_cfg.get("session_token")
    profile_name = aws_cfg.get("profile_name") or aws_cfg.get("profile")

    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token
        return boto3.client("bedrock-runtime", **kwargs)

    if profile_name:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        return session.client("bedrock-runtime")

    # fallback to default AWS provider chain (env vars, shared config, IAM role)
    return boto3.client("bedrock-runtime", **kwargs)


def _prompt(chunk_a: str, chunk_b: str) -> str:
    return f"""You are given two excerpts (segments) from a patient's clinical record, written in Greek at different times.

Your task is to determine whether there is a diagnostic shift between them.

A diagnostic shift refers to any meaningful change in the documented diagnosis.

IMPORTANT:
* Use clinical reasoning based only on the provided text.
* Do NOT introduce information not present in the text.
* Pay attention to negation.

Classify with EXACTLY one shift type:
* "No Shift"
* "Refinement"
* "Evolution / Progression"
* "Contradiction / Overturn"

Output STRICTLY valid JSON with:
* "diagnostic_shift": true or false
* "shift_type": one of ["No Shift", "Refinement", "Evolution / Progression", "Contradiction / Overturn"]
* "justification": concise evidence-based explanation

Segment A:
\"\"\"{chunk_a}\"\"\"

Segment B:
\"\"\"{chunk_b}\"\"\""""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_bool(x: Any) -> str:
    if isinstance(x, bool):
        return "TRUE" if x else "FALSE"
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return "TRUE"
    if s in {"false", "0", "no"}:
        return "FALSE"
    return str(x)


def run(
    input_csv: str,
    output_csv: str,
    model_id: str,
    aws_json: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 300,
    sleep_seconds: float = 0.0,
) -> Dict[str, Any]:
    in_path = Path(input_csv)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    aws_cfg = _load_aws_config(aws_json)
    model_id = aws_cfg.get("model_id") or model_id
    client = _build_bedrock_client(aws_cfg)

    rows_out = []
    total = 0
    ok = 0
    llm_errors = 0
    parse_errors = 0

    with in_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            note_id = str(row.get("note_id") or "").strip()
            orig_i = row.get("orig_i", "")
            orig_j = row.get("orig_j", "")
            i_date = row.get("i_date", "")
            j_date = row.get("j_date", "")
            seg_i_text = row.get("segment_i_text", "") or ""
            seg_j_text = row.get("segment_j_text", "") or ""
            gt_shift_type = row.get("gt_shift_type", "")
            annotator_notes = row.get("annotator_notes", "")
            source_json = row.get("source_json", "")

            prompt = _prompt(seg_i_text, seg_j_text)
            raw_text = ""
            parsed = None
            error = ""
            try:
                resp = client.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={
                        "temperature": float(temperature),
                        "maxTokens": int(max_tokens),
                    },
                )
                content = (((resp or {}).get("output") or {}).get("message") or {}).get("content") or []
                text_blocks = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
                raw_text = "\n".join([t for t in text_blocks if t]).strip()
                parsed = _extract_json_object(raw_text)
                if parsed is None:
                    parse_errors += 1
                    error = "llm_output_not_valid_json"
                else:
                    ok += 1
            except Exception as e:
                llm_errors += 1
                error = str(e)

            llm_diag = _safe_bool(parsed.get("diagnostic_shift")) if isinstance(parsed, dict) else ""
            llm_shift = parsed.get("shift_type", "") if isinstance(parsed, dict) else ""
            llm_just = parsed.get("justification", "") if isinstance(parsed, dict) else ""

            rows_out.append([
                note_id,
                orig_i,
                orig_j,
                i_date,
                j_date,
                seg_i_text,
                seg_j_text,
                llm_diag,
                llm_shift,
                llm_just,
                gt_shift_type,
                "",  # is_llm_justification_correct (for human annotator later)
                annotator_notes,
                json.dumps(parsed, ensure_ascii=False) if parsed else "",
                raw_text,
                error,
                source_json,
            ])

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "note_id",
            "orig_i",
            "orig_j",
            "i_date",
            "j_date",
            "segment_i_text",
            "segment_j_text",
            "llm_diagnostic_shift",
            "llm_shift_type",
            "llm_justification",
            "gt_shift_type",
            "is_llm_justification_correct",
            "annotator_notes",
            "llm_raw_json",
            "llm_raw_text",
            "llm_error",
            "source_json",
        ])
        writer.writerows(rows_out)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

    summary = {
        "input_csv": str(in_path),
        "output_csv": str(out_path),
        "model_id": model_id,
        "rows_total": total,
        "rows_with_parsed_json": ok,
        "llm_errors": llm_errors,
        "parse_errors": parse_errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        default="data/annotations/segment_level_annotation - segment_level_annotation_sample10_seed42.csv",
    )
    ap.add_argument(
        "--output-csv",
        default="data/annotations/llm_annotator_segment_level_llama3-70b - annotation.csv",
    )
    ap.add_argument("--aws-json", default=None, help="Optional AWS config JSON")
    ap.add_argument("--model-id", default="meta.llama3-70b-instruct-v1:0")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--sleep-seconds", type=float, default=0.0)
    args = ap.parse_args()

    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_id=args.model_id,
        aws_json=args.aws_json,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
