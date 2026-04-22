#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ANNOTATIONS_DIR = Path("data/annotations")
OUTPUT_DIR = Path("results/justification_metrics")
NO_SHIFT = "No Shift"


def discover_latest_system_files(annotations_dir: Path) -> Dict[str, Path]:
    pattern = "llm_annotator_*_nli - annotation*.csv"
    files = sorted(annotations_dir.glob(pattern))
    by_system: Dict[str, Path] = {}

    for file_path in files:
        name = file_path.name
        prefix = "llm_annotator_"
        marker = " - annotation"
        if not name.startswith(prefix) or marker not in name:
            continue
        system = name[len(prefix):name.index(marker)]
        prev = by_system.get(system)
        if prev is None or file_path.stat().st_mtime > prev.stat().st_mtime:
            by_system[system] = file_path
    return by_system


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def map_justification_label(raw: str) -> Tuple[Optional[str], Optional[float]]:
    v = normalize_text(raw).lower()
    if v in {"true", "correct", "yes", "1"}:
        return "correct", 1.0
    if v in {"partially", "partial", "partially correct", "0.5"}:
        return "partially correct", 0.5
    if v in {"false", "incorrect", "no", "0"}:
        return "incorrect", 0.0
    return None, None


def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_subset_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    n = len(rows)
    scores = [float(r["jscore"]) for r in rows]
    mean_jscore = safe_mean(scores)

    correct = sum(1 for r in rows if r["just_label"] == "correct")
    partial = sum(1 for r in rows if r["just_label"] == "partially correct")
    incorrect = sum(1 for r in rows if r["just_label"] == "incorrect")

    denom = n if n else 1
    return {
        "n": n,
        "mean_jscore": mean_jscore,
        "count_correct": correct,
        "count_partially_correct": partial,
        "count_incorrect": incorrect,
        "pct_correct": correct / denom * 100.0,
        "pct_partially_correct": partial / denom * 100.0,
        "pct_incorrect": incorrect / denom * 100.0,
    }


def load_rows(path: Path) -> Tuple[List[Dict[str, object]], int]:
    rows: List[Dict[str, object]] = []
    dropped_unmapped_justification = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = normalize_text(row.get("llm_shift_type"))
            gt = normalize_text(row.get("gt_shift_type"))
            just_raw = normalize_text(row.get("is_llm_justification_correct"))
            just_label, jscore = map_justification_label(just_raw)
            if just_label is None:
                dropped_unmapped_justification += 1
                continue
            rows.append({
                "pred": pred,
                "gt": gt,
                "just_raw": just_raw,
                "just_label": just_label,
                "jscore": jscore,
            })
    return rows, dropped_unmapped_justification


def is_positive_shift(label: str) -> bool:
    return normalize_text(label).lower() != NO_SHIFT.lower() and normalize_text(label) != ""


def main() -> None:
    selected_files = discover_latest_system_files(ANNOTATIONS_DIR)
    if not selected_files:
        raise SystemExit(f"No system annotation CSVs found in {ANNOTATIONS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    per_type_rows: List[Dict[str, object]] = []
    payload: Dict[str, object] = {
        "selected_files": {k: str(v) for k, v in selected_files.items()},
        "systems": {},
    }

    for system, path in sorted(selected_files.items()):
        rows, dropped = load_rows(path)

        all_rows = rows
        pred_positive = [r for r in rows if is_positive_shift(str(r["pred"]))]
        correct_positive = [
            r for r in rows if is_positive_shift(str(r["pred"])) and str(r["pred"]) == str(r["gt"])
        ]

        m_all = compute_subset_metrics(all_rows)
        m_pred_positive = compute_subset_metrics(pred_positive)
        m_correct_positive = compute_subset_metrics(correct_positive)

        for subset_name, m in [
            ("all_rows", m_all),
            ("predicted_positive_shifts", m_pred_positive),
            ("correctly_classified_positive_shifts", m_correct_positive),
        ]:
            summary_rows.append({
                "system": system,
                "subset": subset_name,
                **m,
                "dropped_unmapped_justification": dropped,
                "source_file": str(path),
            })

        pos_types = sorted({str(r["gt"]) for r in correct_positive if is_positive_shift(str(r["gt"]))})
        per_type_metrics: Dict[str, Dict[str, object]] = {}
        for shift_type in pos_types:
            rows_t = [r for r in correct_positive if str(r["gt"]) == shift_type]
            mt = compute_subset_metrics(rows_t)
            per_type_metrics[shift_type] = mt
            per_type_rows.append({
                "system": system,
                "shift_type": shift_type,
                **mt,
                "source_file": str(path),
            })

        payload["systems"][system] = {
            "source_file": str(path),
            "dropped_unmapped_justification": dropped,
            "metrics": {
                "all_rows": m_all,
                "predicted_positive_shifts": m_pred_positive,
                "correctly_classified_positive_shifts": m_correct_positive,
                "correctly_classified_positive_shifts_per_type": per_type_metrics,
            },
        }

    summary_path = OUTPUT_DIR / "justification_summary_table.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "subset",
                "n",
                "mean_jscore",
                "count_correct",
                "count_partially_correct",
                "count_incorrect",
                "pct_correct",
                "pct_partially_correct",
                "pct_incorrect",
                "dropped_unmapped_justification",
                "source_file",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    per_type_path = OUTPUT_DIR / "justification_correct_positive_per_shift_type.csv"
    with per_type_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "shift_type",
                "n",
                "mean_jscore",
                "count_correct",
                "count_partially_correct",
                "count_incorrect",
                "pct_correct",
                "pct_partially_correct",
                "pct_incorrect",
                "source_file",
            ],
        )
        writer.writeheader()
        for row in per_type_rows:
            writer.writerow(row)

    json_path = OUTPUT_DIR / "justification_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Selected files per system:")
    for system in sorted(selected_files):
        print(f"  - {system}: {selected_files[system]}")
    print()
    print("System comparison (mean JScore):")
    for row in summary_rows:
        print(
            f"  {row['system']:10s} | {row['subset']:<38s} | n={row['n']:>3d} | "
            f"JScore={row['mean_jscore']:.4f} | "
            f"%correct={row['pct_correct']:.1f} | %partial={row['pct_partially_correct']:.1f} | %incorrect={row['pct_incorrect']:.1f}"
        )

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {per_type_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
