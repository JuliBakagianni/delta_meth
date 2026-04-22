#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Dict, List, Tuple


SEGMENT_GT_FILE = Path("data/annotations/segment_level_annotation - segment_level_annotation_sample10_seed42.csv")
SYSTEM_FILES = {
    "greek_nli": Path("data/annotations/llm_annotator_greek_nli - annotation.csv"),
    "multi_nli": Path("data/annotations/llm_annotator_multi_nli - annotation.csv"),
    "transl_nli": Path("data/annotations/llm_annotator_transl_nli - annotation.csv"),
}
OUT_DIR = Path("results/segment_gt_comparison")
NO_SHIFT = "No Shift"


def norm_note_id(value: str) -> str:
    s = str(value or "").strip()
    if s.lower().endswith(".txt"):
        s = s[:-4]
    return s


def norm_idx(value: str) -> str:
    s = str(value or "").strip()
    if s == "":
        return s
    try:
        return str(int(float(s)))
    except Exception:
        return s


def row_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        norm_note_id(row.get("note_id", "")),
        norm_idx(row.get("orig_i", "")),
        norm_idx(row.get("orig_j", "")),
    )


def is_shift(label: str) -> bool:
    s = str(label or "").strip().lower()
    return s != "" and s != NO_SHIFT.lower()


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, object]:
    classes = sorted({gt for gt, _ in pairs} | {pred for _, pred in pairs})
    per_class = []

    for cls in classes:
        tp = sum(1 for gt, pred in pairs if gt == cls and pred == cls)
        fp = sum(1 for gt, pred in pairs if gt != cls and pred == cls)
        fn = sum(1 for gt, pred in pairs if gt == cls and pred != cls)
        support = sum(1 for gt, _ in pairs if gt == cls)
        p, r, f1 = prf(tp, fp, fn)
        per_class.append(
            {
                "class": cls,
                "precision": p,
                "recall": r,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    n = len(pairs)
    accuracy = sum(1 for gt, pred in pairs if gt == pred) / n if n else 0.0
    macro_p = sum(x["precision"] for x in per_class) / len(per_class) if per_class else 0.0
    macro_r = sum(x["recall"] for x in per_class) / len(per_class) if per_class else 0.0
    macro_f1 = sum(x["f1"] for x in per_class) / len(per_class) if per_class else 0.0
    total_support = sum(x["support"] for x in per_class)
    weighted_p = (
        sum(x["precision"] * x["support"] for x in per_class) / total_support if total_support else 0.0
    )
    weighted_r = (
        sum(x["recall"] * x["support"] for x in per_class) / total_support if total_support else 0.0
    )
    weighted_f1 = sum(x["f1"] * x["support"] for x in per_class) / total_support if total_support else 0.0
    total_tp = sum(x["tp"] for x in per_class)
    total_fp = sum(x["fp"] for x in per_class)
    total_fn = sum(x["fn"] for x in per_class)
    micro_p, micro_r, micro_f1 = prf(total_tp, total_fp, total_fn)

    y_true = [1 if is_shift(gt) else 0 for gt, _ in pairs]
    y_pred = [1 if is_shift(pred) else 0 for _, pred in pairs]
    b_tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    b_fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    b_fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    b_tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    b_p, b_r, b_f1 = prf(b_tp, b_fp, b_fn)
    b_acc = (b_tp + b_tn) / len(y_true) if y_true else 0.0

    return {
        "n": n,
        "per_class": per_class,
        "overall": {
            "accuracy": accuracy,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_p,
            "weighted_recall": weighted_r,
            "weighted_f1": weighted_f1,
        },
        "binary": {
            "tp": b_tp,
            "fp": b_fp,
            "fn": b_fn,
            "tn": b_tn,
            "precision": b_p,
            "recall": b_r,
            "f1": b_f1,
            "accuracy": b_acc,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with SEGMENT_GT_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        seg_rows = list(csv.DictReader(f))
    seg_gt = {row_key(r): str(r.get("gt_shift_type", "")).strip() for r in seg_rows}
    seg_keys = list(seg_gt.keys())

    summary_rows = []
    per_class_rows = []

    for system, path in SYSTEM_FILES.items():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sys_rows = list(csv.DictReader(f))
        pred_map = {row_key(r): str(r.get("llm_shift_type", "")).strip() for r in sys_rows}

        aligned_keys = [k for k in seg_keys if k in pred_map and pred_map[k] != ""]
        non_aligned_count = len(seg_keys) - len(aligned_keys)

        # Setting 1: all segment pairs; missing predictions become No Shift
        pairs_all = []
        for k in seg_keys:
            pred = pred_map.get(k, NO_SHIFT)
            if pred == "":
                pred = NO_SHIFT
            pairs_all.append((seg_gt[k], pred))
        metrics_all = evaluate_pairs(pairs_all)

        # Setting 2: aligned only
        pairs_aligned = [(seg_gt[k], pred_map[k]) for k in aligned_keys]
        metrics_aligned = evaluate_pairs(pairs_aligned)

        for setting_name, metrics in [
            ("all_segment_pairs_missing_as_no_shift", metrics_all),
            ("aligned_only", metrics_aligned),
        ]:
            ov = metrics["overall"]
            bi = metrics["binary"]
            summary_rows.append(
                {
                    "system": system,
                    "setting": setting_name,
                    "segment_total_rows": len(seg_keys),
                    "system_rows": len(sys_rows),
                    "aligned_rows": len(aligned_keys),
                    "non_aligned_rows": non_aligned_count,
                    "n_eval_rows": metrics["n"],
                    "binary_precision": bi["precision"],
                    "binary_recall": bi["recall"],
                    "binary_f1": bi["f1"],
                    "binary_accuracy": bi["accuracy"],
                    "binary_tp": bi["tp"],
                    "binary_fp": bi["fp"],
                    "binary_fn": bi["fn"],
                    "binary_tn": bi["tn"],
                    "accuracy": ov["accuracy"],
                    "micro_precision": ov["micro_precision"],
                    "micro_recall": ov["micro_recall"],
                    "micro_f1": ov["micro_f1"],
                    "macro_precision": ov["macro_precision"],
                    "macro_recall": ov["macro_recall"],
                    "macro_f1": ov["macro_f1"],
                    "weighted_precision": ov["weighted_precision"],
                    "weighted_recall": ov["weighted_recall"],
                    "weighted_f1": ov["weighted_f1"],
                }
            )

            for row in metrics["per_class"]:
                per_class_rows.append(
                    {
                        "system": system,
                        "setting": setting_name,
                        **row,
                    }
                )

    summary_path = OUT_DIR / "three_systems_vs_segment_gt_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    per_class_path = OUT_DIR / "three_systems_vs_segment_gt_per_class.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_class_rows)

    print(f"Saved: {summary_path}")
    print(f"Saved: {per_class_path}")


if __name__ == "__main__":
    main()
