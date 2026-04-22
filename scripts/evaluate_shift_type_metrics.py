#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANNOTATIONS_DIR = Path("data/annotations")
OUTPUT_DIR = Path("results/shift_type_metrics")


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


def safe_label(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_gt_pred_pairs(csv_path: Path) -> Tuple[List[Tuple[str, str]], int]:
    pairs: List[Tuple[str, str]] = []
    dropped = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = safe_label(row.get("gt_shift_type"))
            pred = safe_label(row.get("llm_shift_type"))
            if not gt or not pred:
                dropped += 1
                continue
            pairs.append((gt, pred))

    return pairs, dropped


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_metrics(pairs: List[Tuple[str, str]]) -> Dict[str, object]:
    classes = sorted({gt for gt, _ in pairs} | {pred for _, pred in pairs})
    per_class = []
    total_tp = total_fp = total_fn = 0
    support_sum = 0

    for cls in classes:
        tp = sum(1 for gt, pred in pairs if gt == cls and pred == cls)
        fp = sum(1 for gt, pred in pairs if gt != cls and pred == cls)
        fn = sum(1 for gt, pred in pairs if gt == cls and pred != cls)
        support = sum(1 for gt, _ in pairs if gt == cls)

        p, r, f1 = prf(tp, fp, fn)
        per_class.append({
            "class": cls,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

        total_tp += tp
        total_fp += fp
        total_fn += fn
        support_sum += support

    micro_p, micro_r, micro_f1 = prf(total_tp, total_fp, total_fn)
    accuracy = sum(1 for gt, pred in pairs if gt == pred) / len(pairs) if pairs else 0.0

    macro_precision = sum(x["precision"] for x in per_class) / len(per_class) if per_class else 0.0
    macro_recall = sum(x["recall"] for x in per_class) / len(per_class) if per_class else 0.0
    macro_f1 = sum(x["f1"] for x in per_class) / len(per_class) if per_class else 0.0

    weighted_precision = (
        sum(x["precision"] * x["support"] for x in per_class) / support_sum if support_sum else 0.0
    )
    weighted_recall = (
        sum(x["recall"] * x["support"] for x in per_class) / support_sum if support_sum else 0.0
    )
    weighted_f1 = (
        sum(x["f1"] * x["support"] for x in per_class) / support_sum if support_sum else 0.0
    )

    return {
        "n": len(pairs),
        "classes": classes,
        "per_class": per_class,
        "overall": {
            "accuracy": accuracy,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
        },
    }


def build_confusion_matrix(pairs: List[Tuple[str, str]], class_order: List[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(class_order)}
    mat = np.zeros((len(class_order), len(class_order)), dtype=int)
    for gt, pred in pairs:
        mat[idx[gt], idx[pred]] += 1
    return mat


def write_confusion_csv(out_path: Path, class_order: List[str], mat: np.ndarray) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["gt\\pred"] + class_order
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, gt in enumerate(class_order):
            row = {"gt\\pred": gt}
            for j, pred in enumerate(class_order):
                row[pred] = int(mat[i, j])
            writer.writerow(row)


def plot_confusion_matrices(
    out_path: Path,
    confusion_by_system: Dict[str, np.ndarray],
    class_order: List[str],
) -> None:
    systems = sorted(confusion_by_system)
    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    vmax = max(int(confusion_by_system[s].max()) for s in systems) if systems else 1
    for k, system in enumerate(systems):
        ax = axes[0, k]
        mat = confusion_by_system[system]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(vmax, 1))
        ax.set_title(system)
        ax.set_xlabel("Predicted (llm_shift_type)")
        ax.set_ylabel("Ground Truth (gt_shift_type)")
        ax.set_xticks(range(len(class_order)))
        ax.set_yticks(range(len(class_order)))
        ax.set_xticklabels(class_order, rotation=30, ha="right")
        ax.set_yticklabels(class_order)

        threshold = mat.max() * 0.6 if mat.size else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                color = "white" if mat[i, j] > threshold else "black"
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Count")
    fig.suptitle("Confusion Matrices by System", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_results(
    out_dir: Path,
    selected_files: Dict[str, Path],
    metrics_by_system: Dict[str, Dict[str, object]],
    dropped_by_system: Dict[str, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_class_rows = []

    for system in sorted(metrics_by_system):
        m = metrics_by_system[system]
        ov = m["overall"]
        summary_rows.append({
            "system": system,
            "source_file": str(selected_files[system]),
            "n_rows_used": m["n"],
            "n_rows_dropped_missing_labels": dropped_by_system[system],
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
        })

        for row in m["per_class"]:
            per_class_rows.append({
                "system": system,
                **row,
            })

    with (out_dir / "summary_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["system"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with (out_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()) if per_class_rows else ["system"])
        writer.writeheader()
        for row in per_class_rows:
            writer.writerow(row)

    payload = {
        "selected_files": {k: str(v) for k, v in selected_files.items()},
        "dropped_rows": dropped_by_system,
        "metrics_by_system": metrics_by_system,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_report(
    selected_files: Dict[str, Path],
    metrics_by_system: Dict[str, Dict[str, object]],
    dropped_by_system: Dict[str, int],
) -> None:
    print("Selected files per system:")
    for system in sorted(selected_files):
        print(f"  - {system}: {selected_files[system]}")
    print()

    for system in sorted(metrics_by_system):
        m = metrics_by_system[system]
        ov = m["overall"]
        print(f"=== {system} ===")
        print(f"rows_used={m['n']}, rows_dropped_missing_labels={dropped_by_system[system]}")
        print(
            "overall: "
            f"acc={ov['accuracy']:.4f}, "
            f"micro_p={ov['micro_precision']:.4f}, micro_r={ov['micro_recall']:.4f}, micro_f1={ov['micro_f1']:.4f}, "
            f"macro_p={ov['macro_precision']:.4f}, macro_r={ov['macro_recall']:.4f}, macro_f1={ov['macro_f1']:.4f}, "
            f"weighted_p={ov['weighted_precision']:.4f}, weighted_r={ov['weighted_recall']:.4f}, weighted_f1={ov['weighted_f1']:.4f}"
        )
        print("per-class:")
        for row in m["per_class"]:
            print(
                f"  {row['class']}: p={row['precision']:.4f}, r={row['recall']:.4f}, f1={row['f1']:.4f}, "
                f"support={row['support']}, tp={row['tp']}, fp={row['fp']}, fn={row['fn']}"
            )
        print()


def main() -> None:
    selected_files = discover_latest_system_files(ANNOTATIONS_DIR)
    if not selected_files:
        raise SystemExit(f"No system annotation CSVs found in {ANNOTATIONS_DIR}")

    metrics_by_system: Dict[str, Dict[str, object]] = {}
    dropped_by_system: Dict[str, int] = defaultdict(int)
    pairs_by_system: Dict[str, List[Tuple[str, str]]] = {}

    for system, path in sorted(selected_files.items()):
        pairs, dropped = load_gt_pred_pairs(path)
        dropped_by_system[system] = dropped
        pairs_by_system[system] = pairs
        metrics_by_system[system] = compute_metrics(pairs)

    write_results(OUTPUT_DIR, selected_files, metrics_by_system, dropped_by_system)

    class_order = sorted(
        {
            label
            for pairs in pairs_by_system.values()
            for (gt, pred) in pairs
            for label in (gt, pred)
        }
    )
    confusion_by_system: Dict[str, np.ndarray] = {}
    for system, pairs in sorted(pairs_by_system.items()):
        mat = build_confusion_matrix(pairs, class_order)
        confusion_by_system[system] = mat
        write_confusion_csv(OUTPUT_DIR / f"confusion_matrix_{system}.csv", class_order, mat)
    plot_confusion_matrices(OUTPUT_DIR / "confusion_matrices.png", confusion_by_system, class_order)

    print_report(selected_files, metrics_by_system, dropped_by_system)
    print(f"Saved: {OUTPUT_DIR / 'summary_metrics.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'per_class_metrics.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'metrics.json'}")
    for system in sorted(confusion_by_system):
        print(f"Saved: {OUTPUT_DIR / f'confusion_matrix_{system}.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
