#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANNOTATIONS_DIR = Path("data/annotations")
SEGMENT_FILE = ANNOTATIONS_DIR / "segment_level_annotation - segment_level_annotation_sample10_seed42.csv"
OUTPUT_DIR = Path("results/segment_vs_sentence_gt")
NO_SHIFT = "No Shift"


def discover_sentence_system_files(annotations_dir: Path) -> Dict[str, Path]:
    files = sorted(annotations_dir.glob("llm_annotator_*_nli - annotation.csv"))
    out: Dict[str, Path] = {}
    for p in files:
        name = p.name
        prefix = "llm_annotator_"
        marker = " - annotation.csv"
        if name.startswith(prefix) and marker in name:
            system = name[len(prefix):name.index(marker)]
            out[system] = p
    return out


def normalize_note_id(value: Optional[str]) -> str:
    s = str(value or "").strip()
    if s.lower().endswith(".txt"):
        s = s[:-4]
    return s


def normalize_idx(value: Optional[str]) -> str:
    s = str(value or "").strip()
    if not s:
        return s
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def normalize_label(value: Optional[str]) -> str:
    return str(value or "").strip()


def key_of(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        normalize_note_id(row.get("note_id")),
        normalize_idx(row.get("orig_i")),
        normalize_idx(row.get("orig_j")),
    )


def is_positive(label: str) -> bool:
    return normalize_label(label).lower() != NO_SHIFT.lower() and normalize_label(label) != ""


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    classes = sorted(set(y_true) | set(y_pred))
    per_class = []
    total_tp = total_fp = total_fn = 0
    support_sum = 0

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        support = sum(1 for t in y_true if t == cls)
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
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    macro_p = sum(x["precision"] for x in per_class) / len(per_class) if per_class else 0.0
    macro_r = sum(x["recall"] for x in per_class) / len(per_class) if per_class else 0.0
    macro_f1 = sum(x["f1"] for x in per_class) / len(per_class) if per_class else 0.0
    weighted_p = sum(x["precision"] * x["support"] for x in per_class) / support_sum if support_sum else 0.0
    weighted_r = sum(x["recall"] * x["support"] for x in per_class) / support_sum if support_sum else 0.0
    weighted_f1 = sum(x["f1"] * x["support"] for x in per_class) / support_sum if support_sum else 0.0

    return {
        "n": len(y_true),
        "classes": classes,
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
    }


def confusion_matrix(y_true: List[str], y_pred: List[str], class_order: List[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(class_order)}
    mat = np.zeros((len(class_order), len(class_order)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[idx[t], idx[p]] += 1
    return mat


def save_confusion_csv(path: Path, class_order: List[str], mat: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred"] + class_order)
        for i, cls in enumerate(class_order):
            writer.writerow([cls] + [int(x) for x in mat[i]])


def plot_confusions(path: Path, mats: Dict[str, np.ndarray], class_order: List[str], title: str) -> None:
    systems = sorted(mats)
    fig, axes = plt.subplots(1, len(systems), figsize=(6 * len(systems), 5), squeeze=False)
    vmax = max(int(mats[s].max()) for s in systems) if systems else 1

    for i, s in enumerate(systems):
        ax = axes[0, i]
        mat = mats[s]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1, vmax))
        ax.set_title(s)
        ax.set_xlabel("Sentence-level GT shift type")
        ax.set_ylabel("Segment-level GT shift type")
        ax.set_xticks(range(len(class_order)))
        ax.set_yticks(range(len(class_order)))
        ax.set_xticklabels(class_order, rotation=30, ha="right")
        ax.set_yticklabels(class_order)
        threshold = mat.max() * 0.6 if mat.size else 0
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                color = "white" if mat[r, c] > threshold else "black"
                ax.text(c, r, str(int(mat[r, c])), ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Count")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system_files = discover_sentence_system_files(ANNOTATIONS_DIR)
    if not SEGMENT_FILE.exists():
        raise SystemExit(f"Segment file not found: {SEGMENT_FILE}")
    if not system_files:
        raise SystemExit("No sentence-level system annotation files found.")

    with SEGMENT_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        segment_rows = list(csv.DictReader(f))

    segment_map = {key_of(r): normalize_label(r.get("gt_shift_type")) for r in segment_rows}
    positive_keys = [k for k, v in segment_map.items() if is_positive(v)]

    system_maps: Dict[str, Dict[Tuple[str, str, str], str]] = {}
    for system, path in sorted(system_files.items()):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        system_maps[system] = {key_of(r): normalize_label(r.get("gt_shift_type")) for r in rows}

    missed_rows = []
    for system, smap in sorted(system_maps.items()):
        missing_alignment = 0
        predicted_no_shift = 0
        caught_positive = 0
        aligned_positive = 0
        for k in positive_keys:
            v = smap.get(k)
            if v is None:
                missing_alignment += 1
            else:
                aligned_positive += 1
                if not is_positive(v):
                    predicted_no_shift += 1
                else:
                    caught_positive += 1
        missed_on_aligned = predicted_no_shift
        missed_rows.append({
            "system": system,
            "segment_positive_rows": len(positive_keys),
            "aligned_positive_rows": aligned_positive,
            "non_aligned_positive_rows": missing_alignment,
            "caught_positive": caught_positive,
            "missed_on_aligned": missed_on_aligned,
            "miss_rate_aligned_pct": (missed_on_aligned / aligned_positive * 100.0) if aligned_positive else 0.0,
        })

    caught_any_keys = []
    for k in positive_keys:
        caught = any(is_positive(system_maps[s].get(k, NO_SHIFT)) for s in system_maps)
        if caught:
            caught_any_keys.append(k)

    metrics_by_system = {}
    confusion_by_system: Dict[str, np.ndarray] = {}
    class_order = sorted({segment_map[k] for k in caught_any_keys} | {NO_SHIFT})

    for system, smap in sorted(system_maps.items()):
        y_true = [segment_map[k] for k in caught_any_keys]
        y_pred = [smap.get(k, NO_SHIFT) or NO_SHIFT for k in caught_any_keys]
        y_pred = [p if p else NO_SHIFT for p in y_pred]
        metrics = compute_metrics(y_true, y_pred)
        metrics_by_system[system] = metrics
        mat = confusion_matrix(y_true, y_pred, class_order)
        confusion_by_system[system] = mat
        save_confusion_csv(OUTPUT_DIR / f"caught_confusion_matrix_{system}.csv", class_order, mat)

    plot_confusions(
        OUTPUT_DIR / "caught_confusion_matrices.png",
        confusion_by_system,
        class_order,
        "Caught Positive Rows: Segment GT vs Sentence-level GT",
    )

    missed_path = OUTPUT_DIR / "missed_positive_rows_by_system.csv"
    with missed_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(missed_rows[0].keys()) if missed_rows else ["system"])
        writer.writeheader()
        writer.writerows(missed_rows)

    summary_rows = []
    per_class_rows = []
    for system, m in sorted(metrics_by_system.items()):
        ov = m["overall"]
        summary_rows.append({
            "system": system,
            "n_caught_any_rows": m["n"],
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
        for r in m["per_class"]:
            per_class_rows.append({"system": system, **r})

    summary_path = OUTPUT_DIR / "caught_metrics_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["system"])
        writer.writeheader()
        writer.writerows(summary_rows)

    per_class_path = OUTPUT_DIR / "caught_metrics_per_class.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()) if per_class_rows else ["system"])
        writer.writeheader()
        writer.writerows(per_class_rows)

    json_payload = {
        "segment_file": str(SEGMENT_FILE),
        "system_files": {k: str(v) for k, v in system_files.items()},
        "segment_positive_rows_total": len(positive_keys),
        "caught_any_positive_rows_total": len(caught_any_keys),
        "missed_positive_rows_by_system": missed_rows,
        "caught_metrics_by_system": metrics_by_system,
        "class_order_for_confusion": class_order,
    }
    json_path = OUTPUT_DIR / "segment_vs_sentence_gt_metrics.json"
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Segment-level file:", SEGMENT_FILE)
    print("System files:")
    for s, p in sorted(system_files.items()):
        print(f"  - {s}: {p}")
    print(f"\nSegment-level positive rows total: {len(positive_keys)}")
    print(f"Caught by at least one system (positive) rows: {len(caught_any_keys)}\n")
    print("Missed positive rows by system:")
    for r in missed_rows:
        print(
            f"  {r['system']}: missed_on_aligned={r['missed_on_aligned']}/{r['aligned_positive_rows']} "
            f"({r['miss_rate_aligned_pct']:.1f}%), non_aligned={r['non_aligned_positive_rows']}, "
            f"caught={r['caught_positive']}"
        )
    print("\nCaught-subset overall metrics:")
    for r in summary_rows:
        print(
            f"  {r['system']}: n={r['n_caught_any_rows']}, micro_f1={r['micro_f1']:.4f}, "
            f"macro_f1={r['macro_f1']:.4f}, accuracy={r['accuracy']:.4f}"
        )
    print(f"\nSaved: {missed_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {per_class_path}")
    for s in sorted(confusion_by_system):
        print(f"Saved: {OUTPUT_DIR / f'caught_confusion_matrix_{s}.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'caught_confusion_matrices.png'}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
