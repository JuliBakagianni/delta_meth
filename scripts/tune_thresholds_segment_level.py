#!/usr/bin/env python3
import argparse
import csv
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from src.alignment.embeddings import encode_chunks
from src.alignment.similarity import compute_similarity_matrix
from src.nli.nli_model import predict_nli_batch
from src.preprocessing.chunking import chunk_notes


DEFAULT_SIM_GRID = [0.20, 0.30, 0.40, 0.50, 0.60]
DEFAULT_CONTRA_GRID = [0.40, 0.50, 0.60, 0.70]
NO_SHIFT = "No Shift"


@dataclass
class SystemConfig:
    name: str
    nli_model: str
    translate_nli: bool = False
    translation_model: Optional[str] = None


DEFAULT_SYSTEMS = [
    SystemConfig(
        name="greek_nli",
        nli_model="lighteternal/nli-xlm-r-greek",
        translate_nli=False,
        translation_model=None,
    ),
    SystemConfig(
        name="multi_nli",
        nli_model="joeddav/xlm-roberta-large-xnli",
        translate_nli=False,
        translation_model=None,
    ),
    SystemConfig(
        name="transl_nli",
        nli_model="facebook/bart-large-mnli",
        translate_nli=True,
        translation_model="facebook/nllb-200-distilled-1.3B",
    ),
]


def parse_systems_json(path: Optional[str]) -> List[SystemConfig]:
    if not path:
        return DEFAULT_SYSTEMS
    p = Path(path)
    arr = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for x in arr:
        out.append(
            SystemConfig(
                name=str(x["name"]),
                nli_model=str(x["nli_model"]),
                translate_nli=bool(x.get("translate_nli", False)),
                translation_model=x.get("translation_model"),
            )
        )
    return out


def load_aws_config(aws_json_path: Optional[str]) -> Dict[str, Any]:
    if not aws_json_path:
        return {}
    p = Path(aws_json_path)
    if not p.exists():
        raise FileNotFoundError(f"AWS config JSON not found: {p}")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    return cfg if isinstance(cfg, dict) else {}


def build_bedrock_client(aws_cfg: Dict[str, Any]):
    import boto3

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

    return boto3.client("bedrock-runtime", **kwargs)


def llm_prompt(chunk_a: str, chunk_b: str) -> str:
    return f"""You are given two excerpts (chunks) from a patient's clinical record, written in Greek at different times.

These are partial pieces of clinical documentation. Your task is to determine whether there is a diagnostic shift between them.

A diagnostic shift refers to any meaningful change in the documented diagnosis.

IMPORTANT:
* The text is in Greek.
* Use clinical reasoning to interpret the meaning of the text.
* Your decision and justification must be supported by explicit evidence from the provided chunks.
* Do NOT introduce information that is not present in the text.

Classify the relationship using EXACTLY one of the following categories:
* "No Shift"
* "Refinement"
* "Evolution / Progression"
* "Contradiction / Overturn"

Output STRICTLY a valid JSON object with:
* "diagnostic_shift": true or false
* "shift_type": one of ["No Shift", "Refinement", "Evolution / Progression", "Contradiction / Overturn"]
* "justification": concise explanation

Chunk A:
\"\"\"{chunk_a}\"\"\"

Chunk B:
\"\"\"{chunk_b}\"\"\""""


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
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


def normalize_shift_type(x: str) -> str:
    s = str(x or "").strip()
    allowed = {
        "No Shift",
        "Refinement",
        "Evolution / Progression",
        "Contradiction / Overturn",
    }
    if s in allowed:
        return s
    low = s.lower()
    if "no shift" in low:
        return "No Shift"
    if "refinement" in low:
        return "Refinement"
    if "evolution" in low or "progression" in low:
        return "Evolution / Progression"
    if "contradiction" in low or "overturn" in low:
        return "Contradiction / Overturn"
    return "Contradiction / Overturn"


def make_cache_key(model_id: str, chunk1: str, chunk2: str) -> str:
    raw = f"{model_id}\n{chunk1}\n{chunk2}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def is_shift(label: str) -> bool:
    return str(label or "").strip().lower() != NO_SHIFT.lower() and str(label or "").strip() != ""


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    binary_true = [1 if is_shift(x) else 0 for x in y_true]
    binary_pred = [1 if is_shift(x) else 0 for x in y_pred]
    binary_f1 = f1_score(binary_true, binary_pred, average="binary", zero_division=0)
    return {"macro_f1": float(macro_f1), "binary_f1": float(binary_f1)}


def clear_model_caches() -> None:
    """Best-effort cleanup to reduce GPU memory pressure between systems."""
    try:
        from src.nli import nli_model as _nli_mod

        for obj in list(getattr(_nli_mod, "_models", {}).values()):
            model = obj.get("model")
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
        for obj in list(getattr(_nli_mod, "_seq2seq_models", {}).values()):
            model = obj.get("model")
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
        getattr(_nli_mod, "_models", {}).clear()
        getattr(_nli_mod, "_seq2seq_models", {}).clear()
    except Exception:
        pass

    try:
        from src.alignment import embeddings as _emb_mod

        for obj in list(getattr(_emb_mod, "_embed_models", {}).values()):
            model = obj.get("model")
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
        getattr(_emb_mod, "_embed_models", {}).clear()
    except Exception:
        pass

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def parse_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def heatmap_for_system(
    out_path: Path,
    sim_grid: List[float],
    contra_grid: List[float],
    macro_map: Dict[Tuple[float, float], float],
    binary_map: Dict[Tuple[float, float], float],
    title: str,
) -> None:
    macro = np.zeros((len(sim_grid), len(contra_grid)), dtype=float)
    binary = np.zeros((len(sim_grid), len(contra_grid)), dtype=float)
    for i, s in enumerate(sim_grid):
        for j, c in enumerate(contra_grid):
            macro[i, j] = macro_map[(s, c)]
            binary[i, j] = binary_map[(s, c)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, name in [
        (axes[0], macro, "Macro F1 (CV mean)"),
        (axes[1], binary, "Binary F1 (CV mean)"),
    ]:
        im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(contra_grid)))
        ax.set_yticks(range(len(sim_grid)))
        ax.set_xticklabels([f"{x:.2f}" for x in contra_grid])
        ax.set_yticklabels([f"{x:.2f}" for x in sim_grid])
        ax.set_xlabel("Contradiction threshold")
        ax.set_ylabel("Similarity threshold")
        ax.set_title(name)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--segment-csv",
        default="data/annotations/segment_level_annotation - segment_level_annotation_sample10_seed42.csv",
    )
    ap.add_argument("--out-dir", default="results/tuning")
    ap.add_argument("--systems-json", default=None, help="Optional systems config JSON override")
    ap.add_argument("--aws-json", default=None, help="Optional AWS config JSON")
    ap.add_argument("--llm-model-id", default="meta.llama3-70b-instruct-v1:0")
    ap.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    ap.add_argument("--sim-grid", default="0.20,0.30,0.40,0.50,0.60")
    ap.add_argument("--contra-grid", default="0.40,0.50,0.60,0.70")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--cache-path", default="results/tuning/llm_cache.json")
    ap.add_argument("--nli-batch-size", type=int, default=8)
    ap.add_argument("--translation-batch-size", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_grid = [float(x.strip()) for x in args.sim_grid.split(",") if x.strip()]
    contra_grid = [float(x.strip()) for x in args.contra_grid.split(",") if x.strip()]
    min_sim = min(sim_grid)
    systems = parse_systems_json(args.systems_json)

    rows = parse_csv_rows(Path(args.segment_csv))
    if not rows:
        raise SystemExit("No rows found in segment CSV.")

    for required in ["note_id", "orig_i", "orig_j", "segment_i_text", "segment_j_text", "gt_shift_type"]:
        if required not in rows[0]:
            raise ValueError(f"Missing required column in segment CSV: {required}")

    row_ids = []
    groups = []
    gt_labels = []
    row_candidates = {}

    print("[tuning] Building candidate chunk pairs (embedding + similarity)...")
    for r in rows:
        note_id = str(r.get("note_id", "")).strip()
        orig_i = str(r.get("orig_i", "")).strip()
        orig_j = str(r.get("orig_j", "")).strip()
        rid = f"{note_id}|{orig_i}|{orig_j}"
        row_ids.append(rid)
        groups.append(note_id)
        gt_labels.append(str(r.get("gt_shift_type", "")).strip())

        a = str(r.get("segment_i_text", "") or "")
        b = str(r.get("segment_j_text", "") or "")
        chunks_a = chunk_notes(a, chunk_size=1, chunk_unit="sentences")
        chunks_b = chunk_notes(b, chunk_size=1, chunk_unit="sentences")
        if not chunks_a or not chunks_b:
            row_candidates[rid] = []
            continue

        emb_a = encode_chunks(chunks_a, model_name=args.embedding_model)
        emb_b = encode_chunks(chunks_b, model_name=args.embedding_model)
        sim_mat = compute_similarity_matrix(emb_a, emb_b)

        pairs = []
        n1, n2 = sim_mat.shape
        for i in range(n1):
            for j in range(n2):
                sim = float(sim_mat[i, j])
                if sim >= min_sim:
                    pairs.append(
                        {
                            "i": i,
                            "j": j,
                            "sim": sim,
                            "chunk1": chunks_a[i],
                            "chunk2": chunks_b[j],
                        }
                    )
        row_candidates[rid] = pairs

    # Embedding model is no longer needed after candidate extraction.
    clear_model_caches()

    aws_cfg = load_aws_config(args.aws_json)
    llm_model_id = aws_cfg.get("model_id") or args.llm_model_id
    client = build_bedrock_client(aws_cfg)

    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        llm_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        llm_cache = {}

    all_cv_rows = []
    best_rows = []
    compare_rows = []

    unique_groups = sorted(set(groups))
    n_splits = min(max(2, args.cv_folds), len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    idx_all = np.arange(len(row_ids))

    for sys_cfg in systems:
        print(f"[tuning] System={sys_cfg.name} | NLI model={sys_cfg.nli_model}")

        # Run NLI once for all candidate pairs of this system.
        per_row_nli = {}
        for rid in row_ids:
            pairs = row_candidates[rid]
            if not pairs:
                per_row_nli[rid] = []
                continue
            premises = [p["chunk1"] for p in pairs]
            hypotheses = [p["chunk2"] for p in pairs]
            results = predict_nli_batch(
                premises,
                hypotheses,
                model_name=sys_cfg.nli_model,
                batch_size=int(args.nli_batch_size),
                translate=sys_cfg.translate_nli,
                translation_model=sys_cfg.translation_model,
                translation_batch_size=int(args.translation_batch_size),
            )
            enriched = []
            for p, (label, conf) in zip(pairs, results):
                z = dict(p)
                z["nli_label"] = str(label).lower()
                z["nli_conf"] = float(conf)
                enriched.append(z)
            per_row_nli[rid] = enriched

        combo_predictions: Dict[Tuple[float, float], List[str]] = {}

        for sim_thr in sim_grid:
            for contra_thr in contra_grid:
                y_pred = []
                for rid in row_ids:
                    candidates = per_row_nli[rid]
                    eligible = [
                        p
                        for p in candidates
                        if p["sim"] >= sim_thr and p["nli_label"] == "contradiction" and p["nli_conf"] >= contra_thr
                    ]
                    if not eligible:
                        y_pred.append(NO_SHIFT)
                        continue

                    best = max(eligible, key=lambda x: (x["nli_conf"], x["sim"]))
                    c1 = best["chunk1"]
                    c2 = best["chunk2"]

                    cache_key = make_cache_key(llm_model_id, c1, c2)
                    if cache_key in llm_cache:
                        parsed = llm_cache[cache_key]
                    else:
                        prompt = llm_prompt(c1, c2)
                        resp = client.converse(
                            modelId=llm_model_id,
                            messages=[{"role": "user", "content": [{"text": prompt}]}],
                            inferenceConfig={
                                "temperature": float(args.temperature),
                                "maxTokens": int(args.max_tokens),
                            },
                        )
                        content = (((resp or {}).get("output") or {}).get("message") or {}).get("content") or []
                        blocks = [x.get("text", "") for x in content if isinstance(x, dict) and "text" in x]
                        raw_text = "\n".join([b for b in blocks if b]).strip()
                        parsed = extract_json_object(raw_text)
                        llm_cache[cache_key] = parsed if isinstance(parsed, dict) else {}

                    pred_shift = normalize_shift_type((parsed or {}).get("shift_type", "Contradiction / Overturn"))
                    y_pred.append(pred_shift)

                combo_predictions[(sim_thr, contra_thr)] = y_pred

        # Persist cache after each system.
        cache_path.write_text(json.dumps(llm_cache, ensure_ascii=False), encoding="utf-8")

        macro_map: Dict[Tuple[float, float], float] = {}
        binary_map: Dict[Tuple[float, float], float] = {}

        for sim_thr in sim_grid:
            for contra_thr in contra_grid:
                y_pred = combo_predictions[(sim_thr, contra_thr)]
                fold_macro = []
                fold_binary = []
                for _, test_idx in splitter.split(idx_all, groups=groups):
                    y_t = [gt_labels[i] for i in test_idx]
                    y_p = [y_pred[i] for i in test_idx]
                    m = compute_metrics(y_t, y_p)
                    fold_macro.append(m["macro_f1"])
                    fold_binary.append(m["binary_f1"])

                macro_mean = float(np.mean(fold_macro))
                macro_std = float(np.std(fold_macro))
                binary_mean = float(np.mean(fold_binary))
                binary_std = float(np.std(fold_binary))
                full_data = compute_metrics(gt_labels, y_pred)
                macro_map[(sim_thr, contra_thr)] = macro_mean
                binary_map[(sim_thr, contra_thr)] = binary_mean

                all_cv_rows.append(
                    {
                        "system": sys_cfg.name,
                        "sim_threshold": sim_thr,
                        "contradiction_threshold": contra_thr,
                        "cv_macro_f1_mean": macro_mean,
                        "cv_macro_f1_std": macro_std,
                        "cv_binary_f1_mean": binary_mean,
                        "cv_binary_f1_std": binary_std,
                        "full_data_macro_f1": full_data["macro_f1"],
                        "full_data_binary_f1": full_data["binary_f1"],
                    }
                )

        scored = [r for r in all_cv_rows if r["system"] == sys_cfg.name]
        best = max(scored, key=lambda x: (x["cv_macro_f1_mean"], x["cv_binary_f1_mean"]))
        default = next(
            (
                r
                for r in scored
                if abs(float(r["sim_threshold"]) - 0.5) < 1e-9
                and abs(float(r["contradiction_threshold"]) - 0.7) < 1e-9
            ),
            None,
        )

        best_rows.append(
            {
                "system": sys_cfg.name,
                "best_sim_threshold": best["sim_threshold"],
                "best_contradiction_threshold": best["contradiction_threshold"],
                "best_cv_macro_f1_mean": best["cv_macro_f1_mean"],
                "best_cv_binary_f1_mean": best["cv_binary_f1_mean"],
            }
        )
        if default is not None:
            compare_rows.append(
                {
                    "system": sys_cfg.name,
                    "default_sim_threshold": 0.5,
                    "default_contradiction_threshold": 0.7,
                    "default_cv_macro_f1_mean": default["cv_macro_f1_mean"],
                    "default_cv_binary_f1_mean": default["cv_binary_f1_mean"],
                    "best_sim_threshold": best["sim_threshold"],
                    "best_contradiction_threshold": best["contradiction_threshold"],
                    "best_cv_macro_f1_mean": best["cv_macro_f1_mean"],
                    "best_cv_binary_f1_mean": best["cv_binary_f1_mean"],
                    "macro_f1_gain": best["cv_macro_f1_mean"] - default["cv_macro_f1_mean"],
                    "binary_f1_gain": best["cv_binary_f1_mean"] - default["cv_binary_f1_mean"],
                }
            )

        heatmap_for_system(
            out_path=out_dir / f"heatmap_{sys_cfg.name}.png",
            sim_grid=sim_grid,
            contra_grid=contra_grid,
            macro_map=macro_map,
            binary_map=binary_map,
            title=f"{sys_cfg.name} threshold tuning",
        )

        print(
            f"[tuning] {sys_cfg.name} best sim={best['sim_threshold']:.2f}, contra={best['contradiction_threshold']:.2f}, "
            f"macro_f1={best['cv_macro_f1_mean']:.4f}, binary_f1={best['cv_binary_f1_mean']:.4f}"
        )
        clear_model_caches()

    # Save tables.
    cv_path = out_dir / "cv_grid_scores.csv"
    with cv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "sim_threshold",
                "contradiction_threshold",
                "cv_macro_f1_mean",
                "cv_macro_f1_std",
                "cv_binary_f1_mean",
                "cv_binary_f1_std",
                "full_data_macro_f1",
                "full_data_binary_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(all_cv_rows)

    best_path = out_dir / "best_thresholds_by_system.csv"
    with best_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "best_sim_threshold",
                "best_contradiction_threshold",
                "best_cv_macro_f1_mean",
                "best_cv_binary_f1_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(best_rows)

    compare_path = out_dir / "best_vs_default.csv"
    with compare_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "default_sim_threshold",
                "default_contradiction_threshold",
                "default_cv_macro_f1_mean",
                "default_cv_binary_f1_mean",
                "best_sim_threshold",
                "best_contradiction_threshold",
                "best_cv_macro_f1_mean",
                "best_cv_binary_f1_mean",
                "macro_f1_gain",
                "binary_f1_gain",
            ],
        )
        writer.writeheader()
        writer.writerows(compare_rows)

    # Also save resolved system config for reproducibility.
    (out_dir / "systems_used.json").write_text(
        json.dumps([s.__dict__ for s in systems], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved: {cv_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {compare_path}")
    print(f"Saved heatmaps under: {out_dir}")


if __name__ == "__main__":
    main()
