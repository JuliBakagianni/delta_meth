from typing import Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


_models: Dict[str, Dict] = {}


def _load_model(model_name: str, device: str = None):
    if not model_name:
        raise ValueError("model_name must be provided to load model")
    if model_name in _models:
        return _models[model_name]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    id2label = getattr(model.config, "id2label", None) or {}
    _models[model_name] = {"tokenizer": tokenizer, "model": model, "device": device, "id2label": id2label}
    return _models[model_name]


def predict_nli(premise: str, hypothesis: str, model_name: str) -> Tuple[str, float]:
    """
    Predict NLI label and confidence using HF AutoModelForSequenceClassification.

    Tokenization uses `padding=True` and `truncation=True` to ensure batched
    tensors have consistent lengths and avoid the tensor creation error.

    Returns `(label, confidence)` where `label` is lower-cased.
    """
    m = _load_model(model_name)
    tokenizer = m["tokenizer"]
    model = m["model"]
    device = m["device"]

    # Ensure inputs are strings
    premise = "" if premise is None else str(premise)
    hypothesis = "" if hypothesis is None else str(hypothesis)

    # Tokenize with padding/truncation to produce tensors of equal length
    inputs = tokenizer(premise, hypothesis, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Map predicted index to label
    pred_idx = int(probs.argmax())
    id2label = m.get("id2label") or {}
    label = id2label.get(pred_idx, str(pred_idx)).lower()
    # Normalize common label variants
    if label in ("contradiction", "contradictory"):
        norm_label = "contradiction"
    elif label == "entailment":
        norm_label = "entailment"
    elif label == "neutral":
        norm_label = "neutral"
    else:
        # Some models use different words like 'contradiction' may be 'Contradiction'
        norm_label = label.lower()

    confidence = float(probs[pred_idx])
    return norm_label, confidence


def predict_nli_batch(premises, hypotheses, model_name: str, batch_size: int = 32):
    """Batch NLI predictions for lists of (premise, hypothesis) pairs.

    Returns a list of (label, confidence) tuples in the same order.
    """
    if len(premises) != len(hypotheses):
        raise ValueError("premises and hypotheses must have the same length")

    m = _load_model(model_name)
    tokenizer = m["tokenizer"]
    model = m["model"]
    device = m["device"]

    results = []
    n = len(premises)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_p = ["" if x is None else str(x) for x in premises[start:end]]
        batch_h = ["" if x is None else str(x) for x in hypotheses[start:end]]

        inputs = tokenizer(batch_p, batch_h, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        id2label = m.get("id2label") or {}
        for prob in probs:
            pred_idx = int(prob.argmax())
            label = id2label.get(pred_idx, str(pred_idx)).lower()
            if label in ("contradiction", "contradictory"):
                norm_label = "contradiction"
            elif label == "entailment":
                norm_label = "entailment"
            elif label == "neutral":
                norm_label = "neutral"
            else:
                norm_label = label.lower()
            confidence = float(prob[pred_idx])
            results.append((norm_label, confidence))

    return results
