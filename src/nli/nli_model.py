from typing import Tuple, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


_models: Dict[str, Dict] = {}

# Cache for seq2seq translation models
_seq2seq_models: Dict[str, Dict] = {}


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


def predict_nli_batch(premises, hypotheses, model_name: str, batch_size: int = 32,
                      translate: bool = False, translation_model: Optional[str] = None,
                      translation_batch_size: int = 8) -> list:
    """Batch NLI predictions for lists of (premise, hypothesis) pairs.

    Returns a list of (label, confidence) tuples in the same order.
    """
    if len(premises) != len(hypotheses):
        raise ValueError("premises and hypotheses must have the same length")

    # Optionally translate inputs to target language (e.g., English) before NLI
    if translate and translation_model:
        premises = translate_batch(premises, translation_model, batch_size=translation_batch_size)
        hypotheses = translate_batch(hypotheses, translation_model, batch_size=translation_batch_size)

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


def _load_seq2seq_model(model_name: str, device: str = None):
    """Load and cache a seq2seq model/tokenizer for translation (e.g., mBART, Marian)."""
    if not model_name:
        raise ValueError("model_name must be provided to load seq2seq model")
    if model_name in _seq2seq_models:
        return _seq2seq_models[model_name]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    _seq2seq_models[model_name] = {"tokenizer": tokenizer, "model": model, "device": device}
    return _seq2seq_models[model_name]


def translate_batch(texts, model_name: str, batch_size: int = 8, max_length: int = 512):
    """Translate a list of texts using a seq2seq model.

    Returns list of translated strings in same order.
    """
    if not texts:
        return []
    loaded = _load_seq2seq_model(model_name)
    tokenizer = loaded["tokenizer"]
    model = loaded["model"]
    device = loaded["device"]

    out = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = ["" if t is None else str(t) for t in texts[start:start+batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        out.extend(decoded)
    return out

