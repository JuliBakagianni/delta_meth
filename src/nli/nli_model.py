from typing import Tuple, Dict
from transformers import pipeline


_pipelines: Dict[str, object] = {}


def _get_pipeline(model_name: str):
    if not model_name:
        raise ValueError("model_name must be provided to _get_pipeline")
    if model_name not in _pipelines:
        _pipelines[model_name] = pipeline("text-classification", model=model_name, return_all_scores=False)
    return _pipelines[model_name]


def predict_nli(premise: str, hypothesis: str, model_name: str) -> Tuple[str, float]:
    """
    Predict NLI label and confidence for a single premise-hypothesis pair.

    Args:
        premise: The premise text.
        hypothesis: The hypothesis text.
        model_name: HF model name for the XNLI model.

    Returns:
        A tuple `(label, confidence)` where `label` is a lower-case string
        (e.g. "contradiction", "entailment", "neutral") and `confidence` is
        the softmax score for that label.

    Note: `model_name` must be provided by the caller (from config).
    """
    nlp = _get_pipeline(model_name)
    # Pipe accepts a tuple (premise, hypothesis) for pair classification
    out = nlp((premise, hypothesis))
    if isinstance(out, list) and out:
        res = out[0]
    else:
        res = out
    label = res["label"].lower()
    score = float(res["score"])
    return label, score
