from __future__ import annotations

import numpy as np

from .sensitive import PRIVACY_WEIGHTS, detect_sensitive_items, normalized_key


def _claims(models, candidate: str, new_items: list[dict]) -> list[str]:
    spans = {item["span"].strip().lower() for item in new_items if item["span"].strip()}
    claims = []
    for sent in models.nlp(str(candidate)).sents:
        text = sent.text.strip()
        low = text.lower()
        if text and any(span in low for span in spans):
            claims.append(text)
    return list(dict.fromkeys(claims))


def xaileakage(models, config, source: str, candidate: str) -> dict:
    source_items = detect_sensitive_items(models, config, source)
    candidate_items = detect_sensitive_items(models, config, candidate)
    source_keys = {normalized_key(item) for item in source_items}

    by_key = {}
    for item in candidate_items:
        by_key.setdefault(normalized_key(item), item)

    new_items = [item for key, item in by_key.items() if key not in source_keys]
    if not new_items:
        return {
            "XAILeakage_NLP": 0.0,
            "L_new": 0.0,
            "L_unsup": 0.0,
            "new_sensitive_items": [],
            "claims": [],
            "entailment_probabilities": [],
        }

    numerator = sum(PRIVACY_WEIGHTS.get(item["category"], 0.30) for item in new_items)
    denominator = sum(
        PRIVACY_WEIGHTS.get(item["category"], 0.30) for item in by_key.values()
    ) + float(config.epsilon_num)
    l_new = float(numerator / denominator)

    claims = _claims(models, candidate, new_items)
    if claims:
        entailments = models.nli_entailment(source, claims)
        l_unsup = 1.0 - float(np.mean(entailments))
    else:
        entailments = []
        l_unsup = 1.0

    return {
        "XAILeakage_NLP": float(l_new * l_unsup),
        "L_new": l_new,
        "L_unsup": float(l_unsup),
        "new_sensitive_items": [normalized_key(item) for item in new_items],
        "claims": claims,
        "entailment_probabilities": entailments,
    }
