from __future__ import annotations

import math

import numpy as np
from rapidfuzz.distance import Levenshtein


def tokenize(models, text: str):
    return models.clf_tok.tokenize(str(text))


def structural_quality(models, config, source: str, candidate: str) -> int:
    candidate = str(candidate).strip()
    if not candidate:
        return 0
    refusal_patterns = (
        "i cannot", "i can't", "i am unable", "i'm unable",
        "as an ai", "cannot assist", "can't assist",
        "cannot comply", "can't comply",
    )
    if any(item in candidate.lower() for item in refusal_patterns):
        return 0
    src_n = max(1, len(tokenize(models, source)))
    cf_n = len(tokenize(models, candidate))
    if cf_n == 0 or cf_n > int(config.max_cf_tokens):
        return 0
    if cf_n > math.ceil(float(config.max_length_ratio) * src_n):
        return 0
    return 1


def xaistrength(models, config, source: str, candidate: str) -> dict:
    p_non, p_tox = models.toxic_probabilities([str(source), str(candidate)])
    embeddings = models.sentence_embeddings([str(source), str(candidate)])

    delta_p = float(p_non[1] - p_non[0])
    delta_p_pos = max(0.0, delta_p)
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    similarity = max(0.0, min(1.0, similarity))

    src_tokens = tokenize(models, source)
    cf_tokens = tokenize(models, candidate)
    td = int(Levenshtein.distance(src_tokens, cf_tokens))
    td_norm = float(td / (len(src_tokens) + float(config.epsilon_num)))
    minimality = 1.0 - min(1.0, td_norm)
    strength = float((delta_p_pos + similarity + minimality) / 3.0)

    return {
        "P_non_src": float(p_non[0]),
        "P_non_cf": float(p_non[1]),
        "P_tox_src": float(p_tox[0]),
        "P_tox_cf": float(p_tox[1]),
        "DeltaP": delta_p,
        "DeltaP_pos": delta_p_pos,
        "SBERT_cos": similarity,
        "TD": td,
        "TD_norm": td_norm,
        "Minimality": minimality,
        "XAIStrength_NLP": strength,
    }
