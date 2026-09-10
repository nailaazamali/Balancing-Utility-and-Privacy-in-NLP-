from __future__ import annotations

import math
import re

import numpy as np
import torch

from .metrics import structural_quality
from .sensitive import detect_sensitive_items, normalized_key


def _mask_first(models, text: str, span: str):
    match = re.search(re.escape(str(span).strip()), str(text), flags=re.IGNORECASE)
    if not match:
        return None
    return str(text)[:match.start()] + models.mlm_tok.mask_token + str(text)[match.end():]


@torch.inference_mode()
def _topk_replacements(models, config, masked_text: str, original_span: str) -> list[dict]:
    enc = models.mlm_tok(masked_text, return_tensors="pt", truncation=True)
    enc = {k: v.to(models.device) for k, v in enc.items()}
    positions = (enc["input_ids"][0] == models.mlm_tok.mask_token_id).nonzero(as_tuple=False).flatten()
    if len(positions) != 1:
        return []

    logits = models.mlm(**enc).logits[0, int(positions[0])].float()
    k = min(int(config.K_MLM), int(logits.numel()))
    values, ids = torch.topk(logits, k=k)

    original = str(original_span).strip().lower()
    rows, seen = [], set()
    for value, token_id in zip(values.cpu().tolist(), ids.cpu().tolist()):
        replacement = models.mlm_tok.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()
        key = replacement.lower()
        if not replacement or key == original or key in seen:
            continue
        if models.mlm_tok.mask_token in replacement or not re.search(r"[A-Za-z0-9]", replacement):
            continue
        seen.add(key)
        rows.append({"replacement": replacement, "raw_utility": float(value)})

    if rows:
        raw = np.asarray([row["raw_utility"] for row in rows], dtype=float)
        if float(raw.max() - raw.min()) <= float(config.epsilon_num):
            scaled = np.ones_like(raw)
        else:
            scaled = (raw - raw.min()) / (raw.max() - raw.min())
        for row, utility in zip(rows, scaled):
            row["utility"] = float(utility)
    return rows


def _sample(options: list[dict], epsilon_token: float, delta_c_u: float, rng):
    if not options:
        return None
    utilities = np.asarray([row["utility"] for row in options], dtype=float)
    logits = float(epsilon_token) * utilities / (2.0 * float(delta_c_u))
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    idx = int(rng.choice(len(options), p=probs))
    selected = dict(options[idx])
    selected["sampling_probability"] = float(probs[idx])
    return selected


def new_sensitive_occurrences(models, config, source: str, candidate: str) -> list[dict]:
    source_keys = {normalized_key(x) for x in detect_sensitive_items(models, config, source)}
    return [
        item for item in detect_sensitive_items(models, config, candidate)
        if normalized_key(item) not in source_keys
    ]


def sanitise(models, config, source: str, candidate: str, epsilon_token: float,
             rewrite_fraction: float, seed: int) -> dict:
    new_items = new_sensitive_occurrences(models, config, source, candidate)
    if not new_items:
        return {
            "text": candidate,
            "epsilon_total": 0.0,
            "successful_rewrites": 0,
            "rewrite_trace": [],
        }

    new_items = sorted(new_items, key=lambda x: len(str(x["span"])), reverse=True)
    k_r = min(
        int(config.K_rw),
        int(math.ceil(float(rewrite_fraction) * len(new_items))),
        len(new_items),
    )
    current = str(candidate)
    rng = np.random.default_rng(int(seed))
    trace = []
    count = 0

    for item in new_items[:k_r]:
        previous = current
        masked = _mask_first(models, current, item["span"])
        if masked is None:
            trace.append({"span": item["span"], "status": "mask_failed"})
            continue
        options = _topk_replacements(models, config, masked, item["span"])
        sampled = _sample(options, epsilon_token, config.delta_c_u, rng)
        if sampled is None:
            trace.append({"span": item["span"], "status": "no_replacement"})
            continue
        proposed = masked.replace(models.mlm_tok.mask_token, sampled["replacement"], 1).strip()
        if not proposed or proposed == previous or not structural_quality(models, config, source, proposed):
            trace.append({
                "span": item["span"],
                "status": "reverted",
                "replacement": sampled["replacement"],
            })
            continue
        current = proposed
        count += 1
        trace.append({
            "span": item["span"],
            "category": item["category"],
            "status": "rewritten",
            "replacement": sampled["replacement"],
            "replacement_utility": sampled["utility"],
            "sampling_probability": sampled["sampling_probability"],
        })

    return {
        "text": current,
        "epsilon_total": float(count * float(epsilon_token)),
        "successful_rewrites": int(count),
        "rewrite_trace": trace,
    }
