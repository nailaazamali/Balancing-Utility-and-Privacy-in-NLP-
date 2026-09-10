from __future__ import annotations

import json

import numpy as np

from .generation import generate_candidates
from .leakage import xaileakage
from .metrics import structural_quality, xaistrength
from .sanitisation import sanitise


def evaluate_candidate(models, config, source: str, candidate: str) -> dict:
    utility = xaistrength(models, config, source, candidate)
    leakage = xaileakage(models, config, source, candidate)
    q = structural_quality(models, config, source, candidate)
    textually_admissible = int(
        utility["SBERT_cos"] >= float(config.tau_S)
        and utility["TD_norm"] <= float(config.tau_TD)
        and q == 1
    )
    flip = int(utility["P_tox_src"] >= 0.5 and utility["P_tox_cf"] < 0.5)
    valid = int(textually_admissible == 1 and flip == 1)
    return {
        **utility,
        **leakage,
        "Q_structural": q,
        "textually_admissible": textually_admissible,
        "flip_to_non_toxic": flip,
        "valid_counterfactual": valid,
        "privacy_feasible": int(valid == 1 and leakage["XAILeakage_NLP"] <= float(config.tau_L_max)),
    }


def _rank(row: dict):
    return (
        float(row["XAIStrength_NLP"]),
        -float(row["XAILeakage_NLP"]),
        float(row["DeltaP"]),
        float(row["SBERT_cos"]),
        -float(row["TD_norm"]),
    )


def build_pool(models, config, row: dict, sample_id: int) -> list[dict]:
    source = str(row["tweet"])
    source_seed = int(config.seed * 100000 + sample_id * 1000)
    originals = generate_candidates(models, config, source, source_seed)
    pool = []

    for original_idx, item in enumerate(originals):
        original = item["text"]
        pool.append({
            "sample_id": sample_id,
            "dataset_index": int(row["dataset_index"]),
            "dataset_label": row["label_name"],
            "src": source,
            "candidate": original,
            "candidate_type": "original",
            "original_candidate_index": original_idx,
            "epsilon_token": np.nan,
            "rewrite_fraction": np.nan,
            "epsilon_total": 0.0,
            "successful_rewrites": 0,
            **item,
        })

        for eps_idx, epsilon_token in enumerate(config.epsilon_grid):
            for r_idx, rewrite_fraction in enumerate(config.rewrite_fractions):
                seed = source_seed + 100 + original_idx * 10000 + eps_idx * 100 + r_idx
                rewritten = sanitise(
                    models,
                    config,
                    source,
                    original,
                    float(epsilon_token),
                    float(rewrite_fraction),
                    seed,
                )
                pool.append({
                    "sample_id": sample_id,
                    "dataset_index": int(row["dataset_index"]),
                    "dataset_label": row["label_name"],
                    "src": source,
                    "candidate": rewritten["text"],
                    "candidate_type": "sanitised",
                    "original_candidate_index": original_idx,
                    "epsilon_token": float(epsilon_token),
                    "rewrite_fraction": float(rewrite_fraction),
                    "epsilon_total": rewritten["epsilon_total"],
                    "successful_rewrites": rewritten["successful_rewrites"],
                    "rewrite_trace": json.dumps(rewritten["rewrite_trace"], ensure_ascii=False),
                    **item,
                })

    cache = {}
    for candidate in dict.fromkeys(row["candidate"] for row in pool):
        cache[candidate] = evaluate_candidate(models, config, source, candidate)
    for row_out in pool:
        row_out.update(cache[row_out["candidate"]])
    return pool


def select_source(pool: list[dict]) -> dict:
    originals = [
        row for row in pool
        if row["candidate_type"] == "original" and row["valid_counterfactual"] == 1
    ]
    feasible = [
        row for row in pool
        if row["valid_counterfactual"] == 1 and row["privacy_feasible"] == 1
    ]
    best_original = max(originals, key=_rank) if originals else None
    selected = max(feasible, key=_rank) if feasible else None

    source = pool[0]
    return {
        "sample_id": int(source["sample_id"]),
        "dataset_index": int(source["dataset_index"]),
        "dataset_label": source["dataset_label"],
        "src": source["src"],
        "selection_status": "FEASIBLE" if selected is not None else "INFEASIBLE",
        "original_cf": None if best_original is None else best_original["candidate"],
        "original_XAIStrength_NLP": np.nan if best_original is None else best_original["XAIStrength_NLP"],
        "original_XAILeakage_NLP": np.nan if best_original is None else best_original["XAILeakage_NLP"],
        "selected_cf": None if selected is None else selected["candidate"],
        "selected_candidate_type": None if selected is None else selected["candidate_type"],
        "selected_epsilon_token": np.nan if selected is None else selected["epsilon_token"],
        "selected_rewrite_fraction": np.nan if selected is None else selected["rewrite_fraction"],
        "selected_XAIStrength_NLP": np.nan if selected is None else selected["XAIStrength_NLP"],
        "selected_XAILeakage_NLP": np.nan if selected is None else selected["XAILeakage_NLP"],
        "selected_DeltaP": np.nan if selected is None else selected["DeltaP"],
        "selected_SBERT_cos": np.nan if selected is None else selected["SBERT_cos"],
        "selected_TD_norm": np.nan if selected is None else selected["TD_norm"],
    }
