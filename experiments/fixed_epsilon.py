from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rank_tuple(row):
    return (
        float(row["XAIStrength_NLP"]),
        -float(row["XAILeakage_NLP"]),
        float(row["DeltaP"]),
        float(row["SBERT_cos"]),
        -float(row["TD_norm"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_pool", help="CSV or CSV.GZ with evaluated candidate rows")
    parser.add_argument("--output", default="outputs/fixed_epsilon_summary.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.candidate_pool)
    required = {
        "sample_id", "candidate_type", "epsilon_token", "valid_counterfactual",
        "XAIStrength_NLP", "XAILeakage_NLP", "DeltaP", "SBERT_cos", "TD_norm",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows = []
    for epsilon in sorted(frame["epsilon_token"].dropna().unique()):
        subset = frame[
            (frame["candidate_type"] == "sanitised")
            & np.isclose(frame["epsilon_token"].astype(float), float(epsilon))
            & (frame["valid_counterfactual"] == 1)
        ]
        selected = []
        for _, group in subset.groupby("sample_id"):
            selected.append(max(group.to_dict("records"), key=rank_tuple))
        if not selected:
            continue
        selected = pd.DataFrame(selected)
        rows.append({
            "epsilon_token": float(epsilon),
            "n": len(selected),
            "mean_strength": selected["XAIStrength_NLP"].mean(),
            "mean_leakage": selected["XAILeakage_NLP"].mean(),
            "sd_strength": selected["XAIStrength_NLP"].std(ddof=1),
            "sd_leakage": selected["XAILeakage_NLP"].std(ddof=1),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
