from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    "strength": ["selected_XAIStrength_NLP", "XAIStrength_NLP"],
    "leakage": ["selected_XAILeakage_NLP", "XAILeakage_NLP"],
    "sbert": ["selected_SBERT_cos", "SBERT_cos"],
    "td_norm": ["selected_TD_norm", "TD_norm"],
}


def _pick_column(frame: pd.DataFrame, candidates: list[str], label: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"Missing {label} column. Tried: {candidates}")


def summarise(path: str, name: str) -> dict:
    frame = pd.read_csv(path)
    if "selection_status" not in frame.columns:
        raise ValueError(f"{path} must contain selection_status.")

    strength_col = _pick_column(frame, COLUMN_ALIASES["strength"], "strength")
    leakage_col = _pick_column(frame, COLUMN_ALIASES["leakage"], "leakage")
    sbert_col = _pick_column(frame, COLUMN_ALIASES["sbert"], "SBERT")
    td_col = _pick_column(frame, COLUMN_ALIASES["td_norm"], "token-distance")

    feasible = frame[frame["selection_status"] == "FEASIBLE"].copy()
    return {
        "configuration": name,
        "n": len(frame),
        "feasible_n": len(feasible),
        "infeasible_n": len(frame) - len(feasible),
        "feasible_rate": len(feasible) / max(1, len(frame)),
        "mean_strength": feasible[strength_col].mean(),
        "sd_strength": feasible[strength_col].std(ddof=1),
        "mean_leakage": feasible[leakage_col].mean(),
        "sd_leakage": feasible[leakage_col].std(ddof=1),
        "mean_sbert": feasible[sbert_col].mean(),
        "mean_td_norm": feasible[td_col].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--davidson-qwen")
    parser.add_argument("--davidson-gemma")
    parser.add_argument("--tweeteval-qwen")
    parser.add_argument("--tweeteval-gemma")
    parser.add_argument("--manifest", default="configs/robustness_manifest.json")
    parser.add_argument("--output", default="outputs/robustness_summary.csv")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    labels = {row["name"]: row for row in manifest["configurations"]}

    inputs = [
        (args.davidson_qwen, "Davidson-Qwen2.5-0.5B"),
        (args.davidson_gemma, "Davidson-Gemma-2-2B"),
        (args.tweeteval_qwen, "TweetEval-Qwen2.5-0.5B"),
        (args.tweeteval_gemma, "TweetEval-Gemma-2-2B"),
    ]
    rows = [summarise(path, name) for path, name in inputs if path]
    if not rows:
        raise ValueError("Provide at least one result file.")

    for row in rows:
        reference = labels.get(row["configuration"])
        if reference:
            row["reported_mean_strength"] = reference["reported_mean_strength"]
            row["reported_mean_leakage"] = reference["reported_mean_leakage"]
            row["strength_difference"] = (
                row["mean_strength"] - reference["reported_mean_strength"]
            )
            row["leakage_difference"] = (
                row["mean_leakage"] - reference["reported_mean_leakage"]
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
