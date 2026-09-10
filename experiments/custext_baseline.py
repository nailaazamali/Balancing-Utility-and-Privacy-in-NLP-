from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import ExperimentConfig
from src.models import ModelBundle
from src.selection import evaluate_candidate
from src.utils import set_seed


def rank_tuple(row):
    return (
        float(row["XAIStrength_NLP"]),
        -float(row["XAILeakage_NLP"]),
        float(row["DeltaP"]),
        float(row["SBERT_cos"]),
        -float(row["TD_norm"]),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score candidate texts produced by the official CusText implementation "
            "with the common counterfactual validity, utility, and leakage measures."
        )
    )
    parser.add_argument("candidates", help="CSV with sample_id, src and candidate columns")
    parser.add_argument("--config", default="configs/archived_run.json")
    parser.add_argument("--output-dir", default="outputs/custext")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    set_seed(int(config.seed))
    models = ModelBundle.load(config, args.device)

    frame = pd.read_csv(args.candidates)
    required = {"sample_id", "src", "candidate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    scored = []
    for row in frame.dropna(subset=["candidate"]).to_dict("records"):
        candidate = str(row["candidate"]).strip()
        if not candidate:
            continue
        scored.append({
            **row,
            **evaluate_candidate(models, config, str(row["src"]), candidate),
        })

    scored = pd.DataFrame(scored)
    selected = []
    for sample_id, group in scored.groupby("sample_id", sort=True):
        feasible = group[
            (group["valid_counterfactual"] == 1)
            & (group["XAILeakage_NLP"] <= float(config.tau_L_max))
        ]
        if len(feasible):
            best = max(feasible.to_dict("records"), key=rank_tuple)
            best["selection_status"] = "FEASIBLE"
            selected.append(best)
        else:
            selected.append({"sample_id": sample_id, "selection_status": "INFEASIBLE"})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "scored_candidates.csv", index=False)
    pd.DataFrame(selected).to_csv(output_dir / "selected_results.csv", index=False)
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
