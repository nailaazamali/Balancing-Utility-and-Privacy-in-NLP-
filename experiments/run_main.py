from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm.auto import tqdm

from src.config import ExperimentConfig
from src.data import load_archived_cohorts
from src.models import ModelBundle
from src.selection import build_pool, select_source
from src.utils import ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/archived_run.json")
    parser.add_argument("--output-dir", default="outputs/main")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--check-archived-status-counts",
        action="store_true",
        help="Compare a fresh run with the archived status counts when they are available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    set_seed(int(config.seed))

    screening, positive_all, calibration, evaluation = load_archived_cohorts(
        config, ROOT
    )

    models = ModelBundle.load(config, args.device)
    _, p_tox = models.toxic_probabilities(screening["tweet"].astype(str).tolist())
    toxic_positive = int((p_tox >= 0.5).sum())
    if toxic_positive != int(config.toxic_positive_n):
        raise RuntimeError(
            f"The screening cohort produced {toxic_positive} Toxic-BERT-positive cases; "
            f"the archived cohort contains {config.toxic_positive_n}. "
            "Check the model environment before continuing."
        )

    eval_frame = evaluation if args.limit is None else evaluation.head(args.limit)
    output_dir = ensure_dir(args.output_dir)
    candidate_dir = ensure_dir(output_dir / "candidate_pools")

    selections = []
    for sample_id, row in enumerate(
        tqdm(eval_frame.to_dict("records"), desc="sources"), start=1
    ):
        pool = build_pool(models, config, row, sample_id)
        pd.DataFrame(pool).to_csv(
            candidate_dir / f"source_{sample_id:04d}.csv.gz",
            index=False,
            compression="gzip",
        )
        selections.append(select_source(pool))

    selected = pd.DataFrame(selections)
    selected.to_csv(output_dir / "selected_results.csv", index=False)

    if args.limit is None and args.check_archived_status_counts:
        if hasattr(config, "archived_feasible_n") and hasattr(config, "archived_infeasible_n"):
            feasible = int((selected["selection_status"] == "FEASIBLE").sum())
            infeasible = int((selected["selection_status"] == "INFEASIBLE").sum())
            print(
                "Fresh-run status counts:",
                {"feasible": feasible, "infeasible": infeasible},
            )
            print(
                "Archived status counts:",
                {
                    "feasible": int(config.archived_feasible_n),
                    "infeasible": int(config.archived_infeasible_n),
                },
            )

    print(selected["selection_status"].value_counts(dropna=False))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
