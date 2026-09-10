from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
archived = json.loads((ROOT / "configs/archived_run.json").read_text(encoding="utf-8"))
reference = json.loads((ROOT / "configs/reference.json").read_text(encoding="utf-8"))


def load(path):
    return pd.read_csv(ROOT / path)


screening = load(archived["cohort_files"]["screening"])
positive = load(archived["cohort_files"]["positive_all"])
calibration = load(archived["cohort_files"]["calibration"])
evaluation = load(archived["cohort_files"]["evaluation"])

assert len(screening) == archived["screening_pool_n"]
assert len(positive) == archived["toxic_positive_n"]
assert len(calibration) == archived["calibration_n"]
assert len(evaluation) == archived["main_eval_n"]

screening_idx = set(screening["dataset_index"].astype(int))
positive_idx = set(positive["dataset_index"].astype(int))
calibration_idx = set(calibration["dataset_index"].astype(int))
evaluation_idx = set(evaluation["dataset_index"].astype(int))

assert not (calibration_idx & evaluation_idx)
assert calibration_idx.issubset(screening_idx)
assert evaluation_idx.issubset(screening_idx)
assert calibration_idx.issubset(positive_idx)
assert evaluation_idx.issubset(positive_idx)

# Raw source text/usernames must not be redistributed in the bundled cohort manifests.
for frame, path in [
    (screening, archived["cohort_files"]["screening"]),
    (positive, archived["cohort_files"]["positive_all"]),
    (calibration, archived["cohort_files"]["calibration"]),
    (evaluation, archived["cohort_files"]["evaluation"]),
]:
    assert "tweet" not in frame.columns, f"Raw tweet column found in {path}"

baseline = pd.read_csv(ROOT / "results/baseline_comparison.csv")
assert set(baseline["method"]) == {"Fixed-epsilon DP-MLM", "CusText", "Proposed"}

feasibility = pd.read_csv(ROOT / "results/feasibility_summary.csv").iloc[0]
assert int(feasibility["main_evaluation_n"]) == reference["main_eval_n"]
assert int(feasibility["feasible_n"]) == reference["expected_feasible_n"]
assert int(feasibility["infeasible_n"]) == reference["expected_infeasible_n"]

human = pd.read_csv(ROOT / "results/human_evaluation_summary.csv").iloc[0]
assert int(human["reviewed_n"]) == 450
assert int(human["initial_agreement_n"]) == 412
assert round(float(human["cohen_kappa"]), 3) == 0.831
assert int(human["resolved_disagreements_n"]) == 38

# Obsolete release artifacts and interpreter caches must not be present.
assert not list(ROOT.rglob("*.xlsx")), "Legacy Excel workbooks are present."

# The public human-evaluation directory should contain only final aligned schema/protocol files.
allowed_human = {"README.md", "annotation_protocol.md", "annotation_template.csv"}
actual_human = {p.name for p in (ROOT / "human_evaluation").iterdir() if p.is_file()}
assert actual_human == allowed_human, f"Unexpected human-evaluation files: {sorted(actual_human - allowed_human)}"

# No raw @username strings should appear in CSV release artifacts.
handle_re = re.compile(r"@[A-Za-z0-9_]{2,}")
for csv_path in ROOT.rglob("*.csv"):
    text = csv_path.read_text(encoding="utf-8", errors="ignore")
    assert not handle_re.search(text), f"Raw username-like handle found in {csv_path.relative_to(ROOT)}"

print("Archived runnable cohort manifests are internally consistent.")
print("Bundled cohort manifests do not redistribute raw tweet text.")
print("Paper aggregate feasibility and human-evaluation summaries are present.")
print("Legacy human spreadsheets and redistributed usernames are absent.")
print("Repository checks passed.")
