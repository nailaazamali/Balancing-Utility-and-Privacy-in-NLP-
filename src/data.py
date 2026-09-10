from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset


LABEL_MAP = {0: "hate speech", 1: "offensive language", 2: "neither"}


def load_davidson(dataset_name: str, split: str = "train") -> pd.DataFrame:
    frame = load_dataset(dataset_name, split=split).to_pandas()
    frame = frame.copy()
    frame["dataset_index"] = frame.index.astype(int)
    frame["label_name"] = frame["class"].map(LABEL_MAP)
    return frame


def load_index_manifest(path: str | Path, expected_n: int) -> list[int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    frame = pd.read_csv(path)
    if "dataset_index" not in frame.columns:
        raise ValueError(f"{path} must contain a dataset_index column")
    indices = frame["dataset_index"].dropna().astype(int).tolist()
    if len(indices) != expected_n:
        raise ValueError(
            f"{path} contains {len(indices)} indices; expected {expected_n}."
        )
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate indices found in {path}")
    return indices


def select_by_manifest(frame: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    lookup = frame.set_index("dataset_index", drop=False)
    missing = [idx for idx in indices if idx not in lookup.index]
    if missing:
        raise ValueError(f"Dataset indices not found: {missing[:10]}")
    return lookup.loc[indices].reset_index(drop=True).copy()


def load_reference_splits(frame: pd.DataFrame, split_dir: str | Path, config):
    split_dir = Path(split_dir)
    screening_idx = load_index_manifest(
        split_dir / "screening_indices.csv", config.screening_pool_n
    )
    calibration_idx = load_index_manifest(
        split_dir / "calibration_indices.csv", config.calibration_n
    )
    evaluation_idx = load_index_manifest(
        split_dir / "evaluation_indices.csv", config.main_eval_n
    )

    screening = select_by_manifest(frame, screening_idx)
    calibration = select_by_manifest(frame, calibration_idx)
    evaluation = select_by_manifest(frame, evaluation_idx)

    if set(calibration_idx) & set(evaluation_idx):
        raise ValueError("Calibration and evaluation manifests overlap.")
    if not set(calibration_idx).issubset(set(screening_idx)):
        raise ValueError("Calibration indices must belong to the screening cohort.")
    if not set(evaluation_idx).issubset(set(screening_idx)):
        raise ValueError("Evaluation indices must belong to the screening cohort.")

    return screening, calibration, evaluation


def load_cohort_csv(path: str | Path, expected_n: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing cohort file: {path}")
    frame = pd.read_csv(path)
    if "dataset_index" not in frame.columns:
        raise ValueError(f"{path} must contain a dataset_index column")
    if frame["dataset_index"].duplicated().any():
        raise ValueError(f"Duplicate dataset_index values found in {path}")
    if expected_n is not None and len(frame) != int(expected_n):
        raise ValueError(f"{path} contains {len(frame)} rows; expected {expected_n}.")
    return frame.copy()


def _hydrate_text_from_upstream(
    frame: pd.DataFrame, upstream: pd.DataFrame
) -> pd.DataFrame:
    """Attach tweet text by dataset_index without redistributing it in local manifests."""
    if "tweet" in frame.columns:
        return frame.copy()

    lookup = upstream[["dataset_index", "tweet"]].copy()
    hydrated = frame.merge(lookup, on="dataset_index", how="left", validate="one_to_one")
    if hydrated["tweet"].isna().any():
        missing = hydrated.loc[hydrated["tweet"].isna(), "dataset_index"].astype(int).tolist()
        raise ValueError(f"Could not hydrate upstream tweet text for indices: {missing[:10]}")
    return hydrated


def load_archived_cohorts(config, root_dir: str | Path = "."):
    if not hasattr(config, "cohort_files"):
        raise ValueError("The selected configuration does not define cohort_files.")

    base = Path(root_dir)
    files = config.cohort_files
    screening = load_cohort_csv(base / files["screening"], config.screening_pool_n)
    positive_all = load_cohort_csv(base / files["positive_all"], config.toxic_positive_n)
    calibration = load_cohort_csv(base / files["calibration"], config.calibration_n)
    evaluation = load_cohort_csv(base / files["evaluation"], config.main_eval_n)

    screening_idx = set(screening["dataset_index"].astype(int))
    positive_idx = set(positive_all["dataset_index"].astype(int))
    calibration_idx = set(calibration["dataset_index"].astype(int))
    evaluation_idx = set(evaluation["dataset_index"].astype(int))

    if calibration_idx & evaluation_idx:
        raise ValueError("Calibration and evaluation cohorts overlap.")
    if not calibration_idx.issubset(screening_idx):
        raise ValueError("Calibration rows must belong to the screening cohort.")
    if not evaluation_idx.issubset(screening_idx):
        raise ValueError("Evaluation rows must belong to the screening cohort.")
    if not calibration_idx.issubset(positive_idx):
        raise ValueError("Calibration rows must belong to the Toxic-BERT-positive cohort.")
    if not evaluation_idx.issubset(positive_idx):
        raise ValueError("Evaluation rows must belong to the Toxic-BERT-positive cohort.")

    # Public release manifests intentionally omit raw tweet text/usernames.
    # Hydrate text from the official upstream dataset only when a run needs it.
    if any("tweet" not in frame.columns for frame in (screening, positive_all, calibration, evaluation)):
        upstream = load_davidson(config.dataset, config.dataset_split)
        screening = _hydrate_text_from_upstream(screening, upstream)
        positive_all = _hydrate_text_from_upstream(positive_all, upstream)
        calibration = _hydrate_text_from_upstream(calibration, upstream)
        evaluation = _hydrate_text_from_upstream(evaluation, upstream)

    return screening, positive_all, calibration, evaluation
