# Data provenance

## Paper experiment

The final paper reports the following Davidson data counts:

- 600 samples in the initial screening set;
- 566 Toxic-BERT-positive samples;
- 100 calibration samples;
- 466 main evaluation samples;
- 450 feasible final counterfactuals;
- 16 infeasible cases.

These counts and the experimental settings are recorded in `configs/reference.json`. The exact row-level paper split is not included in this repository, so the CSV files under `results/` should be read as the aggregate results reported in the manuscript.

## Archived runnable experiment

`data/archived_run/` contains a different completed experiment:

- 700 screened source indices;
- 609 Toxic-BERT-positive source indices;
- 100 calibration source indices;
- 450 evaluation source indices.

The calibration and evaluation sets are disjoint and both come from the archived screened and Toxic-BERT-positive sets.

This archived experiment is included only to provide runnable data indices for the code. It is not the paper's 600/566/100/466 split.

## Raw tweet text

The archived CSV files omit the Davidson `tweet` column. They retain `dataset_index`, labels, and the experiment metadata needed to identify each sample.

When an archived run is started, `src/data.py` loads the corresponding text from `tdavidson/hate_speech_offensive` using `dataset_index`.

## Human evaluation

The repository contains the final annotation protocol, a blank annotation template, and the aggregate agreement results in `results/human_evaluation_summary.csv`.

The original row-level annotations for the 450 paper counterfactuals are not included. Older human-evaluation spreadsheets from exploratory versions of the work have been removed.
