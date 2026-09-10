# Baseline evaluation

The baseline results reported in the paper are stored in `results/baseline_comparison.csv`.

The scripts in this repository apply the same validity, XAIStrengthNLP, and XAILeakageNLP calculations to candidate texts produced by the baseline methods.

## Fixed-ε DP-MLM

`experiments/fixed_dpmlm_baseline.py` evaluates candidate texts produced by a fixed-ε DP-MLM setup. Candidate generation is external to this repository.

## CusText

`experiments/custext_baseline.py` evaluates candidate texts produced by CusText using the same downstream metrics.

`scripts/fetch_custext.sh` clones the upstream CusText repository and prints the fetched commit hash. The exact historical commit used for the paper baseline is not stored here, so any new baseline run should record the commit used.

The wrappers therefore provide a common evaluation layer; they do not reproduce the complete third-party generation pipelines.
