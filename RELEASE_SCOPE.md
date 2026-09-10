# Release scope

This repository provides the implementation and aggregate results for the final paper.

## Included

The code follows the paper definitions of:

- XAIStrengthNLP;
- XAILeakageNLP;
- Toxic-BERT toxic-to-non-toxic validity;
- sensitive-span detection and contextual checking;
- targeted DP-MLM-inspired rewriting; and
- final selection under the XAILeakageNLP limit.

The result CSV files under `results/` contain the aggregate values reported in the manuscript.

## Not included

The exact row-level manifests for the paper's 600/566/100/466 data split are not included. The original row-level annotations from the two human annotators are also not included.

Because of this, the repository is not an exact row-by-row reproduction package.

## Archived runnable cohort

A separate archived experiment is included under `data/archived_run/` so the pipeline can be executed. It contains 700 screened samples, 609 Toxic-BERT-positive samples, 100 calibration samples, and 450 evaluation samples.

These data belong to a different archived run and should not be presented as the paper's experimental split.

## Raw text

The bundled data files do not contain Davidson tweet text. Source text is loaded from the upstream dataset at runtime using `dataset_index`, so raw usernames from the original dataset are not copied into this repository.
