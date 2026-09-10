# Data manifests

Raw Davidson tweet text is not included in this repository.

`archived_run/` contains dataset-index manifests from a separate archived experiment:

- `screening_pool_700.csv`: 700 screened samples;
- `toxic_bert_positive_609.csv`: 609 Toxic-BERT-positive samples;
- `calibration_100.csv`: 100 calibration samples;
- `evaluation_450.csv`: 450 evaluation samples.

The files contain `dataset_index` and experiment metadata but omit the raw `tweet` field. When the archived example is run, the text is loaded from `tdavidson/hate_speech_offensive` using `dataset_index`.

The calibration and evaluation sets are disjoint and both are drawn from the archived screened and Toxic-BERT-positive sets.

These manifests are for the archived 700/609/100/450 experiment, not the paper's 600/566/100/466 split. See `DATA_PROVENANCE.md`.
