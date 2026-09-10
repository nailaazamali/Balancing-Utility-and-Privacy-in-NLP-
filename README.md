# XAILeakageNLP

Code and aggregate results for the paper:

> *Balancing Utility and Privacy in NLP Counterfactual Explanations: An ε_token-Optimised DP-MLM-Inspired Framework*

The repository contains the implementation of XAIStrengthNLP, XAILeakageNLP, sensitive-span detection, targeted DP-MLM-inspired sanitisation, and the final counterfactual selection procedure used in the paper.

## Repository scope

The code and aggregate result tables follow the final paper. The exact row-level split used for the paper experiments is not included, so this repository should be treated as an **implementation and aggregate-results release**, not as an exact end-to-end reproduction package.

The paper reports:

- 600 Davidson samples in the initial screening set;
- 566 samples classified as toxic by Toxic-BERT;
- 100 calibration samples;
- 466 main evaluation samples;
- 450 feasible final counterfactuals and 16 infeasible cases.

`configs/reference.json` records these paper settings and `results/` contains the reported aggregate values.

A separate archived cohort is included so that the pipeline can still be run. It uses 700 screened samples, 609 Toxic-BERT-positive samples, 100 calibration samples, and 450 evaluation samples. This archived cohort is **not** the paper split.

See `RELEASE_SCOPE.md` and `DATA_PROVENANCE.md` for details.

## Main method

The repository implements four main parts of the framework:

- **XAIStrengthNLP**: combines positive classifier change, semantic similarity, and textual minimality.
- **XAILeakageNLP**: measures newly introduced sensitive information and how unsupported that information is by the source text.
- **DP-MLM-inspired sanitisation**: rewrites newly introduced sensitive spans using masked-language-model candidates and ε-controlled sampling.
- **Counterfactual selection**: selects the strongest valid toxic-to-non-toxic counterfactual that satisfies the maximum leakage threshold.

## Main settings

| Setting | Value |
| --- | --- |
| Dataset | `tdavidson/hate_speech_offensive` |
| Toxicity classifier | `unitary/toxic-bert` |
| Primary generator | `Qwen/Qwen2.5-0.5B-Instruct` |
| Sentence encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Sensitive-context model | `facebook/bart-large-mnli` |
| Unsupportedness NLI model | `cross-encoder/nli-deberta-v3-base` |
| Masked LM | `roberta-base` |
| ε_token grid | 0.25, 0.50, 1.0, 2.0, 4.0 |
| Rewrite fractions | 0.25, 0.50, 0.75, 1.0 |
| Sensitive-context threshold | 0.60 |
| Semantic-similarity threshold | 0.60 |
| Maximum normalised token distance | 0.80 |
| Preferred leakage target | 0.15 |
| Maximum leakage | 0.30 |
| Seed | 42 |

## Data included in the repository

Raw Davidson tweet text is not stored in the bundled CSV files. The archived manifests contain dataset indices and experiment metadata only. When the archived example is run, `src/data.py` loads the source text from the upstream Davidson dataset using `dataset_index`.

This avoids republishing usernames from the original dataset in this repository. The upstream dataset contains offensive and hateful language and should be handled accordingly.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Check the repository

```bash
python scripts/validate_repository.py
```

The validation script checks the bundled cohort structure, the paper aggregate summaries, and the absence of raw tweet text, cached Python files, and old human-evaluation spreadsheets.

## Run the archived example

```bash
python experiments/run_main.py \
  --config configs/archived_run.json \
  --device cpu \
  --output-dir outputs/archived_demo
```

For a short pipeline check:

```bash
python experiments/run_main.py --limit 2 --device cpu
```

The notebook `notebooks/run_archived_example.ipynb` uses the same archived configuration.

## Fixed-ε analysis

After running the archived example:

```bash
python scripts/merge_candidates.py outputs/archived_demo/candidate_pools
python experiments/fixed_epsilon.py outputs/archived_demo/all_candidates.csv.gz
```

The fixed-ε values reported in the paper are stored in `results/fixed_epsilon_ablation.csv`.

## Baselines

The baseline scripts evaluate externally generated candidate texts using the same validity, XAIStrengthNLP, and XAILeakageNLP calculations used for the proposed method. Baseline candidate generation itself is not reproduced here.

See `baselines/README.md` for the fixed-ε DP-MLM and CusText setup.

## Robustness experiments

The reported Davidson/TweetEval and Qwen/Gemma aggregate results are stored in `results/robustness_summary.csv`. `experiments/robustness.py` can be used to summarise compatible new result files.

## Human evaluation

The paper reports a two-annotator review of 450 feasible counterfactuals. The two annotators initially agreed on 412 cases (91.6%), with Cohen's κ = 0.831. The 38 disagreements were resolved through discussion.

This repository includes the annotation protocol, a blank annotation template, and the aggregate agreement statistics. The original row-level annotations are not included. Older exploratory human-evaluation spreadsheets have been removed because they used earlier versions of the metrics.

## Reported results

The main aggregate values stored under `results/` are:

- XAILeakageNLP: 0.436 → 0.135;
- XAIStrengthNLP: 0.689 → 0.589;
- feasible final counterfactuals: 450 / 466;
- fixed-ε results;
- fixed-ε DP-MLM and CusText comparison;
- robustness results across dataset-generator combinations; and
- human-evaluation agreement statistics.

## Repository structure

```text
configs/            paper settings and archived-run configuration
data/               archived dataset-index manifests
experiments/        main, baseline, fixed-epsilon, and robustness scripts
human_evaluation/   annotation protocol and blank annotation template
notebooks/          archived example notebook
results/            aggregate results reported in the paper
scripts/            validation and utility scripts
src/                implementation of the framework
```

## Licence and citation

See `LICENSE` for the repository reuse terms and `CITATION.cff` for citation information. External datasets, models, and baseline implementations remain subject to their own licences and terms.
