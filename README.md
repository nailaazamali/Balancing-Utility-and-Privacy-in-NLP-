# Balancing Utility and Privacy in NLP

This repository contains the implementation and reported experimental results for our work on privacy-aware counterfactual explanations in NLP.

The study examines a practical problem in counterfactual explanations: an explanation can improve the prediction outcome while also introducing or revealing sensitive information that was not present in the original text. Our framework therefore evaluates both the **usefulness of a counterfactual explanation** and the **privacy leakage it may introduce**.

The repository accompanies the paper:

**“Balancing Utility and Privacy in NLP Counterfactual Explanations: An ε_token-Optimised DP-MLM-Inspired Framework”**

## What this work does

The framework generates counterfactual text and evaluates it using two main measures:

- **XAIStrength** measures how useful the counterfactual explanation is. It combines the prediction change, semantic similarity to the original text, and the amount of textual modification.
- **XAILeakage** measures whether the counterfactual introduces new sensitive information and whether that information is unsupported by the original text.

Sensitive information is detected using a combination of pattern matching, lexical categories, named-entity recognition, and contextual validation. Newly introduced sensitive content can then be rewritten using a **DP-MLM-inspired sanitisation step**. Candidate explanations are checked for validity and privacy constraints before the final explanation is selected.

The main experiments use the Davidson hate-speech/offensive-language dataset with **Toxic-BERT** as the classifier. The paper reports **466 evaluation instances**, of which **450 produced feasible counterfactual explanations** and **16 were infeasible** under the final constraints.

## Main result

The proposed optimisation reduced the average **XAILeakage from 0.436 to 0.135**, corresponding to about a **69% reduction in leakage**. Over the same evaluation, average **XAIStrength changed from 0.689 to 0.589**, representing a **14.5% reduction in explanation strength**.

These results show the trade-off studied in the paper: privacy leakage can be reduced substantially while retaining useful counterfactual explanations.

## Repository contents

The folders in this repository contain the implementation of the main parts of the framework, including counterfactual generation, sensitive-information detection, XAIStrength and XAILeakage calculation, DP-MLM-inspired sanitisation, candidate validation and selection, experiment scripts, and configuration files.

The following result files are included at the top level for quick access:

### `main_summary.csv`

Contains the main before-and-after results reported in the paper, including:

- initial and optimised XAIStrength;
- initial and optimised XAILeakage;
- percentage reduction in leakage; and
- percentage change in explanation strength.

This file summarises the main utility–privacy result of the proposed framework.

### `feasibility_summary.csv`

Reports the number of evaluation cases for which the framework found a valid explanation satisfying the required constraints.

For the main experiment:

- total evaluation instances: **466**;
- feasible counterfactuals: **450**;
- infeasible cases: **16**.

### `fixed_epsilon_ablation.csv`

Contains the fixed-ε analysis used to study how different privacy settings affect explanation strength and leakage.

The evaluated ε values are:

`0.25, 0.50, 1.00, 2.00, 4.00`

This experiment helps show how the privacy setting changes the utility–privacy trade-off.

### `baseline_comparison.csv`

Contains the comparison between the proposed framework and the two privacy-oriented baselines used in the paper:

- **Fixed-ε DP-MLM**
- **CusText**
- **Proposed framework**

The comparison includes feasibility and the main explanation-quality and privacy measures.

### `robustness_summary.csv`

Contains the robustness results reported across different dataset and generator combinations.

The paper evaluates combinations involving:

- Davidson + Qwen;
- Davidson + Gemma;
- TweetEval + Qwen; and
- TweetEval + Gemma.

This checks whether the observed utility–privacy behaviour is limited to one model or dataset.

### `human_evaluation_summary.csv`

Contains the aggregate human-evaluation results reported in the paper.

The final feasible counterfactuals were manually reviewed for sensitive-information behaviour and explanation quality. The paper reports agreement on **412 of 450 cases (91.6%)**, with **Cohen’s κ = 0.831**.

### `archived_run_data_summary.csv`

Summarises the separate archived runnable cohort included with the repository.

This archived run is provided to demonstrate the implementation and workflow. It is **not the exact source-level split used to produce the main paper results**, so it should not be interpreted as a row-by-row reproduction of the 466-case paper experiment.

## Scope of this repository

This repository should be read as a **paper-aligned implementation and aggregate-results release**.

The code follows the methodology described in the paper, and the included summary CSV files record the aggregate results reported in the experiments. The exact source-level split used for every paper experiment is not included, so the repository does not claim exact row-by-row reproduction of all reported results.

## Main components implemented

The repository includes code for:

- counterfactual candidate generation;
- Toxic-BERT prediction scoring;
- semantic similarity using SBERT;
- textual-distance and minimality calculation;
- XAIStrength calculation;
- sensitive-information detection;
- source-relative sensitive-information comparison;
- unsupported-sensitive-content estimation;
- XAILeakage calculation;
- DP-MLM-inspired sensitive-span rewriting;
- validity and privacy-constraint checks; and
- final counterfactual selection.

## Citation

If you use this implementation or the reported results, please cite the associated paper:

**Naila Azam, Vasilis Efthymiou, and Georgios Loukas.  
“Balancing Utility and Privacy in NLP Counterfactual Explanations: An ε_token-Optimised DP-MLM-Inspired Framework.”**

