# Early Warning Model for Air Cargo Delays

## Overview

This repository presents a proof of concept for identifying air cargo shipments that may be at risk of delay. It tests whether free-text handling notes contain useful warning signals and whether combining those notes with structured shipment attributes and Cargo iQ milestone indicators improves predictive performance.

The workflow compares three binary classification models:

- a text-only Logistic Regression baseline;
- a multimodal Random Forest; and
- a multimodal multilayer perceptron (MLP).

The project also introduces **contradiction flags**. These identify cases where a handling note describes an operational issue while the related structured milestone indicators still appear on time and correct. The purpose is to test whether disagreement between structured and unstructured data can help prioritise shipments for earlier review.

This is an analytical proof of concept, not a production delay-prediction system.

## Background and Business Context

Air cargo shipments pass through a series of linked operational milestones, including documentation, acceptance, departure, arrival and final availability. Structured milestone reporting provides consistent performance monitoring, but an operational issue may be described in a handling note before it is reflected in a formal status flag.

An early-warning approach could support exception management by directing limited review capacity towards shipments with a higher apparent risk of delay. The intended use is decision support: the model would complement existing monitoring and human judgement rather than automate operational decisions.

## Research Questions

The analysis addresses three questions:

1. Do handling notes contain a measurable signal associated with shipment delay?
2. Do structured shipment attributes, Cargo iQ indicators and contradiction features improve on a text-only baseline?
3. Does a more complex neural network outperform a simpler tree-based model when both use the same multimodal feature set?

## Dataset

The analysis uses a **synthetic dataset of 50,000 shipment records** created for experimentation. It contains no live operational, customer, employee or commercially sensitive data.

The synthetic data is divided across three input files joined by a unique Master Air Waybill identifier (`MAWB`):

| File | Contents | Reference size |
| --- | --- | ---: |
| `shipment_master_enriched_v2.csv` | Shipment attributes and handling notes | 50,000 × 15 |
| `ciq_flags_corrected_v3.csv` | Cargo iQ milestone status, timeliness and correctness flags | 50,000 × 22 |
| `nfd_milestone_target_file.csv` | Planned and actual final-milestone times and the target label | 50,000 × 5 |

A shipment is labelled late when the actual final notification occurs more than 30 minutes after its planned time. In the reference dataset, 23.6% of shipments are labelled late.

The three source CSV files are not included in this repository. The code can be reviewed without them, but running the complete workflow requires synthetic input files with the filenames and schema listed above. Publishing a data generator or a small representative sample would be required for full end-to-end reproducibility.

## Repository Structure

```text
.
├── README.md
└── early_warning_air_cargo_delays.ipynb
```

The notebook writes its generated CSV, NumPy and PNG outputs to the current working directory. These outputs do not need to be committed to the repository.

## Environment and Installation

Python 3.10 or later is recommended. The main dependencies are:

- pandas
- NumPy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Matplotlib
- SHAP
- Jupyter

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn torch transformers matplotlib shap jupyter
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux, use:

```bash
source .venv/bin/activate
```

The first run downloads the `distilbert-base-uncased` model from Hugging Face. A GPU is helpful for generating embeddings, but the workflow can run on CPU. The MLP SHAP analysis is computationally expensive and may take considerably longer than the model training.

## Quick Start

1. Place the three synthetic CSV files in the same directory as the notebook.
2. Start Jupyter:

   ```bash
   jupyter lab
   ```

3. Open `early_warning_air_cargo_delays.ipynb`.
4. Run the cells sequentially from data validation through SHAP analysis.
5. Review the generated metrics, confusion matrices, ROC curves, prediction files and feature-importance outputs.

The notebook uses fixed random seeds where supported. All learned transformations are fitted on the training data and then applied to the held-out test data.

## Data Preparation and Feature Engineering

The workflow performs the following steps:

- validates file shapes, required columns, duplicate identifiers and key alignment;
- merges the three sources on `MAWB`;
- checks consistency between milestone status, timeliness and correctness fields;
- creates the target using the configurable 30-minute delay threshold;
- removes final-milestone timestamps and flags used to derive the target, reducing direct target leakage;
- cleans handling notes and expands operational abbreviations for rule-based matching;
- creates 13 interpretable issue flags from handling-note text;
- creates nine note-versus-milestone contradiction flags;
- derives note-presence, note-length, word-count and issue-count features;
- calculates the total number of failed Cargo iQ checks; and
- applies a stratified 80/20 train/test split with `random_state=42`.

The test set contains 10,000 records and retains the reference late-shipment rate of approximately 23.6%.

## Text Representation

Handling notes are encoded with the pretrained `distilbert-base-uncased` transformer. The notebook tokenises each note to a maximum length of 128 tokens and mean-pools the final hidden state to produce a 768-dimensional embedding.

Two versions of the note text are retained:

| Text field | Purpose |
| --- | --- |
| `Notes_Clean` | Input to DistilBERT |
| `Notes_Expanded` | Abbreviation-expanded text used by regular-expression issue rules |

## Models

### Text-only Logistic Regression

The baseline uses 781 features: 768 DistilBERT dimensions and 13 note-derived issue flags. Structured shipment attributes, Cargo iQ flags and contradiction flags are deliberately excluded so that the predictive value of the notes can be assessed separately.

Class weighting is used to address the imbalanced target distribution. The model uses the `lbfgs` solver and a maximum of 1,000 iterations.

### Multimodal Random Forest

The Random Forest uses the complete 821-feature matrix:

- 768 DistilBERT dimensions;
- 14 encoded or numeric shipment attributes;
- 12 Cargo iQ milestone flags;
- one Cargo iQ failure aggregate;
- four note-metadata features;
- 13 issue flags; and
- nine contradiction flags.

The reference configuration uses 300 trees, a maximum depth of 15, a minimum leaf size of five and balanced class weights.

### Multimodal MLP

The MLP uses the same 821 features after standardisation. Its hidden layers contain 256, 128 and 64 units with ReLU activation. Training uses early stopping, a batch size of 512, a maximum of 300 iterations and L2 regularisation through `alpha=0.001`.

## Evaluation

The models are evaluated on the same held-out test records using:

- ROC-AUC as the primary discrimination metric;
- precision, recall, F1-score and accuracy;
- confusion matrices;
- combined ROC curves; and
- AUC by shipment group.

The group analysis separates records into:

- **Contradiction:** a note identifies a milestone-linked issue while the corresponding milestone flags remain clean;
- **Cargo iQ-linked, non-contradictory:** the note and milestone indicators both identify a related issue;
- **Unrelated to Cargo iQ:** the note describes an issue without a corresponding milestone mapping; and
- **Neutral or no note:** no mapped issue is identified.

## Reference Results

The following results were produced by the stored reference run on the synthetic held-out test set:

| Model | ROC-AUC | F1 | Precision | Recall | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression, text only | 0.6158 | 0.3962 | 0.7290 | 0.2720 | 0.8040 |
| Random Forest, multimodal | **0.6249** | 0.3959 | 0.7065 | **0.2750** | 0.8016 |
| MLP, multimodal | 0.6240 | **0.3963** | **0.7299** | 0.2720 | **0.8041** |

The best overall multimodal result improved ROC-AUC by 0.0091 over the text-only baseline. The Random Forest slightly outperformed the MLP overall, so the additional neural-network complexity did not provide a meaningful aggregate advantage in this run.

### Group-level ROC-AUC

| Test-set group | Records | Text-only LR | Random Forest | MLP | Best change from baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contradiction | 934 | 0.654 | 0.676 | **0.683** | **+0.029** |
| Cargo iQ-linked, non-contradictory | 1,383 | 0.805 | 0.804 | **0.808** | +0.003 |
| Unrelated to Cargo iQ | 559 | **0.531** | 0.508 | 0.524 | -0.007 |
| Neutral or no note | 7,124 | 0.495 | **0.508** | 0.507 | +0.013 |

Across the full synthetic dataset, 4,487 shipments, or approximately 9.0%, met the contradiction definition. Their late rate was 37.9%, compared with 22.2% for records without a contradiction. This is evidence of an association within the synthetic data, not proof that the same relationship exists in live operations.

The results support a narrow conclusion: combining the data sources added a modest amount of overall discrimination, with the clearest improvement concentrated in the contradiction group. Low recall across all three models means the approach is better suited to prioritised human review than autonomous decision-making.

## Explainability

SHAP is used to examine how different feature groups contribute to the two multimodal models:

- `TreeExplainer` is used for the Random Forest;
- `KernelExplainer` is used for the MLP;
- a stratified sample of 1,000 test records is used for the explanation stage; and
- feature importance is reported both by individual feature and by feature group.

In the reference run, DistilBERT embeddings accounted for most of the measured absolute SHAP contribution in both models: 79.6% for the Random Forest and 93.5% for the MLP. These percentages describe the selected models and SHAP sample only; they should not be interpreted as causal effects.

## Generated Outputs

The notebook creates the following outputs during execution.

### Processed data and predictions

- `preprocessed_data.csv`
- `train_data.csv`
- `test_data.csv`
- `part1_predictions.csv`
- `part2_rf_predictions.csv`
- `part2_mlp_predictions.csv`
- `comparison_results.csv`

### Evaluation figures

- `part1_confusion_matrix.png`
- `part2_rf_confusion_matrix.png`
- `part2_mlp_confusion_matrix.png`
- `confusion_matrices_combined.png`
- `comparison_roc_curves.png`

### SHAP outputs

- `shap_group_importance.png`
- `shap_rf_top_features.png`
- `shap_mlp_top_features.png`
- `shap_rf_beeswarm.png`
- `shap_rf_group_comparison.png`
- `shap_rf_feature_importance.csv`
- `shap_mlp_feature_importance.csv`
- `shap_group_importance.csv`
- `shap_rf_values.npy`
- `shap_mlp_values.npy`

## Limitations and Generalisation Risk

- **Synthetic data:** the relationships, text patterns and class distribution are simulated. Reported metrics demonstrate the workflow, not expected performance in a live logistics network.
- **Data availability:** The source CSVs are not published, so the public repository is not independently executable without compatible synthetic inputs.
- **Temporal uncertainty:** reliable note timestamps are unavailable. A contradiction may represent an early signal, but the workflow cannot prove that every note preceded the related milestone deviation.
- **Moderate discrimination and low recall:** overall ROC-AUC is approximately 0.62 and recall is approximately 0.27 at the default threshold. Most delayed shipments are not detected.
- **Single hold-out split:** the reference results use one stratified random split rather than repeated cross-validation or an out-of-time test.
- **Rule sensitivity:** regex-based issue flags depend on the vocabulary included in the rules and may miss new terminology, spelling variations or context.
- **Categorical encoding:** label encoding is convenient for the proof of concept but can introduce artificial ordering and requires explicit handling of unseen categories.
- **No probability calibration:** probabilities are not calibrated, and the default 0.50 decision threshold is not tied to operational costs.
- **Approximate explainability:** SHAP analysis is performed on a sample, and MLP explanations use an approximate, computationally expensive model-agnostic method.
- **No deployment pipeline:** monitoring, retraining, model persistence, access controls and automated data-quality checks are outside the current scope.

## Future Work

- Publish a synthetic-data generator or a non-sensitive sample dataset with schema documentation.
- Add note timestamps and use an out-of-time evaluation to measure genuine warning lead time.
- Group related records before splitting and check for duplicate or near-duplicate notes across partitions.
- Replace basic label encoding with a pipeline that handles unseen categories safely.
- Compare repeated cross-validation and temporal validation results with confidence intervals.
- Evaluate Precision-Recall AUC alongside ROC-AUC for the imbalanced target.
- Tune and calibrate decision thresholds using the relative cost of unnecessary reviews and missed delays.
- Explore domain-adapted embeddings or a learned issue classifier while retaining interpretable rule-based checks.
- Test the approach on representative operational data before drawing deployment conclusions.
- Add model versioning, drift monitoring, scheduled validation and retraining controls before any production use.

## Responsible Use

This code is intended for research, demonstration and portfolio purposes. Predictions should not be used to make autonomous decisions about shipments, customers, routes, carriers or employees. Any future operational evaluation should include data-governance review, human oversight, performance monitoring and validation for the specific environment in which the model would be used.

## Acknowledgements

The project uses the open-source Python ecosystem, including pandas, NumPy, scikit-learn, PyTorch, Hugging Face Transformers, Matplotlib and SHAP. `distilbert-base-uncased` is used as the pretrained text encoder.
