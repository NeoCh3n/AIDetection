# AIDetection – Unified Threat Detection Pipeline

Supervised threat detection using QRadar rule trigger frequencies. The project trains a Random Forest classifier on labeled attack simulations and reuses the exact same processing path for detection to eliminate training–serving skew.

## Project Goal

- Build a high-recall detector that separates simulated attack windows from normal behavior.
- Share the exact ingestion, aggregation, and feature engineering code between training and detection.
- Produce interpretable outputs (feature importances, SHAP explanations) that security analysts can act on.

## Quick Start

1. Activate the local venv (mandatory): `source venv/bin/activate`
2. Install dependencies (Makefile enforces venv usage): `make install`
3. Run the unified orchestrator from the repo root:
   - Training: `python -m pipeline.main_pipeline train --config pipeline/config.json`
   - Detection: `python -m pipeline.main_pipeline detect --config pipeline/config.json`
4. Optional flags:
   - `--config PATH` to supply an alternate JSON config.
   - `--verbose` for debug-level logging.

All code targets Python 3.6.8; pinned packages are listed in `requirements.txt`.

## Unified Pipeline Overview

```
data_loader.py → feature_aggregator.py → feature_generator.py →
  ├─ train: model_training.train_threat_detector → evaluate_and_report
  └─ detect: model_predictor.Predictor → shap_explainer.Explainer → logging_utils
```

- `pipeline/main_pipeline.py` (`UnifiedPipeline`) is the single entry point.
- Modules are shared verbatim between training and detection to eliminate skew.
- Aggregation output uses the `aggregated_rules` column and mirrors it to `aggregated_rules_dict` for backward compatibility.

## Module Highlights

- `pipeline/data_loader.py` – unified ingestion:
  - `mode='train'`: reads CSVs from `Training_data/{normal,attack}` per paths in config.
  - `mode='detect'`: queries MongoDB through `mongodb/mongodb_connection.py` for the last N minutes.
  - Standardizes rows to `['hostname','rule_id','timestamp','count','source_label']` using `shared_utils/time_utils.parse_qradar_timestamp`.
  - Coerces types (`rule_id`/`count` ints, `hostname` str) and applies basic NA handling.

- `pipeline/feature_aggregator.py` – 30-minute window aggregation:
  - Applies `np.log1p(count)` before grouping; stores raw integer totals in `aggregated_rules`.
  - Adds helper metrics (total events, unique rules, window boundaries, optional `is_attack` in train mode).
  - Provides both `aggregated_rules` and `aggregated_rules_dict` keys for downstream consumers.

- `pipeline/feature_generator.py` – dense feature vectors:
  - Uses `shared_utils/qradar_rule_manager.py` to obtain the ordered rule universe defined by the latest mapping.
  - Generates consistent feature matrices (X) and labels (y, only in train mode).

- `model_training/model_training.py` – training + evaluation helpers:
  - `train_threat_detector(...)` loads data via the pipeline, stratified splits, trains `RandomForestClassifier`.
  - `evaluate_and_report(...)` (in the same module) outputs confusion matrix, classification report, ROC AUC, feature importances.

- `model_predictor.py` – inference wrapper for detection mode (`Predictor` class).
- `system/shap_explainer.py` – SHAP explanations for malicious predictions.
- `system/logging_utils.py` – central logging (daily rotating files in `running_log/` + stdout).

## Training Workflow

1. Place CSVs:
   - Normal activity in `Training_data/normal/`
   - Simulated attack windows in `Training_data/attack/`
2. Update `pipeline/config.json` if custom paths or parameters are needed.
3. Run `python -m pipeline.main_pipeline train --config pipeline/config.json`
4. Outputs (default paths):
   - Model artifact: `model/threat_detector.joblib`
   - Reports: `model/threat_detector_evaluation_report.json`, `model/threat_detector_top_20_features.csv`
5. Random Forest defaults (configurable):
   - `n_estimators=200`, `class_weight='balanced_subsample'`, `max_features='sqrt'`, `random_state=42`, `n_jobs=-1`
6. Stratified train/test split ensures attack prevalence is preserved across splits.

## Detection Workflow

1. Upstream QRadar jobs (see “MongoDB & QRadar Ingestion Jobs”) insert rule counts into MongoDB.
2. Run `python -m pipeline.main_pipeline detect --config pipeline/config.json`
3. Pipeline generates features, loads the trained model, and returns predictions and probabilities.
4. Alerts trigger when the probability exceeds `detection.alert_threshold` from the config.
5. Malicious predictions receive SHAP explanations (Top-N contributing rule IDs) and are logged via `system/logging_utils`.

## Model Evaluation Deliverables

- Confusion matrix (TP/FP/FN/TN).
- Classification report (precision, recall, F1 – recall for label `1` is the primary KPI).
- ROC AUC and Average Precision (configurable).
- Top 20 most important QRadar rules (feature importances) exported to CSV.

## MongoDB & QRadar Ingestion Jobs

- `mongodb/mongodb_connection.py` – central manager reused across ingestion jobs and the pipeline.
- `mongodb/insert_DB.py` (`AQLDataInserter`) – loads AQL JSON results into MongoDB for detection tests.
- `api_integration/create_searches_Qradar.py`, `status_searches_Qradar.py`, `result_searches_Qradar.py`, `delete_searches_Qradar.py` – schedule and manage QRadar searches.
- `mongodb/delete_DB.py` – retention cleanup (daily or on demand).
- `mongodb/setup_mongodb_offline.py` – bootstrap collections/indexes for local development.
- Configurations:
  - Pipeline: `pipeline/config.json`
  - MongoDB/AQL inserter: `mongodb/mongodb_config.json`

## Project Layout (selected)

```
AIDetection/
├── pipeline/                 # Unified pipeline modules and config
├── model_training/           # Training + evaluation helpers
├── model_predictor.py        # Detection-time predictor wrapper
├── system/                   # Logging + SHAP explainers
├── mongodb/                  # MongoDB utilities for detection ingestion
├── api_integration/          # QRadar search orchestration scripts
├── shared_utils/             # Time utilities and rule manager
├── Training_data/            # Normal/attack CSV inputs
├── model/                    # Generated model artifacts and reports
├── tests/                    # Pytest suite covering pipeline components
├── Makefile                  # `make install` / `make test` (enforces venv)
└── requirements.txt          # Pinned Python dependencies
```

Additional directories of interest:
- `Qradar_rule/` – rule definitions used by `QRadarRuleManager` in file mode.
- `Multi_model/` – experimental OOP pipeline prototypes (not part of the unified flow).
- `test_output/` – generated plots/reports from exploratory runs.
- `running_log/` – daily log files written by `logging_utils`.

## Testing & QA

- Run the complete suite: `make test`
- Frequently targeted tests:
  - `tests/test_data_loader_detailed.py`
  - `tests/test_feature_aggregator.py`
  - `tests/test_pipeline_simple.py`
  - `tests/test_model_evaluation.py`
- Tests expect the local venv to be active and dependencies installed via `make install`.

## Data Schema Reference

- Loader output: `hostname (str)`, `rule_id (int)`, `timestamp (datetime)`, `count (int)`, `source_label (str)`
- Aggregated windows: `window_id`, `hostname`, `aggregated_rules` (dict of `{rule_id: total_count}`), mirrored `aggregated_rules_dict`, `total_events`, `unique_rules`, `window_start`, `window_end`, optional `is_attack`

## Security Notes

- Training uses internally generated attack simulations; treat all data as internal-only.
- Follow enterprise data handling policies when moving CSVs or MongoDB dumps.
- Unified preprocessing and shared code paths prevent training–serving drift.

## Troubleshooting

- Python or dependency mismatch: recreate the venv and rerun `make install`.
- MongoDB connectivity: verify `mongodb://localhost:27017/` (or configured URI) is reachable.
- Timestamp parsing errors: confirm QRadar timestamps match `parse_qradar_timestamp` expectations.
- Model loading issues: ensure the joblib model was produced with the pinned scikit-learn version.

---

Built with Python 3.6.8, MongoDB, and scikit-learn (Random Forest).
