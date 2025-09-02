# AIDetection – Unified Threat Detection Pipeline

Supervised threat detection using QRadar rule trigger frequencies. The project trains a Random Forest classifier on Picus BAS–labeled data and reuses the exact same processing path for detection to eliminate training–serving skew.

## How To Use

- Activate venv (required): `source venv/bin/activate`
- Install dependencies: `make install` (preferred) or `pip install -r requirements.txt`
- Run training (from repo root):
  - `python -m pipeline.main_pipeline train`
  - or `python ./pipeline/main_pipeline.py train`
- Run detection:
  - `python -m pipeline.main_pipeline detect`
  - or `python ./pipeline/main_pipeline.py detect`
- Options:
  - `--config PATH` to use a custom JSON config (default: `pipeline/config.json`)
  - `--verbose` to enable more verbose logging

## Overview

- Primary model: `RandomForestClassifier` (sklearn) on tabular features (2898 QRadar Rule IDs)
- Windowing: 30-minute aggregated windows; event-level `log1p(count)` before aggregation
- Python 3.6.8 compatible; pinned dependencies for reproducibility
- Single orchestrator: `pipeline/main_pipeline.py` with `--mode train|detect`

## Architecture

Data path (both modes share modules):

```
data_loader.py → feature_aggregator.py → feature_generator.py →
  ├─ train: model_training (save .joblib, evaluate)
  └─ detect: model_predictor (inference, optional SHAP explain)
```

Key modules:
- `pipeline/data_loader.py`: Unified ingestion
  - train: CSVs from `Training_data/normal` and `Training_data/attack`
  - detect: MongoDB via `mongodb/mongodb_connection.py` (last N minutes)
  - Output schema: `['hostname','rule_id','timestamp','count','source_label']`
- `pipeline/feature_aggregator.py`: 30-min windows, applies `np.log1p(count)` pre-aggregation, outputs `aggregated_rules` (alias `aggregated_rules_dict`).
- `pipeline/feature_generator.py`: Dense vectors from rule IDs using `shared_utils/qradar_rule_manager.py`.
- `model_training/model_training.py`: `train_threat_detector(...)` and `evaluate_and_report(...)`.
- `model_predictor.py`: Loads `.joblib` and returns `(label, probability)` per row.
- `pipeline/main_pipeline.py`: UnifiedPipeline orchestrator (train/detect).

MongoDB & QRadar (detection ingestion):
- `mongodb/insert_DB.py` (`AQLDataInserter`): Insert AQL JSON results as detection windows.
- `mongodb/mongodb_connection.py`: Centralized manager used by `data_loader.py`.

## Environment Setup

Always use the local venv via the Makefile.

```bash
# Verify Python version (expect 3.6.8)
python --version

# Create venv and install pinned deps
make install

# Run lightweight tests
make test
```

Pinned dependencies (Python 3.6.8): pandas==1.1.5, numpy==1.19.5, scikit-learn==0.24.2, joblib==1.0.1, pymongo==3.11.0, matplotlib==3.3.4.

## Training

Prepare CSVs:
- Place normal data under `Training_data/normal/`
- Place Picus attack data under `Training_data/attack/`

Run training (saves model and evaluation):

```bash
venv/bin/python pipeline/main_pipeline.py train --config pipeline/config.json
```

Outputs:
- Model: `model/threat_detector.joblib`
- Reports: `model/threat_detector_evaluation_report.json`, `model/threat_detector_top_20_features.csv`

Evaluation highlights:
- Confusion matrix, classification report (precision/recall/F1 for class 1), ROC AUC
- Top 20 most important Rule IDs (feature importances)

## Detection

Online path (scheduled): QRadar API → `mongodb/insert_DB.py` → MongoDB → unified pipeline → predictions.

Run detection pipeline:

```bash
venv/bin/python pipeline/main_pipeline.py detect --config pipeline/config.json
```

Offline AQL JSON insertion (optional for testing):

```bash
# Insert AQL JSON results into MongoDB
venv/bin/python mongodb/insert_DB.py AQLjsonResult.json

# Then run detection
venv/bin/python pipeline/main_pipeline.py detect --config pipeline/config.json
```

Predictions return alerts when probability exceeds the configured threshold (`detection.alert_threshold`).

## MongoDB Utilities

- `mongodb/mongodb_connection.py`: Central connection manager used by the pipeline and data loader.
  - Key methods: `connect()`, `create_indexes()`, `get_unlabeled_windows()`, `get_events_for_window()`, `insert_window()`, `insert_prediction()`, `cleanup_old_data()`.
- `mongodb/insert_DB.py`: AQL JSON → detection windows inserter.
  - Class: `AQLDataInserter`; CLI: `venv/bin/python mongodb/insert_DB.py AQLjsonResult.json`.
- `mongodb/query_DB.py`: Detection-only query helpers for windows/results/events.
  - Class: `AQLQueryManager`; CLI available via `--help`.
- `mongodb/delete_DB.py`: Detection-only cleanup utilities (apply retention).
  - Class: `DetectionDataCleanup`.
- `mongodb/get_DB.py`: Thin helpers to fetch DB/manager for AQL flows.
  - Functions: `get_database()`, `get_mongodb_manager()`, `get_aql_collections()`.
- `mongodb/setup_mongodb_offline.py`: Bootstrap MongoDB locally (collections, indexes, validation).
- `mongodb/mongodb_config.json`: Connection and collection names for detection mode.

## Configuration

- Pipeline config: `pipeline/config.json`
  - `training.model_path`, `training.test_size`, `training.random_state`
  - `detection.qradar_config` (AQL), `detection.mongodb_config`, `alert_threshold`, `window_size_minutes`
  - `rule_manager`: rule source (`file` or `api`)
- MongoDB config (AQL inserter): `mongodb/mongodb_config.json`

## Data Schema

Unified loader output (row-level):

```
hostname (str), rule_id (int), timestamp (datetime), count (int), source_label (str)
```

Aggregated windows (feature_aggregator):
- Columns: `window_id`, `hostname`, `aggregated_rules` (dict of `{rule_id: value}`), optional label in train mode.
- Note: `np.log1p(count)` is applied before aggregation in both modes.

## Project Structure

```
AIDetection/
├── pipeline/
│   ├── main_pipeline.py        # Orchestrator (train/detect)
│   ├── data_loader.py          # Unified ingestion
│   ├── feature_aggregator.py   # 30-min aggregation + log1p
│   └── feature_generator.py    # Dense vectors via rule manager
├── model_training/
│   ├── model_training.py       # Train + evaluate + reports
│   └── model_evaluation.py     # (helpers/standalone evaluation)
├── model_predictor.py          # Inference wrapper
├── mongodb/
│   ├── mongodb_connection.py   # Central manager (connect/query/insert/cleanup)
│   ├── insert_DB.py            # AQL JSON inserter (AQLDataInserter)
│   ├── query_DB.py             # Query helpers for detection windows/events (AQLQueryManager)
│   ├── delete_DB.py            # Cleanup utilities (retention policies)
│   ├── get_DB.py               # Thin DB/manager accessor for AQL flows
│   ├── setup_mongodb_offline.py# Offline setup: create collections/indexes
│   └── mongodb_config.json     # MongoDB config for detection mode
├── shared_utils/
│   ├── time_utils.py           # Timestamp parsing/window ids
│   └── qradar_rule_manager.py  # Rule list + index mapping
├── Training_data/              # CSVs: normal/ and attack/
├── Qradar_rule/                # Rule definitions (file mode)
├── Makefile                    # venv + tests
└── requirements.txt            # Pinned deps (Py 3.6.8)
```

## Testing

Run the lightweight suite:

```bash
make test
```

Common focused tests:
- `tests/test_data_loader_detailed.py`
- `tests/test_feature_aggregator.py`
- `tests/test_pipeline_simple.py`

## Security Notes

- Uses simulated Picus BAS attack data for training
- Internal data only; follow enterprise data handling policies
- Unified preprocessing prevents training–serving skew

## Troubleshooting

- MongoDB connection: ensure local instance reachable (`mongodb://localhost:27017/`)
- Imports: keep `mongodb/__init__.py` to avoid namespace/package import issues
- Version mismatch: ensure pinned sklearn/joblib when loading models
- Timestamp parsing: see `shared_utils/time_utils.parse_qradar_timestamp`

---

Built with Python 3.6.8, MongoDB, and scikit-learn (Random Forest).
