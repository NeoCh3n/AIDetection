"""
Model Training - Pipeline Integration

Quick Guide
- Use local venv only: `source venv/bin/activate` then `make install`.
- Preferred run (via orchestrator):
  - Train: `python -m pipeline.main_pipeline train --config pipeline/config.json`
  - Detect: `python -m pipeline.main_pipeline detect --config pipeline/config.json`
- Direct CLI (this file):
  - `python model_training/model_training.py --config pipeline/config.json --evaluate`

Enabling GridSearchCV
- Add a `grid_search` block under the `training` section of your config, or pass it in
  the `training_config` dict to `train_threat_detector(...)`.
- Example (pipeline/config.json):
  {
    "training": {
      "model_path": "./model/threat_detector.joblib",
      "test_size": 0.2,
      "random_state": 42,
      "grid_search": {
        "enabled": true,
        "scoring": "roc_auc",
        "cv": 3,
        "verbose": 1,
        "param_grid": {
          "n_estimators": [200, 400],
          "max_depth": [null, 20, 40],
          "max_features": ["sqrt", "log2"],
          "min_samples_split": [2, 5],
          "min_samples_leaf": [1, 2],
          "class_weight": ["balanced_subsample"]
        }
      }
    }
  }

Notes
- Python 3.6.8 and scikit-learn 0.24.2 required (see requirements.txt).
- Data loader defaults to `./Training_data/normal` and `./Training_data/attack` for CSVs.
- RandomForest default: n_estimators=200, class_weight='balanced_subsample', max_features='sqrt', n_jobs=-1.
"""

import sys
import os
import json
import logging
import argparse
import math
from typing import Optional, Dict, Tuple, cast
import joblib
import numpy as np

# Add parent directories to path for pipeline imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'system'))
from system import logging_utils

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

# Import pipeline modules
from pipeline.data_loader import load_data
from pipeline.feature_aggregator import aggregate_to_windows
from pipeline.feature_generator import FeatureGenerator
from shared_utils.qradar_rule_manager import QRadarRuleManager

# Configure logging: route all logs to running_log/YYYY-MM-DD.log
try:
    logging_utils.setup_global_daily_file_logging(level=logging.INFO, include_stdout=True)
except Exception:
    pass
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _safe_extract_metric(cvres: Dict, key: str, index: int) -> Optional[float]:
    """Safely extract a float metric from GridSearchCV results."""
    if not isinstance(cvres, dict) or key not in cvres:
        return None
    try:
        values = cvres[key]
        value = values[index]
    except (TypeError, KeyError, IndexError):
        return None
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_float):
        return None
    return value_float


# --- Lightweight joblib progress bar (no extra dependencies) ---
from contextlib import contextmanager
import time


@contextmanager
def joblib_progress_bar(total: int, desc: str = "GridSearchCV", width: int = 30):
    """
    Render a simple in-place text progress bar for joblib-backed tasks.

    Parameters
    ----------
    total : int
        Total number of tasks (e.g., n_candidates * cv_splits).
    desc : str
        Description prefix for the progress bar.
    width : int
        Width of the progress bar in characters.
    """
    import joblib as _joblib

    start_time = time.time()
    completed = [0]

    def _render(final: bool = False):
        done = min(completed[0], total)
        pct = (float(done) / total) if total else 1.0
        filled = int(width * pct)
        bar = f"[{'#' * filled}{'-' * (width - filled)}]"
        elapsed = time.time() - start_time
        rate = (done / elapsed) if elapsed > 0 else 0.0
        msg = f"\r{desc} {bar} {done}/{total} ({pct*100:5.1f}%) | {rate:.1f} it/s"
        try:
            sys.stdout.write(msg)
            if final:
                sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass

    class _PBCallback(_joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            try:
                completed[0] += self.batch_size
                _render()
            except Exception:
                pass
            return super(_PBCallback, self).__call__(*args, **kwargs)

    old_cb = _joblib.parallel.BatchCompletionCallBack
    _joblib.parallel.BatchCompletionCallBack = _PBCallback
    try:
        _render()
        yield
    finally:
        _joblib.parallel.BatchCompletionCallBack = old_cb
        _render(final=True)


def train_threat_detector(training_config: Dict, model_save_path: str = "./model/threat_detector.joblib") -> Optional[Tuple[RandomForestClassifier, pd.DataFrame, pd.Series]]:
    """
    Train Random Forest model using the unified data pipeline.
    
    This function orchestrates the complete training pipeline:
    1. Loads training data via data_loader
    2. Aggregates features into 30-minute windows
    3. Generates 1128-dimensional feature vectors
    4. Trains Random Forest with CLAUDE.md specifications
    5. Saves model and generates evaluation metrics
    
    Parameters:
    -----------
    training_config : Dict
        Configuration dictionary for training parameters and paths
    model_save_path : str
        Path where the trained model will be saved
        
    Returns:
    --------
    Tuple[RandomForestClassifier, pd.DataFrame, pd.Series] or None
        The trained model, test features, and test labels if successful, None if training fails.
    """
    
    try:
        logger.info("Starting model training with pipeline integration...")
        
        # Create model directory if it doesn't exist
        model_dir = os.path.dirname(model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created model directory: {model_dir}")
        
        # Step 1: Load training data via pipeline
        logger.info("Loading training data via data_loader...")
        df = load_data('train', training_config)
        if df.empty:
            raise ValueError("No training data loaded")
        logger.info(f"Loaded {len(df)} training records")
        
        # Step 2: Aggregate features into 30-minute windows
        logger.info("Aggregating features into 30-minute windows...")
        df_agg = aggregate_to_windows(df, window_size_minutes=30)
        if df_agg.empty:
            raise ValueError("No aggregated windows generated")
        logger.info(f"Created {len(df_agg)} aggregated windows")
        
        # Step 3: Generate 1128-dimensional feature vectors
        logger.info("Generating 1128-dimensional feature vectors...")
        feature_gen = FeatureGenerator()
        feature_gen.initialize_rules()
        X, y = feature_gen.generate_feature_vectors(df_agg, mode='train')
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Label distribution - Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
        
        # Step 4: Split data with stratification (80/20 split)
        logger.info("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y  # Handle class imbalance
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Step 5: Train model (optionally with GridSearchCV)
        gs_cfg = training_config.get('grid_search', {}) if isinstance(training_config, dict) else {}
        use_grid = bool(gs_cfg.get('enabled', False))

        if use_grid:
            logger.info("GridSearchCV enabled — tuning RandomForest hyperparameters...")

            # Reasonable defaults compatible with sklearn 0.24.2
            default_param_grid = {
                'n_estimators': [i for i in range(300, 1000, 200)],
                'max_depth': [i for i in range(10, 20, 5)] + [None],
                'max_features': ['sqrt', 'log2'],
                'min_samples_split': [i for i in range(10, 50, 10)],
                'min_samples_leaf': [i for i in range(10, 20, 5)],
                'class_weight': ['balanced'],
                #'min_impurity_decrease': [0.0, 0.0001, 0.001, 0.01],
                #'ccp_alpha': [0.0, 0.0001, 0.001, 0.01],
                #'criterion': ['gini', 'entropy'],
                }

            param_grid = gs_cfg.get('param_grid', default_param_grid)
            # Enable composite multi-metric scoring (ROC AUC + PR AUC) by default
            # Users can override via grid_search.use_composite=false
            use_composite = bool(gs_cfg.get('use_composite', True))
            # Composite weights can be customized in config; default to equal weights
            composite_weights = gs_cfg.get('composite_weights', {'roc_auc': 0.5, 'average_precision': 0.5})
            n_splits = int(gs_cfg.get('cv', 3))
            verbose = int(gs_cfg.get('verbose', 1))

            if use_composite:
                scoring = {
                    'roc_auc': 'roc_auc',
                    'average_precision': 'average_precision'
                }
                def _refit_composite(cv_results):
                    try:
                        roc = np.array(cv_results['mean_test_roc_auc'], dtype=float)
                        ap = np.array(cv_results['mean_test_average_precision'], dtype=float)
                        w_roc = float(composite_weights.get('roc_auc', 0.5))
                        w_ap = float(composite_weights.get('average_precision', 0.5))
                        comp = w_roc * roc + w_ap * ap
                        # Return index of best candidate
                        return int(np.nanargmax(comp))
                    except Exception:
                        # Fallback to best ROC AUC if composite fails
                        return int(np.nanargmax(cv_results.get('mean_test_roc_auc', [0.0])))
                refit_arg = _refit_composite
            else:
                # Single metric path (backward compatible)
                scoring = gs_cfg.get('scoring', 'roc_auc')
                refit_arg = True

            base_rf = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
            )
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            grid = GridSearchCV(
                estimator=base_rf,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=refit_arg,
                verbose=verbose,
            )
            
            # Compute expected total fits for progress bar
            def _count_candidates(pg) -> int:
                if isinstance(pg, dict):
                    sizes = [len(v) for v in pg.values()] if pg else [0]
                    prod = int(np.prod(sizes)) if sizes else 0
                    return prod
                if isinstance(pg, (list, tuple)):
                    return int(sum(_count_candidates(p) for p in pg))
                return 0

            try:
                n_candidates = _count_candidates(param_grid)
                n_splits_effective = cv.get_n_splits(X_train, y_train)
                total_fits = int(n_candidates * n_splits_effective)
            except Exception:
                total_fits = 0

            # Wrap fit in a joblib progress bar (no external deps)
            with joblib_progress_bar(total=total_fits, desc="GridSearchCV"):
                grid.fit(X_train, y_train)
            rf_model = grid.best_estimator_
            try:
                logger.info(f"GridSearch best params: {grid.best_params_}")
            except Exception:
                pass
            # Log best scores for both metrics (if available) and composite
            try:
                best_idx = int(getattr(grid, 'best_index_', -1))
            except Exception:
                best_idx = -1
            try:
                cvres = grid.cv_results_
                if best_idx >= 0 and isinstance(cvres, dict):
                    roc_list = cvres.get('mean_test_roc_auc')
                    ap_list = cvres.get('mean_test_average_precision')
                    if roc_list is not None and ap_list is not None:
                        w_roc = float(composite_weights.get('roc_auc', 0.5))
                        w_ap = float(composite_weights.get('average_precision', 0.5))
                        best_roc = float(roc_list[best_idx])
                        best_ap = float(ap_list[best_idx])
                        best_comp = w_roc * best_roc + w_ap * best_ap
                        logger.info(f"GridSearch best ROC AUC: {best_roc:.4f}, best PR AUC (AP): {best_ap:.4f}, composite: {best_comp:.4f}")
                # If single metric, keep legacy logging
                elif hasattr(grid, 'best_score_'):
                    logger.info(f"GridSearch best score: {getattr(grid, 'best_score_', 0.0):.4f}")
            except Exception:
                pass

            # Compute held-out test scores for the selected best estimator
            test_roc_auc = None
            test_ap = None
            test_comp = None
            try:
                y_test_proba = rf_model.predict_proba(X_test)[:, 1]
                test_roc_auc = float(roc_auc_score(y_test, y_test_proba))
                test_ap = float(average_precision_score(y_test, y_test_proba))
                if use_composite:
                    w_roc = float(composite_weights.get('roc_auc', 0.5))
                    w_ap = float(composite_weights.get('average_precision', 0.5))
                    test_comp = w_roc * test_roc_auc + w_ap * test_ap
                logger.info(
                    f"Held-out test scores — ROC AUC: {test_roc_auc:.4f}, PR AUC (AP): {test_ap:.4f}"
                    + (f", composite: {test_comp:.4f}" if test_comp is not None else "")
                )
            except Exception as e:
                logger.warning(f"Failed to compute held-out test scores: {e}")

            # Persist full grid search results for auditability
            try:
                model_dir = os.path.dirname(model_save_path) or "."
                base_name = os.path.splitext(os.path.basename(model_save_path))[0]

                # Save full cv_results_ as CSV sorted by rank_test_score
                results_df = pd.DataFrame(grid.cv_results_)
                # If composite multi-metric, compute composite_score column and sort by it
                if use_composite and 'mean_test_roc_auc' in results_df.columns and 'mean_test_average_precision' in results_df.columns:
                    w_roc = float(composite_weights.get('roc_auc', 0.5))
                    w_ap = float(composite_weights.get('average_precision', 0.5))
                    results_df['composite_score'] = (
                        w_roc * results_df['mean_test_roc_auc'].astype(float) +
                        w_ap * results_df['mean_test_average_precision'].astype(float)
                    )
                    results_df = results_df.sort_values('composite_score', ascending=False)
                elif 'rank_test_score' in results_df.columns:
                    # Single-metric legacy ranking
                    results_df = results_df.sort_values('rank_test_score')

                # Insert held-out test scores as the left-most columns for quick visibility
                try:
                    if test_ap is not None:
                        results_df.insert(0, 'best_test_average_precision', test_ap)
                    if test_roc_auc is not None:
                        results_df.insert(0, 'best_test_roc_auc', test_roc_auc)
                    if test_comp is not None:
                        results_df.insert(0, 'best_test_composite', test_comp)
                except Exception:
                    pass
                csv_path = os.path.join(model_dir, f"{base_name}_gridsearch_results.csv")
                results_df.to_csv(csv_path, index=False)
                logger.info(f"GridSearchCV results saved to: {csv_path}")

                # Save concise summary JSON (best params and top-10 rows)
                summary = {
                    'scoring': scoring,
                    'cv': n_splits,
                    'n_candidates': int(len(results_df)) if hasattr(results_df, '__len__') else None,
                    'best_params': getattr(grid, 'best_params_', {}),
                    'top10': results_df.head(10).to_dict(orient='records') if hasattr(results_df, 'head') else []
                }
                # Add best scores including composite when available
                try:
                    best_idx = int(getattr(grid, 'best_index_', -1))
                    cvres = grid.cv_results_
                    if best_idx >= 0 and isinstance(cvres, dict):
                        best_roc = _safe_extract_metric(cvres, 'mean_test_roc_auc', best_idx)
                        best_ap = _safe_extract_metric(cvres, 'mean_test_average_precision', best_idx)
                        if best_roc is not None or best_ap is not None:
                            w_roc = float(composite_weights.get('roc_auc', 0.5))
                            w_ap = float(composite_weights.get('average_precision', 0.5))
                            comp = (w_roc * best_roc + w_ap * best_ap) if (best_roc is not None and best_ap is not None) else None
                            summary['best_scores'] = {
                                'roc_auc': best_roc,
                                'average_precision': best_ap,
                                'composite': comp,
                                'composite_weights': {'roc_auc': w_roc, 'average_precision': w_ap}
                            }
                            # Backward-compat: expose composite as best_score when using composite
                            if comp is not None:
                                summary['best_score'] = comp
                    if not use_composite and hasattr(grid, 'best_score_'):
                        summary['best_score'] = float(getattr(grid, 'best_score_', 0.0))
                except Exception:
                    pass
                summary_path = os.path.join(model_dir, f"{base_name}_gridsearch_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"GridSearchCV summary saved to: {summary_path}")
            except Exception as e:
                logger.warning(f"Failed to persist grid search results: {e}")
            logger.info("GridSearchCV training completed successfully")
        else:
            logger.info("Training RandomForest with fixed hyperparameters...")
            rf_model = RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced_subsample',  # Handle class imbalance
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            logger.info("Model training completed successfully")
        
        # Step 6: Save trained model
        logger.info(f"Saving model to: {model_save_path}")
        joblib.dump(rf_model, model_save_path)
        logger.info("Model saved successfully")
        
        # Log model characteristics
        logger.info(f"Model trained with {rf_model.n_features_in_} features and {len(rf_model.estimators_)} trees")
        
        return rf_model, X_test, y_test
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        return None


def train_threat_detector_from_csv(feature_data_path: str, model_save_path: str) -> str:
    """
    Convenience training entrypoint for labeled CSVs.

    Loads a labeled feature CSV (columns are rule IDs + 'is_attack' or 'label'),
    splits with stratification, trains a RandomForestClassifier, and saves the model.

    Parameters:
    - feature_data_path: path to CSV containing features and 'is_attack' (or 'label')
    - model_save_path: path to save the trained model (.joblib)

    Returns:
    - Path to the saved model
    """
    # Lazy imports to reuse existing dependencies
    import pandas as _pd

    if not os.path.exists(feature_data_path):
        raise FileNotFoundError(f"Feature CSV not found: {feature_data_path}")

    logger.info(f"Loading labeled features from {feature_data_path}")
    df = _pd.read_csv(feature_data_path)

    # Determine label column
    label_col = 'is_attack' if 'is_attack' in df.columns else ('label' if 'label' in df.columns else None)
    if label_col is None:
        raise ValueError("Input CSV must contain 'is_attack' or 'label' column")

    y = df[label_col].astype(int)
    drop_cols = {label_col, 'window_id', 'hostname', 'total_events', 'unique_rules', 'window_start', 'window_end', 'source_label'}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].apply(_pd.to_numeric, errors='coerce').fillna(0.0)

    logger.info("Splitting dataset (stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Training RandomForestClassifier (n_estimators=200, class_weight=balanced_subsample)")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced_subsample',
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    model_dir = os.path.dirname(model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_model, model_save_path)
    logger.info(f"Model saved to: {model_save_path}")

    return model_save_path

def evaluate_and_report(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series, 
                       rule_list: list, model_save_path: str,
                       feature_name_options: Optional[Dict] = None) -> Dict:
    """
    Evaluate the trained model and generate comprehensive reports.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        The trained Random Forest model
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        Test labels
    rule_list : list
        List of rule IDs for feature importance mapping
    model_save_path : str
        Base path for saving evaluation reports
        
    Returns:
    --------
    Dict
        Dictionary containing evaluation metrics and paths to saved reports
    """
    
    try:
        logger.info("Evaluating trained model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        # Log key metrics
        logger.info("=== Model Evaluation Results ===")
        logger.info(f"ROC AUC Score: {roc_auc:.4f}")
        logger.info(f"PR AUC (Average Precision): {pr_auc:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Extract positive class metrics
        if isinstance(report, dict):
            positive_metrics = report.get('1', {})
            if positive_metrics:
                logger.info(f"Positive Class Precision: {positive_metrics.get('precision', 0):.4f}")
                logger.info(f"Positive Class Recall: {positive_metrics.get('recall', 0):.4f}")
                logger.info(f"Positive Class F1-Score: {positive_metrics.get('f1-score', 0):.4f}")
        else:
            logger.warning("Could not extract positive class metrics from report")
        
        # Feature importance analysis
        logger.info("Extracting feature importance...")
        feature_importances = model.feature_importances_

        # Configure rule name mapping via feature_name_options
        include_names = False
        csv_paths: list = []
        direct_map: Dict[int, str] = {}
        if isinstance(feature_name_options, dict):
            include_names = bool(feature_name_options.get('include_rule_names', False))
            csv_paths = feature_name_options.get('csv_paths', []) or []
            raw_map = feature_name_options.get('name_map') or {}
            try:
                direct_map = {int(k): str(v) for k, v in raw_map.items()}
            except Exception:
                direct_map = {}

        # Build rule_id -> rule_name map from provided sources
        rule_name_map: Dict[int, str] = {}
        if include_names:
            # Prefer direct map entries
            if direct_map:
                rule_name_map.update(direct_map)
            # Then load from CSVs if provided
            for path in csv_paths:
                try:
                    if path and os.path.exists(path):
                        df_rules = pd.read_csv(path)
                        if 'id' in df_rules.columns and 'name' in df_rules.columns:
                            for rid, nm in zip(df_rules['id'], df_rules['name']):
                                try:
                                    rid_int = int(rid)
                                    if rid_int not in rule_name_map:
                                        rule_name_map[rid_int] = str(nm)
                                except Exception:
                                    continue
                except Exception:
                    continue

        # Create rule importance mapping and enrich with rule names
        importance_df = pd.DataFrame({
            'rule_id': rule_list,
            'importance': feature_importances
        })
        if include_names:
            importance_df['rule_name'] = [
                rule_name_map.get(int(rid), f"Rule {rid}") for rid in importance_df['rule_id']
            ]

        # Filter out zero-importance features
        # Drop entries with missing or non-positive importance values
        importance_numeric = pd.to_numeric(importance_df['importance'], errors='coerce')
        importance_df = importance_df.assign(importance=importance_numeric)
        importance_df = importance_df.loc[importance_df['importance'].notna()].copy()
        importance_df = importance_df.loc[importance_df['importance'] > 0].copy()
        importance_df = cast(pd.DataFrame, importance_df)
        # Sort by importance descending; ensure deterministic order for ties
        if not importance_df.empty:
            importance_df = importance_df.sort_values(by='importance', ascending=False, kind='mergesort')

        # Prioritize BOC-related features if rule names are available
        # Heuristic keywords; can be extended via feature_name_options in future
        def _is_boc(name: str) -> bool:
            try:
                n = str(name).lower()
                return any(k in n for k in ['boc', 'behavior of compromise', 'behaviordefinition'])
            except Exception:
                return False

        if include_names and 'rule_name' in importance_df.columns:
            importance_df['boc_priority'] = importance_df['rule_name'].apply(_is_boc).astype(int)
            importance_df = importance_df.sort_values(by=['boc_priority', 'importance'], ascending=[False, False])

        # Select Top-10 after filtering and prioritization
        top_10_features = importance_df.head(10)
        logger.info("Top 10 most important features (non-zero, BOC prioritized):")
        for _, row in top_10_features.iterrows():
            if include_names and 'rule_name' in row:
                logger.info(f"  {row['rule_id']} | {row['rule_name']}: {row['importance']:.4f}")
            else:
                logger.info(f"  Rule {row['rule_id']}: {row['importance']:.4f}")
        
        # Save evaluation reports
        model_dir = os.path.dirname(model_save_path)
        base_name = os.path.splitext(os.path.basename(model_save_path))[0]
        
        # Save top features (Top-10). Keep legacy Top-20 filename for compatibility; also write a Top-10 file.
        top10_path = os.path.join(model_dir, f"{base_name}_top_10_features.csv")
        top_10_features.to_csv(top10_path, index=False)
        logger.info(f"Top 10 features saved to: {top10_path}")
        # Backward compatibility file
        legacy_top_path = os.path.join(model_dir, f"{base_name}_top_20_features.csv")
        try:
            top_10_features.to_csv(legacy_top_path, index=False)
            logger.info(f"(Compat) Also wrote: {legacy_top_path}")
        except Exception:
            pass
        
        # Save comprehensive evaluation report
        evaluation_report = {
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'roc_auc_score': roc_auc,
            'pr_auc_average_precision': pr_auc,
            'model_path': model_save_path,
            'feature_count': len(rule_list),
            'top_10_features': top_10_features.to_dict(orient='records'),
            # Backward compatibility key with Top-10 content
            'top_20_features': top_10_features.to_dict(orient='records'),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        report_path = os.path.join(model_dir, f"{base_name}_evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        logger.info(f"Evaluation report saved to: {report_path}")
        
        return evaluation_report
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        return {}


def main():
    """
    Main function for CLI execution of model training.
    """
    parser = argparse.ArgumentParser(description='Train Random Forest model using unified pipeline')
    parser.add_argument('--config', required=True, 
                       help='Path to training configuration JSON file')
    parser.add_argument('--model-output', default='./model/threat_detector.joblib',
                       help='Path to save trained model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run model evaluation after training')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            training_config = json.load(f)
        
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Train the model
        result = train_threat_detector(training_config, args.model_output)
        
        if result is None:
            logger.error("Model training failed")
            return 1
            
        model, X_test, y_test = result
        
        # Run evaluation if requested
        if args.evaluate:
            # Get rule list for feature importance mapping
            rule_manager = QRadarRuleManager()
            rule_list = rule_manager.get_rule_list()
            
            evaluation_report = evaluate_and_report(
                model, X_test, y_test, rule_list, args.model_output
            )
            
            if not evaluation_report:
                logger.warning("Model evaluation failed")
        
        logger.info("Model training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit(main())
