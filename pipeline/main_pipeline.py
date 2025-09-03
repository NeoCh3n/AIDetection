#!/usr/bin/env python3
"""
Unified Threat Detection Pipeline
Single orchestrator for training and detection modes.

How To Use
- Activate venv: `source venv/bin/activate` (required).
- Install deps: `make install` (uses local venv) or `pip install -r requirements.txt`.
- Run training (from repo root):
  - `python -m pipeline.main_pipeline train`
  - or `python ./pipeline/main_pipeline.py train`
- Run detection:
  - `python -m pipeline.main_pipeline detect`
  - or `python ./pipeline/main_pipeline.py detect`

Options
- `--config PATH`: custom JSON config (default `pipeline/config.json`).
- `--verbose`: enable more verbose logging.

Notes
- Requires Python 3.6.8-compatible environment and local venv.
- Training: data_loader → feature_aggregator → feature_generator → model_training (+ evaluation).
- Detection: data_loader → feature_aggregator → feature_generator → model_predictor (alerts/logging).
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
import logging
import numpy as np
from typing import List, Optional

"""
Ensure project root is on sys.path so top-level packages (e.g., `system`,
`api_integration`, `mongodb`, `shared_utils`) are importable when running this
file directly (python ./pipeline/main_pipeline.py).
"""
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system import logging_utils

# Add parent directories to path for imports (kept for safety; root already added)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api_integration'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mongodb'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))

# Import existing pipeline modules
from data_loader import load_data
from feature_aggregator import aggregate_to_windows
from feature_generator import FeatureGenerator

# Training/Evaluation from model_training package
from model_training.model_training import train_threat_detector, evaluate_and_report

# Predictor is only needed in detection; import at callsite to avoid training-only failures.

# Import existing API/MongoDB modules (use package-qualified paths for IDEs)
try:
    from api_integration.create_searches_Qradar import create_searches_Qradar
    from api_integration.status_searches_Qradar import status_searches_Qradar
    from api_integration.result_searches_Qradar import result_searches_Qradar
    from api_integration.delete_searches_Qradar import delete_searches_Qradar
    # MongoDB utilities
    # Align with available functions in mongodb/delete_DB.py
    from mongodb.delete_DB import cleanup_old_data as delete_old_rule_triggers
    # Removed unused and nonexistent import of query_DB
    # AQL JSON inserter (from mongodb/insert_DB.py)
    from mongodb.insert_DB import AQLDataInserter  # type: ignore
except ImportError as e:
    logging.warning(f"API/MongoDB modules not fully available: {e}")
    AQLDataInserter = None  # type: ignore

# Import shared utilities
from shared_utils.qradar_rule_manager import QRadarRuleManager
from shared_utils.time_utils import parse_qradar_timestamp, get_window_id

class UnifiedPipeline:
    def __init__(self, mode='detect', config_path=None, verbose: bool = False):
        self.mode = mode
        self.verbose = verbose
        self.config = self.load_config(config_path)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup unified logging for pipeline"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'running_log')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'pipeline_{self.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('UnifiedPipeline')
        
    def load_config(self, config_path=None):
        """Load unified configuration (JSON). Defaults to pipeline/config.json."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')

        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Failed to load {config_path}: {e}. Falling back to defaults.")
            return {
                "training": {
                    "data_path": "./Training_data",
                    "model_path": "./model/threat_detector.joblib",
                    "test_size": 0.2,
                    "random_state": 42
                },
                "detection": {
                    "qradar_config": {
                        "host": os.getenv("QRADAR_ADDRESS", "192.168.153.123"),
                        "token": os.getenv("QRADAR_API_TOKEN", "REPLACE_WITH_TOKEN"),
                        "query_30min": "SELECT \"sysmon_hostname\" AS 'sysmon_hostname (custom)', \"creEventList\" AS 'Custom Rule', DATEFORMAT(MIN(\"deviceTime\"), 'MMM dd, yyyy, h:mm:ss a') AS 'Log Source Time (Minimum)', COUNT(*) AS 'Count' FROM events GROUP BY \"sysmon_hostname\", \"creEventList\" ORDER BY \"Count\" DESC START '{start_time}' STOP '{end_time}'"
                    },
                    "mongodb_config": {
                        "connection_string": "mongodb://localhost:27017/",
                        "database": "qradar_ml",
                        "collection": "detection_data"
                    },
                    "retention_days": 7,
                    "alert_threshold": 0.8
                },
                "rule_manager": {
                    "mode": "file",
                    "api_config": {}
                }
            }
    
    def run_training(self):
        """Training mode: CSV → Model (delegates to model_training exports)."""
        self.logger.info("Starting training pipeline...")

        try:
            training_cfg = self.config.get('training', {})
            model_path = training_cfg.get('model_path', './model/threat_detector.joblib')

            # Delegate training to integrated trainer
            self.logger.info("Training via train_threat_detector...")
            result = train_threat_detector(training_cfg, model_path)
            if result is None:
                raise RuntimeError("train_threat_detector returned None")

            model, X_test, y_test = result

            # Evaluate model
            self.logger.info("Evaluating via evaluate_and_report...")
            rm_cfg = self.config.get('rule_manager', {})
            rule_manager = QRadarRuleManager(
                mode=rm_cfg.get('mode', 'file'),
                config=rm_cfg.get('api_config', {}),
            )
            rule_list = rule_manager.get_rule_list()
            _report = evaluate_and_report(model, X_test, y_test, rule_list, model_path)

            self.logger.info("Training pipeline completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            return False
    
    def fetch_qradar_data(self, start_time, end_time):
        """Fetch data from QRadar for detection mode"""
        self.logger.info(f"Fetching QRadar data from {start_time} to {end_time}")
        
        try:
            qcfg = self.config['detection']['qradar_config']
            # Construct AQL query
            aql = qcfg['query_30min'].format(
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=end_time.strftime("%Y-%m-%d %H:%M:%S")
            )
            # Build request header from config (avoid logging token)
            request_header = {
                'SEC': qcfg.get('token', ''),
                'Version': str(qcfg.get('version', '20.0')),
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Connection': 'Close'
            }
            http_timeout = int(qcfg.get('timeout', 300))
            poll_interval = max(3, int(qcfg.get('poll_interval', 5)))
            max_wait = int(qcfg.get('max_wait_seconds', http_timeout))
            
            # Create search
            search_response = create_searches_Qradar(
                qradar_address=qcfg['host'],
                AQL=aql,
                request_header=request_header,
                timeout=min(http_timeout, max_wait)
            )
            
            if not search_response:
                self.logger.error("Failed to create QRadar search")
                try:
                    logging_utils.run_log("ERROR", "QRadar create_search failed")
                except Exception:
                    pass
                return None
            
            search_id = search_response.get('search_id')
            self.logger.info(f"Created search: {search_id}")
            
            # Poll for completion and get results
            import time
            start_wait = time.monotonic()
            last_status = None
            last_progress = None
            while True:
                # Check timeout
                elapsed = time.monotonic() - start_wait
                if elapsed > max_wait:
                    msg = (
                        f"QRadar search timed out after {int(elapsed)}s waiting for completion; "
                        f"search_id={search_id}, last_status={last_status}, last_progress={last_progress}"
                    )
                    self.logger.error(msg)
                    try:
                        logging_utils.run_log("ERROR", msg)
                    except Exception:
                        pass
                    # Attempt cleanup
                    try:
                        delete_searches_Qradar(Qradar_address=qcfg['host'], search_id=search_id, request_header=request_header, timeout=10)
                    except Exception:
                        pass
                    return None

                # Query status
                status_resp = status_searches_Qradar(
                    Qradar_address=qcfg['host'],
                    search_id=search_id,
                    request_header=request_header,
                    timeout=min(30, http_timeout)
                )
                if not status_resp or not isinstance(status_resp, dict):
                    time.sleep(poll_interval)
                    continue

                last_status = str(status_resp.get('status', '')).upper()
                last_progress = status_resp.get('progress')
                # Log periodic progress
                self.logger.info(f"QRadar search status: {last_status} ({last_progress}%) for search_id={search_id}")
                if last_status in ("COMPLETED", "COMPLETE", "SORTED", "DONE", "FINISHED"):
                    break
                time.sleep(poll_interval)
            
            result_data = result_searches_Qradar(
                Qradar_address=qcfg['host'],
                search_id=search_id,
                request_header=request_header,
                timeout=min(60, max(15, http_timeout))
            )
            
            # Cleanup search
            delete_searches_Qradar(
                Qradar_address=qcfg['host'],
                search_id=search_id,
                request_header=request_header,
                timeout=15
            )
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch QRadar data: {str(e)}")
            try:
                logging_utils.run_log("ERROR", f"QRadar fetch failure: {str(e)}")
            except Exception:
                pass
            return None
    
    def store_detection_data(self, data, start_time, end_time):
        """Store detection data in MongoDB"""
        self.logger.info("Storing detection data in MongoDB...")
        
        try:
            if AQLDataInserter is None:
                self.logger.error("AQLDataInserter not available; cannot store detection data")
                return []

            inserter = AQLDataInserter()
            if not inserter.connect():
                self.logger.error("Failed to connect to MongoDB for AQL insertion")
                return []

            # Transform AQL JSON results into detection windows and insert
            documents = inserter.parse_aql_json_result(data) if data else []
            inserted = inserter.insert_detection_windows(documents) if documents else 0
            inserter.close()

            self.logger.info(f"Inserted {inserted} detection windows")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to store detection data: {str(e)}")
            return []
    
    def cleanup_old_data(self):
        """Clean up old detection data"""
        try:
            deleted_count = delete_old_rule_triggers(
                self.config['detection']['retention_days']
            )
            self.logger.info(f"Cleaned up {deleted_count} old records")
            return deleted_count
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {str(e)}")
            return 0
    
    def run_detection(self):
        """Detection mode: QRadar → MongoDB → Prediction"""
        self.logger.info("Starting detection pipeline...")
        
        try:
            # Calculate time window
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=30)
            
            # Cleanup old data
            self.cleanup_old_data()
            
            # Fetch QRadar data
            qradar_data = self.fetch_qradar_data(start_time, end_time)
            if not qradar_data:
                self.logger.warning("No QRadar data retrieved")
                return None
            
            # Store in MongoDB
            documents = self.store_detection_data(qradar_data, start_time, end_time)
            if not documents:
                self.logger.warning("No documents to process")
                return None
            
            # Load data for prediction
            df = load_data('detect', self.config['detection'])
            if df.empty:
                self.logger.warning("No data loaded for prediction")
                return None
            
            # Aggregate features (ensure detection-mode to avoid training labels)
            df_agg = aggregate_to_windows(df, mode='detect')
            if df_agg.empty:
                self.logger.warning("No aggregated windows")
                return None
            
            # Generate features
            feature_gen = FeatureGenerator()
            feature_gen.initialize_rules()
            X, _ = feature_gen.generate_feature_vectors(df_agg, mode='detect')

            # Build feature name mapping for explainability/top-k logging
            try:
                prod_rule_to_index = feature_gen.rule_manager.get_production_rule_to_index_map()
                dim = feature_gen.get_feature_vector_dimension()
                index_to_rule: List[Optional[int]] = [None] * dim
                for rid, idx in prod_rule_to_index.items():
                    j = int(idx)
                    if 0 <= j < dim:
                        index_to_rule[j] = int(rid)
                feature_names = [f"rule_{r}" if r is not None else f"feature_{i}" for i, r in enumerate(index_to_rule)]
            except Exception:
                index_to_rule = None
                feature_names = [f"feature_{i}"] * (X.shape[1] if hasattr(X, 'shape') else 0)
            
            # Make predictions
            try:
                from model_predictor import Predictor as ModelPredictor
            except Exception as e:
                self.logger.error(f"Predictor not available: {e}. Ensure model_predictor.py exists and the model path is correct.")
                return None

            predictor = ModelPredictor(self.config['training']['model_path'])
            predictions = predictor.predict(X)

            # Initialize SHAP explainer once if available
            shap_explainer = None
            try:
                from system.shap_explainer import Explainer as ShapExplainer
                shap_explainer = ShapExplainer(
                    getattr(predictor, 'model', None),
                    X if hasattr(X, 'shape') and X.shape[0] > 1 else (X if hasattr(X, 'shape') else []),
                    feature_names=feature_names
                )
            except Exception as e:
                self.logger.warning(f"SHAP explainer unavailable: {e}")
            
            # Process results; persist alerts to MongoDB
            results = []
            try:
                from mongodb.mongodb_connection import get_mongodb_manager
            except Exception:
                get_mongodb_manager = None  # type: ignore

            mongo_ctx = get_mongodb_manager() if get_mongodb_manager else None
            ctx_manager = mongo_ctx if mongo_ctx else None

            # Use context if available; else no-op context
            class _NoopCtx:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc, tb):
                    return False

            with (ctx_manager or _NoopCtx()) as manager:
                for idx, (pred, prob) in enumerate(predictions):
                    is_alert = prob > self.config['detection']['alert_threshold']
                    result = {
                        'timestamp': datetime.now().isoformat(),
                        'hostname': df_agg.iloc[idx]['hostname'],
                        'window_id': df_agg.iloc[idx]['window_id'],
                        'prediction': int(pred),
                        'probability': float(prob),
                        'alert': bool(is_alert)
                    }
                    results.append(result)

                    # Unified detection logging
                    try:
                        label_str = 'malicious' if int(pred) == 1 else 'normal'
                        logging_utils.log_detection(
                            hostname=result['hostname'],
                            window_id=result['window_id'],
                            prediction=label_str,
                            confidence=result['probability']
                        )
                    except Exception:
                        pass

                    # On alert, add detailed top-10 rules and SHAP explanation
                    if is_alert:
                        try:
                            # Top-10 rules by feature magnitude
                            top_n = int(self.config.get('detection', {}).get('top_rules_count', 10))
                            if hasattr(X, 'shape') and X.shape[0] > idx:
                                row_vec = X[idx]
                                # argsort descending
                                top_idx = list(reversed(np.argsort(row_vec)[:top_n].tolist()))
                                top_rules = []
                                for j in top_idx:
                                    val = float(row_vec[j])
                                    if val <= 0:
                                        continue
                                    rule_id = None
                                    if index_to_rule and j < len(index_to_rule):
                                        rule_id = index_to_rule[j]
                                    top_rules.append({'rule_id': int(rule_id) if rule_id is not None else j, 'value': val})
                            else:
                                top_rules = []

                            # Log alert details with top rules
                            try:
                                logging_utils.run_log(
                                    "ALERT_DETAIL",
                                    f"Alert details | hostname={result['hostname']} | window_id={result['window_id']} | confidence={result['probability']:.4f}",
                                    payload={
                                        'hostname': result['hostname'],
                                        'window_id': result['window_id'],
                                        'confidence': result['probability'],
                                        'top_rules_by_count': top_rules,
                                    }
                                )
                            except Exception:
                                pass

                            # SHAP explanation (top-N contributing features)
                            if shap_explainer is not None and hasattr(X, 'shape'):
                                try:
                                    # Explain single instance
                                    shap_vals = shap_explainer.explain(row_vec.reshape(1, -1), log_results=False)

                                    # Build ranking from explainer util to align features
                                    ranking = []
                                    try:
                                        ranking = shap_explainer.get_feature_importance(row_vec.reshape(1, -1))
                                    except Exception:
                                        pass

                                    top_r = ranking[:top_n] if ranking else []
                                    shap_top_rules = [
                                        int(str(item.get('feature', '').replace('rule_', ''))) if str(item.get('feature', '')).startswith('rule_') else item.get('feature')
                                        for item in top_r
                                    ]
                                    shap_top_values = [float(item.get('importance', 0.0)) for item in top_r]

                                    try:
                                        logging_utils.log_shap_results(
                                            hostname=result['hostname'],
                                            window_id=result['window_id'],
                                            top_rules=shap_top_rules,
                                            shap_values=shap_top_values,
                                            prediction=label_str,
                                            confidence=result['probability']
                                        )
                                    except Exception:
                                        pass
                                except Exception as e:
                                    self.logger.warning(f"SHAP explanation failed for {result['window_id']}: {e}")
                        except Exception as e:
                            self.logger.warning(f"Alert detail logging failed: {e}")

                    # Persist alerts to detection_results
                    if is_alert and manager:
                        try:
                            manager.insert_prediction({
                                'window_id': result['window_id'],
                                'hostname': result['hostname'],
                                'predicted_label': int(pred),
                                'confidence': float(prob)
                            })
                        except Exception:
                            pass

                    if is_alert:
                        self.logger.warning(
                            f"ALERT: Threat detected on {result['hostname']} (p={prob:.2f})"
                        )
            
            self.logger.info(f"Detection completed. {len(results)} alerts generated")
            return results
            
        except Exception as e:
            self.logger.error(f"Detection pipeline failed: {str(e)}")
            return None
    
    def execute(self):
        """Main execution based on mode"""
        self.logger.info(f"Starting pipeline in {self.mode} mode")
        
        if self.mode == 'train':
            return self.run_training()
        elif self.mode == 'detect':
            return self.run_detection()
        else:
            self.logger.error(f"Invalid mode: {self.mode}")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Unified Threat Detection Pipeline',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            'Examples:\n'
            '  source venv/bin/activate\n'
            '  make install\n'
            '  python -m pipeline.main_pipeline train --config pipeline/config.json\n'
            '  python -m pipeline.main_pipeline detect --config pipeline/config.json\n\n'
            'Options:\n'
            '  --config PATH   Use a custom JSON config (default: pipeline/config.json)\n'
            '  --verbose       Enable more verbose logging\n'
        ),
    )
    parser.add_argument('mode', choices=['train', 'detect'], help='Pipeline mode')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    pipeline = UnifiedPipeline(mode=args.mode, config_path=args.config, verbose=args.verbose)
    result = pipeline.execute()
    
    if result:
        logging_utils.run_log("INFO", f"Pipeline completed successfully in {args.mode} mode")
        sys.exit(0)
    else:
        logging_utils.run_log("ERROR", f"Pipeline failed in {args.mode} mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
