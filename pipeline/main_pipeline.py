#!/usr/bin/env python3
"""
Unified Threat Detection Pipeline
Single orchestrator for training and detection modes.

How To Use
- Activate venv: `source venv/bin/activate` (required).
- Install deps: `make install` (uses local venv) or `pip install -r requirements.txt`.
- Run training (from repo root):
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
import csv
from datetime import datetime, timedelta
import logging
import random
import time
from contextlib import contextmanager
from typing import List, Optional, Dict, Any

import numpy as np
import importlib

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

# Import existing pipeline modules (support both package and script execution)
try:
    # When run as a package: python -m pipeline.main_pipeline
    from .data_loader import load_data  # type: ignore
    from .feature_aggregator import aggregate_to_windows  # type: ignore
    from .feature_generator import FeatureGenerator  # type: ignore
except Exception:
    # When run as a script: python ./pipeline/main_pipeline.py
    load_data = importlib.import_module('data_loader').load_data  # type: ignore
    aggregate_to_windows = importlib.import_module('feature_aggregator').aggregate_to_windows  # type: ignore
    FeatureGenerator = importlib.import_module('feature_generator').FeatureGenerator  # type: ignore
from system.shap_explainer import Explainer

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
        """Setup unified logging: all logs to running_log/YYYY-MM-DD.log"""
        # Route all Python logging to the daily running_log file and keep console output.
        # This avoids creating per-run pipeline_* log files.
        try:
            level = logging.DEBUG if self.verbose else logging.INFO
            logging_utils.setup_global_daily_file_logging(level=level, include_stdout=True)
        except Exception:
            # Fallback: ensure at least a basic console logger is present
            logging.basicConfig(level=logging.DEBUG if self.verbose else logging.INFO)
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
                        "query_15min": "SELECT \"sourceIP\" AS 'Source IP', \"sysmon_hostname\" AS 'sysmon_hostname (custom)', \"creEventList\" AS 'Custom Rule', DATEFORMAT(MIN(\"deviceTime\"), 'MMM dd, yyyy, h:mm:ss a') AS 'Log Source Time (Minimum)', COUNT(*) AS 'Count' FROM events WHERE (\"Sysmon_hostname\" != NULL) GROUP BY \"sysmon_hostname\", \"sourceIP\", \"creEventList\" ORDER BY \"Count\" DESC START '{start_time}' STOP '{end_time}'"
                    },
                    "mongodb_config": {
                        "connection_string": "mongodb://localhost:27017/",
                        "database": "qradar_ml",
                        "collection": "detection_data"
                },
                "retention_days": 7,
                "alert_threshold": 0.8,
                "min_fetch_interval_seconds": 300,
                "lock_max_age_seconds": 1800
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
            # Optional feature name enrichment for top features
            fname_opts = training_cfg.get('feature_names', {}) if isinstance(training_cfg, dict) else None
            _report = evaluate_and_report(model, X_test, y_test, rule_list, model_path, feature_name_options=fname_opts)

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
            aql = qcfg['query_15min'].format(
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
            base_poll_interval = poll_interval
            max_wait = int(qcfg.get('max_wait_seconds', http_timeout))
            backoff_factor = float(qcfg.get('status_backoff_factor', 1.5))
            max_poll_interval = max(poll_interval, int(qcfg.get('max_poll_interval', 60)))
            jitter_seconds = max(0.0, float(qcfg.get('status_poll_jitter', 1.0)))
            backoff_step_limit = max(1, int(qcfg.get('status_backoff_steps', 6)))
            max_status_checks = int(qcfg.get('max_status_checks', 0))
            if max_status_checks <= 0:
                approx_checks = int(max_wait / max(poll_interval, 1)) + 1
                max_status_checks = max(approx_checks, 5)
            
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
            attempt_count = 0
            consecutive_errors = 0
            consecutive_pending = 0
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
                if attempt_count >= max_status_checks:
                    msg = (
                        f"QRadar search exceeded maximum status checks ({max_status_checks}); "
                        f"search_id={search_id}, last_status={last_status}, last_progress={last_progress}, "
                        f"elapsed={int(elapsed)}s"
                    )
                    self.logger.error(msg)
                    try:
                        logging_utils.run_log("ERROR", msg)
                    except Exception:
                        pass
                    try:
                        delete_searches_Qradar(Qradar_address=qcfg['host'], search_id=search_id, request_header=request_header, timeout=10)
                    except Exception:
                        pass
                    return None

                attempt_count += 1

                # Query status
                status_resp = status_searches_Qradar(
                    Qradar_address=qcfg['host'],
                    search_id=search_id,
                    request_header=request_header,
                    timeout=min(30, http_timeout)
                )
                if not status_resp or not isinstance(status_resp, dict):
                    consecutive_errors += 1
                    exponent = min(consecutive_errors, backoff_step_limit)
                    sleep_seconds = min(
                        max_poll_interval,
                        base_poll_interval * (backoff_factor ** exponent)
                    )
                    if jitter_seconds:
                        sleep_seconds += random.uniform(0, jitter_seconds)
                    self.logger.warning(
                        f"QRadar status check yielded no data (attempt {attempt_count}/{max_status_checks}); "
                        f"backing off for {sleep_seconds:.1f}s"
                    )
                    time.sleep(sleep_seconds)
                    continue

                http_status = status_resp.get('__http_status')
                try:
                    http_status = int(http_status) if http_status is not None else None
                except (TypeError, ValueError):
                    http_status = None

                retry_after_header = status_resp.get('__retry_after')
                retry_after_seconds = None
                if retry_after_header:
                    try:
                        retry_after_seconds = float(retry_after_header)
                    except (TypeError, ValueError):
                        retry_after_seconds = None

                if http_status is not None and http_status >= 400:
                    consecutive_errors += 1
                    exponent = min(consecutive_errors, backoff_step_limit)
                    if http_status == 429 and retry_after_seconds is not None:
                        sleep_seconds = min(max_poll_interval, max(retry_after_seconds, base_poll_interval))
                    else:
                        sleep_seconds = min(
                            max_poll_interval,
                            base_poll_interval * (backoff_factor ** exponent)
                        )
                    if jitter_seconds:
                        sleep_seconds += random.uniform(0, jitter_seconds)
                    self.logger.warning(
                        f"QRadar status check HTTP {http_status} (attempt {attempt_count}/{max_status_checks}); "
                        f"backing off for {sleep_seconds:.1f}s"
                    )
                    time.sleep(sleep_seconds)
                    continue

                if 'status' not in status_resp:
                    consecutive_errors += 1
                    exponent = min(consecutive_errors, backoff_step_limit)
                    sleep_seconds = min(
                        max_poll_interval,
                        base_poll_interval * (backoff_factor ** exponent)
                    )
                    if jitter_seconds:
                        sleep_seconds += random.uniform(0, jitter_seconds)
                    self.logger.warning(
                        f"QRadar status payload missing 'status' field (attempt {attempt_count}/{max_status_checks}); "
                        f"sleeping {sleep_seconds:.1f}s before retry"
                    )
                    time.sleep(sleep_seconds)
                    continue

                # Valid response resets error streak
                consecutive_errors = 0

                prev_status = last_status
                prev_progress = last_progress

                status_value = str(status_resp.get('status', '')).upper()
                progress_value = status_resp.get('progress')

                # Log periodic progress
                self.logger.info(f"QRadar search status: {status_value} ({progress_value}%) for search_id={search_id}")
                if status_value in ("COMPLETED", "COMPLETE", "SORTED", "DONE", "FINISHED"):
                    last_status = status_value
                    last_progress = progress_value
                    break

                status_changed = prev_status is None or status_value != prev_status
                progress_improved = False
                if progress_value is not None and prev_progress is not None:
                    try:
                        progress_improved = float(progress_value) > float(prev_progress)
                    except (TypeError, ValueError):
                        progress_improved = False

                last_status = status_value
                last_progress = progress_value

                if status_changed or progress_improved:
                    consecutive_pending = 0
                else:
                    consecutive_pending += 1

                exponent = min(max(consecutive_pending - 1, 0), backoff_step_limit)
                sleep_seconds = min(
                    max_poll_interval,
                    base_poll_interval * (backoff_factor ** exponent)
                )
                if jitter_seconds:
                    sleep_seconds += random.uniform(0, jitter_seconds)

                self.logger.debug(
                    f"QRadar search pending (status={status_value}, progress={progress_value}); "
                    f"sleeping {sleep_seconds:.1f}s before next status check "
                    f"(attempt {attempt_count}/{max_status_checks})"
                )
                time.sleep(sleep_seconds)
            
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

    def _get_detection_runtime_dir(self) -> str:
        """Directory used to persist detection runtime state (locks, metadata)."""
        paths_cfg = self.config.get('paths', {})
        base_dir = paths_cfg.get('logs')
        if base_dir:
            base_dir = os.path.abspath(base_dir)
        else:
            base_dir = os.path.join(PROJECT_ROOT, 'running_log')

        runtime_dir = os.path.join(base_dir, 'detection_runtime')
        try:
            os.makedirs(runtime_dir, exist_ok=True)
        except Exception as exc:
            self.logger.warning(f"Failed to ensure detection runtime directory: {exc}")
        return runtime_dir

    def _get_detection_lock_path(self) -> str:
        """Path to the detection lock file."""
        return os.path.join(self._get_detection_runtime_dir(), 'detection.lock')

    def _get_detection_state_path(self) -> str:
        """Path to the detection state file."""
        return os.path.join(self._get_detection_runtime_dir(), 'detection_state.json')

    def _load_detection_state(self) -> Dict[str, Any]:
        """Load persisted detection state metadata."""
        state_path = self._get_detection_state_path()
        try:
            with open(state_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            self.logger.warning(f"Detection state file malformed ({exc}); resetting state")
        except Exception as exc:
            self.logger.warning(f"Failed to read detection state file: {exc}")
        return {}

    def _persist_detection_state(self, state: Dict[str, Any]) -> None:
        """Persist detection state metadata atomically."""
        state_path = self._get_detection_state_path()
        tmp_path = f"{state_path}.tmp"
        try:
            with open(tmp_path, 'w', encoding='utf-8') as fh:
                json.dump(state, fh)
            os.replace(tmp_path, state_path)
        except Exception as exc:
            self.logger.warning(f"Unable to write detection state file: {exc}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _get_min_fetch_interval_seconds(self) -> float:
        """Minimum seconds to wait between QRadar queries."""
        detection_cfg = self.config.get('detection', {})
        interval = detection_cfg.get('min_fetch_interval_seconds')
        if interval is None:
            # Default throttle prevents aggressive re-queries; override via config if needed.
            interval = 300
        try:
            interval_val = float(interval)
        except (TypeError, ValueError):
            interval_val = 300.0
        return max(0.0, interval_val)

    def _seconds_until_next_fetch(self, state: Dict[str, Any]) -> float:
        """Return remaining seconds until the next QRadar fetch is allowed."""
        min_interval = self._get_min_fetch_interval_seconds()
        if min_interval <= 0:
            return 0.0

        last_fetch_epoch = state.get('last_fetch_epoch')
        if last_fetch_epoch is None:
            return 0.0

        try:
            elapsed = time.time() - float(last_fetch_epoch)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min_interval - elapsed)

    def _acquire_detection_lock(self):
        """Attempt to create the detection lock file. Returns lock file handle or None."""
        lock_path = self._get_detection_lock_path()
        detection_cfg = self.config.get('detection', {})
        stale_seconds = detection_cfg.get('lock_max_age_seconds', 1800)

        try:
            lock_file = open(lock_path, 'x')
        except FileExistsError:
            if stale_seconds:
                try:
                    mtime = os.path.getmtime(lock_path)
                except OSError:
                    mtime = None
                if mtime is not None and (time.time() - mtime) > stale_seconds:
                    self.logger.warning(
                        f"Found stale detection lock (> {stale_seconds}s); attempting cleanup"
                    )
                    try:
                        os.remove(lock_path)
                    except Exception as exc:
                        self.logger.error(f"Failed to remove stale detection lock: {exc}")
                        return None
                    try:
                        lock_file = open(lock_path, 'x')
                    except FileExistsError:
                        return None
                else:
                    self.logger.info("Detection lock already held by another runner; skipping")
                    return None
            else:
                self.logger.info("Detection lock already held by another runner; skipping")
                return None
        except Exception as exc:
            self.logger.error(f"Unexpected error acquiring detection lock: {exc}")
            return None

        try:
            lock_file.write(json.dumps({
                "pid": os.getpid(),
                "acquired_at": time.time()
            }))
            lock_file.flush()
        except Exception as exc:
            try:
                lock_file.close()
            finally:
                try:
                    os.remove(lock_path)
                except Exception:
                    pass
            self.logger.error(f"Failed to initialize detection lock: {exc}")
            return None

        return lock_file

    @contextmanager
    def _detection_run_guard(self):
        """
        Context manager managing inter-process locking for detection runs.

        Yields:
            bool: True if the lock was acquired and the caller may proceed.
        """
        lock_file = self._acquire_detection_lock()
        if lock_file is None:
            yield False
            return

        try:
            yield True
        finally:
            try:
                lock_file.close()
            except Exception:
                pass
            try:
                os.remove(self._get_detection_lock_path())
            except FileNotFoundError:
                pass
            except Exception as exc:
                self.logger.warning(f"Failed to remove detection lock: {exc}")
    
    def load_production_rule_name_map(self) -> Dict[int, str]:
        """
        Load production rule ID to rule name mapping from shared mapping file.
        
        Returns:
            Dictionary mapping production rule IDs to human-readable names.
        """
        mapping: Dict[int, str] = {}
        mapping_path = os.path.join(PROJECT_ROOT, 'shared_utils', 'uat_to_prod_mapping.csv')

        try:
            if not os.path.exists(mapping_path):
                self.logger.debug(f"Production rule mapping file not found: {mapping_path}")
                return mapping

            with open(mapping_path, 'r', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if not row:
                        continue
                    prod_value = row.get('prod_rule_id')
                    name_value = row.get('rule_name') or row.get('prod_rule_name')
                    try:
                        prod_rule_id = int(str(prod_value).strip()) if prod_value is not None else None
                    except (TypeError, ValueError):
                        continue

                    if prod_rule_id is None:
                        continue

                    if name_value is None:
                        continue

                    rule_name = str(name_value).strip()
                    if not rule_name:
                        continue

                    mapping[prod_rule_id] = rule_name

        except Exception as exc:
            self.logger.warning(f"Failed to load production rule names: {exc}")

        return mapping
    
    def run_detection(self):
        """Detection mode: QRadar → MongoDB → Prediction"""
        self.logger.info("Starting detection pipeline...")
        with self._detection_run_guard() as lock_acquired:
            if not lock_acquired:
                self.logger.info("Detection run skipped: another execution is in progress")
                return None

            state = self._load_detection_state()
            wait_seconds = self._seconds_until_next_fetch(state)
            if wait_seconds > 0:
                self.logger.info(
                    "Detection run skipped: next QRadar fetch allowed in %.1fs",
                    wait_seconds
                )
                return None

            # Calculate time window
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=15)

            state['last_attempt_epoch'] = time.time()
            self._persist_detection_state(state)

            fetch_attempted = False
            fetch_status = 'not_started'
            qradar_data = None

            try:
                # Cleanup old data before fetching new batch
                self.cleanup_old_data()

                fetch_attempted = True
                fetch_status = 'error'
                qradar_data = self.fetch_qradar_data(start_time, end_time)
                if not qradar_data:
                    fetch_status = 'empty'
                    self.logger.warning("No QRadar data retrieved")
                    return None
                fetch_status = 'success'

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

                # Build family-aware feature names and optional rule-name enrichment
                try:
                    feature_names = feature_gen.get_feature_names()
                    dim = feature_gen.get_feature_vector_dimension()
                    # Optional rule name enrichment from config plus production mapping
                    fn_cfg = (self.config.get('detection', {}) or {}).get('feature_names', {}) or {}
                    rule_name_map: Dict[int, str] = {}
                    production_name_map = self.load_production_rule_name_map()
                    include_rule_names = bool(fn_cfg.get('include_rule_names', False)) or bool(production_name_map)

                    if production_name_map:
                        # Use production mapping as authoritative baseline
                        rule_name_map.update(production_name_map)

                    if fn_cfg:
                        # Start with direct name map if provided (may contain overrides)
                        raw_map = fn_cfg.get('name_map') or {}
                        try:
                            for key, value in raw_map.items():
                                rid_int = int(key)
                                rule_name_map[rid_int] = str(value)
                        except Exception:
                            pass

                        if include_rule_names:
                            # Add from CSV paths (fills any remaining gaps)
                            csv_paths = fn_cfg.get('csv_paths', []) or []
                            for path in csv_paths:
                                try:
                                    if path and os.path.exists(path):
                                        import pandas as _pdcsv
                                        df_rules = _pdcsv.read_csv(path)
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
                    # Build index metadata for quick lookup
                    feature_index_meta: List[Dict[str, Any]] = []
                    for j, fname in enumerate(feature_names):
                        meta: Dict[str, Any] = {'feature_name': fname}
                        if isinstance(fname, str) and fname.startswith('rule_'):
                            try:
                                meta['rule_id'] = int(fname.replace('rule_', ''))
                            except Exception:
                                meta['rule_id'] = None
                        elif isinstance(fname, str) and fname.startswith('family_'):
                            meta['family_name'] = fname.replace('family_', '')
                        feature_index_meta.append(meta)
                except Exception:
                    feature_names = [f"feature_{k}" for k in range(int(X.shape[1]))]
                    include_rule_names = False
                    rule_name_map = {}
                    feature_index_meta = []
        
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
                    shap_explainer = ShapExplainer()
                
                    self.logger.info("SHAP explainer initialized and ready for alert explanations")
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
                    # Model metadata for logging
                    model_attr = getattr(predictor, "model", None)
                    if model_attr is not None:
                        model_cls = getattr(model_attr, "__class__", None)
                        model_name = getattr(model_cls, "__name__", str(model_cls)) if model_cls is not None else str(model_attr)
                    else:
                        model_name = "UnknownModel"
                    try:
                        feature_count_total = int(X.shape[1]) if hasattr(X, "shape") else None
                    except Exception:
                        feature_count_total = None

                    for idx, (pred, prob) in enumerate(predictions):
                        is_alert = prob > self.config['detection']['alert_threshold']
                        result = {
                            'timestamp': datetime.now().isoformat(),
                            'hostname': df_agg.iloc[idx]['hostname'],
                            'source_ip': df_agg.iloc[idx]['source_ip'],
                            'window_id': df_agg.iloc[idx]['window_id'],
                            'prediction': int(pred),
                            'probability': float(prob),
                            'alert': bool(is_alert)
                        }
                        results.append(result)

                        label_str = 'malicious' if int(pred) == 1 else 'normal'
                        instances_analyzed = 1

                        # Prepare per-alert feature context
                        payload_top_features: List[Dict[str, Any]] = []
                        row_vec = None
                        try:
                            row_candidate = X[idx]
                            row_vec = np.array(row_candidate, dtype=float, copy=False)
                            if row_vec.ndim > 1:
                                row_vec = row_vec.reshape(-1)
                        except Exception:
                            row_vec = None

                        payload_rule_limit = int((self.config.get('detection', {}) or {}).get('alert_payload_rule_count', 5))
                        if payload_rule_limit <= 0:
                            payload_rule_limit = 5

                        # On alert, add detailed top-10 rules and SHAP explanation
                        if is_alert:
                            try:
                                # Top-10 rules by feature magnitude
                                top_n = int(self.config.get('detection', {}).get('top_rules_count', 10))
                                top_rules: List[Dict[str, Any]] = []
                                top_idx: List[int] = []
                                if row_vec is not None and row_vec.size > 0:
                                    try:
                                        sorted_idx = np.argsort(row_vec)[::-1]
                                        max_count = max(top_n, payload_rule_limit)
                                        top_idx = [int(i) for i in sorted_idx[:max_count]]
                                    except Exception:
                                        top_idx = []
                                for j in top_idx:
                                    try:
                                        val = float(row_vec[j]) if row_vec is not None else 0.0
                                    except Exception:
                                        continue
                                    if val <= 0:
                                        continue
                                    fname = feature_names[j] if j < len(feature_names) else f"feature_{j}"
                                    meta = feature_index_meta[j] if feature_index_meta and j < len(feature_index_meta) else {}
                                    rule_id_val = meta.get('rule_id')
                                    family_name_val = meta.get('family_name')
                                    rule_name = None
                                    if 'include_rule_names' in locals() and include_rule_names and isinstance(rule_id_val, int):
                                        rule_name = rule_name_map.get(int(rule_id_val))

                                    entry: Dict[str, Any] = {
                                        'feature': fname,
                                        'rule_id': rule_id_val if rule_id_val is not None else fname,
                                        'value': val
                                    }
                                    if family_name_val:
                                        entry['family'] = family_name_val
                                    if rule_name is not None:
                                        entry['rule_name'] = rule_name
                                    top_rules.append(entry)

                                top_rules_for_payload: List[Dict[str, Any]] = []
                                if top_rules:
                                    payload_top_features = [
                                        {
                                            'feature': rule_entry.get('feature') or (f"rule_{rule_entry['rule_id']}" if isinstance(rule_entry.get('rule_id'), int) else rule_entry.get('rule_id')),
                                            'rule_id': rule_entry.get('rule_id'),
                                            'rule_name': rule_entry.get('rule_name'),
                                            'importance': rule_entry.get('value')
                                        }
                                        for rule_entry in top_rules[:payload_rule_limit]
                                    ]
                                    top_rules_for_payload = top_rules[:payload_rule_limit]

                                alert_description = (
                                    f"Alert description | {label_str} activity detected "
                                    f"on {result['hostname']} (confidence={result['probability']:.4f})"
                                )

                                # Enhanced SHAP explanation for this specific alert instance
                                shap_success = False
                                enriched_shap = []

                                if shap_explainer is not None and row_vec is not None and row_vec.size > 0:
                                    try:
                                        self.logger.info(f"Applying SHAP explanation for alert instance {result['window_id']}")

                                        # Create single instance data for explanation
                                        instance_data = row_vec.reshape(1, -1)

                                        # Apply SHAP explainer to this specific alert instance
                                        # Use a subset of X as background data for efficiency
                                        background_sample_size = min(100, X.shape[0])
                                        background_indices = np.random.choice(X.shape[0], size=background_sample_size, replace=False)
                                        background_data = X[background_indices]

                                        # SHAP frequent path mining options from config (optional)
                                        shap_cfg = (self.config.get('detection', {}) or {}).get('shap', {}) or {}
                                        fpm_opts = shap_cfg.get('frequent_path_mining', None)

                                        shap_results = shap_explainer.explain(
                                            model=predictor.model,
                                            background_data=background_data,  # Background data for SHAP baseline
                                            instance_data=instance_data,  # Single alert instance to explain
                                            feature_name_list=feature_names,
                                            persist_outputs=False,
                                            plot=False,
                                            plot_in_terminal=True,
                                            summary_report=False,
                                            frequent_path_mining=fpm_opts
                                        )

                                        if shap_results and 'feature_importance' in shap_results:
                                            # Extract top contributing features for this alert
                                            alert_top_features = shap_results['feature_importance'][:top_n]
                                            
                                            # Get raw rule counts for this window to map families back to specific rules
                                            current_window_rules = df_agg.iloc[idx].get('aggregated_rules_dict', {})
                                            if isinstance(current_window_rules, str):
                                                import ast
                                                try:
                                                    current_window_rules = ast.literal_eval(current_window_rules)
                                                except:
                                                    current_window_rules = {}

                                            # Enrich with rule_id and rule_name when available
                                            for fi in alert_top_features:
                                                fname = fi.get('feature', '')
                                                
                                                # Logic to expand Family features into specific rules for the payload
                                                expanded_items = []
                                                if isinstance(fname, str) and fname.startswith('family_'):
                                                    family_name = fname.replace('family_', '')
                                                    found_children = []
                                                    # Find rules in this window that belong to this family
                                                    for r_id, r_count in current_window_rules.items():
                                                        try:
                                                            r_id_int = int(r_id)
                                                            if feature_gen.rule_manager.get_rule_family(r_id_int) == family_name:
                                                                r_name = rule_name_map.get(r_id_int, str(r_id_int))
                                                                found_children.append({
                                                                    'rule_id': r_id_int, 
                                                                    'rule_name': r_name, 
                                                                    'count': r_count
                                                                })
                                                        except:
                                                            continue
                                                    # Sort by count descending and take top 1
                                                    found_children.sort(key=lambda x: x['count'], reverse=True)
                                                    for child in found_children[:1]:
                                                        # Use family's importance for the child
                                                        item = dict(fi)
                                                        item['rule_id'] = child['rule_id']
                                                        item['rule_name'] = child['rule_name']
                                                        item['family'] = family_name
                                                        item['value'] = item.get('importance', 0)
                                                        expanded_items.append(item)
                                                    if not found_children:
                                                        try:
                                                            fallback_ids = feature_gen.rule_manager.get_rules_for_family(family_name, limit=1)
                                                        except Exception:
                                                            fallback_ids = []
                                                        if fallback_ids:
                                                            fallback_rule_id = fallback_ids[0]
                                                            fallback_rule_name = rule_name_map.get(fallback_rule_id, str(fallback_rule_id))
                                                            item = dict(fi)
                                                            item['rule_id'] = fallback_rule_id
                                                            item['rule_name'] = fallback_rule_name
                                                            item['family'] = family_name
                                                            if 'value' not in item:
                                                                item['value'] = item.get('importance', 0)
                                                            expanded_items.append(item)
                                                
                                                # If not a family or no children found, treat as single item
                                                if not expanded_items:
                                                    rid_val = None
                                                    if isinstance(fname, str) and fname.startswith('rule_'):
                                                        try:
                                                            rid_val = int(fname.replace('rule_', ''))
                                                        except Exception:
                                                            rid_val = None
                                                    
                                                    fi_en = dict(fi)
                                                    if rid_val is not None:
                                                        fi_en['rule_id'] = rid_val
                                                        if 'include_rule_names' in locals() and include_rule_names:
                                                            name = rule_name_map.get(int(rid_val))
                                                            if name is not None:
                                                                fi_en['rule_name'] = name
                                                    
                                                    # Ensure 'value' key exists
                                                    if 'importance' in fi_en:
                                                        fi_en['value'] = fi_en['importance']
                                                        
                                                    expanded_items.append(fi_en)

                                                enriched_shap.extend(expanded_items)

                                            # Log SHAP-based feature importance
                                            shap_top_rules = []
                                            shap_top_values = []
                                            


                                            # Expand Family features into specific rules
                                            expanded_features = []
                                            
                                            for feature_info in alert_top_features:
                                                feature_name = feature_info.get('feature', '')
                                                importance = feature_info.get('importance', 0.0)
                                                
                                                # Handle Family Features: Map back to specific rules
                                                if feature_name.startswith('family_'):
                                                    family_name = feature_name.replace('family_', '')
                                                    found_constituent = False
                                                    
                                                    # Find rules in this window that belong to this family
                                                    for r_id, r_count in current_window_rules.items():
                                                        try:
                                                            r_id_int = int(r_id)
                                                            # Check if this rule belongs to the current family
                                                            if feature_gen.rule_manager.get_rule_family(r_id_int) == family_name:
                                                                found_constituent = True
                                                                # Get Rule Name
                                                                r_name = rule_name_map.get(r_id_int, str(r_id_int))
                                                                
                                                                # Create a feature entry for this specific rule
                                                                # We assign the Family's importance to the specific rule
                                                                # This ensures it appears in the top list
                                                                expanded_features.append({
                                                                    'display_name': f"{r_name} (ID:{r_id_int})",
                                                                    'importance': importance,
                                                                    'rule_id': r_id_int
                                                                })
                                                        except:
                                                            continue
                                                    
                                                    # If no specific rules found for this family in this window (rare/edge case),
                                                    # keep the family name so we don't lose the signal
                                                    if not found_constituent:
                                                        try:
                                                            fallback_ids = feature_gen.rule_manager.get_rules_for_family(family_name, limit=1)
                                                        except Exception:
                                                            fallback_ids = []

                                                        if fallback_ids:
                                                            fallback_rule_id = fallback_ids[0]
                                                            fallback_rule_name = rule_name_map.get(fallback_rule_id, str(fallback_rule_id))
                                                            expanded_features.append({
                                                                'display_name': f"{fallback_rule_name} (ID:{fallback_rule_id})",
                                                                'importance': importance,
                                                                'rule_id': fallback_rule_id
                                                            })
                                                        else:
                                                            expanded_features.append({
                                                                'display_name': family_name,
                                                                'importance': importance,
                                                                'rule_id': None
                                                            })

                                                # Handle legacy "rule_" features (if any)
                                                elif feature_name.startswith('rule_'):
                                                    try:
                                                        rule_id = int(feature_name.replace('rule_', ''))
                                                        r_name = rule_name_map.get(rule_id)
                                                        if r_name:
                                                            display_name = f"{r_name} (ID:{rule_id})"
                                                        else:
                                                            display_name = str(rule_id)
                                                        
                                                        expanded_features.append({
                                                            'display_name': display_name,
                                                            'importance': importance,
                                                            'rule_id': rule_id
                                                        })
                                                    except ValueError:
                                                        expanded_features.append({
                                                            'display_name': feature_name,
                                                            'importance': importance,
                                                            'rule_id': None
                                                        })
                                                else:
                                                    # Other features
                                                    expanded_features.append({
                                                        'display_name': feature_name,
                                                        'importance': importance,
                                                        'rule_id': None
                                                    })

                                            # Filter out specific unwanted rules from the output
                                            expanded_features = [
                                                item for item in expanded_features 
                                                if "BOC_OfficeHour" not in item['display_name']
                                            ]

                                            # Sort expanded features by importance (descending) to ensure top rules are first
                                            # Note: If a family had multiple rules, they will all have the same importance
                                            # and appear together.
                                            expanded_features.sort(key=lambda x: x['importance'], reverse=True)
                                            
                                            # Populate the lists for logging
                                            for item in expanded_features:
                                                shap_top_rules.append(item['display_name'])
                                                shap_top_values.append(float(item['importance']))

                                            # Enhanced logging with SHAP results
                                            try:
                                                logging_utils.log_shap_results(
                                                    hostname=result['hostname'],
                                                    source_ip=result['source_ip'],
                                                    window_id=result['window_id'],
                                                    top_rules=shap_top_rules,
                                                    shap_values=shap_top_values,
                                                    prediction='malicious',
                                                    confidence=result['probability']
                                                )

                                                # Additional detailed SHAP logging - SWAPPED CONTENT: Now logs count-based rules
                                                logging_utils.run_log(
                                                    "SHAP_EXPLANATION",
                                                    f"SHAP explanation for alert | hostname={result['hostname']} | window_id={result['window_id']}",
                                                    payload={
                                                        'hostname': result['hostname'],
                                                        'source_ip': result['source_ip'],
                                                        'window_id': result['window_id'],
                                                        'confidence': result['probability'],
                                                        'shap_top_features': top_rules_for_payload, # Swapped: using count-based rules
                                                        'shap_output_files': shap_results.get('output_files', {}),
                                                        'shap_summary': {
                                                            'most_important_feature': enriched_shap[0].get('rule_name', enriched_shap[0].get('feature')) if enriched_shap else None,
                                                            'max_importance_score': alert_top_features[0]['importance'] if alert_top_features else 0,
                                                            'features_analyzed': len(shap_results.get('feature_importance', [])),
                                                            'explanation_timestamp': shap_results.get('timestamp')
                                                        }
                                                    }
                                                )
                                                shap_success = True
                                            except Exception as log_e:
                                                self.logger.warning(f"Failed to log SHAP results for {result['window_id']}: {log_e}")

                                            # Store SHAP results in the result for potential further use
                                            result['shap_explanation'] = {
                                                'top_features': enriched_shap,
                                                'output_files': shap_results.get('output_files', {}),
                                                'explanation_available': True
                                            }

                                            self.logger.info(f"SHAP explanation completed for alert {result['window_id']} - "
                                                           f"Top feature: {alert_top_features[0]['feature'] if alert_top_features else 'N/A'}")

                                        else:
                                            self.logger.warning(f"SHAP explanation returned no results for {result['window_id']}")
                                            result['shap_explanation'] = {'explanation_available': False, 'error': 'No SHAP results'}

                                    except Exception as shap_e:
                                        self.logger.error(f"SHAP explanation failed for alert {result['window_id']}: {shap_e}")
                                        result['shap_explanation'] = {'explanation_available': False, 'error': str(shap_e)}

                                        # Fallback to basic feature importance logging
                                        try:
                                            if top_rules:
                                                basic_rules = [rule['rule_id'] for rule in top_rules[:top_n]]
                                                basic_values = [rule['value'] for rule in top_rules[:top_n]]
                                                logging_utils.log_shap_results(
                                                    hostname=result['hostname'],
                                                    source_ip=result['source_ip'],
                                                    window_id=result['window_id'],
                                                    top_rules=basic_rules,
                                                    shap_values=basic_values,
                                                    prediction='malicious',
                                                    confidence=result['probability']
                                                )
                                        except Exception:
                                            pass

                                else:
                                    self.logger.info(f"SHAP explainer not available for alert {result['window_id']}")
                                    result['shap_explanation'] = {'explanation_available': False, 'error': 'SHAP explainer not initialized'}

                                # Log alert details - SWAPPED CONTENT: Now logs SHAP results (if available), otherwise fallback to counts
                                # Must be logged AFTER SHAP calculation to include SHAP results
                                try:
                                    # Format floating numbers to 3 decimal places specific for payload validation
                                    raw_top_rules = enriched_shap[:payload_rule_limit] if shap_success and enriched_shap else top_rules_for_payload
                                    formatted_top_rules = []
                                    for rule in raw_top_rules:
                                        r_copy = dict(rule)
                                        family_val = r_copy.pop('family', None)
                                        if family_val and r_copy.get('rule_name'):
                                            r_copy['rule_name'] = f"{r_copy['rule_name']}({family_val} family)"
                                        if 'value' in r_copy:
                                            try:
                                                r_copy['value'] = round(float(r_copy['value']), 3)
                                            except (ValueError, TypeError):
                                                pass
                                        formatted_top_rules.append(r_copy)

                                    payload_data = {
                                        'hostname': result['hostname'],
                                        'source_ip': result['source_ip'],
                                        'window_id': result['window_id'],
                                        'confidence': round(float(result['probability']), 3),
                                        # Use SHAP results if successful, otherwise fallback to count-based rules
                                        'top_rules_by_count': formatted_top_rules,
                                    }

                                    logging_utils.run_log(
                                        "ALERT_DETAIL",
                                        f"{alert_description} | "
                                        f"Alert details | hostname={result['hostname']} | window_id={result['window_id']} | confidence={result['probability']:.4f}",
                                        payload=payload_data
                                    )
                                except Exception as e:
                                    self.logger.warning(f"Failed to log ALERT_DETAIL: {e}")

                            except Exception as e:
                                self.logger.warning(f"Alert detail processing failed for {result['window_id']}: {e}")
                                result['shap_explanation'] = {'explanation_available': False, 'error': f'Processing failed: {str(e)}'}

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
                                f"ALERT: Threat detected on {result['hostname']}(IP: {result['source_ip']})(p={prob:.2f})"
                            )
                            if int(pred) == 1:
                                try:
                                    logging_utils.log_detection(
                                        hostname=result['hostname'],
                                        window_id=result['window_id'],
                                        prediction=label_str,
                                        confidence=result['probability'],
                                        model_name=model_name,
                                        feature_count=feature_count_total,
                                        instances_analyzed=instances_analyzed,
                                        top_features=payload_top_features if payload_top_features else None
                                    )
                                except Exception:
                                    pass
                    #summerize alerts count
                    alert_count = sum(1 for r in results if r.get('alert', False))
                    self.logger.info(f"Detection completed. {alert_count} alerts generated in {len(results)} windows.")
                    return results

            except Exception as e:
                self.logger.error(f"Detection pipeline failed: {str(e)}")
                fetch_status = 'error'
                return None

            finally:
                if fetch_attempted:
                    state['last_fetch_epoch'] = time.time()
                    state['last_window_start_epoch'] = start_time.timestamp()
                    state['last_window_end_epoch'] = end_time.timestamp()
                    state['last_fetch_status'] = fetch_status
                    self._persist_detection_state(state)
    
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
