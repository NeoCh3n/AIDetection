#!/usr/bin/env python3
"""
Unified Threat Detection Pipeline
Single orchestrator for training and detection modes
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
import logging

# Add system to path for logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'system'))
from system import logging_utils

# Add parent directories to path for imports
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
    def __init__(self, mode='detect', config_path=None):
        self.mode = mode
        self.config = self.load_config(config_path)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup unified logging for pipeline"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'running_log')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'pipeline_{self.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
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
                        "query_30min": "SELECT \"qidEventId\" as rule_id, \"sysmon_hostname\" as hostname, \"startTime\" as timestamp, COUNT(*) as count FROM events WHERE \"startTime\" >= '{start_time}' AND \"startTime\" < '{end_time}' GROUP BY rule_id, hostname, timestamp"
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
            # Construct AQL query
            aql = self.config['detection']['qradar_config']['query_30min'].format(
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=end_time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Create search
            search_response = create_searches_Qradar(
                qradar_address=self.config['detection']['qradar_config']['host'],
                AQL=aql
            )
            
            if not search_response:
                self.logger.error("Failed to create QRadar search")
                return None
            
            search_id = search_response.get('search_id')
            self.logger.info(f"Created search: {search_id}")
            
            # Wait for completion and get results
            # Note: This is a simplified flow - implement polling in production
            import time
            time.sleep(10)  # Wait for search completion
            
            result_data = result_searches_Qradar(
                Qradar_address=self.config['detection']['qradar_config']['host'],
                search_id=search_id
            )
            
            # Cleanup search
            delete_searches_Qradar(
                Qradar_address=self.config['detection']['qradar_config']['host'],
                search_id=search_id
            )
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch QRadar data: {str(e)}")
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
            
            # Aggregate features
            df_agg = aggregate_to_windows(df)
            if df_agg.empty:
                self.logger.warning("No aggregated windows")
                return None
            
            # Generate features
            feature_gen = FeatureGenerator()
            feature_gen.initialize_rules()
            X = feature_gen.generate_feature_vectors(df_agg, mode='detect')
            
            # Make predictions
            try:
                from model_predictor import Predictor as ModelPredictor
            except Exception as e:
                self.logger.error(f"Predictor not available: {e}. Ensure model_predictor.py exists and the model path is correct.")
                return None

            predictor = ModelPredictor(self.config['training']['model_path'])
            predictions = predictor.predict(X)
            
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
                def __enter__(self_inner):
                    return None
                def __exit__(self_inner, exc_type, exc, tb):
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
    parser = argparse.ArgumentParser(description='Unified Ransomware Detection Pipeline')
    parser.add_argument('mode', choices=['train', 'detect'], help='Pipeline mode')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    pipeline = UnifiedPipeline(mode=args.mode, config_path=args.config)
    result = pipeline.execute()
    
    if result:
        logging_utils.run_log("INFO", f"Pipeline completed successfully in {args.mode} mode")
        sys.exit(0)
    else:
        logging_utils.run_log("ERROR", f"Pipeline failed in {args.mode} mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
