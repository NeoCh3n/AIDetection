#!/usr/bin/env python3
"""
Unified Ransomware Detection Pipeline
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
from pipeline.feature_generator import FeatureGenerator
from model_training import train_model
from model_evaluation import evaluate_model
from model_predictor import Predictor

# Import existing API/MongoDB modules
try:
    from create_searches_Qradar import create_searches_Qradar
    from status_searches_Qradar import status_searches_Qradar
    from result_searches_Qradar import result_searches_Qradar
    from insert_DB import insert_DB
    from delete_searches_Qradar import delete_searches_Qradar
    from delete_DB import delete_old_rule_triggers
    from query_DB import query_DB
except ImportError as e:
    logging.warning(f"API/MongoDB modules not found: {e}")

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
        """Load unified configuration"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.py')
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default configuration
            return {
                "training": {
                    "data_path": "./Training_data",
                    "model_path": "./model/ransomware_detector.joblib",
                    "test_size": 0.2,
                    "random_state": 42
                },
                "detection": {
                    "qradar_config": {
                        "host": "192.168.153.123",
                        "token": "677f60e2-3d58-4275-a1f0-c13d1975fdbe",
                        "query_30min": """SELECT "qidEventId" as rule_id, "sysmon_hostname" as hostname, "startTime" as timestamp, COUNT(*) as count FROM events WHERE "startTime" >= '{start_time}' AND "startTime" < '{end_time}' GROUP BY rule_id, hostname, timestamp"""
                    },
                    "mongodb_config": {
                        "connection_string": "mongodb://localhost:27017/",
                        "database": "qradar_ml",
                        "collection": "detection_data"
                    },
                    "retention_days": 7,
                    "alert_threshold": 0.8
                }
            }
    
    def run_training(self):
        """Training mode: CSV → Model"""
        self.logger.info("Starting training pipeline...")
        
        try:
            # Load training data
            self.logger.info("Loading training data...")
            df = load_data('train', self.config['training'])
            self.logger.info(f"Loaded {len(df)} training records")
            
            # Aggregate to 30-minute windows
            self.logger.info("Aggregating to 30-minute windows...")
            df_agg = aggregate_to_windows(df)
            self.logger.info(f"Created {len(df_agg)} aggregated windows")
            
            # Generate feature vectors
            self.logger.info("Generating feature vectors...")
            feature_gen = FeatureGenerator()
            feature_gen.initialize_rules()
            X, y = feature_gen.generate_feature_vectors(df_agg, mode='train')
            self.logger.info(f"Feature matrix shape: {X.shape}")
            
            # Train model
            self.logger.info("Training model...")
            model = train_model(X, y, self.config['training'])
            
            # Evaluate model
            self.logger.info("Evaluating model...")
            rule_manager = QRadarRuleManager(
                mode=self.config['rule_manager']['mode'],
                config=self.config['rule_manager']['api_config']
            )
            rule_list = rule_manager.get_rule_list()
            evaluate_model(
                model_path=self.config['training']['model_path'],
                test_data_path=None,  # Use split from training
                rule_list=rule_list
            )
            
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
                qradar_address=self.config['detection']['qradar_config']['host'],
                search_id=search_id
            )
            
            # Cleanup search
            delete_searches_Qradar(
                qradar_address=self.config['detection']['qradar_config']['host'],
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
            # Process data into MongoDB format
            documents = []
            if data and 'events' in data:
                for event in data['events']:
                    doc = {
                        'rule_id': str(event.get('rule_id', '')),
                        'hostname': str(event.get('hostname', 'unknown')),
                        'timestamp': event.get('timestamp'),
                        'count': int(event.get('count', 0)),
                        'window_start': start_time.isoformat(),
                        'window_end': end_time.isoformat(),
                        'inserted_at': datetime.now().isoformat()
                    }
                    documents.append(doc)
            
            # Insert into MongoDB
            if documents:
                insert_DB.insert_documents(
                    documents,
                    connection_string=self.config['detection']['mongodb_config']['connection_string'],
                    db_name=self.config['detection']['mongodb_config']['database'],
                    collection_name=self.config['detection']['mongodb_config']['collection']
                )
                self.logger.info(f"Inserted {len(documents)} documents")
            
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
            predictor = Predictor(self.config['training']['model_path'])
            predictions = predictor.predict(X)
            
            # Process results
            results = []
            for idx, (pred, prob) in enumerate(predictions):
                if prob > self.config['detection']['alert_threshold']:
                    result = {
                        'timestamp': datetime.now().isoformat(),
                        'hostname': df_agg.iloc[idx]['hostname'],
                        'window_id': df_agg.iloc[idx]['window_id'],
                        'prediction': int(pred),
                        'probability': float(prob),
                        'alert': prob > self.config['detection']['alert_threshold']
                    }
                    results.append(result)
                    self.logger.warning(f"ALERT: Ransomware detected on {result['hostname']} (probability: {prob:.2f})")
            
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