"""
MongoDB Connection Utility (AQL Detection Only)

This manager is specialized for the detection pipeline that ingests QRadar
AQL search results (see AQLjsonResult.json) and stores 30-minute sliding
windows into MongoDB. It focuses on three collections defined in
mongodb/mongodb_config.json:

- aql_events: optional raw AQL event rows (hostname, rule_id, timestamp, count)
- detection_windows (qradar_sliding_windows): aggregated windows with
  feature_vector and host_triggers
- detection_results: model predictions per window (and optional host)

Python 3.6.8 Compatible
"""

import os
import sys
import pymongo
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime, timedelta
import logging

# Add shared_utils to path for shared utilities integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_utils.time_utils import parse_qradar_timestamp, get_window_id, get_window_start_end
from shared_utils.qradar_rule_manager import get_rule_list
from system import logging_utils


class MongoDBConnectionManager:
    """
    MongoDB connection manager for AQL detection-only pipeline.
    
    Provides minimal operations required by the detection path and data loader:
    - connect() / close()
    - create_indexes() for detection collections
    - get_unlabeled_windows(start, end) for fetching AQL windows
    - insert_prediction(data) for saving detection outcomes
    - insert_event(event) supported against aql_events for simple tests
    - get_data_summary() for basic health checks
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MongoDB connection manager.
        
        Args:
            config_path: Path to mongodb_config.json. Defaults to mongodb/mongodb_config.json
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 'mongodb_config.json'
        )
        self.config = self._load_config()
        self.client = None
        self.db = None
        
        # Initialize detection-only collections from config
        # Single canonical key set: aql_events, detection_windows, detection_results
        collection_config = self.config.get('collections', {})
        self.events_collection_name = collection_config.get('aql_events', 'aql_events')
        self.windows_collection_name = collection_config.get('detection_windows', 'qradar_sliding_windows')
        self.predictions_collection_name = collection_config.get('detection_results', 'detection_results')
        # Kept for completeness; not used in AQL-only flow
        self.rule_mappings_collection_name = collection_config.get('rule_mappings', 'rule_mappings')
        
        # Collection references
        self.events_collection = None
        self.windows_collection = None
        self.predictions_collection = None
        self.rule_mappings_collection = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load MongoDB configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logging_utils.run_log("ERROR", f"Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logging_utils.run_log("ERROR", f"Invalid JSON in config file: {e}")
            raise
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB using configuration.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            mongo_config = self.config['mongodb']
            connection_string = mongo_config.get('connection_string', 
                                               f"mongodb://{mongo_config['host']}:{mongo_config['port']}/")
            
            self.client = pymongo.MongoClient(connection_string)
            self.db = self.client[mongo_config['db_name']]
            
            # Initialize collections for unified pipeline
            self.events_collection = self.db[self.events_collection_name]
            self.windows_collection = self.db[self.windows_collection_name]
            self.predictions_collection = self.db[self.predictions_collection_name]
            self.rule_mappings_collection = self.db[self.rule_mappings_collection_name]
            
            # Test connection
            self.client.admin.command('ping')
            logging_utils.run_log("INFO", f"Connected to MongoDB: {mongo_config['db_name']}")
            return True
            
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to connect to MongoDB: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """
        Create optimized indexes for unified pipeline schema.
        
        Returns:
            True if indexes created successfully, False otherwise
        """
        try:
            if not self.db and not self.connect():
                logging_utils.run_log("ERROR", "Cannot create indexes: MongoDB not connected")
                return False

            # aql_events collection indexes (optional raw events)
            events_indexes = [
                ([('timestamp', -1)], "aql_event_ts_idx"),
                ([('rule_id', 1)], "aql_rule_idx"),
                ([('hostname', 1)], "aql_host_idx"),
                ([('window_id', 1)], "aql_window_idx")
            ]

            # detection_windows collection indexes (aggregated windows)
            windows_indexes = [
                ([('window_start', -1), ('window_end', -1)], "det_window_time_idx"),
                ([('window_id', 1)], "det_window_id_idx"),
                ([('query_time', -1)], "det_query_time_idx"),
                ([('total_triggers', -1)], "det_total_triggers_idx")
            ]

            # detection_results collection indexes (predictions)
            predictions_indexes = [
                ([('window_id', 1)], "det_result_window_idx"),
                ([('prediction_time', -1)], "det_result_time_idx"),
                ([('predicted_label', 1)], "det_result_label_idx")
            ]
            
            # Create indexes only if collections exist
            if self.events_collection:
                for index_spec, index_name in events_indexes:
                    try:
                        self.events_collection.create_index(index_spec, name=index_name)
                    except Exception as e:
                        logging_utils.run_log("WARNING", f"Failed to create events index {index_name}: {e}")
            
            if self.windows_collection:
                for index_spec, index_name in windows_indexes:
                    try:
                        # Unique on window_id provides idempotent upserts
                        if index_name == "det_window_id_idx":
                            self.windows_collection.create_index(index_spec, name=index_name, unique=True)
                        else:
                            self.windows_collection.create_index(index_spec, name=index_name)
                    except Exception as e:
                        logging_utils.run_log("WARNING", f"Failed to create windows index {index_name}: {e}")
            
            if self.predictions_collection:
                for index_spec, index_name in predictions_indexes:
                    try:
                        self.predictions_collection.create_index(index_spec, name=index_name)
                    except Exception as e:
                        logging_utils.run_log("WARNING", f"Failed to create predictions index {index_name}: {e}")
            
            logging_utils.run_log("INFO", "MongoDB indexes created successfully for AQL detection")
            return True
            
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to create indexes: {e}")
            return False
    
    def insert_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Insert a single raw AQL event into aql_events (optional helper).
        
        Args:
            event_data: Event data with ['hostname', 'rule_id', 'timestamp', 'count']
            
        Returns:
            True if insertion successful, False otherwise
        """
        try:
            if not self.db and not self.connect():
                logging_utils.run_log("ERROR", "Cannot insert event: MongoDB not connected")
                return False
            
            if not self.events_collection:
                logging_utils.run_log("ERROR", "aql_events collection not initialized")
                return False
            
            # Ensure proper timestamp parsing using time_utils
            if isinstance(event_data.get('timestamp'), str):
                event_data['timestamp'] = parse_qradar_timestamp(event_data['timestamp'])
            
            # Calculate window_id using time_utils
            if 'timestamp' in event_data:
                event_data['window_id'] = get_window_id(event_data['timestamp'])
            
            # Ensure required fields
            required_fields = ['hostname', 'rule_id', 'timestamp', 'count']
            for field in required_fields:
                if field not in event_data:
                    logging_utils.run_log("ERROR", f"Missing required field: {field}")
                    return False
            
            # Ensure correct data types
            event_data['rule_id'] = int(event_data['rule_id'])
            event_data['count'] = int(event_data['count'])
            event_data['hostname'] = str(event_data['hostname'])
            
            # Insert with upsert
            self.events_collection.replace_one(
                {
                    'window_id': event_data['window_id'],
                    'hostname': event_data['hostname'],
                    'rule_id': event_data['rule_id']
                },
                event_data,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to insert event: {e}")
            return False
    
    def batch_insert_events(self, events: List[Dict[str, Any]], 
                           batch_size: int = 1000) -> int:
        """
        Batch insert raw QRadar events with optimized performance.
        
        Args:
            events: List of event documents
            batch_size: Number of documents to process per batch
            
        Returns:
            Number of documents successfully inserted
        """
        if not self.db and not self.connect():
            logging_utils.run_log("ERROR", "Cannot batch insert events: MongoDB not connected")
            return 0
        
        if not self.events_collection:
            logging_utils.run_log("ERROR", "Events collection not initialized")
            return 0
        
        if not events:
            return 0
        
        total_inserted = 0
        
        try:
            for i in range(0, len(events), batch_size):
                batch = events[i:i + batch_size]
                
                # Prepare bulk operations
                operations = []
                for event_data in batch:
                    # Process timestamps
                    if isinstance(event_data.get('timestamp'), str):
                        event_data['timestamp'] = parse_qradar_timestamp(event_data['timestamp'])
                    
                    if 'timestamp' in event_data:
                        event_data['window_id'] = get_window_id(event_data['timestamp'])
                    
                    # Ensure data types
                    event_data['rule_id'] = int(event_data['rule_id'])
                    event_data['count'] = int(event_data['count'])
                    event_data['hostname'] = str(event_data['hostname'])
                    
                    operations.append({
                        'replaceOne': {
                            'filter': {
                                'window_id': event_data['window_id'],
                                'hostname': event_data['hostname'],
                                'rule_id': event_data['rule_id']
                            },
                            'replacement': event_data,
                            'upsert': True
                        }
                    })
                
                if operations:
                    result = self.events_collection.bulk_write(operations)
                    total_inserted += result.upserted_count + result.modified_count
                    
                    if i % (batch_size * 10) == 0:
                        logging_utils.run_log("INFO", f"Processed {i + len(batch)} events...")
            
            logging_utils.run_log("INFO", f"Batch insert completed: {total_inserted} events")
            return total_inserted
            
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed batch insert events: {e}")
            return total_inserted
    
    def insert_window(self, window_data: Dict[str, Any]) -> bool:
        """
        Insert an aggregated AQL detection window into detection_windows.
        
        Args:
            window_data: Window data with keys like window_start/window_end,
                         feature_vector, host_triggers, total_triggers, etc.
            
        Returns:
            True if insertion successful, False otherwise
        """
        try:
            if not self.db and not self.connect():
                logging_utils.run_log("ERROR", "Cannot insert window: MongoDB not connected")
                return False
            
            if not self.windows_collection:
                logging_utils.run_log("ERROR", "detection_windows collection not initialized")
                return False
            
            # Compute window_id if window_start provided
            if 'window_id' not in window_data:
                if 'window_start' in window_data:
                    try:
                        window_data['window_id'] = window_data['window_start'].strftime("%Y-%m-%d_%H-%M-%S")
                    except Exception:
                        pass
                elif 'timestamp' in window_data:
                    # Backward compat: build from a single timestamp
                    event_time = parse_qradar_timestamp(window_data['timestamp'])
                    ws, we = get_window_start_end(event_time)
                    window_data['window_start'] = ws
                    window_data['window_end'] = we
                    window_data['window_id'] = get_window_id(event_time)

            # Ensure required fields
            required_fields = ['window_id', 'window_start', 'window_end']
            for field in required_fields:
                if field not in window_data:
                    logging_utils.run_log("ERROR", f"Missing required field: {field}")
                    return False

            # Normalize feature keys
            if 'feature_vector' not in window_data and 'features' in window_data:
                window_data['feature_vector'] = window_data.get('features', {})
            
            # Insert with upsert
            self.windows_collection.replace_one(
                {'_id': window_data.get('_id', window_data['window_id'])},
                {'_id': window_data.get('_id', window_data['window_id']), **window_data},
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to insert window: {e}")
            return False
    
    def cleanup_old_data(self, retention_days: int = 7) -> int:
        """
        Clean up old data based on retention policy for unified pipeline.
        
        Args:
            retention_days: Number of days to retain data
            
        Returns:
            Number of documents deleted
        """
        if not self.db and not self.connect():
            logging_utils.run_log("ERROR", "Cannot cleanup old data: MongoDB not connected")
            return 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            total_deleted = 0
            
            # Clean events
            if self.events_collection:
                events_query = {'timestamp': {'$lt': cutoff_date}}
                events_count = self.events_collection.count_documents(events_query)
                if events_count > 0:
                    events_result = self.events_collection.delete_many(events_query)
                    total_deleted += events_result.deleted_count
                    logging_utils.run_log("INFO", f"Deleted {events_result.deleted_count} old events")
            
            # Clean windows
            if self.windows_collection:
                windows_query = {'window_start': {'$lt': cutoff_date}}
                windows_count = self.windows_collection.count_documents(windows_query)
                if windows_count > 0:
                    windows_result = self.windows_collection.delete_many(windows_query)
                    total_deleted += windows_result.deleted_count
                    logging_utils.run_log("INFO", f"Deleted {windows_result.deleted_count} old windows")
            
            # Clean old predictions (keep last 30 days)
            if self.predictions_collection:
                predictions_cutoff = datetime.now() - timedelta(days=30)
                predictions_query = {'prediction_time': {'$lt': predictions_cutoff}}
                predictions_count = self.predictions_collection.count_documents(predictions_query)
                if predictions_count > 0:
                    predictions_result = self.predictions_collection.delete_many(predictions_query)
                    total_deleted += predictions_result.deleted_count
                    logging_utils.run_log("INFO", f"Deleted {predictions_result.deleted_count} old predictions")
            
            if total_deleted == 0:
                logging_utils.run_log("INFO", "No old data found to delete")
            
            return total_deleted
                
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to cleanup old data: {e}")
            return 0
    
    def get_events_for_window(self, window_start: datetime, window_end: datetime, 
                            hostname: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve raw events for a specific time window and hostname.
        
        Args:
            window_start: Start of time window
            window_end: End of time window
            hostname: Optional hostname filter
            
        Returns:
            List of events
        """
        if not self.db and not self.connect():
            logging_utils.run_log("ERROR", "Cannot retrieve events: MongoDB not connected")
            return []
        
        if not self.events_collection:
            logging_utils.run_log("ERROR", "Events collection not initialized")
            return []
        
        try:
            from typing import Dict, Any
            query: Dict[str, Any] = {
                'timestamp': {
                    '$gte': window_start,
                    '$lt': window_end
                }
            }
            
            if hostname:
                query['hostname'] = hostname
            
            events = list(self.events_collection.find(query).sort('timestamp', 1))
            return events
            
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to retrieve events: {e}")
            return []
    
    def get_windows_for_training(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve windows with labels (not used in AQL-only mode).
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of windows with labels
        """
        if not self.db and not self.connect():
            logging_utils.run_log("ERROR", "Cannot retrieve training windows: MongoDB not connected")
            return []
        
        if not self.windows_collection:
            logging_utils.run_log("ERROR", "Windows collection not initialized")
            return []
        
        try:
            query = {
                'window_start': {'$gte': start_time},
                'window_end': {'$lte': end_time},
                'label': {'$exists': True}
            }
            
            windows = list(self.windows_collection.find(query).sort('window_start', 1))
            return windows
            
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to retrieve training windows: {e}")
            return []
    
    def get_unlabeled_windows(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve AQL detection windows without labels (standard case).
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of windows without labels
        """
        if not self.db and not self.connect():
            logging_utils.run_log("ERROR", "Cannot retrieve unlabeled windows: MongoDB not connected")
            return []
        
        if not self.windows_collection:
            logging_utils.run_log("ERROR", "Windows collection not initialized")
            return []
        
        try:
            query = {
                'window_start': {'$gte': start_time},
                'window_end': {'$lte': end_time},
                'label': {'$exists': False}
            }
            
            windows = list(self.windows_collection.find(query).sort('window_start', 1))
            return windows
            
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to retrieve unlabeled windows: {e}")
            return []
    
    def insert_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """
        Insert a prediction result into detection_results.
        
        Args:
            prediction_data: Prediction with at least window_id, predicted_label, confidence;
                             hostname optional in AQL-only mode
            
        Returns:
            True if insertion successful, False otherwise
        """
        try:
            if not self.db and not self.connect():
                logging_utils.run_log("ERROR", "Cannot insert prediction: MongoDB not connected")
                return False
            
            if not self.predictions_collection:
                logging_utils.run_log("ERROR", "detection_results collection not initialized")
                return False
            
            # Ensure required fields
            required_fields = ['window_id', 'predicted_label', 'confidence']
            for field in required_fields:
                if field not in prediction_data:
                    logging_utils.run_log("ERROR", f"Missing required field: {field}")
                    return False
            
            # Add prediction timestamp
            prediction_data['prediction_time'] = datetime.now()
            
            # Insert with upsert
            self.predictions_collection.replace_one(
                {'window_id': prediction_data['window_id']},
                prediction_data,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to insert prediction: {e}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of detection data for unified pipeline.
        
        Returns:
            Dictionary with data statistics for all collections
        """
        if not self.db and not self.connect():
            logging_utils.run_log("ERROR", "Cannot get data summary: MongoDB not connected")
            return {}

        try:
            # Initialize results
            events_result = {'total_events': 0, 'unique_hosts': 0, 'unique_rules': 0, 'time_range': {}}
            windows_result = {'total_windows': 0, 'unique_hosts': 0, 'labeled_windows': 0, 'attack_windows': 0, 'normal_windows': 0, 'time_range': {}}
            predictions_result = {'total_predictions': 0, 'unique_hosts': 0}

            # Events collection
            if self.events_collection:
                try:
                    total_events = self.events_collection.count_documents({})
                    if total_events > 0:
                        unique_hosts_events = len(self.events_collection.distinct("hostname"))
                        unique_rules_events = len(self.events_collection.distinct("rule_id"))
                        
                        events_time_range = list(self.events_collection.aggregate([
                            {"$group": {
                                "_id": None,
                                "min_timestamp": {"$min": "$timestamp"},
                                "max_timestamp": {"$max": "$timestamp"}
                            }}
                        ]))
                        
                        events_result = {
                            "total_events": total_events,
                            "unique_hosts": unique_hosts_events,
                            "unique_rules": unique_rules_events,
                            "time_range": events_time_range[0] if events_time_range else {}
                        }
                except Exception as e:
                    logging_utils.run_log("ERROR", f"Failed to get events summary: {e}")

            # Windows collection
            if self.windows_collection:
                try:
                    total_windows = self.windows_collection.count_documents({})
                    if total_windows > 0:
                        unique_hosts_windows = len(self.windows_collection.distinct("hostname"))
                        labeled_windows = self.windows_collection.count_documents({'label': {'$exists': True}})
                        attack_windows = self.windows_collection.count_documents({'label': 1})
                        normal_windows = self.windows_collection.count_documents({'label': 0})
                        
                        windows_time_range = list(self.windows_collection.aggregate([
                            {"$group": {
                                "_id": None,
                                "min_window_start": {"$min": "$window_start"},
                                "max_window_end": {"$max": "$window_end"}
                            }}
                        ]))
                        
                        windows_result = {
                            "total_windows": total_windows,
                            "unique_hosts": unique_hosts_windows,
                            "labeled_windows": labeled_windows,
                            "attack_windows": attack_windows,
                            "normal_windows": normal_windows,
                            "time_range": windows_time_range[0] if windows_time_range else {}
                        }
                except Exception as e:
                    logging_utils.run_log("ERROR", f"Failed to get windows summary: {e}")

            # Predictions collection
            if self.predictions_collection:
                try:
                    total_predictions = self.predictions_collection.count_documents({})
                    if total_predictions > 0:
                        unique_hosts_predictions = len(self.predictions_collection.distinct("hostname"))
                        predictions_result = {
                            "total_predictions": total_predictions,
                            "unique_hosts": unique_hosts_predictions
                        }
                except Exception as e:
                    logging_utils.run_log("ERROR", f"Failed to get predictions summary: {e}")

            summary = {
                "events_collection": events_result,
                "windows_collection": windows_result,
                "predictions_collection": predictions_result,
                "overall": {
                    "total_hosts": max(events_result.get("unique_hosts", 0), windows_result.get("unique_hosts", 0)),
                    "training_ready": windows_result.get("labeled_windows", 0) > 0
                }
            }

            return summary

        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to get data summary: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logging_utils.run_log("INFO", "MongoDB connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def get_mongodb_manager(config_path: Optional[str] = None) -> MongoDBConnectionManager:
    """
    Factory function to get MongoDB connection manager.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        MongoDBConnectionManager instance
    """
    return MongoDBConnectionManager(config_path)


if __name__ == "__main__":
    # Test the MongoDB connection utility for unified pipeline
    print("Testing MongoDB Connection Utility for Unified Pipeline...")
    print("=" * 70)
    
    # Test connection
    with get_mongodb_manager() as manager:
        # Create indexes
        manager.create_indexes()
        
        # Get data summary
        summary = manager.get_data_summary()
        print("Data Summary:")
        for collection, data in summary.items():
            print(f"  {collection}: {data}")
        
        # Test with sample unified schema data
        from datetime import datetime
        
        # Sample raw event
        sample_event = {
            "hostname": "192.168.153.166",
            "rule_id": 100227,
            "timestamp": datetime(2025, 8, 8, 10, 15, 0),
            "count": 6561
        }
        
        # Sample aggregated window
        sample_window = {
            "hostname": "192.168.153.166",
            "timestamp": datetime(2025, 8, 8, 10, 30, 0),
            "features": {"100227": 211656, "100221": 211656},
            "label": 1  # 1 for attack, 0 for normal
        }
        
        # Sample prediction
        sample_prediction = {
            "window_id": "2025-08-08_10-00-00_W20",
            "hostname": "192.168.153.166",
            "predicted_label": 1,
            "confidence": 0.95,
            "top_features": ["100227", "100221"]
        }
        
        # Test event insertion
        event_success = manager.insert_event(sample_event)
        print(f"Test event insert: {'Success' if event_success else 'Failed'}")
        
        # Test window insertion
        window_success = manager.insert_window(sample_window)
        print(f"Test window insert: {'Success' if window_success else 'Failed'}")
        
        # Test prediction insertion
        prediction_success = manager.insert_prediction(sample_prediction)
        print(f"Test prediction insert: {'Success' if prediction_success else 'Failed'}")
        
        # Test data retrieval
        windows = manager.get_windows_for_training(
            datetime(2025, 8, 8, 0, 0, 0),
            datetime(2025, 8, 8, 23, 59, 59)
        )
        print(f"Retrieved training windows: {len(windows)}")
        
        # Test cleanup
        # cleanup_count = manager.cleanup_old_data(retention_days=1)
        # print(f"Cleanup deleted: {cleanup_count} documents")
