"""
Unified MongoDB Connection Utility for Ransomware Detection Pipeline

This module provides a centralized MongoDB connection utility that integrates
with time_utils.py for consistent timestamp processing across both training
and detection modes.

Python 3.6.8 Compatible
"""

import os
import sys
import pymongo
from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta

# Add shared_utils to path for time_utils integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'system'))

from time_utils import parse_qradar_timestamp, get_window_id, get_window_start_end
import run_log


class MongoDBConnectionManager:
    """
    Unified MongoDB connection manager for ransomware detection pipeline.
    
    Provides centralized database operations with consistent timestamp handling
    using time_utils.py across both training and detection modes.
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
        self.detection_windows_collection = None
        self.detection_results_collection = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load MongoDB configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            run_log.run_log("ERROR", f"Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            run_log.run_log("ERROR", f"Invalid JSON in config file: {e}")
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
            
            # Initialize collections
            collection_config = self.config['collections']
            self.detection_windows_collection = self.db[collection_config['detection_windows']]
            self.detection_results_collection = self.db[collection_config['detection_results']]
            
            # Test connection
            self.client.admin.command('ping')
            run_log.run_log("INFO", f"Connected to MongoDB: {mongo_config['db_name']}")
            return True
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to connect to MongoDB: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """
        Create optimized indexes for detection pipeline.
        
        Returns:
            True if indexes created successfully, False otherwise
        """
        try:
            # Detection windows collection indexes
            indexes = [
                ([("window_start", -1), ("window_end", -1)], "window_time_idx"),
                ([("query_time", -1)], "query_time_idx"),
                ([("host_triggers", 1)], "host_triggers_idx"),
                ([("total_triggers", -1)], "total_triggers_idx"),
                ([("window_start", 1), ("window_end", 1)], "window_range_idx")
            ]
            
            for index_spec, index_name in indexes:
                self.detection_windows_collection.create_index(index_spec, name=index_name)
            
            run_log.run_log("INFO", "MongoDB indexes created successfully")
            return True
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to create indexes: {e}")
            return False
    
    def insert_detection_window(self, window_data: Dict[str, Any]) -> bool:
        """
        Insert a single detection window with time_utils integration.
        
        Args:
            window_data: Detection window data to insert
            
        Returns:
            True if insertion successful, False otherwise
        """
        try:
            if not self.db:
                if not self.connect():
                    return False
            
            # Ensure proper window ID and timestamps using time_utils
            if 'window_start' not in window_data or 'window_end' not in window_data:
                # Use time_utils to calculate window boundaries
                if 'timestamp' in window_data:
                    event_time = parse_qradar_timestamp(window_data['timestamp'])
                    window_start, window_end = get_window_start_end(event_time)
                    window_data['window_start'] = window_start
                    window_data['window_end'] = window_end
                    window_data['_id'] = get_window_id(event_time)
            
            # Ensure query_time is set
            if 'query_time' not in window_data:
                window_data['query_time'] = datetime.now()
            
            # Insert with upsert to handle duplicates
            self.detection_windows_collection.replace_one(
                {'_id': window_data['_id']},
                window_data,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to insert detection window: {e}")
            return False
    
    def batch_insert_detection_windows(self, windows: List[Dict[str, Any]], 
                                     batch_size: int = 1000) -> int:
        """
        Batch insert detection windows with optimized performance.
        
        Args:
            windows: List of detection window documents
            batch_size: Number of documents to process per batch
            
        Returns:
            Number of documents successfully inserted/updated
        """
        if not self.db:
            if not self.connect():
                return 0
        
        total_inserted = 0
        
        try:
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size]
                
                # Prepare bulk operations
                operations = []
                for window_data in batch:
                    # Process timestamps using time_utils
                    if 'timestamp' in window_data:
                        event_time = parse_qradar_timestamp(window_data['timestamp'])
                        window_start, window_end = get_window_start_end(event_time)
                        window_data['window_start'] = window_start
                        window_data['window_end'] = window_end
                        window_data['_id'] = get_window_id(event_time)
                    
                    if 'query_time' not in window_data:
                        window_data['query_time'] = datetime.now()
                    
                    operations.append({
                        'replaceOne': {
                            'filter': {'_id': window_data['_id']},
                            'replacement': window_data,
                            'upsert': True
                        }
                    })
                
                if operations:
                    result = self.detection_windows_collection.bulk_write(operations)
                    total_inserted += result.upserted_count + result.modified_count
                    
                    if i % (batch_size * 10) == 0:
                        run_log.run_log("INFO", f"Processed {i + len(batch)} detection windows...")
            
            run_log.run_log("INFO", f"Batch insert completed: {total_inserted} documents")
            return total_inserted
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed batch insert: {e}")
            return total_inserted
    
    def get_detection_windows(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve detection windows within a time range using time_utils.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of detection windows
        """
        if not self.db:
            if not self.connect():
                return []
        
        try:
            # Ensure timezone consistency
            if start_time.tzinfo is None:
                from pytz import UTC
                start_time = UTC.localize(start_time)
            if end_time.tzinfo is None:
                from pytz import UTC
                end_time = UTC.localize(end_time)
            
            query = {
                'window_start': {'$gte': start_time},
                'window_end': {'$lte': end_time}
            }
            
            windows = list(self.detection_windows_collection.find(query).sort('window_start', 1))
            return windows
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to retrieve detection windows: {e}")
            return []
    
    def cleanup_old_data(self, retention_days: int = 7) -> int:
        """
        Clean up old detection data based on retention policy.
        
        Args:
            retention_days: Number of days to retain data
            
        Returns:
            Number of documents deleted
        """
        if not self.db:
            if not self.connect():
                return 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            query = {'window_start': {'$lt': cutoff_date}}
            count_to_delete = self.detection_windows_collection.count_documents(query)
            
            if count_to_delete > 0:
                result = self.detection_windows_collection.delete_many(query)
                run_log.run_log("INFO", f"Deleted {result.deleted_count} old detection windows")
                return result.deleted_count
            else:
                run_log.run_log("INFO", "No old detection windows found to delete")
                return 0
                
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to cleanup old data: {e}")
            return 0
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of detection data with time_utils integration.
        
        Returns:
            Dictionary with data statistics
        """
        if not self.db:
            if not self.connect():
                return {}
        
        try:
            # Basic counts
            total_docs = self.detection_windows_collection.count_documents({})
            unique_hosts = len(self.detection_windows_collection.distinct("host_triggers"))
            
            # Time range analysis
            time_range = list(self.detection_windows_collection.aggregate([
                {"$group": {
                    "_id": None,
                    "min_window_start": {"$min": "$window_start"},
                    "max_window_end": {"$max": "$window_end"}
                }}
            ]))
            
            # Rule statistics using feature_vector
            rule_stats = list(self.detection_windows_collection.aggregate([
                {"$project": {"rules": {"$objectToArray": "$feature_vector"}}},
                {"$unwind": "$rules"},
                {"$group": {
                    "_id": None,
                    "unique_rules": {"$addToSet": "$rules.k"},
                    "total_rule_triggers": {"$sum": "$rules.v"}
                }}
            ]))
            
            # Host-level statistics
            host_stats = list(self.detection_windows_collection.aggregate([
                {"$project": {"hosts": {"$objectToArray": "$host_triggers"}}},
                {"$unwind": "$hosts"},
                {"$group": {
                    "_id": "$hosts.k",
                    "total_triggers": {"$sum": "$hosts.v.total_triggers"}
                }},
                {"$group": {
                    "_id": None,
                    "total_unique_hosts": {"$sum": 1},
                    "total_host_triggers": {"$sum": "$total_triggers"}
                }}
            ]))
            
            summary = {
                "total_detection_windows": total_docs,
                "unique_hosts": host_stats[0]["total_unique_hosts"] if host_stats else 0,
                "total_host_triggers": host_stats[0]["total_host_triggers"] if host_stats else 0,
                "window_range": time_range[0] if time_range else {},
                "unique_rules": len(rule_stats[0]["unique_rules"]) if rule_stats else 0,
                "total_rule_triggers": rule_stats[0]["total_rule_triggers"]) if rule_stats else 0
            }
            
            return summary
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to get data summary: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            run_log.run_log("INFO", "MongoDB connection closed")
    
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
    # Test the MongoDB connection utility
    print("Testing MongoDB Connection Utility...")
    print("=" * 50)
    
    # Test connection
    with get_mongodb_manager() as manager:
        # Create indexes
        manager.create_indexes()
        
        # Get data summary
        summary = manager.get_data_summary()
        print(f"Data Summary: {summary}")
        
        # Test with sample AQL data
        from datetime import datetime
        sample_window = {
            "window_start": datetime(2025, 8, 8, 10, 0, 0),
            "window_end": datetime(2025, 8, 8, 10, 30, 0),
            "feature_vector": {"100227": 211656, "100221": 211656},
            "host_triggers": {
                "192.168.153.166": {"total_triggers": 6561, "rules": {"100227": 6561}},
                "DESKTOP-64-EDR": {"total_triggers": 5610, "rules": {"100227": 5610}}
            },
            "total_triggers": 217266,
            "total_rules_triggered": 2
        }
        
        success = manager.insert_detection_window(sample_window)
        print(f"Test insert: {'Success' if success else 'Failed'}")