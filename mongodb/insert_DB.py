import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from system import run_log
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Generator
import delete_DB  # # Import delete module for data lifecycle management

#### MongoDB configuration loaded from config file
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')

def load_config():
    """Load configuration from mongodb_config.json"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        run_log.run_log("ERROR", f"Failed to load config: {str(e)}")
        return None

# Load configuration
config = load_config()
if config:
    CONNECTION_STRING_default = config['mongodb']['connection_string']
    NAME_DB_default = config['mongodb']['db_name']
    COLLECTION_default = config['collections']['detection_windows']
    RETENTION_DAYS = config['pipeline']['retention_days']
else:
    # Fallback defaults
    CONNECTION_STRING_default = "mongodb://localhost:27017/"
    NAME_DB_default = "qradar_detection"
    COLLECTION_default = "qradar_sliding_windows"
    RETENTION_DAYS = 7

# Import get_DB with corrected path
import get_DB

#### Data processing constants
MAX_BSON_SIZE = 16 * 1024 * 1024  # # 16MB MongoDB document limit
BATCH_SIZE = 1000  # # Batch size for bulk operations

class QRadarDataProcessor:
    """
    Process real QRadar rule trigger data for ML training and testing
    # # Handles production data from QRadar searches (6000+ hosts, 1+ month)
    """
    
    def __init__(self, connection_string=CONNECTION_STRING_default, db_name=NAME_DB_default):
        self.connection_string = connection_string
        self.db_name = db_name
        self.db = None
        self.collection = None
        
    def connect(self):
        """# # Connect to MongoDB for offline deployment"""
        self.db = get_DB.get_database()
        if self.db is not None:
            self.collection = self.db[COLLECTION_default]
            return True
        return False
    
    def create_production_indexes(self):
        """
        # # Create optimized indexes for detection windows
        """
        if not self.connect():
            return False
        
        try:
            # # Primary time window index
            self.collection.create_index([
                ("window_start", -1),
                ("window_end", -1)
            ], name="window_time_idx")
            
            # # Query time index for sliding windows
            self.collection.create_index([
                ("query_time", -1)
            ], name="query_time_idx")
            
            # # Host-based detection queries
            self.collection.create_index([
                ("host_triggers", 1)
            ], name="host_triggers_idx")
            
            # # Rule trigger analysis
            self.collection.create_index([
                ("total_triggers", -1)
            ], name="total_triggers_idx")
            
            # # Composite index for window lookups
            self.collection.create_index([
                ("window_start", 1),
                ("window_end", 1)
            ], name="window_range_idx")
            
            run_log.run_log("INFO", "# # Detection window indexes created successfully")
            return True
            
        except Exception as e:
            run_log.run_log("ERROR", f"# # Failed to create indexes: {str(e)}")
            return False

    def parse_qradar_search_result(self, result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        # # Parse QRadar search results into sliding window detection format
        # # Uses time_utils.py for consistent timestamp processing
        """
        # Import time_utils for consistent processing
        from shared_utils.time_utils import parse_qradar_timestamp, get_window_id, get_window_start_end
        
        # Group events by 30-minute windows with host-level breakdown
        window_groups = {}
        
        if 'events' not in result_data:
            run_log.run_log("WARNING", "# # No events found in QRadar result")
            return []
            
        run_log.run_log("INFO", f"# # Processing {len(result_data['events'])} QRadar events")
        
        # Group events by 30-minute window with host breakdown
        for event in result_data['events']:
            try:
                # Extract real AQL data fields
                rule_id = str(int(event.get('Custom Rule', 0)))
                count = int(event.get('Count', 0))
                hostname = str(event.get('sysmon_hostname (custom)', 'global'))
                
                # Parse timestamp using time_utils
                event_time_str = event.get('Log Source Time (Minimum)', '')
                if not event_time_str:
                    continue
                
                event_time = parse_qradar_timestamp(event_time_str)
                window_id = get_window_id(event_time, 30)
                window_start, window_end = get_window_start_end(event_time, 30)
                
                # Initialize window group if new
                if window_id not in window_groups:
                    window_groups[window_id] = {
                        '_id': window_id,
                        'window_start': window_start,
                        'window_end': window_end,
                        'query_time': datetime.now(),
                        'feature_vector': {},
                        'rule_counts': {},
                        'host_triggers': {},
                        'total_triggers': 0,
                        'total_rules_triggered': 0
                    }
                
                # Update feature vector (aggregate counts across all hosts)
                window_groups[window_id]['feature_vector'][rule_id] = (
                    window_groups[window_id]['feature_vector'].get(rule_id, 0) + count
                )
                
                # Update rule counts
                window_groups[window_id]['rule_counts'][rule_id] = (
                    window_groups[window_id]['rule_counts'].get(rule_id, 0) + count
                )
                
                # Update host-level breakdown
                if hostname not in window_groups[window_id]['host_triggers']:
                    window_groups[window_id]['host_triggers'][hostname] = {
                        'total_triggers': 0,
                        'rules': {}
                    }
                
                window_groups[window_id]['host_triggers'][hostname]['total_triggers'] += count
                window_groups[window_id]['host_triggers'][hostname]['rules'][rule_id] = (
                    window_groups[window_id]['host_triggers'][hostname]['rules'].get(rule_id, 0) + count
                )
                
                # Update totals
                window_groups[window_id]['total_triggers'] += count
                window_groups[window_id]['total_rules_triggered'] = len(window_groups[window_id]['feature_vector'])
                
            except Exception as e:
                run_log.run_log("WARNING", f"# # Failed to parse event: {str(e)}")
                continue
        
        # Convert window groups to list of documents
        documents = list(window_groups.values())
        run_log.run_log("INFO", f"# # Created {len(documents)} detection windows")
        return documents
    
    
    def insert_qradar_data(self, documents: List[Dict[str, Any]], batch_size: int = BATCH_SIZE):
        """
        # # Insert processed QRadar data in batches
        # # Handles production scale efficiently
        """
        if not self.connect():
            return 0
            
        total_inserted = 0
        
        try:
            # # Process in batches for memory efficiency
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # # Insert with upsert to handle duplicates
                operations = []
                for doc in batch:
                    operations.append({
                        'replaceOne': {
                            'filter': {'_id': doc['_id']},
                            'replacement': doc,
                            'upsert': True
                        }
                    })
                
                result = self.collection.bulk_write(operations)
                total_inserted += result.upserted_count + result.modified_count
                
                if i % (batch_size * 10) == 0:
                    run_log.run_log("INFO", f"# # Processed {i + len(batch)} documents...")
            
            run_log.run_log("INFO", f"# # Completed: {total_inserted} documents inserted/updated")
            return total_inserted
            
        except Exception as e:
            run_log.run_log("ERROR", f"# # Failed to insert QRadar data: {str(e)}")
            return total_inserted
    
    def process_qradar_json_files(self, json_files: List[str]):
        """
        # # Process QRadar JSON files for detection-only pipeline
        # # Creates 30-minute detection windows without training data splitting
        """
        all_documents = []
        
        # # Process each JSON file
        for json_file in json_files:
            try:
                run_log.run_log("INFO", f"# # Processing QRadar file: {json_file}")
                
                with open(json_file, 'r') as f:
                    result_data = json.load(f)
                
                # # Parse QRadar data into detection windows
                documents = self.parse_qradar_search_result(result_data)
                all_documents.extend(documents)
                
                run_log.run_log("INFO", f"# # Processed {len(documents)} detection windows from {json_file}")
                
            except Exception as e:
                run_log.run_log("ERROR", f"# # Failed to process {json_file}: {str(e)}")
                continue
        
        # # Insert all processed data
        if all_documents:
            total_count = self.insert_qradar_data(all_documents)
            return total_count
        else:
            run_log.run_log("WARNING", "# # No documents to insert")
            return 0
    
    def cleanup_old_data(self, retention_days: int = 7):
        """
        # # Clean up old detection windows based on retention policy
        # # Integrates with delete_DB module for data lifecycle management
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            run_log.run_log("INFO", f"# # Starting cleanup: deleting detection windows older than {retention_days} days")
            
            if not self.connect():
                return 0
            
            # # Query to find old detection windows
            query = {'window_start': {'$lt': cutoff_date}}
            
            # # Count documents to be deleted
            count_to_delete = self.collection.count_documents(query)
            
            if count_to_delete > 0:
                # # Delete old documents
                result = self.collection.delete_many(query)
                run_log.run_log("INFO", f"# # Deleted {result.deleted_count} old detection windows")
                return result.deleted_count
            else:
                run_log.run_log("INFO", "# # No old detection windows found to delete")
                return 0
                
        except Exception as e:
            run_log.run_log("ERROR", f"# # Failed to cleanup old data: {str(e)}")
            return 0

    def check_storage_usage(self):
        """
        # # Check current storage usage and data distribution
        """
        if not self.connect():
            return {}
            
        try:
            # # Database statistics
            db_stats = self.db.command("dbStats")
            collection_stats = self.db.command("collStats", COLLECTION_default)
            
            # # Document counts by age
            now = datetime.now()
            age_buckets = {
                "last_24h": now - timedelta(hours=24),
                "last_7d": now - timedelta(days=7),
                "last_30d": now - timedelta(days=30)
            }
            
            age_counts = {}
            for bucket_name, cutoff_date in age_buckets.items():
                count = self.collection.count_documents({
                    'window_start': {'$gte': cutoff_date}
                })
                age_counts[bucket_name] = count
            
            storage_info = {
                "database_size_mb": round(db_stats.get("dataSize", 0) / (1024 * 1024), 2),
                "collection_size_mb": round(collection_stats.get("size", 0) / (1024 * 1024), 2),
                "total_documents": collection_stats.get("count", 0),
                "average_doc_size": round(collection_stats.get("avgObjSize", 0), 2),
                "age_distribution": age_counts
            }
            
            run_log.run_log("INFO", f"# # Storage usage: {storage_info['database_size_mb']} MB")
            return storage_info
            
        except Exception as e:
            run_log.run_log("ERROR", f"# # Failed to check storage usage: {str(e)}")
            return {}

    def manage_data_lifecycle(self, auto_cleanup: bool = True, retention_days: int = 7):
        """
        # # Manage complete data lifecycle: insert, retention, cleanup
        # # Default 7-day retention for ML training requirements
        """
        lifecycle_info = {
            "cleanup_performed": False,
            "documents_deleted": 0,
            "storage_before": {},
            "storage_after": {}
        }
        
        try:
            # # Check storage before cleanup
            lifecycle_info["storage_before"] = self.check_storage_usage()
            
            if auto_cleanup:
                # # Perform automatic cleanup
                deleted_count = self.cleanup_old_data(retention_days)
                lifecycle_info["cleanup_performed"] = True
                lifecycle_info["documents_deleted"] = deleted_count
                
                # # Check storage after cleanup
                lifecycle_info["storage_after"] = self.check_storage_usage()
                
                run_log.run_log("INFO", f"# # Data lifecycle management completed: {deleted_count} documents cleaned")
            
            return lifecycle_info
            
        except Exception as e:
            run_log.run_log("ERROR", f"# # Failed data lifecycle management: {str(e)}")
            return lifecycle_info

    def get_data_summary(self) -> Dict[str, Any]:
        """
        # # Get comprehensive summary of QRadar data
        """
        if not self.connect():
            return {}
            
        try:
            # # Basic counts
            total_docs = self.collection.count_documents({})
            unique_hosts = len(self.collection.distinct("host_triggers"))
            
            # # Time range for detection windows
            time_range = list(self.collection.aggregate([
                {"$group": {
                    "_id": None,
                    "min_window_start": {"$min": "$window_start"},
                    "max_window_end": {"$max": "$window_end"}
                }}
            ]))
            
            # # Rule statistics from feature vectors
            rule_stats = list(self.collection.aggregate([
                {"$project": {"rules": {"$objectToArray": "$feature_vector"}}},
                {"$unwind": "$rules"},
                {"$group": {
                    "_id": None,
                    "unique_rules": {"$addToSet": "$rules.k"},
                    "total_rule_triggers": {"$sum": "$rules.v"}
                }}
            ]))
            
            # # Host statistics
            host_stats = list(self.collection.aggregate([
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
                "total_rule_triggers": rule_stats[0]["total_rule_triggers"] if rule_stats else 0
            }
            
            return summary
            
        except Exception as e:
            run_log.run_log("ERROR", f"# # Failed to get data summary: {str(e)}")
            return {}

    def integrate_with_delete_db(self, retention_days: int = 30):
        """
        # # Integrate with delete_DB module for coordinated data management
        """
        try:
            # # Use delete_DB module for cleanup with ML training retention period
            deleted_count = delete_DB.delete_old_rule_triggers(retention_days)
            run_log.run_log("INFO", f"# # Integrated cleanup completed: {deleted_count} documents deleted")
            return deleted_count
            
        except Exception as e:
            run_log.run_log("ERROR", f"# # Failed to integrate with delete_DB: {str(e)}")
            return 0

def process_qradar_data(connection_string=CONNECTION_STRING_default, db_name=NAME_DB_default,
                       json_files=None, auto_cleanup=True):
    """
    # # Main function to process QRadar search results for detection-only pipeline
    # # Creates 30-minute sliding window detection windows from AQL results
    # # Integrates with delete_DB module for automatic data lifecycle management
    """
    if not json_files:
        run_log.run_log("ERROR", "# # No QRadar JSON files provided")
        return False
    
    processor = QRadarDataProcessor(connection_string, db_name)
    
    try:
        # # Pre-processing cleanup if enabled
        if auto_cleanup:
            run_log.run_log("INFO", "# # Running pre-processing cleanup...")
            lifecycle_info = processor.manage_data_lifecycle(auto_cleanup=True, retention_days=7)
            run_log.run_log("INFO", f"# # Pre-cleanup deleted {lifecycle_info['documents_deleted']} old documents")
        
        # # Create optimized indexes
        processor.create_production_indexes()
        
        # # Process QRadar data files
        run_log.run_log("INFO", f"# # Processing {len(json_files)} QRadar JSON files")
        
        total_count = processor.process_qradar_json_files(json_files)
        
        # # Post-processing cleanup and storage check
        if auto_cleanup:
            run_log.run_log("INFO", "# # Running post-processing cleanup...")
            processor.integrate_with_delete_db(retention_days=7)
            storage_info = processor.check_storage_usage()
            run_log.run_log("INFO", f"# # Final storage: {storage_info.get('database_size_mb', 0)} MB")
        
        # # Print summary
        summary = processor.get_data_summary()
        if summary:
            print(f"# # QRadar Detection Data Summary:")
            print(f"   Total detection windows: {summary.get('total_detection_windows', 0)}")
            print(f"   Unique hosts: {summary.get('unique_hosts', 0)}")
            print(f"   Total host triggers: {summary.get('total_host_triggers', 0)}")
            print(f"   Unique rules triggered: {summary.get('unique_rules', 0)}")
            print(f"   Total rule triggers: {summary.get('total_rule_triggers', 0)}")
            print(f"   Window range: {summary.get('window_range', {})}")
            
        return total_count > 0
        
    except Exception as e:
        run_log.run_log("ERROR", f"# # Failed to process QRadar data: {str(e)}")
        return False

if __name__ == "__main__":
    # # Example usage for detection-only pipeline
    print("# # Starting QRadar detection data processing...")
    
    # # Example: Process QRadar search results for detection windows
    qradar_files = [
        "result.json",  # # Use actual QRadar search result files
        # "/path/to/qradar_search_results.json"
    ]
    
    success = process_qradar_data(
        connection_string=CONNECTION_STRING_default,
        db_name=NAME_DB_default,
        json_files=qradar_files,
        auto_cleanup=True  # 启用自动清理
    )
    
    print(f"# # QRadar data processing completed: {'Success' if success else 'Failed'}")
    
    # # Optional: Manual cleanup using delete_DB integration
    if success:
        processor = QRadarDataProcessor()
        cleanup_count = processor.integrate_with_delete_db(retention_days=7)
        print(f"# # Additional cleanup completed: {cleanup_count} documents deleted")