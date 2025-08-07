import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'system'))
import run_log
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Generator
import delete_DB  # # Import delete module for data lifecycle management

#### MongoDB configuration for detection-only mode
CONNECTION_STRING_default = "mongodb://localhost:27017/"  # # Local MongoDB connection for offline use
NAME_DB_default = "qradar_detection"  # # Database for detection pipeline
COLLECTION_default = "qradar_events"  # # Collection for real AQL data

# Import get_DB with path fix
import importlib.util
spec = importlib.util.spec_from_file_location("get_DB", os.path.join(os.path.dirname(__file__), 'get_DB.py'))
get_DB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(get_DB)

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
        self.db = get_DB.get_database(self.connection_string, self.db_name)
        self.collection = self.db[COLLECTION_default]
        return self.db is not None
    
    def create_production_indexes(self):
        """
        # # Create optimized indexes for production QRadar data
        """
        if not self.connect():
            return False
        
        try:
            # # Compound index for time-based queries
            self.collection.create_index([
                ("date", 1),
                ("work_hour", 1),
                ("hostname", 1)
            ], name="date_hour_host_idx")
            
            # # Data type index for train/test separation
            self.collection.create_index([
                ("data_type", 1),
                ("date", 1)
            ], name="datatype_date_idx")
            
            # # Host-based queries
            self.collection.create_index("hostname", name="hostname_idx")
            
            # # Time bucket queries
            self.collection.create_index("time_bucket", name="timebucket_idx")
            
            # # Rule analysis
            self.collection.create_index("total_triggers", name="triggers_idx")
            
            run_log.run_log("INFO", "# # Production indexes created successfully")
            return True
            
        except Exception as e:
            run_log.run_log("ERROR", f"# # Failed to create indexes: {str(e)}")
            return False

    def parse_qradar_search_result(self, result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        # # Parse QRadar search results into detection-ready format
        # # Uses real AQL schema from result.json with 30-minute windows
        """
        documents = []
        
        if 'events' not in result_data:
            run_log.run_log("WARNING", "# # No events found in QRadar result")
            return documents
            
        run_log.run_log("INFO", f"# # Processing {len(result_data['events'])} QRadar events")
        
        # Parse events in 30-minute windows for detection
        for event in result_data['events']:
            try:
                # Extract real AQL data fields
                rule_id = int(event.get('Custom Rule', 0))
                count = int(event.get('Count', 0))
                
                # Parse timestamp from "Jul 24, 2025, 2:29:57 PM" format
                event_time_str = event.get('Log Source Time (Minimum)', '')
                if event_time_str:
                    # Handle "Jul 24, 2025, 2:29:57 PM" format
                    from dateutil import parser
                    event_time = parser.parse(event_time_str)
                else:
                    continue  # Skip events without timestamp
                
                # Create 30-minute window aligned to detection pipeline
                window_minute = (event_time.minute // 30) * 30
                window_start = event_time.replace(minute=window_minute, second=0, microsecond=0)
                
                # Create document for detection pipeline
                doc = {
                    'rule_id': rule_id,
                    'timestamp': window_start,
                    'count': count,
                    'hostname': event.get('sysmon_hostname (custom) (Unique Count)', None),  # Real AQL has null
                    'source': 'qradar_aql',
                    'window_id': f"window_{window_start.isoformat()}"
                }
                documents.append(doc)
                
            except Exception as e:
                run_log.run_log("WARNING", f"# # Failed to parse event: {str(e)}")
                continue
        
        run_log.run_log("INFO", f"# # Created {len(documents)} AQL-style documents")
        return documents
    
    def split_data_by_time(self, start_date: str, end_date: str, 
                          train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """
        # # Split QRadar data chronologically for ML pipeline
        # # 70% training, 15% validation, 15% testing
        # # Ensures no temporal leakage
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days
        
        # # Calculate split points
        train_days = int(total_days * train_ratio)
        val_days = int(total_days * val_ratio)
        
        train_end = start_dt + timedelta(days=train_days)
        val_end = train_end + timedelta(days=val_days)
        
        return {
            "training": {"start": start_dt, "end": train_end, "days": train_days},
            "validation": {"start": train_end, "end": val_end, "days": val_days}, 
            "testing": {"start": val_end, "end": end_dt, "days": total_days - train_days - val_days}
        }
    
    def assign_data_types(self, documents: List[Dict[str, Any]], time_splits: Dict):
        """
        # # Assign data_type based on time splits
        """
        for doc in documents:
            doc_date = datetime.strptime(doc['date'], "%Y-%m-%d")
            
            if doc_date < time_splits['training']['end']:
                doc['data_type'] = 'training'
            elif doc_date < time_splits['validation']['end']:
                doc['data_type'] = 'validation'
            else:
                doc['data_type'] = 'testing'
        
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
    
    def process_qradar_json_files(self, json_files: List[str], start_date: str, end_date: str):
        """
        # # Process multiple QRadar JSON files for complete dataset
        # # Handles month-long data from multiple searches
        """
        all_documents = []
        
        # # Calculate time splits first
        time_splits = self.split_data_by_time(start_date, end_date)
        run_log.run_log("INFO", f"# # Time splits: Train({time_splits['training']['days']}d), "
                                f"Val({time_splits['validation']['days']}d), Test({time_splits['testing']['days']}d)")
        
        # # Process each JSON file
        for json_file in json_files:
            try:
                run_log.run_log("INFO", f"# # Processing QRadar file: {json_file}")
                
                with open(json_file, 'r') as f:
                    result_data = json.load(f)
                
                # # Parse QRadar data
                documents = self.parse_qradar_search_result(result_data)
                
                # # Assign data types based on time
                documents = self.assign_data_types(documents, time_splits)
                
                all_documents.extend(documents)
                
                run_log.run_log("INFO", f"# # Processed {len(documents)} documents from {json_file}")
                
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
        # # Clean up old data based on retention policy (7 days for ML training)
        # # Integrates with delete_DB module for data lifecycle management
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            run_log.run_log("INFO", f"# # Starting cleanup: deleting data older than {retention_days} days")
            
            if not self.connect():
                return 0
            
            # # Query to find old documents
            query = {'time_bucket': {'$lt': cutoff_date}}
            
            # # Count documents to be deleted
            count_to_delete = self.collection.count_documents(query)
            
            if count_to_delete > 0:
                # # Delete old documents
                result = self.collection.delete_many(query)
                run_log.run_log("INFO", f"# # Deleted {result.deleted_count} old documents")
                return result.deleted_count
            else:
                run_log.run_log("INFO", "# # No old documents found to delete")
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
                    'time_bucket': {'$gte': cutoff_date}
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
            unique_hosts = len(self.collection.distinct("hostname"))
            
            # # Data type distribution
            training_docs = self.collection.count_documents({"data_type": "training"})
            validation_docs = self.collection.count_documents({"data_type": "validation"})
            testing_docs = self.collection.count_documents({"data_type": "testing"})
            
            # # Time range
            time_range = list(self.collection.aggregate([
                {"$group": {
                    "_id": None,
                    "min_date": {"$min": "$date"},
                    "max_date": {"$max": "$date"}
                }}
            ]))
            
            # # Rule statistics
            rule_stats = list(self.collection.aggregate([
                {"$unwind": "$rule_triggers"},
                {"$group": {
                    "_id": None,
                    "unique_rules": {"$addToSet": "$rule_triggers.k"},
                    "total_rule_triggers": {"$sum": "$rule_triggers.v"}
                }}
            ]))
            
            summary = {
                "total_documents": total_docs,
                "unique_hosts": unique_hosts,
                "training_docs": training_docs,
                "validation_docs": validation_docs,
                "testing_docs": testing_docs,
                "date_range": time_range[0] if time_range else {},
                "unique_rules": len(rule_stats[0]["unique_rules"]) if rule_stats else 0
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
                       json_files=None, start_date=None, end_date=None, auto_cleanup=True):
    """
    # # Main function to process real QRadar search results for ML pipeline
    # # Handles production scale: 6000+ hosts, 1+ month data, work hours 10:00-18:00
    # # Integrates with delete_DB for automatic data lifecycle management
    """
    if not json_files:
        run_log.run_log("ERROR", "# # No QRadar JSON files provided")
        return False
    
    if not start_date or not end_date:
        run_log.run_log("ERROR", "# # Start date and end date required")
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
        run_log.run_log("INFO", f"# # Date range: {start_date} to {end_date}")
        
        total_count = processor.process_qradar_json_files(json_files, start_date, end_date)
        
        # # Post-processing cleanup and storage check
        if auto_cleanup:
            run_log.run_log("INFO", "# # Running post-processing cleanup...")
            processor.integrate_with_delete_db(retention_days=7)
            storage_info = processor.check_storage_usage()
            run_log.run_log("INFO", f"# # Final storage: {storage_info.get('database_size_mb', 0)} MB")
        
        # # Print summary
        summary = processor.get_data_summary()
        if summary:
            print(f"# # QRadar Data Processing Summary:")
            print(f"   Total documents: {summary.get('total_documents', 0)}")
            print(f"   Training documents: {summary.get('training_docs', 0)}")
            print(f"   Validation documents: {summary.get('validation_docs', 0)}")
            print(f"   Testing documents: {summary.get('testing_docs', 0)}")
            print(f"   Unique hosts: {summary.get('unique_hosts', 0)}")
            print(f"   Unique rules: {summary.get('unique_rules', 0)}")
            print(f"   Date range: {summary.get('date_range', {})}")
            
        return total_count > 0
        
    except Exception as e:
        run_log.run_log("ERROR", f"# # Failed to process QRadar data: {str(e)}")
        return False

if __name__ == "__main__":
    # # Example usage for production QRadar data with automatic cleanup
    print("# # Starting QRadar data processing for ML pipeline...")
    
    # # Example: Process QRadar search results with lifecycle management
    qradar_files = [
        "result.json",  # # Use actual QRadar search result files
        # "/path/to/qradar_search_week2.json", 
        # "/path/to/qradar_search_week3.json",
        # "/path/to/qradar_search_week4.json"
    ]
    
    success = process_qradar_data(
        connection_string=CONNECTION_STRING_default,
        db_name=NAME_DB_default,
        json_files=qradar_files,
        start_date="2024-06-01",  # 数据开始日期
        end_date="2024-06-30",    # 数据结束日期
        auto_cleanup=True         # 启用自动清理
    )
    
    print(f"# # QRadar data processing completed: {'Success' if success else 'Failed'}")
    
    # # Optional: Manual cleanup using delete_DB integration
    if success:
        processor = QRadarDataProcessor()
        cleanup_count = processor.integrate_with_delete_db(retention_days=7)
        print(f"# # Additional cleanup completed: {cleanup_count} documents deleted")