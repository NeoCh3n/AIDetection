#!/usr/bin/env python3
"""
MongoDB Data Cleanup Script for Detection-Only Mode

This script specifically handles cleanup of AQL JSON data 
in detection-only mode. It processes QRadar search results (JSON format) and
manages the corresponding MongoDB collections.

Python 3.6.8 Compatible
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add mongodb directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from system import logging_utils

# Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')


class DetectionDataCleanup:
    """
    Specialized cleanup manager for detection-only mode with AQL JSON data.
    
    This class handles cleanup of MongoDB collections that store QRadar AQL
    search results in JSON format.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize cleanup manager with AQL-specific configuration."""
        self.config_path = config_path or CONFIG_PATH
        self.config = self._load_config()
        self.client = None
        self.db = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load MongoDB configuration for detection-only mode."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging_utils.run_log("ERROR", f"Config file not found: {self.config_path}")
            return self._create_default_config()
        except json.JSONDecodeError as e:
            logging_utils.run_log("ERROR", f"Invalid JSON in config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration for detection-only mode."""
        return {
            "mongodb": {
                "host": "localhost",
                "port": 27017,
                "db_name": "qradar_detection",
                "connection_string": "mongodb://localhost:27017/"
            },
            "pipeline": {
                "mode": "detection_only",
                "retention_days": 7
            },
            "collections": {
                "detection_windows": "qradar_sliding_windows",
                "detection_results": "detection_results",
                "aql_events": "aql_events"
            }
        }
    
    def connect(self) -> bool:
        """Connect to MongoDB for AQL data processing."""
        try:
            mongo_config = self.config['mongodb']
            self.client = MongoClient(mongo_config['connection_string'])
            self.db = self.client[mongo_config['db_name']]
            
            # Test connection
            self.client.admin.command('ping')
            logging_utils.run_log("INFO", f"Connected to MongoDB: {mongo_config['db_name']} (detection mode)")
            return True
            
        except ConnectionFailure as e:
            logging_utils.run_log("ERROR", f"MongoDB connection failed: {e}")
            return False
        except Exception as e:
            logging_utils.run_log("ERROR", f"Connection error: {e}")
            return False
    
    def cleanup_detection_data(self, retention_days: int = 7) -> Dict[str, Any]:
        """
        Clean up AQL JSON detection data based on retention policy.
        
        Args:
            retention_days: Number of days to retain AQL data
            
        Returns:
            Dictionary with cleanup results by collection
        """
        if self.db is None:
            logging_utils.run_log("ERROR", "Database not connected")
            return {}
        
        collections_to_clean = [
            'qradar_sliding_windows',
            'detection_results', 
            'aql_events'
        ]
        
        results = {}
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        try:
            available_collections = self.db.list_collection_names()
            for collection_name in collections_to_clean:
                if collection_name not in available_collections:
                    results[collection_name] = 0
                    continue
                    
                collection = self.db[collection_name]
                
                # Build time-based query for AQL data
                if collection_name == 'aql_events':
                    time_field = 'timestamp'
                elif collection_name == 'qradar_sliding_windows':
                    time_field = 'window_start'
                elif collection_name == 'detection_results':
                    time_field = 'timestamp'
                else:
                    time_field = 'created_at'
                
                query = {time_field: {'$lt': cutoff_date}}
                
                # Count documents to be deleted
                old_docs = collection.count_documents(query)
                
                if old_docs > 0:
                    # Delete old documents
                    result = collection.delete_many(query)
                    deleted_count = result.deleted_count
                    results[collection_name] = deleted_count
                    logging_utils.run_log("INFO", f"🗑️  Deleted {deleted_count} documents from {collection_name}")
                else:
                    results[collection_name] = 0
                    logging_utils.run_log("INFO", f"No old documents to delete from {collection_name}")
            
            return results
            
        except Exception as e:
            logging_utils.run_log("ERROR", f"Cleanup error: {e}")
            return {"error": str(e)}
    
    def cleanup_by_window_range(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Clean up AQL data by specific window range.
        
        Args:
            start_time: Start of cleanup range
            end_time: End of cleanup range
            
        Returns:
            Cleanup results by collection
        """
        if not self.db:
            return {}
        
        collections = [
            'qradar_sliding_windows',
            'detection_results',
            'aql_events'
        ]
        
        results = {}
        
        try:
            for collection_name in collections:
                collection = self.db[collection_name]
                
                # Build time-based query
                if collection_name == 'aql_events':
                    time_field = 'timestamp'
                elif collection_name == 'qradar_sliding_windows':
                    time_field = 'window_start'
                elif collection_name == 'detection_results':
                    time_field = 'timestamp'
                else:
                    time_field = 'created_at'
                
                query = {time_field: {'$gte': start_time, '$lt': end_time}}
                
                count = collection.count_documents(query)
                if count > 0:
                    result = collection.delete_many(query)
                    results[collection_name] = result.deleted_count
                    logging_utils.run_log("INFO", f"Deleted {result.deleted_count} documents from {collection_name}")
                else:
                    results[collection_name] = 0
                    
        except Exception as e:
            logging_utils.run_log("ERROR", f"Range cleanup error: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_cleanup_summary(self) -> Dict[str, Any]:
        """Get summary of AQL data before cleanup."""
        if not self.db:
            return {}
        
        collections = [
            'qradar_sliding_windows',
            'detection_results',
            'aql_events'
        ]
        
        summary = {}
        
        try:
            for collection_name in collections:
                if collection_name not in self.db.list_collection_names():
                    summary[collection_name] = {
                        "total_documents": 0,
                        "time_range": {}
                    }
                    continue
                    
                collection = self.db[collection_name]
                total_docs = collection.count_documents({})
                
                if total_docs > 0:
                    # Get time range
                    time_field = 'timestamp' if collection_name in ['aql_events', 'detection_results'] else 'window_start'
                    time_range = list(collection.aggregate([
                        {"$group": {
                            "_id": None,
                            "min_time": {"$min": f"${time_field}"},
                            "max_time": {"$max": f"${time_field}"}
                        }}
                    ]))
                    
                    summary[collection_name] = {
                        "total_documents": total_docs,
                        "time_range": time_range[0] if time_range and len(time_range) > 0 else {}
                    }
                else:
                    summary[collection_name] = {
                        "total_documents": 0,
                        "time_range": {}
                    }
        
        except Exception as e:
            logging_utils.run_log("ERROR", f"Failed to get summary: {e}")
            summary["error"] = str(e)
        
        return summary
    
    def execute_cleanup_plan(self, retention_days: int = 7, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute comprehensive cleanup plan for AQL detection data.
        
        Args:
            retention_days: Days to retain AQL data
            dry_run: If True, only show what would be deleted
            
        Returns:
            Cleanup execution results
        """
        if not self.connect():
            return {"error": "Failed to connect to MongoDB"}
        
        results = {
            "mode": "detection_only",
            "data_source": "AQL_JSON",
            "retention_days": retention_days,
            "dry_run": dry_run,
            "summary_before": self.get_cleanup_summary()
        }
        
        if dry_run:
            # Calculate what would be deleted
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            collections = [
                'qradar_sliding_windows',
                'detection_results',
                'aql_events'
            ]
            
            would_delete = {}
            
            if self.db is None:
                logging_utils.run_log("ERROR", "Database not connected")
                return {"error": "Database not connected"}
            
            try:
                available_collections = self.db.list_collection_names()
            except Exception as e:
                logging_utils.run_log("ERROR", f"Failed to list collections: {e}")
                return {"error": str(e)}
            
            for collection_name in collections:
                if collection_name not in available_collections:
                    would_delete[collection_name] = 0
                    continue
                    
                collection = self.db[collection_name]
                
                time_field = 'timestamp' if collection_name in ['aql_events', 'detection_results'] else 'window_start'
                query = {time_field: {'$lt': cutoff_date}}
                
                try:
                    count = collection.count_documents(query)
                    would_delete[collection_name] = count
                except Exception as e:
                    logging_utils.run_log("ERROR", f"Failed to count documents in {collection_name}: {e}")
                    would_delete[collection_name] = 0
            
            results["would_delete"] = would_delete
            total_would_delete = sum(would_delete.values())
            logging_utils.run_log("INFO", f"Dry run: would delete {total_would_delete} AQL documents")
            
        else:
            # Execute actual cleanup
            cleanup_results = self.cleanup_detection_data(retention_days)
            results["cleanup_results"] = cleanup_results
            results["summary_after"] = self.get_cleanup_summary()
        
        self.close()
        return results
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()


def cleanup_old_data(retention_days: int = 7) -> Dict[str, Any]:
    """
    Legacy cleanup function for AQL JSON detection data.
    
    Args:
        retention_days: Number of days to retain AQL data
        
    Returns:
        Cleanup results dictionary
    """
    cleanup = DetectionDataCleanup()
    
    try:
        if cleanup.connect():
            results = cleanup.cleanup_detection_data(retention_days)
            cleanup.close()
            return results
        else:
            return {"error": "Failed to connect to MongoDB"}
    except Exception as e:
        return {"error": str(e)}


def main():
    """CLI interface for AQL detection data cleanup."""
    parser = argparse.ArgumentParser(description='Clean up AQL JSON detection data')
    parser.add_argument('--days', type=int, default=7,
                       help='Days to retain AQL detection data (default: 7)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actual deletion')
    parser.add_argument('--config', type=str,
                       help='Path to mongodb_config.json')
    
    args = parser.parse_args()
    
    print("🧹 Starting AQL detection data cleanup...")
    print(f"   Mode: Detection-only")
    print(f"   Data source: AQL JSON")
    print(f"   Retention: {args.days} days")
    print(f"   Dry run: {'Yes' if args.dry_run else 'No'}")
    
    cleanup = DetectionDataCleanup(args.config)
    
    try:
        results = cleanup.execute_cleanup_plan(
            retention_days=args.days,
            dry_run=args.dry_run
        )
        
        if "error" in results:
            print(f"❌ Cleanup failed: {results['error']}")
            return 1
        
        if args.dry_run:
            print("\n📊 Dry run results for AQL data:")
            would_delete = results.get("would_delete", {})
            total_would_delete = sum(would_delete.values())
            
            for collection, count in would_delete.items():
                print(f"   {collection}: {count} AQL documents")
            print(f"   Total: {total_would_delete} AQL documents would be deleted")
            
        else:
            print("\n✅ AQL detection data cleanup completed")
            cleanup_results = results.get("cleanup_results", {})
            total_deleted = sum(cleanup_results.values())
            
            print(f"   Total AQL documents deleted: {total_deleted}")
            
            if total_deleted > 0:
                print("   Storage space freed up for new AQL data")
            else:
                print("   No old AQL documents to delete")
        
        return 0
        
    except Exception as e:
        print(f"❌ AQL cleanup error: {e}")
        return 1
    finally:
        cleanup.close()


if __name__ == "__main__":
    exit(main())