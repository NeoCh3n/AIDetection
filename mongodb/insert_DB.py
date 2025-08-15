#!/usr/bin/env python3
"""
MongoDB Data Insertion Script for Detection-Only Mode

This script processes QRadar AQL search results (JSON format) and inserts them
into MongoDB collections in detection-only mode.
Handles 30-minute sliding window aggregation of AQL rule trigger data.

Python 3.6.8 Compatible
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import math

# Add required paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from system import run_log

# Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')


class AQLDataInserter:
    """
    Specialized inserter for QRadar AQL JSON data in detection-only mode.
    
    Processes QRadar search results formatted as JSON and creates 30-minute
    sliding windows for ransomware detection without training labels.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize AQL data inserter with detection-specific configuration."""
        self.config_path = config_path or CONFIG_PATH
        self.config = self._load_config()
        self.client = None
        self.db = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration for detection-only AQL processing."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            run_log.run_log("ERROR", f"Config file not found: {self.config_path}")
            return self._create_default_config()
        except json.JSONDecodeError as e:
            run_log.run_log("ERROR", f"Invalid JSON in config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration for AQL detection mode."""
        return {
            "mongodb": {
                "host": "localhost",
                "port": 27017,
                "db_name": "qradar_detection",
                "connection_string": "mongodb://localhost:27017/"
            },
            "pipeline": {
                "mode": "detection_only",
                "window_size_minutes": 30,
                "retention_days": 7
            },
            "collections": {
                "detection_windows": "qradar_sliding_windows",
                "aql_events": "aql_events"
            }
        }
    
    def connect(self) -> bool:
        """Connect to MongoDB for AQL data insertion."""
        try:
            mongo_config = self.config['mongodb']
            self.client = MongoClient(mongo_config['connection_string'])
            self.db = self.client[mongo_config['db_name']]
            
            # Test connection
            self.client.admin.command('ping')
            run_log.run_log("INFO", f"Connected to MongoDB: {mongo_config['db_name']} (AQL detection)")
            return True
            
        except ConnectionFailure as e:
            run_log.run_log("ERROR", f"MongoDB connection failed: {e}")
            return False
        except Exception as e:
            run_log.run_log("ERROR", f"Connection error: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """Create optimized indexes for AQL detection data."""
        if not self.db:
            return False
        
        try:
            from pymongo import ASCENDING, DESCENDING
            
            # Indexes for detection_windows collection
            windows_collection = self.db['qradar_sliding_windows']
            windows_indexes = [
                [("window_start", ASCENDING), ("window_end", ASCENDING)],
                [("query_time", DESCENDING)],
                [("total_triggers", DESCENDING)],
                [("window_id", ASCENDING)]
            ]
            
            for index_spec in windows_indexes:
                try:
                    windows_collection.create_index(index_spec)
                except Exception as e:
                    run_log.run_log("WARNING", f"Index creation warning: {e}")
            
            # Indexes for aql_events collection
            events_collection = self.db['aql_events']
            events_indexes = [
                [("timestamp", DESCENDING)],
                [("rule_id", ASCENDING)],
                [("hostname", ASCENDING)],
                [("window_id", ASCENDING)]
            ]
            
            for index_spec in events_indexes:
                try:
                    events_collection.create_index(index_spec)
                except Exception as e:
                    run_log.run_log("WARNING", f"Index creation warning: {e}")
            
            run_log.run_log("INFO", "AQL detection indexes created successfully")
            return True
            
        except Exception as e:
            run_log.run_log("ERROR", f"Index creation failed: {e}")
            return False
    
    def parse_aql_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse QRadar AQL timestamp string to datetime."""
        try:
            # Handle QRadar AQL timestamp format: "Jul 29, 2025, 9:50:55 AM"
            return datetime.strptime(timestamp_str, "%b %d, %Y, %I:%M:%S %p")
        except ValueError:
            try:
                return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                run_log.run_log("WARNING", f"Failed to parse timestamp: {timestamp_str}")
                return None
    
    def get_window_boundaries(self, timestamp: datetime) -> tuple:
        """Calculate 30-minute window boundaries for AQL data."""
        minutes = timestamp.minute
        window_start = timestamp.replace(
            minute=(minutes // 30) * 30, 
            second=0, 
            microsecond=0
        )
        window_end = window_start + timedelta(minutes=30)
        window_id = window_start.strftime("%Y-%m-%d_%H-%M-%S")
        
        return window_id, window_start, window_end
    
    def parse_aql_json_result(self, result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse QRadar AQL JSON results into detection windows.
        
        Args:
            result_data: AQL search result in JSON format
            
        Returns:
            List of detection window documents
        """
        if 'events' not in result_data:
            run_log.run_log("WARNING", "No events found in AQL result")
            return []
        
        events = result_data['events']
        run_log.run_log("INFO", f"Processing {len(events)} AQL events")
        
        # Group events by 30-minute windows
        window_groups = {}
        
        for event in events:
            try:
                # Extract AQL data fields
                rule_id = str(event.get('Custom Rule', '0'))
                count = int(event.get('Count', 0))
                hostname = str(event.get('sysmon_hostname (custom)', 'global'))
                
                # Parse timestamp
                event_time_str = event.get('Log Source Time (Minimum)', '')
                if not event_time_str:
                    continue
                
                event_time = self.parse_aql_timestamp(event_time_str)
                if not event_time:
                    continue
                
                window_id, window_start, window_end = self.get_window_boundaries(event_time)
                
                # Initialize window group
                if window_id not in window_groups:
                    window_groups[window_id] = {
                        '_id': window_id,
                        'window_start': window_start,
                        'window_end': window_end,
                        'query_time': datetime.now(),
                        'feature_vector': {},
                        'host_triggers': {},
                        'total_triggers': 0,
                        'total_rules_triggered': 0,
                        'metadata': {
                            'source': 'qradar_aql',
                            'data_type': 'detection',
                            'window_size_minutes': 30
                        }
                    }
                
                # Update feature vector (aggregate counts)
                window_groups[window_id]['feature_vector'][rule_id] = (
                    window_groups[window_id]['feature_vector'].get(rule_id, 0) + count
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
                window_groups[window_id]['total_rules_triggered'] = len(
                    window_groups[window_id]['feature_vector']
                )
                
            except Exception as e:
                run_log.run_log("WARNING", f"Failed to parse AQL event: {e}")
                continue
        
        documents = list(window_groups.values())
        run_log.run_log("INFO", f"Created {len(documents)} detection windows from AQL data")
        return documents
    
    def insert_detection_windows(self, documents: List[Dict[str, Any]], 
                               batch_size: int = 1000) -> int:
        """
        Insert detection windows into MongoDB with batch processing.
        
        Args:
            documents: List of detection window documents
            batch_size: Number of documents per batch
            
        Returns:
            Number of documents successfully inserted
        """
        if not self.db or not documents:
            return 0
        
        collection = self.db['qradar_sliding_windows']
        total_inserted = 0
        
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare bulk operations with upsert
                operations = []
                for doc in batch:
                    operations.append({
                        'replaceOne': {
                            'filter': {'_id': doc['_id']},
                            'replacement': doc,
                            'upsert': True
                        }
                    })
                
                if operations:
                    result = collection.bulk_write(operations)
                    total_inserted += result.upserted_count + result.modified_count
                    
                    if i % (batch_size * 5) == 0:
                        run_log.run_log("INFO", f"Processed {i + len(batch)} detection windows...")
            
            run_log.run_log("INFO", f"AQL insertion completed: {total_inserted} detection windows")
            return total_inserted
            
        except Exception as e:
            run_log.run_log("ERROR", f"AQL insertion failed: {e}")
            return total_inserted
    
    def process_aql_json_files(self, json_files: List[str]) -> int:
        """
        Process multiple AQL JSON files for detection mode.
        
        Args:
            json_files: List of AQL JSON file paths
            
        Returns:
            Total number of detection windows inserted
        """
        if not json_files:
            run_log.run_log("ERROR", "No AQL JSON files provided")
            return 0
        
        if not self.connect():
            return 0
        
        # Create indexes
        self.create_indexes()
        
        total_windows = 0
        
        try:
            for json_file in json_files:
                if not os.path.exists(json_file):
                    run_log.run_log("WARNING", f"AQL file not found: {json_file}")
                    continue
                
                run_log.run_log("INFO", f"Processing AQL file: {json_file}")
                
                with open(json_file, 'r') as f:
                    result_data = json.load(f)
                
                # Parse and insert
                documents = self.parse_aql_json_result(result_data)
                if documents:
                    inserted = self.insert_detection_windows(documents)
                    total_windows += inserted
                    run_log.run_log("INFO", f"Inserted {inserted} detection windows from {json_file}")
            
            return total_windows
            
        except Exception as e:
            run_log.run_log("ERROR", f"AQL processing failed: {e}")
            return total_windows
        finally:
            self.close()
    
    def get_insertion_summary(self) -> Dict[str, Any]:
        """Get summary of inserted AQL data."""
        if not self.db:
            return {}
        
        try:
            windows_collection = self.db['qradar_sliding_windows']
            events_collection = self.db['aql_events']
            
            summary = {
                "aql_detection_windows": {
                    "total_documents": windows_collection.count_documents({}),
                    "unique_hosts": len(windows_collection.distinct("host_triggers")),
                    "time_range": {}
                },
                "aql_events": {
                    "total_documents": events_collection.count_documents({})
                }
            }
            
            # Get time range
            if windows_collection.count_documents({}) > 0:
                time_range = list(windows_collection.aggregate([
                    {"$group": {
                        "_id": None,
                        "min_window_start": {"$min": "$window_start"},
                        "max_window_end": {"$max": "$window_end"}
                    }}
                ]))
                summary["aql_detection_windows"]["time_range"] = time_range[0] if time_range else {}
            
            return summary
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to get insertion summary: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()


def insert_aql_data(json_files: List[str], config_path: Optional[str] = None) -> int:
    """
    Main function to process and insert AQL JSON data.
    
    Args:
        json_files: List of AQL JSON file paths
        config_path: Optional path to mongodb_config.json
        
    Returns:
        Total number of detection windows inserted
    """
    processor = AQLDataInserter(config_path)
    return processor.process_aql_json_files(json_files)


def main():
    """CLI interface for AQL data insertion."""
    parser = argparse.ArgumentParser(description='Insert AQL JSON data for detection mode')
    parser.add_argument('json_files', nargs='+', help='AQL JSON files to process')
    parser.add_argument('--config', type=str, help='Path to mongodb_config.json')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for insertion')
    
    args = parser.parse_args()
    
    print("📊 Starting AQL JSON data processing...")
    print(f"   Mode: Detection-only")
    print(f"   Data source: AQL JSON")
    print(f"   Files to process: {len(args.json_files)}")
    print(f"   Batch size: {args.batch_size}")
    
    # Validate files exist
    valid_files = []
    for file_path in args.json_files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"File not found: {file_path}")
    
    if not valid_files:
        print("No valid AQL JSON files to process")
        return 1
    
    try:
        total_windows = insert_aql_data(valid_files, args.config)
        
        if total_windows > 0:
            print(f"AQL processing completed successfully")
            print(f"   Total detection windows created: {total_windows}")
            
            # Show summary
            processor = AQLDataInserter(args.config)
            if processor.connect():
                summary = processor.get_insertion_summary()
                processor.close()
                
                if "error" not in summary:
                    windows = summary.get("aql_detection_windows", {})
                    print(f"   Unique AQL hosts: {windows.get('unique_hosts', 0)}")
                    time_range = windows.get("time_range", {})
                    if time_range:
                        print(f"   AQL time range: {time_range.get('min_window_start', 'N/A')} to {time_range.get('max_window_end', 'N/A')}")
        else:
            print("No AQL detection windows created")
        
        return 0
        
    except Exception as e:
        print(f"AQL processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())