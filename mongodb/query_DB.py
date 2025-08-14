#!/usr/bin/env python3
"""
MongoDB Query Interface for Detection-Only Mode with AQL JSON

This module provides specialized querying capabilities for QRadar AQL data
in detection-only mode. It handles 30-minute sliding windows and AQL-specific
time formats.

Python 3.6.8 Compatible
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add required paths for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import get_DB
from system import run_log

# Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')

def load_config():
    """Load configuration for AQL detection mode."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        run_log.run_log("ERROR", f"Failed to load config: {str(e)}")
        return None

# Load configuration
config = load_config()
if config:
    CONNECTION_STRING = config['mongodb']['connection_string']
    DB_NAME = config['mongodb']['db_name']
    DETECTION_WINDOWS = config['collections']['detection_windows']
    DETECTION_RESULTS = config['collections']['detection_results']
    AQL_EVENTS = config['collections'].get('aql_events', 'aql_events')
else:
    # Fallback defaults for AQL detection mode
    CONNECTION_STRING = "mongodb://localhost:27017/"
    DB_NAME = "qradar_detection"
    DETECTION_WINDOWS = "qradar_sliding_windows"
    DETECTION_RESULTS = "detection_results"
    AQL_EVENTS = "aql_events"

class AQLQueryManager:
    """
    Specialized query manager for AQL JSON detection data.
    
    Provides AQL-specific query methods for 30-minute sliding windows,
    detection results, and raw AQL events.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize AQL query manager with detection-specific configuration."""
        self.config_path = config_path or CONFIG_PATH
        self.config = self._load_config()
        self.db = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load AQL detection configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to load config: {str(e)}")
            return {}
    
    def connect(self) -> bool:
        """Connect to MongoDB for AQL data querying."""
        try:
            self.db = get_DB.get_database(self.config_path)
            if self.db is None:
                run_log.run_log("ERROR", "Failed to connect to MongoDB")
                return False
            return True
        except Exception as e:
            run_log.run_log("ERROR", f"Connection error: {str(e)}")
            return False
    
    def query_detection_windows(self, 
                              time_range: Optional[Dict[str, datetime]] = None,
                              hostname: Optional[str] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query 30-minute detection windows from AQL data.
        
        Args:
            time_range: Optional dict with 'start' and 'end' datetime keys
            hostname: Optional hostname filter
            limit: Maximum number of documents to return
            
        Returns:
            List of detection window documents
        """
        if not self.db:
            return []
        
        try:
            collection = self.db[DETECTION_WINDOWS]
            query = {}
            
            # Build time-based query for AQL windows
            if time_range:
                query['window_start'] = {}
                if 'start' in time_range:
                    query['window_start']['$gte'] = time_range['start']
                if 'end' in time_range:
                    query['window_start']['$lt'] = time_range['end']
            
            # Add hostname filter if provided
            if hostname:
                query['host_triggers.' + hostname] = {'$exists': True}
            
            # Execute query with sorting
            cursor = collection.find(query).sort('window_start', -1).limit(limit)
            results = list(cursor)
            
            run_log.run_log("INFO", f"Queried {len(results)} detection windows")
            return results
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to query detection windows: {str(e)}")
            return []
    
    def query_detection_results(self,
                              time_range: Optional[Dict[str, datetime]] = None,
                              prediction_filter: Optional[int] = None,
                              min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Query detection results from AQL processing.
        
        Args:
            time_range: Optional dict with 'start' and 'end' datetime keys
            prediction_filter: Optional filter for predictions (0=normal, 1=anomaly)
            min_confidence: Optional minimum confidence threshold
            
        Returns:
            List of detection result documents
        """
        if not self.db:
            return []
        
        try:
            collection = self.db[DETECTION_RESULTS]
            query = {}
            
            # Build time-based query
            if time_range:
                query['timestamp'] = {}
                if 'start' in time_range:
                    query['timestamp']['$gte'] = time_range['start']
                if 'end' in time_range:
                    query['timestamp']['$lt'] = time_range['end']
            
            # Add prediction filter
            if prediction_filter is not None:
                query['prediction'] = prediction_filter
            
            # Add confidence filter
            if min_confidence is not None:
                query['confidence'] = {'$gte': min_confidence}
            
            # Execute query
            cursor = collection.find(query).sort('timestamp', -1)
            results = list(cursor)
            
            run_log.run_log("INFO", f"Queried {len(results)} detection results")
            return results
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to query detection results: {str(e)}")
            return []
    
    def query_aql_events(self,
                        time_range: Optional[Dict[str, datetime]] = None,
                        rule_id: Optional[str] = None,
                        hostname: Optional[str] = None,
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Query raw AQL events.
        
        Args:
            time_range: Optional dict with 'start' and 'end' datetime keys
            rule_id: Optional rule ID filter
            hostname: Optional hostname filter
            limit: Maximum number of documents to return
            
        Returns:
            List of AQL event documents
        """
        if not self.db:
            return []
        
        try:
            collection = self.db[AQL_EVENTS]
            query = {}
            
            # Build time-based query
            if time_range:
                query['timestamp'] = {}
                if 'start' in time_range:
                    query['timestamp']['$gte'] = time_range['start']
                if 'end' in time_range:
                    query['timestamp']['$lt'] = time_range['end']
            
            # Add rule ID filter
            if rule_id:
                query['rule_id'] = rule_id
            
            # Add hostname filter
            if hostname:
                query['hostname'] = hostname
            
            # Execute query
            cursor = collection.find(query).sort('timestamp', -1).limit(limit)
            results = list(cursor)
            
            run_log.run_log("INFO", f"Queried {len(results)} AQL events")
            return results
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to query AQL events: {str(e)}")
            return []
    
    def get_latest_window(self) -> Optional[Dict[str, Any]]:
        """Get the most recent detection window."""
        if not self.db:
            return None
        
        try:
            collection = self.db[DETECTION_WINDOWS]
            result = collection.find_one(sort=[('window_start', -1)])
            return result
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to get latest window: {str(e)}")
            return None
    
    def get_window_by_id(self, window_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific detection window by ID."""
        if not self.db:
            return None
        
        try:
            collection = self.db[DETECTION_WINDOWS]
            result = collection.find_one({'_id': window_id})
            return result
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to get window by ID: {str(e)}")
            return None
    
    def get_detection_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get summary of detection activity for the specified time period.
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Summary dictionary with counts and statistics
        """
        if not self.db:
            return {}
        
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Query detection windows
            windows_collection = self.db[DETECTION_WINDOWS]
            windows_query = {
                'window_start': {
                    '$gte': start_time,
                    '$lt': end_time
                }
            }
            
            windows = list(windows_collection.find(windows_query))
            
            # Query detection results
            results_collection = self.db[DETECTION_RESULTS]
            results_query = {
                'timestamp': {
                    '$gte': start_time,
                    '$lt': end_time
                }
            }
            
            results = list(results_collection.find(results_query))
            
            # Calculate summary
            total_windows = len(windows)
            total_anomalies = sum(1 for r in results if r.get('prediction') == 1)
            avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results) if results else 0
            
            summary = {
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'total_detection_windows': total_windows,
                'total_anomaly_detections': total_anomalies,
                'average_confidence': round(avg_confidence, 3),
                'detection_rate': round(total_anomalies / total_windows, 3) if total_windows > 0 else 0
            }
            
            return summary
            
        except Exception as e:
            run_log.run_log("ERROR", f"Failed to get detection summary: {str(e)}")
            return {}
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'db') and self.db:
            try:
                self.db.client.close()
            except:
                pass

def query_database(collection_name: str = DETECTION_WINDOWS, 
                  query: Dict[str, Any] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
    """
    Legacy query function for backward compatibility.
    
    Args:
        collection_name: MongoDB collection name
        query: MongoDB query dictionary
        limit: Maximum number of documents
        
    Returns:
        List of documents
    """
    query = query or {}
    
    manager = AQLQueryManager()
    try:
        if manager.connect():
            if collection_name == DETECTION_WINDOWS:
                results = manager.query_detection_windows(limit=limit)
            elif collection_name == DETECTION_RESULTS:
                results = manager.query_detection_results()
            elif collection_name == AQL_EVENTS:
                results = manager.query_aql_events(limit=limit)
            else:
                # Generic query for any collection
                if manager.db:
                    collection = manager.db[collection_name]
                    results = list(collection.find(query).limit(limit))
                else:
                    results = []
            
            run_log.run_log("INFO", f"Queried {len(results)} documents from {collection_name}")
            return results
        else:
            return []
    except Exception as e:
        run_log.run_log("ERROR", f"Query failed: {str(e)}")
        return []
    finally:
        manager.close()

def main():
    """CLI interface for AQL data querying."""
    parser = argparse.ArgumentParser(description='Query AQL JSON detection data')
    parser.add_argument('collection', choices=['windows', 'results', 'events', 'all'],
                       help='Collection to query')
    parser.add_argument('--hours-back', type=int, default=24,
                       help='Hours to look back (default: 24)')
    parser.add_argument('--limit', type=int, default=100,
                       help='Limit number of results (default: 100)')
    parser.add_argument('--config', type=str,
                       help='Path to mongodb_config.json')
    
    args = parser.parse_args()
    
    print("🔍 Starting AQL data query...")
    print(f"   Mode: Detection-only")
    print(f"   Collection: {args.collection}")
    print(f"   Hours back: {args.hours_back}")
    print(f"   Limit: {args.limit}")
    
    manager = AQLQueryManager(args.config)
    
    try:
        if not manager.connect():
            print("Failed to connect to MongoDB")
            return 1
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=args.hours_back)
        time_range = {'start': start_time, 'end': end_time}
        
        if args.collection == 'windows':
            results = manager.query_detection_windows(time_range=time_range, limit=args.limit)
        elif args.collection == 'results':
            results = manager.query_detection_results(time_range=time_range)
        elif args.collection == 'events':
            results = manager.query_aql_events(time_range=time_range, limit=args.limit)
        elif args.collection == 'all':
            results = {
                'summary': manager.get_detection_summary(args.hours_back),
                'latest_window': manager.get_latest_window(),
                'windows': manager.query_detection_windows(time_range=time_range, limit=args.limit),
                'results': manager.query_detection_results(time_range=time_range),
                'events': manager.query_aql_events(time_range=time_range, limit=args.limit)
            }
        
        print(f"\n Query completed successfully")
        
        if args.collection == 'all':
            print(f"\n Detection Summary:")
            summary = results['summary']
            print(f"   Detection windows: {summary.get('total_detection_windows', 0)}")
            print(f"   Anomaly detections: {summary.get('total_anomaly_detections', 0)}")
            print(f"   Detection rate: {summary.get('detection_rate', 0)}")
            print(f"   Average confidence: {summary.get('average_confidence', 0)}")
        else:
            print(f"   Total results: {len(results)}")
            
            # Show sample data
            if results and isinstance(results, list):
                print(f"\n📋 Sample data:")
                for i, doc in enumerate(results[:3]):
                    print(f"   {i+1}. _id: {doc.get('_id', 'N/A')}")
                    if 'window_start' in doc:
                        print(f"      Window: {doc.get('window_start')} -> {doc.get('window_end')}")
                        print(f"      Total triggers: {doc.get('total_triggers', 0)}")
                    elif 'timestamp' in doc:
                        print(f"      Timestamp: {doc.get('timestamp')}")
                        print(f"      Prediction: {doc.get('prediction', 'N/A')}")
        
        return 0
        
    except Exception as e:
        print(f" Query failed: {e}")
        return 1
    finally:
        manager.close()

if __name__ == "__main__":
    exit(main())