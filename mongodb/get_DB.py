#!/usr/bin/env python3
"""
AQL JSON Database Access Utility for Detection-Only Mode

This module provides specialized database access for QRadar AQL data processing
in detection-only mode. It handles AQL-specific collections and maintains
compatibility with the detection pipeline architecture.

Python 3.6.8 Compatible
"""

import os
import sys
import json
from typing import Optional, Dict, Any

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the unified MongoDB manager
try:
    from mongodb_connection import MongoDBConnectionManager
    from system import logging_utils
except ImportError as e:
    # Fallback to pymongo for backward compatibility
    from pymongo import MongoClient
    run_log = None

def get_database(config_path: Optional[str] = None) -> Optional[Any]:
    """
    Get MongoDB database connection for AQL JSON detection data.
    
    This function provides database access specifically for detection-only mode
    with AQL JSON data processing.
    
    Args:
        config_path: Optional path to mongodb_config.json
        
    Returns:
        MongoDB database object or None if connection fails
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')
    
    try:
        # Use the unified MongoDBConnectionManager
        manager = MongoDBConnectionManager(config_path)
        
        # Connect to MongoDB
        if manager.connect():
            # Return the database object for AQL detection
            return manager.db
        else:
            if run_log:
                logging_utils.run_log("ERROR", "Failed to connect to MongoDB for AQL detection")
            return None
            
    except Exception as e:
        # Fallback to direct pymongo connection for backward compatibility
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            mongo_config = config['mongodb']
            client = MongoClient(mongo_config['connection_string'])
            db = client[mongo_config['db_name']]
            
            # Test connection
            client.admin.command('ping')
            
            if run_log:
                logging_utils.run_log("INFO", f"Connected to MongoDB: {mongo_config['db_name']} (AQL detection)")
            return db
            
        except Exception as fallback_error:
            if run_log:
                logging_utils.run_log("ERROR", f"Failed to connect to MongoDB: {str(fallback_error)}")
            return None

def get_aql_database(config_path: Optional[str] = None) -> Optional[Any]:
    """
    Get MongoDB database connection specifically for AQL JSON data.
    
    Args:
        config_path: Optional path to mongodb_config.json
        
    Returns:
        MongoDB database object configured for AQL detection or None if connection fails
    """
    return get_database(config_path)

def get_mongodb_manager(config_path: Optional[str] = None) -> Optional[MongoDBConnectionManager]:
    """
    Get the full MongoDBConnectionManager instance for AQL operations.
    
    Args:
        config_path: Optional path to mongodb_config.json
        
    Returns:
        MongoDBConnectionManager instance or None if connection fails
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')
    
    try:
        manager = MongoDBConnectionManager(config_path)
        if manager.connect():
            return manager
        return None
    except Exception as e:
        if run_log:
            logging_utils.run_log("ERROR", f"Failed to create MongoDB manager: {str(e)}")
        return None

def get_aql_collections(config_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get AQL-specific collection names from configuration.
    
    Args:
        config_path: Optional path to mongodb_config.json
        
    Returns:
        Dictionary mapping collection types to collection names
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        collections = config.get('collections', {})
        return {
            'detection_windows': collections.get('detection_windows', 'qradar_sliding_windows'),
            'detection_results': collections.get('detection_results', 'detection_results'),
            'aql_events': collections.get('aql_events', 'aql_events')
        }
    except Exception as e:
        if run_log:
            logging_utils.run_log("ERROR", f"Failed to load collection names: {str(e)}")
        return {
            'detection_windows': 'qradar_sliding_windows',
            'detection_results': 'detection_results',
            'aql_events': 'aql_events'
        }

if __name__ == "__main__":
    # Test MongoDB connection for AQL detection mode
    print("Testing MongoDB connection for AQL detection mode...")
    
    # Test basic database access
    db = get_database()
    if db:
        print(f"✓ Connected to AQL database: {db.name}")
        
        # Test collection access
        collections = db.list_collection_names()
        print(f"✓ Available collections: {collections}")
        
        # Test AQL-specific collections
        aql_collections = get_aql_collections()
        print(f"✓ AQL collection mapping:")
        for key, value in aql_collections.items():
            print(f"   {key}: {value}")
        
        # Test unified manager
        manager = get_mongodb_manager()
        if manager:
            print("✓ MongoDBConnectionManager initialized for AQL detection")
            try:
                summary = manager.get_data_summary()
                if summary:
                    print("✓ AQL data summary retrieved successfully")
                    print(f"   Detection windows: {summary.get('windows_collection', {}).get('total_windows', 0)}")
                    print(f"   Detection results: {summary.get('predictions_collection', {}).get('total_predictions', 0)}")
                else:
                    print("⚠ No AQL data summary available")
            except AttributeError:
                print("✓ MongoDBConnectionManager ready (data summary not implemented)")
        else:
            print("✗ Failed to initialize MongoDBConnectionManager for AQL detection")
            
    else:
        print("✗ Failed to connect to MongoDB for AQL detection")