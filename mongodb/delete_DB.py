#!/usr/bin/env python3
"""
MongoDB Data Cleanup Script
Deletes data older than 7 days from detection collections
"""

import os
import sys
from datetime import datetime, timedelta
import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    MONGODB_CONFIG = config['mongodb']
except FileNotFoundError:
    MONGODB_CONFIG = {
        "host": "localhost",
        "port": 27017,
        "db_name": "qradar_detection"
    }

def cleanup_old_data(retention_days: int = 7):
    """
    Delete data older than 7 days from MongoDB collections.
    
    Args:
        retention_days: Number of days to retain data
    """
    try:
        # Connect to MongoDB
        client = MongoClient(
            f"mongodb://{MONGODB_CONFIG['host']}:{MONGODB_CONFIG['port']}/"
        )
        db = client[MONGODB_CONFIG['db_name']]
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Collections to clean up
        collections = ['qradar_events', 'detection_results']
        
        total_deleted = 0
        
        for collection_name in collections:
            collection = db[collection_name]
            
            # Count documents to be deleted
            old_docs = collection.count_documents({"timestamp": {"$lt": cutoff_date}})
            
            if old_docs > 0:
                # Delete old documents
                result = collection.delete_many({"timestamp": {"$lt": cutoff_date}})
                print(f"🗑️  Deleted {result.deleted_count} documents from {collection_name}")
                total_deleted += result.deleted_count
            else:
                print(f"No old documents to delete from {collection_name}")
        
        if total_deleted > 0:
            print(f"🧹 Cleanup complete. Total deleted: {total_deleted} documents")
        else:
            print("No cleanup needed")
            
    except ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
        return False
    except Exception as e:
        print(f"Cleanup error: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()
    
    return True

def main():
    """CLI interface for cleanup"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up old MongoDB data')
    parser.add_argument('--days', type=int, default=7, help='Days to retain')
    
    args = parser.parse_args()
    
    print("🧹 Starting MongoDB cleanup...")
    print(f"   Retention: {args.days} days")
    
    success = cleanup_old_data(args.days)
    
    if success:
        print("Cleanup completed successfully")
    else:
        print("Cleanup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()