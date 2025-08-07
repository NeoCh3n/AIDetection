#!/usr/bin/env python3
"""
Test script to insert real AQL data from result.json into MongoDB
"""

import sys
import os
import json
from datetime import datetime
from dateutil import parser

# Add system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'system'))

# Import required modules
import run_log
from pymongo import MongoClient

def insert_aql_data():
    """Insert real AQL data from result.json into MongoDB"""
    
    # MongoDB connection
    client = MongoClient("mongodb://localhost:27017/")
    db = client["qradar_detection"]
    collection = db["qradar_events"]
    
    try:
        # Load result.json
        result_path = os.path.join(os.path.dirname(__file__), '..', 'result.json')
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        print(f"📊 Loaded {len(result_data.get('events', []))} events from result.json")
        
        documents = []
        
        # Process each event
        for event in result_data.get('events', []):
            try:
                # Extract real AQL data
                rule_id = int(event.get('Custom Rule', 0))
                count = int(event.get('Count', 0))
                
                # Parse timestamp
                time_str = event.get('Log Source Time (Minimum)', '')
                if time_str:
                    timestamp = parser.parse(time_str)
                else:
                    continue
                
                # Create 30-minute window
                window_minute = (timestamp.minute // 30) * 30
                window_start = timestamp.replace(minute=window_minute, second=0, microsecond=0)
                
                doc = {
                    'rule_id': rule_id,
                    'timestamp': window_start,
                    'count': count,
                    'hostname': event.get('sysmon_hostname (custom) (Unique Count)', None),
                    'source': 'qradar_aql',
                    'original_time': time_str
                }
                documents.append(doc)
                
            except Exception as e:
                print(f"⚠️  Failed to process event: {e}")
                continue
        
        if documents:
            # Insert into MongoDB
            result = collection.insert_many(documents)
            print(f"✅ Inserted {len(result.inserted_ids)} AQL documents")
            
            # Show sample
            sample = collection.find_one()
            if sample:
                print(f"📋 Sample document: {sample}")
            
        else:
            print("❌ No documents to insert")
            
        # Create indexes
        collection.create_index([("timestamp", -1)])
        collection.create_index([("rule_id", 1)])
        collection.create_index([("timestamp", 1), ("rule_id", 1)])
        print("✅ Created indexes")
        
        # Count documents
        count = collection.count_documents({})
        print(f"📊 Total documents in collection: {count}")
        
        # Count unique rules
        unique_rules = collection.distinct("rule_id")
        print(f"🔢 Unique rule IDs: {len(unique_rules)} - {sorted(unique_rules)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        client.close()

if __name__ == "__main__":
    print("🚀 Testing AQL data insertion...")
    success = insert_aql_data()
    print(f"✅ Test completed: {'Success' if success else 'Failed'}")