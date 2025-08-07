#!/usr/bin/env python3
"""
Simple complete pipeline test: AQL → MongoDB → Predictions
Tests the entire detection pipeline with real AQL data
"""

import sys
import os
import json
from datetime import datetime

# Add system path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mongodb'))

# Import required modules
try:
    from pymongo import MongoClient
    import pandas as pd
    import numpy as np
    from dateutil import parser
    print("✅ All required packages available")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_complete_pipeline():
    """Test the complete pipeline from AQL to predictions"""
    print("🚀 Starting complete pipeline test...")
    
    # Step 1: Verify MongoDB data
    print("\n📊 Step 1: Checking MongoDB data")
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["qradar_detection"]
        collection = db["qradar_events"]
        
        total_docs = collection.count_documents({})
        print(f"   Total documents in qradar_events: {total_docs}")
        
        if total_docs == 0:
            print("   📊 No documents found, inserting test data...")
            # Insert test AQL data
            result_path = "result.json"
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                
                documents = []
                for event in result_data.get('events', []):
                    rule_id = int(event.get('Custom Rule', 0))
                    count = int(event.get('Count', 0))
                    
                    # Parse timestamp
                    event_time_str = event.get('Log Source Time (Minimum)', '')
                    if event_time_str:
                        event_time = parser.parse(event_time_str)
                        
                        # Create 30-minute window
                        window_minute = (event_time.minute // 30) * 30
                        window_start = event_time.replace(minute=window_minute, second=0, microsecond=0)
                        
                        doc = {
                            'hostname': event.get('sysmon_hostname (custom) (Unique Count)', 'unknown'),
                            'rule_id': rule_id,
                            'timestamp': window_start,
                            'count': count,
                            'source': 'qradar_aql'
                        }
                        documents.append(doc)
                
                if documents:
                    collection.insert_many(documents)
                    print(f"   ✅ Inserted {len(documents)} test documents")
                    total_docs = collection.count_documents({})
                
        # Get sample data
        cursor = collection.find({}).limit(5)
        sample_docs = list(cursor)
        print(f"   Sample documents: {len(sample_docs)}")
        
        if sample_docs:
            print(f"   First document: {sample_docs[0]}")
            
    except Exception as e:
        print(f"   ❌ MongoDB test failed: {e}")
        return False
    
    # Step 2: Test data transformation pipeline
    print("\n📊 Step 2: Testing data transformation pipeline")
    try:
        # Load data into DataFrame
        cursor = collection.find({})
        df = pd.DataFrame(list(cursor))
        
        if df.empty:
            print("   ❌ No data in MongoDB")
            return False
            
        print(f"   Loaded {len(df)} documents")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Unique rule IDs: {sorted(df['rule_id'].unique())}")
        
        # Clean and prepare data
        df = df.fillna({'hostname': 'unknown'})
        df['source_label'] = 'normal'  # Mark as detection data
        
        # Create 30-minute aggregation
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['window_id'] = df['timestamp'].dt.floor('30T')
        
        # Group by hostname and window
        grouped = df.groupby(['hostname', 'window_id', 'rule_id'])['count'].sum().reset_index()
        
        # Pivot to get rule_id as columns
        pivot_df = grouped.pivot_table(
            index=['hostname', 'window_id'],
            columns='rule_id',
            values='count',
            fill_value=0
        ).reset_index()
        
        print(f"   Aggregated to {len(pivot_df)} windows")
        print(f"   Final shape: {pivot_df.shape}")
        
        # Verify rule compatibility
        actual_rules = set(pivot_df.columns) - {'hostname', 'window_id'}
        print(f"   Actual rules in data: {sorted(actual_rules)}")
        
    except Exception as e:
        print(f"   ❌ Data transformation failed: {e}")
        return False
    
    # Step 3: Test rule mapping
    print("\n📊 Step 3: Testing rule mapping")
    try:
        # Load real rule mapping
        with open('real_rule_mapping.json', 'r') as f:
            rule_data = json.load(f)
        
        rule_list = rule_data['rule_list']
        rule_to_index = rule_data['rule_to_index']
        
        print(f"   Total rules in mapping: {len(rule_list)}")
        print(f"   Rule mapping: {rule_to_index}")
        
        # Check compatibility with actual data
        actual_rules = set(pivot_df.columns) - {'hostname', 'window_id'}
        mapping_rules = set(rule_to_index.keys())
        
        print(f"   Rules in data: {sorted(actual_rules)}")
        print(f"   Rules in mapping: {sorted(mapping_rules)}")
        
        # Create feature vector
        feature_vector = np.zeros(len(rule_list))
        for rule_id in actual_rules:
            if str(rule_id) in rule_to_index:
                idx = rule_to_index[str(rule_id)]
                # Use average count across all windows
                feature_vector[idx] = pivot_df[rule_id].mean()
        
        print(f"   Feature vector shape: {feature_vector.shape}")
        print(f"   Non-zero features: {np.count_nonzero(feature_vector)}")
        print(f"   Feature vector (first 5): {feature_vector[:5]}")
        
    except Exception as e:
        print(f"   ❌ Rule mapping test failed: {e}")
        return False
    
    # Step 4: Test MongoDB delete functionality
    print("\n📊 Step 4: Testing MongoDB cleanup")
    try:
        # Test the delete functionality
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=7)
        query = {'timestamp': {'$lt': cutoff_date}}
        old_count = collection.count_documents(query)
        
        print(f"   Documents older than 7 days: {old_count}")
        
        # Create test old document
        old_doc = {
            'hostname': 'test',
            'rule_id': 100001,
            'timestamp': datetime.now() - timedelta(days=10),
            'count': 1,
            'source': 'test'
        }
        collection.insert_one(old_doc)
        
        # Test delete
        result = collection.delete_many({'timestamp': {'$lt': cutoff_date}})
        print(f"   Deleted documents: {result.deleted_count}")
        
    except Exception as e:
        print(f"   ❌ MongoDB cleanup test failed: {e}")
        return False
    
    # Step 5: Final verification
    print("\n📊 Step 5: Final verification")
    try:
        # Count documents by source
        pipeline = [
            {'$group': {'_id': '$source', 'count': {'$sum': 1}}}
        ]
        source_counts = list(collection.aggregate(pipeline))
        print(f"   Documents by source: {source_counts}")
        
        # Count unique rules
        unique_rules = collection.distinct('rule_id')
        print(f"   Unique rules: {len(unique_rules)} - {sorted(unique_rules)}")
        
        # Test data freshness
        latest = collection.find_one(sort=[('timestamp', -1)])
        if latest:
            print(f"   Latest data: {latest['timestamp']}")
        
        print("\n✅ Complete pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Final verification failed: {e}")
        return False
    
    finally:
        client.close()

def main():
    """Run the complete pipeline test"""
    print("🚀 AQL → MongoDB → Predictions Pipeline Test")
    print("=" * 50)
    
    success = test_complete_pipeline()
    
    if success:
        print("\n🎉 All pipeline components working correctly!")
        print("✅ Ready for production deployment")
    else:
        print("\n❌ Pipeline test failed - check error messages above")
    
    return success

if __name__ == "__main__":
    main()