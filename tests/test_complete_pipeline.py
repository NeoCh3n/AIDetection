#!/usr/bin/env python3
"""
Complete pipeline test: AQL → MongoDB → Feature Generation → Predictions
Tests the entire detection pipeline with real AQL data
"""

import sys
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Configure import paths using the project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'system'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'mongodb'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'shared_utils'))

# Import required modules
from pymongo import MongoClient

# Import pipeline modules
from pipeline.feature_aggregator import aggregate_to_windows
from pipeline.feature_generator import FeatureGenerator
from shared_utils.qradar_rule_manager import get_rule_list, get_rule_to_index_map

class PipelineTester:
    """Test the complete AQL → MongoDB → predictions pipeline"""
    
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["qradar_detection"]
        self.collection = self.db["qradar_events"]
        
    def test_data_flow(self):
        """Test the complete data flow from AQL to predictions"""
        print("Starting complete pipeline test...")
        
        # Step 1: Verify MongoDB data
        print("\nStep 1: Checking MongoDB data")
        total_docs = self.collection.count_documents({})
        print(f"   Total documents in qradar_events: {total_docs}")
        
        if total_docs == 0:
            print("   No documents found, running AQL insertion test...")
            self._insert_test_aql_data()
            total_docs = self.collection.count_documents({})
        
        # Step 2: Load data into DataFrame
        print("\nStep 2: Loading data into DataFrame")
        cursor = self.collection.find({})
        df = pd.DataFrame(list(cursor))
        print(f"   Loaded {len(df)} documents")
        
        # Display schema info
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample rule IDs: {df['rule_id'].unique()[:5]}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Prepare container for aggregated data
        df_agg = pd.DataFrame()

        # Step 3: Feature aggregation
        print("\nStep 3: Feature aggregation (30-minute windows)")
        if not df.empty:
            # Prepare data for aggregation
            df_prepared = df[['hostname', 'rule_id', 'timestamp', 'count']].copy()

            if 'source_label' in df.columns:
                df_prepared['source_label'] = df['source_label'].values
            elif 'source' in df.columns:
                df_prepared['source_label'] = df['source'].values
            else:
                df_prepared['source_label'] = 'unknown'
            
            # Aggregate to windows
            df_agg = aggregate_to_windows(df_prepared, window_size_minutes=30, mode='detect')
            print(f"   Aggregated to {len(df_agg)} windows")
            
            if not df_agg.empty:
                print(f"   Sample aggregated data:")
                print(df_agg.head())
        
        # Step 4: Feature generation
        print("\nStep 4: Feature generation")
        try:
            # Get rule mapping
            rule_list = get_rule_list()
            rule_to_index = get_rule_to_index_map()
            
            print(f"   Total rules: {len(rule_list)}")
            print(f"   Rule mapping loaded: {len(rule_to_index)} rules")
            
            # Generate feature vectors
            if not df_agg.empty:
                feature_generator = FeatureGenerator()
                X, y = feature_generator.generate_feature_vectors(df_agg, mode='detect')
                print(f"   Generated feature matrix: {X.shape}")
                print(f"   Feature vector sample (first 5 values):")
                if X.shape[0] > 0:
                    print(f"   {X[0][:5]}")
            else:
                print("   No aggregated data for feature generation")
        
        except Exception as e:
            print(f"   Feature generation failed: {e}")
        
        # Step 5: Verify rule mapping consistency
        print("\nStep 5: Rule mapping verification")
        self._verify_rule_mapping()
        
        # Step 6: Test detection pipeline
        print("\nStep 6: Detection pipeline test")
        self._test_detection_pipeline()
        
        print("\nPipeline test completed!")
        
    def _insert_test_aql_data(self):
        """Insert test AQL data if none exists"""
        try:
            result_path = "result.json"
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                
                documents = []
                for event in result_data.get('events', []):
                    rule_id = int(event.get('Custom Rule', 0))
                    count = int(event.get('Count', 0))
                    
                    # Parse timestamp
                    import dateutil.parser
                    event_time = dateutil.parser.parse(event.get('Log Source Time (Minimum)', ''))
                    
                    # Create 30-minute window
                    window_minute = (event_time.minute // 30) * 30
                    window_start = event_time.replace(minute=window_minute, second=0, microsecond=0)
                    
                    doc = {
                        'rule_id': rule_id,
                        'timestamp': window_start,
                        'count': count,
                        'hostname': event.get('sysmon_hostname (custom) (Unique Count)', None),
                        'source': 'qradar_aql'
                    }
                    documents.append(doc)
                
                if documents:
                    self.collection.insert_many(documents)
                    print(f"   Inserted {len(documents)} test documents")
            else:
                print("   result.json not found, creating sample data")
                self._create_sample_data()
                
        except Exception as e:
            print(f"   Failed to insert test data: {e}")
    
    def _create_sample_data(self):
        """Create sample AQL-style data for testing"""
        try:
            # Use real rule IDs from rule_mapping.json
            with open('real_rule_mapping.json', 'r') as f:
                rule_data = json.load(f)
                rule_ids = rule_data['rule_list']
            
            # Create sample documents based on real counts
            sample_docs = []
            base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            
            for rule_id in rule_ids:
                doc = {
                    'rule_id': rule_id,
                    'timestamp': base_time,
                    'count': rule_data['counts'].get(str(rule_id), 100),
                    'hostname': None,
                    'source': 'qradar_aql'
                }
                sample_docs.append(doc)
            
            if sample_docs:
                self.collection.insert_many(sample_docs)
                print(f"   Created {len(sample_docs)} sample documents with real rule IDs")
                
        except Exception as e:
            print(f"   Failed to create sample data: {e}")
    
    def _verify_rule_mapping(self):
        """Verify rule mapping consistency"""
        try:
            rule_list = get_rule_list()
            rule_to_index = get_rule_to_index_map()
            
            print(f"   Rule list length: {len(rule_list)}")
            print(f"   Rule mapping length: {len(rule_to_index)}")
            
            # Check if rule IDs from MongoDB match rule mapping
            mongo_rules = set(self.collection.distinct("rule_id"))
            mapping_rules = set(rule_to_index.keys())
            
            print(f"   MongoDB rules: {sorted(mongo_rules)}")
            print(f"   Mapping rules: {sorted(mapping_rules)}")
            
            # Find mismatches
            missing_in_mapping = mongo_rules - mapping_rules
            missing_in_mongo = mapping_rules - mongo_rules
            
            if missing_in_mapping:
                print(f"   Rules in MongoDB but not in mapping: {missing_in_mapping}")
            if missing_in_mongo:
                print(f"   Rules in mapping but not in MongoDB: {missing_in_mongo}")
            
            if not missing_in_mapping and not missing_in_mongo:
                print("   Rule mapping is consistent")
                
        except Exception as e:
            print(f"   Rule mapping verification failed: {e}")
    
    def _test_detection_pipeline(self):
        """Test the detection pipeline components"""
        print("   Testing detection pipeline components...")
        
        try:
            # Test feature aggregation
            cursor = self.collection.find({})
            df = pd.DataFrame(list(cursor))
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                # Prepare for aggregation
                df['hostname'] = df['hostname'].fillna('unknown')
                
                # Group by hostname and 30-minute windows
                df['window_start'] = df['timestamp'].dt.floor('30T')
                
                # Create rule-count pairs
                grouped = df.groupby(['hostname', 'window_start', 'rule_id'])['count'].sum().reset_index()
                
                # Pivot to get rule_id as columns
                pivot_df = grouped.pivot_table(
                    index=['hostname', 'window_start'],
                    columns='rule_id',
                    values='count',
                    fill_value=0
                ).reset_index()
                
                print(f"   Detection pipeline ready: {len(pivot_df)} windows")
                print(f"   Sample detection window: {pivot_df.iloc[0].to_dict() if len(pivot_df) > 0 else 'No data'}")
            else:
                print("   No data for detection pipeline")
                
        except Exception as e:
            print(f"   Detection pipeline test failed: {e}")
    
    def cleanup(self):
        """Cleanup test data"""
        try:
            # Optionally clear test data
            # self.collection.delete_many({})
            pass
        except:
            pass
        finally:
            self.client.close()

def main():
    """Run complete pipeline test"""
    tester = PipelineTester()
    try:
        tester.test_data_flow()
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
