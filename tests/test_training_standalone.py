#!/usr/bin/env python3
"""
Updated standalone training pipeline test using actual pipeline modules
Tests the entire process: Training_data → data_loader → feature_aggregator → feature_generator → model_training
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd

# Add project root to path for all imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import actual pipeline modules by temporarily adjusting the system
# We'll use the actual pipeline logic but handle the imports carefully

def import_pipeline_modules():
    """Import pipeline modules with proper path handling"""
    try:
        # Fix system import issues
        if 'system' not in sys.modules:
            sys.path.insert(0, os.path.join(project_root, 'system'))
        
        # Import actual modules
        from pipeline.data_loader import load_data
        from pipeline.feature_aggregator import aggregate_to_windows
        from pipeline.feature_generator import FeatureGenerator
        from shared_utils.qradar_rule_manager import QRadarRuleManager
        from model_training.model_training import train_ransomware_detector
        
        return load_data, aggregate_to_windows, FeatureGenerator, QRadarRuleManager, train_ransomware_detector
    except ImportError as e:
        import traceback
        print(f"Import error: {e}")
        traceback.print_exc()
        # Fallback to simple implementation if imports fail
        raise ImportError(f"Failed to import pipeline modules: {e}")


def run_actual_pipeline_test():
    """Test using actual pipeline modules"""
    
    print("=== COMPLETE TRAINING PIPELINE TEST (ACTUAL MODULES) ===\n")
    
    # Try to import actual modules
    try:
        load_data, aggregate_to_windows, FeatureGenerator, QRadarRuleManager, train_ransomware_detector = import_pipeline_modules()
    except ImportError as e:
        print(f"Cannot import actual modules: {e}")
        return {'success': False, 'error': f'Import failed: {str(e)}'}
    
    # Setup configuration
    config = {
        'training_data_path': 'Training_data/normal',
        'attack_data_path': 'Training_data/attack',
        'rule_mapping_file': 'rule_mapping.json',
        'window_size': 30,  # 30 minutes
        'random_state': 42
    }
    
    # Create temporary directory for test outputs
    temp_dir = tempfile.mkdtemp()
    model_save_path = os.path.join(temp_dir, "test_model.joblib")
    
    try:
        print("1. LOADING TRAINING DATA...")
        
        # Load normal and attack data using real data_loader
        normal_df = load_data(mode='train', config={'training_data_path': 'Training_data/normal', 'rule_mapping_file': 'rule_mapping.json'})
        attack_df = load_data(mode='train', config={'training_data_path': 'Training_data/attack', 'rule_mapping_file': 'rule_mapping.json'})
        
        # Ensure we have DataFrames (type checking)
        assert normal_df is not None, "Normal data loading returned None"
        assert attack_df is not None, "Attack data loading returned None"
        
        print(f"   Normal data loaded: {len(normal_df)} rows")
        print(f"   Attack data loaded: {len(attack_df)} rows")
        
        # Validate data structure
        required_cols = ['hostname', 'rule_id', 'timestamp', 'count']
        for df_name, df in [('normal', normal_df), ('attack', attack_df)]:
            assert not df.empty, f"{df_name} data is empty"
            for col in required_cols:
                assert col in df.columns, f"{col} missing from {df_name} data"
        
        print("   ✓ Data loading completed successfully")
        
        print("\n2. PREPARING DATA WITH LABELS...")
        
        # Add labels
        normal_df = normal_df.copy()
        attack_df = attack_df.copy()
        normal_df['source_label'] = 0  # Normal
        attack_df['source_label'] = 1  # Attack
        
        # Combine datasets
        combined_df = pd.concat([normal_df, attack_df], ignore_index=True)
        print(f"   Combined dataset: {len(combined_df)} total rows")
        print(f"   Normal samples: {(combined_df['source_label'] == 0).sum()}")
        print(f"   Attack samples: {(combined_df['source_label'] == 1).sum()}")
        
        print("\n3. AGGREGATING FEATURES...")
        
        # Aggregate to 30-minute windows using real feature_aggregator
        aggregated_df = aggregate_to_windows(combined_df, window_size_minutes=30)
        
        print(f"   Aggregated windows: {len(aggregated_df)} windows")
        assert not aggregated_df.empty, "Aggregation produced no windows"
        
        # Validate aggregation structure
        required_agg_cols = ['window_id', 'hostname', 'aggregated_rules', 'is_attack']
        for col in required_agg_cols:
            assert col in aggregated_df.columns, f"{col} missing from aggregated data"
        
        # Check sample aggregated data
        sample_window = aggregated_df.iloc[0]
        assert isinstance(sample_window['aggregated_rules'], dict), "Rules should be aggregated as dict"
        assert len(sample_window['aggregated_rules']) > 0, "No rules in aggregated window"
        
        print("   ✓ Feature aggregation completed")
        
        print("\n4. GENERATING FEATURE VECTORS...")
        
        # Initialize feature generator with real rule manager
        feature_gen = FeatureGenerator(environment='prod')
        rule_manager = QRadarRuleManager(mode='file', environment='prod')
        rule_list = rule_manager.get_rule_list()
        
        print(f"   Total rules: {len(rule_list)}")
        
        # Generate feature vectors using real feature_generator
        result = feature_gen.generate_feature_vectors(aggregated_df, mode='train')
        if result is None:
            raise ValueError("Feature generation returned None")
        
        X, y = result
        if X is None or y is None:
            raise ValueError("Feature generation returned None arrays")
        
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        print(f"   Features per sample: {X.shape[1]}")
        
        # Validate feature generation
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == len(rule_list), f"Expected {len(rule_list)} features"
        assert not pd.isna(X).any().any(), "No NaN values in features"
        assert not pd.isna(y).any(), "No NaN values in labels"
        
        print("   ✓ Feature vector generation completed")
        
        print("\n5. TRAINING MODEL...")
        
        # Prepare training configuration
        training_config = {
            'training_data_path': 'Training_data/normal',
            'attack_data_path': 'Training_data/attack',
            'rule_mapping_file': 'rule_mapping.json',
            'window_size': 30,  # 30 minutes
            'random_state': 42
        }
        
        # Train model using real model_training
        result = train_ransomware_detector(
            training_config=training_config,
            model_save_path=model_save_path
        )
        
        if result is None:
            raise ValueError("Model training failed")
        
        model, X_test, y_test = result
        if model is None or X_test is None or y_test is None:
            raise ValueError("Model training returned None values")
        
        print(f"   Model trained successfully")
        print(f"   Model saved to: {model_save_path}")
        print(f"   Model file exists: {os.path.exists(model_save_path)}")
        
        # Validate trained model
        assert os.path.exists(model_save_path), "Model file was not created"
        
        # Test model predictions
        predictions = model.predict(X_test)
        accuracy = float(np.mean(predictions == y_test))
        
        print(f"   Test accuracy: {accuracy:.3f}")
        print(f"   Normal samples: {np.sum(y_test == 0)}")
        print(f"   Attack samples: {np.sum(y_test == 1)}")
        
        # Check feature importance
        feature_importance = model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        top_features_importance = feature_importance[top_features_idx]
        
        print(f"   Top 10 most important features:")
        for i, (idx, imp) in enumerate(zip(top_features_idx, top_features_importance)):
            rule_id = rule_list[idx]
            print(f"     {i+1}. Rule {rule_id}: {imp:.4f}")
        
        print("\n=== TRAINING PIPELINE TEST COMPLETED SUCCESSFULLY ===")
        
        return {
            'success': True,
            'normal_samples': len(normal_df),
            'attack_samples': len(attack_df),
            'aggregated_windows': len(aggregated_df),
            'feature_count': X.shape[1],
            'model_accuracy': accuracy,
            'model_path': model_save_path,
            'temp_dir': temp_dir
        }
        
    except Exception as e:
        print(f"\n❌ ERROR with actual modules: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)




if __name__ == "__main__":
    # Run with actual pipeline modules
    result = run_actual_pipeline_test()
    
    if result['success']:
        print(f"\nSUCCESS: All pipeline steps completed successfully!")
        print(f"   - Loaded {result['normal_samples']} normal + {result['attack_samples']} attack events")
        print(f"   - Created {result['aggregated_windows']} 30-minute windows")
        print(f"   - Generated {result['feature_count']} features per window")
        print(f"   - Achieved {result['model_accuracy']:.3f} model accuracy")
    else:
        print(f"\nFAILED: {result.get('error', 'Unknown error')}")
        # Optionally run fallback
        # run_fallback_pipeline_test()