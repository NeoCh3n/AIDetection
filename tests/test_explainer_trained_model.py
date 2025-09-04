#!/usr/bin/env python3
"""
Test SHAP Explainer with Trained Model and Training Data

This test demonstrates the SHAP explainer's new simplified interface using:
- The actual trained threat_detector.joblib model
- Real training data from the Training_data directory
- The complete pipeline for data processing

Usage:
    cd /path/to/AIDetection
    python tests/test_explainer_trained_model.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the explainer and pipeline components
from system.shap_explainer import Explainer
from pipeline.data_loader import load_data
from pipeline.feature_aggregator import aggregate_to_windows
from pipeline.feature_generator import FeatureGenerator
from shared_utils.qradar_rule_manager import QRadarRuleManager

def test_explainer_with_trained_model():
    """
    Test the SHAP explainer using the actual trained model and training data.
    
    This test follows the complete data pipeline:
    1. Load training data
    2. Aggregate into 30-minute windows  
    3. Generate feature vectors
    4. Load the trained model
    5. Run SHAP explanation with the new interface
    """
    print("=" * 80)
    print("TESTING SHAP EXPLAINER WITH TRAINED MODEL AND TRAINING DATA")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Load configuration
        config_path = os.path.join(PROJECT_ROOT, 'pipeline', 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded configuration from {config_path}")
        
        # 2. Check if trained model exists
        model_path = os.path.join(PROJECT_ROOT, 'model', 'threat_detector.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        print(f"✓ Found trained model at {model_path}")
        
        # 3. Load training data using the pipeline
        print("\n--- Loading Training Data ---")
        training_config = config.get('training', {})
        raw_data = load_data('train', training_config)
        
        if raw_data.empty:
            raise ValueError("No training data loaded")
        
        print(f"✓ Loaded {len(raw_data)} training records")
        print(f"  Columns: {list(raw_data.columns)}")
        
        # Show data distribution if we have labels
        if 'source_label' in raw_data.columns:
            label_counts = raw_data['source_label'].value_counts()
            print(f"  Label distribution: {dict(label_counts)}")
        
        # 4. Process data through the pipeline
        print("\n--- Processing Data Through Pipeline ---")
        
        # Aggregate features into 30-minute windows
        print("Aggregating features into 30-minute windows...")
        df_agg = aggregate_to_windows(raw_data, window_size_minutes=30)
        
        if df_agg.empty:
            raise ValueError("No aggregated windows generated")
        
        print(f"✓ Created {len(df_agg)} aggregated windows")
        
        # Generate feature vectors
        print("Generating feature vectors...")
        feature_gen = FeatureGenerator()
        feature_gen.initialize_rules()
        X, y = feature_gen.generate_feature_vectors(df_agg, mode='train')
        
        print(f"✓ Generated feature matrix: {X.shape}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        if y is not None:
            print(f"  Label distribution - Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
        
        # 5. Load the trained model
        print("\n--- Loading Trained Model ---")
        model = joblib.load(model_path)
        print(f"✓ Loaded model: {type(model).__name__}")
        print(f"  Model expects {model.n_features_in_} features")
        print(f"  Model has {len(model.estimators_)} trees")
        
        # Verify feature compatibility
        if X.shape[1] != model.n_features_in_:
            raise ValueError(f"Feature mismatch: model expects {model.n_features_in_}, got {X.shape[1]}")
        
        # 6. Prepare data for SHAP explanation
        print("\n--- Preparing SHAP Explanation ---")
        
        # Get feature names from the feature generator
        feature_names = feature_gen.get_feature_names()
        print(f"✓ Retrieved {len(feature_names)} feature names")
        
        # Use a subset of training data as background for SHAP (for performance)
        max_background_samples = min(50, X.shape[0])
        if hasattr(X, 'iloc'):
            # X is a DataFrame
            background_data = X.iloc[:max_background_samples]
        else:
            # X is a numpy array
            background_data = X[:max_background_samples]
        
        print(f"✓ Prepared background data: {background_data.shape}")
        
        # 7. Test the new explainer interface
        print("\n--- Testing SHAP Explainer (New Interface) ---")
        
        # Create output directory
        output_dir = os.path.join(PROJECT_ROOT, 'test_output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize explainer with no parameters (new interface)
        explainer = Explainer()
        print("✓ Initialized SHAP explainer")
        
        # Run explanation with all features enabled
        print("Running SHAP explanation with visualizations...")
        results = explainer.explain(
            model=model,
            data=background_data,
            feature_name_list=feature_names,
            output_dir=output_dir,
            plot=True,  # Generate plots
            plot_in_terminal=True,  # Display in terminal
            summary_report=True  # Generate markdown report
        )
        
        print(f"✓ SHAP explanation completed successfully!")
        
        # 8. Display results summary
        print("\n--- Results Summary ---")
        print(f"Model Type: {results['model_type']}")
        print(f"Samples Analyzed: {results['samples_analyzed']}")
        print(f"Features Count: {results['features_count']}")
        print(f"Analysis Timestamp: {results['timestamp']}")
        
        # Show top features
        if results.get('feature_importance'):
            top_features = results['feature_importance'][:10]
            print(f"\nTop 10 Most Important Features:")
            print("-" * 60)
            for feature in top_features:
                print(f"  {feature['rank']:2d}. {feature['feature']:<40} {feature['importance']:.6f}")
        
        # Show generated files
        if results.get('output_files'):
            print(f"\nGenerated Output Files:")
            for output_type, file_path in results['output_files'].items():
                if file_path and os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  {output_type}: {file_path} ({file_size:,} bytes)")
        
        # 9. Test model predictions for verification
        print("\n--- Model Prediction Verification ---")
        
        # Test on a few samples to verify the model is working
        test_samples = 5
        test_data = background_data[:test_samples] if hasattr(background_data, 'iloc') else background_data[:test_samples]
        
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)
        
        print(f"Predictions on {test_samples} samples:")
        for i in range(test_samples):
            pred = predictions[i]
            prob_max = probabilities[i].max()
            print(f"  Sample {i+1}: Prediction = {pred}, Confidence = {prob_max:.2%}")
        
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("✅ The SHAP explainer works correctly with the trained model and training data")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return False

def test_explainer_simple_example():
    """
    Simple test with minimal synthetic data to verify basic functionality.
    """
    print("\n" + "=" * 80)
    print("SIMPLE FUNCTIONALITY TEST")
    print("=" * 80)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create simple synthetic data
        X, y = make_classification(
            n_samples=100, 
            n_features=20, 
            n_classes=2, 
            random_state=42
        )
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(20)]
        
        # Test the explainer
        explainer = Explainer()
        
        # Use first 10 samples as background
        background_data = X[:10]
        
        output_dir = os.path.join(PROJECT_ROOT, 'test_output')
        
        results = explainer.explain(
            model=model,
            data=background_data,
            feature_name_list=feature_names,
            output_dir=output_dir,
            plot=False,  # Skip plots for simple test
            plot_in_terminal=True,  # Show terminal output
            summary_report=False  # Skip report for simple test
        )
        
        print(f"✅ Simple test passed! Analyzed {results['samples_analyzed']} samples")
        return True
        
    except Exception as e:
        print(f"❌ Simple test failed: {str(e)}")
        return False

if __name__ == "__main__":
    """
    Main test execution.
    """
    print("SHAP Explainer Test Suite")
    print("Testing the new simplified explainer interface")
    print(f"Project root: {PROJECT_ROOT}")
    
    # Run simple test first
    print("\n1. Running simple functionality test...")
    simple_success = test_explainer_simple_example()
    
    # Run full test with trained model
    print("\n2. Running full test with trained model...")
    full_success = test_explainer_with_trained_model()
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Simple Test:    {'✅ PASSED' if simple_success else '❌ FAILED'}")
    print(f"Full Test:      {'✅ PASSED' if full_success else '❌ FAILED'}")
    print(f"Overall:        {'✅ ALL TESTS PASSED' if simple_success and full_success else '❌ SOME TESTS FAILED'}")
    print(f"{'='*80}")
    
    # Exit with appropriate code
    sys.exit(0 if simple_success and full_success else 1)
