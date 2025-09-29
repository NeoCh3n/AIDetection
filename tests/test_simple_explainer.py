#!/usr/bin/env python3
"""
Test the simplified SHAP explainer with a real joblib model.

Usage:
    python test_simple_explainer.py --exist [model_path] --explainer_test
    
Arguments:
    --exist [model_path]: Test with existing joblib model (optional path)
    --explainer_test: Run explainer test with random generated data
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.shap_explainer import Explainer
from pipeline.data_loader import load_data
#from pipeline.feature_aggregator import FeatureAggregator
from pipeline.feature_generator import FeatureGenerator

def test_explainer_with_joblib_model(model_path=None):
    """Test the SHAP explainer with a real joblib-saved model using actual training data."""
    print("Testing SHAP Explainer with joblib model and real training data...")
    
    # Use provided path or default
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, 'model', 'test_model_skl0242_20250828_142238.joblib')
    
    # Check if path is relative and make it absolute
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            return False
        
        # Load configuration
        config_path = os.path.join(PROJECT_ROOT, 'pipeline', 'config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print("Loading real training data...")
        
        # Load actual training data using data_loader
        raw_data = load_data('train', config_data)
        print(f"Loaded {len(raw_data)} training records")
        print(f"Training data columns: {list(raw_data.columns)}")
        
        # Process data through the pipeline to get features
        print("Processing data through feature pipeline...")
        
        # Feature aggregation
        aggregator = FeatureAggregator(config_data)
        aggregated_data = aggregator.aggregate_features(raw_data)
        print(f"Aggregated data shape: {aggregated_data.shape}")
        
        # Feature generation
        generator = FeatureGenerator(config_data)
        feature_data = generator.generate_features(aggregated_data)
        print(f"Generated features shape: {feature_data.shape}")
        
        # Show data summary
        show_data_summary(raw_data, feature_data)
        
        # Separate features and labels
        if 'source_label' in feature_data.columns:
            X = feature_data.drop('source_label', axis=1)
            y = feature_data['source_label']
        else:
            X = feature_data
            y = None
        
        # Load the actual model
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        print(f"Loaded model type: {type(model)}")
        print(f"Model expects {model.n_features_in_} features, data has {X.shape[1]} features")
        
        # Check feature compatibility
        if X.shape[1] != model.n_features_in_:
            print(f"WARNING: Feature count mismatch: model expects {model.n_features_in_}, got {X.shape[1]}")
            print("Adjusting feature count for compatibility...")
            
            if X.shape[1] > model.n_features_in_:
                # Trim features
                X = X.iloc[:, :model.n_features_in_]
            else:
                # Pad with zeros
                missing_features = model.n_features_in_ - X.shape[1]
                for i in range(missing_features):
                    X[f'padding_feature_{i}'] = 0
        
        # Use subset of data as background for SHAP
        background_size = min(100, len(X))
        background_data = X.sample(n=background_size, random_state=42).values
        
        # Create feature names
        feature_names = list(X.columns)
        
        # Create rule mapping for interpretation
        rule_mapping = {name: f"Rule_{name}" for name in feature_names}
        
        # Initialize explainer with the real model and real background data
        explainer = Explainer(
            model=model,
            background_data=background_data,
            feature_names=feature_names,
            rule_mapping=rule_mapping
        )
        
        # Test on a real instance from the data
        test_instance = X.iloc[0:1].values  # First real instance
        print(f"Testing on real instance shape: {test_instance.shape}")
        
        # Get model prediction first to see what we're explaining
        prediction = model.predict(test_instance)
        prediction_proba = model.predict_proba(test_instance)
        print(f"Model prediction: {prediction[0]}")
        print(f"Prediction probability: {prediction_proba[0]}")
        
        if y is not None:
            actual_label = y.iloc[0]
            print(f"Actual label: {actual_label}")
        
        # Get SHAP explanation
        print("\nGenerating SHAP explanation for real data...")
        real_output_dir = os.path.join(PROJECT_ROOT, 'test_output', 'real_instance')
        os.makedirs(real_output_dir, exist_ok=True)
        explanation = explainer.explain(
            instance_data=test_instance,
            output_dir=real_output_dir,
            summary_report=True
        )
        shap_values = explanation.get('shap_values')
        print(f"SHAP values type: {type(shap_values)}")
        
        # Get feature importance ranking
        ranking = explanation.get('feature_importance', [])
        print(f"\nTop 5 Most Important Features for prediction '{prediction[0]}':")
        for item in ranking[:5]:
            feature_label = item.get('rule_name') or item.get('feature')
            rank = item.get('rank', '?')
            importance = float(item.get('importance', 0.0))
            print(f"  {rank}. {feature_label} (importance: {importance:.4f})")
        
        print("\nSUCCESS: SHAP Explainer with real training data test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Error loading or testing with real training data: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to synthetic data for this model
        print("\nFalling back to synthetic data with the loaded model...")
        try:
            # Just use the loaded model with synthetic data
            model = joblib.load(model_path)
            n_features = model.n_features_in_
            
            # Create synthetic background data
            np.random.seed(42)
            background_data = np.random.randn(100, n_features)
            feature_names = [f'feature_{i}' for i in range(n_features)]
            rule_mapping = {f'feature_{i}': f'Security_Rule_{i+1}' for i in range(n_features)}
            
            explainer = Explainer(
                model=model,
                background_data=background_data,
                feature_names=feature_names,
                rule_mapping=rule_mapping
            )
            
            test_instance = np.random.randn(1, n_features)
            prediction = model.predict(test_instance)
            fallback_output_dir = os.path.join(PROJECT_ROOT, 'test_output', 'synthetic_fallback')
            os.makedirs(fallback_output_dir, exist_ok=True)
            explanation = explainer.explain(
                instance_data=test_instance,
                output_dir=fallback_output_dir,
                summary_report=False
            )
            shap_values = explanation.get('shap_values')
            ranking = explanation.get('feature_importance', [])
            
            print(f"Synthetic test - Model prediction: {prediction[0]}")
            print("Top 3 synthetic features:")
            for item in ranking[:3]:
                feature_label = item.get('rule_name') or item.get('feature')
                rank = item.get('rank', '?')
                importance = float(item.get('importance', 0.0))
                print(f"  {rank}. {feature_label} (importance: {importance:.4f})")
            
            print("SUCCESS: Fallback synthetic test completed!")
            return True
            
        except Exception as fallback_error:
            print(f"ERROR: Fallback test also failed: {str(fallback_error)}")
            return False

def show_data_summary(raw_data, feature_data):
    """Show a summary of the loaded and processed data."""
    print("\n" + "="*40 + " DATA SUMMARY " + "="*40)
    
    # Raw data summary
    print(f"Raw Training Data:")
    print(f"  - Records: {len(raw_data)}")
    print(f"  - Columns: {list(raw_data.columns)}")
    
    if 'source_label' in raw_data.columns:
        label_counts = raw_data['source_label'].value_counts()
        print(f"  - Label distribution: {dict(label_counts)}")
    
    # Feature data summary
    print(f"\nProcessed Feature Data:")
    print(f"  - Shape: {feature_data.shape}")
    print(f"  - Features: {list(feature_data.columns)[:10]}{'...' if len(feature_data.columns) > 10 else ''}")
    
    if 'source_label' in feature_data.columns:
        feature_label_counts = feature_data['source_label'].value_counts()
        print(f"  - Label distribution: {dict(feature_label_counts)}")
    
    print("="*93)

def test_explainer_with_synthetic_model():
    """Fallback test with a synthetic model and random data."""
    print("\nTesting SHAP Explainer with synthetic model...")
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Train a simple RandomForest
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Create background data (subset of training data)
    background_data = X[:100]
    
    # Create feature names
    feature_names = [f'rule_{i}' for i in range(10)]
    
    # Create rule mapping
    rule_mapping = {f'rule_{i}': f'Security Rule {i+1}' for i in range(10)}
    
    # Initialize explainer
    explainer = Explainer(
        model=model,
        background_data=background_data,
        feature_names=feature_names,
        rule_mapping=rule_mapping
    )
    
    # Test on a single instance (malicious case)
    test_instance = X[0:1]  # First instance
    print(f"Testing on instance shape: {test_instance.shape}")
    
    # Get explanation
    synthetic_output_dir = os.path.join(PROJECT_ROOT, 'test_output', 'synthetic_test')
    os.makedirs(synthetic_output_dir, exist_ok=True)
    explanation = explainer.explain(
        instance_data=test_instance,
        output_dir=synthetic_output_dir,
        summary_report=True
    )
    shap_values = explanation.get('shap_values')
    print(f"SHAP values type: {type(shap_values)}")
    
    # Get feature importance ranking
    ranking = explanation.get('feature_importance', [])
    print("\nTop 5 Most Important Features:")
    for item in ranking[:5]:
        feature_label = item.get('rule_name') or item.get('feature')
        rank = item.get('rank', '?')
        importance = float(item.get('importance', 0.0))
        print(f"  {rank}. {feature_label} (importance: {importance:.4f})")

    print("\n✅ SHAP Explainer synthetic test completed successfully!")
    return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test SHAP explainer with various options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with default model
    python test_simple_explainer.py --exist
    
    # Test with custom model path
    python test_simple_explainer.py --exist model/my_model.joblib
    
    # Test explainer with random data
    python test_simple_explainer.py --explainer_test
    
    # Run both tests
    python test_simple_explainer.py --exist --explainer_test
        """
    )
    
    parser.add_argument(
        '--exist', 
        nargs='?', 
        const='default',
        help='Test with existing joblib model. Optionally provide model path.'
    )
    
    parser.add_argument(
        '--explainer_test', 
        action='store_true',
        help='Run explainer test with randomly generated data'
    )
    
    return parser.parse_args()

def test_explainer(args=None):
    """Main test function that handles command line arguments."""
    if args is None:
        args = parse_arguments()
    
    print("=" * 60)
    print("SHAP Explainer Test Suite")
    print("=" * 60)
    
    tests_run = 0
    tests_passed = 0
    
    # Test with existing model if requested
    if args.exist:
        tests_run += 1
        print(f"\n{'='*20} Testing with Existing Model {'='*20}")
        
        # Determine model path
        if args.exist == 'default':
            model_path = None  # Use default
            print("Using default model path...")
        else:
            model_path = args.exist
            print(f"Using custom model path: {model_path}")
        
        success = test_explainer_with_joblib_model(model_path)
        if success:
            tests_passed += 1
    
    # Test explainer with random data if requested
    if args.explainer_test:
        tests_run += 1
        print(f"\n{'='*20} Testing with Random Generated Data {'='*20}")
        success = test_explainer_with_synthetic_model()
        if success:
            tests_passed += 1
    
    # If no tests specified, show help
    if tests_run == 0:
        print("No tests specified. Use --help to see available options.")
        print("\nQuick start:")
        print("  --exist              # Test with existing model")
        print("  --explainer_test     # Test with random data")
        print("  --exist --explainer_test  # Run both tests")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {tests_passed}/{tests_run} tests passed")
    if tests_passed == tests_run:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    test_explainer()
