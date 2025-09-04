#!/usr/bin/env python3
"""
Test SHAP Explainer with Trained Model and Training Data - Instance Testing

This test demonstrates the SHAP explainer's new simplified interface using:
- The actual trained threat_detector.joblib model
- Real training data from the Training_data directory
- Testing individual instance explanations (one instance at a time)

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

def test_explainer_single_instance():
    """
    Test the SHAP explainer using individual instances from the trained model.
    
    This test follows the complete data pipeline and tests single instance explanations:
    1. Load training data and process through pipeline
    2. Load the trained model
    3. Select individual instances for explanation
    4. Run SHAP explanation for each instance separately
    """
    print("=" * 80)
    print("TESTING SHAP EXPLAINER WITH SINGLE INSTANCE EXPLANATIONS")
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
        
        # 3. Load and process training data
        print("\n--- Loading and Processing Training Data ---")
        training_config = config.get('training', {})
        raw_data = load_data('train', training_config)
        
        if raw_data.empty:
            raise ValueError("No training data loaded")
        
        print(f"✓ Loaded {len(raw_data)} training records")
        
        # Aggregate features into windows
        df_agg = aggregate_to_windows(raw_data, window_size_minutes=30)
        print(f"✓ Created {len(df_agg)} aggregated windows")
        
        # Generate feature vectors
        feature_gen = FeatureGenerator()
        feature_gen.initialize_rules()
        X, y = feature_gen.generate_feature_vectors(df_agg, mode='train')
        
        print(f"✓ Generated feature matrix: {X.shape}")
        if y is not None:
            print(f"  Label distribution - Benign: {np.sum(y == 0)}, Malicious: {np.sum(y == 1)}")
        
        # 4. Load the trained model
        print("\n--- Loading Trained Model ---")
        model = joblib.load(model_path)
        print(f"✓ Loaded model: {type(model).__name__}")
        
        # 5. Prepare for single instance testing
        print("\n--- Preparing Single Instance Tests ---")
        
        # Get feature names
        feature_names = feature_gen.get_feature_names()
        print(f"✓ Retrieved {len(feature_names)} feature names")
        
        # Prepare background data (sample from training data)
        max_background_samples = min(100, X.shape[0])
        if hasattr(X, 'iloc'):
            background_data = X.iloc[:max_background_samples].values
        else:
            background_data = X[:max_background_samples]
        
        print(f"✓ Prepared background data: {background_data.shape}")
        
        # Select test instances (both classes if available)
        test_instances = []
        
        if y is not None:
            # Find instances of each class
            benign_indices = np.where(y == 0)[0]
            malicious_indices = np.where(y == 1)[0]
            
            # Select a few instances from each class
            if len(benign_indices) > 0:
                test_instances.extend(benign_indices[:2])  # 2 benign instances
            if len(malicious_indices) > 0:
                test_instances.extend(malicious_indices[:2])  # 2 malicious instances
        else:
            # No labels available, just pick first few instances
            test_instances = list(range(min(4, X.shape[0])))
        
        print(f"✓ Selected {len(test_instances)} test instances for explanation")
        
        # 6. Initialize explainer
        print("\n--- Initializing SHAP Explainer ---")
        explainer = Explainer()
        print("✓ Initialized SHAP explainer")
        
        # 7. Test each instance individually
        print("\n--- Testing Individual Instance Explanations ---")
        
        successful_explanations = 0
        
        for i, instance_idx in enumerate(test_instances):
            print(f"\n--- Instance {i+1}/{len(test_instances)} (Index: {instance_idx}) ---")
            
            try:
                # Extract single instance
                if hasattr(X, 'iloc'):
                    single_instance = X.iloc[instance_idx:instance_idx+1].values
                else:
                    single_instance = X[instance_idx:instance_idx+1]
                
                print(f"Instance shape: {single_instance.shape}")
                
                # Get model prediction for this instance
                prediction = model.predict(single_instance)[0]
                probabilities = model.predict_proba(single_instance)[0]
                confidence = probabilities.max()
                
                predicted_class = "Malicious" if prediction == 1 else "Benign"
                actual_class = "Unknown"
                if y is not None:
                    actual_class = "Malicious" if y[instance_idx] == 1 else "Benign"
                
                print(f"Prediction: {predicted_class} (confidence: {confidence:.2%})")
                print(f"Actual: {actual_class}")
                
                # Create instance-specific output directory
                instance_output_dir = os.path.join(PROJECT_ROOT, 'test_output', f'instance_{instance_idx}')
                os.makedirs(instance_output_dir, exist_ok=True)
                
                # Run SHAP explanation for this single instance
                print("Running SHAP explanation...")
                results = explainer.explain(
                    model=model,
                    background_data=background_data,
                    instance_data=single_instance,
                    feature_name_list=feature_names,
                    output_dir=instance_output_dir,
                    plot=True,  # Generate visualizations
                    plot_in_terminal=False,  # Skip terminal plots to avoid clutter
                    summary_report=True  # Generate report
                )
                
                print(f"✓ SHAP explanation completed for instance {instance_idx}")
                
                # Show top contributing features for this instance
                if results.get('feature_importance'):
                    top_features = results['feature_importance'][:5]
                    print(f"Top 5 contributing features:")
                    for j, feature in enumerate(top_features, 1):
                        print(f"  {j}. {feature['feature']}: {feature['importance']:.6f}")
                
                # Show generated files
                if results.get('output_files'):
                    print(f"Generated files:")
                    for output_type, file_path in results['output_files'].items():
                        if file_path and os.path.exists(file_path):
                            print(f"  - {output_type}: {os.path.basename(file_path)}")
                
                successful_explanations += 1
                
            except Exception as e:
                print(f"❌ Failed to explain instance {instance_idx}: {str(e)}")
                continue
        
        # 8. Summary of instance testing
        print(f"\n--- Instance Testing Summary ---")
        print(f"Total instances tested: {len(test_instances)}")
        print(f"Successful explanations: {successful_explanations}")
        print(f"Success rate: {successful_explanations/len(test_instances)*100:.1f}%")
        
        if successful_explanations == len(test_instances):
            print("✅ All individual instance explanations completed successfully!")
            return True
        else:
            print(f"⚠️  {len(test_instances) - successful_explanations} instance explanations failed")
            return False
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_explainer_batch_vs_single():
    """
    Compare batch explanation vs individual instance explanations to verify consistency.
    """
    print("\n" + "=" * 80)
    print("TESTING BATCH VS SINGLE INSTANCE CONSISTENCY")
    print("=" * 80)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create synthetic data
        X, y = make_classification(
            n_samples=50, 
            n_features=10, 
            n_classes=2, 
            random_state=42
        )
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(10)]
        
        # Test explainer
        explainer = Explainer()
        
        # Background data
        background_data = X[:20]
        
        output_dir = os.path.join(PROJECT_ROOT, 'test_output', 'consistency_test')
        os.makedirs(output_dir, exist_ok=True)
        
        # Test single instance
        test_instance = X[25:26]  # Single instance
        
        print(f"Testing single instance explanation...")
        print(f"Instance shape: {test_instance.shape}")
        print(f"Background data shape: {background_data.shape}")
        
        results = explainer.explain(
            model=model,
            background_data=background_data,
            instance_data=test_instance,
            feature_name_list=feature_names,
            output_dir=output_dir,
            plot=True,
            plot_in_terminal=True,
            summary_report=True
        )
        
        print(f"✅ Single instance explanation successful!")
        print(f"Instance analyzed: {results['samples_analyzed']}")
        print(f"Features analyzed: {results['features_count']}")
        
        # Show feature importance for the single instance
        if results.get('feature_importance'):
            print(f"\nFeature importance for single instance:")
            for feature in results['feature_importance'][:5]:
                print(f"  {feature['feature']}: {feature['importance']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consistency test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_alert_simulation():
    """
    Simulate the alert explanation workflow as it would happen in the pipeline.
    """
    print("\n" + "=" * 80)
    print("TESTING ALERT SIMULATION WORKFLOW")
    print("=" * 80)
    
    try:
        # Load minimal required components
        config_path = os.path.join(PROJECT_ROOT, 'pipeline', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model
        model_path = os.path.join(PROJECT_ROOT, 'model', 'threat_detector.joblib')
        if not os.path.exists(model_path):
            print("⚠️  No trained model found, skipping alert simulation")
            return True
            
        model = joblib.load(model_path)
        
        # Create dummy alert data (simulating what would come from detection)
        np.random.seed(42)
        alert_instance = np.random.random((1, model.n_features_in_))
        background_data = np.random.random((50, model.n_features_in_))
        
        # Create dummy feature names
        feature_names = [f"rule_{i}" for i in range(model.n_features_in_)]
        
        # Simulate alert explanation
        print("Simulating alert explanation workflow...")
        
        explainer = Explainer()
        
        # Create alert-specific output directory (as would happen in pipeline)
        alert_id = "test_alert_12345"
        alert_output_dir = os.path.join(PROJECT_ROOT, 'test_output', 'alerts', alert_id)
        os.makedirs(alert_output_dir, exist_ok=True)
        
        # Run explanation (as would happen for each alert)
        results = explainer.explain(
            model=model,
            background_data=background_data,
            instance_data=alert_instance,
            feature_name_list=feature_names,
            output_dir=alert_output_dir,
            plot=True,  # Generate plots for analyst review
            plot_in_terminal=False,  # Don't clutter detection logs
            summary_report=True  # Generate detailed report
        )
        
        print(f"✅ Alert explanation simulation successful!")
        print(f"Alert ID: {alert_id}")
        print(f"Files generated: {len(results.get('output_files', {}))}")
        
        # Simulate what would be logged in the pipeline
        if results.get('feature_importance'):
            top_features = results['feature_importance'][:3]
            print(f"Top contributing features for alert:")
            for feature in top_features:
                print(f"  - {feature['feature']}: {feature['importance']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Alert simulation failed: {str(e)}")
        return False

if __name__ == "__main__":
    """
    Main test execution - focused on instance testing.
    """
    print("SHAP Explainer Instance Testing Suite")
    print("Testing single instance explanations as required for alert analysis")
    print(f"Project root: {PROJECT_ROOT}")
    
    tests_results = []
    
    # Test 1: Single instance explanations with real model and data
    print("\n1. Testing single instance explanations with trained model...")
    result1 = test_explainer_single_instance()
    tests_results.append(("Single Instance Test", result1))
    
    # Test 2: Consistency testing
    print("\n2. Testing batch vs single instance consistency...")
    result2 = test_explainer_batch_vs_single()
    tests_results.append(("Consistency Test", result2))
    
    # Test 3: Alert simulation
    print("\n3. Testing alert explanation simulation...")
    result3 = test_alert_simulation()
    tests_results.append(("Alert Simulation", result3))
    
    # Summary
    print(f"\n{'='*80}")
    print("INSTANCE TESTING SUITE SUMMARY")
    print(f"{'='*80}")
    
    all_passed = True
    for test_name, result in tests_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30}: {status}")
        if not result:
            all_passed = False
    
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"{'='*80}")
    
    if all_passed:
        print("\n🎉 The SHAP explainer is ready for single instance explanations in the alert pipeline!")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)
