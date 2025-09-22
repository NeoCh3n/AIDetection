#!/usr/bin/env python3
"""
OOP Pipeline Usage Examples
Demonstrates how to use the new object-oriented pipeline with different models.
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.pipe import PipelineOrchestrator, ModelFactory


def example_basic_training():
    """Basic training example with Random Forest."""
    print("=== Basic Training Example ===")
    
    # Create orchestrator with default config
    orchestrator = PipelineOrchestrator()
    
    print(f"Available models: {orchestrator.list_available_models()}")
    
    try:
        # Train with Random Forest (default)
        results = orchestrator.train(model_type='random_forest')
        
        print("Training Results:")
        print(f"- Model Type: {results['model_type']}")
        print(f"- Model Path: {results['model_path']}")
        print(f"- Training Samples: {results['training_samples']}")
        print(f"- Test Samples: {results['test_samples']}")
        
        evaluation = results.get('evaluation', {})
        if 'roc_auc' in evaluation:
            print(f"- ROC AUC: {evaluation['roc_auc']:.4f}")
        
        if 'top_features' in evaluation:
            print(f"- Top 5 Important Rules:")
            for i, feature in enumerate(evaluation['top_features'][:5]):
                print(f"  {i+1}. Rule {feature['rule_id']}: {feature['importance']:.4f}")
                
    except Exception as e:
        print(f"Training failed: {e}")


def example_model_comparison():
    """Compare different models."""
    print("\n=== Model Comparison Example ===")
    
    orchestrator = PipelineOrchestrator()
    available_models = orchestrator.list_available_models()
    
    results = {}
    
    for model_type in available_models[:3]:  # Test first 3 models
        print(f"\nTraining {model_type}...")
        try:
            # Train model with unique save path
            model_path = f"./model/threat_detector_{model_type}.joblib"
            result = orchestrator.train(
                model_type=model_type,
                save_path=model_path
            )
            
            results[model_type] = {
                'roc_auc': result.get('evaluation', {}).get('roc_auc', 'N/A'),
                'model_path': result['model_path']
            }
            
            print(f"✓ {model_type} trained successfully")
            
        except Exception as e:
            print(f"✗ {model_type} training failed: {e}")
            results[model_type] = {'error': str(e)}
    
    # Print comparison
    print("\nModel Comparison Results:")
    print("-" * 50)
    for model_type, result in results.items():
        if 'error' in result:
            print(f"{model_type:20}: ERROR - {result['error']}")
        else:
            print(f"{model_type:20}: ROC AUC = {result['roc_auc']}")


def example_custom_config():
    """Example using custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Use custom config file
    config_path = "./pipeline/config_oop.json"
    
    if os.path.exists(config_path):
        orchestrator = PipelineOrchestrator(config_path)
        print(f"✓ Loaded custom configuration from {config_path}")
        
        # Show model info
        model_info = ModelFactory.get_model_info()
        print("Available Models:")
        for model_type, info in model_info.items():
            scaling = "Yes" if info['requires_scaling'] else "No"
            importance = "Yes" if info['supports_feature_importance'] else "No"
            print(f"  {model_type:20} | Scaling: {scaling:3} | Importance: {importance}")
    else:
        print(f"✗ Config file not found: {config_path}")


def example_detection():
    """Example detection pipeline."""
    print("\n=== Detection Pipeline Example ===")
    
    # Check if trained model exists
    model_path = "./model/threat_detector.joblib"
    
    if os.path.exists(model_path):
        try:
            orchestrator = PipelineOrchestrator()
            results = orchestrator.detect(model_path=model_path)
            
            print("Detection Results:")
            print(f"- Status: {results['status']}")
            print(f"- Total Windows: {results.get('total_windows', 'N/A')}")
            print(f"- Threats Detected: {results.get('threat_count', 'N/A')}")
            print(f"- Alerts Generated: {len(results.get('alerts', []))}")
            
            # Show first few alerts
            alerts = results.get('alerts', [])
            if alerts:
                print("Sample Alerts:")
                for i, alert in enumerate(alerts[:3]):
                    print(f"  Alert {i+1}:")
                    print(f"    - Window: {alert['window_id']}")
                    print(f"    - Hosts: {len(alert['hostnames'])}")
                    print(f"    - Confidence: {alert.get('confidence', 'N/A')}")
            
        except Exception as e:
            print(f"Detection failed: {e}")
    else:
        print(f"✗ No trained model found at {model_path}")
        print("Run training first with: python pipeline/pipe.py train")


def example_interactive_menu():
    """Interactive menu for testing different functionality."""
    print("\n=== Interactive Pipeline Demo ===")
    
    while True:
        print("\nChoose an option:")
        print("1. List available models")
        print("2. Train Random Forest model")
        print("3. Train SVM model")
        print("4. Compare all models")
        print("5. Run detection")
        print("6. Show model information")
        print("0. Exit")
        
        try:
            choice = input("Enter choice (0-6): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                orchestrator = PipelineOrchestrator()
                models = orchestrator.list_available_models()
                print(f"Available models: {', '.join(models)}")
                
            elif choice == '2':
                print("Training Random Forest...")
                example_basic_training()
                
            elif choice == '3':
                print("Training SVM...")
                orchestrator = PipelineOrchestrator()
                try:
                    results = orchestrator.train('svm')
                    print(f"✓ SVM training completed: {results['model_path']}")
                except Exception as e:
                    print(f"✗ SVM training failed: {e}")
                    
            elif choice == '4':
                example_model_comparison()
                
            elif choice == '5':
                example_detection()
                
            elif choice == '6':
                model_info = ModelFactory.get_model_info()
                print("\nModel Information:")
                print("-" * 60)
                print(f"{'Model':20} | {'Scaling':8} | {'Importance':10} | {'Class'}")
                print("-" * 60)
                for model_type, info in model_info.items():
                    scaling = "Required" if info['requires_scaling'] else "Optional"
                    importance = "Yes" if info['supports_feature_importance'] else "No"
                    print(f"{model_type:20} | {scaling:8} | {importance:10} | {info['class_name']}")
                    
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("OOP Threat Detection Pipeline - Usage Examples")
    print("=" * 50)
    
    # Run examples
    try:
        example_basic_training()
        example_custom_config()
        example_detection()
        
        # Interactive menu (optional)
        if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
            example_interactive_menu()
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()