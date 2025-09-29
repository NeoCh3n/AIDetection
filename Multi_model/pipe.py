#!/usr/bin/env python3
"""
Object-Oriented Threat Detection Pipeline - Main Entry Point
Modern OOP implementation with model switching capability.

This module provides a simple main entry point for the restructured
threat detection pipeline using the orchestrator pattern.

Architecture Components (now modularized):
- orchestrator.py: Main pipeline orchestrator
- data/: Data handling components
- features/: Feature engineering components  
- models/: ML model implementations (supervised + clustering)
- pipelines/: Training and detection pipeline orchestrators

Usage:
    python pipe.py train --model-type random_forest --verbose
    python pipe.py detect --model-path ./model/threat_detector.joblib
    python pipe.py train --model-type kmeans  # clustering example

Python 3.6.8 Compatible
"""

import sys
import os
import argparse
import logging

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the orchestrator
from orchestrator import PipelineOrchestrator


def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(
        description="Object-Oriented Threat Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train supervised models
  python pipe.py train --model-type random_forest --verbose
  python pipe.py train --model-type svm
  python pipe.py train --model-type gradient_boosting
  
  # Train clustering models  
  python pipe.py train --model-type kmeans --verbose
  python pipe.py train --model-type dbscan
  python pipe.py train --model-type gaussian_mixture
  
  # Run detection
  python pipe.py detect --model-path ./model/threat_detector.joblib
  python pipe.py detect --model-path ./model/kmeans_cluster.joblib
  
  # List available models
  python pipe.py list-models
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Pipeline operation mode')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument(
        '--model-type', 
        default='random_forest',
        help='Model type to train (see list-models for options)'
    )
    train_parser.add_argument('--config', help='Configuration file path')
    train_parser.add_argument('--save-path', help='Custom path to save trained model')
    train_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # Detection command
    detect_parser = subparsers.add_parser('detect', help='Run threat detection')
    detect_parser.add_argument('--model-path', help='Path to trained model')
    detect_parser.add_argument('--config', help='Configuration file path')
    detect_parser.add_argument('--threshold', type=float, default=0.5, 
                              help='Threat probability threshold (for supervised models)')
    detect_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')
    list_parser.add_argument('--type', choices=['supervised', 'clustering', 'all'], 
                            default='all', help='Type of models to list')
    list_parser.add_argument('--detailed', action='store_true', 
                            help='Show detailed model information')
    
    # Get recommendations command
    recommend_parser = subparsers.add_parser('recommend', help='Get model recommendations')
    recommend_parser.add_argument('--use-case', help='Use case for recommendations')
    
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.mode:
        parser.print_help()
        return 0
    
    # Setup logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Create orchestrator
        config_path = getattr(args, 'config', None)
        orchestrator = PipelineOrchestrator(config_path)
        
        if args.mode == 'train':
            return handle_training(orchestrator, args)
        elif args.mode == 'detect':
            return handle_detection(orchestrator, args)
        elif args.mode == 'list-models':
            return handle_list_models(orchestrator, args)
        elif args.mode == 'recommend':
            return handle_recommendations(orchestrator, args)
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_training(orchestrator, args):
    """Handle training mode."""
    print(f"Starting training with {args.model_type} model")
    
    # Validate model type
    if not orchestrator.validate_model_type(args.model_type):
        available = orchestrator.list_available_models()
        print(f"Unknown model type: {args.model_type}")
        print(f"Available models: {', '.join(available)}")
        return 1
    
    # Show model type info
    if orchestrator.is_clustering_model(args.model_type):
        print(f"Training clustering model: {args.model_type}")
    else:
        print(f"Training supervised model: {args.model_type}")
    
    # Training parameters
    train_kwargs = {}
    if args.save_path:
        train_kwargs['save_path'] = args.save_path
    
    # Execute training
    results = orchestrator.train(args.model_type, **train_kwargs)
    
    # Print results
    print("✅ Training completed successfully!")
    print(f"📁 Model saved to: {results['model_path']}")
    print(f"📊 Training samples: {results.get('training_samples', 'N/A')}")
    
    if 'test_samples' in results:
        print(f"🧪 Test samples: {results['test_samples']}")
    
    # Show evaluation metrics
    evaluation = results.get('evaluation', {})
    if 'roc_auc' in evaluation:
        print(f"📈 ROC AUC: {evaluation['roc_auc']:.4f}")
    
    if 'silhouette_score' in evaluation:
        print(f"📈 Silhouette Score: {evaluation['silhouette_score']:.4f}")
    
    if 'n_clusters' in evaluation:
        print(f"🔍 Clusters found: {evaluation['n_clusters']}")
        if 'noise_points' in evaluation:
            print(f"🔍 Noise points: {evaluation['noise_points']} ({evaluation.get('noise_ratio', 0)*100:.1f}%)")
    
    return 0


def handle_detection(orchestrator, args):
    """Handle detection mode."""
    model_path = args.model_path or "./model/threat_detector.joblib"
    print(f"🔍 Starting detection with model: {model_path}")
    
    # Detection parameters
    detect_kwargs = {}
    if hasattr(args, 'threshold'):
        detect_kwargs['threshold'] = args.threshold
    
    # Execute detection
    results = orchestrator.detect(model_path, **detect_kwargs)
    
    # Print results
    print("✅ Detection completed!")
    print(f"📊 Status: {results['status']}")
    
    if results['status'] == 'no_data':
        print("ℹ️  No new data available for detection")
        return 0
    
    # Show results based on model type
    if results.get('model_type') == 'clustering':
        print(f"🔍 Total windows: {results.get('total_windows', 0)}")
        print(f"📊 Clusters found: {results.get('clusters_found', 0)}")
        print(f"⚠️  Anomalies detected: {results.get('anomaly_count', 0)}")
        
        if results.get('cluster_distribution'):
            print("📈 Cluster distribution:")
            for cluster_id, count in results['cluster_distribution'].items():
                print(f"   Cluster {cluster_id}: {count} windows")
    else:
        print(f"🔍 Total windows: {results.get('total_windows', 0)}")
        print(f"⚠️  Threats detected: {results.get('threat_count', 0)}")
    
    if results.get('alerts'):
        print(f"🚨 Alerts generated: {len(results['alerts'])}")
        for i, alert in enumerate(results['alerts'][:5], 1):  # Show first 5 alerts
            alert_type = alert.get('alert_type', 'unknown')
            window_id = alert.get('window_id', 'unknown')
            hosts = len(alert.get('hostnames', []))
            print(f"   Alert {i}: {alert_type} - Window {window_id} - {hosts} hosts")
        
        if len(results['alerts']) > 5:
            print(f"   ... and {len(results['alerts']) - 5} more alerts")
    
    return 0


def handle_list_models(orchestrator, args):
    """Handle list-models mode."""
    print("🔧 Available Models")
    print("=" * 60)
    
    if args.type in ['supervised', 'all']:
        supervised_models = orchestrator.get_supervised_models()
        print(f"\n📊 Supervised Models ({len(supervised_models)}):")
        for model in sorted(supervised_models):
            print(f"   • {model}")
    
    if args.type in ['clustering', 'all']:
        clustering_models = orchestrator.get_clustering_models()
        print(f"\n🔍 Clustering Models ({len(clustering_models)}):")
        for model in sorted(clustering_models):
            print(f"   • {model}")
    
    if args.detailed:
        print("\n📋 Detailed Model Information:")
        print("=" * 60)
        model_info = orchestrator.get_model_info()
        
        for model_type, info in sorted(model_info.items()):
            model_category = info.get('type', 'unknown')
            class_name = info.get('class_name', 'Unknown')
            scaling = "✓" if info.get('requires_scaling', False) else "✗"
            grid_search = "✓" if info.get('supports_grid_search', False) else "✗"
            
            print(f"\n{model_type} ({model_category}):")
            print(f"   Class: {class_name}")
            print(f"   Requires Scaling: {scaling}")
            print(f"   Grid Search: {grid_search}")
            
            if 'description' in info:
                print(f"   Description: {info['description']}")
    
    return 0


def handle_recommendations(orchestrator, args):
    """Handle recommendations mode."""
    if args.use_case:
        recommendations = orchestrator.get_model_recommendations(args.use_case)
        print(f"🎯 Recommended models for '{args.use_case}':")
        if recommendations:
            for model in recommendations:
                model_type = "clustering" if orchestrator.is_clustering_model(model) else "supervised"
                print(f"   • {model} ({model_type})")
        else:
            print("   No specific recommendations found for this use case")
    else:
        print("💡 Available use cases for recommendations:")
        print("\nSupervised Models:")
        print("   • interpretability")
        print("   • performance") 
        print("   • speed")
        print("   • high_dimensional")
        print("   • large_datasets")
        print("   • small_datasets")
        
        print("\nClustering Models:")
        print("   • anomaly_detection")
        print("   • exploration")
        print("   • fast_clustering")
        print("   • irregular_shapes")
        print("   • probability_estimates")
        print("   • noise_handling")
        
        print(f"\nUsage: python {sys.argv[0]} recommend --use-case <use_case>")
    
    return 0


def component_test():
    """Basic test of the pipeline components."""
    print("🧪 Object-Oriented Threat Detection Pipeline - Component Test")
    print("=" * 60)
    
    try:
        # Test orchestrator creation
        orchestrator = PipelineOrchestrator()
        print("✅ Created pipeline orchestrator successfully")
        
        # Test model listing
        available_models = orchestrator.list_available_models()
        supervised_models = orchestrator.get_supervised_models()
        clustering_models = orchestrator.get_clustering_models()
        
        print(f"✅ Available models: {len(available_models)} total")
        print(f"   📊 Supervised: {len(supervised_models)}")
        print(f"   🔍 Clustering: {len(clustering_models)}")
        
        # Test model creation for a few models
        test_models = ['random_forest', 'svm', 'kmeans', 'dbscan']
        for model_type in test_models:
            if model_type in available_models:
                try:
                    from models.model_factory import ModelFactory
                    config = orchestrator.get_config()
                    model = ModelFactory.create_model(model_type, config)
                    model_category = "clustering" if orchestrator.is_clustering_model(model_type) else "supervised"
                    print(f"✅ Created {model_type} ({model_category}) model successfully")
                except Exception as e:
                    print(f"❌ Failed to create {model_type} model: {e}")
        
        print("\n🎉 Component test completed successfully!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if running as main script or as component test
    if len(sys.argv) > 1 and sys.argv[1] in ['train', 'detect', 'list-models', 'recommend']:
        sys.exit(main())
    else:
        # Run component test if no valid command provided
        component_test()