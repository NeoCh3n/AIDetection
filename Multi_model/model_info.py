#!/usr/bin/env python3
"""
Model Information Utility

Display comprehensive information about available models,
their configurations, and recommendations.

Usage:
    python model_info.py [--model MODEL_TYPE] [--use-case USE_CASE]

Examples:
    python model_info.py                           # Show all models
    python model_info.py --model random_forest     # Show specific model
    python model_info.py --use-case interpretability  # Show recommended models

Python 3.6.8 Compatible
"""

import sys
import os
import json
import argparse
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model_factory import ModelFactory


def print_separator(char='-', length=80):
    """Print a separator line."""
    print(char * length)


def print_model_info(model_type: str, config: Dict[str, Any]):
    """Print detailed information for a specific model."""
    # Check supervised models first
    model_info = config.get('models', {}).get(model_type, {})
    model_category = "supervised"
    
    # If not found, check clustering models
    if not model_info:
        model_info = config.get('clustering_models', {}).get(model_type, {})
        model_category = "clustering"
    
    if not model_info:
        print(f"No configuration found for model: {model_type}")
        return
    
    print(f"\n{'=' * 20} {model_type.upper().replace('_', ' ')} {'=' * 20}")
    print(f"Category: {model_category.title()}")
    print(f"Class: {model_info.get('class_name', 'Unknown')}")
    print(f"SKlearn Class: {model_info.get('sklearn_class', 'Unknown')}")
    print(f"Description: {model_info.get('description', 'No description')}")
    
    print(f"\nCapabilities:")
    print(f"  • Requires Scaling: {model_info.get('requires_scaling', False)}")
    print(f"  • Feature Importance: {model_info.get('supports_feature_importance', False)}")
    print(f"  • Grid Search: {model_info.get('supports_grid_search', False)}")
    
    availability = model_info.get('availability', 'standard')
    if availability != 'standard':
        print(f"  • Availability: {availability}")
        if 'availability_note' in model_info:
            print(f"    Note: {model_info['availability_note']}")
    
    recommended_for = model_info.get('recommended_for', [])
    if recommended_for:
        print(f"\nRecommended for: {', '.join(recommended_for)}")
    
    # Default parameters
    default_params = model_info.get('default_params', {})
    if default_params:
        print(f"\nDefault Parameters:")
        for param, value in default_params.items():
            print(f"  • {param}: {value}")
    
    # Grid search parameters
    grid_params = model_info.get('grid_search_params', {})
    if grid_params:
        print(f"\nGrid Search Parameters:")
        for param, values in grid_params.items():
            if isinstance(values, list) and len(values) > 5:
                print(f"  • {param}: {values[:3]} ... (total: {len(values)} values)")
            else:
                print(f"  • {param}: {values}")


def print_all_models_summary(config: Dict[str, Any]):
    """Print summary of all available models."""
    factory = ModelFactory()
    availability_info = factory.get_availability_info()
    models_config = config.get('models', {})
    clustering_config = config.get('clustering_models', {})
    
    print("AVAILABLE MODELS SUMMARY")
    print_separator('=')
    
    # Create summary table
    print(f"{'Model':<20} {'Type':<12} {'Status':<12} {'Scaling':<8} {'Grid Search':<12}")
    print_separator()
    
    # Show supervised models
    print("SUPERVISED MODELS:")
    for model_type in sorted(factory._BASE_MODELS.keys()):
        model_info = models_config.get(model_type, {})
        scaling = "Yes" if model_info.get('requires_scaling', False) else "No"
        grid_search = "Yes" if model_info.get('supports_grid_search', False) else "No"
        status = "Available"
        
        print(f"{model_type:<20} {'Supervised':<12} {status:<12} {scaling:<8} {grid_search:<12}")
    
    # Show optional supervised models with their status
    optional_models = availability_info.get('optional_models', {})
    for model_type, info in optional_models.items():
        model_info = models_config.get(model_type, {})
        scaling = "Yes" if model_info.get('requires_scaling', False) else "No"
        grid_search = "Yes" if model_info.get('supports_grid_search', False) else "No"
        status = "Available" if info['status'] == 'available' else "Unavailable"
        
        print(f"{model_type:<20} {'Supervised':<12} {status:<12} {scaling:<8} {grid_search:<12}")
        
        # Show error or fix instructions for unavailable models
        if info['status'] == 'unavailable':
            print(f"  └─ Error: {info.get('error', 'Unknown error')}")
    
    print()
    print("CLUSTERING MODELS:")
    # Show clustering models
    for model_type in sorted(factory._CLUSTERING_MODELS.keys()):
        model_info = clustering_config.get(model_type, {})
        scaling = "Yes" if model_info.get('requires_scaling', False) else "No"
        grid_search = "Yes" if model_info.get('supports_grid_search', False) else "No"
        status = "Available"
        
        print(f"{model_type:<20} {'Clustering':<12} {status:<12} {scaling:<8} {grid_search:<12}")


def print_recommendations(config: Dict[str, Any]):
    """Print model recommendations by use case."""
    guidelines = config.get('model_selection_guidelines', {})
    
    print("MODEL RECOMMENDATIONS BY USE CASE")
    print_separator('=')
    
    # Handle new nested structure with supervised/clustering categories
    if 'supervised' in guidelines or 'clustering' in guidelines:
        # New format with categories
        supervised_guidelines = guidelines.get('supervised', {})
        clustering_guidelines = guidelines.get('clustering', {})
        
        if supervised_guidelines:
            print("\nSUPERVISED MODELS:")
            for use_case, models in supervised_guidelines.items():
                if use_case == 'default_recommendation':
                    continue
                print(f"\n{use_case.replace('_', ' ').title()}:")
                for model in models:
                    print(f"  • {model}")
            
            default = supervised_guidelines.get('default_recommendation')
            if default:
                print(f"\nDefault Supervised: {default}")
        
        if clustering_guidelines:
            print("\nCLUSTERING MODELS:")
            for use_case, models in clustering_guidelines.items():
                if use_case == 'default_recommendation':
                    continue
                print(f"\n{use_case.replace('_', ' ').title()}:")
                for model in models:
                    print(f"  • {model}")
            
            default = clustering_guidelines.get('default_recommendation')
            if default:
                print(f"\nDefault Clustering: {default}")
    
    else:
        # Legacy format (flat structure)
        for use_case, models in guidelines.items():
            if use_case == 'default_recommendation':
                continue
            
            print(f"\n{use_case.replace('_', ' ').title()}:")
            for model in models:
                print(f"  • {model}")
        
        default = guidelines.get('default_recommendation')
        if default:
            print(f"\nDefault Recommendation: {default}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Display model information and recommendations"
    )
    parser.add_argument(
        '--model', 
        help='Show detailed info for specific model'
    )
    parser.add_argument(
        '--use-case',
        help='Show recommended models for use case',
        choices=[
            'interpretability', 'performance', 'speed', 'high_dimensional', 'large_datasets', 'small_datasets',
            'anomaly_detection', 'exploration', 'fast_clustering', 'irregular_shapes', 'probability_estimates', 'noise_handling'
        ]
    )
    
    args = parser.parse_args()
    
    # Load model configuration
    try:
        factory = ModelFactory()
        config = factory.load_model_config()
        
        if not config:
            print("Error: Could not load model configuration")
            return 1
        
        if args.model:
            # Show specific model info
            print_model_info(args.model, config)
        elif args.use_case:
            # Show recommendations for use case
            recommendations = factory.get_model_recommendations(args.use_case)
            print(f"Recommended models for {args.use_case}:")
            for model in recommendations:
                print(f"  • {model}")
                print_model_info(model, config)
        else:
            # Show all models summary
            print_all_models_summary(config)
            print()
            print_recommendations(config)
            
            # Show training notes
            notes = config.get('training_notes', {})
            if notes:
                print("\nTRAINING NOTES")
                print_separator('=')
                for key, value in notes.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())