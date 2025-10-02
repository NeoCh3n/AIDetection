#!/usr/bin/env python3
"""
Test Model Evaluation Script

This script tests the saved model using the evaluate_model function.
It loads the saved model from the model directory and evaluates its performance.
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to the import path so package imports resolve consistently.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_training.model_evaluation import evaluate_model
from shared_utils.qradar_rule_manager import QRadarRuleManager

def test_model_evaluation():
    """Test the model evaluation with the saved model"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model evaluation test...")
    
    try:
        # Define paths
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'test_model_skl0242_20250828_142238.joblib')
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        logger.info(f"Found model at: {model_path}")
        
        # Get rule list (you can modify this based on your needs)
        try:
            rule_manager = QRadarRuleManager(mode='file')  # Use file mode for now
            rule_list = rule_manager.get_rule_list()
            logger.info(f"Loaded {len(rule_list)} rules")
        except Exception as e:
            logger.warning(f"Could not load rules: {e}")
            rule_list = None
        
        # Evaluate model
        logger.info("Starting model evaluation...")
        results = evaluate_model(
            model_path=model_path,
            test_data_path=None,  # Use split from training data
            rule_list=rule_list
        )
        
        # Display results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Test Samples: {results['test_samples']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        if results['roc_auc']:
            print(f"ROC-AUC: {results['roc_auc']:.4f}")
        print(f"Actual Positives: {results['actual_positives']}")
        print(f"Predicted Positives: {results['positive_predictions']}")
        
        print("\nConfusion Matrix:")
        cm = results['confusion_matrix']
        print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
        
        print("\nClassification Report:")
        print(results['classification_report'])
        print("="*50)
        
        logger.info("Model evaluation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    print("Model Evaluation Test Script")
    print("=" * 30)
    
    success = test_model_evaluation()
    
    if success:
        print("\n✓ Model evaluation completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Model evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
