"""
Model Training - Pipeline Integration

Integrated training module that uses the unified data pipeline to train
Random Forest models for classification tasks using QRadar rule frequencies.
"""

import sys
import os
import json
import logging
import argparse
from typing import Optional, Dict, Tuple
import joblib
import numpy as np

# Add parent directories to path for pipeline imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'system'))
from system import logging_utils

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Import pipeline modules
from pipeline.data_loader import load_data
from pipeline.feature_aggregator import aggregate_to_windows
from pipeline.feature_generator import FeatureGenerator
from shared_utils.qradar_rule_manager import QRadarRuleManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_threat_detector(training_config: Dict, model_save_path: str = "./model/threat_detector.joblib") -> Optional[Tuple[RandomForestClassifier, pd.DataFrame, pd.Series]]:
    """
    Train Random Forest model using the unified data pipeline.
    
    This function orchestrates the complete training pipeline:
    1. Loads training data via data_loader
    2. Aggregates features into 30-minute windows
    3. Generates 2898-dimensional feature vectors
    4. Trains Random Forest with CLAUDE.md specifications
    5. Saves model and generates evaluation metrics
    
    Parameters:
    -----------
    training_config : Dict
        Configuration dictionary for training parameters and paths
    model_save_path : str
        Path where the trained model will be saved
        
    Returns:
    --------
    Tuple[RandomForestClassifier, pd.DataFrame, pd.Series] or None
        The trained model, test features, and test labels if successful, None if training fails.
    """
    
    try:
        logger.info("Starting model training with pipeline integration...")
        
        # Create model directory if it doesn't exist
        model_dir = os.path.dirname(model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created model directory: {model_dir}")
        
        # Step 1: Load training data via pipeline
        logger.info("Loading training data via data_loader...")
        df = load_data('train', training_config)
        if df.empty:
            raise ValueError("No training data loaded")
        logger.info(f"Loaded {len(df)} training records")
        
        # Step 2: Aggregate features into 30-minute windows
        logger.info("Aggregating features into 30-minute windows...")
        df_agg = aggregate_to_windows(df, window_size_minutes=30)
        if df_agg.empty:
            raise ValueError("No aggregated windows generated")
        logger.info(f"Created {len(df_agg)} aggregated windows")
        
        # Step 3: Generate 2898-dimensional feature vectors
        logger.info("Generating 2898-dimensional feature vectors...")
        feature_gen = FeatureGenerator()
        feature_gen.initialize_rules()
        X, y = feature_gen.generate_feature_vectors(df_agg, mode='train')
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Label distribution - Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
        
        # Step 4: Split data with stratification (80/20 split)
        logger.info("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y  # Handle class imbalance
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Step 5: Train Random Forest with optimized specifications
        logger.info("Training Random Forest with optimized specifications...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced_subsample',  # Handle class imbalance
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        logger.info("Model training completed successfully")
        
        # Step 6: Save trained model
        logger.info(f"Saving model to: {model_save_path}")
        joblib.dump(rf_model, model_save_path)
        logger.info("Model saved successfully")
        
        # Log model characteristics
        logger.info(f"Model trained with {rf_model.n_features_in_} features and {len(rf_model.estimators_)} trees")
        
        return rf_model, X_test, y_test
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        return None


def evaluate_and_report(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series, 
                       rule_list: list, model_save_path: str) -> Dict:
    """
    Evaluate the trained model and generate comprehensive reports.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        The trained Random Forest model
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        Test labels
    rule_list : list
        List of rule IDs for feature importance mapping
    model_save_path : str
        Base path for saving evaluation reports
        
    Returns:
    --------
    Dict
        Dictionary containing evaluation metrics and paths to saved reports
    """
    
    try:
        logger.info("Evaluating trained model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log key metrics
        logger.info("=== Model Evaluation Results ===")
        logger.info(f"ROC AUC Score: {roc_auc:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Extract positive class metrics
        if isinstance(report, dict):
            positive_metrics = report.get('1', {})
            if positive_metrics:
                logger.info(f"Positive Class Precision: {positive_metrics.get('precision', 0):.4f}")
                logger.info(f"Positive Class Recall: {positive_metrics.get('recall', 0):.4f}")
                logger.info(f"Positive Class F1-Score: {positive_metrics.get('f1-score', 0):.4f}")
        else:
            logger.warning("Could not extract positive class metrics from report")
        
        # Feature importance analysis
        logger.info("Extracting feature importance...")
        feature_importances = model.feature_importances_
        
        # Create rule importance mapping
        importance_df = pd.DataFrame({
            'rule_id': rule_list,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        # Get top 20 most important features
        top_20_features = importance_df.head(20)
        logger.info("Top 20 most important features:")
        for _, row in top_20_features.iterrows():
            logger.info(f"  Rule {row['rule_id']}: {row['importance']:.4f}")
        
        # Save evaluation reports
        model_dir = os.path.dirname(model_save_path)
        base_name = os.path.splitext(os.path.basename(model_save_path))[0]
        
        # Save top 20 features
        top_features_path = os.path.join(model_dir, f"{base_name}_top_20_features.csv")
        top_20_features.to_csv(top_features_path, index=False)
        logger.info(f"Top 20 features saved to: {top_features_path}")
        
        # Save comprehensive evaluation report
        evaluation_report = {
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'roc_auc_score': roc_auc,
            'model_path': model_save_path,
            'feature_count': len(rule_list),
            'top_20_features': top_20_features.to_dict('records'),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        report_path = os.path.join(model_dir, f"{base_name}_evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        logger.info(f"Evaluation report saved to: {report_path}")
        
        return evaluation_report
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        return {}


def main():
    """
    Main function for CLI execution of model training.
    """
    parser = argparse.ArgumentParser(description='Train Random Forest model using unified pipeline')
    parser.add_argument('--config', required=True, 
                       help='Path to training configuration JSON file')
    parser.add_argument('--model-output', default='./model/threat_detector.joblib',
                       help='Path to save trained model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run model evaluation after training')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            training_config = json.load(f)
        
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Train the model
        result = train_threat_detector(training_config, args.model_output)
        
        if result is None:
            logger.error("Model training failed")
            return 1
            
        model, X_test, y_test = result
        
        # Run evaluation if requested
        if args.evaluate:
            # Get rule list for feature importance mapping
            rule_manager = QRadarRuleManager()
            rule_list = rule_manager.get_rule_list()
            
            evaluation_report = evaluate_and_report(
                model, X_test, y_test, rule_list, args.model_output
            )
            
            if not evaluation_report:
                logger.warning("Model evaluation failed")
        
        logger.info("Model training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit(main())