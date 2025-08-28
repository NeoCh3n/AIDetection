import joblib
import numpy as np
import pandas as pd
import logging
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'system'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

from system import logging_utils
from pipeline.data_loader import load_data
from pipeline.feature_aggregator import aggregate_to_windows
from pipeline.feature_generator import FeatureGenerator


def evaluate_model(model_path, test_data_path=None, rule_list=None):
    """
    Evaluate the trained model using either provided test data or split from training data.

    Parameters:
    model_path: str - Path to the saved model file (.joblib)
    test_data_path: str - Path to test data (optional, if None will use split from training)
    rule_list: list - List of rules for feature generation
    
    Returns:
    dict: Dictionary containing accuracy, precision, recall, F1-score, ROC-AUC score and other metrics
    """
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load the trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as load_error:
            logger.error(f"Failed to load model: {str(load_error)}")
            logger.error("This might be due to scikit-learn version incompatibility.")
            logger.error("Try upgrading scikit-learn: pip install scikit-learn>=1.0.0")
            raise ValueError(f"Model loading failed - version incompatibility: {str(load_error)}")
        
        # Prepare test data
        if test_data_path:
            # Load separate test data
            logger.info(f"Loading test data from: {test_data_path}")
            config = {'data_path': os.path.dirname(test_data_path)}
            df = load_data('test', config)
        else:
            # Use training data and split it
            logger.info("Using training data split for evaluation")
            config = {'data_path': './Training_data'}
            df = load_data('train', config)
        
        if df.empty:
            raise ValueError("No test data available")
            
        logger.info(f"Loaded {len(df)} test records")
        
        # Aggregate to 30-minute windows
        logger.info("Aggregating to 30-minute windows...")
        df_agg = aggregate_to_windows(df)
        logger.info(f"Created {len(df_agg)} aggregated windows")
        
        # Generate feature vectors
        logger.info("Generating feature vectors...")
        feature_gen = FeatureGenerator()
        feature_gen.initialize_rules()
        
        if test_data_path:
            # For separate test data
            X, y = feature_gen.generate_feature_vectors(df_agg, mode='test')
        else:
            # For training data split
            X, y = feature_gen.generate_feature_vectors(df_agg, mode='train')
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X, y = X_test, y_test
        
        logger.info(f"Test feature matrix shape: {X.shape}")
        logger.info(f"Test labels shape: {y.shape}")
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        logger.info("Calculating performance metrics...")
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='binary')
        recall = recall_score(y, y_pred, average='binary')
        f1 = f1_score(y, y_pred, average='binary')
        
        # ROC-AUC only if we have probability predictions
        try:
            roc_auc = roc_auc_score(y, y_pred_proba)
        except:
            roc_auc = None
            logger.warning("Could not calculate ROC-AUC score")
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        
        # Classification Report
        class_report = classification_report(y, y_pred, target_names=['Normal', 'Attack'])
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'test_samples': len(y),
            'positive_predictions': int(np.sum(y_pred)),
            'actual_positives': int(np.sum(y))
        }
        
        # Print results
        logger.info("=== MODEL EVALUATION RESULTS ===")
        logger.info(f"Test Samples: {results['test_samples']}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        if roc_auc:
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Actual Positives: {results['actual_positives']}")
        logger.info(f"Predicted Positives: {results['positive_predictions']}")
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        logger.info(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        logger.info("\nClassification Report:")
        logger.info(class_report)
        
        return results
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise


def evaluate_model_simple(model, X_test, y_test):
    """
    Simple evaluation function for direct model testing.
    
    Parameters:
    model: Trained machine learning model
    X_test: Test features
    y_test: True labels for the test set
    
    Returns:
    dict: Dictionary containing performance metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        return results
        
    except Exception as e:
        logging.error(f"Simple model evaluation failed: {str(e)}")
        raise
