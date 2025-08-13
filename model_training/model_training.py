"""
Module for training and tuning the Random Forest model for ransomware detection.

This module handles the training process for the supervised machine learning model
designed to detect ransomware activity using QRadar rule trigger frequencies as features.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import logging
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_ransomware_detector(feature_data_path: str, model_save_path: str) -> Optional[RandomForestClassifier]:
    """
    Train, tune, and save the Random Forest classifier for ransomware detection.
    
    This function loads the labeled dataset, splits it into training and testing sets
    with stratification to handle class imbalance, trains a Random Forest classifier
    with optimized hyperparameters, and saves the trained model.
    
    Parameters:
    -----------
    feature_data_path : str
        Path to the final labeled dataset CSV file containing feature vectors and labels.
        Expected columns include ~1500 rule ID columns plus 'is_attack' label column.
    model_save_path : str
        Path where the trained model will be saved (e.g., './model/ransomware_detector.joblib').
        
    Returns:
    --------
    RandomForestClassifier or None
        The trained Random Forest model if successful, None if training fails.
        
    Raises:
    -------
    FileNotFoundError
        If the feature_data_path file does not exist.
    ValueError
        If the data format is invalid or missing required columns.
    Exception
        For any other training-related errors.
    """
    
    try:
        # Validate input paths
        if not os.path.exists(feature_data_path):
            raise FileNotFoundError(f"Feature data file not found: {feature_data_path}")
            
        # Create model directory if it doesn't exist
        model_dir = os.path.dirname(model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created model directory: {model_dir}")
        
        # Load the labeled DataFrame
        logger.info(f"Loading feature data from: {feature_data_path}")
        df = pd.read_csv(feature_data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Validate required columns
        if 'is_attack' not in df.columns:
            raise ValueError("Dataset missing required 'is_attack' label column")
            
        # Separate features (X) and labels (y)
        X = df.drop('is_attack', axis=1)
        y = df['is_attack']
        
        # Log class distribution for debugging
        class_counts = y.value_counts()
        logger.info(f"Class distribution - Normal: {class_counts.get(0, 0)}, Attack: {class_counts.get(1, 0)}")
        
        # Split the data into training and testing sets
        # Using stratify=y to maintain class proportion in train/test splits
        logger.info("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y  # Crucial for handling class imbalance
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Testing set shape: {X_test.shape}")
        
        # Configure Random Forest classifier with optimized hyperparameters
        logger.info("Initializing Random Forest classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=200,                    # Good starting point for robust model
            class_weight='balanced_subsample',   # Handle class imbalance
            max_features='sqrt',                 # Default often performs well
            random_state=42,                     # Reproducibility
            n_jobs=-1,                          # Use all available cores
            verbose=1                           # Show training progress
        )
        
        # Train the model
        logger.info("Training Random Forest model...")
        rf_model.fit(X_train, y_train)
        
        # Log training completion
        logger.info("Model training completed successfully")
        
        # Save the trained model
        logger.info(f"Saving trained model to: {model_save_path}")
        joblib.dump(rf_model, model_save_path)
        logger.info("Model saved successfully")
        
        # Log model characteristics
        logger.info(f"Model type: {type(rf_model).__name__}")
        logger.info(f"Number of features: {rf_model.n_features_in_}")
        logger.info(f"Number of trees: {len(rf_model.estimators_)}")
        
        return rf_model
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        logger.exception("Full traceback:")
        return None


def validate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Validate the trained model using test data.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        The trained Random Forest model
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        Test labels
        
    Returns:
    --------
    dict
        Dictionary containing validation metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log results
        logger.info("Model validation completed")
        accuracy = report.get('accuracy', 0) if isinstance(report, dict) else 0
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC Score: {roc_auc:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Log class-specific metrics for attack detection
        attack_metrics = report.get('1', {}) if isinstance(report, dict) else {}
        if attack_metrics:
            logger.info(f"Attack class precision: {attack_metrics.get('precision', 0):.4f}")
            logger.info(f"Attack class recall: {attack_metrics.get('recall', 0):.4f}")
            logger.info(f"Attack class f1-score: {attack_metrics.get('f1-score', 0):.4f}")
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'roc_auc_score': roc_auc
        }
        
    except Exception as e:
        logger.error(f"Error during model validation: {str(e)}")
        return {}


def main():
    """
    Main function for standalone execution of model training.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ransomware detection model')
    parser.add_argument('--feature-data', required=True, 
                       help='Path to feature data CSV file')
    parser.add_argument('--model-path', required=True,
                       help='Path to save trained model')
    parser.add_argument('--validate', action='store_true',
                       help='Perform model validation after training')
    
    args = parser.parse_args()
    
    # Train the model
    model = train_ransomware_detector(args.feature_data, args.model_path)
    
    if model is None:
        logger.error("Model training failed")
        return 1
    
    # Optional validation
    if args.validate:
        logger.info("Loading data for validation...")
        df = pd.read_csv(args.feature_data)
        X = df.drop('is_attack', axis=1)
        y = df['is_attack']
        
        # Split data again to get test set
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Validate model
        metrics = validate_model(model, X_test, y_test)
        
        if metrics:
            logger.info("Model validation metrics saved successfully")
    
    logger.info("Model training process completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())