"""
Simple usage
------------
- Evaluate with default training split as test:
    python model_training/model_evaluation.py --model ./model/threat_detector.joblib

- Evaluate with explicit test data (directory containing normal/ and attack/):
    python model_training/model_evaluation.py --model ./model/threat_detector.joblib \
        --test-data ./Training_data

 - Evaluate a single class by pointing directly to ./Training_data/normal or ./Training_data/attack.
   Only that directory is used (no auto-include of sibling).

Options:
- --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]

Notes:
- Uses the unified pipeline: data_loader -> feature_aggregator -> feature_generator.
- Writes a summary line to running_log/<today>.log via logging_utils.
"""

import joblib
import numpy as np
import pandas as pd
import logging
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

from system import logging_utils
from pipeline.data_loader import load_data
from pipeline.feature_aggregator import aggregate_to_windows
from pipeline.feature_generator import FeatureGenerator


class EvaluationResults:
    """Data class to hold evaluation results"""
    def __init__(self, accuracy, precision, recall, f1_score, roc_auc, 
                 confusion_matrix, classification_report, test_samples, 
                 positive_predictions, actual_positives):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.roc_auc = roc_auc
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
        self.test_samples = test_samples
        self.positive_predictions = positive_predictions
        self.actual_positives = actual_positives


class MetricsCalculator:
    """Class responsible for calculating model evaluation metrics"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None, pred_counts=None, actual_counts=None):
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            pred_counts: Dictionary with prediction counts {'attack': int, 'normal': int}
            actual_counts: Dictionary with actual counts {'attack': int, 'normal': int}
            
        Returns:
            EvaluationResults object containing all metrics
        """
        self.logger.info("Calculating performance metrics...")
        
        # Store counts for use in calculations
        self.pred_counts = pred_counts
        self.actual_counts = actual_counts
        
        # Log count information if provided
        if pred_counts and actual_counts:
            self.logger.info(f"Using provided counts - Predictions: {pred_counts}, Actual: {actual_counts}")
        
        # Validate inputs
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred cannot be None")
        
        # Check if inputs are empty
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("y_true and y_pred cannot be empty")
        
        # Debug information
        self._log_data_info(y_true, y_pred)
        
        # Ensure proper data types and compatibility
        try:
            y_true, y_pred = self._prepare_data(y_true, y_pred)
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise ValueError(f"Cannot prepare data for metric calculation: {e}")
        
        # Calculate basic metrics
        accuracy = self._calculate_accuracy(y_true, y_pred)
        precision, recall, f1 = self._calculate_precision_recall_f1(y_true, y_pred)
        
        # Calculate ROC-AUC
        roc_auc = self._calculate_roc_auc(y_true, y_pred_proba)
        
        # Generate confusion matrix and classification report
        cm = self._calculate_confusion_matrix(y_true, y_pred)
        class_report = self._generate_classification_report(y_true, y_pred)
        
        return EvaluationResults(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm.tolist(),
            classification_report=class_report,
            test_samples=len(y_true),
            positive_predictions=int(np.sum(y_pred)),
            actual_positives=int(np.sum(y_true))
        )
    
    def _log_data_info(self, y_true, y_pred):
        """Log debugging information about the data"""
        self.logger.info(f"Data shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        self.logger.info(f"Unique values - y_true: {np.unique(y_true)}, y_pred: {np.unique(y_pred)}")
        
        # Check if model is predicting only one class
        if len(np.unique(y_pred)) == 1:
            self.logger.warning(f"WARNING: MODEL IS PREDICTING ONLY ONE CLASS: {np.unique(y_pred)[0]}")
            self.logger.warning("This suggests the model may be poorly trained or there's a data issue")
            
            # Additional diagnostics
            if np.unique(y_pred)[0] == 0:
                self.logger.warning("DIAGNOSIS: Model only predicts 'Normal' (0) - possible causes:")
                self.logger.warning("   1. Model wasn't trained on enough attack samples")
                self.logger.warning("   2. Feature preprocessing differs between training/testing")
                self.logger.warning("   3. Model learned to always predict majority class")
                self.logger.warning("   4. Feature values are out of expected range")
            else:
                self.logger.warning("DIAGNOSIS: Model only predicts 'Attack' (1) - possible causes:")
                self.logger.warning("   1. Model is overly sensitive/poorly calibrated")
                self.logger.warning("   2. Feature preprocessing issue")
                
            # Check if we actually have both classes in true labels
            unique_true = np.unique(y_true)
            if len(unique_true) == 1:
                self.logger.warning(f"WARNING: TEST DATA ONLY HAS ONE CLASS: {unique_true[0]}")
                self.logger.warning("Cannot properly evaluate model with single-class test data!")
            else:
                self.logger.info(f"✓ Test data has both classes: {unique_true}")
                
                # Show class distribution
                attack_count = np.sum(y_true == 1)
                normal_count = np.sum(y_true == 0)
                total = len(y_true)
                attack_pct = (attack_count / total) * 100
                normal_pct = (normal_count / total) * 100
                
                self.logger.info(f"Test data distribution:")
                self.logger.info(f"   Normal: {normal_count} samples ({normal_pct:.1f}%)")
                self.logger.info(f"   Attack: {attack_count} samples ({attack_pct:.1f}%)")
    
    def _prepare_data(self, y_true, y_pred):
        """Prepare data for metric calculation"""
        # Convert to numpy arrays first
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Check shapes
        if y_true.shape != y_pred.shape:
            self.logger.error(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
            raise ValueError(f"Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}")
        
        # Handle different data types
        if y_true.dtype != y_pred.dtype:
            self.logger.info(f"Converting data types: y_true {y_true.dtype} -> y_pred {y_pred.dtype}")
            
            # Try to find common type
            # If either array is floating, convert to float then to int
            # Use dtype.kind == 'f' to avoid static checker issues with np.floating
            if (getattr(y_true, 'dtype', None) is not None and y_true.dtype.kind == 'f') or \
               (getattr(y_pred, 'dtype', None) is not None and y_pred.dtype.kind == 'f'):
                y_true = np.array(y_true, dtype=float)
                y_pred = np.array(y_pred, dtype=float)
                
                # Check for NaN values using pandas for broader dtype support
                import pandas as pd  # local import to avoid circulars in some environments
                if pd.isna(y_true).any() or pd.isna(y_pred).any():
                    self.logger.error("NaN values detected in labels or predictions")
                    raise ValueError("NaN values found in data")
                
                # Convert to int
                y_true = np.array(y_true, dtype=int)
                y_pred = np.array(y_pred, dtype=int)
            else:
                # Both are integers or can be converted to int
                try:
                    y_true = np.array(y_true, dtype=int)
                    y_pred = np.array(y_pred, dtype=int)
                except (ValueError, OverflowError) as e:
                    self.logger.error(f"Cannot convert to int: {e}")
                    raise ValueError(f"Data type conversion failed: {e}")
        else:
            # Same data type, but ensure it's int
            try:
                y_true = np.array(y_true, dtype=int)
                y_pred = np.array(y_pred, dtype=int)
            except (ValueError, OverflowError) as e:
                self.logger.error(f"Cannot convert to int: {e}")
                raise ValueError(f"Data type conversion failed: {e}")
        
        # Validate value ranges
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        # Check for valid binary classification values
        valid_values = {0, 1}
        if not set(unique_true).issubset(valid_values):
            self.logger.warning(f"y_true contains non-binary values: {unique_true}")
        if not set(unique_pred).issubset(valid_values):
            self.logger.warning(f"y_pred contains non-binary values: {unique_pred}")
        
        # Only log final distribution
        self.logger.info(f"Label distribution - True: {np.bincount(y_true)}, Predicted: {np.bincount(y_pred)}")
        
        return y_true, y_pred
    
    def _calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy score"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            return accuracy
        except Exception as e:
            self.logger.error(f"Error calculating accuracy: {e}")
            return 0.0
    
    def _calculate_precision_recall_f1(self, y_true, y_pred):
        """Calculate precision, recall, and F1-score"""
        try:
            unique_y = np.unique(y_true)
            
            if len(unique_y) < 2:
                self.logger.warning("Only one class in true labels - using 'macro' average")
                precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            else:
                precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            return precision, recall, f1
            
        except Exception as e:
            self.logger.error(f"Error calculating precision/recall/f1: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_roc_auc(self, y_true, y_pred_proba):
        """Calculate ROC-AUC score"""
        try:
            unique_y = np.unique(y_true)
            if len(unique_y) >= 2 and y_pred_proba is not None:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                return roc_auc
            else:
                if y_pred_proba is None:
                    self.logger.warning("No probability predictions available for ROC-AUC")
                else:
                    self.logger.warning("Cannot calculate ROC-AUC: only one class in true labels")
                return None
        except Exception as e:
            self.logger.warning(f"Could not calculate ROC-AUC score: {e}")
            return None
    
    def _calculate_confusion_matrix(self, y_true, y_pred):
        """Calculate confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            return cm
        except Exception as e:
            self.logger.error(f"Error calculating confusion matrix: {e}")
            return np.array([[0, 0], [0, 0]])
    
    def _generate_classification_report(self, y_true, y_pred):
        """Generate classification report"""
        try:
            class_report = classification_report(y_true, y_pred, target_names=['Normal', 'Attack'], zero_division=0)
            return class_report
        except Exception as e:
            self.logger.error(f"Error generating classification report: {e}")
            return "Classification report could not be generated"


class ModelEvaluator:
    """Class responsible for model evaluation workflow"""
    
    def __init__(self, logger=None):
        self.logger = logger or self._setup_logger()
        self.metrics_calculator = MetricsCalculator(self.logger)
    
    def _setup_logger(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def evaluate_model(self, model_path, test_data_path=None, rule_list=None):
        """
        Main evaluation method
        
        Args:
            model_path: Path to the saved model file (.joblib)
            test_data_path: Path to test data (optional)
            rule_list: List of rules for feature generation (optional)
            
        Returns:
            EvaluationResults object containing all metrics
        """
        try:
            self.logger.info(f"Starting model evaluation...")
            
            # Load model
            model = self._load_model(model_path)
            
            # Prepare data
            X, y = self._prepare_test_data(test_data_path)
            
            # Make predictions
            y_pred, y_pred_proba = self._make_predictions(model, X)
            
            # Log prediction summary and get counts
            pred_counts, actual_counts = self._log_prediction_summary(y, y_pred)
            
            # Calculate metrics with counts
            results = self.metrics_calculator.calculate_metrics(y, y_pred, y_pred_proba, pred_counts, actual_counts)
            
            # Display results
            self._display_results(results)
            
            self.logger.info("Model evaluation completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def _load_model(self, model_path):
        """Load the trained model"""
        self.logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model = joblib.load(model_path)
            self.logger.info("Model loaded successfully")
            return model
        except Exception as load_error:
            self.logger.error(f"Failed to load model: {str(load_error)}")
            self.logger.error("This might be due to scikit-learn version incompatibility with your environment.")
            self.logger.error("Ensure the model is loaded with the same scikit-learn version used for training (e.g., 0.24.2 on Python 3.6.8).")
            raise ValueError(f"Model loading failed - version incompatibility: {str(load_error)}")
    
    def _prepare_test_data(self, test_data_path=None):
        """Prepare test data for evaluation.

        If test_data_path is provided, expects a directory containing
        'normal' and 'attack' subdirectories (like Training_data).
        Otherwise, uses the default './Training_data' path.
        """
        # Load data
        if test_data_path:
            self.logger.info(f"Loading test data from: {test_data_path}")
            base = os.path.abspath(test_data_path)

            # If a file path is provided, use its directory
            if os.path.isfile(base):
                base = os.path.dirname(base)

            # Determine normal and attack directories
            normal_dir = None
            attack_dir = None

            # Case 1: base contains 'normal' and/or 'attack' subdirs
            cand_normal = os.path.join(base, 'normal')
            cand_attack = os.path.join(base, 'attack')
            if os.path.isdir(cand_normal):
                normal_dir = cand_normal
            if os.path.isdir(cand_attack):
                attack_dir = cand_attack

            # Case 2: base itself is a 'normal' or 'attack' directory
            base_name = os.path.basename(base.rstrip(os.sep))
            parent = os.path.dirname(base)
            if normal_dir is None and base_name.lower() == 'normal':
                # Honor explicit single-class directory: only load this directory
                normal_dir = base
                attack_dir = None
            if attack_dir is None and base_name.lower() == 'attack':
                # Honor explicit single-class directory: only load this directory
                attack_dir = base
                normal_dir = None

            # No sibling inference: if one side missing, treat as single-class dataset

            # Build config; missing dir paths will be treated as empty by the loader
            config = {
                'training_data_path': normal_dir if normal_dir is not None else "__MISSING__",
                'attack_data_path': attack_dir if attack_dir is not None else "__MISSING__"
            }
            df = load_data('train', config)
        else:
            self.logger.info("Using training data split for evaluation")
            config = {
                'training_data_path': './Training_data/normal',
                'attack_data_path': './Training_data/attack'
            }
            df = load_data('train', config)
        
        if df.empty:
            raise ValueError("No test data available")
        
        self.logger.info(f"Loaded {len(df)} test records")
        
        # Aggregate to 30-minute windows
        self.logger.info("Aggregating to 30-minute windows...")
        df_agg = aggregate_to_windows(df)
        self.logger.info(f"Created {len(df_agg)} aggregated windows")
        
        # Generate feature vectors
        self.logger.info("Generating feature vectors...")
        feature_gen = FeatureGenerator()
        feature_gen.initialize_rules()
        
        if test_data_path:
            # Use training mode to ensure labels are generated for provided test set
            X, y = feature_gen.generate_feature_vectors(df_agg, mode='train')
            if y is None:
                raise ValueError("Labels could not be generated from provided test data. Ensure data includes source labels.")
        else:
            X, y = feature_gen.generate_feature_vectors(df_agg, mode='train')
            self.logger.info(f"Before split - Total samples: {len(X)}")
            if y is not None:
                self.logger.info(f"Labels shape: {y.shape}")
                self.logger.info(f"Before split - Label distribution: {np.bincount(y) if len(np.unique(y)) <= 2 else 'Multiple classes'}")
            else:
                self.logger.warning("No labels produced; cannot perform stratified split.")
            
            if len(X) < 10:
                self.logger.warning(f"WARNING: Very few samples ({len(X)}) for train/test split!")
            
            # Split the data
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                self.logger.info(f"After split - Test samples: {len(X_test)}, Train samples: {len(X_train)}")
                self.logger.info(f"Test set label distribution: {np.bincount(y_test) if len(np.unique(y_test)) <= 2 else 'Multiple classes'}")
                X, y = X_test, y_test
            else:
                self.logger.warning("Skipping train/test split due to missing labels.")
        
        self.logger.info(f"Test feature matrix shape: {X.shape}")
        if y is not None:
            self.logger.info(f"Test labels shape: {y.shape}")
        
        return X, y
    
    def _make_predictions(self, model, X):
        """Make predictions using the model"""
        self.logger.info("Making predictions...")
        
        # Log feature statistics for debugging
        self.logger.info(f"Feature matrix statistics:")
        self.logger.info(f"   Shape: {X.shape}")
        self.logger.info(f"   Min value: {np.min(X):.4f}")
        self.logger.info(f"   Max value: {np.max(X):.4f}")
        self.logger.info(f"   Mean: {np.mean(X):.4f}")
        self.logger.info(f"   Std: {np.std(X):.4f}")
        
        # Check for unusual feature values
        import pandas as pd  # local import to satisfy static analyzers
        if pd.isna(X).any():
            self.logger.warning("WARNING: NaN values detected in features!")
        # Detect infinities without relying on np.isinf to satisfy static checkers
        try:
            if np.any((X == np.inf) | (X == -np.inf)):
                self.logger.warning("WARNING: Infinite values detected in features!")
        except Exception:
            # Fallback: very large magnitude indicates potential infinities
            if np.any(np.abs(X) > 1e308):
                self.logger.warning("WARNING: Extremely large values detected; possible infinities!")
        
        # Check if all features are zeros
        if np.all(X == 0):
            self.logger.warning("WARNING: All features are zero! This will cause prediction issues.")
        elif np.sum(X == 0) > (0.9 * X.size):
            zero_pct = (np.sum(X == 0) / X.size) * 100
            self.logger.warning(f"WARNING: {zero_pct:.1f}% of features are zero - very sparse features!")
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Log prediction probabilities for debugging single-class prediction
        if y_pred_proba is not None:
            self.logger.info(f"Prediction probabilities:")
            self.logger.info(f"   Min probability: {np.min(y_pred_proba):.4f}")
            self.logger.info(f"   Max probability: {np.max(y_pred_proba):.4f}")
            self.logger.info(f"   Mean probability: {np.mean(y_pred_proba):.4f}")
            self.logger.info(f"   Std probability: {np.std(y_pred_proba):.4f}")
            
            # Check if probabilities are all very low or very high
            if np.max(y_pred_proba) < 0.1:
                self.logger.warning("WARNING: All prediction probabilities < 0.1 - model very biased toward Normal")
            elif np.min(y_pred_proba) > 0.9:
                self.logger.warning("WARNING: All prediction probabilities > 0.9 - model very biased toward Attack")
        
        return y_pred, y_pred_proba
    
    def _log_prediction_summary(self, y_true, y_pred):
        """Log prediction results vs actual samples and return counts"""
        # Count predictions
        pred_attack = np.sum(y_pred == 1)
        pred_normal = np.sum(y_pred == 0)
        
        # Count actual samples
        actual_attack = np.sum(y_true == 1)
        actual_normal = np.sum(y_true == 0)
        
        self.logger.info("=== PREDICTION vs ACTUAL SUMMARY ===")
        self.logger.info(f"Predict:   Attack: {pred_attack:4d}   Normal: {pred_normal:4d}")
        self.logger.info(f"Sample:    Attack: {actual_attack:4d}   Normal: {actual_normal:4d}")
        self.logger.info("=" * 40)
        
        # Return counts as dictionaries
        pred_counts = {'attack': int(pred_attack), 'normal': int(pred_normal)}
        actual_counts = {'attack': int(actual_attack), 'normal': int(actual_normal)}
        
        return pred_counts, actual_counts
    
    def _display_results(self, results):
        """Display evaluation results"""
        self.logger.info("=== MODEL EVALUATION RESULTS ===")
        self.logger.info(f"Test Samples: {results.test_samples}")
        self.logger.info(f"Accuracy: {results.accuracy:.4f}")
        self.logger.info(f"Precision: {results.precision:.4f}")
        self.logger.info(f"Recall: {results.recall:.4f}")
        self.logger.info(f"F1-Score: {results.f1_score:.4f}")
        if results.roc_auc:
            self.logger.info(f"ROC-AUC: {results.roc_auc:.4f}")
        self.logger.info(f"Actual Positives: {results.actual_positives}")
        self.logger.info(f"Predicted Positives: {results.positive_predictions}")
        
        self.logger.info("\nConfusion Matrix:")
        try:
            cm = np.array(results.confusion_matrix)
            self.logger.info(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
            self.logger.info(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        except Exception as e:
            self.logger.error(f"Error displaying confusion matrix: {e}")
            self.logger.info(f"Confusion matrix: {results.confusion_matrix}")
        
        self.logger.info("\nClassification Report:")
        try:
            self.logger.info(str(results.classification_report))
        except Exception:
            # Defensive: ensure logging does not fail on unexpected types
            self.logger.info(repr(results.classification_report))

        # Persist a compact summary into running_log via centralized logger
        try:
            summary_payload = {
                'test_samples': results.test_samples,
                'accuracy': results.accuracy,
                'precision': results.precision,
                'recall': results.recall,
                'f1_score': results.f1_score,
                'roc_auc': results.roc_auc,
                'confusion_matrix': results.confusion_matrix,
                'positive_predictions': results.positive_predictions,
                'actual_positives': results.actual_positives,
            }
            # Use standardized pipeline event logging
            try:
                from system import logging_utils as _lu
            except Exception:
                _lu = logging_utils  # fallback to top-level import
            _lu.log_pipeline_event('EVALUATION_RESULTS', 'Model evaluation summary', metadata=summary_payload)
        except Exception as _e:
            self.logger.warning(f"Could not write evaluation summary to running_log: {_e}")


# Legacy function for backward compatibility
def evaluate_model(model_path, test_data_path=None, rule_list=None):
    """
    Legacy function for backward compatibility
    
    Parameters:
    model_path: str - Path to the saved model file (.joblib)
    test_data_path: str - Path to test data (optional, if None will use split from training)
    rule_list: list - List of rules for feature generation
    
    Returns:
    dict: Dictionary containing accuracy, precision, recall, F1-score, ROC-AUC score and other metrics
    """
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model_path, test_data_path, rule_list)
    
    # Convert to dictionary for backward compatibility
    return {
        'accuracy': results.accuracy,
        'precision': results.precision,
        'recall': results.recall,
        'f1_score': results.f1_score,
        'roc_auc': results.roc_auc,
        'confusion_matrix': results.confusion_matrix,
        'classification_report': results.classification_report,
        'test_samples': results.test_samples,
        'positive_predictions': results.positive_predictions,
        'actual_positives': results.actual_positives
    }


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
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate counts
        pred_attack = int(np.sum(y_pred == 1))
        pred_normal = int(np.sum(y_pred == 0))
        actual_attack = int(np.sum(y_test == 1))
        actual_normal = int(np.sum(y_test == 0))
        
        pred_counts = {'attack': pred_attack, 'normal': pred_normal}
        actual_counts = {'attack': actual_attack, 'normal': actual_normal}
        
        # Use MetricsCalculator with counts
        calculator = MetricsCalculator()
        results = calculator.calculate_metrics(y_test, y_pred, y_pred_proba, pred_counts, actual_counts)
        
        return {
            'accuracy': results.accuracy,
            'precision': results.precision,
            'recall': results.recall,
            'f1_score': results.f1_score,
            'roc_auc': results.roc_auc
        }
        
    except Exception as e:
        logging.error(f"Simple model evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Command-line entrypoint to evaluate a trained model.

    Usage examples:
      - Evaluate with default training split for test:
          python model_training/model_evaluation.py --model ./model/threat_detector.joblib

      - Evaluate with explicit test data directory (must contain normal/ and attack/):
          python model_training/model_evaluation.py --model ./model/threat_detector.joblib \
              --test-data ./Training_data
    """
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained threat detection model")
    parser.add_argument("--model", required=True, help="Path to the trained .joblib model file")
    parser.add_argument("--test-data", dest="test_data", default=None,
                        help="Optional path to test data directory (with normal/ and attack/)")
    parser.add_argument("--log-level", dest="log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")

    args = parser.parse_args()

    # Configure basic logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    evaluator = ModelEvaluator()
    try:
        results = evaluator.evaluate_model(model_path=args.model, test_data_path=args.test_data)

        # Print concise summary to stdout
        print("\n=== MODEL EVALUATION SUMMARY ===")
        print(f"Accuracy:   {results.accuracy:.4f}")
        print(f"Precision:  {results.precision:.4f}")
        print(f"Recall:     {results.recall:.4f}")
        print(f"F1-Score:   {results.f1_score:.4f}")
        if results.roc_auc is not None:
            print(f"ROC-AUC:    {results.roc_auc:.4f}")
        print(f"Test Size:  {results.test_samples}")
        print(f"TP/FP/FN/TN from confusion matrix: {results.confusion_matrix}")
        print("\nClassification Report:")
        try:
            # classification_report is expected to be a string; force str for safety
            print(str(results.classification_report))
        except Exception:
            import json as _json
            try:
                print(_json.dumps(results.classification_report, indent=2))
            except Exception:
                print(repr(results.classification_report))
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        sys.exit(1)
