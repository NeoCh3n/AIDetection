#!/usr/bin/env python3
"""
Object-Oriented Threat Detection Pipeline
Modern OOP implementation with model switching capability.

This module provides a flexible, extensible pipeline architecture that supports
multiple machine learning models (RandomForest, SVM, etc.) with consistent
interfaces for training and detection workflows.

Architecture Components:
- DataHandler: Unified data loading from CSV/MongoDB
- FeatureManipulator: Feature engineering and preprocessing
- ModelBase: Abstract base class for ML models
- TrainingPipeline: Orchestrates model training workflow
- DetectionPipeline: Handles real-time threat detection
- ModelFactory: Creates models based on configuration

Python 3.6.8 Compatible
"""

import sys
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import existing modules
from shared_utils.time_utils import get_window_id, parse_qradar_timestamp
from shared_utils.qradar_rule_manager import QRadarRuleManager
from mongodb.mongodb_connection import get_mongodb_manager
from system.logging_utils import setup_global_daily_file_logging
from system.config import get_config

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.ensemble import GradientBoostingClassifier
    GRADIENT_BOOSTING_AVAILABLE = True
except ImportError:
    GRADIENT_BOOSTING_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class DataHandler:
    """
    Unified data loading and preprocessing handler.
    
    Abstracts data sources (CSV files for training, MongoDB for detection)
    and provides consistent output format for downstream processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataHandler with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataHandler")
    
    def load_data(self, mode: str) -> pd.DataFrame:
        """
        Load data based on mode with unified output format.
        
        Args:
            mode: Either 'train' or 'detect'
            
        Returns:
            Standardized DataFrame with columns: hostname, rule_id, timestamp, count, source_label
        """
        if mode not in ['train', 'detect']:
            raise ValueError("Mode must be either 'train' or 'detect'")
        
        self.logger.info(f"Loading data in {mode} mode...")
        
        try:
            if mode == 'train':
                df = self._load_training_data()
            else:
                df = self._load_detection_data()
            
            # Validate and preprocess
            df = self._validate_and_preprocess(df)
            
            self.logger.info(f"Successfully loaded {len(df)} records in {mode} mode")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data in {mode} mode: {e}")
            raise
    
    def _load_training_data(self) -> pd.DataFrame:
        """Load training data from CSV files."""
        from pipeline.data_loader import _load_training_data
        return _load_training_data(self.config)
    
    def _load_detection_data(self) -> pd.DataFrame:
        """Load detection data from MongoDB."""
        from pipeline.data_loader import _load_detection_data
        return _load_detection_data(self.config)
    
    def _validate_and_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and preprocess loaded data."""
        required_columns = ['hostname', 'rule_id', 'timestamp', 'count']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure proper data types
        df['rule_id'] = pd.to_numeric(df['rule_id'], errors='coerce')
        df['count'] = pd.to_numeric(df['count'], errors='coerce') 
        df['hostname'] = df['hostname'].astype(str)
        
        # Parse timestamps
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = df['timestamp'].apply(parse_qradar_timestamp)
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Drop invalid rows
        df = df.dropna(subset=['rule_id', 'count', 'hostname'])
        
        return df


class FeatureManipulator:
    """
    Feature engineering and preprocessing handler.
    
    Handles time-window aggregation, feature generation, and preprocessing
    with consistent interface for both training and detection modes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureManipulator with configuration.
        
        Args:
            config: Configuration dictionary containing feature engineering settings
        """
        self.config = config
        self.window_size_minutes = config.get('feature_engineering', {}).get('window_size_minutes', 30)
        self.log_transform = config.get('feature_engineering', {}).get('log_transform', True)
        self.rule_manager = QRadarRuleManager(config.get('rule_manager', {}))
        self.logger = logging.getLogger(f"{__name__}.FeatureManipulator")
    
    def process_features(self, df: pd.DataFrame, mode: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process raw data into model-ready features.
        
        Args:
            df: Raw data DataFrame
            mode: 'train' or 'detect'
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is labels (None for detect mode)
        """
        self.logger.info("Processing features...")
        
        # Aggregate to time windows
        df_agg = self._aggregate_to_windows(df)
        
        # Generate feature vectors
        X, y = self._generate_feature_vectors(df_agg, mode)
        
        self.logger.info(f"Generated feature matrix: {X.shape}")
        
        return X, y
    
    def _aggregate_to_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data to time windows."""
        from pipeline.feature_aggregator import aggregate_to_windows
        return aggregate_to_windows(df, window_size_minutes=self.window_size_minutes)
    
    def _generate_feature_vectors(self, df_agg: pd.DataFrame, mode: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate feature vectors from aggregated data."""
        from pipeline.feature_generator import FeatureGenerator
        
        feature_gen = FeatureGenerator(self.rule_manager)
        return feature_gen.generate_feature_vectors(df_agg, mode)


class ModelBase(ABC):
    """
    Abstract base class for machine learning models.
    
    Defines the interface that all model implementations must follow
    to ensure consistent behavior across different algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration.
        
        Args:
            config: Model-specific configuration parameters
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create and return the model instance."""
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Get model hyperparameters from config."""
        pass
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'ModelBase':
        """
        Train the model on provided data.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Training model...")
        
        # Create model if not exists
        if self.model is None:
            self.model = self.create_model()
        
        # Apply scaling if needed
        if self.needs_scaling():
            if self.scaler is None:
                self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Train model
        self.model.fit(X_scaled, y)
        
        self.logger.info("Model training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on provided data."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(f"{self.__class__.__name__} does not support probability predictions")
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    @abstractmethod
    def needs_scaling(self) -> bool:
        """Return True if this model requires feature scaling."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'model_type': self.__class__.__name__
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ModelBase':
        """Load model from file."""
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.scaler = model_data.get('scaler')
        
        return instance


class RandomForestModel(ModelBase):
    """Random Forest model implementation."""
    
    def create_model(self) -> RandomForestClassifier:
        """Create RandomForest model instance."""
        params = self.get_model_params()
        return RandomForestClassifier(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get RandomForest hyperparameters."""
        rf_config = self.config.get('model', {}).get('random_forest', {})
        return {
            'n_estimators': rf_config.get('n_estimators', 200),
            'class_weight': rf_config.get('class_weight', 'balanced_subsample'),
            'max_features': rf_config.get('max_features', 'sqrt'),
            'random_state': rf_config.get('random_state', 42),
            'n_jobs': rf_config.get('n_jobs', -1),
            'max_depth': rf_config.get('max_depth'),
            'min_samples_split': rf_config.get('min_samples_split', 2),
            'min_samples_leaf': rf_config.get('min_samples_leaf', 1)
        }
    
    def needs_scaling(self) -> bool:
        """RandomForest doesn't require feature scaling."""
        return False
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        return self.model.feature_importances_


class SVMModel(ModelBase):
    """Support Vector Machine model implementation."""
    
    def create_model(self) -> SVC:
        """Create SVM model instance."""
        params = self.get_model_params()
        return SVC(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get SVM hyperparameters."""
        svm_config = self.config.get('model', {}).get('svm', {})
        return {
            'C': svm_config.get('C', 1.0),
            'kernel': svm_config.get('kernel', 'rbf'),
            'gamma': svm_config.get('gamma', 'scale'),
            'probability': svm_config.get('probability', True),
            'random_state': svm_config.get('random_state', 42),
            'class_weight': svm_config.get('class_weight', 'balanced')
        }
    
    def needs_scaling(self) -> bool:
        """SVM requires feature scaling."""
        return True


class LogisticRegressionModel(ModelBase):
    """Logistic Regression model implementation."""
    
    def create_model(self) -> LogisticRegression:
        """Create Logistic Regression model instance."""
        params = self.get_model_params()
        return LogisticRegression(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Logistic Regression hyperparameters."""
        lr_config = self.config.get('model', {}).get('logistic_regression', {})
        return {
            'C': lr_config.get('C', 1.0),
            'penalty': lr_config.get('penalty', 'l2'),
            'solver': lr_config.get('solver', 'liblinear'),
            'max_iter': lr_config.get('max_iter', 1000),
            'random_state': lr_config.get('random_state', 42),
            'class_weight': lr_config.get('class_weight', 'balanced')
        }
    
    def needs_scaling(self) -> bool:
        """Logistic Regression benefits from feature scaling."""
        return True


class GradientBoostingModel(ModelBase):
    """Gradient Boosting model implementation."""
    
    def create_model(self):
        """Create Gradient Boosting model instance."""
        if not GRADIENT_BOOSTING_AVAILABLE:
            raise ImportError("Gradient Boosting not available - sklearn version may be too old")
        
        params = self.get_model_params()
        return GradientBoostingClassifier(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Gradient Boosting hyperparameters."""
        gb_config = self.config.get('model', {}).get('gradient_boosting', {})
        return {
            'n_estimators': gb_config.get('n_estimators', 100),
            'learning_rate': gb_config.get('learning_rate', 0.1),
            'max_depth': gb_config.get('max_depth', 3),
            'min_samples_split': gb_config.get('min_samples_split', 2),
            'min_samples_leaf': gb_config.get('min_samples_leaf', 1),
            'random_state': gb_config.get('random_state', 42)
        }
    
    def needs_scaling(self) -> bool:
        """Gradient Boosting doesn't require feature scaling."""
        return False
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_


class XGBoostModel(ModelBase):
    """XGBoost model implementation."""
    
    def create_model(self):
        """Create XGBoost model instance."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available - install with 'pip install xgboost'")
        
        params = self.get_model_params()
        return xgb.XGBClassifier(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get XGBoost hyperparameters."""
        xgb_config = self.config.get('model', {}).get('xgboost', {})
        return {
            'n_estimators': xgb_config.get('n_estimators', 100),
            'learning_rate': xgb_config.get('learning_rate', 0.1),
            'max_depth': xgb_config.get('max_depth', 6),
            'subsample': xgb_config.get('subsample', 1.0),
            'colsample_bytree': xgb_config.get('colsample_bytree', 1.0),
            'random_state': xgb_config.get('random_state', 42),
            'scale_pos_weight': xgb_config.get('scale_pos_weight', 1),
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
    
    def needs_scaling(self) -> bool:
        """XGBoost doesn't require feature scaling."""
        return False
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_


class ModelFactory:
    """Factory class for creating model instances."""
    
    # Base model registry
    _BASE_MODELS = {
        'random_forest': RandomForestModel,
        'svm': SVMModel,
        'logistic_regression': LogisticRegressionModel
    }
    
    @classmethod
    def _get_model_registry(cls) -> Dict[str, Any]:
        """Get available model registry based on installed packages."""
        registry = cls._BASE_MODELS.copy()
        
        # Add optional models if available
        if GRADIENT_BOOSTING_AVAILABLE:
            registry['gradient_boosting'] = GradientBoostingModel
        
        if XGBOOST_AVAILABLE:
            registry['xgboost'] = XGBoostModel
        
        return registry
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> ModelBase:
        """
        Create model instance based on type.
        
        Args:
            model_type: Type of model to create ('random_forest', 'svm', etc.)
            config: Model configuration
            
        Returns:
            Model instance
        """
        registry = cls._get_model_registry()
        
        if model_type not in registry:
            available_models = list(registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
        
        model_class = registry[model_type]
        return model_class(config)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls._get_model_registry().keys())
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        registry = cls._get_model_registry()
        
        info = {}
        for model_type, model_class in registry.items():
            info[model_type] = {
                'class_name': model_class.__name__,
                'requires_scaling': model_class({'model': {}}).needs_scaling(),
                'supports_feature_importance': hasattr(model_class, 'get_feature_importance')
            }
        
        return info


class TrainingPipeline:
    """
    Training pipeline orchestrator.
    
    Manages the complete training workflow including data loading,
    feature processing, model training, and evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training pipeline.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.data_handler = DataHandler(config)
        self.feature_manipulator = FeatureManipulator(config)
        self.model = None
        self.logger = logging.getLogger(f"{__name__}.TrainingPipeline")
    
    def train(self, model_type: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute complete training pipeline.
        
        Args:
            model_type: Type of model to train ('random_forest', 'svm', etc.)
            save_path: Optional path to save trained model
            
        Returns:
            Training results including metrics and model path
        """
        self.logger.info(f"Starting training pipeline with {model_type} model")
        
        try:
            # Load and process data
            raw_data = self.data_handler.load_data('train')
            X, y = self.feature_manipulator.process_features(raw_data, 'train')
            
            # Split data
            test_size = self.config.get('training', {}).get('test_size', 0.2)
            random_state = self.config.get('training', {}).get('random_state', 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            self.logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Create and train model
            self.model = ModelFactory.create_model(model_type, self.config)
            
            # Check for grid search configuration
            grid_search_config = self.config.get('training', {}).get('grid_search', {})
            if grid_search_config.get('enabled', False):
                self.model = self._train_with_grid_search(self.model, X_train, y_train, grid_search_config)
            else:
                self.model.train(X_train, y_train)
            
            # Evaluate model
            evaluation_results = self._evaluate_model(X_test, y_test)
            
            # Save model if path provided
            model_path = save_path or self.config.get('training', {}).get('model_path', './model/threat_detector.joblib')
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.model.save_model(model_path)
            
            results = {
                'model_type': model_type,
                'model_path': model_path,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'evaluation': evaluation_results
            }
            
            self.logger.info("Training pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise
    
    def _train_with_grid_search(self, model: ModelBase, X_train: np.ndarray, y_train: np.ndarray, 
                               grid_config: Dict[str, Any]) -> ModelBase:
        """Train model with grid search optimization."""
        self.logger.info("Performing grid search optimization...")
        
        # Create base model
        base_model = model.create_model()
        
        # Setup grid search
        param_grid = grid_config.get('param_grid', {})
        scoring = grid_config.get('scoring', 'roc_auc')
        cv = grid_config.get('cv', 3)
        verbose = grid_config.get('verbose', 1)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            verbose=verbose,
            n_jobs=-1
        )
        
        # Apply scaling if needed
        if model.needs_scaling():
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model.scaler = scaler
        else:
            X_train_scaled = X_train
        
        # Perform grid search
        grid_search.fit(X_train_scaled, y_train)
        
        # Update model with best parameters
        model.model = grid_search.best_estimator_
        
        self.logger.info(f"Grid search completed. Best score: {grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return model
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate trained model performance."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        self.logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        results = {
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Add probability-based metrics if available
        try:
            y_proba = self.model.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                results['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
        except (AttributeError, NotImplementedError):
            pass
        
        # Add feature importance if available
        try:
            if hasattr(self.model, 'get_feature_importance'):
                feature_importance = self.model.get_feature_importance()
                rule_list = self.feature_manipulator.rule_manager.get_rule_list()
                
                if len(feature_importance) == len(rule_list):
                    # Get top 20 most important features
                    importance_df = pd.DataFrame({
                        'rule_id': rule_list,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=False)
                    
                    results['top_features'] = importance_df.head(20).to_dict('records')
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
        
        return results


class DetectionPipeline:
    """
    Detection pipeline for real-time threat detection.
    
    Loads trained model and processes new data for threat detection
    with alerting and logging capabilities.
    """
    
    def __init__(self, config: Dict[str, Any], model_path: str):
        """
        Initialize detection pipeline.
        
        Args:
            config: Pipeline configuration dictionary
            model_path: Path to trained model file
        """
        self.config = config
        self.model_path = model_path
        self.data_handler = DataHandler(config)
        self.feature_manipulator = FeatureManipulator(config)
        self.model = None
        self.logger = logging.getLogger(f"{__name__}.DetectionPipeline")
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load model using the base class method
            model_data = joblib.load(self.model_path)
            model_type = model_data.get('model_type', 'random_forest')
            
            # Create model instance and load state
            self.model = ModelFactory.create_model(model_type.lower(), model_data['config'])
            self.model.model = model_data['model']
            self.model.scaler = model_data.get('scaler')
            
            self.logger.info(f"Model loaded successfully: {model_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run detection on latest data.
        
        Args:
            threshold: Probability threshold for threat classification
            
        Returns:
            Detection results including predictions and alerts
        """
        self.logger.info("Starting detection pipeline...")
        
        try:
            # Load latest data
            raw_data = self.data_handler.load_data('detect')
            
            if raw_data.empty:
                self.logger.info("No new data available for detection")
                return {'status': 'no_data', 'alerts': []}
            
            # Process features
            X, _ = self.feature_manipulator.process_features(raw_data, 'detect')
            
            # Make predictions
            predictions = self.model.predict(X)
            
            try:
                probabilities = self.model.predict_proba(X)
                threat_probabilities = probabilities[:, 1] if probabilities.shape[1] == 2 else None
            except (AttributeError, NotImplementedError):
                threat_probabilities = None
            
            # Process results and generate alerts
            alerts = self._process_detections(raw_data, predictions, threat_probabilities, threshold)
            
            results = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'total_windows': len(predictions),
                'threat_count': int(np.sum(predictions)),
                'alerts': alerts
            }
            
            self.logger.info(f"Detection completed - {results['threat_count']} threats detected")
            return results
            
        except Exception as e:
            self.logger.error(f"Detection pipeline failed: {e}")
            raise
    
    def _process_detections(self, raw_data: pd.DataFrame, predictions: np.ndarray,
                           probabilities: Optional[np.ndarray], threshold: float) -> List[Dict[str, Any]]:
        """Process detections and generate alerts."""
        alerts = []
        
        # Group raw data by window for context
        raw_data['window_id'] = raw_data['timestamp'].apply(
            lambda ts: get_window_id(ts, self.feature_manipulator.window_size_minutes)
        )
        
        window_groups = raw_data.groupby('window_id')
        
        for i, (prediction, window_id) in enumerate(zip(predictions, window_groups.groups.keys())):
            if prediction == 1:  # Threat detected
                window_data = window_groups.get_group(window_id)
                
                alert = {
                    'window_id': window_id,
                    'timestamp': window_data['timestamp'].min().isoformat(),
                    'hostnames': window_data['hostname'].unique().tolist(),
                    'prediction': int(prediction),
                    'confidence': float(probabilities[i]) if probabilities is not None else None,
                    'event_count': len(window_data),
                    'unique_rules': window_data['rule_id'].nunique()
                }
                
                # Add explanation if available
                if probabilities is not None and probabilities[i] >= threshold:
                    alert['explanation'] = self._generate_explanation(window_data, i)
                
                alerts.append(alert)
                
                # Log the detection
                self._log_detection(alert)
        
        return alerts
    
    def _generate_explanation(self, window_data: pd.DataFrame, sample_index: int) -> Dict[str, Any]:
        """Generate explanation for threat detection using SHAP or feature importance."""
        try:
            from system.shap_explainer import SHAPExplainer
            
            explainer = SHAPExplainer(self.model.model, self.feature_manipulator.rule_manager)
            
            # This would need the processed feature vector for this specific window
            # For now, return basic explanation based on rules triggered
            rule_counts = window_data.groupby('rule_id')['count'].sum().sort_values(ascending=False)
            
            return {
                'method': 'rule_frequency',
                'top_rules': [
                    {'rule_id': int(rule_id), 'count': int(count)}
                    for rule_id, count in rule_counts.head(5).items()
                ]
            }
            
        except Exception as e:
            self.logger.warning(f"Could not generate explanation: {e}")
            return {'method': 'unavailable', 'error': str(e)}
    
    def _log_detection(self, alert: Dict[str, Any]) -> None:
        """Log detection alert."""
        from system.logging_utils import log_detection_alert
        
        try:
            # Format message for logging
            message = (
                f"THREAT DETECTED - Window: {alert['window_id']}, "
                f"Hosts: {len(alert['hostnames'])}, "
                f"Confidence: {alert.get('confidence', 'N/A')}, "
                f"Events: {alert['event_count']}"
            )
            
            self.logger.warning(message)
            
            # Send to syslog/QRadar if configured
            if hasattr(log_detection_alert, '__call__'):
                log_detection_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to log detection: {e}")


class PipelineOrchestrator:
    """
    Main orchestrator for the OOP pipeline.
    
    Provides unified interface for training and detection operations
    with model switching and configuration management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(f"{__name__}.PipelineOrchestrator")
        
        # Setup logging
        try:
            setup_global_daily_file_logging(level=logging.INFO, include_stdout=True)
        except Exception as e:
            self.logger.warning(f"Could not setup global logging: {e}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return get_config()
    
    def train(self, model_type: str = 'random_forest', **kwargs) -> Dict[str, Any]:
        """
        Execute training pipeline.
        
        Args:
            model_type: Type of model to train
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        self.logger.info(f"Starting training with {model_type} model")
        
        # Validate model type
        if model_type not in ModelFactory.get_available_models():
            available = ModelFactory.get_available_models()
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        
        # Create and run training pipeline
        training_pipeline = TrainingPipeline(self.config)
        results = training_pipeline.train(model_type, **kwargs)
        
        self.logger.info("Training completed successfully")
        return results
    
    def detect(self, model_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute detection pipeline.
        
        Args:
            model_path: Path to trained model (uses config default if not provided)
            **kwargs: Additional detection parameters
            
        Returns:
            Detection results
        """
        # Use configured model path if not provided
        if model_path is None:
            model_path = self.config.get('training', {}).get('model_path', './model/threat_detector.joblib')
        
        self.logger.info(f"Starting detection with model: {model_path}")
        
        # Create and run detection pipeline
        detection_pipeline = DetectionPipeline(self.config, model_path)
        results = detection_pipeline.detect(**kwargs)
        
        self.logger.info("Detection completed successfully")
        return results
    
    def list_available_models(self) -> List[str]:
        """Get list of available model types."""
        return ModelFactory.get_available_models()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()


def main():
    """Main entry point with CLI support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OOP Threat Detection Pipeline")
    parser.add_argument('mode', choices=['train', 'detect'], help='Pipeline mode')
    parser.add_argument('--model-type', default='random_forest', help='Model type to use')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--model-path', help='Path to trained model (for detection)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(args.config)
        
        if args.mode == 'train':
            print(f"Available models: {orchestrator.list_available_models()}")
            results = orchestrator.train(args.model_type)
            print(f"Training completed successfully!")
            print(f"Model saved to: {results['model_path']}")
            print(f"ROC AUC: {results.get('evaluation', {}).get('roc_auc', 'N/A')}")
            
        elif args.mode == 'detect':
            results = orchestrator.detect(args.model_path)
            print(f"Detection completed!")
            print(f"Status: {results['status']}")
            print(f"Threats detected: {results.get('threat_count', 0)}")
            if results.get('alerts'):
                print(f"Alerts generated: {len(results['alerts'])}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if running as main script or as component test
    if len(sys.argv) > 1 and sys.argv[1] in ['train', 'detect']:
        main()
    else:
        # Basic test of the classes
        print("OOP Threat Detection Pipeline - Component Test")
        
        # Test configuration loading
        config = get_config()
        print(f"Loaded configuration with {len(config)} sections")
        
        # Test model factory
        available_models = ModelFactory.get_available_models()
        print(f"Available models: {available_models}")
        
        # Test model creation
        for model_type in available_models:
            try:
                model = ModelFactory.create_model(model_type, config)
                print(f"✓ Created {model_type} model successfully")
            except Exception as e:
                print(f"✗ Failed to create {model_type} model: {e}")
        
        # Test orchestrator
        try:
            orchestrator = PipelineOrchestrator()
            print(f"✓ Created pipeline orchestrator successfully")
            print(f"Available models: {orchestrator.list_available_models()}")
        except Exception as e:
            print(f"✗ Failed to create orchestrator: {e}")
