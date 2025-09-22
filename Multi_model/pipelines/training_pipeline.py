#!/usr/bin/env python3
"""
TrainingPipeline - Orchestrates the complete training workflow.

This module handles data loading, feature processing, model training,
and evaluation in a unified pipeline with support for grid search
optimization and comprehensive model evaluation.

Python 3.6.8 Compatible
"""

import sys
import os
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import required modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Import our modular components
from data.data_handler import DataHandler
from features.feature_manipulator import FeatureManipulator
from models.model_factory import ModelFactory
from models.base import ModelBase


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
            
            # Handle both supervised and clustering models
            if ModelFactory.is_clustering_model(model_type):
                return self._train_clustering_model(model_type, raw_data, save_path)
            else:
                return self._train_supervised_model(model_type, raw_data, save_path)
                
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise
    
    def _train_supervised_model(self, model_type: str, raw_data: pd.DataFrame, 
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train supervised learning model."""
        # Process features for supervised learning
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
        evaluation_results = self._evaluate_supervised_model(X_test, y_test)
        
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
        
        self.logger.info("Supervised training pipeline completed successfully")
        return results
    
    def _train_clustering_model(self, model_type: str, raw_data: pd.DataFrame,
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train clustering model."""
        # Process features for clustering (no labels needed)
        X, _ = self.feature_manipulator.process_features(raw_data, 'detect')  # Use detect mode for unlabeled data
        
        self.logger.info(f"Clustering data shape: {X.shape}")
        
        # Create and train model
        self.model = ModelFactory.create_model(model_type, self.config)
        
        # Check for grid search configuration  
        grid_search_config = self.config.get('training', {}).get('grid_search', {})
        if grid_search_config.get('enabled', False):
            self.model = self._train_clustering_with_grid_search(self.model, X, grid_search_config)
        else:
            self.model.fit(X)
        
        # Evaluate clustering model
        evaluation_results = self._evaluate_clustering_model(X)
        
        # Save model if path provided
        model_path = save_path or self.config.get('training', {}).get('model_path', f'./model/{model_type}_cluster.joblib')
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save_model(model_path)
        
        results = {
            'model_type': model_type,
            'model_path': model_path,
            'training_samples': len(X),
            'evaluation': evaluation_results
        }
        
        self.logger.info("Clustering training pipeline completed successfully")
        return results
    
    def _train_with_grid_search(self, model: ModelBase, X_train: np.ndarray, y_train: np.ndarray, 
                               grid_config: Dict[str, Any]) -> ModelBase:
        """Train supervised model with grid search optimization."""
        self.logger.info("Performing supervised grid search optimization...")
        
        # Create base model
        base_model = model.create_model()
        
        # Setup grid search
        param_grid = grid_config.get('param_grid') or model.get_grid_search_params()
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
    
    def _train_clustering_with_grid_search(self, model, X: np.ndarray, 
                                          grid_config: Dict[str, Any]):
        """Train clustering model with grid search optimization."""
        self.logger.info("Performing clustering grid search optimization...")
        
        # For clustering, we need to use a different approach since there are no labels
        # We'll use silhouette score as the default metric
        from sklearn.metrics import silhouette_score
        from sklearn.model_selection import ParameterGrid
        
        param_grid = grid_config.get('param_grid') or model.get_grid_search_params()
        scoring_metric = grid_config.get('scoring', 'silhouette')
        model_type = model.__class__.__name__.lower().replace('model', '')  # Extract model type from class name
        
        # Apply scaling if needed
        if model.needs_scaling():
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.scaler = scaler
        else:
            X_scaled = X
        
        best_score = -1
        best_params = None
        best_model = None
        
        # Manual grid search for clustering
        for params in ParameterGrid(param_grid):
            try:
                # Create model with these parameters
                temp_config = self.config.copy()
                temp_config['model'] = {model_type: params}
                temp_model = ModelFactory.create_model(model_type, temp_config)
                
                # Fit model
                temp_model.fit(X_scaled)
                
                # Get cluster labels
                if hasattr(temp_model, 'predict'):
                    labels = temp_model.predict(X_scaled)
                else:
                    labels = temp_model.fit_predict(X_scaled)
                
                # Calculate score
                if scoring_metric == 'silhouette' and len(np.unique(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_model = temp_model
                        
            except Exception as e:
                self.logger.warning(f"Failed to evaluate params {params}: {e}")
        
        if best_model is not None:
            model = best_model
            self.logger.info(f"Best clustering score ({scoring_metric}): {best_score:.4f}")
            self.logger.info(f"Best parameters: {best_params}")
        else:
            # Fallback to normal training
            model.fit(X_scaled)
            self.logger.warning("Grid search failed, using default parameters")
        
        return model
    
    def _evaluate_supervised_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate trained supervised model performance."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        self.logger.info("Evaluating supervised model performance...")
        
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
    
    def _evaluate_clustering_model(self, X: np.ndarray) -> Dict[str, Any]:
        """Evaluate trained clustering model performance."""
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        self.logger.info("Evaluating clustering model performance...")
        
        # Apply scaling if used during training
        if self.model.scaler is not None:
            X_scaled = self.model.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get cluster labels
        try:
            if hasattr(self.model, 'predict'):
                labels = self.model.predict(X_scaled)
            else:
                labels = self.model.fit_predict(X_scaled)
        except Exception as e:
            self.logger.error(f"Failed to get cluster labels: {e}")
            return {'error': str(e)}
        
        # Calculate clustering metrics
        results = {
            'n_clusters': len(np.unique(labels)),
            'cluster_labels': np.unique(labels).tolist(),
            'cluster_counts': {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
        }
        
        # Add clustering quality metrics
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for these metrics
                results['silhouette_score'] = silhouette_score(X_scaled, labels)
                results['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
                results['davies_bouldin_score'] = davies_bouldin_score(X_scaled, labels)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate clustering metrics: {e}")
        
        # Add noise points info for DBSCAN
        if -1 in labels:
            results['noise_points'] = int(np.sum(labels == -1))
            results['noise_ratio'] = float(np.sum(labels == -1) / len(labels))
        
        return results