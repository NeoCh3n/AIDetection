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
from tqdm import tqdm
from contextlib import contextmanager
import time

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import required modules
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

# Import our modular components
from data.data_handler import DataHandler
from features.feature_manipulator import FeatureManipulator
from models.model_factory import ModelFactory
from models.base import ModelBase

sys.path.insert(0, '..')
from system.shap_explainer import Explainer


@contextmanager
def joblib_progress_bar(total: int, desc: str = "GridSearchCV", width: int = 30):
    """
    Render a simple in-place text progress bar for joblib-backed tasks.

    Parameters
    ----------
    total : int
        Total number of tasks (e.g., n_candidates * cv_splits).
    desc : str
        Description prefix for the progress bar.
    width : int
        Width of the progress bar in characters.
    """
    import joblib as _joblib

    start_time = time.time()
    completed = [0]

    def _render(final: bool = False):
        done = min(completed[0], total)
        pct = (float(done) / total) if total else 1.0
        filled = int(width * pct)
        bar = f"[{'#' * filled}{'-' * (width - filled)}]"
        elapsed = time.time() - start_time
        rate = (done / elapsed) if elapsed > 0 else 0.0
        msg = f"\r{desc} {bar} {done}/{total} ({pct*100:5.1f}%) | {rate:.1f} it/s"
        try:
            sys.stdout.write(msg)
            if final:
                sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass

    class _PBCallback(_joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            try:
                completed[0] += self.batch_size
                _render()
            except Exception:
                pass
            return super(_PBCallback, self).__call__(*args, **kwargs)

    old_cb = _joblib.parallel.BatchCompletionCallBack
    _joblib.parallel.BatchCompletionCallBack = _PBCallback
    try:
        _render()
        yield
    finally:
        _joblib.parallel.BatchCompletionCallBack = old_cb
        _render(final=True)


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
        self.model = self._train_with_grid_search(self.model, X_train, y_train, grid_search_config)
        
        # Evaluate model
        evaluation_results = self._evaluate_supervised_model(X_test, y_test)
        
        # Prepare results dict
        model_path = save_path or self.config.get('training', {}).get('model_path', './model/threat_detector.joblib')
        results = {
            'model_type': model_type,
            'model_path': model_path,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'evaluation': evaluation_results
        }
        
        # Explain model using SHAP explainer
        try:
            explainer = Explainer()
            rule_list = self.feature_manipulator.rule_manager.get_rule_list()
            feature_name_list = [f"rule_{rid}" for rid in rule_list]
            output_dir = os.path.join(PROJECT_ROOT, 'results')
            os.makedirs(output_dir, exist_ok=True)
            
            # Use a sample of test data for explanation
            sample_size = min(10, len(X_test))
            explanation_results = explainer.explain(
                model=self.model.model if not isinstance(self.model.model, tuple) else self.model.model[2],  # Handle scaler case
                background_data=X_train,
                instance_data=X_test[:sample_size],
                feature_name_list=feature_name_list,
                output_dir=output_dir,
                plot=False,
                plot_in_terminal=True,
                summary_report=True
            )
            results['explanation'] = explanation_results
        except Exception as e:
            self.logger.warning(f"Model explanation failed: {e}")
            results['explanation'] = {'error': str(e)}
        
        # Save model if path provided
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save_model(model_path)
        
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
        """Train supervised model with GridSearchCV optimization and joblib progress bar."""
        self.logger.info("Performing supervised grid search optimization with GridSearchCV...")
        
        # Create base model
        base_estimator = model.create_model()
        
        # Setup grid search parameters
        param_grid = model.get_grid_search_params()
        scoring_metric = grid_config.get('scoring', 'roc_auc')
        cv = grid_config.get('cv', 3)
        
        if not param_grid:
            self.logger.warning("No parameter grid provided for grid search. Skipping.")
            # Fit base model
            if model.needs_scaling():
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                base_estimator.fit(X_train_scaled, y_train)
                model.model = ('scaler', scaler, base_estimator)
            else:
                base_estimator.fit(X_train, y_train)
                model.model = base_estimator
            return model
        
        # Use GridSearchCV for optimization with joblib progress bar
        if model.needs_scaling():
            # Use Pipeline for scaling
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', base_estimator)
            ])
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring_metric,
                cv=cv,
                n_jobs=-1,
                verbose=0
            )
            total_tasks = len(ParameterGrid(param_grid)) * cv
            with joblib_progress_bar(total=total_tasks, desc="Grid Search"):
                grid_search.fit(X_train, y_train)
            model.model = grid_search.best_estimator_
        else:
            grid_search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring=scoring_metric,
                cv=cv,
                n_jobs=-1,
                verbose=0
            )
            total_tasks = len(ParameterGrid(param_grid)) * cv
            with joblib_progress_bar(total=total_tasks, desc="Grid Search"):
                grid_search.fit(X_train, y_train)
            model.model = grid_search.best_estimator_
        
        self.logger.info(f"Grid search completed. Best score: {grid_search.best_score_:.4f} with params: {grid_search.best_params_}")
        
        return model
    
    def _train_clustering_with_grid_search(self, model, X: np.ndarray, 
                                          grid_config: Dict[str, Any]):
        """Train clustering model with grid search optimization."""
        self.logger.info("Performing clustering grid search optimization...")
        
        param_grid = grid_config.get('param_grid') or model.get_grid_search_params()
        
        if not param_grid:
            self.logger.warning("No parameter grid provided for clustering grid search. Skipping.")
            estimator = model.create_model()
            if model.needs_scaling():
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                estimator.fit(X_scaled)
                model.model = ('scaler', scaler, estimator)
            else:
                estimator.fit(X)
                model.model = estimator
            return model
        
        best_score = -np.inf
        best_params = None
        best_estimator = None
        
        pg = list(ParameterGrid(param_grid))
        self.logger.info(f"Clustering grid search: {len(pg)} parameter combinations")
        
        for params in tqdm(pg, desc="Clustering Grid Search", unit="combination"):
            try:
                est = clone(model.create_model()).set_params(**params)
                if model.needs_scaling():
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    est.fit(X_scaled)
                    labels = est.predict(X_scaled)
                    score = silhouette_score(X_scaled, labels)
                else:
                    est.fit(X)
                    labels = est.predict(X)
                    score = silhouette_score(X, labels)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    if model.needs_scaling():
                        best_estimator = ('scaler', scaler, est)
                    else:
                        best_estimator = est
                        
            except Exception as e:
                self.logger.debug(f"Clustering failed on params {params}: {e}")
                continue
        
        if best_params is None:
            self.logger.warning("Clustering grid search found no valid combinations. Using default estimator.")
            est = model.create_model()
            if model.needs_scaling():
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                est.fit(X_scaled)
                model.model = ('scaler', scaler, est)
            else:
                est.fit(X)
                model.model = est
        else:
            model.model = best_estimator
            self.logger.info(f"Clustering grid search completed. Best score: {best_score:.4f} with params: {best_params}")
        
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
        if hasattr(self.model, 'scaler') and self.model.scaler is not None:
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