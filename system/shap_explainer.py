#!/usr/bin/env python3
"""
SHAP Explainer for RandomForest Threat Detection Models

This module provides SHAP (SHapley Additive exPlanations) explainability for
RandomForest models used in malicious activity detection. It integrates with
the logging system to provide detailed explanations when malicious predictions
are made.

Features:
- Simple 2-parameter constructor (model, background_data)
- Automatic rule mapping configuration
- Logging integration for threat detection records
- Feature importance ranking
- Markdown report generation

Usage:
    explainer = Explainer(model, background_data)
    shap_values = explainer.explain(suspicious_data)
    ranking = explainer.get_feature_importance(suspicious_data)
"""

import shap
import json
import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

class Explainer:
    """
    SHAP Explainer for RandomForest threat detection models.
    
    Provides explainability for malicious predictions using SHAP values,
    with integrated logging and rule mapping for security analysis.
    """
    
    def __init__(self, model, background_data, feature_names=None, rule_mapping=None):
        """
        Initialize SHAP explainer for RandomForest models.
        
        Args:
            model: Trained RandomForest model with predict() and predict_proba() methods
            background_data: Background dataset for SHAP baseline (numpy array or pandas DataFrame)
            feature_names: Optional list of feature names (auto-generated if None)
            rule_mapping: Optional dict mapping feature names to rule descriptions
        """
        self.model = model
        self.background_data = self._validate_background_data(background_data)
        self.feature_names = self._setup_feature_names(feature_names)
        self.rule_mapping = rule_mapping or self._load_rule_mapping()
        
        # Validate model compatibility
        self._validate_model()
        
        # Initialize SHAP TreeExplainer optimized for RandomForest
        self.explainer = shap.TreeExplainer(self.model, data=self.background_data)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("SHAP Explainer initialized successfully")

    def _validate_background_data(self, background_data):
        """Validate and convert background data to numpy array."""
        if hasattr(background_data, 'values'):  # pandas DataFrame
            return background_data.values
        elif isinstance(background_data, np.ndarray):
            return background_data
        else:
            try:
                return np.array(background_data)
            except Exception as e:
                raise ValueError(f"Invalid background_data format: {str(e)}")

    def _setup_feature_names(self, feature_names):
        """Setup feature names based on background data shape."""
        n_features = self.background_data.shape[1]
        
        if feature_names is None:
            return [f'feature_{i}' for i in range(n_features)]
        elif len(feature_names) != n_features:
            self.logger.warning(f"Feature names length ({len(feature_names)}) doesn't match data features ({n_features})")
            return [f'feature_{i}' for i in range(n_features)]
        else:
            return list(feature_names)

    def _validate_model(self):
        """Validate model compatibility with SHAP TreeExplainer."""
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("Model must be a tree-based ensemble (e.g., RandomForest)")
        
        if not hasattr(self.model, 'predict') or not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model must have predict() and predict_proba() methods")
        
        # Check feature compatibility
        expected_features = getattr(self.model, 'n_features_in_', None)
        if expected_features and expected_features != self.background_data.shape[1]:
            self.logger.warning(f"Feature count mismatch: model expects {expected_features}, got {self.background_data.shape[1]}")

    def _load_rule_mapping(self):
        """Load rule mapping from configuration if available."""
        try:
            # Try to load from system config
            import system.config as config
            config_data = config.get_config()
            
            rule_mapping_path = config_data.get('rule_mapping_path', 'rule_mapping.json')
            if not os.path.isabs(rule_mapping_path):
                # Make path relative to project root
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                rule_mapping_path = os.path.join(project_root, rule_mapping_path)
            
            if os.path.exists(rule_mapping_path):
                with open(rule_mapping_path, 'r') as f:
                    rule_mapping = json.load(f)
                self.logger.info(f"Loaded rule mapping from {rule_mapping_path}")
                return rule_mapping
            else:
                self.logger.info(f"Rule mapping file not found at {rule_mapping_path}")
                
        except Exception as e:
            self.logger.warning(f"Could not load rule mapping: {str(e)}")
        
        # Return default mapping
        return {name: f"Rule_{name}" for name in self.feature_names}

    def explain(self, X, log_results=True):
        """
        Generate SHAP explanations for input data.
        
        Args:
            X: Input data to explain (single instance or batch)
            log_results: Whether to log explanation results
            
        Returns:
            SHAP values for the input data
        """
        try:
            # Validate and prepare input
            X_processed = self._prepare_input(X)
            
            # Generate SHAP values
            shap_values = self.explainer.shap_values(X_processed)
            
            if log_results:
                self._log_explanation(X_processed, shap_values)
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation: {str(e)}")
            raise

    def _prepare_input(self, X):
        """Prepare input data for SHAP explanation."""
        # Convert to numpy array if needed
        if hasattr(X, 'values'):  # pandas DataFrame
            X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            X_array = np.array(X)
        
        # Ensure 2D
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        
        # Check feature count
        if X_array.shape[1] != self.background_data.shape[1]:
            raise ValueError(f"Input features ({X_array.shape[1]}) don't match background data ({self.background_data.shape[1]})")
        
        return X_array

    def _log_explanation(self, X, shap_values):
        """Log detailed SHAP explanation summary."""
        try:
            # Handle multi-class output (take class 1 for binary classification)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                values = shap_values[1]  # Malicious class
                class_explained = "malicious"
            elif isinstance(shap_values, list):
                values = shap_values[0]
                class_explained = "class_0"
            else:
                values = shap_values
                class_explained = "prediction"
            
            # Calculate importance for logging
            if values.ndim > 1:
                avg_importance = np.abs(values).mean(axis=0)
                instance_count = values.shape[0]
            else:
                avg_importance = np.abs(values)
                instance_count = 1
            
            # Get top contributing features
            top_indices = np.argsort(avg_importance)[-10:][::-1]  # Top 10
            
            # Create explanation summary
            explanation_summary = {
                'timestamp': datetime.now().isoformat(),
                'class_explained': class_explained,
                'instance_count': instance_count,
                'total_features': len(avg_importance),
                'top_contributing_features': []
            }
            
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    rule_name = self.rule_mapping.get(feature_name, feature_name)
                    
                    explanation_summary['top_contributing_features'].append({
                        'rank': len(explanation_summary['top_contributing_features']) + 1,
                        'feature': feature_name,
                        'rule_description': rule_name,
                        'importance_score': float(avg_importance[idx])
                    })
            
            # Log the explanation
            self.logger.info("="*60)
            self.logger.info("SHAP EXPLANATION GENERATED")
            self.logger.info("="*60)
            self.logger.info(f"Explanation Summary:\n{json.dumps(explanation_summary, indent=2)}")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.warning(f"Could not log SHAP explanation: {str(e)}")

    def get_feature_importance(self, X):
        """
        Get feature importance ranking for given input.
        
        Args:
            X: Input data to analyze
            
        Returns:
            List of dicts with feature importance ranking
        """
        try:
            shap_values = self.explain(X, log_results=False)
            
            # Handle multi-class output
            if isinstance(shap_values, list) and len(shap_values) > 1:
                values = shap_values[1]  # Malicious class
            elif isinstance(shap_values, list):
                values = shap_values[0]
            else:
                values = shap_values
            
            # Calculate importance
            if values.ndim > 1:
                importance = np.abs(values).mean(axis=0)
            else:
                importance = np.abs(values)
            
            # Create ranking
            indices = np.argsort(importance)[::-1]
            ranking = []
            
            for rank, idx in enumerate(indices, 1):
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    rule_name = self.rule_mapping.get(feature_name, feature_name)
                    
                    ranking.append({
                        'rank': rank,
                        'feature': feature_name,
                        'rule': rule_name,
                        'importance': float(importance[idx])
                    })
            
            return ranking
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return []

    def generate_markdown_report(self, X, output_path=None):
        """
        Generate a detailed markdown report for the SHAP explanation.
        
        Args:
            X: Input data to analyze
            output_path: Optional path to save the report
            
        Returns:
            String containing the markdown report
        """
        try:
            # Get predictions and explanations
            X_processed = self._prepare_input(X)
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(X_processed)
            ranking = self.get_feature_importance(X)
            
            # Generate report
            report_lines = [
                "# SHAP Threat Detection Explanation Report",
                "",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Instances Analyzed:** {len(X_processed)}",
                "",
                "## Model Predictions",
                ""
            ]
            
            # Add prediction details
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                report_lines.extend([
                    f"### Instance {i+1}",
                    f"- **Prediction:** {pred}",
                    f"- **Confidence:** {prob.max():.4f}",
                    f"- **Class Probabilities:** {prob.tolist()}",
                    ""
                ])
            
            # Add top features
            report_lines.extend([
                "## Top Contributing Features",
                "",
                "| Rank | Feature | Rule Description | Importance Score |",
                "|------|---------|------------------|------------------|"
            ])
            
            for item in ranking[:15]:  # Top 15 features
                report_lines.append(
                    f"| {item['rank']} | {item['feature']} | {item['rule']} | {item['importance']:.6f} |"
                )
            
            report_lines.extend([
                "",
                "## Analysis Notes",
                "",
                "- Higher importance scores indicate features that contributed more to the prediction",
                "- This explanation helps understand which security rules triggered the threat detection",
                "- Review the top contributing features for threat analysis and rule refinement",
                ""
            ])
            
            markdown_report = "\n".join(report_lines)
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_report)
                self.logger.info(f"SHAP report saved to {output_path}")
            
            return markdown_report
            
        except Exception as e:
            self.logger.error(f"Error generating markdown report: {str(e)}")
            return f"# Error\n\nCould not generate report: {str(e)}"

    def __str__(self):
        """String representation of the explainer."""
        return (f"SHAP Explainer(model={type(self.model).__name__}, "
                f"features={len(self.feature_names)}, "
                f"background_size={len(self.background_data)})")

    def __repr__(self):
        """Detailed representation of the explainer."""
        return self.__str__()