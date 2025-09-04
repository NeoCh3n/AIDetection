#!/usr/bin/env python3
"""
SHAP Explainer for Machine Learning Models

This module provides SHAP (SHapley Additive exPlanations) explainability for
machine learning models used in threat detection and classification tasks.

Features:
- Simple initialization with no parameters
- Main explain() function with required and optional parameters
- Support for tree-based models (RandomForest, XGBoost, etc.)
- Feature importance ranking and visualization
- Terminal and file-based reporting
- Comprehensive logging integration

Usage:
    explainer = Explainer()
    results = explainer.explain(
        model=my_model,
        data=input_data,
        feature_name_list=feature_names,
        output_dir="./results",
        plot=True,
        plot_in_terminal=True,
        summary_report=True
    )
"""

import shap
import json
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotext as plt_terminal
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

class Explainer:
    """
    SHAP Explainer for machine learning models.
    
    Provides explainability for model predictions using SHAP values,
    with support for various model types and comprehensive analysis features.
    
    The class is initialized without parameters and provides a main explain()
    function that accepts all necessary inputs.
    """
    
    def __init__(self):
        """
        Initialize SHAP explainer with no parameters.
        
        The explainer is ready to use with the explain() method which accepts
        all necessary parameters for model explanation.
        """
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("SHAP Explainer initialized and ready to use")
    
    def explain(self, 
                model, 
                data: Union[np.ndarray, pd.DataFrame], 
                feature_name_list: List[str], 
                output_dir: str,
                plot: bool = False,
                plot_in_terminal: bool = False, 
                summary_report: bool = False) -> Dict[str, Any]:
        """
        Explain a machine learning model using SHAP values.
        
        Args:
            model: Trained ML model with predict() and predict_proba() methods
            data: Input data to explain (background data for SHAP baseline)
            feature_name_list: List of feature names for meaningful explanations
            output_dir: Directory to save outputs (plots, reports)
            plot: Whether to generate and save visualization plots (default: False)
            plot_in_terminal: Whether to display plots in terminal (default: False)
            summary_report: Whether to generate markdown summary report (default: False)
            
        Returns:
            Dict containing SHAP values, feature importance, and file paths
            
        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        try:
            # Validate inputs
            self._validate_inputs(model, data, feature_name_list, output_dir)
            
            # Prepare data and initialize SHAP
            background_data = self._prepare_background_data(data)
            feature_names = self._validate_feature_names(feature_name_list, background_data)
            
            # Initialize SHAP explainer
            explainer = self._initialize_shap_explainer(model, background_data)
            
            # Generate SHAP explanations
            self.logger.info(f"Generating SHAP explanations for {background_data.shape[0]} samples...")
            shap_values = explainer.shap_values(background_data)
            
            # Calculate feature importance
            importance_ranking = self._calculate_feature_importance(shap_values, feature_names)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare results
            results = {
                'shap_values': shap_values,
                'feature_importance': importance_ranking,
                'model_type': type(model).__name__,
                'samples_analyzed': background_data.shape[0],
                'features_count': len(feature_names),
                'timestamp': datetime.now().isoformat(),
                'output_files': {}
            }
            
            # Generate visualizations if requested
            if plot:
                plot_files = self._generate_plots(
                    shap_values, feature_names, importance_ranking, output_dir
                )
                results['output_files'].update(plot_files)
            
            # Display terminal plots if requested
            if plot_in_terminal:
                self._display_terminal_visualizations(
                    shap_values, feature_names, importance_ranking, model, background_data
                )
            
            # Generate summary report if requested
            if summary_report:
                report_file = self._generate_summary_report(
                    results, model, background_data, feature_names, output_dir
                )
                results['output_files']['summary_report'] = report_file
            
            self.logger.info(f"SHAP analysis completed successfully. Results saved to {output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in SHAP explanation: {e}")
            raise
    
    def _validate_inputs(self, model, data, feature_name_list, output_dir):
        """Validate all input parameters."""
        # Validate model
        if model is None:
            raise ValueError("Model cannot be None")
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a 'predict' method")
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model must have a 'predict_proba' method")
        
        # Validate data
        if data is None:
            raise ValueError("Data cannot be None")
        
        # Validate feature names
        if not feature_name_list or not isinstance(feature_name_list, (list, tuple)):
            raise ValueError("feature_name_list must be a non-empty list or tuple")
        
        # Validate output directory
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("output_dir must be a non-empty string")
        
        self.logger.info("Input validation completed successfully")
    
    def _prepare_background_data(self, data):
        """Prepare and validate background data."""
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data.copy()
        else:
            try:
                data_array = np.array(data)
            except Exception as e:
                raise ValueError(f"Cannot convert data to numpy array: {e}")
        
        # Validate shape
        if data_array.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data_array.shape}")
        
        if data_array.shape[0] == 0 or data_array.shape[1] == 0:
            raise ValueError(f"Data cannot be empty, got shape {data_array.shape}")
        
        # Check for invalid values
        if not np.isfinite(data_array).all():
            self.logger.warning("Data contains non-finite values (NaN, inf)")
        
        return data_array
    
    def _validate_feature_names(self, feature_name_list, data_array):
        """Validate feature names against data dimensions."""
        feature_names = list(feature_name_list)
        n_features = data_array.shape[1]
        
        if len(feature_names) != n_features:
            raise ValueError(f"Number of feature names ({len(feature_names)}) must match "
                           f"number of data features ({n_features})")
        
        # Check for duplicates
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Feature names must be unique")
        
        # Ensure all names are strings
        feature_names = [str(name) for name in feature_names]
        
        return feature_names
    
    def _initialize_shap_explainer(self, model, background_data):
        """Initialize the appropriate SHAP explainer."""
        try:
            # Try TreeExplainer first (fastest for tree models)
            if (hasattr(model, 'estimators_') or hasattr(model, 'tree_') or 
                hasattr(model, 'booster')):
                explainer = shap.TreeExplainer(model, data=background_data)
                self.logger.info("Using SHAP TreeExplainer")
                return explainer
        except Exception as e:
            self.logger.warning(f"TreeExplainer failed: {e}. Falling back to KernelExplainer")
        
        try:
            # Fallback to KernelExplainer (works with any model)
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            self.logger.info("Using SHAP KernelExplainer")
            return explainer
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SHAP explainer: {e}")
    
    def _calculate_feature_importance(self, shap_values, feature_names):
        """Calculate feature importance ranking from SHAP values."""
        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class: use class 1 for binary or first class for multi-class
                if len(shap_values) == 2:
                    values = shap_values[1]  # Binary classification - positive class
                else:
                    values = shap_values[0]  # Multi-class - first class
            else:
                values = shap_values
            
            # Convert to numpy array if needed
            if hasattr(values, 'values'):
                values = values.values
            values = np.array(values)
            
            # Calculate importance
            if values.ndim == 2:
                importance = np.abs(values).mean(axis=0)
            else:
                importance = np.abs(values)
            
            # Create ranking
            indices = np.argsort(importance)[::-1]
            ranking = []
            
            for rank, idx in enumerate(indices, 1):
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                    
                    ranking.append({
                        'rank': rank,
                        'feature': feature_name,
                        'importance': float(importance[idx])
                    })
            
            return ranking
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return []
    
    def _generate_plots(self, shap_values, feature_names, importance_ranking, output_dir):
        """Generate visualization plots and save to files."""
        plot_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Generate feature importance plot
            importance_path = os.path.join(output_dir, f"feature_importance_{timestamp}.png")
            self._create_importance_plot(importance_ranking, importance_path)
            plot_files['importance_plot'] = importance_path
            
            # Generate SHAP heatmap
            heatmap_path = os.path.join(output_dir, f"shap_heatmap_{timestamp}.png")
            self._create_heatmap_plot(shap_values, feature_names, importance_ranking, heatmap_path)
            plot_files['heatmap'] = heatmap_path
            
            self.logger.info(f"Plots generated successfully in {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
        
        return plot_files
    
    def _create_importance_plot(self, importance_ranking, output_path, top_features=15):
        """Create and save feature importance plot."""
        if not importance_ranking:
            self.logger.warning("No feature importance data available for plotting")
            return
        
        # Prepare data
        ranking = importance_ranking[:top_features]
        features = [item['feature'] for item in ranking]
        scores = [item['importance'] for item in ranking]
        
        # Truncate long feature names
        features = [f[:30] + "..." if len(f) > 30 else f for f in features]
        
        # Create plot
        plt.figure(figsize=(10, max(6, len(features) * 0.4)))
        
        # Color gradient
        colors = plt.cm.Reds(np.linspace(0.8, 0.3, len(scores)))
        
        bars = plt.barh(range(len(features)), scores, color=colors)
        
        # Customize
        plt.yticks(range(len(features)), features)
        plt.xlabel('SHAP Importance Score')
        plt.title(f'Top {len(features)} Feature Importance Ranking')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_width() + max(scores) * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}', ha='left', va='center')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Feature importance plot saved to {output_path}")
    
    def _create_heatmap_plot(self, shap_values, feature_names, importance_ranking, output_path, top_features=20):
        """Create and save SHAP heatmap plot."""
        try:
            # Handle multi-class output
            if isinstance(shap_values, list) and len(shap_values) > 1:
                values = shap_values[1]  # Malicious class
                class_name = "Malicious Class"
            elif isinstance(shap_values, list):
                values = shap_values[0]
                class_name = "Class 0"
            else:
                values = shap_values
                class_name = "Prediction"
            
            # Ensure numpy array
            if hasattr(values, 'values'):
                values = values.values
            values = np.array(values)
            
            # Prepare data for heatmap
            n_instances = min(values.shape[0], 10)  # Limit instances for readability
            n_features = min(top_features, len(importance_ranking))
            
            # Get top feature indices
            top_feature_names = [item['feature'] for item in importance_ranking[:n_features]]
            top_feature_indices = []
            
            for fname in top_feature_names:
                try:
                    idx = feature_names.index(fname)
                    top_feature_indices.append(idx)
                except ValueError:
                    continue
            
            if not top_feature_indices:
                self.logger.warning("No valid feature indices found for heatmap")
                return
            
            # Create heatmap data
            if values.ndim > 1:
                heatmap_data = values[:n_instances, top_feature_indices]
                instance_labels = [f"Instance {i+1}" for i in range(n_instances)]
            else:
                heatmap_data = values[top_feature_indices].reshape(1, -1)
                instance_labels = ["Instance 1"]
            
            # Create feature labels
            feature_labels = []
            for fname in top_feature_names[:len(top_feature_indices)]:
                label = fname if len(fname) <= 25 else fname[:22] + "..."
                feature_labels.append(label)
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            
            # Symmetric color scale around 0
            absmax = float(np.nanmax(np.abs(heatmap_data))) if np.size(heatmap_data) else 1.0
            if absmax == 0:
                absmax = 1.0
            
            img = plt.imshow(
                heatmap_data,
                aspect='auto',
                cmap='RdBu_r',  # Red positive, Blue negative
                vmin=-absmax,
                vmax=absmax,
                interpolation='nearest'
            )
            
            # Add colorbar
            cbar = plt.colorbar(img)
            cbar.set_label('SHAP Value (Impact on Prediction)')
            
            # Set labels
            plt.xticks(ticks=np.arange(len(feature_labels)), labels=feature_labels, rotation=45, ha='right')
            plt.yticks(ticks=np.arange(len(instance_labels)), labels=instance_labels)
            
            # Annotate cells
            h, w = heatmap_data.shape
            for i in range(h):
                for j in range(w):
                    val = heatmap_data[i, j]
                    color = 'black' if abs(val) < (0.6 * absmax) else 'white'
                    plt.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=8, color=color)
            
            # Title and labels
            plt.title(
                f'SHAP Feature Importance Heatmap - {class_name}\n'
                f'Top {n_features} Contributing Features',
                fontsize=14, fontweight='bold'
            )
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Data Instances', fontsize=12)
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"SHAP heatmap saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating heatmap: {e}")
    
    def _display_terminal_visualizations(self, shap_values, feature_names, importance_ranking, model, background_data):
        """Display visualizations in terminal."""
        try:
            # Get model predictions
            predictions = model.predict(background_data)
            probabilities = model.predict_proba(background_data)
            
            # Display header
            print("\n" + "=" * 70)
            print("SHAP MODEL EXPLANATION SUMMARY")
            print("=" * 70)
            
            # Model info
            print(f"\nModel: {type(model).__name__}")
            print(f"Samples analyzed: {len(background_data)}")
            print(f"Features: {len(feature_names)}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Predictions summary
            print(f"\nPredictions:")
            unique_preds, counts = np.unique(predictions, return_counts=True)
            for pred, count in zip(unique_preds, counts):
                confidence = probabilities[predictions == pred].max(axis=1).mean()
                print(f"  {pred}: {count} samples (avg confidence: {confidence:.2%})")
            
            # Feature importance table
            top_features = importance_ranking[:10]  # Top 10
            if top_features:
                print(f"\nTop {len(top_features)} Most Important Features:")
                print("-" * 70)
                print(f"{'Rank':<4} {'Importance':<12} {'Feature':<50}")
                print("-" * 70)
                
                for item in top_features:
                    feature = item['feature']
                    if len(feature) > 45:
                        feature = feature[:42] + "..."
                    
                    # Visual bar
                    bar_length = min(20, int(item['importance'] * 100))
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    
                    print(f"{item['rank']:<4} {item['importance']:<12.4f} {feature:<50}")
                    print(f"     {bar}")
                
                print("-" * 70)
                print(f"Most critical feature: {top_features[0]['feature']}")
                print(f"Importance score: {top_features[0]['importance']:.4f}")
            
            print("=" * 70)
            
        except Exception as e:
            self.logger.error(f"Error displaying terminal visualizations: {e}")
            print(f"ERROR: {e}")
    
    def _generate_summary_report(self, results, model, background_data, feature_names, output_dir):
        """Generate markdown summary report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_dir, f"shap_analysis_report_{timestamp}.md")
            
            # Get predictions
            predictions = model.predict(background_data)
            probabilities = model.predict_proba(background_data)
            
            # Generate report content
            report_lines = [
                "# SHAP Model Explanation Report",
                "",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Model Type:** {type(model).__name__}",
                f"**Samples Analyzed:** {len(background_data)}",
                f"**Total Features:** {len(feature_names)}",
                "",
                "## Model Predictions Summary",
                ""
            ]
            
            # Add prediction details
            unique_preds, counts = np.unique(predictions, return_counts=True)
            for pred, count in zip(unique_preds, counts):
                confidence = probabilities[predictions == pred].max(axis=1).mean()
                report_lines.extend([
                    f"- **Prediction {pred}:** {count} samples (avg confidence: {confidence:.2%})",
                ])
            
            # Add feature importance table
            top_features = results['feature_importance'][:15]  # Top 15
            if top_features:
                report_lines.extend([
                    "",
                    "## Top Contributing Features",
                    "",
                    "| Rank | Feature | Importance Score |",
                    "|------|---------|------------------|"
                ])
                
                for item in top_features:
                    report_lines.append(
                        f"| {item['rank']} | {item['feature']} | {item['importance']:.6f} |"
                    )
                
                # Add key findings
                report_lines.extend([
                    "",
                    "## Key Findings",
                    "",
                    f"- **Most Critical Feature:** {top_features[0]['feature']}",
                    f"- **Highest Importance Score:** {top_features[0]['importance']:.4f}",
                    f"- **Analysis Scope:** Top {len(top_features)} out of {len(results['feature_importance'])} total features",
                    ""
                ])
            
            # Add visualizations section if files exist
            if results['output_files']:
                report_lines.extend([
                    "## Generated Visualizations",
                    ""
                ])
                
                for viz_type, file_path in results['output_files'].items():
                    if viz_type != 'summary_report' and file_path:
                        filename = os.path.basename(file_path)
                        viz_name = viz_type.replace('_', ' ').title()
                        report_lines.extend([
                            f"### {viz_name}",
                            f"![{viz_name}]({filename})",
                            ""
                        ])
            
            # Add technical details
            report_lines.extend([
                "## Technical Details",
                "",
                f"- **SHAP Analysis Timestamp:** {results['timestamp']}",
                f"- **Background Data Shape:** {background_data.shape}",
                f"- **Feature Count:** {results['features_count']}",
                f"- **Samples Analyzed:** {results['samples_analyzed']}",
                ""
            ])
            
            # Write report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            
            self.logger.info(f"Summary report saved to {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return None
    
    def __str__(self) -> str:
        """String representation of the explainer."""
        return "SHAP Explainer (ready to use with explain() method)"
    
    def __repr__(self) -> str:
        """Detailed representation of the explainer."""
        return self.__str__()
        """
        Generate SHAP explanations for input data.
        
        Args:
            X: Input data to explain (single instance or batch)
            max_samples: Maximum number of samples to explain (for performance)
            log_results: Whether to log explanation results
            
        Returns:
            SHAP values array with shape (n_samples, n_features) or (n_samples, n_features, n_classes)
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Prepare input data
            X_processed = self._prepare_input_data(X)
            
            # Limit samples for performance
            if X_processed.shape[0] > max_samples:
                self.logger.warning(f"Limiting explanation to {max_samples} samples (got {X_processed.shape[0]})")
                X_processed = X_processed[:max_samples]
            
            # Generate SHAP values
            self.logger.info(f"Generating SHAP explanations for {X_processed.shape[0]} samples...")
            shap_values = self.explainer.shap_values(X_processed)
            
            # Log results if requested
            if log_results:
                self._log_explanation_summary(X_processed, shap_values)
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation: {e}")
            raise
    
    def _prepare_input_data(self, X):
        """Prepare and validate input data for SHAP explanation."""
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X.copy()
        else:
            try:
                X_array = np.array(X)
            except Exception as e:
                raise ValueError(f"Cannot convert input data to numpy array: {e}")
        
        # Ensure 2D
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        elif X_array.ndim != 2:
            raise ValueError(f"Input data must be 1D or 2D, got shape {X_array.shape}")
        
        # Validate feature count
        if X_array.shape[1] != self.background_data.shape[1]:
            raise ValueError(f"Input features ({X_array.shape[1]}) don't match "
                           f"background data features ({self.background_data.shape[1]})")
        
        # Check for invalid values
        if not np.isfinite(X_array).all():
            raise ValueError("Input data contains non-finite values (NaN, inf)")
        
        return X_array
    
    def _log_explanation_summary(self, X, shap_values):
        """Log a summary of the SHAP explanation results."""
        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class: use class 1 for binary or first class for multi-class
                if len(shap_values) == 2:
                    values = shap_values[1]  # Binary classification - positive class
                    class_name = "positive_class"
                else:
                    values = shap_values[0]  # Multi-class - first class
                    class_name = f"class_0"
            else:
                values = shap_values
                class_name = "prediction"
            
            # Convert to numpy array if needed
            if hasattr(values, 'values'):
                values = values.values
            values = np.array(values)
            
            # Calculate feature importance
            if values.ndim == 2:
                importance = np.abs(values).mean(axis=0)
            else:
                importance = np.abs(values)
            
            # Get top features
            top_indices = np.argsort(importance)[-5:][::-1]  # Top 5
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'samples_explained': X.shape[0],
                'class_explained': class_name,
                'top_features': []
            }
            
            for i, idx in enumerate(top_indices):
                if idx < len(self.feature_names):
                    summary['top_features'].append({
                        'rank': i + 1,
                        'feature': self.feature_names[idx],
                        'rule': self.rule_mapping.get(self.feature_names[idx], self.feature_names[idx]),
                        'avg_importance': float(importance[idx])
                    })
            
            self.logger.info("SHAP Explanation Summary:")
            self.logger.info(json.dumps(summary, indent=2))
            
        except Exception as e:
            self.logger.warning(f"Failed to log explanation summary: {e}")
    
    def get_feature_importance(self, X: Union[np.ndarray, pd.DataFrame, list], 
                             class_index: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get feature importance ranking for input data.
        
        Args:
            X: Input data to analyze
            class_index: For multi-class models, which class to analyze (None for auto-select)
            
        Returns:
            List of dicts with feature importance ranking
        """
        try:
            shap_values = self.explain(X, log_results=False)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                if class_index is not None and class_index < len(shap_values):
                    values = shap_values[class_index]
                elif len(shap_values) == 2:
                    values = shap_values[1]  # Binary - positive class
                else:
                    values = shap_values[0]  # Multi-class - first class
            else:
                values = shap_values
            
            # Convert to numpy array if needed
            if hasattr(values, 'values'):
                values = values.values
            values = np.array(values)
            
            # Calculate importance
            if values.ndim == 2:
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
                        'rule': str(rule_name),
                        'importance': float(importance[idx])
                    })
            
            return ranking
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return []
    
    def generate_summary_plot(self, X: Union[np.ndarray, pd.DataFrame, list], 
                            output_path: Optional[str] = None, 
                            top_features: int = 15) -> Optional[str]:
        """
        Generate a summary plot showing feature importance.
        
        Args:
            X: Input data to analyze
            output_path: Path to save the plot (None to display only)
            top_features: Number of top features to show
            
        Returns:
            Path to saved plot or None if display only
        """
        try:
            ranking = self.get_feature_importance(X)[:top_features]
            
            if not ranking:
                self.logger.warning("No feature importance data available")
                return None
            
            # Prepare data
            features = [item['rule'][:30] + "..." if len(item['rule']) > 30 
                       else item['rule'] for item in ranking]
            scores = [item['importance'] for item in ranking]
            
            # Create plot
            plt.figure(figsize=(10, max(6, len(features) * 0.4)))
            
            # Color gradient
            colors = plt.cm.Reds(np.linspace(0.8, 0.3, len(scores)))
            
            bars = plt.barh(range(len(features)), scores, color=colors)
            
            # Customize
            plt.yticks(range(len(features)), features)
            plt.xlabel('SHAP Importance Score')
            plt.title(f'Top {len(features)} Feature Importance Ranking')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                plt.text(bar.get_width() + max(scores) * 0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{score:.4f}', ha='left', va='center')
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Feature importance plot saved to {output_path}")
                return output_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating summary plot: {e}")
            return None
    
    def display_terminal_summary(self, X: Union[np.ndarray, pd.DataFrame, list], 
                               top_features: int = 10):
        """
        Display feature importance summary in terminal.
        
        Args:
            X: Input data to analyze
            top_features: Number of top features to display
        """
        try:
            # Get model predictions
            X_processed = self._prepare_input_data(X)
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(X_processed)
            
            # Get feature importance
            ranking = self.get_feature_importance(X)[:top_features]
            
            # Display header
            print("\n" + "=" * 70)
            print("SHAP MODEL EXPLANATION SUMMARY")
            print("=" * 70)
            
            # Model info
            print(f"\nModel: {type(self.model).__name__}")
            print(f"Samples analyzed: {len(X_processed)}")
            print(f"Features: {len(self.feature_names)}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Predictions summary
            print(f"\nPredictions:")
            unique_preds, counts = np.unique(predictions, return_counts=True)
            for pred, count in zip(unique_preds, counts):
                confidence = probabilities[predictions == pred].max(axis=1).mean()
                print(f"  {pred}: {count} samples (avg confidence: {confidence:.2%})")
            
            # Feature importance table
            if ranking:
                print(f"\nTop {len(ranking)} Most Important Features:")
                print("-" * 70)
                print(f"{'Rank':<4} {'Importance':<12} {'Feature/Rule':<50}")
                print("-" * 70)
                
                for item in ranking:
                    rule = item['rule']
                    if len(rule) > 45:
                        rule = rule[:42] + "..."
                    
                    # Visual bar
                    bar_length = min(20, int(item['importance'] * 100))
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    
                    print(f"{item['rank']:<4} {item['importance']:<12.4f} {rule:<50}")
                    print(f"     {bar}")
                
                print("-" * 70)
                print(f"Most critical feature: {ranking[0]['rule']}")
                print(f"Importance score: {ranking[0]['importance']:.4f}")
            
            print("=" * 70)
            
        except Exception as e:
            self.logger.error(f"Error displaying terminal summary: {e}")
            print(f"ERROR: {e}")
    
    def __str__(self) -> str:
        """String representation of the explainer."""
        return "SHAP Explainer (ready to use with explain() method)"
    
    def __repr__(self) -> str:
        """Detailed representation of the explainer."""
        return self.__str__()
