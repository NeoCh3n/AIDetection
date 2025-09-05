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
        background_data=background_data,
        instance_data=instance_data,
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
                background_data: Union[np.ndarray, pd.DataFrame], 
                instance_data: Union[np.ndarray, pd.DataFrame],
                feature_name_list: List[str], 
                output_dir: str,
                plot: bool = False,
                plot_in_terminal: bool = False, 
                summary_report: bool = False) -> Dict[str, Any]:
        """
        Explain a machine learning model using SHAP values for specific instances.
        
        Args:
            model: Trained ML model with predict() and predict_proba() methods
            background_data: Background data for SHAP baseline (representative sample)
            instance_data: Specific instance(s) to explain
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
            self._validate_inputs(model, background_data, instance_data, feature_name_list, output_dir)
            
            # Prepare data
            background_array = self._prepare_background_data(background_data)
            instance_array = self._prepare_instance_data(instance_data)
            feature_names = self._validate_feature_names(feature_name_list, background_array)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")
            
            # Initialize SHAP explainer with background data
            explainer = self._initialize_shap_explainer(model, background_array)
            
            # Calculate SHAP values for the specific instances
            self.logger.info(f"Calculating SHAP values for {instance_array.shape[0]} instance(s)")
            shap_values = explainer.shap_values(instance_array)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(shap_values, feature_names)
            
            # Build results
            results = {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'feature_names': feature_names,
                'instance_shape': instance_array.shape,
                'background_shape': background_array.shape,
                'samples_analyzed': instance_array.shape[0],
                'features_count': len(feature_names),
                'model_type': type(model).__name__,
                'timestamp': datetime.now().isoformat(),
                'output_files': {}
            }
            
            # Generate visualizations if requested
            if plot:
                plot_files = self._generate_plots(shap_values, feature_names, instance_array, output_dir)
                results['output_files'].update(plot_files)
            
            # Terminal visualization
            if plot_in_terminal:
                self._display_terminal_plots(shap_values, feature_names, model, instance_array)
            
            # Summary report
            if summary_report:
                report_path = self._generate_summary_report(results, model, background_array, instance_array, feature_names, output_dir)
                if report_path:
                    results['output_files']['summary_report'] = report_path
            
            self.logger.info("SHAP explanation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in SHAP explanation: {e}")
            raise
    
    def _validate_inputs(self, model, background_data, instance_data, feature_name_list, output_dir):
        """Validate all input parameters."""
        # Validate model
        if model is None:
            raise ValueError("Model cannot be None")
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a 'predict' method")
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model must have a 'predict_proba' method")
        
        # Validate background data
        if background_data is None:
            raise ValueError("Background data cannot be None")
        
        # Validate instance data
        if instance_data is None:
            raise ValueError("Instance data cannot be None")
        
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
                raise ValueError(f"Cannot convert background data to numpy array: {e}")
        
        # Validate shape
        if data_array.ndim != 2:
            raise ValueError(f"Background data must be 2D, got shape {data_array.shape}")
        
        if data_array.shape[0] == 0 or data_array.shape[1] == 0:
            raise ValueError(f"Background data cannot be empty, got shape {data_array.shape}")
        
        # Check for invalid values (NaN/inf) with compatibility fallback
        if self._has_nonfinite(data_array):
            self.logger.warning("Background data contains non-finite values (NaN, inf)")
        
        return data_array
    
    def _prepare_instance_data(self, data):
        """Prepare and validate instance data."""
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data.copy()
        else:
            try:
                data_array = np.array(data)
            except Exception as e:
                raise ValueError(f"Cannot convert instance data to numpy array: {e}")
        
        # Ensure 2D
        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)
        elif data_array.ndim != 2:
            raise ValueError(f"Instance data must be 1D or 2D, got shape {data_array.shape}")
        
        if data_array.shape[0] == 0 or data_array.shape[1] == 0:
            raise ValueError(f"Instance data cannot be empty, got shape {data_array.shape}")
        
        # Check for invalid values (NaN/inf) with compatibility fallback
        if self._has_nonfinite(data_array):
            raise ValueError("Instance data contains non-finite values (NaN, inf)")
        
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
    
    def _generate_plots(self, shap_values, feature_names, instance_data, output_dir):
        """Generate visualization plots and save to files."""
        plot_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Generate feature importance plot
            importance_path = os.path.join(output_dir, f"feature_importance_{timestamp}.png")
            self._create_importance_plot(shap_values, feature_names, importance_path)
            plot_files['importance_plot'] = importance_path
            
            # Generate SHAP heatmap
            heatmap_path = os.path.join(output_dir, f"shap_heatmap_{timestamp}.png")
            self._create_heatmap_plot(shap_values, feature_names, instance_data, heatmap_path)
            plot_files['heatmap'] = heatmap_path
            
            self.logger.info(f"Plots generated successfully in {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
        
        return plot_files
    
    def _create_importance_plot(self, shap_values, feature_names, output_path, top_features=15):
        """Create and save feature importance plot."""
        try:
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(shap_values, feature_names)
            
            if not feature_importance:
                self.logger.warning("No feature importance data available for plotting")
                return
            
            # Prepare data
            ranking = feature_importance[:top_features]
            features = [item['feature'] for item in ranking]
            scores = [item['importance'] for item in ranking]
            
            # Truncate long feature names
            features = [f[:30] + "..." if len(f) > 30 else f for f in features]
            
            # Create plot
            plt.figure(figsize=(10, max(6, len(features) * 0.4)))
            
            # Color gradient using a named colormap via get_cmap (Pylance-friendly)
            reds = cm.get_cmap('Reds')
            # Build gradient as ndarray; ensure ndarray type for Colormap.__call__
            gradient_vals = np.linspace(0.8, 0.3, len(scores), endpoint=True, dtype=float)
            colors = reds(np.array(gradient_vals, dtype=float))
            
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
            
        except Exception as e:
            self.logger.error(f"Error creating importance plot: {e}")
    
    def _create_heatmap_plot(self, shap_values, feature_names, instance_data, output_path, top_features=20):
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
            
            # Calculate feature importance for ordering
            feature_importance = self._calculate_feature_importance(shap_values, feature_names)
            n_features = min(top_features, len(feature_importance))
            
            # Get top feature indices
            top_feature_names = [item['feature'] for item in feature_importance[:n_features]]
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
            
            # Prepare heatmap data
            n_instances = min(values.shape[0], 10)  # Limit instances for readability
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
            plt.xticks(range(len(feature_labels)), feature_labels, rotation=45, ha='right')
            plt.yticks(range(len(instance_labels)), instance_labels)
            
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
    
    def _display_terminal_plots(self, shap_values, feature_names, model, instance_data):
        """Display visualizations in terminal."""
        try:
            # Get model predictions
            predictions = model.predict(instance_data)
            probabilities = model.predict_proba(instance_data)
            
            # Display header
            print("\n" + "=" * 70)
            print("SHAP MODEL EXPLANATION SUMMARY")
            print("=" * 70)
            
            # Model info
            print(f"\nModel: {type(model).__name__}")
            print(f"Instances analyzed: {len(instance_data)}")
            print(f"Features: {len(feature_names)}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Predictions summary
            print(f"\nPredictions:")
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                confidence = prob.max()
                pred_label = "Malicious" if pred == 1 else "Benign"
                print(f"  Instance {i+1}: {pred_label} (confidence: {confidence:.1%})")
            
            # Feature importance table
            feature_importance = self._calculate_feature_importance(shap_values, feature_names)
            top_features = feature_importance[:10]  # Top 10
            if top_features:
                print(f"\nTop {len(top_features)} Most Important Features:")
                print("-" * 70)
                print(f"{'Rank':<4} {'Importance':<12} {'Feature':<50}")
                print("-" * 70)
                
                for item in top_features:
                    feature = item['feature']
                    if len(feature) > 45:
                        feature = feature[:42] + "..."
                    
                    print(f"{item['rank']:<4} {item['importance']:<12.4f} {feature:<50}")
                
                print("-" * 70)
                print(f"Most critical feature: {top_features[0]['feature']}")
                print(f"Importance score: {top_features[0]['importance']:.4f}")
            
            print("=" * 70)
            
        except Exception as e:
            self.logger.error(f"Error displaying terminal plots: {e}")
            print(f"ERROR: {e}")
    
    def _generate_summary_report(self, results, model, background_data, instance_data, feature_names, output_dir):
        """Generate markdown summary report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_dir, f"shap_analysis_report_{timestamp}.md")
            
            # Get predictions
            predictions = model.predict(instance_data)
            probabilities = model.predict_proba(instance_data)
            
            # Generate report content
            report_lines = [
                "# SHAP Model Explanation Report",
                "",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Model Type:** {type(model).__name__}",
                f"**Instances Analyzed:** {results['samples_analyzed']}",
                f"**Total Features:** {results['features_count']}",
                f"**Background Data Shape:** {background_data.shape}",
                "",
                "## Instance Predictions",
                ""
            ]
            
            # Add prediction details
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                pred_label = "Malicious" if pred == 1 else "Benign"
                confidence = prob.max()
                report_lines.extend([
                    f"### Instance {i+1}",
                    f"- **Prediction:** {pred_label}",
                    f"- **Confidence:** {confidence:.1%}",
                    ""
                ])
            
            # Add feature importance table
            top_features = results['feature_importance'][:15]  # Top 15
            if top_features:
                report_lines.extend([
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
                f"- **Instance Data Shape:** {instance_data.shape}",
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

    def _has_nonfinite(self, data_array) -> bool:
        """Return True if the array contains any NaN or inf values.

        Uses numpy.isfinite when available; otherwise falls back to isnan/isinf or
        a safe element-wise check. Implemented this way to avoid static-analysis
        warnings in some environments while keeping runtime behavior robust.
        """
        try:
            arr = np.array(data_array, dtype=float)
        except Exception:
            # If coercion to float fails, proceed with best-effort checks
            arr = np.array(data_array)

        # Preferred path: np.isfinite
        try:
            isfinite = getattr(np, "isfinite", None)
            if isfinite is not None:
                result = isfinite(arr)
                return not bool(np.all(result))
        except Exception:
            pass

        # Fallback: combine isnan and isinf if available
        try:
            _isnan = getattr(np, "isnan", None)
            _isinf = getattr(np, "isinf", None)
            if _isnan is not None and _isinf is not None:
                mask_nan = _isnan(arr)
                mask_inf = _isinf(arr)
                return bool(np.any(mask_nan | mask_inf))
        except Exception:
            pass

        # Last resort: element-wise Python check
        try:
            import math
            it = np.nditer(arr, flags=["refs_ok"])
            for x in it:
                try:
                    v = float(x)
                except Exception:
                    # Skip non-numeric entries in best-effort mode
                    continue
                if math.isnan(v) or math.isinf(v):
                    return True
            return False
        except Exception:
            # If all else fails, assume no non-finite values
            return False
