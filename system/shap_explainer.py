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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotext as plt_terminal
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
        # Initialize logging FIRST so helper methods can use it safely
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.background_data = self._validate_background_data(background_data)
        self.feature_names = self._setup_feature_names(feature_names)
        self.rule_mapping = rule_mapping or self._load_rule_mapping()
        
        # Validate model compatibility
        self._validate_model()
        
        # Initialize SHAP TreeExplainer optimized for RandomForest
        self.explainer = shap.TreeExplainer(self.model, data=self.background_data)
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

    def explain(self, X, log_results=True, generate_plots=False, output_dir=None, show_terminal=False):
        """
        Generate SHAP explanations for input data.
        
        Args:
            X: Input data to explain (single instance or batch)
            log_results: Whether to log explanation results
            generate_plots: Whether to generate heat map and importance plots
            output_dir: Directory for saving plots (if generate_plots=True)
            show_terminal: Whether to display results in terminal using plotext
            
        Returns:
            SHAP values for the input data, and optionally plot paths
        """
        try:
            # Validate and prepare input
            X_processed = self._prepare_input(X)
            
            # Generate SHAP values
            shap_values = self.explainer.shap_values(X_processed)
            
            if log_results:
                self._log_explanation(X_processed, shap_values)
            
            # Show terminal summary if requested
            if show_terminal:
                self.display_terminal_summary(X)
            
            # Generate visualizations if requested
            if generate_plots:
                plot_results = self.generate_summary_visualization(X, output_dir)
                self.logger.info(f"Visualizations generated: {plot_results}")
                return shap_values, plot_results
            
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
            # Unwrap (shap_values, plot_results) if present
            if isinstance(shap_values, tuple):
                shap_values = shap_values[0]

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
            
            # Ensure numpy array (supports SHAP Explanation objects)
            if hasattr(values, 'values') and not isinstance(values, np.ndarray):
                values = values.values
            values = np.array(values)
            
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
                    # Safely resolve rule name as string
                    rule_name = self.rule_mapping.get(feature_name) or feature_name
                    if not isinstance(rule_name, str):
                        rule_name = str(rule_name)

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
            if isinstance(shap_values, tuple):
                shap_values = shap_values[0]
            if isinstance(shap_values, list) and len(shap_values) > 1:
                values = shap_values[1]  # Malicious class
            elif isinstance(shap_values, list):
                values = shap_values[0]
            else:
                values = shap_values
            
            # Ensure numpy array (supports SHAP Explanation objects)
            if hasattr(values, 'values') and not isinstance(values, np.ndarray):
                values = values.values
            values = np.array(values)

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
                    # Safely resolve rule name as string
                    rule_name = self.rule_mapping.get(feature_name) or feature_name
                    if not isinstance(rule_name, str):
                        rule_name = str(rule_name)

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

    def generate_heatmap(self, X, output_path=None, top_features=20, figsize=(12, 8)):
        """
        Generate a heat map visualization of SHAP feature importance.
        
        Args:
            X: Input data to analyze
            output_path: Optional path to save the heat map image
            top_features: Number of top features to display (default: 20)
            figsize: Figure size tuple (width, height)
            
        Returns:
            String path to the saved image or None if display only
        """
        try:
            # Get SHAP values and feature importance
            shap_values = self.explain(X, log_results=False)
            ranking = self.get_feature_importance(X)
            
            # Handle multi-class output
            if isinstance(shap_values, tuple):
                shap_values = shap_values[0]
            if isinstance(shap_values, list) and len(shap_values) > 1:
                values = shap_values[1]  # Malicious class
                class_name = "Malicious Class"
            elif isinstance(shap_values, list):
                values = shap_values[0]
                class_name = "Class 0"
            else:
                values = shap_values
                class_name = "Prediction"
            
            # Ensure numpy array (supports SHAP Explanation objects)
            if hasattr(values, 'values') and not isinstance(values, np.ndarray):
                values = values.values
            values = np.array(values)
            
            # Prepare data for heat map
            X_processed = self._prepare_input(X)
            n_instances = min(X_processed.shape[0], 10)  # Limit to 10 instances for readability
            n_features = min(top_features, len(ranking))
            
            # Get top feature indices
            top_feature_names = [item['feature'] for item in ranking[:n_features]]
            top_feature_indices = []
            
            for fname in top_feature_names:
                try:
                    idx = self.feature_names.index(fname)
                    top_feature_indices.append(idx)
                except ValueError:
                    continue
            
            if not top_feature_indices:
                self.logger.warning("No valid feature indices found for heat map")
                return None
            
            # Create heat map data
            if values.ndim > 1:
                heatmap_data = values[:n_instances, top_feature_indices]
                instance_labels = [f"Instance {i+1}" for i in range(n_instances)]
            else:
                heatmap_data = values[top_feature_indices].reshape(1, -1)
                instance_labels = ["Instance 1"]
            
            # Create rule labels for features
            rule_labels = []
            for fname in top_feature_names[:len(top_feature_indices)]:
                # Safely resolve rule name and ensure it is a string
                rule_name = self.rule_mapping.get(fname) or fname
                if not isinstance(rule_name, str):
                    rule_name = str(rule_name)
                # Truncate long rule names for display
                if len(rule_name) > 25:
                    rule_name = rule_name[:22] + "..."
                rule_labels.append(rule_name)
            
            # Create the heat map (pure matplotlib fallback; avoids seaborn dependency)
            plt.figure(figsize=figsize)

            # Symmetric color scale around 0 for SHAP values
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

            # Add colorbar with label
            cbar = plt.colorbar(img)
            cbar.set_label('SHAP Value (Impact on Prediction)')

            # Set tick labels
            plt.xticks(ticks=np.arange(len(rule_labels)), labels=rule_labels, rotation=45, ha='right')
            plt.yticks(ticks=np.arange(len(instance_labels)), labels=instance_labels)

            # Annotate values in each cell
            h, w = heatmap_data.shape
            for i in range(h):
                for j in range(w):
                    val = heatmap_data[i, j]
                    # Choose text color based on background intensity for readability
                    color = 'black' if abs(val) < (0.6 * absmax) else 'white'
                    plt.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=8, color=color)

            # Axes labels and title
            plt.title(
                f'SHAP Feature Importance Heat Map - {class_name}\n'
                f'Top {n_features} Contributing Features',
                fontsize=14,
                fontweight='bold'
            )
            plt.xlabel('Security Rules / Features', fontsize=12)
            plt.ylabel('Data Instances', fontsize=12)

            plt.tight_layout()
            
            # Save or display
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"SHAP heat map saved to {output_path}")
                return output_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating heat map: {str(e)}")
            return None

    def generate_feature_importance_plot(self, X, output_path=None, top_features=15, figsize=(10, 8)):
        """
        Generate a horizontal bar plot of feature importance.
        
        Args:
            X: Input data to analyze
            output_path: Optional path to save the plot
            top_features: Number of top features to display
            figsize: Figure size tuple (width, height)
            
        Returns:
            String path to the saved image or None if display only
        """
        try:
            ranking = self.get_feature_importance(X)[:top_features]
            
            if not ranking:
                self.logger.warning("No feature importance data available for plotting")
                return None
            
            # Prepare data
            features = [item['rule'] for item in ranking]
            importance_scores = [item['importance'] for item in ranking]
            
            # Truncate long feature names
            features = [f[:30] + "..." if len(f) > 30 else f for f in features]
            
            # Create horizontal bar plot
            plt.figure(figsize=figsize)
            
            # Create color map based on importance
            # Ensure the input to the colormap is a concrete ndarray to satisfy type checkers
            cmap = cm.get_cmap('Reds')
            linspace_vals = np.linspace(
                0.9,
                0.3,
                num=len(importance_scores),
                endpoint=True,
                retstep=False,
                dtype=float,
            )
            # Use np.array for broad stub compatibility (Pylance/old numpy stubs)
            colors = cmap(np.array(linspace_vals, dtype=float))
            
            bars = plt.barh(range(len(features)), importance_scores, color=colors)
            
            # Customize plot
            plt.yticks(range(len(features)), features)
            plt.xlabel('SHAP Importance Score', fontsize=12)
            plt.title('Top Security Rules Contributing to Threat Detection', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()  # Highest importance at top
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + max(importance_scores) * 0.01, bar.get_y() + bar.get_height()/2,
                        f'{importance_scores[i]:.4f}', ha='left', va='center', fontsize=9)
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save or display
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Feature importance plot saved to {output_path}")
                return output_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating feature importance plot: {str(e)}")
            return None

    def generate_summary_visualization(self, X, output_dir=None, prefix="shap_analysis"):
        """
        Generate a comprehensive set of visualizations including heat map and importance plot.
        
        Args:
            X: Input data to analyze
            output_dir: Directory to save visualizations (uses current dir if None)
            prefix: Prefix for output filenames
            
        Returns:
            Dict with paths to generated visualizations
        """
        try:
            if output_dir is None:
                output_dir = os.getcwd()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results = {
                'heatmap': None,
                'importance_plot': None,
                'timestamp': timestamp
            }
            
            # Generate heat map
            heatmap_path = os.path.join(output_dir, f"{prefix}_heatmap_{timestamp}.png")
            results['heatmap'] = self.generate_heatmap(X, heatmap_path)
            
            # Generate importance plot
            importance_path = os.path.join(output_dir, f"{prefix}_importance_{timestamp}.png")
            results['importance_plot'] = self.generate_feature_importance_plot(X, importance_path)
            
            self.logger.info(f"Summary visualizations generated in {output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating summary visualizations: {str(e)}")
            return {'error': str(e)}

    def display_terminal_heatmap(self, X, top_features=15, max_instances=8):
        """
        Display a heat map in the terminal using plotext.
        
        Args:
            X: Input data to analyze
            top_features: Number of top features to display
            max_instances: Maximum number of instances to show
        """
        try:
            # Get SHAP values and feature importance
            shap_values = self.explain(X, log_results=False)
            ranking = self.get_feature_importance(X)
            
            # Handle multi-class output
            if isinstance(shap_values, tuple):
                shap_values = shap_values[0]
            if isinstance(shap_values, list) and len(shap_values) > 1:
                values = shap_values[1]  # Malicious class
                class_name = "Malicious Class"
            elif isinstance(shap_values, list):
                values = shap_values[0]
                class_name = "Class 0"
            else:
                values = shap_values
                class_name = "Prediction"
            
            # Ensure numpy array (supports SHAP Explanation objects)
            if hasattr(values, 'values') and not isinstance(values, np.ndarray):
                values = values.values
            values = np.array(values)
            
            # Prepare data
            X_processed = self._prepare_input(X)
            n_instances = min(X_processed.shape[0], max_instances)
            n_features = min(top_features, len(ranking))
            
            # Get top feature names and indices
            top_feature_names = [item['feature'] for item in ranking[:n_features]]
            top_feature_indices = []
            
            for fname in top_feature_names:
                try:
                    idx = self.feature_names.index(fname)
                    top_feature_indices.append(idx)
                except ValueError:
                    continue
            
            if not top_feature_indices:
                print("ERROR: No valid features found for terminal heat map")
                return
            
            # Prepare heat map data
            if values.ndim > 1:
                heatmap_data = values[:n_instances, top_feature_indices]
            else:
                heatmap_data = values[top_feature_indices].reshape(1, -1)
                n_instances = 1
            
            # Create rule labels (truncated for terminal)
            rule_labels = []
            for fname in top_feature_names[:len(top_feature_indices)]:
                # Safely resolve rule name and ensure it is a string
                rule_name = self.rule_mapping.get(fname) or fname
                if not isinstance(rule_name, str):
                    rule_name = str(rule_name)
                # Truncate for terminal display
                if len(rule_name) > 20:
                    rule_name = rule_name[:17] + "..."
                rule_labels.append(rule_name)
            
            # Display header
            print("\n" + "="*80)
            print(f"SHAP HEAT MAP - {class_name}")
            print("="*80)
            print(f"Top {n_features} Features | {n_instances} Instance(s)")
            print("RED/High: Positive SHAP (contributes to malicious)")
            print("BLUE/Low: Negative SHAP (contributes to benign)")
            print("-"*80)
            
            # Display heat map using plotext
            plt_terminal.clear_figure()
            plt_terminal.theme('dark')
            
            # Create a simple text-based heat map
            print("\nFeature Importance Heat Map:")
            print("-" * 60)
            
            # Header row
            header = "Instance".ljust(12)
            for i, label in enumerate(rule_labels):
                if i < 5:  # Limit to 5 features for terminal width
                    header += f"{label[:8]:>10}"
            print(header)
            print("-" * 60)
            
            # Data rows
            for i in range(n_instances):
                row = f"Inst_{i+1}".ljust(12)
                for j in range(min(5, len(top_feature_indices))):
                    value = heatmap_data[i, j] if heatmap_data.ndim > 1 else heatmap_data[j]
                    
                    # Color coding for terminal
                    if value > 0.01:
                        color_symbol = "[+]"
                    elif value < -0.01:
                        color_symbol = "[-]"
                    else:
                        color_symbol = "[0]"
                    
                    row += f"{value:>8.3f}{color_symbol}"
                print(row)
            
            print("-" * 60)
            print("Legend: [+] High Impact | [0] Neutral | [-] Low Impact")
            print("="*80)
            
        except Exception as e:
            self.logger.error(f"Error displaying terminal heat map: {str(e)}")
            print(f"ERROR: Error displaying terminal heat map: {str(e)}")

    def display_terminal_importance_chart(self, X, top_features=10):
        """
        Display feature importance as a horizontal bar chart in terminal using plotext.
        
        Args:
            X: Input data to analyze
            top_features: Number of top features to display
        """
        try:
            ranking = self.get_feature_importance(X)[:top_features]
            
            if not ranking:
                print("ERROR: No feature importance data available")
                return
            
            # Prepare data for plotext
            features = []
            importance_scores = []
            
            for item in ranking:
                rule_name = item['rule']
                # Truncate long names for terminal
                if len(rule_name) > 25:
                    rule_name = rule_name[:22] + "..."
                features.append(rule_name)
                importance_scores.append(item['importance'])
            
            # Display using plotext
            plt_terminal.clear_figure()
            plt_terminal.simple_bar(
                features, 
                importance_scores, 
                width=60,
                title="Top Security Rules - Feature Importance"
            )
            plt_terminal.show()
            
            # Also display as text table
            print("\n" + "="*70)
            print("FEATURE IMPORTANCE RANKING")
            print("="*70)
            print(f"{'Rank':<4} {'Importance':<12} {'Security Rule':<50}")
            print("-"*70)
            
            for item in ranking:
                rank = item['rank']
                importance = item['importance']
                rule = item['rule']
                
                # Truncate rule name if too long
                if len(rule) > 45:
                    rule = rule[:42] + "..."
                
                # Visual importance indicator
                bar_length = int(importance * 20)  # Scale to 20 chars max
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                print(f"{rank:<4} {importance:<12.4f} {rule:<50}")
                print(f"     {bar}")
            
            print("="*70)
            
        except Exception as e:
            self.logger.error(f"Error displaying terminal importance chart: {str(e)}")
            print(f"ERROR: Error displaying terminal chart: {str(e)}")

    def display_terminal_summary(self, X, top_features=8):
        """
        Display a comprehensive terminal summary with both heat map and importance chart.
        
        Args:
            X: Input data to analyze
            top_features: Number of top features to display
        """
        try:
            # Get predictions first
            X_processed = self._prepare_input(X)
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(X_processed)
            
            # Display header
            print("\n" + "=" * 60)
            print("SHAP THREAT DETECTION ANALYSIS")
            print("=" * 60)
            
            # Show predictions
            print(f"\nANALYSIS SUMMARY:")
            print(f"   Instances Analyzed: {len(X_processed)}")
            print(f"   Total Features: {len(self.feature_names)}")
            print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show predictions for each instance
            print(f"\nMODEL PREDICTIONS:")
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                confidence = prob.max()
                status_icon = "[THREAT]" if pred == 1 or pred == "malicious" else "[BENIGN]"
                print(f"   Instance {i+1}: {status_icon} {pred} (confidence: {confidence:.1%})")
            
            # Display terminal heat map
            self.display_terminal_heatmap(X, top_features=top_features, max_instances=5)
            
            # Display importance chart
            print(f"\nFEATURE IMPORTANCE ANALYSIS:")
            self.display_terminal_importance_chart(X, top_features=top_features)
            
            # Show key insights
            ranking = self.get_feature_importance(X)
            if ranking:
                top_rule = ranking[0]
                print(f"\nKEY INSIGHTS:")
                print(f"   Most Critical Rule: {top_rule['rule']}")
                print(f"   Importance Score: {top_rule['importance']:.4f}")
                print(f"   Detection Confidence: {probabilities.max():.1%}")
                
                # Show threat level
                max_confidence = probabilities.max()
                if max_confidence > 0.9:
                    threat_level = "HIGH"
                elif max_confidence > 0.7:
                    threat_level = "MEDIUM"
                else:
                    threat_level = "LOW"
                print(f"   Threat Level: {threat_level}")
            
            print("\n" + "=" * 60)
            print("Terminal analysis complete!")
            print("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Error displaying terminal summary: {str(e)}")
            print(f"ERROR: Error displaying terminal summary: {str(e)}")

    def generate_markdown_report(self, X, output_path=None, include_visualizations=True):
        """
        Generate a detailed markdown report for the SHAP explanation.
        
        Args:
            X: Input data to analyze
            output_path: Optional path to save the report
            include_visualizations: Whether to generate and include visualization plots
            
        Returns:
            String containing the markdown report
        """
        try:
            # Get predictions and explanations
            X_processed = self._prepare_input(X)
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(X_processed)
            ranking = self.get_feature_importance(X)
            
            # Generate visualizations if requested
            plot_results = None
            if include_visualizations and output_path:
                output_dir = os.path.dirname(output_path) if output_path else None
                plot_results = self.generate_summary_visualization(X, output_dir)
            
            # Generate report
            report_lines = [
                "# SHAP Threat Detection Explanation Report",
                "",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Instances Analyzed:** {len(X_processed)}",
                f"**Model Type:** {type(self.model).__name__}",
                f"**Total Features:** {len(self.feature_names)}",
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
            
            # Add visualizations section if available
            if plot_results and not plot_results.get('error'):
                report_lines.extend([
                    "## Visualizations",
                    ""
                ])
                
                if plot_results.get('heatmap'):
                    heatmap_filename = os.path.basename(plot_results['heatmap'])
                    report_lines.extend([
                        "### SHAP Feature Importance Heat Map",
                        f"![SHAP Heat Map]({heatmap_filename})",
                        "",
                        "The heat map shows how each security rule contributes to the threat detection for each analyzed instance. "
                        "Red colors indicate positive contribution to malicious prediction, while blue indicates negative contribution.",
                        ""
                    ])
                
                if plot_results.get('importance_plot'):
                    importance_filename = os.path.basename(plot_results['importance_plot'])
                    report_lines.extend([
                        "### Feature Importance Ranking",
                        f"![Feature Importance]({importance_filename})",
                        "",
                        "This chart ranks the security rules by their overall importance in the threat detection decision.",
                        ""
                    ])
            
            # Add top features table
            report_lines.extend([
                "## Top Contributing Security Rules",
                "",
                "| Rank | Feature | Rule Description | Importance Score |",
                "|------|---------|------------------|------------------|"
            ])
            
            for item in ranking[:15]:  # Top 15 features
                report_lines.append(
                    f"| {item['rank']} | {item['feature']} | {item['rule']} | {item['importance']:.6f} |"
                )
            
            # Add detailed analysis
            if ranking:
                top_rule = ranking[0]
                report_lines.extend([
                    "",
                    "## Key Findings",
                    "",
                    f"- **Most Critical Rule:** {top_rule['rule']} (importance: {top_rule['importance']:.4f})",
                    f"- **Analysis Scope:** Top {min(15, len(ranking))} out of {len(ranking)} total features",
                    f"- **Detection Confidence:** {probabilities.max():.1%}",
                    ""
                ])
            
            report_lines.extend([
                "## Analysis Notes",
                "",
                "- **Importance Scores:** Higher values indicate features that contributed more to the malicious prediction",
                "- **Security Rules:** Each feature represents a security rule or detection pattern",
                "- **Threat Analysis:** Review the top contributing rules for threat hunting and rule refinement",
                "- **Model Explainability:** SHAP values provide local explanations for individual predictions",
                ""
            ])
            
            # Add technical details
            report_lines.extend([
                "## Technical Details",
                "",
                f"- **SHAP Library Version:** {shap.__version__ if hasattr(shap, '__version__') else 'Unknown'}",
                f"- **Background Data Size:** {len(self.background_data)} instances",
                f"- **Feature Count:** {len(self.feature_names)}",
                f"- **Analysis Timestamp:** {datetime.now().isoformat()}",
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
