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
        summary_report=True,
        frequent_path_mining={
            'enabled': True,
            'max_trees': 50,
            'itemset_sizes': [2, 3],
            'top_k': 20,
            'min_support': 0.05
        }
    )
"""

import shap
import json
import os
import csv
import logging
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotext as plt_terminal
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple, Set, cast
from pathlib import Path

class Explainer:
    """
    SHAP Explainer for machine learning models.
    
    Provides explainability for model predictions using SHAP values,
    with support for various model types and comprehensive analysis features.
    
    The class is initialized without parameters and provides a main explain()
    function that accepts all necessary inputs.
    """
    
    def __init__(self,
                 model: Optional[object] = None,
                 background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 feature_names: Optional[List[str]] = None,
                 rule_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize SHAP explainer; optionally preconfigure defaults.

        Args:
            model: Optional trained model to reuse in explain() calls
            background_data: Optional background data for SHAP baseline
            feature_names: Optional list of feature names
            rule_mapping: Optional mapping feature_name -> human-friendly rule name
        """
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("SHAP Explainer initialized and ready to use")
        # Optional defaults used by convenience-style invocations
        self._default_model = model
        self._default_background = background_data
        self._default_feature_names = feature_names
        # Internal caches
        self._rule_name_map: Optional[Dict[int, str]] = None
        self._feature_name_map: Optional[Dict[str, str]] = None
        # Accept external mapping if provided: coerce to int-keyed map when possible
        if isinstance(rule_mapping, dict) and rule_mapping:
            try:
                # If keys are convertible to int, store as rule_id map
                int_map: Dict[int, str] = {}
                convertible = True
                for k, v in rule_mapping.items():
                    try:
                        int_map[int(str(k).replace('rule_', '').replace('Security_Rule_', '').lstrip('0') or '0') if str(k).strip() else int(k)] = str(v)
                    except Exception:
                        convertible = False
                        break
                if convertible:
                    self._rule_name_map = int_map
                else:
                    # Keep as feature-name map for potential future use
                    self._feature_name_map = {str(k): str(v) for k, v in rule_mapping.items()}
            except Exception:
                # Fallback: ignore invalid external mapping
                self._rule_name_map = None
                self._feature_name_map = None
        # Lazy cache for BOC rule IDs discovered from rule CSVs
        self._boc_rule_ids: Optional[Set[int]] = None
    
    def explain(self,
                *args,
                model: Optional[object] = None,
                background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                instance_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                feature_name_list: Optional[List[str]] = None,
                output_dir: Optional[str] = None,
                persist_outputs: bool = True,
                plot: bool = False,
                plot_in_terminal: bool = False,
                summary_report: bool = False,
                frequent_path_mining: Optional[Dict[str, Any]] = None,
                **legacy_kwargs) -> Dict[str, Any]:
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
            # Support legacy positional usage where only instance data is provided
            if args:
                if len(args) == 1 and instance_data is None and model is None and background_data is None and feature_name_list is None:
                    instance_data = args[0]
                else:
                    # Allow positional mapping for (model, background, instance, features, output_dir)
                    positional = list(args)
                    if model is None and positional:
                        model = positional.pop(0)
                    if background_data is None and positional:
                        background_data = positional.pop(0)
                    if instance_data is None and positional:
                        instance_data = positional.pop(0)
                    if feature_name_list is None and positional:
                        feature_name_list = positional.pop(0)
                    if output_dir is None and positional:
                        output_dir = positional.pop(0)

            # Backwards compatibility for legacy keyword arguments
            if 'generate_plots' in legacy_kwargs:
                plot = plot or bool(legacy_kwargs.pop('generate_plots'))
            if 'show_terminal' in legacy_kwargs:
                plot_in_terminal = plot_in_terminal or bool(legacy_kwargs.pop('show_terminal'))
            if 'log_results' in legacy_kwargs:
                summary_report = summary_report or bool(legacy_kwargs.pop('log_results'))

            if legacy_kwargs:
                # Keep a trace without breaking execution
                self.logger.debug(
                    "Ignoring unsupported legacy kwargs: %s",
                    sorted(legacy_kwargs.keys())
                )

            # Use defaults from initialization when explicit parameters are missing
            if instance_data is None and isinstance(model, (np.ndarray, pd.DataFrame, list, tuple)) and self._default_model is not None:
                # Heuristic: legacy call provided only instance data; shift values
                instance_data = model
                model = None

            model_to_use = model if model is not None else self._default_model
            background_to_use = background_data if background_data is not None else self._default_background
            feature_names_input = feature_name_list if feature_name_list is not None else self._default_feature_names
            if persist_outputs:
                output_dir_to_use = output_dir if output_dir else os.path.join('.', 'shap_output')
            else:
                output_dir_to_use = output_dir if output_dir else os.path.join('.', 'shap_output')
                # When not persisting, disable file-producing options defensively
                plot = False
                summary_report = False

            if model_to_use is None:
                raise ValueError("Model cannot be None and no default model was provided during initialization")
            if background_to_use is None:
                raise ValueError("Background data is required for SHAP explanations")
            if instance_data is None:
                raise ValueError("Instance data cannot be None")
            if feature_names_input is None:
                raise ValueError("Feature names must be provided for explanations")

            # Validate inputs
            self._validate_inputs(model_to_use, background_to_use, instance_data, feature_names_input, output_dir_to_use, persist_outputs)
            
            # Prepare data
            background_array = self._prepare_background_data(background_to_use)
            instance_array = self._prepare_instance_data(instance_data)
            feature_names = self._validate_feature_names(feature_names_input, background_array)
            
            # Create output directory
            if persist_outputs:
                os.makedirs(output_dir_to_use, exist_ok=True)
                self.logger.info(f"Created output directory: {output_dir_to_use}")
            else:
                self.logger.info("persist_outputs=False; skipping SHAP output directory creation")
            
            # Initialize SHAP explainer with background data
            explainer = self._initialize_shap_explainer(model_to_use, background_array)
            
            # Calculate SHAP values for the specific instances
            self.logger.info(f"Calculating SHAP values for {instance_array.shape[0]} instance(s)")
            shap_values = explainer.shap_values(instance_array)
            
            # Calculate feature importance (full ranking)
            feature_importance_full = self._calculate_feature_importance(shap_values, feature_names)

            # Prioritize BOC-related rules and keep only top 5 for output
            feature_importance = self._prioritize_boc_and_top_k(
                feature_importance_full,
                feature_names,
                top_k=5
            )

            # Optional: Frequent Path Mining over RandomForest decision paths
            fpm_results = None
            try:
                fpm_results = self._maybe_mine_frequent_paths(model_to_use, feature_names, frequent_path_mining)
            except Exception as e:
                self.logger.warning(f"Frequent path mining skipped due to error: {e}")
            
            # Build results
            results = {
                'shap_values': shap_values,
                # Only output top-5 rules; BOC-related first by priority
                'feature_importance': feature_importance,
                'feature_names': feature_names,
                'instance_shape': instance_array.shape,
                'background_shape': background_array.shape,
                'samples_analyzed': instance_array.shape[0],
                'features_count': len(feature_names),
                'model_type': type(model_to_use).__name__,
                'timestamp': datetime.now().isoformat(),
                'output_files': {}
            }
            if fpm_results is not None:
                results['frequent_paths'] = fpm_results
            
            # Generate visualizations if requested
            if plot and persist_outputs:
                plot_files = self._generate_plots(shap_values, feature_names, instance_array, output_dir_to_use)
                results['output_files'].update(plot_files)
            
            # Terminal visualization
            if plot_in_terminal:
                self._display_terminal_plots(shap_values, feature_names, model_to_use, instance_array)
            
            # Summary report
            if summary_report and persist_outputs:
                report_path = self._generate_summary_report(results, model_to_use, background_array, instance_array, feature_names, output_dir_to_use)
                if report_path:
                    results['output_files']['summary_report'] = report_path
            
            self.logger.info("SHAP explanation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in SHAP explanation: {e}")
            raise

    def _maybe_mine_frequent_paths(self, model, feature_names: List[str], options: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Optionally mine frequent feature combinations from RandomForest decision paths.

        Controls via 'options' dict:
        - enabled (bool): turn mining on/off
        - max_trees (int): analyze at most this many trees (None=all)
        - max_depth (int): limit path depth considered (None=full path)
        - itemset_sizes (List[int]): sizes of combinations to count (e.g., [2,3])
        - top_k (int): number of top itemsets to return
        - min_support (float): minimum support threshold (0..1)
        """
        try:
            if not isinstance(options, dict) or not options.get('enabled', False):
                return None

            estimators = getattr(model, 'estimators_', None)
            if estimators is None or not hasattr(model, 'n_features_in_'):
                return None

            # Defaults
            max_trees = options.get('max_trees', 100)
            max_depth = options.get('max_depth', None)
            itemset_sizes = options.get('itemset_sizes', [2])
            top_k = int(options.get('top_k', 20))
            min_support = float(options.get('min_support', 0.01))

            # Select trees
            if isinstance(max_trees, int) and max_trees > 0:
                trees = estimators[:max_trees]
            else:
                trees = estimators

            from itertools import combinations
            from collections import Counter

            path_counter = Counter()
            total_paths = 0

            for est in trees:
                tree_ = getattr(est, 'tree_', None)
                if tree_ is None:
                    continue
                children_left = tree_.children_left
                children_right = tree_.children_right
                features = tree_.feature

                # DFS: (node_id, depth, features_on_path)
                stack = [(0, 0, [])]
                while stack:
                    node_id, depth, feats = stack.pop()

                    is_leaf = (children_left[node_id] == -1 and children_right[node_id] == -1)
                    if is_leaf:
                        feat_set = sorted(set([f for f in feats if f is not None and f >= 0]))
                        if feat_set:
                            for k in itemset_sizes:
                                if k <= 0:
                                    continue
                                if len(feat_set) >= k:
                                    for combo in combinations(feat_set, k):
                                        path_counter[combo] += 1
                            total_paths += 1
                        continue

                    # Depth pruning
                    if isinstance(max_depth, int) and max_depth >= 0 and depth >= max_depth:
                        feat_set = sorted(set([f for f in feats if f is not None and f >= 0]))
                        if feat_set:
                            for k in itemset_sizes:
                                if k <= 0:
                                    continue
                                if len(feat_set) >= k:
                                    for combo in combinations(feat_set, k):
                                        path_counter[combo] += 1
                            total_paths += 1
                        continue

                    # Descend
                    feat_idx = features[node_id]
                    next_feats = feats
                    if feat_idx is not None and int(feat_idx) >= 0:
                        next_feats = feats + [int(feat_idx)]
                    left = children_left[node_id]
                    right = children_right[node_id]
                    if left != -1:
                        stack.append((left, depth + 1, next_feats))
                    if right != -1:
                        stack.append((right, depth + 1, next_feats))

            # Build results
            items = []
            if total_paths > 0 and path_counter:
                for combo, count in path_counter.items():
                    support = float(count) / float(total_paths)
                    if support >= min_support:
                        combo_names = []
                        combo_rule_ids = []
                        for idx in combo:
                            name = feature_names[idx] if 0 <= idx < len(feature_names) else f"feature_{idx}"
                            combo_names.append(name)
                            rid = None
                            if isinstance(name, str) and name.startswith('rule_'):
                                try:
                                    rid = int(name.replace('rule_', ''))
                                except Exception:
                                    rid = None
                            combo_rule_ids.append(rid)
                        items.append({
                            'features': combo_names,
                            'feature_indices': list(combo),
                            'rule_ids': combo_rule_ids,
                            'count': int(count),
                            'support': support
                        })

                # Sort and truncate
                items.sort(key=lambda x: (-x['count'], -x['support']))
                if top_k and top_k > 0:
                    items = items[:top_k]

            return {
                'parameters': {
                    'enabled': True,
                    'max_trees': max_trees,
                    'max_depth': max_depth,
                    'itemset_sizes': itemset_sizes,
                    'top_k': top_k,
                    'min_support': min_support,
                    'total_paths': total_paths
                },
                'top_itemsets': items
            }
        except Exception:
            return None
    
    def _validate_inputs(self, model, background_data, instance_data, feature_name_list, output_dir, persist_outputs: bool):
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
        
        # Validate output directory when persistence is enabled
        if persist_outputs:
            if not output_dir or not isinstance(output_dir, str):
                raise ValueError("output_dir must be a non-empty string when persist_outputs is True")
        
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
            prioritized_features = self._prioritize_boc_and_top_k(feature_importance, feature_names, top_k=5)

            if prioritized_features:
                print(f"\nTop {len(prioritized_features)} features (BOC prioritized):")
                print("-" * 70)
                print(f"{'Rank':<4} {'Importance':<12} {'Feature (Rule Name)':<50}")
                print("-" * 70)

                # Build reusable rule name map once for terminal display
                rule_name_map = {}
                try:
                    rule_name_map = self._get_rule_name_map()
                except Exception:
                    rule_name_map = {}

                for item in prioritized_features:
                    raw_feature = item.get('feature', '')
                    display_feature = raw_feature

                    # Prefer explicit rule_name in the item if already present
                    rule_name = item.get('rule_name')

                    # Otherwise, look up by rule_ prefix
                    if not rule_name and isinstance(raw_feature, str) and raw_feature.startswith('rule_'):
                        try:
                            rid = int(raw_feature.split('_', 1)[1])
                            rule_name = rule_name_map.get(rid)
                        except Exception:
                            rule_name = None

                    if rule_name:
                        display_feature = f"{raw_feature} | {rule_name}"

                    if isinstance(display_feature, str) and len(display_feature) > 45:
                        display_feature = display_feature[:42] + "..."

                    print(f"{item.get('rank', ''):<4} {item.get('importance', 0.0):<12.4f} {display_feature:<50}")

                print("-" * 70)

                most_critical_feature = prioritized_features[0].get('feature', '')
                most_critical_name = prioritized_features[0].get('rule_name')
                if not most_critical_name and isinstance(most_critical_feature, str) and most_critical_feature.startswith('rule_'):
                    try:
                        rid = int(most_critical_feature.split('_', 1)[1])
                        most_critical_name = rule_name_map.get(rid)
                    except Exception:
                        most_critical_name = None

                if most_critical_name:
                    print(f"Most critical feature: {most_critical_feature} ({most_critical_name})")
                else:
                    print(f"Most critical feature: {most_critical_feature}")
                print(f"Importance score: {prioritized_features[0].get('importance', 0.0):.4f}")
            else:
                print("\nNo non-zero feature importances were available for display.")
            
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

    # ------------------------
    # BOC prioritization utils
    # ------------------------
    def _extract_rule_id(self, feature_name: str) -> Optional[int]:
        """Extract integer rule_id from a feature name like 'rule_100392'."""
        try:
            if isinstance(feature_name, str) and feature_name.startswith('rule_'):
                rid = int(feature_name.replace('rule_', ''))
                return rid
        except Exception:
            pass
        return None

    def _get_rule_name_map(self) -> Dict[int, str]:
        """Build and cache a mapping of rule_id -> rule_name from Qradar_rule CSVs."""
        if isinstance(self._rule_name_map, dict):
            # Ensure int->str mapping
            try:
                # Coerce keys to int if needed (defensive)
                coerced: Dict[int, str] = {}
                for k, v in self._rule_name_map.items():
                    coerced[int(k)] = str(v)
                self._rule_name_map = coerced
            except Exception:
                # On failure, reset and rebuild from CSVs
                self._rule_name_map = None
            else:
                return self._rule_name_map

        name_map: Dict[int, str] = {}
        try:
            sys_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(sys_dir, '..'))

            # Load production mapping first for canonical names
            mapping_path = os.path.join(project_root, 'shared_utils', 'uat_to_prod_mapping.csv')
            if os.path.isfile(mapping_path):
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            if not row:
                                continue
                            prod_value = row.get('prod_rule_id')
                            name_value = row.get('rule_name') or row.get('prod_rule_name')
                            try:
                                prod_rule_id = int(str(prod_value).strip()) if prod_value is not None else None
                            except (TypeError, ValueError):
                                continue

                            if prod_rule_id is None or name_value is None:
                                continue

                            rule_name = str(name_value).strip()
                            if not rule_name:
                                continue

                            name_map[prod_rule_id] = rule_name
                except Exception:
                    # Ignore mapping load errors; fall back to other sources
                    pass

            qradar_dir = os.path.join(project_root, 'Qradar_rule')
            if os.path.isdir(qradar_dir):
                for fname in os.listdir(qradar_dir):
                    if not fname.lower().endswith('.csv'):
                        continue
                    fpath = os.path.join(qradar_dir, fname)
                    try:
                        df_rules = pd.read_csv(fpath)
                    except Exception:
                        continue
                    if 'id' not in df_rules.columns or 'name' not in df_rules.columns:
                        continue
                    for rid_val, nm in zip(df_rules['id'], df_rules['name']):
                        try:
                            rid_int = int(rid_val)
                            if rid_int not in name_map:
                                name_map[rid_int] = str(nm) if nm is not None else ''
                        except Exception:
                            continue

            self._rule_name_map = name_map
            return self._rule_name_map
        except Exception:
            self._rule_name_map = {}
            return self._rule_name_map

    def _load_boc_rule_ids(self) -> Set[int]:
        """Return cached BOC rule IDs (names containing 'BOC') discovered from CSVs."""
        if self._boc_rule_ids is not None:
            return self._boc_rule_ids
        try:
            name_map = self._get_rule_name_map()
            boc_ids = {rid for rid, nm in name_map.items() if 'boc' in str(nm).lower()}
            self._boc_rule_ids = boc_ids
            return self._boc_rule_ids
        except Exception:
            self._boc_rule_ids = set()
            return self._boc_rule_ids

    def _is_boc_rule(self, rule_id: Optional[int]) -> bool:
        """Return True if rule_id is in the discovered BOC rule set."""
        if rule_id is None:
            return False
        try:
            boc_ids = self._load_boc_rule_ids()
            return int(rule_id) in boc_ids
        except Exception:
            return False

    def _prioritize_boc_and_top_k(
        self,
        feature_importance: List[Dict[str, Any]],
        feature_names: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Reorder importance list to put BOC-related rules first and limit to top_k.

        - Determines BOC-related rules via Qradar_rule CSV names containing 'BOC'.
        - Keeps original relative order within BOC group and within others.
        - Returns only the first top_k items from the reordered list.
        """
        if not feature_importance:
            return []

        # Remove zero-importance items first
        filtered: List[Dict[str, Any]] = [
            item for item in feature_importance
            if float(item.get('importance', 0.0)) > 0.0
        ]
        if not filtered:
            return []

        # Stable partition based on BOC membership
        boc_items: List[Dict[str, Any]] = []
        other_items: List[Dict[str, Any]] = []

        for item in filtered:
            feat_name = str(item.get('feature', ''))
            rid = self._extract_rule_id(feat_name)
            if self._is_boc_rule(rid):
                boc_items.append(item)
            else:
                other_items.append(item)

        # If no BOC items at all, just take top_k from original
        if not boc_items:
            return feature_importance[:max(0, int(top_k))]

        # Combine BOC-first then others, limit to top_k
        combined = boc_items + other_items
        k = max(0, int(top_k))
        limited = combined[:k]

        # Enrich with rule_id and rule_name; fix ranks after filtering
        name_map = self._get_rule_name_map()
        enriched: List[Dict[str, Any]] = []
        for new_rank, item in enumerate(limited, start=1):
            feat_name = str(item.get('feature', ''))
            rid = self._extract_rule_id(feat_name)
            rule_name = name_map.get(int(rid)) if rid is not None else None
            out = dict(item)
            out['rank'] = int(new_rank)
            if rid is not None:
                out['rule_id'] = int(rid)
            if rule_name is not None:
                out['rule_name'] = str(rule_name)
            enriched.append(out)
        return enriched
