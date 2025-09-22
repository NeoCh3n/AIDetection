#!/usr/bin/env python3
"""
DetectionPipeline - Real-time threat detection using trained models.

This module handles loading trained models and processing new data
for threat detection with alerting and logging capabilities.
Supports both supervised and clustering models.

Python 3.6.8 Compatible
"""

import sys
import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import required modules
from shared_utils.time_utils import get_window_id

# Import our modular components
from data.data_handler import DataHandler
from features.feature_manipulator import FeatureManipulator
from models.model_factory import ModelFactory


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
        self.model_type = None
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
            self.model_type = model_data.get('model_type', 'random_forest')
            
            # Create model instance and load state
            self.model = ModelFactory.create_model(self.model_type.lower(), model_data['config'])
            self.model.model = model_data['model']
            self.model.scaler = model_data.get('scaler')
            
            self.logger.info(f"Model loaded successfully: {self.model_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run detection on latest data.
        
        Args:
            threshold: Probability threshold for threat classification (supervised models only)
            
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
            
            # Determine if this is supervised or clustering detection
            if ModelFactory.is_clustering_model(self.model_type):
                return self._detect_clustering(raw_data, X, threshold)
            else:
                return self._detect_supervised(raw_data, X, threshold)
                
        except Exception as e:
            self.logger.error(f"Detection pipeline failed: {e}")
            raise
    
    def _detect_supervised(self, raw_data: pd.DataFrame, X: np.ndarray, 
                          threshold: float) -> Dict[str, Any]:
        """Run supervised detection (classification)."""
        # Make predictions
        predictions = self.model.predict(X)
        
        try:
            probabilities = self.model.predict_proba(X)
            threat_probabilities = probabilities[:, 1] if probabilities.shape[1] == 2 else None
        except (AttributeError, NotImplementedError):
            threat_probabilities = None
        
        # Process results and generate alerts
        alerts = self._process_supervised_detections(raw_data, predictions, threat_probabilities, threshold)
        
        results = {
            'status': 'completed',
            'model_type': 'supervised',
            'timestamp': datetime.now().isoformat(),
            'total_windows': len(predictions),
            'threat_count': int(np.sum(predictions)),
            'alerts': alerts
        }
        
        self.logger.info(f"Supervised detection completed - {results['threat_count']} threats detected")
        return results
    
    def _detect_clustering(self, raw_data: pd.DataFrame, X: np.ndarray, 
                          threshold: float) -> Dict[str, Any]:
        """Run clustering detection (anomaly detection)."""
        # Make cluster predictions
        try:
            if hasattr(self.model, 'predict'):
                cluster_labels = self.model.predict(X)
            else:
                cluster_labels = self.model.fit_predict(X)
        except Exception as e:
            self.logger.error(f"Clustering prediction failed: {e}")
            return {'status': 'error', 'error': str(e)}
        
        # For clustering, anomalies could be:
        # 1. DBSCAN noise points (label -1)
        # 2. Small clusters (outliers)
        # 3. Points far from cluster centers (for distance-based methods)
        anomaly_indices = self._identify_clustering_anomalies(cluster_labels, X)
        
        # Process results and generate alerts
        alerts = self._process_clustering_detections(raw_data, cluster_labels, anomaly_indices)
        
        results = {
            'status': 'completed',
            'model_type': 'clustering',
            'timestamp': datetime.now().isoformat(),
            'total_windows': len(cluster_labels),
            'clusters_found': len(np.unique(cluster_labels)),
            'anomaly_count': len(anomaly_indices),
            'cluster_distribution': {int(label): int(count) for label, count in 
                                   zip(*np.unique(cluster_labels, return_counts=True))},
            'alerts': alerts
        }
        
        self.logger.info(f"Clustering detection completed - {results['anomaly_count']} anomalies detected")
        return results
    
    def _identify_clustering_anomalies(self, cluster_labels: np.ndarray, X: np.ndarray) -> List[int]:
        """Identify anomalies from clustering results."""
        anomaly_indices = []
        
        # Method 1: DBSCAN noise points
        noise_mask = cluster_labels == -1
        anomaly_indices.extend(np.where(noise_mask)[0].tolist())
        
        # Method 2: Very small clusters (potential outliers)
        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
        min_cluster_size = max(2, int(0.05 * len(cluster_labels)))  # At least 5% of data
        
        small_clusters = unique_labels[counts < min_cluster_size]
        for cluster_id in small_clusters:
            cluster_mask = cluster_labels == cluster_id
            anomaly_indices.extend(np.where(cluster_mask)[0].tolist())
        
        # Method 3: Distance-based anomalies for K-Means/Gaussian Mixture
        if hasattr(self.model, 'cluster_centers_') or hasattr(self.model, 'means_'):
            try:
                anomaly_indices.extend(self._find_distance_based_anomalies(cluster_labels, X))
            except Exception as e:
                self.logger.warning(f"Could not compute distance-based anomalies: {e}")
        
        return list(set(anomaly_indices))  # Remove duplicates
    
    def _find_distance_based_anomalies(self, cluster_labels: np.ndarray, X: np.ndarray) -> List[int]:
        """Find anomalies based on distance from cluster centers."""
        anomaly_indices = []
        
        # Get cluster centers
        if hasattr(self.model, 'cluster_centers_'):
            centers = self.model.cluster_centers_
        elif hasattr(self.model, 'get_means'):
            centers = self.model.get_means()
        else:
            return []
        
        # Apply scaling if model was trained with scaling
        if self.model.scaler is not None:
            X_scaled = self.model.scaler.transform(X)
        else:
            X_scaled = X
        
        # Calculate distances for each point to its assigned cluster center
        for i, (point, label) in enumerate(zip(X_scaled, cluster_labels)):
            if label >= 0 and label < len(centers):  # Valid cluster
                center = centers[label]
                distance = np.linalg.norm(point - center)
                
                # Calculate threshold as 95th percentile of distances to this center
                cluster_mask = cluster_labels == label
                cluster_points = X_scaled[cluster_mask]
                cluster_distances = np.array([np.linalg.norm(p - center) for p in cluster_points])
                threshold = np.percentile(cluster_distances, 95)
                
                if distance > threshold:
                    anomaly_indices.append(i)
        
        return anomaly_indices
    
    def _process_supervised_detections(self, raw_data: pd.DataFrame, predictions: np.ndarray,
                                     probabilities: Optional[np.ndarray], threshold: float) -> List[Dict[str, Any]]:
        """Process supervised detection results and generate alerts."""
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
                    'alert_type': 'supervised_threat',
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
    
    def _process_clustering_detections(self, raw_data: pd.DataFrame, cluster_labels: np.ndarray,
                                     anomaly_indices: List[int]) -> List[Dict[str, Any]]:
        """Process clustering detection results and generate alerts."""
        alerts = []
        
        # Group raw data by window for context
        raw_data['window_id'] = raw_data['timestamp'].apply(
            lambda ts: get_window_id(ts, self.feature_manipulator.window_size_minutes)
        )
        
        window_groups = raw_data.groupby('window_id')
        window_ids = list(window_groups.groups.keys())
        
        for anomaly_idx in anomaly_indices:
            if anomaly_idx < len(window_ids):
                window_id = window_ids[anomaly_idx]
                window_data = window_groups.get_group(window_id)
                cluster_label = cluster_labels[anomaly_idx]
                
                alert = {
                    'alert_type': 'clustering_anomaly',
                    'window_id': window_id,
                    'timestamp': window_data['timestamp'].min().isoformat(),
                    'hostnames': window_data['hostname'].unique().tolist(),
                    'cluster_label': int(cluster_label),
                    'anomaly_type': 'noise' if cluster_label == -1 else 'outlier',
                    'event_count': len(window_data),
                    'unique_rules': window_data['rule_id'].nunique()
                }
                
                # Add clustering-specific explanation
                alert['explanation'] = self._generate_clustering_explanation(window_data, cluster_label)
                
                alerts.append(alert)
                
                # Log the detection
                self._log_detection(alert)
        
        return alerts
    
    def _generate_explanation(self, window_data: pd.DataFrame, sample_index: int) -> Dict[str, Any]:
        """Generate explanation for supervised threat detection using SHAP or feature importance."""
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
    
    def _generate_clustering_explanation(self, window_data: pd.DataFrame, cluster_label: int) -> Dict[str, Any]:
        """Generate explanation for clustering anomaly detection."""
        rule_counts = window_data.groupby('rule_id')['count'].sum().sort_values(ascending=False)
        
        explanation = {
            'method': 'clustering_analysis',
            'cluster_label': int(cluster_label),
            'anomaly_reason': 'noise_point' if cluster_label == -1 else 'outlier_cluster',
            'top_rules': [
                {'rule_id': int(rule_id), 'count': int(count)}
                for rule_id, count in rule_counts.head(5).items()
            ],
            'total_events': len(window_data),
            'unique_rules_triggered': window_data['rule_id'].nunique()
        }
        
        return explanation
    
    def _log_detection(self, alert: Dict[str, Any]) -> None:
        """Log detection alert."""
        try:
            from system.logging_utils import log_detection_alert
            
            # Format message for logging
            if alert['alert_type'] == 'supervised_threat':
                message = (
                    f"THREAT DETECTED - Window: {alert['window_id']}, "
                    f"Hosts: {len(alert['hostnames'])}, "
                    f"Confidence: {alert.get('confidence', 'N/A')}, "
                    f"Events: {alert['event_count']}"
                )
            else:  # clustering_anomaly
                message = (
                    f"ANOMALY DETECTED - Window: {alert['window_id']}, "
                    f"Hosts: {len(alert['hostnames'])}, "
                    f"Type: {alert['anomaly_type']}, "
                    f"Cluster: {alert['cluster_label']}, "
                    f"Events: {alert['event_count']}"
                )
            
            self.logger.warning(message)
            
            # Send to syslog/QRadar if configured
            if hasattr(log_detection_alert, '__call__'):
                log_detection_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to log detection: {e}")