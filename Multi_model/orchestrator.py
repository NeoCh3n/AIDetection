#!/usr/bin/env python3
"""
PipelineOrchestrator - Main orchestrator for the OOP pipeline.

Provides unified interface for training and detection operations
with model switching and configuration management.
Supports both supervised and clustering models.

Python 3.6.8 Compatible
"""

import sys
import os
import logging
import json
from typing import Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import required modules
from system.logging_utils import setup_global_daily_file_logging
from system.config import get_config

# Import pipeline components
from pipelines.training_pipeline import TrainingPipeline
from pipelines.detection_pipeline import DetectionPipeline
from models.model_factory import ModelFactory


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
            model_type: Type of model to train (supervised or clustering)
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        self.logger.info(f"Starting training with {model_type} model")
        
        # Validate model type
        available_models = ModelFactory.get_available_models()
        if model_type not in available_models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
        
        # Log model type for user awareness
        if ModelFactory.is_clustering_model(model_type):
            self.logger.info(f"Training clustering model: {model_type}")
        else:
            self.logger.info(f"Training supervised model: {model_type}")
        
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
    
    def get_supervised_models(self) -> List[str]:
        """Get list of available supervised models."""
        return ModelFactory.get_supervised_models()
    
    def get_clustering_models(self) -> List[str]:
        """Get list of available clustering models."""
        return ModelFactory.get_clustering_models()
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all available models."""
        return ModelFactory.get_model_info()
    
    def get_model_recommendations(self, use_case: str = None) -> List[str]:
        """Get model recommendations for specific use case."""
        return ModelFactory.get_model_recommendations(use_case)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def validate_model_type(self, model_type: str) -> bool:
        """Validate if model type is available."""
        return model_type in self.list_available_models()
    
    def is_clustering_model(self, model_type: str) -> bool:
        """Check if model type is a clustering model."""
        return ModelFactory.is_clustering_model(model_type)
    
    def is_supervised_model(self, model_type: str) -> bool:
        """Check if model type is a supervised model."""
        return ModelFactory.is_supervised_model(model_type)