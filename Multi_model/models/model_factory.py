#!/usr/bin/env python3
"""
ModelFactory - Factory class for creating model instances.

This module provides a centralized way to create and manage different ML models
with automatic detection of available dependencies.

Python 3.6.8 Compatible
"""

import sys
import os
import logging
import json
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import base model
from .base import ModelBase
from .clustering_base import ClusteringBase

# Import all available models
from .random_forest import RandomForestModel
from .svm import SVMModel
from .logistic_regression import LogisticRegressionModel
from .decision_tree import DecisionTreeModel
from .adaboost import AdaBoostModel

# Import clustering models
from .kmeans import KMeansModel
from .dbscan import DBSCANModel
from .gaussian_mixture import GaussianMixtureModel
from .extra_trees import ExtraTreesModel
from .naive_bayes import NaiveBayesModel

# Import optional models with availability checks
try:
    from .gradient_boosting import GradientBoostingModel
    GRADIENT_BOOSTING_AVAILABLE = True
    GRADIENT_BOOSTING_ERROR = None
except ImportError as e:
    GRADIENT_BOOSTING_AVAILABLE = False
    GRADIENT_BOOSTING_ERROR = str(e)

# Configure logging
logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating model instances."""
    
    # Base model registry - always available
    _BASE_MODELS = {
        'random_forest': RandomForestModel,
        'svm': SVMModel,
        'logistic_regression': LogisticRegressionModel,
        'decision_tree': DecisionTreeModel,
        'adaboost': AdaBoostModel,
        'extra_trees': ExtraTreesModel,
        'naive_bayes': NaiveBayesModel
    }
    
    # Clustering models - always available
    _CLUSTERING_MODELS = {
        'kmeans': KMeansModel,
        'dbscan': DBSCANModel,
        'gaussian_mixture': GaussianMixtureModel
    }
    
    @classmethod
    def load_model_config(cls) -> Dict[str, Any]:
        """Load model configuration from model_config.json."""
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'model_config.json'
        )
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Model config file not found at {config_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in model config: {e}")
            return {}
    
    @classmethod
    def _get_model_registry(cls) -> Dict[str, Any]:
        """Get available model registry based on installed packages."""
        registry = cls._BASE_MODELS.copy()
        
        # Add optional models if available and actually working
        if GRADIENT_BOOSTING_AVAILABLE:
            registry['gradient_boosting'] = GradientBoostingModel
        
        return registry
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> Any:
        """
        Create model instance based on type.
        
        Args:
            model_type: Type of model to create ('random_forest', 'svm', etc.)
            config: Model configuration
            
        Returns:
            Model instance (supervised or clustering)
        """
        registry = cls._get_model_registry()
        clustering_registry = cls._get_clustering_registry()
        
        if model_type in registry:
            model_class = registry[model_type]
            return model_class(config)
        elif model_type in clustering_registry:
            model_class = clustering_registry[model_type]
            return model_class(config)
        else:
            available_models = list(registry.keys()) + list(clustering_registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
    
    @classmethod
    def _get_clustering_registry(cls) -> Dict[str, Any]:
        """Get clustering model registry."""
        return cls._CLUSTERING_MODELS.copy()
    
    @classmethod
    def is_supervised_model(cls, model_type: str) -> bool:
        """Check if model type is supervised."""
        return model_type in cls._get_model_registry()
    
    @classmethod
    def is_clustering_model(cls, model_type: str) -> bool:
        """Check if model type is clustering."""
        return model_type in cls._get_clustering_registry()
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of all available model types (supervised + clustering)."""
        supervised = list(cls._get_model_registry().keys())
        clustering = list(cls._get_clustering_registry().keys())
        return supervised + clustering
    
    @classmethod
    def get_supervised_models(cls) -> List[str]:
        """Get list of available supervised model types."""
        return list(cls._get_model_registry().keys())
    
    @classmethod
    def get_clustering_models(cls) -> List[str]:
        """Get list of available clustering model types."""
        return list(cls._get_clustering_registry().keys())
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        registry = cls._get_model_registry()
        clustering_registry = cls._get_clustering_registry()
        
        info = {}
        
        # Process supervised models
        for model_type, model_class in registry.items():
            try:
                # Create temporary instance to check capabilities
                temp_instance = model_class({'model': {}})
                info[model_type] = {
                    'type': 'supervised',
                    'class_name': model_class.__name__,
                    'requires_scaling': temp_instance.needs_scaling(),
                    'supports_feature_importance': hasattr(temp_instance, 'get_feature_importance'),
                    'supports_grid_search': bool(temp_instance.get_grid_search_params()),
                    'description': temp_instance.__class__.__doc__ or 'No description available'
                }
            except Exception as e:
                logger.warning(f"Could not get info for model {model_type}: {e}")
                info[model_type] = {
                    'type': 'supervised',
                    'class_name': model_class.__name__,
                    'error': str(e)
                }
        
        # Process clustering models
        for model_type, model_class in clustering_registry.items():
            try:
                # Create temporary instance to check capabilities
                temp_instance = model_class({'model': {}})
                info[model_type] = {
                    'type': 'clustering',
                    'class_name': model_class.__name__,
                    'requires_scaling': temp_instance.needs_scaling(),
                    'supports_grid_search': bool(temp_instance.get_grid_search_params()),
                    'description': temp_instance.__class__.__doc__ or 'No description available'
                }
            except Exception as e:
                logger.warning(f"Could not get info for clustering model {model_type}: {e}")
                info[model_type] = {
                    'type': 'clustering',
                    'class_name': model_class.__name__,
                    'error': str(e)
                }
        
        return info
    
    @classmethod
    def get_model_recommendations(cls, use_case: str = None) -> List[str]:
        """Get model recommendations based on use case."""
        model_config = cls.load_model_config()
        guidelines = model_config.get('model_selection_guidelines', {})
        available_models = cls.get_available_models()  # Only actually working models
        
        if not use_case:
            # Return default recommendations
            supervised_default = guidelines.get('supervised', {}).get('default_recommendation', 'random_forest')
            clustering_default = guidelines.get('clustering', {}).get('default_recommendation', 'kmeans')
            defaults = [supervised_default, clustering_default]
            return [model for model in defaults if model in available_models]
        
        # Handle nested structure (supervised/clustering categories)
        if 'supervised' in guidelines or 'clustering' in guidelines:
            supervised_guidelines = guidelines.get('supervised', {})
            clustering_guidelines = guidelines.get('clustering', {})
            
            # Check supervised recommendations
            supervised_key = f'for_{use_case}'
            if supervised_key in supervised_guidelines:
                recommended = supervised_guidelines[supervised_key]
                return [model for model in recommended if model in available_models]
            
            # Check clustering recommendations  
            clustering_key = f'for_{use_case}'
            if clustering_key in clustering_guidelines:
                recommended = clustering_guidelines[clustering_key]
                return [model for model in recommended if model in available_models]
        
        # Handle legacy flat structure
        use_case_key = f'for_{use_case}'
        if use_case_key in guidelines:
            recommended = guidelines[use_case_key]
            return [model for model in recommended if model in available_models]
        
        # Fallback to default
        default = guidelines.get('default_recommendation', 'random_forest')
        return [default] if default in available_models else ['random_forest']
    
    @classmethod
    def get_model_config_summary(cls) -> Dict[str, Any]:
        """Get a summary of all model configurations."""
        model_config = cls.load_model_config()
        models = model_config.get('models', {})
        available_models = cls.get_available_models()
        
        summary = {}
        for model_type in available_models:
            if model_type in models:
                model_info = models[model_type]
                summary[model_type] = {
                    'description': model_info.get('description', 'No description'),
                    'requires_scaling': model_info.get('requires_scaling', False),
                    'supports_feature_importance': model_info.get('supports_feature_importance', False),
                    'supports_grid_search': model_info.get('supports_grid_search', False),
                    'recommended_for': model_info.get('recommended_for', []),
                    'availability': model_info.get('availability', 'standard')
                }
        
        return summary
    
    @classmethod
    def get_availability_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get detailed availability information for all models."""
        info = {
            'standard_models': {
                'available': list(cls._BASE_MODELS.keys()),
                'description': 'Always available with scikit-learn'
            },
            'optional_models': {}
        }
        
        # Gradient Boosting
        if GRADIENT_BOOSTING_AVAILABLE:
            info['optional_models']['gradient_boosting'] = {
                'status': 'available',
                'description': 'Gradient Boosting available'
            }
        else:
            info['optional_models']['gradient_boosting'] = {
                'status': 'unavailable',
                'error': GRADIENT_BOOSTING_ERROR,
                'description': 'Gradient Boosting not available'
            }
        
        return info