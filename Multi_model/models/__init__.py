#!/usr/bin/env python3
"""
Models module - Machine Learning model implementations.

This module provides various machine learning model implementations
with a consistent interface and GridSearch capabilities.

Available Models:
- RandomForestModel: Random Forest classifier
- SVMModel: Support Vector Machine
- LogisticRegressionModel: Logistic Regression
- DecisionTreeModel: Decision Tree classifier
- AdaBoostModel: AdaBoost classifier
- ExtraTreesModel: Extra Trees classifier
- NaiveBayesModel: Naive Bayes classifier
- GradientBoostingModel: Gradient Boosting (if available)
- XGBoostModel: XGBoost (if xgboost package is installed)

Python 3.6.8 Compatible
"""

from .base import ModelBase
from .model_factory import ModelFactory

# Import all available models
from .random_forest import RandomForestModel
from .svm import SVMModel
from .logistic_regression import LogisticRegressionModel
from .decision_tree import DecisionTreeModel
from .adaboost import AdaBoostModel
from .extra_trees import ExtraTreesModel
from .naive_bayes import NaiveBayesModel

# Import optional models with availability checks
try:
    from .gradient_boosting import GradientBoostingModel
    __all__ = [
        'ModelBase', 'ModelFactory',
        'RandomForestModel', 'SVMModel', 'LogisticRegressionModel',
        'DecisionTreeModel', 'AdaBoostModel', 'ExtraTreesModel',
        'NaiveBayesModel', 'GradientBoostingModel'
    ]
except ImportError:
    __all__ = [
        'ModelBase', 'ModelFactory',
        'RandomForestModel', 'SVMModel', 'LogisticRegressionModel',
        'DecisionTreeModel', 'AdaBoostModel', 'ExtraTreesModel',
        'NaiveBayesModel'
    ]