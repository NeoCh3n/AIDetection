"""Pipeline package providing data processing primitives for the unified pipeline."""

from .feature_aggregator import aggregate_to_windows
from .feature_generator import FeatureGenerator, generate_feature_vectors
from .data_loader import load_data

__all__ = [
    "aggregate_to_windows",
    "FeatureGenerator",
    "generate_feature_vectors",
    "load_data",
]
