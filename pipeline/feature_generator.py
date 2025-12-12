import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
import logging
import sys
import os

# Public exports for static analyzers
__all__ = ["FeatureGenerator"]

# Import QRadarRuleManager from shared_utils
from shared_utils.qradar_rule_manager import QRadarRuleManager

# Add parent directory to path for config imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from system.config import get_config
except ImportError:
    # Fallback if config module not available
    def get_config():
        return {}

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """
    Generates feature vectors from aggregated rule counts for threat detection.
    Uses QRadarRuleManager with UAT-to-Production rule ID mapping support.
    """
    
    def __init__(self, environment: str = 'prod', config: Optional[Dict] = None):
        """
        Initialize feature generator with centralized configuration
        
        Args:
            environment: 'prod' or 'uat' - determines baseline rule set
            config: Configuration dictionary for rule manager (uses system config if None)
        """
        self.environment = environment
        self.config = config or get_config()
        self.rule_manager = QRadarRuleManager(
            mode=self.config.get('rule_manager', {}).get('mode', 'file'),
            config=self.config.get('rule_manager', {}),
            environment=environment
        )
        self._rule_to_index = None
        self._vector_dimension = None
        self._family_to_index = None
        self._rule_feature_to_index: Optional[Dict[int, int]] = None
        
        # Validate configuration on initialization
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        rule_manager_config = self.config.get('rule_manager', {})
        if not isinstance(rule_manager_config, dict):
            self.config['rule_manager'] = {}
        
        logger.info(f"Feature generator initialized with {self.environment} environment")

    def initialize_rules(self) -> None:
        """Initialize feature mappings: families + per-rule features."""
        # Family features
        self._family_to_index = self.rule_manager.get_family_to_index_map()

        # Per-rule features (cover every production rule so we can always emit rule_id)
        prod_rule_list = self.rule_manager.get_production_rule_list()
        start_idx = len(self._family_to_index)
        self._rule_feature_to_index = {
            int(rule_id): start_idx + i for i, rule_id in enumerate(sorted(prod_rule_list))
        }

        self._vector_dimension = len(self._family_to_index) + len(self._rule_feature_to_index)

        # Keep a reference map for compatibility with any legacy callers
        self._rule_to_index = dict(self._rule_feature_to_index)

        logger.info(
            "Initialized feature generator with %d families and %d per-rule features "
            "(total dimension %d)",
            len(self._family_to_index),
            len(self._rule_feature_to_index),
            self._vector_dimension,
        )
    
    def generate_feature_vectors(self, 
                               df_agg: pd.DataFrame, 
                               mode: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate feature vectors from aggregated rule counts with UAT-to-Production mapping.
        
        Args:
            df_agg: DataFrame with aggregated rule counts from feature_aggregator.py
            mode: 'train' or 'detect' mode
            
        Returns:
            Tuple of (X_feature_matrix, y_labels)
            y_labels is None in detect mode
        """
        if self._rule_to_index is None:
            self.initialize_rules()
        
        if df_agg.empty:
            logger.warning("Empty aggregated DataFrame provided")
            return np.array([]), None
        
        # Initialize feature matrix
        n_samples = len(df_agg)
        X = np.zeros((n_samples, self._vector_dimension), dtype=float)
        
        # Get UAT-to-Production mapping for training mode
        uat_to_prod_map: Dict[int, int] = {}
        if mode == 'train' and self.environment == 'uat':
            uat_map = self.rule_manager.get_uat_to_prod_map()
            if uat_map:
                uat_to_prod_map = {int(k): int(v) for k, v in uat_map.items()}
            logger.info(f"Using {len(uat_to_prod_map)} UAT-to-Production mappings")
        
        # Map aggregated rule counts to feature vectors
        for idx, row in df_agg.iterrows():
            # Choose which aggregated counts to use
            if 'aggregated_rules_log1p_sum' in df_agg.columns:
                rule_counts = row['aggregated_rules_log1p_sum']
            elif 'aggregated_rules' in df_agg.columns or 'aggregated_rules_dict' in df_agg.columns:
                rule_counts = row['aggregated_rules'] if 'aggregated_rules' in df_agg.columns else row['aggregated_rules_dict']
            else:
                rule_counts = None

            if isinstance(rule_counts, dict):
                for rule_id, count in rule_counts.items():
                    try:
                        rule_id_int = int(str(rule_id))
                        count_val = float(count)
                    except Exception:
                        continue

                    family = self.rule_manager.get_rule_family(rule_id_int)

                    # Accumulate into family feature if applicable
                    if family != "Uncategorized" and self._family_to_index and family in self._family_to_index:
                        col_idx = self._family_to_index[family]
                        X[idx, col_idx] += count_val

                    # Always accumulate into per-rule feature when available
                    if self._rule_feature_to_index and rule_id_int in self._rule_feature_to_index:
                        col_idx_rule = self._rule_feature_to_index[rule_id_int]
                        X[idx, col_idx_rule] += count_val
                        
            elif 'rule_id' in df_agg.columns and 'count' in df_agg.columns:
                # Handle direct rule_id/count format
                rule_id = int(str(row['rule_id']))
                count = float(str(row['count']))
                
                # Get family for the rule
                family = self.rule_manager.get_rule_family(rule_id)
                
                if family != "Uncategorized" and self._family_to_index and family in self._family_to_index:
                    col_idx = self._family_to_index[family]
                    X[idx, col_idx] += count
                if self._rule_feature_to_index and rule_id in self._rule_feature_to_index:
                    col_idx_rule = self._rule_feature_to_index[rule_id]
                    X[idx, col_idx_rule] += count
        
        # Handle labels for training mode
        y = None
        if mode == 'train':
            if 'label' in df_agg.columns:
                y = df_agg['label'].values.astype(int)
            elif 'is_attack' in df_agg.columns:
                y = df_agg['is_attack'].values.astype(int)
            else:
                logger.warning("No label column found in training mode")
        
        logger.info(f"Generated feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        if y is not None and y.size > 0:
            unique_labels, counts = np.unique(y, return_counts=True)
            logger.info(f"Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get feature names corresponding to rule families."""
        if self._family_to_index is None:
            self.initialize_rules()
        if self._rule_feature_to_index is None:
            self.initialize_rules()

        # Families first (sorted by index)
        sorted_families = sorted(self._family_to_index.items(), key=lambda x: x[1])
        names = [f"family_{name}" for name, _ in sorted_families]

        # Then per-rule features (sorted by index)
        sorted_rules = sorted(self._rule_feature_to_index.items(), key=lambda x: x[1])
        names.extend([f"rule_{rid}" for rid, _ in sorted_rules])

        return names
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered rules and families."""
        if self._family_to_index is None:
            self.initialize_rules()
            
        return {
            'total_families': len(self._family_to_index),
            'total_rule_features': len(self._rule_feature_to_index or {}),
            'families': list(self._family_to_index.keys()),
            'environment': self.environment,
            'vector_dimension': self._vector_dimension
        }

    @classmethod
    def create_from_config(cls, config: Optional[Dict] = None, environment: str = 'prod') -> 'FeatureGenerator':
        """
        Factory method to create feature generator from centralized configuration.
        
        Args:
            config: Configuration dictionary (uses system config if None)
            environment: Environment type ('prod' or 'uat')
            
        Returns:
            Configured FeatureGenerator instance
        """
        return cls(environment=environment, config=config)

    def get_feature_vector_dimension(self) -> int:
        """Get the expected dimension of feature vectors"""
        if self._vector_dimension is None:
            self.initialize_rules()
        return self._vector_dimension or 0
    
    def validate_rule_coverage(self, df_agg: pd.DataFrame) -> Dict[str, Any]:
        """Validate how many rules are actually present in the data with UAT mapping."""
        uat_to_prod_map = {}
        if self.environment == 'uat':
            uat_map = self.rule_manager.get_uat_to_prod_map()
            if uat_map:
                uat_to_prod_map = {int(k): int(v) for k, v in uat_map.items()}
        
        all_rules = set()
        
        if 'aggregated_rules_dict' in df_agg.columns:
            for rules_dict in df_agg['aggregated_rules_dict']:
                if isinstance(rules_dict, dict):
                    for rule_id in rules_dict.keys():
                        rule_id_int = int(str(rule_id))
                        prod_rule_id = uat_to_prod_map.get(rule_id_int, rule_id_int)
                        all_rules.add(prod_rule_id)
        elif 'rule_id' in df_agg.columns:
            for rule_id in df_agg['rule_id'].unique().tolist():
                rule_id_int = int(str(rule_id))
                prod_rule_id = uat_to_prod_map.get(rule_id_int, rule_id_int)
                all_rules.add(prod_rule_id)
        
        discovered_rules = set(self.rule_manager.get_production_rule_list())
        present_rules = len(all_rules.intersection(discovered_rules))
        unknown_rules = len(all_rules - discovered_rules)
        
        return {
            'present_rules': present_rules,
            'missing_rules': len(discovered_rules) - present_rules,
            'unknown_rules': unknown_rules,
            'uat_mappings_used': len(uat_to_prod_map)
        }
