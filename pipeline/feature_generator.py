import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
from shared_utils.rule_manager import RuleManager

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """
    Generates feature vectors from aggregated rule counts for ransomware detection.
    Uses dynamic rule count from RuleManager instead of hardcoded dimensions.
    """
    
    def __init__(self, rules_folder_path: str = "*Qradar_rule"):
        self.rule_manager = RuleManager(rules_folder_path)
        self._rule_to_index = None
        self._vector_dimension = None
    
    def initialize_rules(self) -> None:
        """Initialize rule mappings from discovered rules."""
        self._rule_to_index = self.rule_manager.get_rule_to_index_map()
        self._vector_dimension = self.rule_manager.get_vector_dimension()
        logger.info(f"Initialized feature generator with {self._vector_dimension} rules")
    
    def generate_feature_vectors(self, 
                               df_agg: pd.DataFrame, 
                               mode: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate feature vectors from aggregated rule counts.
        
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
        X = np.zeros((n_samples, self._vector_dimension), dtype=np.float32)
        
        # Map aggregated rule counts to feature vectors
        for idx, row in df_agg.iterrows():
            if 'aggregated_rules_dict' in df_agg.columns:
                rule_counts = row['aggregated_rules_dict']
                if isinstance(rule_counts, dict):
                    for rule_id, count in rule_counts.items():
                        if rule_id in self._rule_to_index:
                            col_idx = self._rule_to_index[rule_id]
                            X[idx, col_idx] = float(count)
            elif 'rule_id' in df_agg.columns and 'count' in df_agg.columns:
                # Handle direct rule_id/count format
                rule_id = row['rule_id']
                count = row['count']
                if rule_id in self._rule_to_index:
                    col_idx = self._rule_to_index[rule_id]
                    X[idx, col_idx] = float(count)
        
        # Handle labels for training mode
        y = None
        if mode == 'train':
            if 'label' in df_agg.columns:
                y = df_agg['label'].values.astype(np.int32)
            elif 'is_attack' in df_agg.columns:
                y = df_agg['is_attack'].values.astype(np.int32)
            else:
                logger.warning("No label column found in training mode")
        
        logger.info(f"Generated feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        if y is not None:
            logger.info(f"Label distribution: {np.bincount(y)}")
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get feature names corresponding to rule IDs."""
        rule_list = self.rule_manager.get_rule_list()
        return [f"rule_{rule_id}" for rule_id in rule_list]
    
    def get_rule_statistics(self) -> Dict[str, int]:
        """Get statistics about discovered rules."""
        return {
            'total_rules': self.rule_manager.get_vector_dimension(),
            'rule_min': min(self.rule_manager.get_rule_list()) if self.rule_manager.get_rule_list() else 0,
            'rule_max': max(self.rule_manager.get_rule_list()) if self.rule_manager.get_rule_list() else 0
        }
    
    def validate_rule_coverage(self, df_agg: pd.DataFrame) -> Dict[str, int]:
        """Validate how many rules are actually present in the data."""
        if 'aggregated_rules_dict' in df_agg.columns:
            all_rules = set()
            for rules_dict in df_agg['aggregated_rules_dict']:
                if isinstance(rules_dict, dict):
                    all_rules.update(rules_dict.keys())
        elif 'rule_id' in df_agg.columns:
            all_rules = set(df_agg['rule_id'].unique())
        else:
            return {'present_rules': 0, 'missing_rules': self._vector_dimension}
        
        discovered_rules = set(self.rule_manager.get_rule_list())
        present_rules = len(all_rules.intersection(discovered_rules))
        
        return {
            'present_rules': present_rules,
            'missing_rules': len(discovered_rules) - present_rules,
            'unknown_rules': len(all_rules - discovered_rules)
        }