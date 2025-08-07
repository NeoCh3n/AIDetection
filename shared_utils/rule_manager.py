import os
import pandas as pd
import json
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RuleManager:
    """
    Manages QRadar rule discovery and mapping for dynamic feature vector creation.
    """
    
    def __init__(self, rules_folder_path: str = "*Qradar_rule"):
        self.rules_folder_path = rules_folder_path
        self.rules_cache_file = "rules_cache.json"
        self._rule_list = None
        self._rule_to_index = None
    
    def discover_rules(self, use_cache: bool = True) -> List[int]:
        """
        Discover all rule IDs from QRadar rule files.
        
        Args:
            use_cache: Whether to use cached rules if available
            
        Returns:
            List of unique rule IDs sorted in ascending order
        """
        if use_cache and self._load_cached_rules():
            logger.info(f"Loaded {len(self._rule_list)} rules from cache")
            return self._rule_list
        
        rule_ids = set()
        
        # Handle asterisk in path - use absolute path
        full_path = os.path.join(os.getcwd(), self.rules_folder_path)
        
        if not os.path.exists(full_path):
            # Try without asterisk
            full_path = os.path.join(os.getcwd(), self.rules_folder_path.replace('*', ''))
            if not os.path.exists(full_path):
                # Try glob pattern
                import glob
                possible_paths = glob.glob(self.rules_folder_path)
                if possible_paths:
                    full_path = possible_paths[0]
                else:
                    raise FileNotFoundError(f"Rules folder not found: {self.rules_folder_path}")
        
        # Scan for CSV files
        csv_files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(full_path, csv_file)
            try:
                df = pd.read_csv(file_path)
                if 'id' in df.columns:
                    ids = df['id'].dropna().astype(int).tolist()
                    rule_ids.update(ids)
                    logger.info(f"Found {len(ids)} rules in {csv_file}")
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
        
        self._rule_list = sorted(list(rule_ids))
        self._save_cached_rules()
        
        logger.info(f"Discovered {len(self._rule_list)} total unique rules")
        return self._rule_list
    
    def get_rule_list(self) -> List[int]:
        """Get the complete list of rule IDs."""
        if self._rule_list is None:
            self._rule_list = self.discover_rules()
        return self._rule_list
    
    def get_rule_to_index_map(self) -> Dict[int, int]:
        """Get mapping from rule ID to feature vector index."""
        if self._rule_to_index is None:
            rule_list = self.get_rule_list()
            self._rule_to_index = {rule_id: idx for idx, rule_id in enumerate(rule_list)}
        return self._rule_to_index
    
    def get_vector_dimension(self) -> int:
        """Get the dimension of feature vectors."""
        return len(self.get_rule_list())
    
    def _load_cached_rules(self) -> bool:
        """Load rules from cache file if it exists."""
        if os.path.exists(self.rules_cache_file):
            try:
                with open(self.rules_cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if 'rule_list' in cached_data:
                        self._rule_list = cached_data['rule_list']
                        return True
            except Exception as e:
                logger.warning(f"Error loading cached rules: {e}")
        return False
    
    def _save_cached_rules(self):
        """Save discovered rules to cache file."""
        try:
            with open(self.rules_cache_file, 'w') as f:
                json.dump({'rule_list': self._rule_list}, f)
            logger.info("Rules cached successfully")
        except Exception as e:
            logger.warning(f"Error caching rules: {e}")
    
    def validate_rule_files(self) -> bool:
        """Validate that rule files exist and are readable."""
        base_path = self.rules_folder_path.replace('*', '')
        
        if not os.path.exists(base_path):
            return False
            
        csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
        return len(csv_files) > 0
    
    def clear_cache(self):
        """Clear the rules cache."""
        if os.path.exists(self.rules_cache_file):
            os.remove(self.rules_cache_file)
            logger.info("Rules cache cleared")
        self._rule_list = None
        self._rule_to_index = None