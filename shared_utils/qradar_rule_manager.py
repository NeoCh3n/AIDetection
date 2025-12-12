#!/usr/bin/env python3
"""
Unified QRadar Rule Manager
Combines API-based rule discovery with file-based rule discovery
"""

import os
import json
import csv
import glob
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import re
from datetime import datetime

class QRadarRuleManager:
    """
    Unified rule manager supporting both API and file-based rule discovery
    with UAT-to-Production rule ID mapping
    """
    
    def __init__(
        self,
        mode: str = 'api',
        config: Optional[Dict[str, Any]] = None,
        environment: str = 'prod',
    ):
        """
        Initialize rule manager
        
        Args:
            mode: 'api' (fetch from QRadar) or 'file' (load from CSV)
            config: Optional configuration dictionary with API/file settings
            environment: 'prod' or 'uat' - determines baseline rule set
        """
        self.mode = mode
        self.config: Dict[str, Any] = config or {}
        self.environment = environment
        self.logger = logging.getLogger('QRadarRuleManager')
        
        # Cache for rule mappings
        self._rule_list = None
        self._rule_to_index = None
        self._rule_mapping = None
        self._uat_to_prod_map = None
        self._prod_rule_list = None
        self._prod_rule_to_index = None
        self._family_mapping = None
        self._family_to_index = None
        self._rule_id_to_family = None
        
        # Default paths
        self.project_root = Path(__file__).parent.parent.absolute()
        self.rule_dir = self.project_root / "Qradar_rule"
        self.mapping_file = self.project_root / "rule_mapping.json"
        self.uat_mapping_file = self.project_root / "shared_utils" / "uat_to_prod_mapping.csv"
        self.family_mapping_file = self.project_root / "rule_family_mapping.json"
    
    def discover_rules(self) -> List[int]:
        """
        Discover rules based on configured mode
        
        Returns:
            List of unique rule IDs
        """
        if self.mode == 'api':
            return self._fetch_from_qradar()
        else:
            return self._discover_from_csv()
    
    def _fetch_from_qradar(self) -> List[int]:
        """
        Fetch rules from QRadar API
        
        Returns:
            List of rule IDs from all rule types
        """
        self.logger.info("Fetching rules from QRadar API...")
        
        # API configuration
        qradar_host = self.config.get('qradar_host', 'https://192.168.153.123')
        api_token = self.config.get('api_token', '677f60e2-3d58-4275-a1f0-c13d1975fdbe')
        api_version = self.config.get('api_version', '20.0')
        verify_ssl = self.config.get('verify_ssl', False)
        
        endpoints = {
            "rules": "/api/analytics/rules",
            "ade_rules": "/api/analytics/ade_rules", 
            "building_blocks": "/api/analytics/building_blocks"
        }
        
        headers = {
            "SEC": api_token,
            "Version": api_version
        }
        
        all_rule_ids = []
        
        try:
            # Ensure output directory exists
            os.makedirs(self.rule_dir, exist_ok=True)
            
            for rule_type, endpoint in endpoints.items():
                url = f"{qradar_host}{endpoint}"
                
                try:
                    response = requests.get(url, headers=headers, verify=verify_ssl)
                    response.raise_for_status()
                    rules = response.json()
                    
                    # Save rules to CSV
                    output_file = self.rule_dir / f"qradar_{rule_type}.csv"
                    self._save_rules_to_csv(rules, output_file, rule_type)
                    
                    # Extract rule IDs
                    rule_ids = [r.get('id') for r in rules if r.get('id')]
                    all_rule_ids.extend(rule_ids)
                    
                    self.logger.info(f"Fetched {len(rule_ids)} {rule_type} from API")
                    
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Failed to fetch {rule_type}: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"API fetch failed: {e}")
            # Fall back to file mode
            return self._discover_from_csv()
        
        return list(set(all_rule_ids))
    
    def _save_rules_to_csv(self, rules: List[Dict], output_file: Path, rule_type: str):
        """
        Save rules to CSV file
        
        Args:
            rules: List of rule dictionaries
            output_file: Output CSV file path
            rule_type: Type of rules being saved
        """
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "name", "type", "enabled", "origin"])
                
                for rule in rules:
                    rule_data = {
                        "id": rule.get("id"),
                        "name": rule.get("name"),
                        "type": rule.get("type", rule.get("rule_type", rule_type)),
                        "enabled": rule.get("enabled"),
                        "origin": rule.get("origin", "")
                    }
                    writer.writerow([
                        rule_data["id"],
                        rule_data["name"],
                        rule_data["type"],
                        rule_data["enabled"],
                        rule_data["origin"]
                    ])
        
        except Exception as e:
            self.logger.error(f"Failed to save rules to {output_file}: {e}")
    
    def _discover_from_csv(self) -> List[int]:
        """
        Discover rules from CSV files in Qradar_rule directory
        
        Returns:
            List of unique rule IDs from all CSV files
        """
        self.logger.info("Discovering rules from CSV files...")
        
        if not self.rule_dir.exists():
            self.logger.warning(f"Rule directory not found: {self.rule_dir}")
            return []
        
        # Find all CSV files
        rule_files = self._find_rule_files()
        if not rule_files:
            self.logger.warning("No rule CSV files found")
            return []
        
        all_rule_ids = []
        
        for file_path in rule_files:
            try:
                rule_ids = self._extract_rule_ids_from_csv(file_path)
                all_rule_ids.extend(rule_ids)
                self.logger.info(f"Loaded {len(rule_ids)} rules from {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                continue
        
        return list(set(all_rule_ids))
    
    def _find_rule_files(self) -> List[Path]:
        """
        Find all rule CSV files in the rule directory
        
        Returns:
            List of rule file paths
        """
        patterns = [
            "qradar_*.csv",
            "*.csv"
        ]
        
        rule_files = []
        for pattern in patterns:
            files = list(self.rule_dir.glob(pattern))
            rule_files.extend(files)
        
        # Remove duplicates and filter
        rule_files = list(set(rule_files))
        rule_files = [f for f in rule_files if f.name != "rule_mapping.json"]
        
        return rule_files
    
    def _extract_rule_ids_from_csv(self, file_path: Path) -> List[int]:
        """
        Extract rule IDs from a CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of rule IDs
        """
        rule_ids = []
        
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        rule_id = int(float(row['id']))
                        rule_ids.append(rule_id)
                    except (ValueError, KeyError):
                        continue
        
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
        
        return rule_ids
    
    def create_rule_mapping(self, force_refresh: bool = False) -> Dict:
        """
        Create unified rule mapping
        
        Args:
            force_refresh: Force recreation of mapping even if cached
            
        Returns:
            Rule mapping dictionary with rule_list and rule_to_index
        """
        if not force_refresh and self._rule_mapping:
            return self._rule_mapping
        
        # Discover rules
        rule_ids = self.discover_rules()
        if not rule_ids:
            raise ValueError("No rules discovered")
        
        # Create mapping
        unique_rules = sorted(set(rule_ids))
        rule_to_index = {rule_id: idx for idx, rule_id in enumerate(unique_rules)}
        
        # Create configuration
        mapping = {
            'rule_to_index': {int(k): int(v) for k, v in rule_to_index.items()},
            'rule_list': [int(r) for r in unique_rules],
            'total_rules': len(unique_rules),
            'mode': self.mode,
            'generated_at': str(datetime.now()),
            'generated_by': 'QRadarRuleManager'
        }
        
        # Cache the mapping
        self._rule_mapping = mapping
        self._rule_list = unique_rules
        self._rule_to_index = rule_to_index
        
        # Save to file
        self._save_rule_mapping(mapping)
        
        return mapping
    
    def _save_rule_mapping(self, mapping: Dict):
        """
        Save rule mapping to JSON file
        
        Args:
            mapping: Rule mapping dictionary
        """
        try:
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, default=str)
            
            self.logger.info(f"Rule mapping saved to {self.mapping_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save rule mapping: {e}")
    
    def load_rule_mapping(self) -> Dict:
        """
        Load existing rule mapping from file
        
        Returns:
            Rule mapping dictionary
        """
        try:
            if self.mapping_file.exists():
                with open(self.mapping_file, 'r') as f:
                    mapping = json.load(f)
                
                # Update cache
                self._rule_mapping = mapping
                self._rule_list = mapping['rule_list']
                self._rule_to_index = mapping['rule_to_index']
                
                self.logger.info(f"Loaded rule mapping from {self.mapping_file}")
                return mapping
            
        except Exception as e:
            self.logger.error(f"Failed to load rule mapping: {e}")
        
        # If loading fails, create new mapping
        return self.create_rule_mapping()
    
    def get_rule_list(self, refresh: bool = False) -> List[int]:
        """
        Get ordered list of rule IDs
        
        Args:
            refresh: Force refresh from source
            
        Returns:
            Ordered list of rule IDs
        """
        if refresh or not self._rule_list:
            mapping = self.create_rule_mapping(force_refresh=True)
            return mapping['rule_list']
        
        return self._rule_list
    
    def get_rule_to_index_map(self, refresh: bool = False) -> Dict[int, int]:
        """
        Get rule ID to index mapping
        
        Args:
            refresh: Force refresh from source
            
        Returns:
            Dictionary mapping rule IDs to indices
        """
        if refresh or not self._rule_to_index:
            mapping = self.create_rule_mapping(force_refresh=True)
            return mapping['rule_to_index']
        
        return self._rule_to_index
    
    def validate_mapping(self) -> bool:
        """
        Validate that rule mapping is complete and correct
        
        Returns:
            True if valid, False otherwise
        """
        try:
            mapping = self.load_rule_mapping()
            
            required_keys = ['rule_to_index', 'rule_list', 'total_rules']
            for key in required_keys:
                if key not in mapping:
                    self.logger.error(f"Missing key in mapping: {key}")
                    return False
            
            if mapping['total_rules'] == 0:
                self.logger.error("No rules found in mapping")
                return False
            
            self.logger.info(f"✅ Rule mapping validation passed - {mapping['total_rules']} rules")
            return True
            
        except Exception as e:
            self.logger.error(f"Mapping validation failed: {e}")
            return False
    
    def switch_mode(self, new_mode: str):
        """
        Switch between API and file modes
        
        Args:
            new_mode: 'api' or 'file'
        """
        self.mode = new_mode
        self._rule_list = None
        self._rule_to_index = None
        self._rule_mapping = None
        self.logger.info(f"Switched to {new_mode} mode")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about discovered rules
        
        Returns:
            Dictionary with statistics
        """
        try:
            mapping = self.load_rule_mapping()
            rule_files = self._find_rule_files()
            
            return {
                'total_rules': mapping['total_rules'],
                'mode': self.mode,
                'rule_files': len(rule_files),
                'source_files': [str(f.name) for f in rule_files],
                'mapping_file': str(self.mapping_file)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def load_uat_to_prod_mapping(self) -> Dict[int, int]:
        """
        Load UAT to Production rule ID mapping from CSV file
        
        Returns:
            Dictionary mapping UAT rule IDs to production rule IDs
        """
        if self._uat_to_prod_map is not None:
            return self._uat_to_prod_map
        
        mapping = {}
        
        try:
            if self.uat_mapping_file.exists():
                with open(self.uat_mapping_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            uat_rule_id = int(row['uat_rule_id'])
                            prod_rule_id = int(row['prod_rule_id'])
                            mapping[uat_rule_id] = prod_rule_id
                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"Invalid mapping row: {row} - {e}")
                            continue
                
                self.logger.info(f"Loaded {len(mapping)} UAT-to-Production mappings")
                self._uat_to_prod_map = mapping
            else:
                self.logger.warning(f"UAT mapping file not found: {self.uat_mapping_file}")
                self._uat_to_prod_map = {}
                
        except Exception as e:
            self.logger.error(f"Failed to load UAT mapping: {e}")
            self._uat_to_prod_map = {}
        
        return self._uat_to_prod_map
    
    def get_uat_to_prod_map(self) -> Dict[int, int]:
        """
        Get UAT to Production rule ID mapping dictionary
        
        Returns:
            Dictionary mapping UAT rule IDs to production rule IDs
        """
        return self.load_uat_to_prod_mapping()
    
    def get_production_rule_list(self, refresh: bool = False) -> List[int]:
        """
        Get ordered list of production rule IDs as baseline for feature vectors
        
        Args:
            refresh: Force refresh from source
            
        Returns:
            Ordered list of production rule IDs
        """
        if refresh or self._prod_rule_list is None:
            # Always use production rules as baseline
            if self.environment == 'uat':
                # For UAT environment, load production rules and apply mapping
                self.logger.info("Loading production rules as baseline for UAT training")
                
                # Discover current environment rules
                current_rules = self.discover_rules()
                
                # Load UAT-to-Production mapping
                uat_to_prod = self.get_uat_to_prod_map()
                
                # Map UAT rules to production rules
                prod_rules = set()
                for rule_id in current_rules:
                    if rule_id in uat_to_prod:
                        prod_rules.add(uat_to_prod[rule_id])
                    else:
                        # If no mapping exists, assume it's already production
                        prod_rules.add(rule_id)
                
                self._prod_rule_list = sorted(prod_rules)
            else:
                # Production environment - use discovered rules directly
                self._prod_rule_list = self.get_rule_list(refresh=refresh)
        
        return self._prod_rule_list
    
    def get_production_rule_to_index_map(self, refresh: bool = False) -> Dict[int, int]:
        """
        Get production rule ID to index mapping for feature vector construction
        
        Args:
            refresh: Force refresh from source
            
        Returns:
            Dictionary mapping production rule IDs to vector indices
        """
        if refresh or self._prod_rule_to_index is None:
            prod_rules = self.get_production_rule_list(refresh=refresh)
            self._prod_rule_to_index = {rule_id: idx for idx, rule_id in enumerate(prod_rules)}
        
        return self._prod_rule_to_index
    
    def map_uat_rule_to_prod(self, uat_rule_id: int) -> Optional[int]:
        """
        Map a single UAT rule ID to its corresponding production rule ID
        
        Args:
            uat_rule_id: UAT rule ID to map
            
        Returns:
            Production rule ID or None if no mapping exists
        """
        uat_to_prod = self.get_uat_to_prod_map()
        return uat_to_prod.get(uat_rule_id, uat_rule_id)
    
        return results

    def _load_family_mapping_config(self) -> Dict[str, str]:
        """
        Load explicit rule ID to family mapping from configuration file
        
        Returns:
            Dictionary mapping rule ID (str) to Family Name
        """
        try:
            if self.family_mapping_file.exists():
                with open(self.family_mapping_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('family_mapping', {})
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load family mapping config: {e}")
            return {}

    def _get_rule_family(self, rule_id: int, rule_name: str, explicit_mapping: Dict[str, str]) -> str:
        """
        Determine the family for a given rule based on ID or Name pattern
        
        Args:
            rule_id: Rule ID
            rule_name: Rule Name
            explicit_mapping: Dictionary of explicit ID->Family mappings
            
        Returns:
            Family name (e.g., "BOC_3xx", "BOC_300xx", "Uncategorized")
        """
        # 1. Explicit Mapping
        if str(rule_id) in explicit_mapping:
            return explicit_mapping[str(rule_id)]
            
        # 2. Pattern Matching (Name-based)
        # BOC_300xx pattern: BOC_300 followed by 2 digits (e.g., BOC_30001...)
        if re.match(r'^BOC_300\d{2}.*', rule_name):
            return "BOC_300xx"

        # BOC_3xx pattern: BOC_3 followed by 2 digits (e.g., BOC_303...)
        if re.match(r'^BOC_3\d{2}.*', rule_name):
            return "BOC_3xx"
            
        # 3. Fallback
        return "Uncategorized"

    def create_family_mapping(self, force_refresh: bool = False) -> Dict:
        """
        Create rule family mapping
        
        Args:
            force_refresh: Force recreation of mapping
            
        Returns:
            Dictionary containing family_to_index and rule_id_to_family
        """
        if not force_refresh and self._family_mapping:
            return self._family_mapping
            
        # Get all rules with metadata (need names)
        # We need to re-discover to get names, as simple ID list isn't enough
        # For efficiency, we'll assume discover_rules or _discover_from_csv 
        # can be adapted or we just re-read the CSVs/API here if needed.
        # But wait, discover_rules only returns IDs.
        # We need a method that returns full rule objects or at least ID+Name.
        
        # Let's fetch full rule details
        rules_data = []
        if self.mode == 'api':
            # In API mode, we might need to fetch again or cache. 
            # For now, let's rely on CSVs which are saved during API fetch.
            # If API fetch hasn't run, we might miss data. 
            # But typically discover_rules is called first.
            pass
        
        # Always read from CSVs as they are the local cache/source
        rule_files = self._find_rule_files()
        explicit_mapping = self._load_family_mapping_config()
        
        family_set = set()
        rule_id_to_family = {}
        
        for file_path in rule_files:
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            rule_id = int(float(row['id']))
                            rule_name = row.get('name', '')
                            
                            family = self._get_rule_family(rule_id, rule_name, explicit_mapping)
                            
                            rule_id_to_family[rule_id] = family
                            family_set.add(family)
                            
                        except (ValueError, KeyError):
                            continue
            except Exception as e:
                self.logger.error(f"Error reading {file_path} for family mapping: {e}")
        
        # Sort families to ensure consistent indexing
        # Remove Uncategorized from families list as they will be handled individually
        if "Uncategorized" in family_set:
            family_set.remove("Uncategorized")
            
        sorted_families = sorted(list(family_set))
        
        # Get list of uncategorized rule IDs
        uncategorized_rules = sorted([
            rid for rid, fam in rule_id_to_family.items() 
            if fam == "Uncategorized"
        ])
        
        family_to_index = {family: idx for idx, family in enumerate(sorted_families)}
        
        mapping = {
            'family_to_index': family_to_index,
            'rule_id_to_family': rule_id_to_family,
            'families': sorted_families,
            'uncategorized_rules': uncategorized_rules,
            'total_families': len(sorted_families),
            'total_uncategorized': len(uncategorized_rules),
            'generated_at': str(datetime.now())
        }
        
        self._family_mapping = mapping
        self._family_to_index = family_to_index
        self._rule_id_to_family = rule_id_to_family
        
        return mapping

    def get_uncategorized_rules(self, refresh: bool = False) -> List[int]:
        """
        Get list of rule IDs that are not in any specific family
        
        Args:
            refresh: Force refresh
            
        Returns:
            List of rule IDs
        """
        if refresh or not self._family_mapping:
            self.create_family_mapping(force_refresh=True)
        return self._family_mapping.get('uncategorized_rules', [])

    def get_family_to_index_map(self, refresh: bool = False) -> Dict[str, int]:
        """
        Get family name to index mapping
        
        Args:
            refresh: Force refresh
            
        Returns:
            Dictionary mapping Family Name to Index
        """
        if refresh or not self._family_to_index:
            mapping = self.create_family_mapping(force_refresh=True)
            return mapping['family_to_index']
        return self._family_to_index

    def get_rule_family(self, rule_id: int) -> str:
        """
        Get family for a specific rule ID
        
        Args:
            rule_id: Rule ID
            
        Returns:
            Family name
        """
        if not self._rule_id_to_family:
            self.create_family_mapping()
        return self._rule_id_to_family.get(rule_id, "Uncategorized")

    def get_rules_for_family(self, family_name: str, limit: Optional[int] = None) -> List[int]:
        """
        Get a sorted list of rule IDs that belong to the specified family.

        Args:
            family_name: Family name (e.g., "BOC_300xx").
            limit: Optional cap on how many rule IDs to return.

        Returns:
            List of rule IDs belonging to the family (possibly empty).
        """
        if not family_name:
            return []

        if not self._rule_id_to_family:
            self.create_family_mapping()

        matches = [rid for rid, fam in self._rule_id_to_family.items() if fam == family_name]
        matches.sort()

        if limit is not None:
            try:
                limit_val = max(int(limit), 0)
            except (TypeError, ValueError):
                limit_val = None
            if limit_val:
                return matches[:limit_val]
        return matches


# Convenience functions for backward compatibility
def get_rule_list(
    mode: str = 'file',
    config: Optional[Dict[str, Any]] = None,
    environment: str = 'prod',
) -> List[int]:
    """
    Get rule list using specified mode
    
    Args:
        mode: 'api' or 'file'
        config: Configuration dictionary
        environment: 'prod' or 'uat'
        
    Returns:
        List of rule IDs
    """
    manager = QRadarRuleManager(mode=mode, config=config, environment=environment)
    return manager.get_rule_list()


def get_rule_to_index_map(
    mode: str = 'file',
    config: Optional[Dict[str, Any]] = None,
    environment: str = 'prod',
) -> Dict[int, int]:
    """
    Get rule to index mapping using specified mode
    
    Args:
        mode: 'api' or 'file'
        config: Configuration dictionary
        environment: 'prod' or 'uat'
        
    Returns:
        Dictionary mapping rule IDs to indices
    """
    manager = QRadarRuleManager(mode=mode, config=config, environment=environment)
    return manager.get_rule_to_index_map()


def get_production_rule_list(
    mode: str = 'file',
    config: Optional[Dict[str, Any]] = None,
    environment: str = 'prod',
) -> List[int]:
    """
    Get production rule list as baseline for feature vectors
    
    Args:
        mode: 'api' or 'file'
        config: Configuration dictionary
        environment: 'prod' or 'uat'
        
    Returns:
        List of production rule IDs
    """
    manager = QRadarRuleManager(mode=mode, config=config, environment=environment)
    return manager.get_production_rule_list()


def get_production_rule_to_index_map(
    mode: str = 'file',
    config: Optional[Dict[str, Any]] = None,
    environment: str = 'prod',
) -> Dict[int, int]:
    """
    Get production rule ID to index mapping
    
    Args:
        mode: 'api' or 'file'
        config: Configuration dictionary
        environment: 'prod' or 'uat'
        
    Returns:
        Dictionary mapping production rule IDs to vector indices
    """
    manager = QRadarRuleManager(mode=mode, config=config, environment=environment)
    return manager.get_production_rule_to_index_map()


def get_uat_to_prod_map(config: Optional[Dict[str, Any]] = None) -> Dict[int, int]:
    """
    Get UAT to Production rule ID mapping
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping UAT rule IDs to production rule IDs
    """
    manager = QRadarRuleManager(config=config)
    return manager.get_uat_to_prod_map()


def get_family_to_index_map(
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, int]:
    """
    Get family to index mapping
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping Family Name to Index
    """
    manager = QRadarRuleManager(config=config)
    return manager.get_family_to_index_map()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified QRadar Rule Manager')
    parser.add_argument('mode', choices=['api', 'file'], help='Rule discovery mode')
    parser.add_argument('--validate', action='store_true', help='Validate existing mapping')
    parser.add_argument('--validate-mapping', action='store_true', help='Validate UAT-to-Production mapping')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--environment', choices=['prod', 'uat'], default='prod', help='Environment type')
    parser.add_argument('--uat-mapping', action='store_true', help='Show UAT-to-Production mappings')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create manager
    manager = QRadarRuleManager(mode=args.mode, environment=args.environment)
    
    if args.validate:
        manager.validate_mapping()
    elif args.validate_mapping:
        results = manager.validate_mapping_consistency()
        print(json.dumps(results, indent=2))
    elif args.stats:
        stats = manager.get_statistics()
        print(json.dumps(stats, indent=2))
    elif args.uat_mapping:
        mapping = manager.get_uat_to_prod_map()
        print(json.dumps({str(k): v for k, v in mapping.items()}, indent=2))
    else:
        # Create new mapping
        mapping = manager.create_rule_mapping(force_refresh=True)
        print(f"Created mapping with {mapping['total_rules']} rules")
        
        # Show production rule list for verification
        prod_rules = manager.get_production_rule_list()
        print(f"Production rule baseline: {len(prod_rules)} rules")
