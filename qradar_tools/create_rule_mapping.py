#!/usr/bin/env python3
"""
Create Rule Mapping - Portable System

Automatically creates rule mapping from all rule files found in Qradar_rule folder.
This ensures the system works across different computers without hardcoded paths.
"""

import os
import json
import csv
import glob
from pathlib import Path

def get_project_root():
    """Get the absolute path of the project root directory"""
    return Path(__file__).parent.absolute()

def find_rule_files(rule_dir=None):
    """
    Find all rule CSV files in the rule directory
    
    Args:
        rule_dir: Directory to search for rule files (default: Qradar_rule)
    
    Returns:
        list: List of rule file paths
    """
    if rule_dir is None:
        rule_dir = get_project_root() / "Qradar_rule"
    
    rule_dir = Path(rule_dir)
    
    if not rule_dir.exists():
        print(f"Rule directory not found: {rule_dir}")
        return []
    
    # Find all CSV files that might contain rules
    rule_files = []
    patterns = [
        "qradar_*.csv",
        "*.csv"
    ]
    
    for pattern in patterns:
        files = list(rule_dir.glob(pattern))
        rule_files.extend(files)
    
    # Remove duplicates and filter out non-rule files
    rule_files = list(set(rule_files))
    rule_files = [f for f in rule_files if f.name != "rule_mapping.json"]
    
    return rule_files

def extract_rule_ids_from_csv(file_path):
    """Extract rule IDs from a CSV file"""
    rule_ids = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rule_id = int(float(row['id']))
                    rule_ids.append(rule_id)
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return rule_ids

def create_rule_mapping(rule_files=None):
    """
    Create rule mapping from all rule files
    
    Args:
        rule_files: List of rule file paths (auto-detected if None)
    
    Returns:
        dict: Rule mapping configuration
    """
    if rule_files is None:
        rule_files = find_rule_files()
    
    if not rule_files:
        print("No rule files found!")
        return None
    
    print(f"Found {len(rule_files)} rule files:")
    for f in rule_files:
        print(f"  - {f.name}")
    
    # Collect all rule IDs
    all_rule_ids = []
    file_stats = {}
    
    for file_path in rule_files:
        rule_ids = extract_rule_ids_from_csv(file_path)
        all_rule_ids.extend(rule_ids)
        file_stats[file_path.name] = len(rule_ids)
        print(f"  {file_path.name}: {len(rule_ids)} rules")
    
    # Create unique sorted list
    unique_rules = sorted(set(all_rule_ids))
    
    # Create mapping
    rule_to_index = {rule_id: idx for idx, rule_id in enumerate(unique_rules)}
    
    # Create configuration
    mapping = {
        'rule_to_index': {int(k): int(v) for k, v in rule_to_index.items()},
        'rule_list': [int(r) for r in unique_rules],
        'total_rules': len(unique_rules),
        'source_files': [str(Path(f).name) for f in rule_files],
        'file_stats': file_stats,
        'generated_at': str(Path.cwd()),
        'generated_by': 'create_rule_mapping.py'
    }
    
    return mapping

def save_rule_mapping(mapping, output_file=None):
    """Save rule mapping to JSON file"""
    if output_file is None:
        output_file = get_project_root() / "rule_mapping.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Rule mapping saved to: {output_file}")
    print(f"Total rules mapped: {mapping['total_rules']}")
    return str(output_file)

def validate_rule_mapping():
    """Validate that rule mapping is complete and correct"""
    project_root = get_project_root()
    mapping_file = project_root / "rule_mapping.json"
    rule_dir = project_root / "Qradar_rule"
    
    if not mapping_file.exists():
        print("❌ rule_mapping.json not found")
        return False
    
    if not rule_dir.exists():
        print("❌ Qradar_rule directory not found")
        return False
    
    # Load mapping
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
    except Exception as e:
        print(f"❌ Error loading rule mapping: {e}")
        return False
    
    # Validate structure
    required_keys = ['rule_to_index', 'rule_list', 'total_rules']
    for key in required_keys:
        if key not in mapping:
            print(f"❌ Missing key in mapping: {key}")
            return False
    
    # Check rule files exist
    rule_files = find_rule_files()
    if not rule_files:
        print("❌ No rule CSV files found")
        return False
    
    print("✅ Rule mapping validation passed")
    print(f"  - {mapping['total_rules']} total rules")
    print(f"  - {len(rule_files)} rule files")
    
    return True

def main():
    """CLI interface"""
    print("=== Creating QRadar Rule Mapping ===")
    print(f"Project root: {get_project_root()}")
    
    # Create mapping
    mapping = create_rule_mapping()
    if mapping is None:
        print("Failed to create rule mapping")
        return False
    
    # Save mapping
    save_rule_mapping(mapping)
    
    # Validate
    validate_rule_mapping()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)