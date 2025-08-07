#!/usr/bin/env python3
"""
Complete System Setup - Portable QRadar Rule Processing

This script provides a complete, portable setup for the QRadar anomaly detection system.
It handles:
1. Fetching rules from QRadar (all 3 endpoints)
2. Creating unified rule mapping
3. Validating system setup
4. Providing usage instructions
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_python_version():
    """Check Python version compatibility"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 6:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.6+")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\nChecking dependencies...")
    required_packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'shap',
        'joblib',
        'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def setup_directories():
    """Ensure required directories exist"""
    print("\nSetting up directories...")
    
    directories = [
        "Qradar_rule",
        "Model",
        "running_log"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")

def check_rule_files():
    """Check if rule files exist"""
    print("\nChecking rule files...")
    
    rule_dir = Path("Qradar_rule")
    if not rule_dir.exists():
        print("❌ Qradar_rule directory not found")
        return False
    
    rule_files = list(rule_dir.glob("*.csv"))
    if not rule_files:
        print("❌ No rule CSV files found in Qradar_rule/")
        print("Run: python get_rule.py (after configuring QRadar credentials)")
        return False
    
    print(f"✅ Found {len(rule_files)} rule files:")
    for f in rule_files:
        print(f"   {f.name}")
    
    return True

def create_rule_mapping():
    """Create rule mapping if it doesn't exist"""
    print("\nCreating rule mapping...")
    
    mapping_file = Path("rule_mapping.json")
    
    if mapping_file.exists():
        print("✅ rule_mapping.json already exists")
        return True
    
    if not check_rule_files():
        return False
    
    try:
        subprocess.run([sys.executable, "create_rule_mapping.py"], check=True)
        if mapping_file.exists():
            print("✅ rule_mapping.json created")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create rule mapping: {e}")
        return False
    
    return False

def show_next_steps():
    """Display next steps for the user"""
    print_header("Next Steps")
    
    print("1. Configure QRadar connection:")
    print("   Edit Qradar_rule/get_rule.py and set:")
    print("   - QRADAR_HOST (e.g., https://your-qradar-server)")
    print("   - API_TOKEN (from QRadar Admin → Authorized Services)")
    print()
    
    print("2. Download rules:")
    print("   python get_rule.py")
    print()
    
    print("3. Create rule mapping:")
    print("   python create_rule_mapping.py")
    print()
    
    print("4. Transform your data:")
    print("   python qradar_to_matrix.py your_data.csv")
    print()
    
    print("5. Train the model:")
    print("   python train_ML_model.py")
    print()

def main():
    """Main setup process"""
    print_header("QRadar Anomaly Detection System Setup")
    
    # Check environment
    if not check_python_version():
        return False
    
    if not check_dependencies():
        return False
    
    # Setup directories
    setup_directories()
    
    # Check rule files
    rule_files_ok = check_rule_files()
    
    # Create rule mapping
    mapping_ok = create_rule_mapping()
    
    # Summary
    print_header("Setup Summary")
    
    if rule_files_ok and mapping_ok:
        print("✅ System ready for use!")
        print("\nYou can now:")
        print("- Use existing rule files and mapping")
        print("- Or download new rules with get_rule.py")
    elif not rule_files_ok:
        print("⚠️  Rule files missing")
        print("   Run: python get_rule.py (after configuring)")
    elif not mapping_ok:
        print("⚠️  Rule mapping missing")
        print("   Run: python create_rule_mapping.py")
    
    show_next_steps()
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)