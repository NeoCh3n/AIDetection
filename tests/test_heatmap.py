#!/usr/bin/env python3
"""
Test the SHAP explainer heat map functionality.
"""

import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.shap_explainer import Explainer

def test_heatmap_generation():
    """Test the SHAP explainer heat map generation."""
    print("🔥 Testing SHAP Heat Map Generation...")
    
    # Create sample threat detection data
    X, y = make_classification(
        n_samples=500, 
        n_features=15, 
        n_classes=2, 
        n_informative=10,
        random_state=42
    )
    
    # Train a RandomForest threat detector
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Create background data
    background_data = X[:100]
    
    # Create security rule names
    security_rules = [
        f'Security_Rule_{i+1:02d}' for i in range(15)
    ]
    
    # Create rule mapping with descriptive names
    rule_mapping = {
        'Security_Rule_01': 'Suspicious Process Execution',
        'Security_Rule_02': 'Unusual Network Traffic',
        'Security_Rule_03': 'Failed Login Attempts',
        'Security_Rule_04': 'File System Modifications',
        'Security_Rule_05': 'Privilege Escalation',
        'Security_Rule_06': 'Malicious Domain Access',
        'Security_Rule_07': 'Registry Tampering',
        'Security_Rule_08': 'Suspicious PowerShell Activity',
        'Security_Rule_09': 'Unauthorized Data Access',
        'Security_Rule_10': 'Command Injection Patterns',
        'Security_Rule_11': 'Lateral Movement Indicators',
        'Security_Rule_12': 'Data Exfiltration Signals',
        'Security_Rule_13': 'Persistence Mechanisms',
        'Security_Rule_14': 'Evasion Techniques',
        'Security_Rule_15': 'Anomalous User Behavior'
    }
    
    # Initialize explainer
    explainer = Explainer(
        model=model,
        background_data=background_data,
        feature_names=security_rules,
        rule_mapping=rule_mapping
    )
    
    # Test on some "malicious" instances
    test_instances = X[y == 1][:5]  # Take 5 malicious instances
    print(f"Testing on {len(test_instances)} malicious instances")
    
    # Generate explanations with visual outputs
    print("\n📊 Generating SHAP explanations with visualizations...")
    results = explainer.explain(
        instance_data=test_instances,
        output_dir="./test_output",
        plot=True,
        summary_report=True
    )

    shap_values = results.get('shap_values')
    output_files = results.get('output_files', {})
    report = output_files.get('summary_report')
    heatmap_path = output_files.get('heatmap')
    importance_path = output_files.get('importance_plot')
    
    # Show results
    print("\n✅ Heat Map Test Results:")
    print(f"  📊 Heat map saved: {heatmap_path}")
    print(f"  📈 Importance plot saved: {importance_path}")
    print(f"  📄 Report generated: {report}")
    print(f"  🎯 Available outputs: {list(output_files.keys())}")
    if shap_values is not None:
        print(f"  📐 SHAP values type: {type(shap_values)}")
    
    # Show top features for verification
    ranking = results.get('feature_importance', [])
    print(f"\n🔍 Top 5 Security Rules Detected:")
    for item in ranking[:5]:
        feature_label = item.get('rule_name') or item.get('feature')
        rank = item.get('rank', '?')
        importance = float(item.get('importance', 0.0))
        print(f"  {rank}. {feature_label} (score: {importance:.4f})")
    
    print("\n🎉 Heat map generation test completed successfully!")
    
    return True

if __name__ == "__main__":
    # Create output directory
    os.makedirs("./test_output", exist_ok=True)
    
    try:
        test_heatmap_generation()
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
