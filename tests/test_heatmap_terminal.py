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
    
    # Generate explanations with heat map
    print("\n📊 Generating SHAP explanations with visualizations...")
    shap_values, plot_results = explainer.explain(
        test_instances, 
        log_results=True, 
        generate_plots=True,
        output_dir="./test_output"
    )
    
    # Test terminal visualization
    print("\n🖥️  Displaying results in terminal...")
    explainer.explain(
        test_instances[:3],  # Limit to 3 instances for terminal display
        log_results=False,
        show_terminal=True
    )
    
    # Test individual terminal methods
    print("\n🔥 Testing individual terminal heat map...")
    explainer.display_terminal_heatmap(test_instances[:2], top_features=8)
    
    print("\n📊 Testing terminal importance chart...")
    explainer.display_terminal_importance_chart(test_instances, top_features=8)
    
    # Test individual heat map generation
    print("\n🎨 Generating individual heat map...")
    heatmap_path = explainer.generate_heatmap(
        test_instances, 
        output_path="./test_output/custom_heatmap.png",
        top_features=12
    )
    
    # Test feature importance plot
    print("\n📈 Generating feature importance plot...")
    importance_path = explainer.generate_feature_importance_plot(
        test_instances,
        output_path="./test_output/importance_plot.png",
        top_features=10
    )
    
    # Generate comprehensive markdown report with visualizations
    print("\n📝 Generating comprehensive report with heat maps...")
    report = explainer.generate_markdown_report(
        test_instances,
        output_path="./test_output/threat_analysis_report.md",
        include_visualizations=True
    )
    
    # Show results
    print("\n✅ Heat Map Test Results:")
    print(f"  📊 Heat map saved: {heatmap_path}")
    print(f"  📈 Importance plot saved: {importance_path}")
    print(f"  📄 Report generated: ./test_output/threat_analysis_report.md")
    
    if plot_results and not plot_results.get('error'):
        print(f"  🎯 Summary visualizations: {plot_results}")
    
    # Show top features for verification
    ranking = explainer.get_feature_importance(test_instances)
    print(f"\n🔍 Top 5 Security Rules Detected:")
    for item in ranking[:5]:
        print(f"  {item['rank']}. {item['rule']} (score: {item['importance']:.4f})")
    
    print("\n🎉 Heat map generation test completed successfully!")
    
    # Show usage examples
    print("\n📚 USAGE EXAMPLES:")
    print("="*50)
    print("# Terminal output only:")
    print("explainer.explain(data, show_terminal=True)")
    print("\n# Both terminal and file output:")
    print("explainer.explain(data, show_terminal=True, generate_plots=True)")
    print("\n# Individual terminal components:")
    print("explainer.display_terminal_heatmap(data)")
    print("explainer.display_terminal_importance_chart(data)")
    print("explainer.display_terminal_summary(data)")
    print("="*50)
    
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
