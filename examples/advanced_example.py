"""Advanced example of using DataSentry.

This example demonstrates advanced features including custom configurations,
individual detector usage, and visualization.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from datasentry import (
    DataSentry,
    ImbalanceDetector,
    MissingValueDetector,
    OutlierDetector,
    ImbalanceFixer,
    MissingValueFixer,
    OutlierFixer,
)
from datasentry.core.base import SeverityLevel


def create_complex_data():
    """Create complex data with multiple quality issues."""
    np.random.seed(42)
    
    # Create imbalanced data
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=4,
        weights=[0.6, 0.25, 0.1, 0.05],
        flip_y=0.08,
        random_state=42
    )
    
    # Add feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')
    
    # Introduce missing values with pattern
    missing_cols = ['feature_0', 'feature_5', 'feature_10']
    for col in missing_cols:
        missing_mask = np.random.random(len(X)) < 0.1
        X.loc[missing_mask, col] = np.nan
    
    # Add outliers to specific features
    outlier_features = ['feature_1', 'feature_6', 'feature_11']
    for col in outlier_features:
        outlier_indices = np.random.choice(len(X), size=30, replace=False)
        X.loc[outlier_indices, col] = X[col].mean() + X[col].std() * 5
    
    # Create shifted test data
    X_test = X.iloc[:400].copy()
    X_test = X_test + np.random.randn(*X_test.shape) * 0.8
    
    return X, y, X_test


def demonstrate_individual_detectors(X, y, X_test):
    """Demonstrate using individual detectors."""
    print("\n" + "=" * 60)
    print("INDIVIDUAL DETECTORS")
    print("=" * 60)
    
    # Imbalance detection with custom threshold
    print("\n1. Imbalance Detection (custom threshold=2.0)")
    imbalance_det = ImbalanceDetector(imbalance_threshold=2.0)
    result = imbalance_det.detect(X, y)
    print(f"   Issue Detected: {result.issue_detected}")
    print(f"   Imbalance Ratio: {result.details.get('imbalance_ratio', 'N/A')}")
    print(f"   Severity: {result.severity}")
    
    # Missing value detection
    print("\n2. Missing Value Detection")
    missing_det = MissingValueDetector(missing_threshold=0.05)
    result = missing_det.detect(X, y)
    print(f"   Issue Detected: {result.issue_detected}")
    print(f"   Total Missing: {result.details.get('total_missing', 0)}")
    print(f"   Missing Ratio: {result.details.get('overall_missing_ratio', 0):.2%}")
    if result.details.get('high_missing_features'):
        print(f"   High Missing Features: {len(result.details['high_missing_features'])}")
    
    # Outlier detection with different methods
    print("\n3. Outlier Detection (Isolation Forest)")
    outlier_det = OutlierDetector(method='isolation_forest', contamination=0.05)
    result = outlier_det.detect(X, y)
    print(f"   Issue Detected: {result.issue_detected}")
    print(f"   Outliers Found: {result.details.get('n_outliers', 0)}")
    print(f"   Outlier Ratio: {result.details.get('outlier_ratio', 0):.2%}")


def demonstrate_individual_fixers(X, y):
    """Demonstrate using individual fixers."""
    print("\n" + "=" * 60)
    print("INDIVIDUAL FIXERS")
    print("=" * 60)
    
    # Fix missing values with KNN imputation
    print("\n1. Fixing Missing Values (KNN Imputation)")
    missing_fixer = MissingValueFixer(strategy='knn', k_neighbors=5)
    result = missing_fixer.fix(X, y)
    print(f"   Success: {result.success}")
    print(f"   Values Imputed: {result.details.get('values_imputed', 0)}")
    X_missing_fixed = result.X_transformed
    
    # Fix outliers with winsorization
    print("\n2. Fixing Outliers (Winsorization)")
    outlier_fixer = OutlierFixer(method='winsorize', winsorize_limits=(0.05, 0.05))
    result = outlier_fixer.fix(X_missing_fixed, y)
    print(f"   Success: {result.success}")
    X_outlier_fixed = result.X_transformed
    
    # Fix imbalance with SMOTE
    print("\n3. Fixing Imbalance (SMOTE)")
    imbalance_fixer = ImbalanceFixer(method='smote')
    result = imbalance_fixer.fix(X_outlier_fixed, y)
    print(f"   Success: {result.success}")
    print(f"   Original Samples: {result.details.get('original_samples', 0)}")
    print(f"   New Samples: {result.details.get('new_samples', 0)}")
    
    return result.X_transformed, result.y_transformed


def demonstrate_custom_workflow(X, y, X_test):
    """Demonstrate custom detection and fixing workflow."""
    print("\n" + "=" * 60)
    print("CUSTOM WORKFLOW")
    print("=" * 60)
    
    ds = DataSentry(random_state=42, verbose=False)
    
    # Step 1: Detect only critical issues
    print("\n1. Detecting critical issues only...")
    results = ds.detect_all(X, y, X_test)
    
    critical_issues = [
        name for name, result in results.items()
        if result.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
    ]
    
    print(f"   Critical/High Issues: {critical_issues}")
    
    # Step 2: Fix only detected issues
    print("\n2. Fixing detected issues only...")
    fix_config = {}
    
    if 'missing_values' in critical_issues:
        fix_config['missing_values'] = {'strategy': 'median'}
    
    if 'outliers' in critical_issues:
        fix_config['outliers'] = {'method': 'cap'}
    
    if 'imbalance' in critical_issues:
        fix_config['imbalance'] = {'method': 'class_weights'}
    
    if fix_config:
        X_fixed, y_fixed = ds.fix_all(X, y, fix_config=fix_config)
        print(f"   Applied fixes: {list(fix_config.keys())}")
    else:
        X_fixed, y_fixed = X, y
        print("   No critical issues to fix")
    
    # Step 3: Re-evaluate
    print("\n3. Re-evaluating after fixes...")
    new_results = ds.detect_all(X_fixed, y_fixed, X_test)
    
    remaining_critical = [
        name for name, result in new_results.items()
        if result.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
    ]
    
    print(f"   Remaining Critical/High Issues: {remaining_critical}")


def demonstrate_report_export(X, y, X_test):
    """Demonstrate report export functionality."""
    print("\n" + "=" * 60)
    print("REPORT EXPORT")
    print("=" * 60)
    
    ds = DataSentry(random_state=42, verbose=False)
    
    # Generate report
    report = ds.generate_full_report(X, y, X_test=X_test)
    
    # Export as JSON
    import json
    json_report = json.dumps(report, indent=2)
    print(f"\n1. JSON Report Length: {len(json_report)} characters")
    
    # Save to file (demonstration - commented out)
    # with open('report.json', 'w') as f:
    #     f.write(json_report)
    # print("   Saved to: report.json")
    
    # Generate HTML report
    from datasentry.core.report import ReportGenerator
    
    detectors = ds.detect_all(X, y, X_test)
    report_gen = ReportGenerator(list(detectors.values()))
    html_report = report_gen.to_html()
    print(f"\n2. HTML Report Length: {len(html_report)} characters")
    
    # Save to file (demonstration - commented out)
    # report_gen.save_html('report.html')
    # print("   Saved to: report.html")


def main():
    """Run the advanced example."""
    print("=" * 60)
    print("DataSentry Advanced Example")
    print("=" * 60)
    
    # Create complex data
    print("\nCreating complex data with multiple quality issues...")
    X, y, X_test = create_complex_data()
    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Demonstrate individual detectors
    demonstrate_individual_detectors(X, y, X_test)
    
    # Demonstrate individual fixers
    X_fixed, y_fixed = demonstrate_individual_fixers(X, y)
    
    # Demonstrate custom workflow
    demonstrate_custom_workflow(X, y, X_test)
    
    # Demonstrate report export
    demonstrate_report_export(X, y, X_test)
    
    print("\n" + "=" * 60)
    print("Advanced Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
