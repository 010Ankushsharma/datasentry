"""Basic example of using DataSentry.

This example demonstrates the basic usage of DataSentry for data quality
detection and remediation.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from datasentry import DataSentry


def create_sample_data():
    """Create sample data with various quality issues."""
    np.random.seed(42)
    
    # Create imbalanced classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        weights=[0.7, 0.2, 0.1],  # Imbalanced
        flip_y=0.05,  # Some label noise
        random_state=42
    )
    
    # Introduce missing values
    missing_mask = np.random.random(X.shape) < 0.05
    X[missing_mask] = np.nan
    
    # Add outliers
    outlier_indices = np.random.choice(len(X), size=20, replace=False)
    X[outlier_indices] = X[outlier_indices] * 5
    
    # Create test data with shift
    X_test = X[:200] + np.random.randn(200, 10) * 0.5
    
    return X, y, X_test


def main():
    """Run the basic example."""
    print("=" * 60)
    print("DataSentry Basic Example")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data with quality issues...")
    X_train, y_train, X_test = create_sample_data()
    print(f"   Training data shape: {X_train.shape}")
    print(f"   Test data shape: {X_test.shape}")
    
    # Initialize DataSentry
    print("\n2. Initializing DataSentry...")
    ds = DataSentry(random_state=42, verbose=True)
    
    # Generate comprehensive report
    print("\n3. Generating comprehensive data quality report...")
    report = ds.generate_full_report(X_train, y_train, X_test=X_test)
    
    # Display report summary
    print("\n" + "=" * 60)
    print("REPORT SUMMARY")
    print("=" * 60)
    
    meta = report["report_metadata"]
    print(f"Overall Status: {meta['overall_status']}")
    print(f"Health Score: {meta['health_score']:.2%}")
    print(f"Issues Detected: {meta['issues_detected']}")
    print(f"Checks Performed: {meta['total_detectors']}")
    
    # Display severity distribution
    print("\nSeverity Distribution:")
    for severity, count in report["severity_distribution"].items():
        print(f"  {severity}: {count}")
    
    # Display detected issues
    print("\nDetected Issues:")
    for result in report["detailed_results"]:
        print(f"\n  [{result['severity']}] {result['detector_name']}")
        print(f"    Score: {result['score']:.3f}")
        if result['recommendations']:
            print(f"    Recommendations: {result['recommendations'][0]}")
    
    # Fix issues
    print("\n" + "=" * 60)
    print("FIXING ISSUES")
    print("=" * 60)
    
    fix_config = {
        'missing_values': {'strategy': 'mean'},
        'outliers': {'method': 'cap'},
        'imbalance': {'method': 'smote'},
        'redundancy': {'method': 'remove'},
    }
    
    print("\n4. Fixing detected issues...")
    X_clean, y_clean = ds.fix_all(X_train, y_train, fix_config=fix_config)
    
    print(f"\n   Original training data: {X_train.shape}")
    print(f"   Cleaned training data: {X_clean.shape}")
    if y_clean is not None:
        print(f"   Original target distribution: {np.bincount(y_train)}")
        print(f"   Cleaned target distribution: {np.bincount(y_clean)}")
    
    # Generate report after fixing
    print("\n5. Generating report after fixing...")
    report_clean = ds.generate_full_report(X_clean, y_clean, X_test=X_test)
    
    print(f"\n   Health Score (before): {report['report_metadata']['health_score']:.2%}")
    print(f"   Health Score (after): {report_clean['report_metadata']['health_score']:.2%}")
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
