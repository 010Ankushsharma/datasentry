"""Pytest configuration and fixtures for DataSentry tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Create sample clean data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    return X, y


@pytest.fixture
def imbalanced_data():
    """Create imbalanced classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # Create imbalanced target: 80% class 0, 15% class 1, 5% class 2
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    return X, y


@pytest.fixture
def data_with_missing():
    """Create data with missing values."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # Introduce missing values
    missing_mask = np.random.random(X.shape) < 0.1
    X[missing_mask] = np.nan
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def data_with_outliers():
    """Create data with outliers."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # Add outliers
    X[0] = [10, 10, 10]
    X[1] = [-10, -10, -10]
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def data_with_duplicates():
    """Create data with duplicate rows."""
    np.random.seed(42)
    X_base = np.random.randn(50, 5)
    X = np.vstack([X_base, X_base[:10]])  # Add 10 duplicates
    y = np.random.randint(0, 2, 60)
    return X, y


@pytest.fixture
def data_with_redundancy():
    """Create data with redundant features."""
    np.random.seed(42)
    X_base = np.random.randn(100, 3)
    # Create highly correlated features
    X_col4 = X_base[:, 0] * 0.95 + np.random.randn(100) * 0.05
    X = np.column_stack([X_base, X_col4])
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def shifted_data():
    """Create train/test data with distribution shift."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(50, 5) * 2 + 1  # Different distribution
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 50)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def sample_dataframe():
    """Create sample pandas DataFrame."""
    np.random.seed(42)
    df = pd.DataFrame({
        'feature_a': np.random.randn(100),
        'feature_b': np.random.randn(100),
        'feature_c': np.random.randint(0, 10, 100),
        'feature_d': np.random.choice(['A', 'B', 'C'], 100),
    })
    y = pd.Series(np.random.randint(0, 2, 100), name='target')
    return df, y
