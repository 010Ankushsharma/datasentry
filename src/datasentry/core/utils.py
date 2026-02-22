"""Utility functions for DataSentry library.

This module provides common utility functions used across the library.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def validate_data(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    accept_sparse: bool = False,
    min_samples: int = 1,
    min_features: int = 1,
) -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[Union[np.ndarray, pd.Series]]]:
    """Validate input data for detectors and fixers.
    
    Args:
        X: Feature matrix to validate.
        y: Target vector to validate (optional).
        accept_sparse: Whether to accept sparse matrices.
        min_samples: Minimum number of samples required.
        min_features: Minimum number of features required.
    
    Returns:
        Tuple of validated (X, y).
    
    Raises:
        TypeError: If X or y has invalid type.
        ValueError: If data doesn't meet requirements.
    
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 1, 0])
        >>> X_val, y_val = validate_data(X, y)
    """
    # Validate X
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError(
            f"X must be a numpy array or pandas DataFrame, got {type(X).__name__}"
        )
    
    # Convert to numpy if needed for easier validation
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    
    # Check dimensions
    if X_array.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {X_array.ndim} dimensions")
    
    n_samples, n_features = X_array.shape
    
    if n_samples < min_samples:
        raise ValueError(
            f"X must have at least {min_samples} samples, got {n_samples}"
        )
    
    if n_features < min_features:
        raise ValueError(
            f"X must have at least {min_features} features, got {n_features}"
        )
    
    # Check for empty data
    if np.all(pd.isna(X_array)):
        raise ValueError("X contains all missing values")
    
    # Validate y if provided
    if y is not None:
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError(
                f"y must be a numpy array or pandas Series, got {type(y).__name__}"
            )
        
        y_array = y.values if isinstance(y, pd.Series) else y
        
        if y_array.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got {y_array.ndim} dimensions")
        
        if len(y_array) != n_samples:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X has {n_samples}, y has {len(y_array)}"
            )
    
    return X, y


def get_feature_names(
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None
) -> List[str]:
    """Get feature names from data or generate default names.
    
    Args:
        X: Feature matrix.
        feature_names: Optional list of feature names.
    
    Returns:
        List of feature names.
    
    Example:
        >>> X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> names = get_feature_names(X)
        >>> print(names)
        ['a', 'b']
    """
    if feature_names is not None:
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names must have length {n_features}, got {len(feature_names)}"
            )
        return list(feature_names)
    
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    
    # Generate default names for numpy arrays
    n_features = X.shape[1]
    return [f"feature_{i}" for i in range(n_features)]


def convert_to_dataframe(
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Convert input to pandas DataFrame.
    
    Args:
        X: Input data (numpy array or DataFrame).
        feature_names: Optional feature names.
    
    Returns:
        Pandas DataFrame.
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()
    
    columns = get_feature_names(X, feature_names)
    return pd.DataFrame(X, columns=columns)


def convert_to_series(
    y: Union[np.ndarray, pd.Series],
    name: str = "target"
) -> pd.Series:
    """Convert input to pandas Series.
    
    Args:
        y: Input target (numpy array or Series).
        name: Name for the series.
    
    Returns:
        Pandas Series.
    """
    if isinstance(y, pd.Series):
        return y.copy()
    
    return pd.Series(y, name=name)


def calculate_missing_ratio(
    X: Union[np.ndarray, pd.DataFrame]
) -> pd.Series:
    """Calculate the ratio of missing values per feature.
    
    Args:
        X: Feature matrix.
    
    Returns:
        Series with missing value ratios per feature.
    """
    df = convert_to_dataframe(X)
    return df.isnull().mean()


def detect_column_types(
    X: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """Detect numeric and categorical columns in a DataFrame.
    
    Args:
        X: Input DataFrame.
    
    Returns:
        Tuple of (numeric_columns, categorical_columns).
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        a: Numerator.
        b: Denominator.
        default: Value to return if b is zero.
    
    Returns:
        a / b or default if b is zero.
    """
    return a / b if b != 0 else default


def get_class_distribution(y: Union[np.ndarray, pd.Series]) -> pd.Series:
    """Get the distribution of classes in target variable.
    
    Args:
        y: Target vector.
    
    Returns:
        Series with class counts.
    """
    series = convert_to_series(y)
    return series.value_counts().sort_index()


def check_array_type(X: Union[np.ndarray, pd.DataFrame]) -> str:
    """Check the type of array provided.
    
    Args:
        X: Input data.
    
    Returns:
        String indicating the type ('numpy', 'pandas', or 'other').
    """
    if isinstance(X, pd.DataFrame):
        return "pandas"
    elif isinstance(X, np.ndarray):
        return "numpy"
    else:
        return "other"


def ensure_2d(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """Ensure array is 2-dimensional.
    
    Args:
        X: Input data.
    
    Returns:
        2D version of X.
    """
    if isinstance(X, pd.DataFrame):
        return X
    
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        return X_arr.reshape(-1, 1)
    return X_arr


def sample_data(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    n_samples: int = 10000,
    random_state: Optional[int] = None
) -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[Union[np.ndarray, pd.Series]]]:
    """Sample data if it exceeds a certain size.
    
    Args:
        X: Feature matrix.
        y: Target vector (optional).
        n_samples: Maximum number of samples.
        random_state: Random seed for reproducibility.
    
    Returns:
        Sampled (X, y) or just X if y is None.
    """
    n_total = len(X)
    
    if n_total <= n_samples:
        return X, y
    
    rng = np.random.RandomState(random_state)
    indices = rng.choice(n_total, size=n_samples, replace=False)
    
    if isinstance(X, pd.DataFrame):
        X_sampled = X.iloc[indices]
    else:
        X_sampled = X[indices]
    
    if y is not None:
        if isinstance(y, pd.Series):
            y_sampled = y.iloc[indices]
        else:
            y_sampled = y[indices]
        return X_sampled, y_sampled
    
    return X_sampled, None
