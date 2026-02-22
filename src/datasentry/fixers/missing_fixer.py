"""Missing value fixer for DataSentry library.

This module provides methods to fix missing value issues.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from datasentry.core.base import BaseFixer, FixResult
from datasentry.core.utils import validate_data, convert_to_dataframe


class MissingValueFixer(BaseFixer):
    """Fix missing values using various imputation strategies.
    
    This fixer provides multiple methods to handle missing values:
    - Simple: Mean, median, mode, constant imputation
    - KNN: K-nearest neighbors imputation
    - Iterative: MICE (Multivariate Imputation by Chained Equations)
    - Forward/Backward fill: For time series data
    - Drop: Remove rows or columns with missing values
    
    Attributes:
        strategy: Imputation strategy.
        fill_value: Value for constant imputation.
    
    Example:
        >>> fixer = MissingValueFixer(strategy='mean')
        >>> X = pd.DataFrame({'a': [1, 2, None], 'b': [4, None, 6]})
        >>> result = fixer.fix(X)
        >>> print(result.success)
        True
    """
    
    def __init__(
        self,
        strategy: str = "mean",
        fill_value: Optional[Any] = None,
        k_neighbors: int = 5,
        max_iter: int = 10,
        drop_threshold: float = 0.5,
        drop_axis: str = "rows"
    ):
        """Initialize the missing value fixer.
        
        Args:
            strategy: Imputation strategy ('mean', 'median', 'most_frequent',
                'constant', 'knn', 'iterative', 'forward', 'backward', 'drop').
            fill_value: Value for constant imputation.
            k_neighbors: Number of neighbors for KNN imputation.
            max_iter: Maximum iterations for iterative imputation.
            drop_threshold: Threshold for dropping columns (ratio of missing).
            drop_axis: Axis to drop ('rows' or 'columns').
        """
        super().__init__("MissingValueFixer")
        self.strategy = strategy
        self.fill_value = fill_value
        self.k_neighbors = k_neighbors
        self.max_iter = max_iter
        self.drop_threshold = drop_threshold
        self.drop_axis = drop_axis
        
        self._imputer: Optional[Any] = None
        self._imputation_values: Optional[Dict[str, Any]] = None
    
    def fix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ) -> FixResult:
        """Fix missing values in the dataset.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            **kwargs: Additional fixer-specific parameters.
        
        Returns:
            FixResult with transformed data and operation details.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        
        # Store original type
        X_is_df = isinstance(X, pd.DataFrame)
        df = convert_to_dataframe(X)
        
        original_n_samples = len(df)
        original_n_features = len(df.columns)
        
        # Calculate missing statistics before
        missing_before = df.isnull().sum().sum()
        
        try:
            # Apply imputation based on strategy
            if self.strategy in ["mean", "median", "most_frequent", "constant"]:
                df_imputed, details = self._simple_impute(df)
            elif self.strategy == "knn":
                df_imputed, details = self._knn_impute(df)
            elif self.strategy == "iterative":
                df_imputed, details = self._iterative_impute(df)
            elif self.strategy in ["forward", "backward"]:
                df_imputed, details = self._fill_direction(df)
            elif self.strategy == "drop":
                df_imputed, details = self._drop_missing(df)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            # Convert back to original type
            if X_is_df:
                X_transformed = df_imputed
            else:
                X_transformed = df_imputed.values
            
            # Calculate statistics
            missing_after = df_imputed.isnull().sum().sum()
            
            details.update({
                "original_samples": original_n_samples,
                "final_samples": len(df_imputed),
                "original_features": original_n_features,
                "final_features": len(df_imputed.columns),
                "missing_before": int(missing_before),
                "missing_after": int(missing_after),
                "values_imputed": int(missing_before - missing_after),
            })
            
            result = FixResult(
                fixer_name=self.name,
                success=True,
                X_transformed=X_transformed,
                y_transformed=y,
                details=details,
            )
            
        except Exception as e:
            result = FixResult(
                fixer_name=self.name,
                success=False,
                X_transformed=X,
                y_transformed=y,
                details={"error": str(e), "strategy": self.strategy},
                warnings=[f"Failed to impute missing values: {str(e)}"],
            )
        
        self._last_result = result
        return result
    
    def _simple_impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply simple imputation strategies.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Tuple of (df_imputed, details).
        """
        df_imputed = df.copy()
        imputation_values = {}
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            if self.strategy in ["mean", "median", "most_frequent"]:
                imputer = SimpleImputer(strategy=self.strategy)
                df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                
                # Store imputation values
                if hasattr(imputer, 'statistics_'):
                    for col, val in zip(numeric_cols, imputer.statistics_):
                        imputation_values[col] = val
        
        # Impute categorical columns with most frequent
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[categorical_cols] = imputer.fit_transform(df[categorical_cols])
            
            for col, val in zip(categorical_cols, imputer.statistics_):
                imputation_values[col] = val
        
        # Handle constant strategy
        if self.strategy == "constant":
            df_imputed = df.fillna(self.fill_value)
            imputation_values = {col: self.fill_value for col in df.columns}
        
        self._imputation_values = imputation_values
        
        details = {
            "strategy": self.strategy,
            "imputation_values": imputation_values,
            "numeric_strategy": self.strategy if self.strategy != "constant" else "constant",
            "categorical_strategy": "most_frequent",
        }
        
        return df_imputed, details
    
    def _knn_impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply KNN imputation.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Tuple of (df_imputed, details).
        """
        # KNN only works with numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        df_imputed = df.copy()
        
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=self.k_neighbors)
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            self._imputer = imputer
        
        # Handle categorical with mode
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        
        details = {
            "strategy": "knn",
            "k_neighbors": self.k_neighbors,
            "numeric_columns": list(numeric_cols),
            "categorical_strategy": "most_frequent",
        }
        
        return df_imputed, details
    
    def _iterative_impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply iterative (MICE) imputation.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Tuple of (df_imputed, details).
        """
        # Iterative only works with numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        df_imputed = df.copy()
        
        if len(numeric_cols) > 0:
            imputer = IterativeImputer(
                max_iter=self.max_iter,
                random_state=42
            )
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            self._imputer = imputer
        
        # Handle categorical with mode
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        
        details = {
            "strategy": "iterative",
            "max_iter": self.max_iter,
            "numeric_columns": list(numeric_cols),
            "categorical_strategy": "most_frequent",
        }
        
        return df_imputed, details
    
    def _fill_direction(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply forward or backward fill.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Tuple of (df_imputed, details).
        """
        if self.strategy == "forward":
            df_imputed = df.fillna(method='ffill')
            method = "forward fill"
        else:  # backward
            df_imputed = df.fillna(method='bfill')
            method = "backward fill"
        
        # Fill any remaining NaN with column mean/mode
        df_imputed = df_imputed.fillna(df.median(numeric_only=True))
        
        details = {
            "strategy": self.strategy,
            "method": method,
        }
        
        return df_imputed, details
    
    def _drop_missing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Drop rows or columns with missing values.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Tuple of (df_clean, details).
        """
        original_shape = df.shape
        
        if self.drop_axis == "rows":
            # Drop rows with missing values
            df_clean = df.dropna()
            dropped = original_shape[0] - len(df_clean)
            details = {
                "strategy": "drop",
                "axis": "rows",
                "rows_dropped": dropped,
            }
        else:  # columns
            # Drop columns with too many missing values
            missing_ratio = df.isnull().mean()
            cols_to_drop = missing_ratio[missing_ratio > self.drop_threshold].index
            df_clean = df.drop(columns=cols_to_drop)
            
            details = {
                "strategy": "drop",
                "axis": "columns",
                "columns_dropped": list(cols_to_drop),
                "n_columns_dropped": len(cols_to_drop),
                "threshold": self.drop_threshold,
            }
        
        return df_clean, details
    
    def get_imputation_values(self) -> Optional[Dict[str, Any]]:
        """Get the values used for imputation.
        
        Returns:
            Dictionary mapping column names to imputation values.
        """
        return self._imputation_values
