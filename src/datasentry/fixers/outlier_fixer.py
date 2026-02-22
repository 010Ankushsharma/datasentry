"""Outlier fixer for DataSentry library.

This module provides methods to fix outlier issues.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from datasentry.core.base import BaseFixer, FixResult
from datasentry.core.utils import validate_data, convert_to_dataframe


class OutlierFixer(BaseFixer):
    """Fix outliers using various strategies.
    
    This fixer provides multiple methods to handle outliers:
    - Remove: Remove outlier samples
    - Cap: Cap values at percentiles
    - Transform: Apply log or other transformations
    - Winsorize: Limit extreme values
    
    Attributes:
        method: Outlier handling method.
        lower_percentile: Lower percentile for capping.
        upper_percentile: Upper percentile for capping.
    
    Example:
        >>> fixer = OutlierFixer(method='cap')
        >>> X = np.array([[1], [2], [3], [100], [2], [3]])  # 100 is outlier
        >>> result = fixer.fix(X)
        >>> print(result.success)
        True
    """
    
    def __init__(
        self,
        method: str = "cap",
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
        outlier_indices: Optional[List[int]] = None,
        transform_method: str = "log",
        winsorize_limits: Tuple[float, float] = (0.05, 0.05)
    ):
        """Initialize the outlier fixer.
        
        Args:
            method: Handling method ('remove', 'cap', 'transform', 'winsorize').
            lower_percentile: Lower percentile for capping.
            upper_percentile: Upper percentile for capping.
            outlier_indices: Specific indices to treat as outliers.
            transform_method: Transformation method ('log', 'sqrt', 'boxcox').
            winsorize_limits: (lower, upper) limits for winsorization.
        """
        super().__init__("OutlierFixer")
        self.method = method.lower()
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.outlier_indices = outlier_indices
        self.transform_method = transform_method
        self.winsorize_limits = winsorize_limits
    
    def fix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ) -> FixResult:
        """Fix outliers in the dataset.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            **kwargs: Additional fixer-specific parameters.
                - outlier_indices: Override specific indices to handle.
        
        Returns:
            FixResult with transformed data and operation details.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        
        # Store original type
        X_is_df = isinstance(X, pd.DataFrame)
        df = convert_to_dataframe(X)
        
        original_n_samples = len(df)
        
        # Get outlier indices from kwargs or instance
        outlier_indices = kwargs.get('outlier_indices', self.outlier_indices)
        
        try:
            # Apply fix based on method
            if self.method == "remove":
                df_fixed, y_fixed, details = self._remove_outliers(
                    df, y, outlier_indices
                )
            elif self.method == "cap":
                df_fixed, details = self._cap_outliers(df)
                y_fixed = y
            elif self.method == "transform":
                df_fixed, details = self._transform_outliers(df)
                y_fixed = y
            elif self.method == "winsorize":
                df_fixed, details = self._winsorize(df)
                y_fixed = y
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Convert back to original type
            if X_is_df:
                X_transformed = df_fixed
            else:
                X_transformed = df_fixed.values
            
            details.update({
                "original_samples": original_n_samples,
                "final_samples": len(df_fixed),
                "samples_affected": original_n_samples - len(df_fixed) if self.method == "remove" else 0,
            })
            
            result = FixResult(
                fixer_name=self.name,
                success=True,
                X_transformed=X_transformed,
                y_transformed=y_fixed,
                details=details,
            )
            
        except Exception as e:
            result = FixResult(
                fixer_name=self.name,
                success=False,
                X_transformed=X,
                y_transformed=y,
                details={"error": str(e), "method": self.method},
                warnings=[f"Failed to fix outliers: {str(e)}"],
            )
        
        self._last_result = result
        return result
    
    def _remove_outliers(
        self,
        df: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]],
        outlier_indices: Optional[List[int]]
    ) -> Tuple[pd.DataFrame, Optional[Union[np.ndarray, pd.Series]], Dict[str, Any]]:
        """Remove outlier samples.
        
        Args:
            df: Input DataFrame.
            y: Target vector.
            outlier_indices: Indices of outliers to remove.
        
        Returns:
            Tuple of (df_clean, y_clean, details).
        """
        if outlier_indices is None:
            # Auto-detect outliers using IQR method
            outlier_indices = self._detect_outliers_iqr(df)
        
        # Create mask for non-outliers
        clean_mask = np.ones(len(df), dtype=bool)
        clean_mask[outlier_indices] = False
        
        df_clean = df.iloc[clean_mask].reset_index(drop=True)
        
        y_clean = None
        if y is not None:
            if isinstance(y, pd.Series):
                y_clean = y.iloc[clean_mask].reset_index(drop=True)
            else:
                y_clean = y[clean_mask]
        
        details = {
            "method": "remove",
            "outliers_removed": len(outlier_indices),
            "outlier_indices": outlier_indices[:100] if len(outlier_indices) <= 100 else outlier_indices[:100].tolist(),
        }
        
        return df_clean, y_clean, details
    
    def _detect_outliers_iqr(self, df: pd.DataFrame) -> List[int]:
        """Detect outliers using IQR method.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            List of outlier indices.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        outlier_mask = pd.Series(False, index=df.index)
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            col_outliers = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
            outlier_mask = outlier_mask | col_outliers
        
        return df.index[outlier_mask].tolist()
    
    def _cap_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cap outlier values at percentiles.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Tuple of (df_capped, details).
        """
        df_capped = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        cap_values = {}
        
        for col in numeric_cols:
            lower_val = df[col].quantile(self.lower_percentile)
            upper_val = df[col].quantile(self.upper_percentile)
            
            df_capped[col] = df_capped[col].clip(lower=lower_val, upper=upper_val)
            
            cap_values[col] = {
                "lower": round(float(lower_val), 4),
                "upper": round(float(upper_val), 4),
            }
        
        details = {
            "method": "cap",
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "cap_values": cap_values,
        }
        
        return df_capped, details
    
    def _transform_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply transformation to reduce outlier impact.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Tuple of (df_transformed, details).
        """
        df_transformed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if self.transform_method == "log":
                # Add constant to handle negative/zero values
                min_val = col_data.min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                df_transformed[col] = np.log1p(df[col] + offset)
            
            elif self.transform_method == "sqrt":
                # Add constant to handle negative values
                min_val = col_data.min()
                offset = abs(min_val) + 1 if min_val < 0 else 0
                df_transformed[col] = np.sqrt(df[col] + offset)
            
            elif self.transform_method == "boxcox":
                from scipy import stats
                # Only for positive values
                min_val = col_data.min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                df_transformed[col], _ = stats.boxcox(df[col] + offset)
        
        details = {
            "method": "transform",
            "transform_method": self.transform_method,
            "columns_transformed": list(numeric_cols),
        }
        
        return df_transformed, details
    
    def _winsorize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply winsorization to limit extreme values.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Tuple of (df_winsorized, details).
        """
        from scipy.stats import mstats
        
        df_winsorized = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        winsorize_stats = {}
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            # Calculate limits
            lower_limit = self.winsorize_limits[0]
            upper_limit = self.winsorize_limits[1]
            
            # Apply winsorization
            winsorized = mstats.winsorize(col_data, limits=self.winsorize_limits)
            
            # Update dataframe
            df_winsorized.loc[col_data.index, col] = winsorized
            
            winsorize_stats[col] = {
                "lower_limit": lower_limit,
                "upper_limit": upper_limit,
                "min_before": round(float(col_data.min()), 4),
                "max_before": round(float(col_data.max()), 4),
                "min_after": round(float(winsorized.min()), 4),
                "max_after": round(float(winsorized.max()), 4),
            }
        
        details = {
            "method": "winsorize",
            "limits": self.winsorize_limits,
            "statistics": winsorize_stats,
        }
        
        return df_winsorized, details
