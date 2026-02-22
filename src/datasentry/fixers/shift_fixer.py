"""Data shift fixer for DataSentry library.

This module provides methods to address data distribution shift issues.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from datasentry.core.base import BaseFixer, FixResult
from datasentry.core.utils import validate_data, convert_to_dataframe


class ShiftFixer(BaseFixer):
    """Fix data distribution shift using various adaptation strategies.
    
    This fixer provides methods to address data shift:
    - Standardize: Apply standardization to both datasets
    - Align: Align distributions using quantile transformation
    - Reweight: Sample reweighting for covariate shift
    - Adapt: Domain adaptation techniques
    
    Note: Some shift types (like concept shift) cannot be fixed without
    retraining the model on new data.
    
    Attributes:
        method: Shift adaptation method.
    
    Example:
        >>> fixer = ShiftFixer(method='standardize')
        >>> X_train = np.random.randn(100, 5)
        >>> X_test = np.random.randn(50, 5) * 2
        >>> result = fixer.fix(X_train, X_test=X_test)
    """
    
    def __init__(
        self,
        method: str = "standardize",
        n_quantiles: int = 1000,
        output_distribution: str = "normal"
    ):
        """Initialize the shift fixer.
        
        Args:
            method: Adaptation method ('standardize', 'quantile', 'robust').
            n_quantiles: Number of quantiles for quantile transformation.
            output_distribution: Output distribution for quantile transform.
        """
        super().__init__("ShiftFixer")
        self.method = method.lower()
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        
        self._scaler: Optional[Any] = None
    
    def fix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ) -> FixResult:
        """Fix data shift in the dataset.
        
        Args:
            X: Training feature matrix.
            y: Training target vector (optional).
            **kwargs: Additional fixer-specific parameters.
                - X_test: Test data to align with training data.
        
        Returns:
            FixResult with transformed data and operation details.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        
        # Store original type
        X_is_df = isinstance(X, pd.DataFrame)
        df_train = convert_to_dataframe(X)
        
        # Get test data if provided
        X_test = kwargs.get('X_test')
        
        try:
            if X_test is not None:
                # Fix both train and test
                X_test, _ = validate_data(X_test)
                df_test = convert_to_dataframe(X_test)
                
                df_train_fixed, df_test_fixed, details = self._fix_with_test(
                    df_train, df_test
                )
                
                # Convert back to original types
                if X_is_df:
                    X_transformed = df_train_fixed
                    X_test_transformed = df_test_fixed
                else:
                    X_transformed = df_train_fixed.values
                    X_test_transformed = df_test_fixed.values
                
                details["X_test_transformed"] = X_test_transformed
                
            else:
                # Fix only training data
                df_train_fixed, details = self._fix_train_only(df_train)
                
                # Convert back to original type
                if X_is_df:
                    X_transformed = df_train_fixed
                else:
                    X_transformed = df_train_fixed.values
            
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
                details={"error": str(e), "method": self.method},
                warnings=[f"Failed to fix shift: {str(e)}"],
            )
        
        self._last_result = result
        return result
    
    def _fix_train_only(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fix training data only.
        
        Args:
            df: Training DataFrame.
        
        Returns:
            Tuple of (df_fixed, details).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        df_fixed = df.copy()
        
        if self.method == "standardize":
            scaler = StandardScaler()
            df_fixed[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self._scaler = scaler
            
            details = {
                "method": "standardize",
                "columns_transformed": list(numeric_cols),
                "transform_description": "Zero mean, unit variance standardization",
            }
        
        elif self.method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            df_fixed[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self._scaler = scaler
            
            details = {
                "method": "robust",
                "columns_transformed": list(numeric_cols),
                "transform_description": "Median and IQR based scaling",
            }
        
        elif self.method == "quantile":
            from sklearn.preprocessing import QuantileTransformer
            transformer = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, len(df)),
                output_distribution=self.output_distribution,
                random_state=42
            )
            df_fixed[numeric_cols] = transformer.fit_transform(df[numeric_cols])
            self._scaler = transformer
            
            details = {
                "method": "quantile",
                "n_quantiles": self.n_quantiles,
                "output_distribution": self.output_distribution,
                "columns_transformed": list(numeric_cols),
            }
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return df_fixed, details
    
    def _fix_with_test(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Fix both training and test data.
        
        Args:
            df_train: Training DataFrame.
            df_test: Test DataFrame.
        
        Returns:
            Tuple of (df_train_fixed, df_test_fixed, details).
        """
        # Ensure same columns
        common_cols = list(set(df_train.columns) & set(df_test.columns))
        numeric_cols = df_train[common_cols].select_dtypes(include=[np.number]).columns
        
        df_train_fixed = df_train.copy()
        df_test_fixed = df_test.copy()
        
        if self.method == "standardize":
            scaler = StandardScaler()
            df_train_fixed[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
            df_test_fixed[numeric_cols] = scaler.transform(df_test[numeric_cols])
            self._scaler = scaler
            
            details = {
                "method": "standardize",
                "columns_transformed": list(numeric_cols),
                "train_mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                "train_scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
            }
        
        elif self.method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            df_train_fixed[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
            df_test_fixed[numeric_cols] = scaler.transform(df_test[numeric_cols])
            self._scaler = scaler
            
            details = {
                "method": "robust",
                "columns_transformed": list(numeric_cols),
            }
        
        elif self.method == "quantile":
            from sklearn.preprocessing import QuantileTransformer
            
            # Combine train and test for fitting
            combined = pd.concat([df_train[numeric_cols], df_test[numeric_cols]], axis=0)
            
            transformer = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, len(combined)),
                output_distribution=self.output_distribution,
                random_state=42
            )
            transformer.fit(combined)
            
            df_train_fixed[numeric_cols] = transformer.transform(df_train[numeric_cols])
            df_test_fixed[numeric_cols] = transformer.transform(df_test[numeric_cols])
            self._scaler = transformer
            
            details = {
                "method": "quantile",
                "n_quantiles": self.n_quantiles,
                "output_distribution": self.output_distribution,
                "columns_transformed": list(numeric_cols),
            }
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return df_train_fixed, df_test_fixed, details
    
    def transform_test(self, X_test: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform new test data using fitted scaler.
        
        Args:
            X_test: Test data to transform.
        
        Returns:
            Transformed test data.
        """
        if self._scaler is None:
            raise ValueError("Fixer has not been fitted yet. Call fix() first.")
        
        X_is_df = isinstance(X_test, pd.DataFrame)
        df_test = convert_to_dataframe(X_test)
        numeric_cols = df_test.select_dtypes(include=[np.number]).columns
        
        df_test_transformed = df_test.copy()
        df_test_transformed[numeric_cols] = self._scaler.transform(df_test[numeric_cols])
        
        if X_is_df:
            return df_test_transformed
        else:
            return df_test_transformed.values
