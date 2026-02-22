"""Feature redundancy fixer for DataSentry library.

This module provides methods to fix feature redundancy issues.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from datasentry.core.base import BaseFixer, FixResult
from datasentry.core.utils import validate_data, convert_to_dataframe


class RedundancyFixer(BaseFixer):
    """Fix feature redundancy using various selection strategies.
    
    This fixer provides methods to address feature redundancy:
    - Remove: Remove highly correlated features
    - PCA: Apply Principal Component Analysis
    - SelectKBest: Select top k features
    - VarianceThreshold: Remove low variance features
    
    Attributes:
        method: Redundancy handling method.
        correlation_threshold: Threshold for removing correlated features.
        n_components: Number of components for PCA.
    
    Example:
        >>> fixer = RedundancyFixer(method='remove')
        >>> X = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [4, 5, 6]})
        >>> result = fixer.fix(X)
        >>> print(result.success)
        True
    """
    
    def __init__(
        self,
        method: str = "remove",
        correlation_threshold: float = 0.95,
        features_to_remove: Optional[List[str]] = None,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.0,
        keep_features: Optional[List[str]] = None
    ):
        """Initialize the redundancy fixer.
        
        Args:
            method: Handling method ('remove', 'pca', 'variance_threshold', 'select_kbest').
            correlation_threshold: Threshold for removing correlated features.
            features_to_remove: Specific features to remove.
            n_components: Number of components for PCA.
            variance_threshold: Minimum variance for variance threshold method.
            keep_features: Features to always keep.
        """
        super().__init__("RedundancyFixer")
        self.method = method.lower()
        self.correlation_threshold = correlation_threshold
        self.features_to_remove = features_to_remove or []
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.keep_features = keep_features or []
        
        self._removed_features: List[str] = []
        self._selector: Optional[Any] = None
    
    def fix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ) -> FixResult:
        """Fix feature redundancy in the dataset.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional, required for some methods).
            **kwargs: Additional fixer-specific parameters.
                - features_to_remove: Override features to remove.
        
        Returns:
            FixResult with transformed data and operation details.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        
        # Store original type
        X_is_df = isinstance(X, pd.DataFrame)
        df = convert_to_dataframe(X)
        
        original_n_features = len(df.columns)
        
        try:
            # Apply fix based on method
            if self.method == "remove":
                features_to_remove = kwargs.get('features_to_remove', self.features_to_remove)
                df_fixed, details = self._remove_features(df, features_to_remove)
            elif self.method == "pca":
                df_fixed, details = self._apply_pca(df)
            elif self.method == "variance_threshold":
                df_fixed, details = self._variance_threshold(df)
            elif self.method == "select_kbest":
                if y is None:
                    raise ValueError("y is required for select_kbest method")
                df_fixed, details = self._select_kbest(df, y)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Convert back to original type
            if X_is_df:
                X_transformed = df_fixed
            else:
                X_transformed = df_fixed.values
            
            details.update({
                "original_features": original_n_features,
                "final_features": len(df_fixed.columns) if hasattr(df_fixed, 'columns') else df_fixed.shape[1],
                "features_removed": original_n_features - (len(df_fixed.columns) if hasattr(df_fixed, 'columns') else df_fixed.shape[1]),
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
                details={"error": str(e), "method": self.method},
                warnings=[f"Failed to fix redundancy: {str(e)}"],
            )
        
        self._last_result = result
        return result
    
    def _remove_features(
        self,
        df: pd.DataFrame,
        features_to_remove: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove specified or auto-detected redundant features.
        
        Args:
            df: Input DataFrame.
            features_to_remove: Features to remove (if empty, auto-detect).
        
        Returns:
            Tuple of (df_clean, details).
        """
        if not features_to_remove:
            # Auto-detect highly correlated features
            features_to_remove = self._detect_correlated_features(df)
        
        # Filter to features that exist and aren't in keep_features
        features_to_remove = [
            f for f in features_to_remove 
            if f in df.columns and f not in self.keep_features
        ]
        
        df_clean = df.drop(columns=features_to_remove)
        
        self._removed_features = features_to_remove
        
        details = {
            "method": "remove",
            "features_removed": features_to_remove,
            "n_features_removed": len(features_to_remove),
            "auto_detected": len(self.features_to_remove) == 0,
        }
        
        return df_clean, details
    
    def _detect_correlated_features(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect highly correlated features to remove.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            List of features to remove.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return []
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()
        
        # Find high correlations
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Identify features to drop (greedy approach)
        to_drop = set()
        
        for col in upper_triangle.columns:
            high_corr = upper_triangle[col][upper_triangle[col] > self.correlation_threshold]
            for idx in high_corr.index:
                if idx not in self.keep_features and col not in self.keep_features:
                    # Drop the feature with more high correlations
                    col_corr_count = (upper_triangle[col] > self.correlation_threshold).sum()
                    idx_corr_count = (corr_matrix.loc[idx] > self.correlation_threshold).sum()
                    
                    if col_corr_count >= idx_corr_count:
                        to_drop.add(col)
                    else:
                        to_drop.add(idx)
        
        return list(to_drop)
    
    def _apply_pca(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply PCA for dimensionality reduction.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Tuple of (df_pca, details).
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        categorical_df = df.select_dtypes(exclude=[np.number])
        
        if numeric_df.empty:
            raise ValueError("PCA requires numeric features")
        
        # Standardize
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_df)
        
        # Determine number of components
        n_components = self.n_components
        if n_components is None:
            n_components = min(10, numeric_df.shape[1], numeric_df.shape[0] - 1)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_scaled)
        
        # Create DataFrame with PCA components
        pca_columns = [f"PC{i+1}" for i in range(pca_result.shape[1])]
        df_pca = pd.DataFrame(pca_result, columns=pca_columns, index=df.index)
        
        # Add categorical columns back if any
        if not categorical_df.empty:
            df_pca = pd.concat([df_pca, categorical_df], axis=1)
        
        self._selector = pca
        
        details = {
            "method": "pca",
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_variance_explained": round(float(sum(pca.explained_variance_ratio_)), 4),
            "numeric_features_transformed": list(numeric_df.columns),
        }
        
        return df_pca, details
    
    def _variance_threshold(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove features with low variance.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Tuple of (df_filtered, details).
        """
        from sklearn.feature_selection import VarianceThreshold
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        categorical_df = df.select_dtypes(exclude=[np.number])
        
        if numeric_df.empty:
            return df, {"method": "variance_threshold", "note": "No numeric features"}
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(numeric_df)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = numeric_df.columns[selected_mask].tolist()
        removed_features = numeric_df.columns[~selected_mask].tolist()
        
        # Filter to features that aren't in keep_features
        removed_features = [f for f in removed_features if f not in self.keep_features]
        final_selected = [f for f in selected_features] + [f for f in removed_features if f in self.keep_features]
        
        df_filtered = numeric_df[final_selected].copy()
        
        # Add categorical columns back
        if not categorical_df.empty:
            df_filtered = pd.concat([df_filtered, categorical_df], axis=1)
        
        self._removed_features = removed_features
        self._selector = selector
        
        details = {
            "method": "variance_threshold",
            "threshold": self.variance_threshold,
            "features_removed": removed_features,
            "n_features_removed": len(removed_features),
            "variances": {col: round(float(var), 6) for col, var in zip(numeric_df.columns, selector.variances_)},
        }
        
        return df_filtered, details
    
    def _select_kbest(
        self,
        df: pd.DataFrame,
        y: Union[np.ndarray, pd.Series]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select top k features using statistical tests.
        
        Args:
            df: Input DataFrame.
            y: Target vector.
        
        Returns:
            Tuple of (df_selected, details).
        """
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        categorical_df = df.select_dtypes(exclude=[np.number])
        
        if numeric_df.empty:
            return df, {"method": "select_kbest", "note": "No numeric features"}
        
        # Determine k
        k = self.n_components
        if k is None:
            k = max(1, numeric_df.shape[1] // 2)
        k = min(k, numeric_df.shape[1])
        
        # Determine if classification or regression
        y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)
        is_classification = len(np.unique(y_array)) < 10
        
        score_func = f_classif if is_classification else f_regression
        
        # Apply selection
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(numeric_df, y_array)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = numeric_df.columns[selected_mask].tolist()
        removed_features = numeric_df.columns[~selected_mask].tolist()
        
        df_selected = numeric_df[selected_features].copy()
        
        # Add categorical columns back
        if not categorical_df.empty:
            df_selected = pd.concat([df_selected, categorical_df], axis=1)
        
        self._removed_features = removed_features
        self._selector = selector
        
        details = {
            "method": "select_kbest",
            "k": k,
            "score_function": score_func.__name__,
            "features_removed": removed_features,
            "n_features_removed": len(removed_features),
            "feature_scores": {col: round(float(score), 4) for col, score in zip(numeric_df.columns, selector.scores_)},
        }
        
        return df_selected, details
    
    def get_removed_features(self) -> List[str]:
        """Get list of features removed in last fix operation.
        
        Returns:
            List of removed feature names.
        """
        return self._removed_features
