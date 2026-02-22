"""Data leakage fixer for DataSentry library.

This module provides methods to fix data leakage issues.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from datasentry.core.base import BaseFixer, FixResult
from datasentry.core.utils import validate_data, convert_to_dataframe


class DataLeakageFixer(BaseFixer):
    """Fix data leakage issues in datasets.
    
    This fixer provides methods to address data leakage:
    - Remove: Remove features that leak target information
    - Deduplicate: Remove duplicate samples
    - Split: Ensure proper train-test split order
    
    Attributes:
        method: Fixing method ('remove_features', 'deduplicate', 'temporal_split').
        features_to_remove: Specific features to remove.
    
    Example:
        >>> fixer = DataLeakageFixer(method='remove_features')
        >>> X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'id': [1, 2, 3]})
        >>> y = np.array([0, 1, 0])
        >>> result = fixer.fix(X, y, features_to_remove=['id'])
    """
    
    def __init__(
        self,
        method: str = "remove_features",
        features_to_remove: Optional[List[str]] = None,
        remove_duplicates: bool = True,
        keep: str = "first"
    ):
        """Initialize the data leakage fixer.
        
        Args:
            method: Fixing method ('remove_features', 'deduplicate', 'temporal_split').
            features_to_remove: List of feature names to remove.
            remove_duplicates: Whether to remove duplicate rows.
            keep: Which duplicates to keep ('first', 'last', False).
        """
        super().__init__("DataLeakageFixer")
        self.method = method
        self.features_to_remove = features_to_remove or []
        self.remove_duplicates = remove_duplicates
        self.keep = keep
    
    def fix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ) -> FixResult:
        """Fix data leakage in the dataset.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            **kwargs: Additional fixer-specific parameters.
                - features_to_remove: Override list of features to remove.
                - duplicate_indices: Specific duplicate indices to remove.
        
        Returns:
            FixResult with transformed data and operation details.
        """
        # Validate inputs
        if y is not None:
            X, y = validate_data(X, y)
        else:
            X, _ = validate_data(X)
        
        # Store original types
        X_is_df = isinstance(X, pd.DataFrame)
        df = convert_to_dataframe(X)
        
        original_n_samples = len(df)
        original_n_features = len(df.columns)
        
        try:
            details = {"actions": []}
            
            # Get features to remove from kwargs or instance variable
            features_to_remove = kwargs.get('features_to_remove', self.features_to_remove)
            
            # Remove features with target leakage
            if features_to_remove:
                df, remove_details = self._remove_features(df, features_to_remove)
                details["actions"].append(remove_details)
            
            # Remove duplicates
            if self.remove_duplicates:
                duplicate_indices = kwargs.get('duplicate_indices')
                df, dedup_details = self._remove_duplicates(df, duplicate_indices)
                details["actions"].append(dedup_details)
            
            # Convert back to original type
            if X_is_df:
                X_transformed = df
            else:
                X_transformed = df.values
            
            # Update details
            details.update({
                "original_samples": original_n_samples,
                "final_samples": len(df),
                "original_features": original_n_features,
                "final_features": len(df.columns),
                "samples_removed": original_n_samples - len(df),
                "features_removed": original_n_features - len(df.columns),
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
                warnings=[f"Failed to fix data leakage: {str(e)}"],
            )
        
        self._last_result = result
        return result
    
    def _remove_features(
        self,
        df: pd.DataFrame,
        features_to_remove: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove specified features.
        
        Args:
            df: Input DataFrame.
            features_to_remove: List of feature names to remove.
        
        Returns:
            Tuple of (df_clean, details).
        """
        # Filter to features that exist in the dataframe
        existing_features = [f for f in features_to_remove if f in df.columns]
        missing_features = [f for f in features_to_remove if f not in df.columns]
        
        df_clean = df.drop(columns=existing_features)
        
        details = {
            "action": "remove_features",
            "features_removed": existing_features,
            "n_features_removed": len(existing_features),
            "features_not_found": missing_features if missing_features else None,
        }
        
        return df_clean, details
    
    def _remove_duplicates(
        self,
        df: pd.DataFrame,
        duplicate_indices: Optional[List[int]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove duplicate rows.
        
        Args:
            df: Input DataFrame.
            duplicate_indices: Specific indices to remove (optional).
        
        Returns:
            Tuple of (df_clean, details).
        """
        original_count = len(df)
        
        if duplicate_indices is not None:
            # Remove specific indices
            df_clean = df.drop(index=duplicate_indices)
            n_duplicates = len(duplicate_indices)
        else:
            # Auto-detect and remove duplicates
            df_clean = df.drop_duplicates(keep=self.keep)
            n_duplicates = original_count - len(df_clean)
        
        details = {
            "action": "remove_duplicates",
            "duplicates_removed": n_duplicates,
            "keep": self.keep,
        }
        
        return df_clean, details
    
    def fix_train_test_contamination(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Tuple[FixResult, FixResult]:
        """Fix train-test contamination by removing overlapping samples.
        
        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training targets (optional).
            y_test: Test targets (optional).
        
        Returns:
            Tuple of (train_result, test_result).
        """
        # Convert to DataFrames
        train_df = convert_to_dataframe(X_train)
        test_df = convert_to_dataframe(X_test)
        
        # Find overlapping samples
        common_cols = list(set(train_df.columns) & set(test_df.columns))
        train_subset = train_df[common_cols].reset_index(drop=True)
        test_subset = test_df[common_cols].reset_index(drop=True)
        
        # Find exact matches
        merged = test_subset.merge(
            train_subset,
            on=list(common_cols),
            how='left',
            indicator=True
        )
        
        contaminated_mask = merged['_merge'] == 'both'
        contaminated_indices = test_df.index[contaminated_mask].tolist()
        
        # Remove contaminated samples from test set
        test_clean = test_df.drop(index=contaminated_indices)
        
        # Create results
        train_result = FixResult(
            fixer_name=self.name,
            success=True,
            X_transformed=X_train,
            y_transformed=y_train,
            details={
                "action": "train_test_contamination_check",
                "contaminated_samples_in_test": len(contaminated_indices),
            },
        )
        
        test_result = FixResult(
            fixer_name=self.name,
            success=True,
            X_transformed=test_clean.values if isinstance(X_test, np.ndarray) else test_clean,
            y_transformed=y_test.drop(index=contaminated_indices) if y_test is not None else None,
            details={
                "action": "remove_contaminated_test_samples",
                "samples_removed": len(contaminated_indices),
                "contaminated_indices": contaminated_indices[:100],  # Limit
            },
        )
        
        return train_result, test_result
