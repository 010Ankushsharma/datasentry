"""Class imbalance fixer for DataSentry library.

This module provides methods to fix class imbalance issues.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from datasentry.core.base import BaseFixer, FixResult, SeverityLevel
from datasentry.core.utils import validate_data, get_class_distribution


class ImbalanceFixer(BaseFixer):
    """Fix class imbalance using various resampling strategies.
    
    This fixer provides multiple methods to address class imbalance:
    - Oversampling: SMOTE, Random Oversampling
    - Undersampling: Random Undersampling
    - Hybrid: SMOTEENN, SMOTETomek
    - Class weights: Adjust model loss function
    
    Attributes:
        method: Resampling method to use.
        sampling_strategy: Target ratio or strategy for resampling.
        random_state: Random seed for reproducibility.
    
    Example:
        >>> fixer = ImbalanceFixer(method='smote')
        >>> X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
        >>> y = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # Imbalanced
        >>> result = fixer.fix(X, y)
        >>> print(result.success)
        True
    """
    
    def __init__(
        self,
        method: str = "smote",
        sampling_strategy: Union[str, float, Dict] = "auto",
        random_state: Optional[int] = 42,
        k_neighbors: int = 5,
        **kwargs
    ):
        """Initialize the imbalance fixer.
        
        Args:
            method: Resampling method ('smote', 'adasyn', 'random_over',
                'random_under', 'smoteenn', 'smotetomek', 'class_weights').
            sampling_strategy: Target ratio or strategy for resampling.
                'auto', 'majority', 'not minority', float, or dict.
            random_state: Random seed for reproducibility.
            k_neighbors: Number of neighbors for SMOTE/ADASYN.
            **kwargs: Additional parameters for resampling methods.
        """
        super().__init__("ImbalanceFixer")
        self.method = method.lower()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.kwargs = kwargs
    
    def fix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> FixResult:
        """Fix class imbalance in the dataset.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            **kwargs: Additional fixer-specific parameters.
        
        Returns:
            FixResult with transformed data and operation details.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        
        # Store original types
        X_is_df = isinstance(X, pd.DataFrame)
        y_is_series = isinstance(y, pd.Series)
        
        # Convert to numpy for processing
        X_array = X.values if X_is_df else np.asarray(X)
        y_array = y.values if y_is_series else np.asarray(y)
        
        # Get original class distribution
        original_dist = get_class_distribution(y)
        
        try:
            if self.method == "class_weights":
                # Calculate class weights (no resampling)
                X_transformed, y_transformed = X_array, y_array
                class_weights = self._compute_class_weights(y_array)
                details = {
                    "method": "class_weights",
                    "class_weights": class_weights,
                    "original_distribution": original_dist.to_dict(),
                }
            elif self.method in ["smote", "adasyn", "random_over", "random_under", "smoteenn", "smotetomek"]:
                X_transformed, y_transformed, details = self._resample(
                    X_array, y_array, original_dist
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Convert back to original types
            if X_is_df:
                X_transformed = pd.DataFrame(
                    X_transformed,
                    columns=X.columns if hasattr(X, 'columns') else None
                )
            
            if y_is_series:
                y_transformed = pd.Series(
                    y_transformed,
                    name=y.name if hasattr(y, 'name') else 'target'
                )
            
            # Calculate new distribution
            new_dist = get_class_distribution(y_transformed)
            
            details.update({
                "new_distribution": new_dist.to_dict(),
                "original_samples": len(y_array),
                "new_samples": len(y_transformed),
            })
            
            result = FixResult(
                fixer_name=self.name,
                success=True,
                X_transformed=X_transformed,
                y_transformed=y_transformed,
                details=details,
            )
            
        except Exception as e:
            result = FixResult(
                fixer_name=self.name,
                success=False,
                X_transformed=X,
                y_transformed=y,
                details={"error": str(e), "method": self.method},
                warnings=[f"Failed to apply {self.method}: {str(e)}"],
            )
        
        self._last_result = result
        return result
    
    def _resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        original_dist: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply resampling method.
        
        Args:
            X: Feature array.
            y: Target array.
            original_dist: Original class distribution.
        
        Returns:
            Tuple of (X_resampled, y_resampled, details).
        """
        # Try to import imbalanced-learn
        try:
            from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.combine import SMOTEENN, SMOTETomek
            
            # Map method names to classes
            resamplers = {
                "smote": SMOTE,
                "adasyn": ADASYN,
                "random_over": RandomOverSampler,
                "random_under": RandomUnderSampler,
                "smoteenn": SMOTEENN,
                "smotetomek": SMOTETomek,
            }
            
            ResamplerClass = resamplers.get(self.method)
            
            if ResamplerClass is None:
                raise ValueError(f"Unknown resampling method: {self.method}")
            
            # Initialize resampler
            if self.method in ["smote", "adasyn"]:
                resampler = ResamplerClass(
                    sampling_strategy=self.sampling_strategy,
                    random_state=self.random_state,
                    k_neighbors=self.k_neighbors,
                )
            elif self.method in ["smoteenn", "smotetomek"]:
                resampler = ResamplerClass(
                    sampling_strategy=self.sampling_strategy,
                    random_state=self.random_state,
                    smote=SMOTE(k_neighbors=self.k_neighbors),
                )
            else:
                resampler = ResamplerClass(
                    sampling_strategy=self.sampling_strategy,
                    random_state=self.random_state,
                )
            
            # Apply resampling
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            
            details = {
                "method": self.method,
                "resampler": ResamplerClass.__name__,
                "sampling_strategy": str(self.sampling_strategy),
                "original_distribution": original_dist.to_dict(),
            }
            
            return X_resampled, y_resampled, details
            
        except ImportError:
            # Fallback to simple random oversampling/undersampling
            return self._simple_resample(X, y, original_dist)
    
    def _simple_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        original_dist: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Simple resampling without imbalanced-learn.
        
        Args:
            X: Feature array.
            y: Target array.
            original_dist: Original class distribution.
        
        Returns:
            Tuple of (X_resampled, y_resampled, details).
        """
        rng = np.random.RandomState(self.random_state)
        
        classes = original_dist.index.tolist()
        max_count = original_dist.max()
        
        X_resampled_list = []
        y_resampled_list = []
        
        for cls in classes:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]
            
            if self.method in ["smote", "adasyn", "random_over"]:
                # Oversample to match majority class
                n_samples = int(max_count)
                indices = rng.choice(len(X_cls), size=n_samples, replace=True)
                X_resampled_list.append(X_cls[indices])
                y_resampled_list.append(y_cls[indices])
            elif self.method == "random_under":
                # Undersample to match minority class
                min_count = original_dist.min()
                if len(X_cls) > min_count:
                    indices = rng.choice(len(X_cls), size=int(min_count), replace=False)
                    X_resampled_list.append(X_cls[indices])
                    y_resampled_list.append(y_cls[indices])
                else:
                    X_resampled_list.append(X_cls)
                    y_resampled_list.append(y_cls)
            else:
                X_resampled_list.append(X_cls)
                y_resampled_list.append(y_cls)
        
        X_resampled = np.vstack(X_resampled_list)
        y_resampled = np.hstack(y_resampled_list)
        
        details = {
            "method": self.method,
            "resampler": "simple_numpy",
            "note": "imbalanced-learn not installed, using simple resampling",
            "original_distribution": original_dist.to_dict(),
        }
        
        return X_resampled, y_resampled, details
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[Any, float]:
        """Compute balanced class weights.
        
        Args:
            y: Target array.
        
        Returns:
            Dictionary mapping class labels to weights.
        """
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y
        )
        return {cls: float(w) for cls, w in zip(classes, weights)}
    
    def get_class_weights(self) -> Optional[Dict[Any, float]]:
        """Get computed class weights from last fix operation.
        
        Returns:
            Dictionary of class weights or None.
        """
        if self._last_result is None:
            return None
        
        return self._last_result.details.get("class_weights")
