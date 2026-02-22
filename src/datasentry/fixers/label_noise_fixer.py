"""Label noise fixer for DataSentry library.

This module provides methods to fix label noise issues.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from datasentry.core.base import BaseFixer, FixResult
from datasentry.core.utils import validate_data


class LabelNoiseFixer(BaseFixer):
    """Fix label noise using various cleaning strategies.
    
    This fixer provides methods to address label noise:
    - Remove: Remove samples with likely incorrect labels
    - Relabel: Change labels based on model confidence
    - Weight: Assign lower weights to uncertain samples
    
    Attributes:
        method: Cleaning method ('remove', 'relabel', 'weight').
        noise_threshold: Threshold for flagging noisy labels.
        n_folds: Number of cross-validation folds.
    
    Example:
        >>> fixer = LabelNoiseFixer(method='remove')
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 3, 100)
        >>> result = fixer.fix(X, y)
        >>> print(result.success)
        True
    """
    
    def __init__(
        self,
        method: str = "remove",
        noise_threshold: float = 0.3,
        n_folds: int = 5,
        min_samples_per_class: int = 10,
        random_state: Optional[int] = 42
    ):
        """Initialize the label noise fixer.
        
        Args:
            method: Cleaning method ('remove', 'relabel', 'weight').
            noise_threshold: Confidence threshold for flagging noisy labels.
            n_folds: Number of cross-validation folds.
            min_samples_per_class: Minimum samples required per class.
            random_state: Random seed for reproducibility.
        """
        super().__init__("LabelNoiseFixer")
        self.method = method.lower()
        self.noise_threshold = noise_threshold
        self.n_folds = n_folds
        self.min_samples_per_class = min_samples_per_class
        self.random_state = random_state
    
    def fix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> FixResult:
        """Fix label noise in the dataset.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            **kwargs: Additional fixer-specific parameters.
                - noisy_indices: Pre-computed indices of noisy samples.
        
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
        
        original_n_samples = len(y_array)
        
        try:
            # Get noisy indices (from kwargs or compute them)
            noisy_indices = kwargs.get('noisy_indices')
            if noisy_indices is None:
                noisy_indices, noise_scores, predicted_labels = self._identify_noisy_labels(
                    X_array, y_array
                )
            else:
                noise_scores = np.zeros(len(y_array))
                predicted_labels = y_array.copy()
            
            # Apply fix based on method
            if self.method == "remove":
                X_transformed, y_transformed, details = self._remove_noisy_samples(
                    X_array, y_array, noisy_indices
                )
            elif self.method == "relabel":
                X_transformed, y_transformed, details = self._relabel_samples(
                    X_array, y_array, noisy_indices, predicted_labels
                )
            elif self.method == "weight":
                X_transformed, y_transformed, details = self._weight_samples(
                    X_array, y_array, noise_scores
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
            
            details.update({
                "original_samples": original_n_samples,
                "final_samples": len(y_transformed) if hasattr(y_transformed, '__len__') else original_n_samples,
                "n_noisy_identified": len(noisy_indices),
                "noise_ratio": round(len(noisy_indices) / original_n_samples, 4),
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
    
    def _identify_noisy_labels(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Identify potentially noisy labels using confident learning.
        
        Args:
            X: Feature array.
            y: Target array.
        
        Returns:
            Tuple of (noisy_indices, noise_scores, predicted_labels).
        """
        # Get cross-validated predicted probabilities
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Get predicted probabilities
        y_probs = cross_val_predict(
            model, X, y, cv=cv, method='predict_proba'
        )
        
        # Get predicted labels
        unique_classes = np.unique(y)
        predicted_labels = unique_classes[np.argmax(y_probs, axis=1)]
        
        # Calculate confidence in given label vs predicted label
        given_label_idx = np.searchsorted(unique_classes, y)
        confidence_in_given = y_probs[np.arange(len(y)), given_label_idx]
        confidence_in_predicted = y_probs.max(axis=1)
        
        # Noise score
        noise_scores = confidence_in_predicted - confidence_in_given
        
        # Flag samples where model is confident about different label
        noisy_mask = (
            (predicted_labels != y) & 
            (confidence_in_predicted > 0.6) &
            (noise_scores > self.noise_threshold)
        )
        noisy_indices = np.where(noisy_mask)[0]
        
        return noisy_indices, noise_scores, predicted_labels
    
    def _remove_noisy_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noisy_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Remove samples with likely incorrect labels.
        
        Args:
            X: Feature array.
            y: Target array.
            noisy_indices: Indices of noisy samples.
        
        Returns:
            Tuple of (X_clean, y_clean, details).
        """
        # Create mask for clean samples
        clean_mask = np.ones(len(y), dtype=bool)
        clean_mask[noisy_indices] = False
        
        X_clean = X[clean_mask]
        y_clean = y[clean_mask]
        
        details = {
            "method": "remove",
            "n_removed": len(noisy_indices),
            "n_remaining": len(y_clean),
            "description": "Removed samples with likely incorrect labels",
        }
        
        return X_clean, y_clean, details
    
    def _relabel_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noisy_indices: np.ndarray,
        predicted_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Relabel samples with model-predicted labels.
        
        Args:
            X: Feature array.
            y: Target array.
            noisy_indices: Indices of noisy samples.
            predicted_labels: Predicted labels for all samples.
        
        Returns:
            Tuple of (X, y_relabeled, details).
        """
        y_relabeled = y.copy()
        
        # Only relabel the identified noisy samples
        y_relabeled[noisy_indices] = predicted_labels[noisy_indices]
        
        # Track changes
        relabel_map = {}
        for idx in noisy_indices:
            old_label = y[idx]
            new_label = predicted_labels[idx]
            key = f"{old_label} -> {new_label}"
            relabel_map[key] = relabel_map.get(key, 0) + 1
        
        details = {
            "method": "relabel",
            "n_relabeled": len(noisy_indices),
            "relabel_distribution": relabel_map,
            "description": "Changed labels based on model confidence",
        }
        
        return X, y_relabeled, details
    
    def _weight_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Assign sample weights based on label confidence.
        
        Args:
            X: Feature array.
            y: Target array.
            noise_scores: Noise scores for each sample.
        
        Returns:
            Tuple of (X, y_with_weights, details).
        """
        # Convert noise scores to weights (higher noise = lower weight)
        # Normalize noise scores to [0, 1]
        max_score = noise_scores.max() if noise_scores.max() > 0 else 1
        normalized_scores = noise_scores / max_score
        
        # Convert to weights: weight = 1 - 0.5 * normalized_noise
        # This gives weights in range [0.5, 1.0]
        sample_weights = 1.0 - 0.5 * normalized_scores
        
        # Store weights in details (can't attach to y directly)
        details = {
            "method": "weight",
            "sample_weights": sample_weights.tolist(),
            "weight_statistics": {
                "min": round(float(sample_weights.min()), 4),
                "max": round(float(sample_weights.max()), 4),
                "mean": round(float(sample_weights.mean()), 4),
            },
            "description": "Assigned lower weights to uncertain samples",
        }
        
        # Return original X, y (weights are in details)
        return X, y, details
    
    def get_clean_indices(self) -> Optional[np.ndarray]:
        """Get indices of clean (non-noisy) samples from last fix.
        
        Returns:
            Array of clean indices or None.
        """
        if self._last_result is None:
            return None
        
        # This would need to be stored during fix operation
        return self._last_result.metadata.get("clean_indices")
