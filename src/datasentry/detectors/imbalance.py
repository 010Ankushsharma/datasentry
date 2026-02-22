"""Class imbalance detection for DataSentry library.

This module provides detection of class imbalance issues in classification datasets.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy

from datasentry.core.base import BaseDetector, DetectionResult, SeverityLevel
from datasentry.core.utils import get_class_distribution, validate_data


class ImbalanceDetector(BaseDetector):
    """Detect class imbalance in classification datasets.
    
    Class imbalance occurs when the classes in a classification problem
    are not represented equally. This can lead to biased models that
    favor the majority class.
    
    Attributes:
        imbalance_threshold: Threshold for considering imbalance severe.
        min_samples_per_class: Minimum samples required per class.
    
    Example:
        >>> detector = ImbalanceDetector(imbalance_threshold=2.0)
        >>> X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        >>> y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8:2 ratio
        >>> result = detector.detect(X, y)
        >>> print(result.issue_detected)
        True
    """
    
    def __init__(
        self,
        imbalance_threshold: float = 3.0,
        min_samples_per_class: int = 5,
        severity_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize the imbalance detector.
        
        Args:
            imbalance_threshold: Ratio threshold for flagging imbalance.
                Default is 3.0 (majority class is 3x larger than minority).
            min_samples_per_class: Minimum number of samples required per class.
            severity_thresholds: Custom thresholds for severity levels.
                Format: {'low': 2.0, 'medium': 5.0, 'high': 10.0, 'critical': 20.0}
        """
        super().__init__("ImbalanceDetector")
        self.imbalance_threshold = imbalance_threshold
        self.min_samples_per_class = min_samples_per_class
        self.severity_thresholds = severity_thresholds or {
            "low": 2.0,
            "medium": 5.0,
            "high": 10.0,
            "critical": 20.0,
        }
    
    def detect(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> DetectionResult:
        """Detect class imbalance in the dataset.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
        
        Returns:
            DetectionResult with imbalance analysis.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        
        # Get class distribution
        class_dist = get_class_distribution(y)
        total_samples = len(y)
        n_classes = len(class_dist)
        
        # Calculate metrics
        max_count = class_dist.max()
        min_count = class_dist.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Calculate class proportions
        proportions = class_dist / total_samples
        max_proportion = proportions.max()
        min_proportion = proportions.min()
        
        # Calculate entropy (measure of balance)
        # Max entropy is log(n_classes), normalized entropy shows balance
        class_probs = class_dist / total_samples
        dataset_entropy = entropy(class_probs, base=2)
        max_entropy = np.log2(n_classes) if n_classes > 1 else 1
        normalized_entropy = dataset_entropy / max_entropy if max_entropy > 0 else 1.0
        
        # Check for rare classes
        rare_classes = class_dist[class_dist < self.min_samples_per_class].index.tolist()
        
        # Determine severity
        severity = self._determine_severity(imbalance_ratio, normalized_entropy)
        
        # Determine if issue exists
        issue_detected = (
            imbalance_ratio >= self.imbalance_threshold or
            len(rare_classes) > 0 or
            normalized_entropy < 0.5
        )
        
        # Calculate score (0 to 1, higher means more imbalanced)
        score = min(1.0, imbalance_ratio / self.severity_thresholds["critical"])
        
        # Prepare details
        details = {
            "n_classes": n_classes,
            "total_samples": total_samples,
            "imbalance_ratio": round(imbalance_ratio, 3),
            "majority_class_count": int(max_count),
            "minority_class_count": int(min_count),
            "majority_class_proportion": round(max_proportion, 3),
            "minority_class_proportion": round(min_proportion, 3),
            "normalized_entropy": round(normalized_entropy, 3),
            "class_distribution": class_dist.to_dict(),
            "class_proportions": {k: round(v, 3) for k, v in proportions.to_dict().items()},
        }
        
        if rare_classes:
            details["rare_classes"] = rare_classes
            details["rare_classes_count"] = len(rare_classes)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            imbalance_ratio, n_classes, rare_classes, normalized_entropy
        )
        
        result = DetectionResult(
            detector_name=self.name,
            issue_detected=issue_detected,
            severity=severity,
            score=score,
            details=details,
            recommendations=recommendations,
        )
        
        self._last_result = result
        return result
    
    def _determine_severity(
        self,
        imbalance_ratio: float,
        normalized_entropy: float
    ) -> SeverityLevel:
        """Determine severity level based on imbalance metrics.
        
        Args:
            imbalance_ratio: Ratio of majority to minority class.
            normalized_entropy: Normalized entropy measure.
        
        Returns:
            SeverityLevel enum value.
        """
        if imbalance_ratio >= self.severity_thresholds["critical"]:
            return SeverityLevel.CRITICAL
        elif imbalance_ratio >= self.severity_thresholds["high"]:
            return SeverityLevel.HIGH
        elif imbalance_ratio >= self.severity_thresholds["medium"]:
            return SeverityLevel.MEDIUM
        elif imbalance_ratio >= self.severity_thresholds["low"] or normalized_entropy < 0.7:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.NONE
    
    def _generate_recommendations(
        self,
        imbalance_ratio: float,
        n_classes: int,
        rare_classes: List,
        normalized_entropy: float
    ) -> List[str]:
        """Generate recommendations based on detected issues.
        
        Args:
            imbalance_ratio: Ratio of majority to minority class.
            n_classes: Number of classes.
            rare_classes: List of rare class labels.
            normalized_entropy: Normalized entropy measure.
        
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        if imbalance_ratio >= self.imbalance_threshold:
            recommendations.append(
                f"Consider using SMOTE or ADASYN to oversample minority classes "
                f"(current ratio: {imbalance_ratio:.1f}:1)"
            )
            recommendations.append(
                "Apply class weights in your model to penalize misclassification "
                "of minority classes"
            )
        
        if rare_classes:
            recommendations.append(
                f"Collect more samples for rare classes: {rare_classes}"
            )
        
        if normalized_entropy < 0.5:
            recommendations.append(
                "Consider combining similar minority classes if applicable"
            )
        
        if imbalance_ratio > 10:
            recommendations.append(
                "Use stratified sampling for train/validation/test splits"
            )
            recommendations.append(
                "Consider ensemble methods designed for imbalanced data "
                "(e.g., BalancedRandomForest)"
            )
        
        if not recommendations:
            recommendations.append("Class distribution appears balanced")
        
        return recommendations
    
    def get_imbalance_summary(self) -> Dict[str, Any]:
        """Get a summary of the last imbalance detection.
        
        Returns:
            Dictionary with imbalance summary or None if detect hasn't been called.
        """
        if self._last_result is None:
            return None
        
        return {
            "imbalance_ratio": self._last_result.details.get("imbalance_ratio"),
            "n_classes": self._last_result.details.get("n_classes"),
            "class_distribution": self._last_result.details.get("class_distribution"),
            "severity": self._last_result.severity.name,
        }
