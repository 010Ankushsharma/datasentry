"""Label noise detection for DataSentry library.

This module provides detection of label noise (incorrect labels) in classification datasets.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from datasentry.core.base import BaseDetector, DetectionResult, SeverityLevel
from datasentry.core.utils import validate_data


class LabelNoiseDetector(BaseDetector):
    """Detect label noise (incorrect labels) in classification datasets.
    
    Label noise occurs when some training examples have incorrect class labels.
    This can significantly degrade model performance. This detector uses
    confident learning techniques to identify potentially mislabeled examples.
    
    Attributes:
        n_folds: Number of cross-validation folds.
        noise_threshold: Threshold for flagging a sample as noisy.
        method: Detection method ('confident_learning' or 'knn_consensus').
    
    Example:
        >>> detector = LabelNoiseDetector()
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 3, 100)
        >>> result = detector.detect(X, y)
        >>> print(result.issue_detected)
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        noise_threshold: float = 0.3,
        method: str = "confident_learning",
        min_samples_per_class: int = 10,
        random_state: Optional[int] = 42
    ):
        """Initialize the label noise detector.
        
        Args:
            n_folds: Number of cross-validation folds for prediction.
            noise_threshold: Probability threshold for flagging noisy labels.
            method: Detection method ('confident_learning' or 'knn_consensus').
            min_samples_per_class: Minimum samples required per class.
            random_state: Random seed for reproducibility.
        """
        super().__init__("LabelNoiseDetector")
        self.n_folds = n_folds
        self.noise_threshold = noise_threshold
        self.method = method
        self.min_samples_per_class = min_samples_per_class
        self.random_state = random_state
        
        self._noisy_indices: Optional[np.ndarray] = None
        self._noise_scores: Optional[np.ndarray] = None
    
    def detect(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> DetectionResult:
        """Detect label noise in the dataset.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
        
        Returns:
            DetectionResult with label noise analysis.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)
        
        # Check for sufficient samples per class
        unique, counts = np.unique(y_array, return_counts=True)
        if np.any(counts < self.min_samples_per_class):
            return DetectionResult(
                detector_name=self.name,
                issue_detected=False,
                severity=SeverityLevel.NONE,
                score=0.0,
                details={
                    "error": "Insufficient samples per class for noise detection",
                    "class_counts": dict(zip(unique.tolist(), counts.tolist())),
                },
                recommendations=["Collect more samples per class before noise detection"],
            )
        
        # Detect noise based on method
        if self.method == "confident_learning":
            noisy_indices, noise_scores, details = self._confident_learning(X, y_array)
        elif self.method == "knn_consensus":
            noisy_indices, noise_scores, details = self._knn_consensus(X, y_array)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Calculate metrics
        n_samples = len(y_array)
        n_noisy = len(noisy_indices)
        noise_ratio = n_noisy / n_samples if n_samples > 0 else 0.0
        
        # Determine severity
        severity = self._determine_severity(noise_ratio)
        
        # Determine if issue exists
        issue_detected = n_noisy > 0 and noise_ratio >= 0.01
        
        # Calculate score
        score = min(1.0, noise_ratio * 5)  # Scale: 20% noise = score 1.0
        
        # Store for later access
        self._noisy_indices = noisy_indices
        self._noise_scores = noise_scores
        
        # Prepare details
        details.update({
            "n_samples": n_samples,
            "n_noisy_labels": n_noisy,
            "noise_ratio": round(noise_ratio, 4),
            "method": self.method,
            "n_folds": self.n_folds,
        })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            noise_ratio, n_noisy, details.get("avg_confidence", 0.5)
        )
        
        result = DetectionResult(
            detector_name=self.name,
            issue_detected=issue_detected,
            severity=severity,
            score=score,
            details=details,
            recommendations=recommendations,
            metadata={
                "noisy_indices": noisy_indices.tolist(),
                "noise_scores": noise_scores.tolist() if noise_scores is not None else [],
            }
        )
        
        self._last_result = result
        return result
    
    def _confident_learning(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Use confident learning to detect label noise.
        
        This method uses cross-validated predicted probabilities to identify
        samples where the model is confident about a different label.
        
        Args:
            X: Feature matrix.
            y: Target array.
        
        Returns:
            Tuple of (noisy_indices, noise_scores, details).
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
        
        # Identify potentially noisy labels
        predicted_labels = np.argmax(y_probs, axis=1)
        unique_classes = np.unique(y)
        predicted_labels = unique_classes[predicted_labels]
        
        # Calculate confidence in the given label vs predicted label
        given_label_idx = np.searchsorted(unique_classes, y)
        confidence_in_given = y_probs[np.arange(len(y)), given_label_idx]
        confidence_in_predicted = y_probs.max(axis=1)
        
        # Noise score: how much more confident is model in different label
        noise_scores = confidence_in_predicted - confidence_in_given
        
        # Flag samples where model is confident about different label
        noisy_mask = (
            (predicted_labels != y) & 
            (confidence_in_predicted > 0.6) &
            (noise_scores > self.noise_threshold)
        )
        noisy_indices = np.where(noisy_mask)[0]
        
        details = {
            "avg_confidence": round(float(np.mean(confidence_in_given)), 3),
            "avg_predicted_confidence": round(float(np.mean(confidence_in_predicted)), 3),
            "method_description": "Cross-validated confident learning with RandomForest",
        }
        
        return noisy_indices, noise_scores, details
    
    def _knn_consensus(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Use KNN consensus to detect label noise.
        
        This method identifies samples whose label disagrees with their
        k-nearest neighbors.
        
        Args:
            X: Feature matrix.
            y: Target array.
        
        Returns:
            Tuple of (noisy_indices, noise_scores, details).
        """
        # Use KNN to find neighbors
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X, y)
        
        # Get neighbor indices
        distances, indices = knn.kneighbors(X)
        
        # Calculate consensus among neighbors
        noise_scores = np.zeros(len(y))
        disagreement_count = np.zeros(len(y))
        
        for i in range(len(y)):
            neighbor_labels = y[indices[i][1:]]  # Exclude self
            label_counts = np.bincount(neighbor_labels, minlength=len(np.unique(y)))
            
            # Calculate disagreement score
            if len(neighbor_labels) > 0:
                agreement_ratio = np.sum(neighbor_labels == y[i]) / len(neighbor_labels)
                noise_scores[i] = 1.0 - agreement_ratio
                disagreement_count[i] = len(neighbor_labels) - np.sum(neighbor_labels == y[i])
        
        # Flag samples with low consensus
        noisy_mask = noise_scores > self.noise_threshold
        noisy_indices = np.where(noisy_mask)[0]
        
        details = {
            "avg_neighbor_agreement": round(float(1 - np.mean(noise_scores)), 3),
            "method_description": "KNN consensus (k=10)",
        }
        
        return noisy_indices, noise_scores, details
    
    def _determine_severity(self, noise_ratio: float) -> SeverityLevel:
        """Determine severity based on noise ratio.
        
        Args:
            noise_ratio: Proportion of noisy labels.
        
        Returns:
            SeverityLevel enum value.
        """
        if noise_ratio >= 0.20:
            return SeverityLevel.CRITICAL
        elif noise_ratio >= 0.10:
            return SeverityLevel.HIGH
        elif noise_ratio >= 0.05:
            return SeverityLevel.MEDIUM
        elif noise_ratio >= 0.01:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.NONE
    
    def _generate_recommendations(
        self,
        noise_ratio: float,
        n_noisy: int,
        avg_confidence: float
    ) -> List[str]:
        """Generate recommendations for label noise.
        
        Args:
            noise_ratio: Proportion of noisy labels.
            n_noisy: Number of noisy labels.
            avg_confidence: Average confidence in given labels.
        
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        if noise_ratio > 0.01:
            recommendations.append(
                f"Review {n_noisy} potentially mislabeled samples "
                f"({noise_ratio:.1%} of dataset)"
            )
        
        if noise_ratio > 0.05:
            recommendations.append(
                "Consider using robust loss functions (e.g., label smoothing, "
                "focal loss) that are less sensitive to label noise"
            )
            recommendations.append(
                "Implement a label cleaning pipeline before model training"
            )
        
        if noise_ratio > 0.10:
            recommendations.append(
                "Consider co-teaching or other noise-robust training methods"
            )
            recommendations.append(
                "Investigate data collection process for systematic labeling errors"
            )
        
        if avg_confidence < 0.5:
            recommendations.append(
                "Model confidence in labels is low - consider re-labeling dataset"
            )
        
        if not recommendations:
            recommendations.append("Labels appear clean")
        
        return recommendations
    
    def get_noisy_samples(self) -> Optional[np.ndarray]:
        """Get indices of samples flagged as potentially noisy.
        
        Returns:
            Array of indices or None if detect hasn't been called.
        """
        return self._noisy_indices
    
    def get_noise_scores(self) -> Optional[np.ndarray]:
        """Get noise scores for all samples.
        
        Returns:
            Array of noise scores or None if detect hasn't been called.
        """
        return self._noise_scores
