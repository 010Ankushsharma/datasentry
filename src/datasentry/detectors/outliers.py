"""Outlier detection for DataSentry library.

This module provides detection of outliers in datasets using various methods.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from datasentry.core.base import BaseDetector, DetectionResult, SeverityLevel
from datasentry.core.utils import validate_data, convert_to_dataframe


class OutlierDetector(BaseDetector):
    """Detect outliers in datasets.
    
    Outliers are data points that differ significantly from other observations.
    They can negatively impact model performance and should be identified
    and handled appropriately.
    
    Attributes:
        method: Detection method ('iqr', 'zscore', 'isolation_forest', 'lof').
        threshold: Threshold for flagging outliers.
        contamination: Expected proportion of outliers (for ML methods).
    
    Example:
        >>> detector = OutlierDetector(method='iqr')
        >>> X = np.array([[1], [2], [3], [100], [2], [3]])  # 100 is outlier
        >>> result = detector.detect(X)
        >>> print(result.issue_detected)
        True
    """
    
    def __init__(
        self,
        method: str = "iqr",
        threshold: float = 3.0,
        contamination: float = 0.1,
        random_state: Optional[int] = 42
    ):
        """Initialize the outlier detector.
        
        Args:
            method: Detection method ('iqr', 'zscore', 'isolation_forest', 'lof').
            threshold: Threshold for statistical methods (IQR multiplier or Z-score).
            contamination: Expected proportion of outliers (for ML methods).
            random_state: Random seed for reproducibility.
        """
        super().__init__("OutlierDetector")
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.random_state = random_state
        
        self._outlier_indices: Optional[np.ndarray] = None
        self._outlier_scores: Optional[np.ndarray] = None
    
    def detect(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DetectionResult:
        """Detect outliers in the dataset.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,), optional.
        
        Returns:
            DetectionResult with outlier analysis.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        df = convert_to_dataframe(X)
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return DetectionResult(
                detector_name=self.name,
                issue_detected=False,
                severity=SeverityLevel.NONE,
                score=0.0,
                details={"error": "No numeric features found for outlier detection"},
                recommendations=["Outlier detection requires numeric features"],
            )
        
        # Detect outliers based on method
        if self.method == "iqr":
            outlier_indices, outlier_scores, details = self._detect_iqr(numeric_df)
        elif self.method == "zscore":
            outlier_indices, outlier_scores, details = self._detect_zscore(numeric_df)
        elif self.method == "isolation_forest":
            outlier_indices, outlier_scores, details = self._detect_isolation_forest(numeric_df)
        elif self.method == "lof":
            outlier_indices, outlier_scores, details = self._detect_lof(numeric_df)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Calculate metrics
        n_samples = len(df)
        n_outliers = len(outlier_indices)
        outlier_ratio = n_outliers / n_samples if n_samples > 0 else 0.0
        
        # Determine severity
        severity = self._determine_severity(outlier_ratio)
        
        # Determine if issue exists
        issue_detected = n_outliers > 0 and outlier_ratio >= 0.01
        
        # Calculate score
        score = min(1.0, outlier_ratio * 5)  # Scale: 20% outliers = score 1.0
        
        # Store for later access
        self._outlier_indices = outlier_indices
        self._outlier_scores = outlier_scores
        
        # Prepare details
        details.update({
            "n_samples": n_samples,
            "n_outliers": n_outliers,
            "outlier_ratio": round(outlier_ratio, 4),
            "method": self.method,
            "threshold": self.threshold if self.method in ['iqr', 'zscore'] else self.contamination,
        })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(outlier_ratio, n_outliers, details)
        
        result = DetectionResult(
            detector_name=self.name,
            issue_detected=issue_detected,
            severity=severity,
            score=score,
            details=details,
            recommendations=recommendations,
            metadata={
                "outlier_indices": outlier_indices.tolist(),
                "outlier_scores": outlier_scores.tolist() if outlier_scores is not None else [],
            }
        )
        
        self._last_result = result
        return result
    
    def _detect_iqr(
        self,
        df: pd.DataFrame
    ) -> tuple:
        """Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            df: Numeric DataFrame.
        
        Returns:
            Tuple of (outlier_indices, outlier_scores, details).
        """
        outlier_mask = pd.Series(False, index=df.index)
        feature_outliers = {}
        
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask = outlier_mask | col_outliers
            
            n_col_outliers = col_outliers.sum()
            if n_col_outliers > 0:
                feature_outliers[col] = {
                    "count": int(n_col_outliers),
                    "lower_bound": round(lower_bound, 3),
                    "upper_bound": round(upper_bound, 3),
                }
        
        outlier_indices = df.index[outlier_mask].values
        
        # Calculate outlier scores (normalized distance from bounds)
        outlier_scores = np.zeros(len(df))
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                median = df[col].median()
                distances = np.abs(df[col] - median) / IQR
                outlier_scores = np.maximum(outlier_scores, distances.fillna(0).values)
        
        details = {
            "feature_outliers": feature_outliers,
            "n_features_with_outliers": len(feature_outliers),
            "method_description": f"IQR method (threshold={self.threshold}x IQR)",
        }
        
        return outlier_indices, outlier_scores, details
    
    def _detect_zscore(
        self,
        df: pd.DataFrame
    ) -> tuple:
        """Detect outliers using Z-score method.
        
        Args:
            df: Numeric DataFrame.
        
        Returns:
            Tuple of (outlier_indices, outlier_scores, details).
        """
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df, nan_policy='omit'))
        
        # Handle NaN values
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        
        # Find outliers
        outlier_mask = (z_scores > self.threshold).any(axis=1)
        outlier_indices = df.index[outlier_mask].values
        
        # Max Z-score per sample as outlier score
        outlier_scores = z_scores.max(axis=1)
        
        # Per-feature statistics
        feature_outliers = {}
        for i, col in enumerate(df.columns):
            n_outliers = (z_scores[:, i] > self.threshold).sum()
            if n_outliers > 0:
                max_zscore = z_scores[:, i].max()
                feature_outliers[col] = {
                    "count": int(n_outliers),
                    "max_zscore": round(float(max_zscore), 3),
                }
        
        details = {
            "feature_outliers": feature_outliers,
            "n_features_with_outliers": len(feature_outliers),
            "method_description": f"Z-score method (threshold={self.threshold})",
            "mean_zscore": round(float(z_scores.mean()), 3),
            "max_zscore": round(float(z_scores.max()), 3),
        }
        
        return outlier_indices, outlier_scores, details
    
    def _detect_isolation_forest(
        self,
        df: pd.DataFrame
    ) -> tuple:
        """Detect outliers using Isolation Forest.
        
        Args:
            df: Numeric DataFrame.
        
        Returns:
            Tuple of (outlier_indices, outlier_scores, details).
        """
        # Handle missing values
        df_clean = df.fillna(df.median())
        
        # Fit Isolation Forest
        clf = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        predictions = clf.fit_predict(df_clean)
        outlier_scores = -clf.score_samples(df_clean)  # Higher = more anomalous
        
        # Find outliers (predictions == -1 are outliers)
        outlier_mask = predictions == -1
        outlier_indices = df.index[outlier_mask].values
        
        details = {
            "method_description": f"Isolation Forest (contamination={self.contamination})",
            "n_trees": 100,
            "avg_outlier_score": round(float(outlier_scores[outlier_mask].mean()), 3) if outlier_mask.any() else 0.0,
        }
        
        return outlier_indices, outlier_scores, details
    
    def _detect_lof(
        self,
        df: pd.DataFrame
    ) -> tuple:
        """Detect outliers using Local Outlier Factor.
        
        Args:
            df: Numeric DataFrame.
        
        Returns:
            Tuple of (outlier_indices, outlier_scores, details).
        """
        # Handle missing values
        df_clean = df.fillna(df.median())
        
        # Fit LOF
        clf = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            n_jobs=-1
        )
        
        predictions = clf.fit_predict(df_clean)
        outlier_scores = -clf.negative_outlier_factor_  # Higher = more anomalous
        
        # Find outliers
        outlier_mask = predictions == -1
        outlier_indices = df.index[outlier_mask].values
        
        details = {
            "method_description": f"Local Outlier Factor (contamination={self.contamination})",
            "n_neighbors": 20,
            "avg_outlier_score": round(float(outlier_scores[outlier_mask].mean()), 3) if outlier_mask.any() else 0.0,
        }
        
        return outlier_indices, outlier_scores, details
    
    def _determine_severity(self, outlier_ratio: float) -> SeverityLevel:
        """Determine severity based on outlier ratio.
        
        Args:
            outlier_ratio: Proportion of outliers.
        
        Returns:
            SeverityLevel enum value.
        """
        if outlier_ratio >= 0.20:
            return SeverityLevel.CRITICAL
        elif outlier_ratio >= 0.10:
            return SeverityLevel.HIGH
        elif outlier_ratio >= 0.05:
            return SeverityLevel.MEDIUM
        elif outlier_ratio >= 0.01:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.NONE
    
    def _generate_recommendations(
        self,
        outlier_ratio: float,
        n_outliers: int,
        details: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for outliers.
        
        Args:
            outlier_ratio: Proportion of outliers.
            n_outliers: Number of outliers.
            details: Detection details.
        
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        if outlier_ratio > 0.01:
            recommendations.append(
                f"Review {n_outliers} detected outliers ({outlier_ratio:.1%} of data)"
            )
        
        if outlier_ratio > 0.05:
            recommendations.append(
                "Consider using robust scalers (RobustScaler) instead of StandardScaler"
            )
            recommendations.append(
                "Apply outlier-robust algorithms (e.g., RANSAC, Huber regression)"
            )
        
        if outlier_ratio > 0.10:
            recommendations.append(
                "High outlier ratio detected - investigate data collection process"
            )
            recommendations.append(
                "Consider winsorization or capping extreme values"
            )
        
        if outlier_ratio > 0.20:
            recommendations.append(
                "CRITICAL: Very high outlier ratio - data quality may be compromised"
            )
        
        # Feature-specific recommendations
        feature_outliers = details.get("feature_outliers", {})
        if len(feature_outliers) > 0:
            worst_features = sorted(
                feature_outliers.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:3]
            worst_names = [f[0] for f in worst_features]
            recommendations.append(
                f"Features with most outliers: {worst_names}"
            )
        
        if not recommendations:
            recommendations.append("No significant outliers detected")
        
        return recommendations
    
    def get_outlier_indices(self) -> Optional[np.ndarray]:
        """Get indices of detected outliers.
        
        Returns:
            Array of outlier indices or None if detect hasn't been called.
        """
        return self._outlier_indices
    
    def get_outlier_scores(self) -> Optional[np.ndarray]:
        """Get outlier scores for all samples.
        
        Returns:
            Array of outlier scores or None if detect hasn't been called.
        """
        return self._outlier_scores
