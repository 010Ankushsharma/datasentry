"""Missing value detection for DataSentry library.

This module provides detection of missing values and patterns in datasets.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from datasentry.core.base import BaseDetector, DetectionResult, SeverityLevel
from datasentry.core.utils import validate_data, convert_to_dataframe


class MissingValueDetector(BaseDetector):
    """Detect missing values and patterns in datasets.
    
    Missing values can significantly impact model performance. This detector
    identifies missing values, analyzes their patterns, and assesses their
    impact on the dataset.
    
    Attributes:
        missing_threshold: Threshold for flagging high missing ratio.
        check_patterns: Whether to analyze missing value patterns.
        check_completeness: Whether to check row completeness.
    
    Example:
        >>> detector = MissingValueDetector()
        >>> X = pd.DataFrame({'a': [1, 2, None], 'b': [None, 2, 3]})
        >>> result = detector.detect(X)
        >>> print(result.issue_detected)
        True
    """
    
    def __init__(
        self,
        missing_threshold: float = 0.05,
        row_completeness_threshold: float = 0.9,
        check_patterns: bool = True,
        check_completeness: bool = True,
        pattern_threshold: float = 0.1
    ):
        """Initialize the missing value detector.
        
        Args:
            missing_threshold: Threshold for flagging features with too many missing values.
            row_completeness_threshold: Minimum ratio of non-missing values per row.
            check_patterns: Whether to analyze missing value patterns.
            check_completeness: Whether to check row completeness.
            pattern_threshold: Threshold for identifying common missing patterns.
        """
        super().__init__("MissingValueDetector")
        self.missing_threshold = missing_threshold
        self.row_completeness_threshold = row_completeness_threshold
        self.check_patterns = check_patterns
        self.check_completeness = check_completeness
        self.pattern_threshold = pattern_threshold
    
    def detect(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DetectionResult:
        """Detect missing values in the dataset.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,), optional.
        
        Returns:
            DetectionResult with missing value analysis.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        
        # Convert to DataFrame for easier analysis
        df = convert_to_dataframe(X)
        
        # Calculate basic missing statistics
        missing_stats = self._calculate_missing_stats(df)
        
        # Check for high missing features
        high_missing_features = self._get_high_missing_features(missing_stats)
        
        # Check row completeness
        completeness_stats = {}
        if self.check_completeness:
            completeness_stats = self._check_row_completeness(df)
        
        # Analyze missing patterns
        pattern_stats = {}
        if self.check_patterns:
            pattern_stats = self._analyze_missing_patterns(df)
        
        # Determine if issue exists
        has_missing = missing_stats["total_missing"] > 0
        has_high_missing = len(high_missing_features) > 0
        has_low_completeness = completeness_stats.get("low_completeness_rows", 0) > 0
        
        issue_detected = has_missing and (has_high_missing or has_low_completeness)
        
        # Calculate severity
        severity = self._determine_severity(
            missing_stats, high_missing_features, completeness_stats
        )
        
        # Calculate score
        score = self._calculate_missing_score(missing_stats, completeness_stats)
        
        # Prepare details
        details = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "total_cells": len(df) * len(df.columns),
            "total_missing": missing_stats["total_missing"],
            "overall_missing_ratio": round(missing_stats["overall_ratio"], 4),
            "features_with_missing": missing_stats["features_with_missing"],
            "high_missing_features": high_missing_features,
            "missing_by_feature": missing_stats["by_feature"],
        }
        
        if completeness_stats:
            details["row_completeness"] = completeness_stats
        
        if pattern_stats:
            details["missing_patterns"] = pattern_stats
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_stats, high_missing_features, completeness_stats
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
    
    def _calculate_missing_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate missing value statistics.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Dictionary with missing statistics.
        """
        # Count missing per feature
        missing_counts = df.isnull().sum()
        missing_ratios = missing_counts / len(df)
        
        # Total missing
        total_missing = missing_counts.sum()
        total_cells = len(df) * len(df.columns)
        overall_ratio = total_missing / total_cells if total_cells > 0 else 0.0
        
        # Features with any missing
        features_with_missing = (missing_counts > 0).sum()
        
        # Per-feature statistics
        by_feature = {}
        for col in df.columns:
            count = int(missing_counts[col])
            ratio = round(float(missing_ratios[col]), 4)
            by_feature[col] = {
                "count": count,
                "ratio": ratio,
            }
        
        return {
            "total_missing": int(total_missing),
            "total_cells": total_cells,
            "overall_ratio": overall_ratio,
            "features_with_missing": int(features_with_missing),
            "by_feature": by_feature,
        }
    
    def _get_high_missing_features(
        self,
        missing_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get features with high missing ratios.
        
        Args:
            missing_stats: Missing value statistics.
        
        Returns:
            List of features exceeding threshold.
        """
        high_missing = []
        for feature, stats in missing_stats["by_feature"].items():
            if stats["ratio"] >= self.missing_threshold:
                high_missing.append({
                    "feature": feature,
                    "missing_count": stats["count"],
                    "missing_ratio": stats["ratio"],
                })
        
        # Sort by missing ratio descending
        high_missing.sort(key=lambda x: x["missing_ratio"], reverse=True)
        return high_missing
    
    def _check_row_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check completeness of each row.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Dictionary with completeness statistics.
        """
        # Calculate completeness ratio per row
        completeness = 1 - df.isnull().sum(axis=1) / len(df.columns)
        
        # Rows with low completeness
        low_completeness_mask = completeness < self.row_completeness_threshold
        low_completeness_rows = low_completeness_mask.sum()
        
        # Completeness distribution
        completeness_bins = pd.cut(completeness, bins=[0, 0.5, 0.75, 0.9, 1.0])
        completeness_dist = completeness_bins.value_counts().to_dict()
        completeness_dist = {str(k): int(v) for k, v in completeness_dist.items()}
        
        return {
            "mean_completeness": round(float(completeness.mean()), 4),
            "min_completeness": round(float(completeness.min()), 4),
            "low_completeness_rows": int(low_completeness_rows),
            "low_completeness_ratio": round(low_completeness_rows / len(df), 4),
            "completeness_distribution": completeness_dist,
        }
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing values.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Dictionary with pattern analysis.
        """
        # Create missing indicator matrix
        missing_matrix = df.isnull()
        
        if not missing_matrix.any().any():
            return {"has_patterns": False, "reason": "No missing values"}
        
        # Find common patterns (combinations of missing features)
        pattern_counts = missing_matrix.value_counts()
        
        # Filter to patterns that appear multiple times
        common_patterns = pattern_counts[pattern_counts >= len(df) * self.pattern_threshold]
        
        # Convert patterns to readable format
        pattern_list = []
        for pattern, count in common_patterns.head(10).items():
            if isinstance(pattern, tuple):
                missing_features = [col for col, is_missing in zip(df.columns, pattern) if is_missing]
            else:
                missing_features = [df.columns[i] for i, is_missing in enumerate(pattern) if is_missing]
            
            if missing_features:  # Only include patterns with actual missing values
                pattern_list.append({
                    "missing_features": missing_features,
                    "count": int(count),
                    "ratio": round(count / len(df), 4),
                })
        
        # Check if missingness is correlated with any feature
        # This is a simplified check - full MCAR/MAR/MNAR analysis would require more
        
        return {
            "has_patterns": len(pattern_list) > 0,
            "n_unique_patterns": len(pattern_counts),
            "common_patterns": pattern_list,
        }
    
    def _determine_severity(
        self,
        missing_stats: Dict[str, Any],
        high_missing_features: List[Dict],
        completeness_stats: Dict[str, Any]
    ) -> SeverityLevel:
        """Determine severity based on missing value analysis.
        
        Args:
            missing_stats: Missing value statistics.
            high_missing_features: List of features with high missing ratios.
            completeness_stats: Row completeness statistics.
        
        Returns:
            SeverityLevel enum value.
        """
        overall_ratio = missing_stats["overall_ratio"]
        
        # Check for completely missing features
        completely_missing = sum(
            1 for f in high_missing_features if f["missing_ratio"] >= 0.99
        )
        if completely_missing > 0:
            return SeverityLevel.CRITICAL
        
        # Check overall missing ratio
        if overall_ratio >= 0.5:
            return SeverityLevel.CRITICAL
        elif overall_ratio >= 0.3:
            return SeverityLevel.HIGH
        elif overall_ratio >= 0.1:
            return SeverityLevel.MEDIUM
        elif overall_ratio >= 0.05:
            return SeverityLevel.LOW
        
        # Check row completeness
        if completeness_stats:
            low_comp_ratio = completeness_stats.get("low_completeness_ratio", 0)
            if low_comp_ratio >= 0.5:
                return SeverityLevel.HIGH
            elif low_comp_ratio >= 0.2:
                return SeverityLevel.MEDIUM
            elif low_comp_ratio >= 0.05:
                return SeverityLevel.LOW
        
        return SeverityLevel.NONE
    
    def _calculate_missing_score(
        self,
        missing_stats: Dict[str, Any],
        completeness_stats: Dict[str, Any]
    ) -> float:
        """Calculate overall missing value score.
        
        Args:
            missing_stats: Missing value statistics.
            completeness_stats: Row completeness statistics.
        
        Returns:
            Score between 0 and 1.
        """
        scores = []
        
        # Overall missing ratio
        overall_ratio = missing_stats["overall_ratio"]
        scores.append(min(1.0, overall_ratio * 2))
        
        # Row completeness
        if completeness_stats:
            low_comp_ratio = completeness_stats.get("low_completeness_ratio", 0)
            scores.append(min(1.0, low_comp_ratio * 2))
        
        return max(scores)
    
    def _generate_recommendations(
        self,
        missing_stats: Dict[str, Any],
        high_missing_features: List[Dict],
        completeness_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for missing values.
        
        Args:
            missing_stats: Missing value statistics.
            high_missing_features: List of features with high missing ratios.
            completeness_stats: Row completeness statistics.
        
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        overall_ratio = missing_stats["overall_ratio"]
        
        if overall_ratio == 0:
            recommendations.append("No missing values detected - dataset is complete")
            return recommendations
        
        # Recommendations for high missing features
        if high_missing_features:
            n_high = len(high_missing_features)
            recommendations.append(
                f"Consider removing {n_high} features with >{self.missing_threshold:.0%} missing values"
            )
            
            # List worst offenders
            worst = high_missing_features[:3]
            worst_names = [f["feature"] for f in worst]
            recommendations.append(
                f"Highest missing: {worst_names}"
            )
        
        # Imputation recommendations
        if overall_ratio < 0.3:
            recommendations.append(
                "Use imputation strategies (mean/median for numeric, mode for categorical)"
            )
        else:
            recommendations.append(
                "Consider collecting more complete data or using advanced imputation (KNN, Iterative)"
            )
        
        # Row completeness recommendations
        if completeness_stats:
            low_comp = completeness_stats.get("low_completeness_rows", 0)
            if low_comp > 0:
                recommendations.append(
                    f"Review {low_comp} rows with low completeness - consider removing "
                    "if they have insufficient information"
                )
        
        # Pattern recommendations
        if overall_ratio > 0.1:
            recommendations.append(
                "Analyze missing value patterns to determine if data is MCAR, MAR, or MNAR"
            )
        
        return recommendations
    
    def get_missing_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of missing values from last detection.
        
        Returns:
            Dictionary with missing summary or None if detect hasn't been called.
        """
        if self._last_result is None:
            return None
        
        return {
            "overall_missing_ratio": self._last_result.details.get("overall_missing_ratio"),
            "features_with_missing": self._last_result.details.get("features_with_missing"),
            "high_missing_features": self._last_result.details.get("high_missing_features"),
            "severity": self._last_result.severity.name,
        }
