"""Feature redundancy detection for DataSentry library.

This module provides detection of redundant features in datasets.
"""

from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

from datasentry.core.base import BaseDetector, DetectionResult, SeverityLevel
from datasentry.core.utils import validate_data, convert_to_dataframe


class RedundancyDetector(BaseDetector):
    """Detect redundant features in datasets.
    
    Redundant features are those that provide similar information to other
    features. They can increase model complexity, cause multicollinearity,
    and reduce model interpretability.
    
    Attributes:
        correlation_threshold: Threshold for flagging correlated features.
        mutual_info_threshold: Threshold for flagging features with high MI.
        check_duplicates: Whether to check for duplicate columns.
    
    Example:
        >>> detector = RedundancyDetector(correlation_threshold=0.95)
        >>> X = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [4, 5, 6]})
        >>> result = detector.detect(X)
        >>> print(result.issue_detected)
        True
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.95,
        mutual_info_threshold: float = 0.9,
        check_duplicates: bool = True,
        check_correlation: bool = True,
        check_mutual_info: bool = False,
        method: str = "pearson"
    ):
        """Initialize the redundancy detector.
        
        Args:
            correlation_threshold: Correlation threshold for flagging redundancy.
            mutual_info_threshold: Mutual information threshold for redundancy.
            check_duplicates: Whether to check for duplicate columns.
            check_correlation: Whether to check for correlated features.
            check_mutual_info: Whether to check mutual information.
            method: Correlation method ('pearson', 'spearman').
        """
        super().__init__("RedundancyDetector")
        self.correlation_threshold = correlation_threshold
        self.mutual_info_threshold = mutual_info_threshold
        self.check_duplicates = check_duplicates
        self.check_correlation = check_correlation
        self.check_mutual_info = check_mutual_info
        self.method = method
    
    def detect(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DetectionResult:
        """Detect redundant features in the dataset.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,), optional.
        
        Returns:
            DetectionResult with redundancy analysis.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        df = convert_to_dataframe(X)
        
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Initialize results
        all_issues = []
        details = {"checks_performed": []}
        
        # Check 1: Duplicate columns
        if self.check_duplicates:
            duplicates = self._check_duplicate_columns(df)
            details["duplicate_columns"] = duplicates
            details["checks_performed"].append("duplicate_columns")
            if duplicates["has_duplicates"]:
                all_issues.append("duplicate_columns")
        
        # Check 2: Correlated features
        if self.check_correlation and not numeric_df.empty:
            correlations = self._check_correlations(numeric_df)
            details["correlated_features"] = correlations
            details["checks_performed"].append("correlated_features")
            if correlations["has_correlations"]:
                all_issues.append("correlated_features")
        
        # Check 3: Mutual information (if enabled)
        if self.check_mutual_info and not numeric_df.empty:
            mi_results = self._check_mutual_information(numeric_df)
            details["mutual_information"] = mi_results
            details["checks_performed"].append("mutual_information")
            if mi_results["has_high_mi"]:
                all_issues.append("mutual_information")
        
        # Calculate metrics
        n_issues = len(all_issues)
        has_redundancy = n_issues > 0
        
        # Calculate severity
        severity = self._determine_severity(details)
        
        # Calculate score
        score = self._calculate_redundancy_score(details)
        
        # Prepare summary
        details["n_issues_found"] = n_issues
        details["issues_found"] = all_issues
        details["n_features"] = len(df.columns)
        details["n_numeric_features"] = len(numeric_df.columns)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(details)
        
        result = DetectionResult(
            detector_name=self.name,
            issue_detected=has_redundancy,
            severity=severity,
            score=score,
            details=details,
            recommendations=recommendations,
        )
        
        self._last_result = result
        return result
    
    def _check_duplicate_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate columns.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Dictionary with duplicate analysis.
        """
        # Find duplicate columns
        duplicate_groups = []
        checked = set()
        
        for i, col1 in enumerate(df.columns):
            if col1 in checked:
                continue
            
            group = [col1]
            for col2 in df.columns[i + 1:]:
                if col2 not in checked and df[col1].equals(df[col2]):
                    group.append(col2)
                    checked.add(col2)
            
            if len(group) > 1:
                duplicate_groups.append(group)
                checked.add(col1)
        
        return {
            "has_duplicates": len(duplicate_groups) > 0,
            "n_duplicate_groups": len(duplicate_groups),
            "duplicate_groups": duplicate_groups,
            "total_duplicate_columns": sum(len(g) - 1 for g in duplicate_groups),
        }
    
    def _check_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for highly correlated features.
        
        Args:
            df: Numeric DataFrame.
        
        Returns:
            Dictionary with correlation analysis.
        """
        # Calculate correlation matrix
        if self.method == "pearson":
            corr_matrix = df.corr(method='pearson').abs()
        else:
            corr_matrix = df.corr(method='spearman').abs()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val) and corr_val >= self.correlation_threshold:
                    high_corr_pairs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": round(float(corr_val), 4),
                    })
        
        # Sort by correlation value
        high_corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)
        
        # Identify features to potentially remove
        features_to_remove = self._identify_features_to_remove(high_corr_pairs)
        
        return {
            "has_correlations": len(high_corr_pairs) > 0,
            "n_high_corr_pairs": len(high_corr_pairs),
            "high_correlation_pairs": high_corr_pairs[:20],  # Limit for memory
            "correlation_threshold": self.correlation_threshold,
            "features_to_consider_removing": features_to_remove,
            "method": self.method,
        }
    
    def _identify_features_to_remove(
        self,
        high_corr_pairs: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify features that could be removed to reduce redundancy.
        
        Uses a greedy approach to minimize the number of features to remove
        while eliminating all high correlations.
        
        Args:
            high_corr_pairs: List of highly correlated feature pairs.
        
        Returns:
            List of features to consider removing.
        """
        if not high_corr_pairs:
            return []
        
        # Build graph of correlated features
        from collections import defaultdict
        graph = defaultdict(set)
        
        for pair in high_corr_pairs:
            f1, f2 = pair["feature_1"], pair["feature_2"]
            graph[f1].add(f2)
            graph[f2].add(f1)
        
        # Greedy vertex cover approximation
        to_remove = set()
        edges_remaining = set((p["feature_1"], p["feature_2"]) for p in high_corr_pairs)
        
        while edges_remaining:
            # Find feature with most connections
            max_feature = max(
                graph.keys() - to_remove,
                key=lambda f: len(graph[f] - to_remove),
                default=None
            )
            
            if max_feature is None:
                break
            
            to_remove.add(max_feature)
            
            # Remove edges connected to this feature
            edges_remaining = {
                (f1, f2) for f1, f2 in edges_remaining
                if f1 != max_feature and f2 != max_feature
            }
        
        return sorted(list(to_remove))
    
    def _check_mutual_information(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for features with high mutual information.
        
        Args:
            df: Numeric DataFrame.
        
        Returns:
            Dictionary with MI analysis.
        """
        # Calculate pairwise mutual information
        n_features = len(df.columns)
        mi_matrix = np.zeros((n_features, n_features))
        
        # Fill MI matrix
        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    mi = mutual_info_regression(
                        df.iloc[:, i:i+1].fillna(df.iloc[:, i].median()),
                        df.iloc[:, j].fillna(df.iloc[:, j].median()),
                        random_state=42
                    )[0]
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi
                except Exception:
                    mi_matrix[i, j] = 0
                    mi_matrix[j, i] = 0
        
        # Normalize MI scores
        max_mi = mi_matrix.max()
        if max_mi > 0:
            mi_matrix = mi_matrix / max_mi
        
        # Find high MI pairs
        high_mi_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi_val = mi_matrix[i, j]
                if mi_val >= self.mutual_info_threshold:
                    high_mi_pairs.append({
                        "feature_1": df.columns[i],
                        "feature_2": df.columns[j],
                        "mutual_info": round(float(mi_val), 4),
                    })
        
        return {
            "has_high_mi": len(high_mi_pairs) > 0,
            "n_high_mi_pairs": len(high_mi_pairs),
            "high_mi_pairs": high_mi_pairs[:20],
            "mi_threshold": self.mutual_info_threshold,
        }
    
    def _determine_severity(self, details: Dict[str, Any]) -> SeverityLevel:
        """Determine severity based on redundancy analysis.
        
        Args:
            details: Dictionary with redundancy analysis.
        
        Returns:
            SeverityLevel enum value.
        """
        n_issues = details.get("n_issues_found", 0)
        
        # Check duplicates
        dup = details.get("duplicate_columns", {})
        if dup.get("has_duplicates"):
            n_dups = dup.get("n_duplicate_groups", 0)
            if n_dups >= 5:
                return SeverityLevel.HIGH
            elif n_dups >= 2:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.LOW
        
        # Check correlations
        corr = details.get("correlated_features", {})
        if corr.get("has_correlations"):
            n_pairs = corr.get("n_high_corr_pairs", 0)
            if n_pairs >= 20:
                return SeverityLevel.HIGH
            elif n_pairs >= 10:
                return SeverityLevel.MEDIUM
            elif n_pairs >= 5:
                return SeverityLevel.LOW
        
        if n_issues == 0:
            return SeverityLevel.NONE
        
        return SeverityLevel.LOW
    
    def _calculate_redundancy_score(self, details: Dict[str, Any]) -> float:
        """Calculate overall redundancy score.
        
        Args:
            details: Dictionary with redundancy analysis.
        
        Returns:
            Score between 0 and 1.
        """
        scores = []
        
        # Duplicate score
        dup = details.get("duplicate_columns", {})
        if dup.get("has_duplicates"):
            n_dups = dup.get("total_duplicate_columns", 0)
            n_features = details.get("n_features", 1)
            scores.append(min(1.0, n_dups / n_features))
        
        # Correlation score
        corr = details.get("correlated_features", {})
        if corr.get("has_correlations"):
            n_pairs = corr.get("n_high_corr_pairs", 0)
            scores.append(min(1.0, n_pairs / 20))
        
        return max(scores) if scores else 0.0
    
    def _generate_recommendations(self, details: Dict[str, Any]) -> List[str]:
        """Generate recommendations for redundancy.
        
        Args:
            details: Dictionary with redundancy analysis.
        
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        # Duplicate recommendations
        dup = details.get("duplicate_columns", {})
        if dup.get("has_duplicates"):
            n_dups = dup.get("total_duplicate_columns", 0)
            recommendations.append(
                f"Remove {n_dups} duplicate columns"
            )
            for group in dup.get("duplicate_groups", [])[:3]:
                recommendations.append(
                    f"  Keep '{group[0]}', remove: {group[1:]}"
                )
        
        # Correlation recommendations
        corr = details.get("correlated_features", {})
        if corr.get("has_correlations"):
            n_pairs = corr.get("n_high_corr_pairs", 0)
            to_remove = corr.get("features_to_consider_removing", [])
            
            recommendations.append(
                f"Found {n_pairs} highly correlated feature pairs "
                f"(threshold: {self.correlation_threshold})"
            )
            
            if to_remove:
                recommendations.append(
                    f"Consider removing features: {to_remove}"
                )
            
            recommendations.append(
                "Use VIF (Variance Inflation Factor) to identify multicollinear features"
            )
            recommendations.append(
                "Consider using PCA or feature selection techniques"
            )
        
        # MI recommendations
        mi = details.get("mutual_information", {})
        if mi.get("has_high_mi"):
            recommendations.append(
                "Features with high mutual information detected - "
                "consider dimensionality reduction"
            )
        
        if not recommendations:
            recommendations.append("No significant redundancy detected")
        
        return recommendations
    
    def get_redundant_features(self) -> List[str]:
        """Get list of features identified as redundant.
        
        Returns:
            List of redundant feature names or empty list.
        """
        if self._last_result is None:
            return []
        
        redundant = []
        
        # From duplicates
        dup = self._last_result.details.get("duplicate_columns", {})
        for group in dup.get("duplicate_groups", []):
            redundant.extend(group[1:])  # Keep first, remove rest
        
        # From correlations
        corr = self._last_result.details.get("correlated_features", {})
        redundant.extend(corr.get("features_to_consider_removing", []))
        
        return list(set(redundant))
