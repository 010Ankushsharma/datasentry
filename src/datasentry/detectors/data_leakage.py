"""Data leakage detection for DataSentry library.

This module provides detection of data leakage issues in ML datasets.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from datasentry.core.base import BaseDetector, DetectionResult, SeverityLevel
from datasentry.core.utils import validate_data, convert_to_dataframe


class DataLeakageDetector(BaseDetector):
    """Detect data leakage in ML datasets.
    
    Data leakage occurs when information from outside the training dataset
    is used to create the model. This includes:
    - Target leakage: Features that contain information about the target
    - Train-test contamination: Duplicate samples between train and test sets
    - Temporal leakage: Using future information to predict the past
    
    Attributes:
        leakage_threshold: Threshold for flagging leakage.
        check_target_leakage: Whether to check for target leakage.
        check_duplicates: Whether to check for duplicate samples.
    
    Example:
        >>> detector = DataLeakageDetector()
        >>> X_train = np.random.randn(100, 5)
        >>> X_test = np.random.randn(50, 5)
        >>> result = detector.detect(X_train, X_test=X_test)
    """
    
    def __init__(
        self,
        leakage_threshold: float = 0.95,
        check_target_leakage: bool = True,
        check_duplicates: bool = True,
        check_multicollinearity: bool = True,
        mi_threshold: float = 0.5,
        random_state: Optional[int] = 42
    ):
        """Initialize the data leakage detector.
        
        Args:
            leakage_threshold: Correlation threshold for flagging leakage.
            check_target_leakage: Whether to check for target leakage.
            check_duplicates: Whether to check for duplicate samples.
            check_multicollinearity: Whether to check feature multicollinearity.
            mi_threshold: Mutual information threshold for target leakage.
            random_state: Random seed for reproducibility.
        """
        super().__init__("DataLeakageDetector")
        self.leakage_threshold = leakage_threshold
        self.check_target_leakage = check_target_leakage
        self.check_duplicates = check_duplicates
        self.check_multicollinearity = check_multicollinearity
        self.mi_threshold = mi_threshold
        self.random_state = random_state
    
    def detect(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> DetectionResult:
        """Detect data leakage in the dataset.
        
        Args:
            X: Feature matrix (training data).
            y: Target vector (optional, required for target leakage detection).
            X_test: Test feature matrix (optional, for contamination check).
        
        Returns:
            DetectionResult with leakage analysis.
        """
        # Validate inputs
        if y is not None:
            X, y = validate_data(X, y)
        else:
            X, _ = validate_data(X)
        
        if X_test is not None:
            X_test, _ = validate_data(X_test)
        
        df = convert_to_dataframe(X)
        
        # Initialize results
        all_issues = []
        details = {"checks_performed": []}
        
        # Check 1: Target leakage
        if self.check_target_leakage and y is not None:
            target_leakage = self._check_target_leakage(df, y)
            details["target_leakage"] = target_leakage
            details["checks_performed"].append("target_leakage")
            if target_leakage["has_leakage"]:
                all_issues.append("target_leakage")
        
        # Check 2: Duplicate samples
        if self.check_duplicates:
            duplicates = self._check_duplicates(df)
            details["duplicates"] = duplicates
            details["checks_performed"].append("duplicates")
            if duplicates["has_duplicates"]:
                all_issues.append("duplicates")
        
        # Check 3: Train-test contamination
        if X_test is not None:
            contamination = self._check_contamination(df, X_test)
            details["train_test_contamination"] = contamination
            details["checks_performed"].append("train_test_contamination")
            if contamination["has_contamination"]:
                all_issues.append("train_test_contamination")
        
        # Check 4: Feature multicollinearity
        if self.check_multicollinearity:
            multicollinearity = self._check_multicollinearity(df)
            details["multicollinearity"] = multicollinearity
            details["checks_performed"].append("multicollinearity")
            if multicollinearity["has_multicollinearity"]:
                all_issues.append("multicollinearity")
        
        # Calculate overall metrics
        n_issues = len(all_issues)
        has_leakage = n_issues > 0
        
        # Calculate severity score
        severity = self._determine_severity(details)
        
        # Calculate overall score
        score = self._calculate_leakage_score(details)
        
        # Prepare summary
        details["n_issues_found"] = n_issues
        details["issues_found"] = all_issues
        
        # Generate recommendations
        recommendations = self._generate_recommendations(details)
        
        result = DetectionResult(
            detector_name=self.name,
            issue_detected=has_leakage,
            severity=severity,
            score=score,
            details=details,
            recommendations=recommendations,
        )
        
        self._last_result = result
        return result
    
    def _check_target_leakage(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """Check for target leakage using mutual information.
        
        Args:
            X: Feature DataFrame.
            y: Target vector.
        
        Returns:
            Dictionary with target leakage analysis.
        """
        y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)
        
        # Determine if classification or regression
        is_classification = len(np.unique(y_array)) < 10
        
        # Calculate mutual information
        try:
            if is_classification:
                mi_scores = mutual_info_classif(
                    X, y_array, random_state=self.random_state
                )
            else:
                mi_scores = mutual_info_regression(
                    X, y_array, random_state=self.random_state
                )
        except Exception:
            # Fallback to correlation for numeric features
            mi_scores = []
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    mi_scores.append(abs(X[col].corr(pd.Series(y_array))))
                else:
                    mi_scores.append(0.0)
            mi_scores = np.array(mi_scores)
        
        # Normalize MI scores
        mi_scores = np.nan_to_num(mi_scores, nan=0.0)
        if mi_scores.max() > 0:
            mi_scores = mi_scores / mi_scores.max()
        
        # Find high MI features
        high_mi_features = X.columns[mi_scores > self.mi_threshold].tolist()
        
        # Check for suspicious patterns (e.g., ID-like features)
        suspicious_features = []
        for col in X.columns:
            col_data = X[col]
            if pd.api.types.is_numeric_dtype(col_data):
                # Check if feature is unique (might be an ID)
                if col_data.nunique() == len(col_data):
                    suspicious_features.append(col)
        
        return {
            "has_leakage": len(high_mi_features) > 0,
            "high_mi_features": high_mi_features,
            "mi_scores": {col: round(float(score), 3) for col, score in zip(X.columns, mi_scores)},
            "suspicious_features": suspicious_features,
            "n_high_mi_features": len(high_mi_features),
        }
    
    def _check_duplicates(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate samples in the dataset.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Dictionary with duplicate analysis.
        """
        # Check for exact duplicates
        duplicated = X.duplicated()
        n_duplicates = duplicated.sum()
        
        # Get indices of duplicates
        duplicate_indices = X.index[duplicated].tolist()
        
        # Calculate duplicate ratio
        duplicate_ratio = n_duplicates / len(X) if len(X) > 0 else 0.0
        
        return {
            "has_duplicates": n_duplicates > 0,
            "n_duplicates": int(n_duplicates),
            "duplicate_ratio": round(duplicate_ratio, 4),
            "duplicate_indices": duplicate_indices[:100],  # Limit for memory
            "n_unique_samples": len(X) - n_duplicates,
        }
    
    def _check_contamination(
        self,
        X_train: pd.DataFrame,
        X_test: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Check for train-test contamination (overlapping samples).
        
        Args:
            X_train: Training feature DataFrame.
            X_test: Test feature matrix.
        
        Returns:
            Dictionary with contamination analysis.
        """
        X_test_df = convert_to_dataframe(X_test)
        
        # Ensure same columns
        common_cols = list(set(X_train.columns) & set(X_test_df.columns))
        X_train_sub = X_train[common_cols]
        X_test_sub = X_test_df[common_cols]
        
        # Check for exact matches
        merged = X_test_sub.merge(
            X_train_sub,
            on=list(common_cols),
            how='left',
            indicator=True
        )
        n_contaminated = (merged['_merge'] == 'both').sum()
        
        # Calculate contamination ratio
        contamination_ratio = n_contaminated / len(X_test) if len(X_test) > 0 else 0.0
        
        return {
            "has_contamination": n_contaminated > 0,
            "n_contaminated_samples": int(n_contaminated),
            "contamination_ratio": round(contamination_ratio, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
    
    def _check_multicollinearity(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Check for feature multicollinearity.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Dictionary with multicollinearity analysis.
        """
        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                "has_multicollinearity": False,
                "high_corr_pairs": [],
                "reason": "Insufficient numeric features",
            }
        
        # Calculate correlation matrix
        corr_matrix = X[numeric_cols].corr().abs()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val >= self.leakage_threshold:
                    high_corr_pairs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": round(float(corr_val), 4),
                    })
        
        return {
            "has_multicollinearity": len(high_corr_pairs) > 0,
            "n_high_corr_pairs": len(high_corr_pairs),
            "high_corr_pairs": high_corr_pairs[:20],  # Limit for memory
            "threshold": self.leakage_threshold,
        }
    
    def _determine_severity(self, details: Dict[str, Any]) -> SeverityLevel:
        """Determine severity based on leakage details.
        
        Args:
            details: Dictionary with leakage analysis.
        
        Returns:
            SeverityLevel enum value.
        """
        # Check for critical issues
        if "target_leakage" in details:
            tl = details["target_leakage"]
            if tl.get("has_leakage") and tl.get("n_high_mi_features", 0) > 2:
                return SeverityLevel.CRITICAL
        
        if "train_test_contamination" in details:
            tc = details["train_test_contamination"]
            if tc.get("has_contamination"):
                ratio = tc.get("contamination_ratio", 0)
                if ratio > 0.1:
                    return SeverityLevel.CRITICAL
                elif ratio > 0.05:
                    return SeverityLevel.HIGH
        
        # Count issues
        n_issues = details.get("n_issues_found", 0)
        
        if n_issues >= 3:
            return SeverityLevel.HIGH
        elif n_issues == 2:
            return SeverityLevel.MEDIUM
        elif n_issues == 1:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.NONE
    
    def _calculate_leakage_score(self, details: Dict[str, Any]) -> float:
        """Calculate overall leakage score.
        
        Args:
            details: Dictionary with leakage analysis.
        
        Returns:
            Score between 0 and 1.
        """
        scores = []
        
        if "target_leakage" in details:
            tl = details["target_leakage"]
            if tl.get("has_leakage"):
                scores.append(min(1.0, tl.get("n_high_mi_features", 0) * 0.2))
        
        if "duplicates" in details:
            dup = details["duplicates"]
            if dup.get("has_duplicates"):
                scores.append(min(1.0, dup.get("duplicate_ratio", 0) * 5))
        
        if "train_test_contamination" in details:
            tc = details["train_test_contamination"]
            if tc.get("has_contamination"):
                scores.append(min(1.0, tc.get("contamination_ratio", 0) * 10))
        
        if "multicollinearity" in details:
            mc = details["multicollinearity"]
            if mc.get("has_multicollinearity"):
                scores.append(min(1.0, mc.get("n_high_corr_pairs", 0) * 0.1))
        
        return max(scores) if scores else 0.0
    
    def _generate_recommendations(self, details: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detected issues.
        
        Args:
            details: Dictionary with leakage analysis.
        
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        if "target_leakage" in details:
            tl = details["target_leakage"]
            if tl.get("has_leakage"):
                features = tl.get("high_mi_features", [])
                recommendations.append(
                    f"Remove features with high target correlation: {features[:5]}"
                )
                if tl.get("suspicious_features"):
                    recommendations.append(
                        f"Review suspicious ID-like features: {tl['suspicious_features']}"
                    )
        
        if "duplicates" in details:
            dup = details["duplicates"]
            if dup.get("has_duplicates"):
                n_dup = dup.get("n_duplicates", 0)
                recommendations.append(
                    f"Remove {n_dup} duplicate samples from training data"
                )
        
        if "train_test_contamination" in details:
            tc = details["train_test_contamination"]
            if tc.get("has_contamination"):
                recommendations.append(
                    "CRITICAL: Train-test contamination detected. "
                    "Ensure proper data splitting before any preprocessing"
                )
        
        if "multicollinearity" in details:
            mc = details["multicollinearity"]
            if mc.get("has_multicollinearity"):
                recommendations.append(
                    f"Remove highly correlated features (>{self.leakage_threshold}) "
                    "or use regularization"
                )
        
        if not recommendations:
            recommendations.append("No data leakage detected")
        
        return recommendations
